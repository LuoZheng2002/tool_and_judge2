use atomic_refcell::AtomicRefCell;
use futures::stream::{self, StreamExt};
use pyo3::{
    Py, Python,
    types::{PyAnyMethods, PyList, PyListMethods},
};
use std::sync::Arc;

use crate::{
    config::{Language, ToolConfig, TranslateMode, TranslateOption},
    models::{
        backend::{WhichBackend, get_or_create_backend}, function_name_mapper::{self, FunctionNameMapper}, model_interface::{ToolCallParsingResult, get_model_interface}
    },
    tool_bfcl_decl::BfclDatasetEntry,
    util::{
        get_model_directory_safe_name, load_json_lines, load_json_lines_with_id, parse_test_cases, sort_and_write_json_lines, write_json_lines_to_file
    },
};

const CATEGORY_CACHE_PATH: &str = "tool_category_cache.json";
const CATEGORY_CACHE_LOCK_PATH: &str = "tool_category_cache.json.lock";

pub async fn tool_run_async(configs: Py<PyList>, num_gpus: usize) {
    let (extracted_configs, config_len): (Vec<ToolConfig>, usize) = Python::attach(|py| {
        let configs = configs.bind(py);
        let config_len = configs.len();
        let extracted_configs = configs
            .iter()
            .map(|config| {
                config
                    .extract()
                    .expect("Failed to extract ToolConfig from Python object")
            })
            .collect();
        (extracted_configs, config_len)
    });

    println!(
        "Tool run implementation called with {} configs and  {} GPUs.",
        config_len, num_gpus
    );

    // load environment variables from .env file
    dotenvy::dotenv().ok();
    println!("Loaded environment variables from .env file.");
    println!("Starting tool run with {} configs.", config_len);

    let function_name_mapper = Arc::new(AtomicRefCell::new(FunctionNameMapper::new()));
    for config in extracted_configs {
        println!("Processing config: {:?}", config);
        let language_tag = match &config.translate_mode {
            TranslateMode::Translated { language, .. } => match language {
                Language::Chinese => "_zh",
                Language::Hindi => "_hi",
            },
            TranslateMode::NotTranslated {} => "_en",
        };
        let (translate_level_tag, pre_translate_tag, prompt_translate_tag, post_translate_tag) =
            match &config.translate_mode {
                TranslateMode::Translated { option, .. } => match option {
                    TranslateOption::FullyTranslated => {
                        ("_fulltrans", "_nopretrans", "_noprompt", "_noposttrans")
                    }
                    TranslateOption::FullyTranslatedPromptTranslate => {
                        ("_fulltrans", "_nopretrans", "_prompt", "_noposttrans")
                    }
                    TranslateOption::FullyTranslatedPreTranslate => {
                        ("_fulltrans", "_pretrans", "_noprompt", "_noposttrans")
                    }
                    TranslateOption::FullyTranslatedPostTranslate => {
                        ("_fulltrans", "_nopretrans", "_noprompt", "_posttrans")
                    }
                    TranslateOption::PartiallyTranslated => {
                        ("_parttrans", "_nopretrans", "_noprompt", "_noposttrans")
                    }
                },
                TranslateMode::NotTranslated {} => {
                    ("_na", "_nopretrans", "_noprompt", "_noposttrans")
                }
            };
        let noise_tag = match &config.add_noise_mode {
            crate::config::AddNoiseMode::NoNoise => "_nonoise",
            crate::config::AddNoiseMode::Synonym => "_syno",
            crate::config::AddNoiseMode::Paraphrase => "_para",
        };
        let model_dir_name = get_model_directory_safe_name(&config.model.to_string());
        let unpretranslated_dataset_path = format!(
            "tool/dataset/BFCL_v4_multiple{language_tag}{translate_level_tag}{noise_tag}.json"
        );
        let ground_truth_path = "tool/dataset/possible_answer/BFCL_v4_multiple.json";
        let pre_translate_output_combined_tags =
            language_tag.to_string() + translate_level_tag + pre_translate_tag + noise_tag;
        let inference_raw_output_combined_tags = language_tag.to_string()
            + translate_level_tag
            + pre_translate_tag
            + noise_tag
            + prompt_translate_tag;
        let post_translate_output_combined_tags = language_tag.to_string()
            + translate_level_tag
            + pre_translate_tag
            + noise_tag
            + prompt_translate_tag
            + post_translate_tag;

        let (pre_translate_input_path, pre_translate_output_path) = if pre_translate_tag
            == "_pretrans"
        {
            (
                unpretranslated_dataset_path.clone(),
                Some(format!(
                    "tool/result/pre_translate/{model_dir_name}/{pre_translate_output_combined_tags}.json"
                )),
            )
        } else {
            assert_eq!(pre_translate_tag, "_nopretrans");
            (unpretranslated_dataset_path.clone(), None)
        };

        let inference_raw_input_path = if pre_translate_tag == "_pretrans" {
            pre_translate_output_path
                .clone()
                .expect("pre_translate_output_path should have value")
        } else {
            unpretranslated_dataset_path.clone()
        };

        let inference_raw_output_path = format!(
            "tool/result/inference_raw/{model_dir_name}/{inference_raw_output_combined_tags}.json"
        );

        let inference_json_input_path = inference_raw_output_path.clone();
        let inference_json_output_path = format!(
            "tool/result/inference_json/{model_dir_name}/{inference_raw_output_combined_tags}.json"
        );
        let post_translate_input_path = inference_json_output_path.clone();
        let post_translate_output_path = if post_translate_tag == "_posttrans" {
            Some(format!(
                "tool/result/post_translate/{model_dir_name}/{post_translate_output_combined_tags}.json"
            ))
        } else {
            assert_eq!(post_translate_tag, "_noposttrans");
            None
        };
        let evaluation_input_path = if post_translate_tag == "_posttrans" {
            post_translate_output_path
                .clone()
                .expect("post_translate_output_path should have value")
        } else {
            post_translate_input_path.clone()
        };
        let evaluation_output_path = format!(
            "tool/result/evaluation/{model_dir_name}/{post_translate_output_combined_tags}.json"
        );
        let score_input_path = evaluation_output_path.clone();
        let score_output_path = format!(
            "tool/result/score/{model_dir_name}/{post_translate_output_combined_tags}.json"
        );
        let categorize_input_path = score_output_path.clone();
        let categorize_output_path = format!(
            "tool/result/categorize/{model_dir_name}/{post_translate_output_combined_tags}.json"
        );
        let categorize_score_input_path = categorize_output_path.clone();
        let categorize_score_output_path = format!(
            "tool/result/categorize_score/{model_dir_name}/{post_translate_output_combined_tags}.json"
        );

        let test_cases =
            load_json_lines(&unpretranslated_dataset_path).expect("Failed to load test cases");
        let ground_truths =
            load_json_lines(ground_truth_path).expect("Failed to load ground truths");

        println!(
            "Loaded {} test cases from {}",
            test_cases.len(),
            unpretranslated_dataset_path
        );
        let test_cases = parse_test_cases(test_cases);

        /* ════════════════════════════════════════════════════════════════════════════════ */
        /* PASS 1: Translated Questions (Pre-Translation)                                   */
        /* ════════════════════════════════════════════════════════════════════════════════ */
        /* Translates questions from the source language to English before inference.       */
        /* This pass runs when FULLY_TRANSLATED_PRE_TRANSLATE option is enabled.            */
        /* Output: tool/result/pre_translate/{model}/{language}.json                        */
        /* ════════════════════════════════════════════════════════════════════════════════ */
        if pre_translate_tag == "_nopretrans" {
            // Skip translation - pass through original test cases
            println!("Skipping question translation (pre-translate not enabled)");
        } else {
            assert_eq!(pre_translate_tag, "_pretrans");
            let pre_translate_output_path = pre_translate_output_path
                .as_ref()
                .expect("pre_translate_output_path should have value");
            let (mut pre_translate_results, existing_pre_translate_ids) =
                match load_json_lines_with_id(&pre_translate_output_path) {
                    Ok(results) => results,
                    Err(_) => {
                        println!(
                            "File {} not found. It will be created.",
                            pre_translate_output_path
                        );
                        (Vec::new(), Vec::new())
                    }
                };
            let cases_to_translate: Vec<BfclDatasetEntry> = test_cases
                .iter()
                .filter(|case|{
                    !existing_pre_translate_ids.contains(&case.id)
                }).cloned().collect();

            if cases_to_translate.is_empty() {
                println!("All test cases have already been translated. Skipping translation.");
            } else {
                println!(
                    "Translating {} questions to English...",
                    cases_to_translate.len()
                );

                // Get backend and interface for translation
                let main_backend =
                    get_or_create_backend(config.model, WhichBackend::Main, num_gpus).await;
                let main_backend = main_backend
                    .as_ref()
                    .expect("Backend should be created by the call above");
                let main_interface = get_model_interface(config.model);

                // Create tasks for all translations
                let total_cases = cases_to_translate.len();
                let mut tasks = Vec::new();

                for case in cases_to_translate.iter() {
                    let question = case.question_content.clone();
                    let case_clone = case.clone();
                    let main_interface = main_interface.clone();
                    let main_backend = main_backend.clone();
                    let task = async move {
                        // Use the dedicated translation method
                        let translated_question = main_interface
                            .translate_tool_question_async(main_backend, question)
                            .await;
                        let modified_case = case_clone
                            .modify_question_content(&translated_question)
                            .expect("Failed to modify question content");
                        modified_case
                    };
                    tasks.push(task);
                }

                // Create a stream from the tasks and process up to 200 concurrently
                let mut translation_stream = stream::iter(tasks).buffer_unordered(200);

                let mut completed_count = 0;
                while let Some(modified_case) = translation_stream.next().await {
                    completed_count += 1;
                    println!(
                        "[{}/{}] Translated question for case {}",
                        completed_count,
                        total_cases,
                        modified_case
                            .get("id")
                            .and_then(|id| id.as_str())
                            .expect("Modified case missing 'id' field")
                    );
                    pre_translate_results.push(modified_case);
                    // Write to file immediately
                    if completed_count % 10 == 0 {
                        write_json_lines_to_file(
                            &pre_translate_output_path,
                            &pre_translate_results,
                        )
                        .expect("Failed to write pre-translation results to file");
                    }
                }
                println!(
                    "All {} questions translated.",
                    cases_to_translate.len()
                );
                // Final sort and write
                if !pre_translate_results.is_empty() {
                    sort_and_write_json_lines(
                        pre_translate_output_path,
                        &mut pre_translate_results,
                    )
                    .expect("Failed to sort and write pre-translation results");
                }
            }
        }
        /* ════════════════════════════════════════════════════════════════════════════════ */
        /* PASS 2: Inference Raw                                                            */
        /* ════════════════════════════════════════════════════════════════════════════════ */
        /* Generates raw model outputs for each test case using function calling.           */
        /* Input: test_cases (from pre_translate if pre-translate enabled, else dataset)    */
        /* Output: tool/result/inference_raw/{model}/{filename}.json                        */
        /* ════════════════════════════════════════════════════════════════════════════════ */
        let (mut inference_raw_outputs, existing_inference_ids) = match
            load_json_lines_with_id(&inference_raw_output_path)
        {
            Ok(results) => results,
            Err(_) => {
                println!(
                    "File {} not found. It will be created.",
                    inference_raw_output_path
                );
                (Vec::new(), Vec::new())
            }
        };

        let preprocessed_test_cases = load_json_lines(&inference_raw_input_path).expect(
            "Failed to load pre-translation test cases for inference",
        );
        let preprocessed_test_cases = parse_test_cases(preprocessed_test_cases);
        let cases_to_process = preprocessed_test_cases
            .into_iter()
            .filter(|case| {
                !existing_inference_ids.contains(&case.id)
            })
            .collect::<Vec<BfclDatasetEntry>>();
        if cases_to_process.is_empty(){
            println!("All test cases for {} have already been processed. Skipping model loading and inference.", config.model.to_string());
        }else{
            let total_cases = cases_to_process.len();
            println!(
                "Generating functions for {} cases...",
                total_cases
            );
            let main_backend = get_or_create_backend(
                config.model,
                WhichBackend::Main,
                num_gpus,
            ).await;
            let main_backend = main_backend
                .as_ref()
                .expect("Backend should be created by the call above");
            // Model interface can be created outside async context
            let main_interface = get_model_interface(config.model);

            let prompt_translate = if prompt_translate_tag == "_prompt" {
                true
            } else {
                assert_eq!(prompt_translate_tag, "_noprompt");
                false
            };
            let mut tasks = Vec::new();

            for case in cases_to_process.iter() {
                let functions = case.functions.clone();
                let user_question = case.question_content.clone();
                let case_clone = case.clone();
                let main_interface = main_interface.clone();
                let main_backend = main_backend.clone();
                let function_name_mapper = function_name_mapper.clone();
                let task = async move {
                    let result = main_interface
                        .generate_tool_call_async(
                            main_backend.clone(),
                            functions,
                            user_question,
                            prompt_translate,
                            function_name_mapper,
                        )
                        .await;
                    (case_clone.id.clone(), result)
                };
                tasks.push(task);
            }
            // Create a stream from the tasks and process up to 200 concurrently
            let mut inference_stream = stream::iter(tasks).buffer_unordered(200);
            let mut completed_count = 0;
            while let Some(result) = inference_stream.next().await {
                completed_count += 1;
                println!(
                    "[{}/{}] Case {} processed",
                    completed_count,
                    total_cases,
                    result.0
                );
                let result_to_write = serde_json::json!({
                    "id": result.0,
                    "result": result.1
                });
                inference_raw_outputs.push(result_to_write);
                // Write to file immediately
                if completed_count % 10 == 0 {
                    write_json_lines_to_file(
                        &inference_raw_output_path,
                        &inference_raw_outputs,
                    )
                    .expect("Failed to write inference raw results to file");
                }
            }
            println!(
                "All {} cases processed.",
                cases_to_process.len()
            );
            // Final sort and write
            if !inference_raw_outputs.is_empty() {
                sort_and_write_json_lines(
                    &inference_raw_output_path,
                    &mut inference_raw_outputs,
                )
                .expect("Failed to sort and write inference raw results");
            }
        }
        /* ═══════════════════════════════════════════════════════════════════════ */
        /* PASS 3: Inference JSON                                                  */
        /* ═══════════════════════════════════════════════════════════════════════ */
        /* Converts raw model outputs into structured JSON format.                 */
        /* Input: tool/result/inference_raw/{model}/{filename}.json                */
        /* Output: tool/result/inference_json/{model}/{filename}.json              */
        /* ═══════════════════════════════════════════════════════════════════════ */
        let inference_json_inputs = load_json_lines(&inference_json_input_path)
            .expect("Failed to load inference raw outputs for JSON conversion");
        let main_interface = get_model_interface(config.model);
        let mut inference_json_outputs = Vec::new();
        for entry in inference_json_inputs.iter() {
            let id = entry
                .get("id")
                .and_then(|v| v.as_str())
                .expect("Missing or invalid 'id' field")
                .to_string();
            let result_str = entry
                .get("result")
                .and_then(|v| v.as_str())
                .expect("Missing or invalid 'result' field")
                .to_string();
            let result = main_interface.postprocess_tool_calls(&result_str, function_name_mapper.clone());
            let valid = match &result {
                ToolCallParsingResult::Success(_) => true,
                ToolCallParsingResult::Failure(_) => false,
            };
            let result_json = serde_json::to_value(result)
                .expect("Failed to serialize post-processed tool call result");
            let output_entry = serde_json::json!({
                "id": id,
                "valid": valid,
                "result": result_json
            });
            inference_json_outputs.push(output_entry);
        }
        sort_and_write_json_lines(
            &inference_json_output_path,
            &mut inference_json_outputs,
        )
        .expect("Failed to sort and write inference JSON results");
        /* ═══════════════════════════════════════════════════════════════════════ */
        /* PASS 4: Post-Translation                                                */
        /* ═══════════════════════════════════════════════════════════════════════ */
        /* Translates model outputs back to the source language.                   */
        /* This pass runs when FULLY_TRANSLATED_POST_TRANSLATE option is enabled.  */
        /* Output: tool/result/post_translate/{model}/{language}.json              */
        /* ═══════════════════════════════════════════════════════════════════════ */
        if post_translate_tag == "_noposttrans" {
            // Skip translation - pass through original inference json results
            println!("Skipping answer translation (post-translate not enabled)");
        } else {
            assert_eq!(post_translate_tag, "_posttrans");
            // Load inference json results
            let inference_json_results = load_json_lines(&post_translate_input_path)
                .expect("Failed to load inference JSON results for post-translation");

            let (mut translated_answers_results, existing_translated_answers_ids) =
                match load_json_lines_with_id(
                    post_translate_output_path
                        .as_ref()
                        .expect("post_translate_output_path should have value"),
                ) {
                    Ok(results) => results,
                    Err(_) => {
                        println!(
                            "File {} not found. It will be created.",
                            post_translate_output_path
                                .as_ref()
                                .expect("post_translate_output_path should have value")
                        );
                        (Vec::new(), Vec::new())
                    }
                };
            
    }
}


//         # ═══════════════════════════════════════════════════════════════════════
//         # PASS 4: Translated Answers (Post-Translation)
//         # ═══════════════════════════════════════════════════════════════════════
//         # Translates function call parameter values from source language to English.
//         # This pass runs when FULLY_TRANSLATED_POST_TRANSLATE option is enabled.
//         # Input: tool/result/inference_json/{model}/{filename}.json
//         # Output: tool/result/translated_answers/{model}/{language}.json
//         # ═══════════════════════════════════════════════════════════════════════
//         if post_translate_tag == "_noposttrans":
//             # Skip translation - pass through original inference json results
//             print(f"Skipping answer translation (post-translate not enabled)")
//         else:
//             assert post_translate_tag == "_posttrans"
//             # Load inference json results
//             try:
//                 inference_json_results = load_json_lines(post_translate_input_path)
//             except FileNotFoundError:
//                 print(f"Error: File {post_translate_input_path} not found.")
//                 exit(1)

//             try:
//                 translated_answers_results, existing_translated_answers_ids = load_json_lines_with_id(post_translate_output_path)
//                 existing_translated_answers_ids = {entry["id"] for entry in translated_answers_results}
//             except FileNotFoundError:
//                 print(f"File {post_translate_output_path} not found. It will be created.")
//                 translated_answers_results = []
//                 existing_translated_answers_ids = set()

//             # Filter samples that haven't been translated yet
//             samples_to_translate = [sample for sample in inference_json_results if sample['id'] not in existing_translated_answers_ids]

//             if len(samples_to_translate) == 0:
//                 print(f"All answers have already been translated. Skipping translation.")
//             else:
//                 print(f"Translating {len(samples_to_translate)} answers to English...")

//                 # Get backend and interface for translation
//                 translation_backend = get_or_create_backend(
//                     model=config.model,
//                     num_gpus=args.num_gpus,
//                     max_model_len=2000,
//                     instance_name="experiment"  # Use experiment instance for post-translation
//                 )
//                 translation_interface = get_or_create_model_interface(config.model)

//                 async def translate_answers_async():
//                     """Translate function call parameters asynchronously."""
//                     async def translate_list_values(items: list) -> list:
//                         """
//                         Recursively translate string values within a list.

//                         For example:
//                         ["鸡肉", "蘑菇"] -> ["chicken", "mushroom"]
//                         """
//                         # Collect all items that need translation
//                         translation_tasks = []
//                         indices_for_strings = []

//                         for i, item in enumerate(items):
//                             if isinstance(item, str) and item.strip():
//                                 # Translate string items
//                                 translation_tasks.append(
//                                     translation_interface.translate_tool_answer_async(
//                                         backend=translation_backend,
//                                         parameter_value=item
//                                     )
//                                 )
//                                 indices_for_strings.append(i)

//                         # Create result list with original items
//                         translated_list = list(items)

//                         # Wait for all string translations to complete
//                         if translation_tasks:
//                             translated_values = await asyncio.gather(*translation_tasks)
//                             # Replace translated strings at their original indices
//                             for idx, translated_value in zip(indices_for_strings, translated_values):
//                                 translated_list[idx] = translated_value

//                         # Second pass: recursively translate nested dicts and lists
//                         for i, item in enumerate(translated_list):
//                             if isinstance(item, dict):
//                                 translated_list[i] = await translate_dict_values(item)
//                             elif isinstance(item, list):
//                                 translated_list[i] = await translate_list_values(item)

//                         return translated_list

//                     async def translate_dict_values(arguments: dict) -> dict:
//                         """
//                         Recursively translate only the VALUES in a dictionary, preserving all KEYS unchanged.

//                         For example:
//                         {"location": "北京"} -> {"location": "Beijing"}  # Key preserved, value translated
//                         """
//                         translated = {}

//                         # First pass: collect all string values that need translation
//                         # IMPORTANT: We only translate VALUES, never KEYS (parameter names)
//                         translation_tasks = []
//                         keys_for_string_values = []  # Store parameter names (not translated)

//                         for param_name, param_value in arguments.items():
//                             if isinstance(param_value, str) and param_value.strip():
//                                 # Translate this string VALUE
//                                 # The parameter NAME (param_name) is preserved as-is
//                                 translation_tasks.append(
//                                     translation_interface.translate_tool_answer_async(
//                                         backend=translation_backend,
//                                         parameter_value=param_value
//                                     )
//                                 )
//                                 keys_for_string_values.append(param_name)
//                             elif isinstance(param_value, (dict, list)):
//                                 # Skip for now - will handle in second pass
//                                 pass
//                             else:
//                                 # Keep non-string values as-is (numbers, booleans, etc.)
//                                 translated[param_name] = param_value

//                         # Wait for all string value translations to complete
//                         if translation_tasks:
//                             translated_values = await asyncio.gather(*translation_tasks)
//                             # Map translated values back to their ORIGINAL parameter names
//                             for param_name, translated_value in zip(keys_for_string_values, translated_values):
//                                 translated[param_name] = translated_value

//                         # Second pass: recursively translate nested dictionaries and lists
//                         # Again, only the values in nested dicts/lists are translated, not the keys
//                         for param_name, param_value in arguments.items():
//                             if isinstance(param_value, dict):
//                                 translated[param_name] = await translate_dict_values(param_value)
//                             elif isinstance(param_value, list):
//                                 translated[param_name] = await translate_list_values(param_value)

//                         return translated

//                     async def translate_single_answer(sample):
//                         """Translate parameters in a single sample and return the modified sample."""
//                         # If sample is invalid (postprocess error), return as-is
//                         if not sample.get("valid", True):  # Default to True for backward compatibility
//                             return sample

//                         result = sample.get("result", [])

//                         # If result is not a list or is empty, return as is
//                         if not isinstance(result, list) or len(result) == 0:
//                             return sample

//                         modified_result = []
//                         for func_call in result:
//                             if not isinstance(func_call, dict):
//                                 modified_result.append(func_call)
//                                 continue

//                             # Get the function name and arguments
//                             func_name = list(func_call.keys())[0] if func_call else None
//                             if not func_name:
//                                 modified_result.append(func_call)
//                                 continue

//                             arguments = func_call.get(func_name, {})

//                             # If no arguments, skip translation
//                             if not arguments or not isinstance(arguments, dict):
//                                 modified_result.append(func_call)
//                                 continue

//                             # Translate all string values in arguments
//                             try:
//                                 translated_arguments = await translate_dict_values(arguments)

//                                 # Create modified function call with translated arguments
//                                 modified_result.append({func_name: translated_arguments})
//                             except Exception as e:
//                                 print(f"Error: Failed to translate parameters for sample {sample['id']}: {e}")
//                                 exit(1)
//                                 # # Keep original if translation fails
//                                 # modified_result.append(func_call)

//                         # Create modified sample with translated parameters
//                         modified_sample = sample.copy()
//                         modified_sample["result"] = modified_result

//                         return modified_sample

//                     # Create all translation tasks
//                     tasks = [translate_single_answer(sample) for sample in samples_to_translate]

//                     # Process results as they complete
//                     completed_count = 0
//                     for coro in asyncio.as_completed(tasks):
//                         modified_sample = await coro
//                         completed_count += 1

//                         print(f"[{completed_count}/{len(samples_to_translate)}] Translated answer parameters for sample {modified_sample['id']}")

//                         translated_answers_results.append(modified_sample)

//                         # Write to file immediately
//                         write_json_lines_to_file(post_translate_output_path, translated_answers_results)

//                 # Run the async translation
//                 await translate_answers_async()

//                 print(f"All {len(samples_to_translate)} answers translated.")

//                 # Final sort and write
//                 if len(translated_answers_results) > 0:
//                     append_and_rewrite_json_lines(post_translate_output_path, translated_answers_results)
