
import json
import os
import uuid

os.environ['HF_HOME'] = "/work/nvme/bfdz/zluo8/huggingface"
from dotenv import load_dotenv
from src_py.utils import load_config_from_file
from src_py.utils import load_json_lines_from_file
from src_py.utils import combine_entries_to_pairs
from src_py.utils import get_model_directory_safe_name
from src_py.vllm_backend import create_vllm_backend
from src_py.utils import calculate_perplexity_from_logits

import argparse
import subprocess
import time
import asyncio
load_dotenv(".env")




# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run BFCL evaluation with custom configuration"
)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to a Python file containing the 'config'"
)
parser.add_argument(
    "--num-gpus",
    type=int,
    default=1,
    help="Number of GPUs to use for local inference (default: 1)"
)
parser.add_argument(
    "--debug-limit",
    type=int,
    default=None,
    help="Limit the number of entries to process for debugging (default: None)"
)

args = parser.parse_args()

# Load config from specified file
if not args.config:
    print("Error: Please specify a config file using --config argument. For example, --config config1.py")
    exit(1)

# Run maturin develop to build and install the Rust extension with file locking
import fcntl
lock_file_path = "/tmp/maturin_build_lock"
print("Acquiring build lock...")
with open(lock_file_path, "w") as lock_file:
    # Acquire exclusive lock (blocks until available)
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    try:
        print("Building Rust extension with maturin develop...")
        # result = subprocess.run(["maturin", "develop"], check=True)
        result = subprocess.run(["maturin", "develop", "--release"], check=True)
        print("Installed Rust extension successfully.")
        time.sleep(2)  # Give some time for the build to complete
    finally:
        # Release lock
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        print("Released build lock.")

# from codebase_rs import concatenate_preference_datasets, concatenate_perplexity_datasets, dispatch_preference_results, dispatch_perplexity_results
from codebase_rs import *

print(f"Loading config from: {args.config}")
config: JudgeConfig = load_config_from_file(args.config, "config")
print("Processing configuration: ", config)


model_name = config.model.to_string() # Get model name from Rust enum
model_safe_name = get_model_directory_safe_name(model_name)
# Start the first pass

experiment_str = config.experiment.to_string()


main_vllm_backend_created = False
main_hf_backend_created = False
assistant_api_backend_created = False
main_vllm_engine = None
main_hf_model = None
main_tokenizer = None
assistant_client = None



async def main_async():
    global main_hf_backend_created, main_hf_model, main_tokenizer
    match config.experiment:
        case JudgeExperiment.PreferenceDirect(lang1=lang1, lang2=lang2):
            # Determine alphabetical order for language codes
            sorted_langs = sorted([lang1, lang2])
            first_lang = sorted_langs[0]
            second_lang = sorted_langs[1]

            preference_aggregated_input_path = preference_aggregated_input_file_path(config)
            preference_aggregated_output_path = preference_aggregated_output_file_path(config)
            # load two answers datasets

            # generate a filename based on uuid
            # uuid_str = str(uuid.uuid4())
            
            # check if there is file at preference_aggregated_output_path
            if os.path.exists(preference_aggregated_output_path):
                dispatch_preference_results(model_safe_name, first_lang, second_lang, preference_aggregated_output_path)
                # delete this file
                os.remove(combined_output_path)
                print(f"Dispatched results from existing file: {combined_output_path}")
            
            # call rust function to concatenate two datasets
            preference_prepare_aggregated_input(model_safe_name, first_lang, second_lang, debug_limit=args.debug_limit)
            combined_entries = load_json_lines_from_file(combined_input_path)
            semaphore = asyncio.Semaphore(200)
            async def collect_single_preference_async(entry: dict) -> dict:
                """
                entry is of type TwoAnswersEntry in src/judge/generate_dataset.rs 
                """
                global main_vllm_backend_created, main_vllm_engine, main_tokenizer
                if not main_vllm_backend_created:
                    print(f"Creating VLLM backend for model {model_name} using {args.num_gpus} GPUs...", flush=True)
                    main_vllm_engine, main_tokenizer = create_vllm_backend(model_name, args.num_gpus)
                    print(f"VLLM backend created for model {model_name}", flush=True)
                    main_vllm_backend_created = True
                engine = main_vllm_engine
                tokenizer = main_tokenizer
                async with semaphore:
                    if config.model == LocalModel.Llama3_3_70B:
                        from src_py.llama3_1_backend import collect_preference_local_async
                    elif config.model in [LocalModel.Qwen3_8B, LocalModel.Qwen3_14B, LocalModel.Qwen3_30bA3b, LocalModel.Qwen3Next80bA3b]:
                        from src_py.qwen3_backend import collect_preference_local_async
                    else:
                        raise ValueError(f"Unsupported model for preference collection: {config.model}")
                    try:
                        logprob1, logprob2 = await collect_preference_local_async(entry['question'], entry['answer1'], entry['answer2'], engine, tokenizer)
                        if logprob1 >= logprob2:
                            preferred_answer = 1
                        else:
                            preferred_answer = 2
                        logprob_signed_difference = logprob1 - logprob2
                        preference = {
                            'Ok': {
                                'preferred_answer': preferred_answer,
                                'logprob_signed_difference': logprob_signed_difference,
                                'logprob1': logprob1,
                                'logprob2': logprob2,
                            }
                        }
                    except Exception as e:
                        error_message = str(e)
                        preference = {
                            'Err': error_message
                        }
                    # The output type is PreferenceResultEntry in src/judge/result_file_model.rs
                    return {
                        "index": entry["index"],
                        "preference": preference,
                        "question": entry["question"],
                        "answer1": entry["answer1"],
                        "answer2": entry["answer2"],
                        "lang1": entry["lang1"],
                        "lang2": entry["lang2"],
                        "is_correct1": entry["is_correct1"],
                        "is_correct2": entry["is_correct2"],
                        "subject": entry["subject"],
                    }
            
            async def collect_all_preference_entries() -> list[dict]:
                tasks = [collect_single_preference_async(entry) for entry in combined_entries]
                with open(combined_output_path, 'w', encoding='utf-8') as f:
                    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                        result = await coro
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()
                        print(f"Written {i}/{len(combined_entries)} entries to file")
            asyncio.run(collect_all_preference_entries())

            # dispatch results
            dispatch_preference_results(model_safe_name, lang1, lang2, combined_output_path)
            # delete this file
            os.remove(combined_output_path)
            print(f"Dispatched results and removed file: {combined_output_path}")
        case JudgeExperiment.Perplexity(lang=lang):
            # collect the response (to the response folder)
            # we only need the questions for the first pass, but will need the two answers for the second pass
            # however, we need two answers of the same language, so we can only load two one-answer datasets and combine them
            # we do not concatenate the datasets because two conjugated cases should be processed at the same time.

            # Use the existing filtering before letting LLMs to merge the dataset
            # Assume all filtered entries can be successfully processed

            # first pass: do the response generation
            generate_response_input_file_path = perplexity_generate_response_input_file_path(config)
            generate_response_output_file_path = perplexity_generate_response_output_file_path(config)
            if os.path.exists(generate_response_output_file_path):
                perplexity_dispatch_response_results(config)
            perplexity_prepare_response_input(config, debug_limit=args.debug_limit)
            
            
            input_entries = load_json_lines_from_file(generate_response_input_file_path)
            print(f"Total entries to generate responses for language {lang}: {len(input_entries)}")

            # Process entries asynchronously using vLLM
            semaphore = asyncio.Semaphore(200)

            async def collect_single_response_async(entry: dict) -> dict:
                """
                entry is of type GenerateResponseInputEntry
                """
                global main_vllm_backend_created, main_vllm_engine, main_tokenizer
                if not main_vllm_backend_created:
                    print(f"Creating VLLM backend for model {model_name} using {args.num_gpus} GPUs...", flush=True)
                    main_vllm_engine, main_tokenizer = create_vllm_backend(model_name, args.num_gpus)
                    print(f"VLLM backend created for model {model_name}", flush=True)
                    main_vllm_backend_created = True
                engine = main_vllm_engine
                tokenizer = main_tokenizer
                async with semaphore:
                    if config.model == LocalModel.Llama3_3_70B:
                        from src_py.llama3_1_backend import generate_response_async
                    elif config.model in [LocalModel.Qwen3_8B, LocalModel.Qwen3_14B, LocalModel.Qwen3_30bA3b, LocalModel.Qwen3Next80bA3b]:
                        from src_py.qwen3_backend import generate_response_async
                    else:
                        raise ValueError(f"Unsupported model for response collection: {config.model}")
                    try:
                        response = await generate_response_async(entry, engine, tokenizer)
                    except Exception as e:
                        error_message = str(e)
                        print(f"Error generating response for entry {entry['index']}: {error_message}")
                        response = f"ERROR: {error_message}"

                    return {
                        'index': entry['index'],
                        'question': entry['question'],
                        'response': response,
                        'lang': entry['lang'],
                        'subject': entry['subject'],
                    }

            async def collect_all_response_entries() -> list[dict]:
                tasks = [collect_single_response_async(entry) for entry in input_entries]
                with open(generate_response_output_file_path, 'w', encoding='utf-8') as f:
                    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                        result = await coro
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()
                        print(f"Written {i}/{len(input_entries)} entries to file")

            await collect_all_response_entries()
            perplexity_dispatch_response_results(config)
            print(f"Completed writing all responses to {generate_response_output_file_path}")

            # debug: stop here
            exit(1)

            
            # pass 2: call gpt-5/deepseek to merge the style with the ground truth (to the perplexity_dataset folder)
            generate_styled_answers_input_file_path = perplexity_generate_styled_answers_input_file_path(config)
            generate_styled_answers_output_file_path = perplexity_generate_styled_answers_output_file_path(config)
            if os.path.exists(generate_styled_answers_output_file_path):
                perplexity_dispatch_styled_answers_results(config)
            perplexity_prepare_generate_styled_answers_input(config, debug_limit=args.debug_limit)
            
            input_entries = load_json_lines_from_file(generate_styled_answers_input_file_path)
            print(f"Total entries to generate styled answers for language {lang}: {len(input_entries)}")

            # then write an async function to process them
            semaphore = asyncio.Semaphore(200)
            async def collect_single_styled_answers_async(
                entry: dict,
            ) -> dict:
                """
                entry is of type GenerateStyledAnswersInputEntry in src/judge/perplexity.rs 
                """
                global assistant_api_backend_created, assistant_client
                assistant_model_name = "gpt-5"
                if not assistant_api_backend_created:
                    print(f"Creating Assistant API client for model {assistant_model_name}...", flush=True)
                    from src_py.api_backend import create_api_backend
                    assistant_client = create_api_backend(assistant_model_name)
                    print(f"Assistant API client created for model {assistant_model_name}", flush=True)
                    assistant_api_backend_created = True
                client = assistant_client
                async with semaphore:
                    from src_py.gpt5_backend import generate_styled_answers_async
                    result = await generate_styled_answers_async(
                        assistant_model_name,
                        client,
                        entry['question'],
                        entry['response'],
                        entry['original_answer_correct'],
                        entry['original_answer_incorrect']
                    )
                # return type is StyledAnswersEntry in src/judge/perplexity.rs
                return {
                    'index': entry['index'],
                    'question': entry['question'],
                    'styled_response_correct': result['styled_response_correct'],
                    'styled_response_incorrect': result['styled_response_incorrect'],
                    'lang': entry['lang'],
                    'subject': entry['subject'],
                }
            async def collect_all_perplexity_dataset_entries() -> list[dict]:
                tasks = [collect_single_styled_answers_async(entry) for entry in input_entries]
                with open(generate_styled_answers_output_file_path, 'a', encoding='utf-8') as f:
                    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                        result = await coro
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()
                        print(f"Written {i}/{len(input_entries)} entries to perplexity dataset file")
            asyncio.run(collect_all_perplexity_dataset_entries())
            perplexity_dispatch_styled_answers_results(config)
            print(f"Completed writing all styled answers to {generate_styled_answers_output_file_path}")

            # third pass: input is perplexity dataset, output is perplexity
            generate_perplexity_aggregated_input_file_path = perplexity_generate_perplexity_aggregated_input_file_path(config)
            generate_perplexity_aggregated_output_file_path = perplexity_generate_perplexity_aggregated_output_file_path(config)
            if os.path.exists(generate_perplexity_aggregated_output_file_path):
                perplexity_dispatch_generate_perplexity_results(config)
            perplexity_prepare_generate_perplexity_aggregated_input(config, debug_limit=args.debug_limit)
            
            
            input_entries = load_json_lines_from_file(generate_perplexity_aggregated_input_file_path)
            print(f"Total entries to calculate perplexity for language {lang}: {len(input_entries)}")
            
            with open(generate_perplexity_aggregated_output_file_path, 'w') as f:
                for i in range(0, len(indices_to_process), batch_size):
                    batch_indices = indices_to_process[i:i+batch_size]
                    batch_entries = [perplexity_dataset_entries[index] for index in batch_indices]
                    print(f"Processing batch {i//batch_size + 1}/{(len(indices_to_process) + batch_size - 1)//batch_size}", flush=True)

                    if not main_hf_backend_created:
                        print(f"Creating HuggingFace backend for model {model_name} using {args.num_gpus} GPUs...", flush=True)
                        from src_py.huggingface_backend import create_huggingface_backend
                        main_hf_model, main_tokenizer = create_huggingface_backend(model_name, args.num_gpus)
                        print(f"HuggingFace backend created for model {model_name}", flush=True)
                        main_hf_backend_created = True
                    hf_model = main_hf_model
                    hf_tokenizer = main_tokenizer

                    # Get model outputs for the batch
                    if config.model == LocalModel.Llama3_3_70B:
                        from src_py.llama3_1_backend import collect_perplexity_batch
                    elif config.model in [LocalModel.Qwen3_8B, LocalModel.Qwen3_14B, LocalModel.Qwen3_30bA3b, LocalModel.Qwen3Next80bA3b]:
                        from src_py.qwen3_backend import collect_perplexity_batch
                    else:
                        raise ValueError(f"Unsupported model for perplexity collection: {config.model}")

                    batch_outputs = collect_perplexity_batch(batch_entries, args.num_gpus, config.model)

                    # Process each entry in the batch and write immediately
                    for entry, output in zip(batch_entries, batch_outputs):
                        result_entry = {
                            'index': entry['index'],
                            'perplexity': output['perplexity'],
                            'question': entry['question'],
                            'styled_response': entry['styled_response'],
                            'original_answer': entry['original_answer'],
                            'is_correct': entry['is_correct'],
                            'lang': lang,
                            'subject': entry['subject'],
                        }
                        # Write result immediately
                        f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
                        f.flush()
                        total_processed += 1

                    # Flush after each batch to ensure results are written
                    f.flush()
                    print(f"Written {total_processed}/{len(indices_to_process)} entries to perplexity result file", flush=True)
            perplexity_dispatch_generate_perplexity_results(config)
            print(f"Completed writing all {total_processed} perplexity results to {generate_perplexity_aggregated_output_file_path}")

if __name__ == "__main__":
    asyncio.run(main_async())






