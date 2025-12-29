use indexmap::IndexMap;
use pyo3::pyfunction;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use crate::{
    config::{JudgeConfig, JudgeExperiment, Model},
    judge::{
        base_paths::{JUDGE_BASE_DATASET_PATH, JUDGE_BASE_RESULT_PATH},
        generate_dataset::{
            PerplexityDatasetMaskEntry, TwoAnswersSameLangEntry,
            generate_perplexity_dataset_mask, generate_two_answers_same_lang_dataset,
            get_valid_perplexity_indices,
        },
    },
    utils::{get_model_safe_name, load_json_lines, write_json_lines_to_file},
};

// first convert two_answers_same_lang dataset to GenerateResponseInputEntry by selecting indices that haven't been processed
// then collect ResponseEntry and write it to result file (in folder response)
// then convert ResponseEntry and TwoAnswersSameLangEntry to GenerateStyledAnswersInputEntry, also selecting unprocessed indices
// then collect StyledAnswersEntry and write it to result file (in folder styled_answers)
// finally convert StyledAnswersEntry to GeneratePerplexityAggregatedInputEntry, selecting unprocessed indices
// then collect PerplexityEntry and write it to result file (in folder perplexity)

#[derive(Clone, Deserialize, Serialize)]
pub struct GenerateResponseInputEntry {
    pub index: usize,
    pub question: String,
    pub lang: String,
    pub subject: String,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct ResponseEntry {
    pub index: usize,
    pub question: String,
    pub response: String,
    pub lang: String,
    pub subject: String,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct GenerateStyledAnswersInputEntry {
    pub index: usize,
    pub question: String,
    pub response: String,
    pub original_answer_correct: String,
    pub original_answer_incorrect: String,
    pub lang: String,
    pub subject: String,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct StyledAnswersEntry {
    pub index: usize,
    pub question: String,
    pub styled_response_correct: String,
    pub styled_response_incorrect: String,
    pub lang: String,
    pub subject: String,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct GeneratePerplexityAggregatedInputEntry {
    pub index: usize,
    pub question: String,
    pub styled_response: String,
    pub is_correct: bool,
    pub lang: String,
    pub subject: String,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct PerplexityEntry {
    pub index: usize,
    pub perplexity: f64,
    pub question: String,
    pub styled_response: String,
    // pub original_answer: String,
    pub is_correct: bool,
    pub lang: String,
    pub subject: String,
}

const PERPLEXITY_GENERATE_RESPONSE_INPUT_FILE_NAME: &str = "perplexity_response_input.jsonl";
const PERPLEXITY_GENERATE_RESPONSE_OUTPUT_FILE_NAME: &str = "perplexity_response_output.jsonl";
const PERPLEXITY_GENERATE_STYLED_ANSWERS_INPUT_FILE_NAME: &str =
    "perplexity_styled_answers_input.jsonl";
const PERPLEXITY_GENERATE_STYLED_ANSWERS_OUTPUT_FILE_NAME: &str =
    "perplexity_styled_answers_output.jsonl";
const PERPLEXITY_GENERATE_PERPLEXITY_AGGREGATED_INPUT_FILE_NAME: &str =
    "perplexity_aggregated_input.jsonl";
const PERPLEXITY_GENERATE_PERPLEXITY_AGGREGATED_OUTPUT_FILE_NAME: &str =
    "perplexity_aggregated_output.jsonl";
#[pyfunction]
pub fn perplexity_generate_response_input_file_path(config: &JudgeConfig) -> String {
    let model = Model::Local(config.model);
    let model_safe_name = get_model_safe_name(model);
    let file_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join(PERPLEXITY_GENERATE_RESPONSE_INPUT_FILE_NAME);
    file_path.to_string_lossy().to_string()
}

#[pyfunction]
pub fn perplexity_generate_response_output_file_path(config: &JudgeConfig) -> String {
    let model = Model::Local(config.model);
    let model_safe_name = get_model_safe_name(model);
    let file_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join(PERPLEXITY_GENERATE_RESPONSE_OUTPUT_FILE_NAME);
    file_path.to_string_lossy().to_string()
}

#[pyfunction]
pub fn perplexity_generate_styled_answers_input_file_path(config: &JudgeConfig) -> String {
    let model = Model::Local(config.model);
    let model_safe_name = get_model_safe_name(model);
    let file_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join(PERPLEXITY_GENERATE_STYLED_ANSWERS_INPUT_FILE_NAME);
    file_path.to_string_lossy().to_string()
}
#[pyfunction]
pub fn perplexity_generate_styled_answers_output_file_path(config: &JudgeConfig) -> String {
    let model = Model::Local(config.model);
    let model_safe_name = get_model_safe_name(model);
    let file_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join(PERPLEXITY_GENERATE_STYLED_ANSWERS_OUTPUT_FILE_NAME);
    file_path.to_string_lossy().to_string()
}

#[pyfunction]
pub fn perplexity_generate_perplexity_aggregated_input_file_path(config: &JudgeConfig) -> String {
    let model = Model::Local(config.model);
    let model_safe_name = get_model_safe_name(model);
    let file_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join(PERPLEXITY_GENERATE_PERPLEXITY_AGGREGATED_INPUT_FILE_NAME);
    file_path.to_string_lossy().to_string()
}
#[pyfunction]
pub fn perplexity_generate_perplexity_aggregated_output_file_path(config: &JudgeConfig) -> String {
    let model = Model::Local(config.model);
    let model_safe_name = get_model_safe_name(model);
    let file_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join(PERPLEXITY_GENERATE_PERPLEXITY_AGGREGATED_OUTPUT_FILE_NAME);
    file_path.to_string_lossy().to_string()
}

#[pyfunction]
pub fn perplexity_prepare_response_input(
    config: &JudgeConfig,
    debug_limit: Option<usize>,
) {
    let model_safe_name = get_model_safe_name(Model::Local(config.model));
    let lang = match &config.experiment {
        JudgeExperiment::Perplexity { lang, .. } => lang,
        _ => panic!("Invalid experiment type for perplexity_prepare_response_input"),
    };
    let output_file_path = perplexity_generate_response_input_file_path(config);
    // read or create
    let dataset_path = JUDGE_BASE_DATASET_PATH
        .join("two_answers_same_lang")
        .join(format!("{}.jsonl", lang));
    if !Path::new(&dataset_path).exists() {
        println!(
            "Two answers same language dataset for language {} not found. Generating...",
            lang
        );
        generate_two_answers_same_lang_dataset(&lang);
    }
    let dataset_entries =
        load_json_lines(&dataset_path).expect("Failed to load two answers same language dataset");
    let dataset_entries_parsed: IndexMap<usize, TwoAnswersSameLangEntry> = dataset_entries
        .into_iter()
        .map(|entry| {
            let parsed: TwoAnswersSameLangEntry = serde_json::from_value(entry)
                .expect("Failed to parse two answers same language entry");
            (parsed.index, parsed)
        })
        .collect();
    let result_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join("response")
        .join(format!("{}.jsonl", lang));
    let processed_ids: HashSet<usize> = match load_json_lines(&result_path) {
        Ok(entries) => entries
            .into_iter()
            .map(|entry| {
                let parsed: ResponseEntry =
                    serde_json::from_value(entry).expect("Failed to parse response entry");
                parsed.index
            })
            .collect(),
        Err(_) => {
            println!(
                "File {} does not exist, assuming no completed entries.",
                result_path.to_string_lossy()
            );
            HashSet::new()
        }
    };
    let perplexity_mask_path = JUDGE_BASE_DATASET_PATH.join("perplexity_mask.jsonl");
    let perplexity_mask_entries =
        load_json_lines(&perplexity_mask_path).expect("Failed to load perplexity mask dataset");
    let perplexity_mask_entries_parsed: IndexMap<usize, PerplexityDatasetMaskEntry> =
        perplexity_mask_entries
            .into_iter()
            .map(|entry| {
                let parsed: PerplexityDatasetMaskEntry =
                    serde_json::from_value(entry).expect("Failed to parse perplexity mask entry");
                (parsed.index, parsed)
            })
            .collect();
    let indices = dataset_entries_parsed.keys();
    let mut aggregated_entries: Vec<GenerateResponseInputEntry> = Vec::new();
    let mut count = 0;
    for index in indices {
        if let Some(limit) = debug_limit {
            if count >= limit {
                break;
            }
            count += 1;
        }
        let mask_entry = &perplexity_mask_entries_parsed
            .get(index)
            .expect("Missing mask entry");
        // only push valid entries
        if mask_entry.valid && !processed_ids.contains(index) {
            let dataset_entry = dataset_entries_parsed
                .get(index)
                .expect("Missing two answers same language entry");
            let input_entry = GenerateResponseInputEntry {
                index: *index,
                question: dataset_entry.question.clone(),
                lang: dataset_entry.lang.clone(),
                subject: dataset_entry.subject.clone(),
            };
            aggregated_entries.push(input_entry);
        }
    }
    // serialize aggregated entries and write to output file
    let aggregated_entries_serialized: Vec<serde_json::Value> = aggregated_entries
        .into_iter()
        .map(|entry| serde_json::to_value(entry).expect("Failed to serialize aggregated entry"))
        .collect();
    write_json_lines_to_file(&output_file_path, &aggregated_entries_serialized)
        .expect("Failed to write aggregated response input dataset");
    println!(
        "Response input dataset for language {} written to {}",
        lang, output_file_path
    );
}

#[pyfunction]
pub fn perplexity_prepare_generate_styled_answers_input(
    config: &JudgeConfig,
    debug_limit: Option<usize>,
) {
    let model_safe_name = get_model_safe_name(Model::Local(config.model));
    let lang = match &config.experiment {
        JudgeExperiment::Perplexity { lang, .. } => lang,
        _ => panic!("Invalid experiment type for perplexity_prepare_styled_answers_input"),
    };
    let output_file_path = perplexity_generate_styled_answers_input_file_path(config);
    let response_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join("response")
        .join(format!("{}.jsonl", lang));
    let two_answers_same_lang_path = JUDGE_BASE_DATASET_PATH
        .join("two_answers_same_lang")
        .join(format!("{}.jsonl", lang));
    let result_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join("styled_answers")
        .join(format!("{}.jsonl", lang));
    let perplexity_mask_path = JUDGE_BASE_DATASET_PATH.join("perplexity_mask.jsonl");
    let response_entries =
        load_json_lines(&response_path).expect("Failed to load response dataset");
    let two_answers_same_lang_entries = load_json_lines(&two_answers_same_lang_path)
        .expect("Failed to load two answers same language dataset");
    // let result_entries =
    //     load_json_lines(&result_path).expect("Failed to load styled answers dataset");
    let perplexity_mask_entries =
        load_json_lines(&perplexity_mask_path).expect("Failed to load perplexity mask dataset");
    let perplexity_mask_entries_parsed: IndexMap<usize, PerplexityDatasetMaskEntry> =
        perplexity_mask_entries
            .into_iter()
            .map(|entry| {
                let parsed: PerplexityDatasetMaskEntry =
                    serde_json::from_value(entry).expect("Failed to parse perplexity mask entry");
                (parsed.index, parsed)
            })
            .collect();
    let response_entries_parsed: IndexMap<usize, ResponseEntry> = response_entries
        .into_iter()
        .map(|entry| {
            let parsed: ResponseEntry =
                serde_json::from_value(entry).expect("Failed to parse response entry");
            (parsed.index, parsed)
        })
        .collect();
    let two_answers_same_lang_entries_parsed: IndexMap<usize, TwoAnswersSameLangEntry> =
        two_answers_same_lang_entries
            .into_iter()
            .map(|entry| {
                let parsed: TwoAnswersSameLangEntry = serde_json::from_value(entry)
                    .expect("Failed to parse two answers same language entry");
                (parsed.index, parsed)
            })
            .collect();
    // let styled_answers_entries_parsed: IndexMap<usize, StyledAnswersEntry> = result_entries
    //     .into_iter()
    //     .map(|entry| {
    //         let parsed: StyledAnswersEntry =
    //             serde_json::from_value(entry).expect("Failed to parse styled answers entry");
    //         (parsed.index, parsed)
    //     })
    //     .collect();
    let processed_ids: HashSet<usize> = match load_json_lines(&result_path) {
        Ok(entries) => entries
            .into_iter()
            .map(|entry| {
                let parsed: StyledAnswersEntry =
                    serde_json::from_value(entry).expect("Failed to parse styled answers entry");
                parsed.index
            })
            .collect(),
        Err(_) => {
            println!(
                "File {} does not exist, assuming no completed entries.",
                result_path.to_string_lossy()
            );
            HashSet::new()
        }
    };
    let indices = response_entries_parsed.keys();
    let mut aggregated_entries: Vec<GenerateStyledAnswersInputEntry> = Vec::new();
    let mut count = 0;
    for index in indices {
        if let Some(limit) = debug_limit {
            if count >= limit {
                break;
            }
            count += 1;
        }
        let mask_entry = &perplexity_mask_entries_parsed
            .get(index)
            .expect("Missing mask entry");
        assert!(
            mask_entry.valid,
            "Mask entry is not valid for index {}, should have been filtered out in previous pass",
            index
        );
        // only push unprocessed entries
        if !processed_ids.contains(index) {
            let response_entry = response_entries_parsed
                .get(index)
                .expect("Missing response entry");
            let two_answers_entry = two_answers_same_lang_entries_parsed
                .get(index)
                .expect("Missing two answers same language entry");
            let original_answer_correct = two_answers_entry.answer_correct.clone();
            let original_answer_incorrect = two_answers_entry.answer_incorrect.clone();
            let input_entry = GenerateStyledAnswersInputEntry {
                index: *index,
                question: response_entry.question.clone(),
                response: response_entry.response.clone(),
                original_answer_correct,
                original_answer_incorrect,
                lang: response_entry.lang.clone(),
                subject: response_entry.subject.clone(),
            };
            aggregated_entries.push(input_entry);
        }
    }
    // serialize aggregated entries and write to output file
    let aggregated_entries_serialized: Vec<serde_json::Value> = aggregated_entries
        .into_iter()
        .map(|entry| serde_json::to_value(entry).expect("Failed to serialize aggregated entry"))
        .collect();
    write_json_lines_to_file(&output_file_path, &aggregated_entries_serialized)
        .expect("Failed to write aggregated styled answers input dataset");
    println!(
        "Styled answers input dataset for language {} written to {}",
        lang, output_file_path
    );
}

#[pyfunction]
pub fn perplexity_prepare_generate_perplexity_aggregated_input(
    config: &JudgeConfig,
    debug_limit: Option<usize>,
) {
    let model_safe_name = get_model_safe_name(Model::Local(config.model));
    let lang = match &config.experiment {
        JudgeExperiment::Perplexity { lang, .. } => lang,
        _ => panic!("Invalid experiment type for perplexity_prepare_aggregated_input"),
    };
    let output_file_path = perplexity_generate_perplexity_aggregated_input_file_path(config);
    let styled_answers_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join("styled_answers")
        .join(format!("{}.jsonl", lang));
    let correct_result_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join("perplexity")
        .join(format!("{}_correct.jsonl", lang));
    let incorrect_result_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join("perplexity")
        .join(format!("{}_incorrect.jsonl", lang));

    let perplexity_mask_path = JUDGE_BASE_DATASET_PATH.join("perplexity_mask.jsonl");

    if !Path::new(&perplexity_mask_path).exists() {
        println!("Perplexity mask dataset not found. Generating...");
        generate_perplexity_dataset_mask();
    }
    let styled_answers_entries =
        load_json_lines(&styled_answers_path).expect("Failed to load styled answers dataset");

    let perplexity_mask_entries =
        load_json_lines(&perplexity_mask_path).expect("Failed to load perplexity mask dataset");

    // parse all entries
    let styled_answers_entries_parsed: IndexMap<usize, StyledAnswersEntry> = styled_answers_entries
        .into_iter()
        .map(|entry| {
            let parsed: StyledAnswersEntry =
                serde_json::from_value(entry).expect("Failed to parse correct one answer entry");
            (parsed.index, parsed)
        })
        .collect();

    let correct_result_ids: HashSet<usize> = match load_json_lines(&correct_result_path) {
        Ok(entries) => entries
            .into_iter()
            .map(|entry| {
                let parsed: PerplexityEntry = serde_json::from_value(entry)
                    .expect("Failed to parse correct one answer result entry");
                parsed.index
            })
            .collect(),
        Err(_) => {
            println!(
                "File {:?} does not exist, assuming no completed entries.",
                correct_result_path
            );
            HashSet::new()
        }
    };
    let incorrect_result_ids: HashSet<usize> = match load_json_lines(&incorrect_result_path) {
        Ok(entries) => entries
            .into_iter()
            .map(|entry| {
                let parsed: PerplexityEntry = serde_json::from_value(entry)
                    .expect("Failed to parse incorrect one answer result entry");
                parsed.index
            })
            .collect(),
        Err(_) => {
            println!(
                "File {:?} does not exist, assuming no completed entries.",
                incorrect_result_path
            );
            HashSet::new()
        }
    };
    let perplexity_mask_entries_parsed: IndexMap<usize, PerplexityDatasetMaskEntry> =
        perplexity_mask_entries
            .into_iter()
            .map(|entry| {
                let parsed: PerplexityDatasetMaskEntry =
                    serde_json::from_value(entry).expect("Failed to parse perplexity mask entry");
                (parsed.index, parsed)
            })
            .collect();
    // let dataset_length = styled_answers_entries_parsed.len();
    // assert_eq!(dataset_length, perplexity_mask_entries_parsed.len());
    let indices = styled_answers_entries_parsed.keys();
    let mut combined_entries: Vec<GeneratePerplexityAggregatedInputEntry> = Vec::new();
    let mut count = 0;
    for index in indices {
        if let Some(limit) = debug_limit {
            if count >= limit {
                break;
            }
            count += 1;
        }
        let mask_entry = &perplexity_mask_entries_parsed
            .get(index)
            .expect("Missing mask entry");
        assert!(
            mask_entry.valid,
            "Mask entry is not valid for index {}, should have been filtered out in previous pass",
            index
        );
        // only push valid entries
        if !correct_result_ids.contains(index) {
            let styled_answers_entry = styled_answers_entries_parsed
                .get(index)
                .expect("Missing correct one answer entry");
            // only extract the correct answer part
            let input_entry = GeneratePerplexityAggregatedInputEntry {
                index: *index,
                question: styled_answers_entry.question.clone(),
                styled_response: styled_answers_entry.styled_response_correct.clone(),
                is_correct: true,
                lang: styled_answers_entry.lang.clone(),
                subject: styled_answers_entry.subject.clone(),
            };
            combined_entries.push(input_entry);
        }
        if !incorrect_result_ids.contains(index) {
            let styled_answers_entry = styled_answers_entries_parsed
                .get(index)
                .expect("Missing incorrect one answer entry");
            // only extract the incorrect answer part
            let input_entry = GeneratePerplexityAggregatedInputEntry {
                index: *index,
                question: styled_answers_entry.question.clone(),
                styled_response: styled_answers_entry.styled_response_incorrect.clone(),
                is_correct: false,
                lang: styled_answers_entry.lang.clone(),
                subject: styled_answers_entry.subject.clone(),
            };
            combined_entries.push(input_entry);
        }
    }
    // serialize combined entries and write to output file
    let combined_entries_serialized: Vec<serde_json::Value> = combined_entries
        .into_iter()
        .map(|entry| serde_json::to_value(entry).expect("Failed to serialize combined entry"))
        .collect();
    write_json_lines_to_file(&output_file_path, &combined_entries_serialized)
        .expect("Failed to write combined perplexity dataset");
    println!(
        "Concatenated perplexity dataset for language {} written to {}",
        lang, output_file_path
    );
}

#[pyfunction]
pub fn perplexity_dispatch_response_results(config: &JudgeConfig) {
    let model_safe_name = get_model_safe_name(Model::Local(config.model));
    let lang = match &config.experiment {
        JudgeExperiment::Perplexity { lang, .. } => lang,
        _ => panic!("Invalid experiment type for dispatch_response_results"),
    };
    let input_file_path = perplexity_generate_response_output_file_path(config);
    println!(
        "Dispatching response results for model: {}, lang: {}, input file: {}",
        model_safe_name, lang, input_file_path
    );
    let result_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join("response")
        .join(format!("{}.jsonl", lang));
    let mut result_entries: Vec<ResponseEntry> = match load_json_lines(&result_path) {
        Ok(entries) => entries
            .into_iter()
            .map(|entry| {
                let parsed = serde_json::from_value::<ResponseEntry>(entry)
                    .expect("Failed to parse response entry");
                parsed
            })
            .collect(),
        Err(e) => {
            println!("Cannot open file: {}, assuming empty result file", e);
            vec![]
        }
    };
    let aggregated_entries = match load_json_lines(&input_file_path) {
        Ok(entries) => entries,
        Err(_) => {
            println!(
                "Input file does not exist: {}, skipping dispatching.",
                input_file_path
            );
            return;
        }
    };
    let aggregated_entries_parsed: HashMap<usize, ResponseEntry> = aggregated_entries
        .into_iter()
        .map(|entry| {
            let parsed = serde_json::from_value::<ResponseEntry>(entry)
                .expect("Failed to parse combined response entry");
            (parsed.index, parsed)
        })
        .collect();
    let mut remaining_indices: HashSet<usize> = get_valid_perplexity_indices();
    for entry in result_entries.iter() {
        let result = remaining_indices.remove(&entry.index);
        assert!(
            result,
            "Duplicate or invalid index found in response results: {}",
            entry.index
        );
    }
    let mut missing_index_count = 0;
    for index in remaining_indices {
        if let Some(entry) = aggregated_entries_parsed.get(&index) {
            result_entries.push(entry.clone());
        } else {
            missing_index_count += 1;
        }
    }
    if missing_index_count > 0 {
        println!(
            "Warning: Missing {} entries for model: {}, lang: {}",
            missing_index_count, model_safe_name, lang
        );
    }
    result_entries.sort_by_key(|e| e.index);
    let serialized = result_entries
        .iter()
        .map(|e| serde_json::to_value(e).expect("Failed to serialize response entry"))
        .collect::<Vec<_>>();
    write_json_lines_to_file(&result_path, &serialized).expect("Failed to write json lines");
    println!(
        "Dispatched response results for model: {}, lang: {}",
        model_safe_name, lang
    );
    // remove the aggregated output file after dispatching
    std::fs::remove_file(&input_file_path)
        .expect("Failed to remove aggregated response output file after dispatching");
    println!(
        "Removed aggregated response output file: {}",
        input_file_path
    );
}

#[pyfunction]
pub fn perplexity_dispatch_styled_answers_results(config: &JudgeConfig) {
    let model_safe_name = get_model_safe_name(Model::Local(config.model));
    let lang = match &config.experiment {
        JudgeExperiment::Perplexity { lang, .. } => lang,
        _ => panic!("Invalid experiment type for dispatch_styled_answers_results"),
    };
    let input_file_path = perplexity_generate_styled_answers_output_file_path(config);
    println!(
        "Dispatching styled answers results for model: {}, lang: {}, input file: {}",
        model_safe_name, lang, input_file_path
    );
    let result_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join("styled_answers")
        .join(format!("{}.jsonl", lang));
    let mut result_entries: Vec<StyledAnswersEntry> = match load_json_lines(&result_path) {
        Ok(entries) => entries
            .into_iter()
            .map(|entry| {
                let parsed = serde_json::from_value::<StyledAnswersEntry>(entry)
                    .expect("Failed to parse styled answers entry");
                parsed
            })
            .collect(),
        Err(e) => {
            println!("Cannot open file: {}, assuming empty result file", e);
            vec![]
        }
    };
    let aggregated_entries = match load_json_lines(&input_file_path) {
        Ok(entries) => entries,
        Err(_) => {
            println!(
                "Input file does not exist: {}, skipping dispatching.",
                input_file_path
            );
            return;
        }
    };
    let aggregated_entries_parsed: HashMap<usize, StyledAnswersEntry> = aggregated_entries
        .into_iter()
        .map(|entry| {
            let parsed = serde_json::from_value::<StyledAnswersEntry>(entry)
                .expect("Failed to parse combined styled answers entry");
            (parsed.index, parsed)
        })
        .collect();
    let mut remaining_indices: HashSet<usize> = get_valid_perplexity_indices();
    for entry in result_entries.iter() {
        let result = remaining_indices.remove(&entry.index);
        assert!(
            result,
            "Duplicate or invalid index found in styled answers results: {}",
            entry.index
        );
    }
    let mut missing_index_count = 0;
    for index in remaining_indices {
        if let Some(entry) = aggregated_entries_parsed.get(&index) {
            result_entries.push(entry.clone());
        } else {
            missing_index_count += 1;
        }
    }
    if missing_index_count > 0 {
        println!(
            "Warning: Missing {} entries for model: {}, lang: {}",
            missing_index_count, model_safe_name, lang
        );
    }
    result_entries.sort_by_key(|e| e.index);
    let serialized = result_entries
        .iter()
        .map(|e| serde_json::to_value(e).expect("Failed to serialize styled answers entry"))
        .collect::<Vec<_>>();
    write_json_lines_to_file(&result_path, &serialized).expect("Failed to write json lines");
    println!(   
        "Dispatched styled answers results for model: {}, lang: {}",
        model_safe_name, lang
    );
    // remove the aggregated output file after dispatching
    std::fs::remove_file(&input_file_path)
        .expect("Failed to remove aggregated styled answers output file after dispatching");
    println!(
        "Removed aggregated styled answers output file: {}",
        input_file_path
    );
}

#[pyfunction]
pub fn perplexity_dispatch_generate_perplexity_results(config: &JudgeConfig) {
    let model_safe_name = get_model_safe_name(Model::Local(config.model));
    let lang = match &config.experiment {
        JudgeExperiment::Perplexity { lang, .. } => lang,
        _ => panic!("Invalid experiment type for dispatch_perplexity_results"),
    };
    let input_file_path = perplexity_generate_perplexity_aggregated_output_file_path(config);
    println!(
        "Dispatching perplexity results for model: {}, lang: {}, input file: {}",
        model_safe_name, lang, input_file_path
    );
    let correct_result_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join("perplexity")
        .join(format!("{}_correct.jsonl", lang));
    let incorrect_result_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join("perplexity")
        .join(format!("{}_incorrect.jsonl", lang));
    let mut correct_result_entries: Vec<PerplexityEntry> =
        match load_json_lines(&correct_result_path) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry| {
                    let parsed = serde_json::from_value::<PerplexityEntry>(entry)
                        .expect("Failed to parse correct perplexity entry");
                    parsed
                })
                .collect(),
            Err(e) => {
                println!("Cannot open file: {}, assuming empty result file", e);
                vec![]
            }
        };
    let mut incorrect_result_entries: Vec<PerplexityEntry> =
        match load_json_lines(&incorrect_result_path) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry| {
                    let parsed = serde_json::from_value::<PerplexityEntry>(entry)
                        .expect("Failed to parse incorrect perplexity entry");
                    parsed
                })
                .collect(),
            Err(e) => {
                println!("Cannot open file: {}, assuming empty result file", e);
                vec![]
            }
        };
    let combined_entries = match load_json_lines(&input_file_path) {
        Ok(entries) => entries,
        Err(_) => {
            println!(
                "Input file does not exist: {}, skipping dispatching.",
                input_file_path
            );
            return;
        }
    };
    let aggregated_entries_parsed: HashMap<(usize, bool), PerplexityEntry> = combined_entries
        .into_iter()
        .map(|entry| {
            let parsed = serde_json::from_value::<PerplexityEntry>(entry)
                .expect("Failed to parse combined perplexity entry");
            ((parsed.index, parsed.is_correct), parsed)
        })
        .collect();
    let mut remaining_correct_indices: HashSet<usize> = get_valid_perplexity_indices();
    let mut remainint_incorrect_indices: HashSet<usize> = remaining_correct_indices.clone();
    for entry in correct_result_entries.iter() {
        let result = remaining_correct_indices.remove(&entry.index);
        assert!(
            result,
            "Duplicate or invalid index found in correct results: {}",
            entry.index
        );
    }
    for entry in incorrect_result_entries.iter() {
        let result = remainint_incorrect_indices.remove(&entry.index);
        assert!(
            result,
            "Duplicate or invalid index found in incorrect results: {}",
            entry.index
        );
    }
    let mut missing_correct_index_count = 0;
    let mut missing_incorrect_index_count = 0;
    for index in remaining_correct_indices {
        let key = (index, true);
        if let Some(entry) = aggregated_entries_parsed.get(&key) {
            correct_result_entries.push(entry.clone());
        } else {
            missing_correct_index_count += 1;
        }
    }
    for index in remainint_incorrect_indices {
        let key = (index, false);
        if let Some(entry) = aggregated_entries_parsed.get(&key) {
            incorrect_result_entries.push(entry.clone());
        } else {
            missing_incorrect_index_count += 1;
        }
    }
    if missing_correct_index_count > 0 {
        println!(
            "Warning: Missing {} correct entries for model: {}, lang: {}",
            missing_correct_index_count, model_safe_name, lang
        );
    }
    if missing_incorrect_index_count > 0 {
        println!(
            "Warning: Missing {} incorrect entries for model: {}, lang: {}",
            missing_incorrect_index_count, model_safe_name, lang
        );
    }
    correct_result_entries.sort_by_key(|e| e.index);
    incorrect_result_entries.sort_by_key(|e| e.index);
    let correct_serialized = correct_result_entries
        .iter()
        .map(|e| serde_json::to_value(e).expect("Failed to serialize correct perplexity entry"))
        .collect::<Vec<_>>();
    let incorrect_serialized = incorrect_result_entries
        .iter()
        .map(|e| serde_json::to_value(e).expect("Failed to serialize incorrect perplexity entry"))
        .collect::<Vec<_>>();
    write_json_lines_to_file(&correct_result_path, &correct_serialized)
        .expect("Failed to write json lines");
    write_json_lines_to_file(&incorrect_result_path, &incorrect_serialized)
        .expect("Failed to write json lines");
    println!(
        "Dispatched perplexity results for model: {}, lang: {}",
        model_safe_name, lang
    );
    // remove the aggregated output file after dispatching
    std::fs::remove_file(&input_file_path)
        .expect("Failed to remove aggregated perplexity output file after dispatching");
    println!(
        "Removed aggregated perplexity output file: {}",
        input_file_path
    );
}
