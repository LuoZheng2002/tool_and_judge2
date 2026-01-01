use core::panic;
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use pyo3::pyfunction;
use serde::{Deserialize, Serialize};

use crate::{
    config::{JudgeConfig, JudgeExperiments, Model},
    judge::{
        base_paths::{JUDGE_BASE_DATASET_PATH, JUDGE_BASE_RESULT_PATH},
        generate_dataset::{TwoAnswersEntry, generate_two_answers_dataset, get_preference_indices},
    },
    utils::{get_model_safe_name, load_json_lines, write_json_lines_to_file},
};

#[derive(Clone, Serialize, Deserialize)]
pub struct Preference {
    pub preferred_answer: usize,
    pub logprob_signed_difference: f32,
    pub logprob1: f32,
    pub logprob2: f32,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct PreferenceResultEntry {
    pub index: usize,
    pub preference: Result<Preference, String>,
    pub question: String,
    pub answer1: String,
    pub answer2: String,
    pub lang1: String,
    pub lang2: String,
    pub is_correct1: bool,
    pub is_correct2: bool,
    pub subject: String,
}

const PREFERENCE_AGGREGATED_INPUT_FILE_NAME: &str = "preference_aggregated_input.jsonl";
const PREFERENCE_AGGREGATED_OUTPUT_FILE_NAME: &str = "preference_aggregated_output.jsonl";

#[pyfunction]
pub fn preference_aggregated_input_file_path(config: &JudgeConfig) -> String {
    let model = Model::Local(config.model);
    let model_safe_name = get_model_safe_name(model);
    let file_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join(PREFERENCE_AGGREGATED_INPUT_FILE_NAME);
    file_path.to_str().unwrap().to_string()
}

#[pyfunction]
pub fn preference_aggregated_output_file_path(config: &JudgeConfig) -> String {
    let model = config.model;
    let model_safe_name = get_model_safe_name(Model::Local(model));
    let file_path = JUDGE_BASE_RESULT_PATH
        .join(&model_safe_name)
        .join(PREFERENCE_AGGREGATED_OUTPUT_FILE_NAME);
    file_path.to_str().unwrap().to_string()
}

#[pyfunction]
pub fn preference_prepare_aggregated_input(config: &JudgeConfig, debug_limit: Option<usize>) {
    let model_safe_name = get_model_safe_name(Model::Local(config.model));
    let output_file_path = preference_aggregated_input_file_path(config);
    let mut aggregated_entries: Vec<TwoAnswersEntry> = Vec::new();
    let language_pairs: Vec<(String, String)> = match &config.experiments {
        JudgeExperiments::Vllm {
            preference_experiments,
            ..
        } => preference_experiments
            .iter()
            .map(|exp| (exp.lang1.clone(), exp.lang2.clone()))
            .collect(),
        JudgeExperiments::HuggingFace { .. } => {
            panic!("Preference experiments are not supported for HuggingFace backend");
        }
    };
    for (lang1, lang2) in language_pairs.iter() {
        for (lang1_correct, lang2_correct) in
            [(true, false), (false, true)].iter()
        {
            let lang1_correct_str = if *lang1_correct {
                "correct"
            } else {
                "incorrect"
            };
            let lang2_correct_str = if *lang2_correct {
                "correct"
            } else {
                "incorrect"
            };
            let dataset_path = JUDGE_BASE_DATASET_PATH.join("two_answers").join(format!(
                "{}_{}_{}_{}.jsonl",
                lang1, lang1_correct_str, lang2, lang2_correct_str
            ));
            let result_path = JUDGE_BASE_RESULT_PATH
                .join(&model_safe_name)
                .join("preference")
                .join(format!(
                    "{}_{}_{}_{}.jsonl",
                    lang1, lang1_correct_str, lang2, lang2_correct_str
                ));
            if !Path::new(&dataset_path).exists() {
                println!(
                    "Two answers datasets for languages {} and {} not found. Generating...",
                    lang1, lang2
                );
                generate_two_answers_dataset(&lang1, &lang2);
            }
            let dataset_entries = load_json_lines(&dataset_path).expect("Failed to load dataset");

            let dataset_parsed: Vec<TwoAnswersEntry> = dataset_entries
                .into_iter()
                .map(|entry| serde_json::from_value(entry).expect("Failed to parse dataset entry"))
                .collect();

            let result_ids = match load_json_lines(&result_path) {
                Ok(entries) => entries
                    .into_iter()
                    .map(|entry| {
                        let parsed: PreferenceResultEntry =
                            serde_json::from_value(entry).expect("Failed to parse result entry");
                        parsed.index
                    })
                    .collect(),
                Err(_) => {
                    println!(
                        "File {:?} does not exist, assuming no completed entries.",
                        result_path
                    );
                    HashSet::new()
                }
            };
            let mut count = 0;
            for entry in dataset_parsed {
                if let Some(limit) = debug_limit {
                    if count >= limit {
                        break;
                    }
                    count += 1;
                }
                if !result_ids.contains(&entry.index) {
                    aggregated_entries.push(entry);
                }
            }
        }
    }
    let aggregated_entries_serialized: Vec<serde_json::Value> = aggregated_entries
        .iter()
        .map(|entry| serde_json::to_value(entry).expect("Failed to serialize aggregated entry"))
        .collect();
    // write to output file
    write_json_lines_to_file(&output_file_path, &aggregated_entries_serialized)
        .expect("Failed to write aggregated dataset");
    println!(
        "Concatenated two answers dataset for language pairs{:?} written to {:?}",
        language_pairs, output_file_path
    );
}

#[pyfunction]
pub fn preference_dispatch_preference_results(config: &JudgeConfig) {
    let model_safe_name = get_model_safe_name(Model::Local(config.model));
    let language_pairs: Vec<(String, String)> = match &config.experiments {
        JudgeExperiments::Vllm {
            preference_experiments,
            ..
        } => preference_experiments
            .iter()
            .map(|exp| (exp.lang1.clone(), exp.lang2.clone()))
            .collect(),
        JudgeExperiments::HuggingFace { .. } => {
            panic!("Preference experiments are not supported for HuggingFace backend");
        }
    };
    let input_file_path = preference_aggregated_output_file_path(config);
    let combined_entries = load_json_lines(&input_file_path).expect("Failed to load input file");
    // key: (lang1, lang2, index, is_correct1, is_correct2)
    let combined_entries_parsed: HashMap<(String, String, usize, bool, bool), PreferenceResultEntry> =
        combined_entries
            .into_iter()
            .map(|entry| {
                let parsed = serde_json::from_value::<PreferenceResultEntry>(entry)
                    .expect("Failed to parse combined preference entry");
                let is_correct1 = parsed.is_correct1;
                let is_correct2 = parsed.is_correct2;
                ((parsed.lang1.clone(), parsed.lang2.clone(), parsed.index, is_correct1, is_correct2), parsed)
            })
            .collect();
    if !Path::new(&input_file_path).exists() {
        println!(
            "Input file does not exist: {}, skipping dispatching.",
            input_file_path
        );
        return;
    }
    println!(
        "Dispatching preference results for model: {}, language pairs: {:?}, input file: {}",
        model_safe_name, language_pairs, input_file_path
    );
    for (lang1, lang2) in language_pairs.iter() {
        for (lang1_correct, lang2_correct) in
            [(true, false), (false, true)].iter()
        {
            let lang1_correct_str = if *lang1_correct {
                "correct"
            } else {
                "incorrect"
            };
            let lang2_correct_str = if *lang2_correct {
                "correct"
            } else {
                "incorrect"
            };

            let result_path = JUDGE_BASE_RESULT_PATH
                .join(&model_safe_name)
                .join("preference")
                .join(format!(
                    "{}_{}_{}_{}.jsonl",
                    lang1, lang1_correct_str, lang2, lang2_correct_str
                ));
            let mut result_entries: Vec<PreferenceResultEntry> = match load_json_lines(&result_path)
            {
                Ok(entries) => entries
                    .into_iter()
                    .map(|entry| {
                        let parsed = serde_json::from_value::<PreferenceResultEntry>(entry)
                            .expect("Failed to parse preference entry");
                        parsed
                    })
                    .collect(),
                Err(e) => {
                    println!("Cannot open file: {}, assuming empty result file", e);
                    vec![]
                }
            };

            let mut remaining_indices: HashSet<usize> = get_preference_indices();
            for entry in result_entries.iter() {
                let result = remaining_indices.remove(&entry.index);
                assert!(
                    result,
                    "Duplicate or invalid index found in results: {}",
                    entry.index
                );
            }
            let mut missing_count = 0;
            for index in remaining_indices {
                let key = (
                    lang1.clone(),
                    lang2.clone(),
                    index,
                    *lang1_correct,
                    *lang2_correct,
                );
                if let Some(entry) = combined_entries_parsed.get(&key) {
                    result_entries.push(entry.clone());
                } else {
                    missing_count += 1;
                }
            }
            if missing_count > 0 {
                println!(
                    "Warning: Missing entries for model: {}, lang1: {}, lang2: {}: missing entries count: {}",
                    model_safe_name, lang1, lang2, missing_count,
                );
            }
            result_entries.sort_by_key(|e| e.index);
            let result_entries_serialized: Vec<serde_json::Value> = result_entries
                .iter()
                .map(|e| {
                    serde_json::to_value(e).expect(
                        "Failed to serialize preference entry",
                    )
                })
                .collect();
            write_json_lines_to_file(&result_path, &result_entries_serialized)
                .expect("Failed to write json lines");
        }
        println!(
            "Dispatched preference results for model: {}, lang1: {}, lang2: {}",
            model_safe_name, lang1, lang2
        );
    }
    // remove the aggregated output file
    std::fs::remove_file(&input_file_path).expect("Failed to remove aggregated output file");
    println!(
        "Removed aggregated output file: {}",
        input_file_path
    );
}
