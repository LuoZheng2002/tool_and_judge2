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
        let lang1_correct_lang2_incorrect_dataset_path = JUDGE_BASE_DATASET_PATH
            .join("two_answers")
            .join(format!("{}_correct_{}_incorrect.jsonl", lang1, lang2));
        let lang1_incorrect_lang2_correct_dataset_path = JUDGE_BASE_DATASET_PATH
            .join("two_answers")
            .join(format!("{}_incorrect_{}_correct.jsonl", lang1, lang2));
        let both_correct_dataset_path = JUDGE_BASE_DATASET_PATH
            .join("two_answers")
            .join(format!("{}_correct_{}_correct.jsonl", lang1, lang2));
        let both_incorrect_dataset_path = JUDGE_BASE_DATASET_PATH
            .join("two_answers")
            .join(format!("{}_incorrect_{}_incorrect.jsonl", lang1, lang2));
        let lang1_correct_lang2_incorrect_result_path = JUDGE_BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("preference")
            .join(format!("{}_correct_{}_incorrect.jsonl", lang1, lang2));
        let lang1_incorrect_lang2_correct_result_path = JUDGE_BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("preference")
            .join(format!("{}_incorrect_{}_correct.jsonl", lang1, lang2));
        let both_correct_result_path = JUDGE_BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("preference")
            .join(format!("{}_correct_{}_correct.jsonl", lang1, lang2));
        let both_incorrect_result_path = JUDGE_BASE_RESULT_PATH
            .join(&model_safe_name)
            .join("preference")
            .join(format!("{}_incorrect_{}_incorrect.jsonl", lang1, lang2));
        let output_paths_exist = [
            &lang1_correct_lang2_incorrect_dataset_path,
            &lang1_incorrect_lang2_correct_dataset_path,
            &both_correct_dataset_path,
            &both_incorrect_dataset_path,
        ]
        .iter()
        .all(|path| Path::new(path).exists());
        if !output_paths_exist {
            println!(
                "Two answers datasets for languages {} and {} not found. Generating...",
                lang1, lang2
            );
            generate_two_answers_dataset(&lang1, &lang2);
        }
        let lang1_correct_lang2_incorrect_dataset_entries =
            load_json_lines(&lang1_correct_lang2_incorrect_dataset_path)
                .expect("Failed to load lang1 correct lang2 incorrect dataset");
        let lang1_incorrect_lang2_correct_dataset_entries =
            load_json_lines(&lang1_incorrect_lang2_correct_dataset_path)
                .expect("Failed to load lang1 incorrect lang2 correct dataset");
        let both_correct_dataset_entries = load_json_lines(&both_correct_dataset_path)
            .expect("Failed to load both correct dataset");
        let both_incorrect_dataset_entries = load_json_lines(&both_incorrect_dataset_path)
            .expect("Failed to load both incorrect dataset");

        let lang1_correct_lang2_incorrect_dataset_parsed: Vec<TwoAnswersEntry> =
            lang1_correct_lang2_incorrect_dataset_entries
                .into_iter()
                .map(|entry| {
                    serde_json::from_value(entry)
                        .expect("Failed to parse lang1 correct lang2 incorrect dataset entry")
                })
                .collect();
        let lang1_incorrect_lang2_correct_dataset_parsed: Vec<TwoAnswersEntry> =
            lang1_incorrect_lang2_correct_dataset_entries
                .into_iter()
                .map(|entry| {
                    serde_json::from_value(entry)
                        .expect("Failed to parse lang1 incorrect lang2 correct dataset entry")
                })
                .collect();
        let both_correct_dataset_parsed: Vec<TwoAnswersEntry> = both_correct_dataset_entries
            .into_iter()
            .map(|entry| {
                serde_json::from_value(entry).expect("Failed to parse both correct dataset entry")
            })
            .collect();
        let both_incorrect_dataset_parsed: Vec<TwoAnswersEntry> = both_incorrect_dataset_entries
            .into_iter()
            .map(|entry| {
                serde_json::from_value(entry).expect("Failed to parse both incorrect dataset entry")
            })
            .collect();

        let lang1_correct_lang2_incorrect_result_ids =
            match load_json_lines(&lang1_correct_lang2_incorrect_result_path) {
                Ok(entries) => entries
                    .into_iter()
                    .map(|entry| {
                        let parsed: PreferenceResultEntry = serde_json::from_value(entry)
                            .expect("Failed to parse lang1 correct lang2 incorrect result entry");
                        parsed.index
                    })
                    .collect(),
                Err(_) => {
                    println!(
                        "File {:?} does not exist, assuming no completed entries.",
                        lang1_correct_lang2_incorrect_result_path
                    );
                    HashSet::new()
                }
            };
        let lang1_incorrect_lang2_correct_result_ids =
            match load_json_lines(&lang1_incorrect_lang2_correct_result_path) {
                Ok(entries) => entries
                    .into_iter()
                    .map(|entry| {
                        let parsed: PreferenceResultEntry = serde_json::from_value(entry)
                            .expect("Failed to parse lang1 incorrect lang2 correct result entry");
                        parsed.index
                    })
                    .collect(),
                Err(_) => {
                    println!(
                        "File {:?} does not exist, assuming no completed entries.",
                        lang1_incorrect_lang2_correct_result_path
                    );
                    HashSet::new()
                }
            };
        let both_correct_result_ids = match load_json_lines(&both_correct_result_path) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry| {
                    let parsed: PreferenceResultEntry = serde_json::from_value(entry)
                        .expect("Failed to parse both correct result entry");
                    parsed.index
                })
                .collect(),
            Err(_) => {
                println!(
                    "File {:?} does not exist, assuming no completed entries.",
                    both_correct_result_path
                );
                HashSet::new()
            }
        };
        let both_incorrect_result_ids = match load_json_lines(&both_incorrect_result_path) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry| {
                    let parsed: PreferenceResultEntry = serde_json::from_value(entry)
                        .expect("Failed to parse both incorrect result entry");
                    parsed.index
                })
                .collect(),
            Err(_) => {
                println!(
                    "File {:?} does not exist, assuming no completed entries.",
                    both_incorrect_result_path
                );
                HashSet::new()
            }
        };

        // conditionally concatenate

        let mut count = 0;
        for entry in lang1_correct_lang2_incorrect_dataset_parsed {
            if let Some(limit) = debug_limit {
                if count >= limit {
                    break;
                }
                count += 1;
            }
            if !lang1_correct_lang2_incorrect_result_ids.contains(&entry.index) {
                aggregated_entries.push(entry);
            }
        }
        count = 0;
        for entry in lang1_incorrect_lang2_correct_dataset_parsed {
            if let Some(limit) = debug_limit {
                if count >= limit {
                    break;
                }
                count += 1;
            }
            if !lang1_incorrect_lang2_correct_result_ids.contains(&entry.index) {
                aggregated_entries.push(entry);
            }
        }
        count = 0;
        for entry in both_correct_dataset_parsed {
            if let Some(limit) = debug_limit {
                if count >= limit {
                    break;
                }
                count += 1;
            }
            if !both_correct_result_ids.contains(&entry.index) {
                aggregated_entries.push(entry);
            }
        }
        count = 0;
        for entry in both_incorrect_dataset_parsed {
            if let Some(limit) = debug_limit {
                if count >= limit {
                    break;
                }
                count += 1;
            }
            if !both_incorrect_result_ids.contains(&entry.index) {
                aggregated_entries.push(entry);
            }
        }
    }
    let aggregated_entries_serialized: Vec<serde_json::Value> = aggregated_entries
        .iter()
        .map(|entry| serde_json::to_value(entry).expect("Failed to serialize combined entry"))
        .collect();
    // write to output file
    write_json_lines_to_file(&output_file_path, &aggregated_entries_serialized)
        .expect("Failed to write combined two answers dataset");
    println!(
        "Concatenated two answers dataset for language pairs{:?} written to {:?}",
        language_pairs, output_file_path
    );
}

#[pyfunction]
pub fn dispatch_preference_results(
    model_safe_name: &str,
    lang1: &str,
    lang2: &str,
    input_file_path: &str,
) {
    println!(
        "Dispatching preference results for model: {}, lang1: {}, lang2: {}, input file: {}",
        model_safe_name, lang1, lang2, input_file_path
    );
    let lang1_correct_lang2_incorrect_path = format!(
        "judge/result/{}/preference/{}_correct_{}_incorrect.jsonl",
        model_safe_name, lang1, lang2
    );
    let lang1_incorrect_lang2_correct_path = format!(
        "judge/result/{}/preference/{}_incorrect_{}_correct.jsonl",
        model_safe_name, lang1, lang2
    );
    let both_correct_path = format!(
        "judge/result/{}/preference/{}_correct_{}_correct.jsonl",
        model_safe_name, lang1, lang2
    );
    let both_incorrect_path = format!(
        "judge/result/{}/preference/{}_incorrect_{}_incorrect.jsonl",
        model_safe_name, lang1, lang2
    );
    let mut lang1_correct_lang2_incorrect_entries: Vec<PreferenceResultEntry> =
        match load_json_lines(&lang1_correct_lang2_incorrect_path) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry| {
                    let parsed = serde_json::from_value::<PreferenceResultEntry>(entry)
                        .expect("Failed to parse lang1 correct lang2 incorrect preference entry");
                    parsed
                })
                .collect(),
            Err(e) => {
                println!("Cannot open file: {}, assuming empty result file", e);
                vec![]
            }
        };
    let mut lang1_incorrect_lang2_correct_entries: Vec<PreferenceResultEntry> =
        match load_json_lines(&lang1_incorrect_lang2_correct_path) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry| {
                    let parsed = serde_json::from_value::<PreferenceResultEntry>(entry)
                        .expect("Failed to parse lang1 incorrect lang2 correct preference entry");
                    parsed
                })
                .collect(),
            Err(e) => {
                println!("Cannot open file: {}, assuming empty result file", e);
                vec![]
            }
        };
    let mut both_correct_entries: Vec<PreferenceResultEntry> =
        match load_json_lines(&both_correct_path) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry| {
                    let parsed = serde_json::from_value::<PreferenceResultEntry>(entry)
                        .expect("Failed to parse both correct preference entry");
                    parsed
                })
                .collect(),
            Err(e) => {
                println!("Cannot open file: {}, assuming empty result file", e);
                vec![]
            }
        };
    let mut both_incorrect_entries: Vec<PreferenceResultEntry> =
        match load_json_lines(&both_incorrect_path) {
            Ok(entries) => entries
                .into_iter()
                .map(|entry| {
                    let parsed = serde_json::from_value::<PreferenceResultEntry>(entry)
                        .expect("Failed to parse both incorrect preference entry");
                    parsed
                })
                .collect(),
            Err(e) => {
                println!("Cannot open file: {}, assuming empty result file", e);
                vec![]
            }
        };
    if !Path::new(input_file_path).exists() {
        println!(
            "Input file does not exist: {}, skipping dispatching.",
            input_file_path
        );
        return;
    }
    let combined_entries = load_json_lines(input_file_path).expect("Failed to load input file");
    let combined_entries_parsed: HashMap<(usize, bool, bool), PreferenceResultEntry> =
        combined_entries
            .into_iter()
            .map(|entry| {
                let parsed = serde_json::from_value::<PreferenceResultEntry>(entry)
                    .expect("Failed to parse combined preference entry");
                let is_correct1 = parsed.is_correct1;
                let is_correct2 = parsed.is_correct2;
                ((parsed.index, is_correct1, is_correct2), parsed)
            })
            .collect();
    let mut remaining_indices_lang1_correct_lang2_incorrect: HashSet<usize> =
        get_preference_indices();
    let mut remaining_indices_lang1_incorrect_lang2_correct: HashSet<usize> =
        remaining_indices_lang1_correct_lang2_incorrect.clone();
    let mut remaining_indices_both_correct: HashSet<usize> =
        remaining_indices_lang1_incorrect_lang2_correct.clone();
    let mut remaining_indices_both_incorrect: HashSet<usize> =
        remaining_indices_both_correct.clone();
    for entry in lang1_correct_lang2_incorrect_entries.iter() {
        let result = remaining_indices_lang1_correct_lang2_incorrect.remove(&entry.index);
        assert!(
            result,
            "Duplicate or invalid index found in lang1 correct lang2 incorrect results: {}",
            entry.index
        );
    }
    for entry in lang1_incorrect_lang2_correct_entries.iter() {
        let result = remaining_indices_lang1_incorrect_lang2_correct.remove(&entry.index);
        assert!(
            result,
            "Duplicate or invalid index found in lang1 incorrect lang2 correct results: {}",
            entry.index
        );
    }
    for entry in both_correct_entries.iter() {
        let result = remaining_indices_both_correct.remove(&entry.index);
        assert!(
            result,
            "Duplicate or invalid index found in both correct results: {}",
            entry.index
        );
    }
    for entry in both_incorrect_entries.iter() {
        let result = remaining_indices_both_incorrect.remove(&entry.index);
        assert!(
            result,
            "Duplicate or invalid index found in both incorrect results: {}",
            entry.index
        );
    }
    let mut missing_lang1_correct_lang2_incorrect_count = 0;
    let mut missing_lang1_incorrect_lang2_correct_count = 0;
    let mut missing_both_correct_count = 0;
    let mut missing_both_incorrect_count = 0;
    for index in remaining_indices_lang1_correct_lang2_incorrect {
        let key = (index, true, false);
        if let Some(entry) = combined_entries_parsed.get(&key) {
            lang1_correct_lang2_incorrect_entries.push(entry.clone());
        } else {
            missing_lang1_correct_lang2_incorrect_count += 1;
        }
    }
    for index in remaining_indices_lang1_incorrect_lang2_correct {
        let key = (index, false, true);
        if let Some(entry) = combined_entries_parsed.get(&key) {
            lang1_incorrect_lang2_correct_entries.push(entry.clone());
        } else {
            missing_lang1_incorrect_lang2_correct_count += 1;
        }
    }
    for index in remaining_indices_both_correct {
        let key = (index, true, true);
        if let Some(entry) = combined_entries_parsed.get(&key) {
            both_correct_entries.push(entry.clone());
        } else {
            missing_both_correct_count += 1;
        }
    }
    for index in remaining_indices_both_incorrect {
        let key = (index, false, false);
        if let Some(entry) = combined_entries_parsed.get(&key) {
            both_incorrect_entries.push(entry.clone());
        } else {
            missing_both_incorrect_count += 1;
        }
    }
    if [
        missing_lang1_correct_lang2_incorrect_count,
        missing_lang1_incorrect_lang2_correct_count,
        missing_both_correct_count,
        missing_both_incorrect_count,
    ]
    .iter()
    .any(|&count| count > 0)
    {
        println!(
            "Warning: Missing entries for model: {}, lang1: {}, lang2: {}: lang1 correct lang2 incorrect: {}, lang1 incorrect lang2 correct: {}, both correct: {}, both incorrect: {}",
            model_safe_name,
            lang1,
            lang2,
            missing_lang1_correct_lang2_incorrect_count,
            missing_lang1_incorrect_lang2_correct_count,
            missing_both_correct_count,
            missing_both_incorrect_count
        );
    }
    lang1_correct_lang2_incorrect_entries.sort_by_key(|e| e.index);
    lang1_incorrect_lang2_correct_entries.sort_by_key(|e| e.index);
    both_correct_entries.sort_by_key(|e| e.index);
    both_incorrect_entries.sort_by_key(|e| e.index);
    let lang1_correct_lang2_incorrect_serialized: Vec<serde_json::Value> =
        lang1_correct_lang2_incorrect_entries
            .iter()
            .map(|e| {
                serde_json::to_value(e)
                    .expect("Failed to serialize lang1 correct lang2 incorrect preference entry")
            })
            .collect();
    let lang1_incorrect_lang2_correct_serialized: Vec<serde_json::Value> =
        lang1_incorrect_lang2_correct_entries
            .iter()
            .map(|e| {
                serde_json::to_value(e)
                    .expect("Failed to serialize lang1 incorrect lang2 correct preference entry")
            })
            .collect();
    let both_correct_serialized: Vec<serde_json::Value> = both_correct_entries
        .iter()
        .map(|e| {
            serde_json::to_value(e).expect("Failed to serialize both correct preference entry")
        })
        .collect();
    let both_incorrect_serialized: Vec<serde_json::Value> = both_incorrect_entries
        .iter()
        .map(|e| {
            serde_json::to_value(e).expect("Failed to serialize both incorrect preference entry")
        })
        .collect();
    write_json_lines_to_file(
        &lang1_correct_lang2_incorrect_path,
        &lang1_correct_lang2_incorrect_serialized,
    )
    .expect("Failed to write json lines");
    write_json_lines_to_file(
        &lang1_incorrect_lang2_correct_path,
        &lang1_incorrect_lang2_correct_serialized,
    )
    .expect("Failed to write json lines");
    write_json_lines_to_file(&both_correct_path, &both_correct_serialized)
        .expect("Failed to write json lines");
    write_json_lines_to_file(&both_incorrect_path, &both_incorrect_serialized)
        .expect("Failed to write json lines");
    println!(
        "Dispatched preference results for model: {}, lang1: {}, lang2: {}",
        model_safe_name, lang1, lang2
    );
}
