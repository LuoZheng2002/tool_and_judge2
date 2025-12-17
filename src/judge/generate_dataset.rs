use std::{fs::File, path::Path};

use pyo3::pyfunction;
use serde::Deserialize;

use crate::util::load_json_lines;


#[derive(Deserialize)]
pub struct MmmluDatasetEntry {
    pub original_index: usize,
    pub question: String,
    pub choices: [String; 4],
    pub answer: usize,
    pub subject: String,
}

fn download_mmmlu_dataset(lang: &str){
    unimplemented!();
}

#[pyfunction]
pub fn generate_normalized_datasets(lang: &str) {
    let input_dataset_path = format!("judge/datasets/mmmlu/{}.jsonl", lang);

    if !Path::new(&input_dataset_path).exists() {
        println!("MMMLU dataset for language {} not found. Downloading...", lang);
        download_mmmlu_dataset(lang);
    }
    let entries = load_json_lines(&input_dataset_path).expect("Failed to load MMMLU dataset");
    let parsed_entries: Vec<MmmluDatasetEntry> = entries
        .into_iter()
        .map(|entry| serde_json::from_value(entry).expect("Failed to parse MMMLU dataset entry"))
        .collect();

    println!("Generating single answer dataset...");
    unimplemented!();
}

fn normalize_english_sample



#[test]
fn test_generate_single_answer_dataset() {
    generate_normalized_datasets("en");
}