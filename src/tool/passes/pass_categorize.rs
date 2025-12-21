use pyo3::pyfunction;
use serde::{Deserialize, Serialize};

use crate::{
    config::ToolConfig,
    tool::{
        base_path::BASE_RESULT_PATH,
        error_analysis::{EvaluationError, ToolErrorCategory},
        experiments::CategorizeFileName,
    },
    utils::get_model_safe_name,
};

#[derive(Clone, Serialize)]
pub struct CategorizeAggregatedInputEntry {
    pub id: String,
    pub error: EvaluationError,
    pub file_name: CategorizeFileName,
}

#[derive(Clone, Deserialize)]
pub struct CategorizeAggregatedOutputEntry {
    pub id: String,
    pub error_category: ToolErrorCategory,
    pub error: EvaluationError,
    pub file_name: CategorizeFileName,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CategorizeEntry {
    pub id: String,
    pub error_category: ToolErrorCategory,
    pub error: EvaluationError,
}

const CATEGORIZE_AGGREGATED_INPUT_FILE_NAME: &str = "categorize_aggregated_input.jsonl";
const CATEGORIZE_AGGREGATED_OUTPUT_FILE_NAME: &str = "categorize_aggregated_output.jsonl";

#[pyfunction]
pub fn pass_categorize_aggregated_input_file_path(config: &ToolConfig) -> String {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    let file_path = BASE_RESULT_PATH
        .join(&model_safe_name)
        .join(CATEGORIZE_AGGREGATED_INPUT_FILE_NAME);
    file_path.to_str().unwrap().to_string()
}

#[pyfunction]
pub fn pass_categorize_aggregated_output_file_path(config: &ToolConfig) -> String {
    let model = config.model;
    let model_safe_name = get_model_safe_name(model);
    let file_path = BASE_RESULT_PATH
        .join(&model_safe_name)
        .join(CATEGORIZE_AGGREGATED_OUTPUT_FILE_NAME);
    file_path.to_str().unwrap().to_string()
}

/// This function only generates entries that need to be categorized by an LLM.
#[pyfunction]
pub fn pass_generate_categorize_aggregated_input(config: &ToolConfig) {
    // TODO: implement this function
}
