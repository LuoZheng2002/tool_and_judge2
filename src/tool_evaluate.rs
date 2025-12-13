use crate::{tool_bfcl_formats::{BfclDatasetEntry, BfclGroundTruthEntry}, tool_file_models::{EvaluationResultEntry, InferenceJsonEntry}};





pub fn evaluate_entry(
    id: &str,
    inference_entry: &InferenceJsonEntry,
    test_case_entry: &BfclDatasetEntry,
    ground_truth_entry: &BfclGroundTruthEntry,
) -> EvaluationResultEntry {
    
    todo!()
}
