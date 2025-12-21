use indexmap::IndexMap;

use crate::tool::{bfcl_formats::{
        BfclDatasetEntry, BfclGroundTruthEntry, BfclParameter,
    }, error_analysis::EvaluationError, passes::pass_parse_output::ParseOutputEntry};


// enum RecursiveCheckResult {
//     Valid,
//     MissingRequiredParam {
//         missing_param: String,
//         required_params: Vec<String>,
//     },
//     UnexpectedParam {
//         unexpected_param: String,
//         expected_params: Vec<String>,
//     },
//     InvalidParamType,
//     InvalidParamValue {
//         param: String,
//         actual_value: serde_json::Value,
//         expected_values: Vec<serde_json::Value>,
//     },
// }

