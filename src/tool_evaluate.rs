use crate::{
    tool_bfcl_formats::{BfclDatasetEntry, BfclGroundTruthEntry, BfclOutputFunctionCall},
    tool_error_analysis::EvaluationError,
    tool_file_models::{EvaluationResultEntry, InferenceJsonEntry},
};

pub fn evaluate_entry(
    id: String,
    inference_entry: &InferenceJsonEntry,
    test_case_entry: &BfclDatasetEntry,
    ground_truth_entry: &BfclGroundTruthEntry,
) -> EvaluationResultEntry {
    let functions = match &inference_entry.result {
        Ok(funcs) => funcs,
        Err(e) => {
            return EvaluationResultEntry {
                id,
                valid: false,
                error: Some(e.clone()),
            };
        }
    };
    if functions.len() != 1 {
        return EvaluationResultEntry {
            id,
            valid: false,
            error: Some(EvaluationError::InvalidEntryCount {
                expected_count: 1,
                actual_count: functions.len(),
                decoded_output: serde_json::to_string(functions)
                    .expect("Should serialize correctly"),
            }),
        };
    }
    let function = &functions[0];
    let function = BfclOutputFunctionCall::deserialize_from_json(function)
        .expect("The function call should be deserializable, otherwise it is intercepted at previous parsing pass");
    let ground_truth_functions = &ground_truth_entry.ground_truth;
    assert_eq!(
        ground_truth_functions.len(),
        1,
        "Each ground truth entry should have exactly one function call"
    );
    let ground_truth_function = &ground_truth_functions[0];
    if function.function_name != ground_truth_function.function_name {
        return EvaluationResultEntry {
            id,
            valid: false,
            error: Some(EvaluationError::WrongFuncName {
                expected_name: ground_truth_function.function_name.clone(),
                actual_name: function.function_name.clone(),
                decoded_output: serde_json::to_string(functions)
                    .expect("Should serialize correctly"),
            }),
        };
    }
    let target_test_case_function = test_case_entry
        .functions
        .iter()
        .find(|f| f.name == function.function_name)
        .expect("The test case should contain the target function");
    let target_function_required_parameters = &target_test_case_function.required;
    let prarameters = &function.parameters;
    for required_param in target_function_required_parameters {
        if !prarameters.contains_key(required_param) {
            return EvaluationResultEntry {
                id,
                valid: false,
                error: Some(EvaluationError::MissingRequiredParam {
                    missing_param: required_param.clone(),
                    required_params: target_function_required_parameters.clone(),
                    decoded_output: serde_json::to_string(functions)
                        .expect("Should serialize correctly"),
                }),
            };
        }
    }
    let ground_truth_parameters = &ground_truth_function.parameters;
    for (param, value) in prarameters.iter() {
        let Some(ground_truth_parameter_values) = ground_truth_parameters.get(param) else {
            return EvaluationResultEntry {
                id,
                valid: false,
                error: Some(EvaluationError::UnexpectedParam {
                    unexpected_param: param.clone(),
                    decoded_output: serde_json::to_string(functions)
                        .expect("Should serialize correctly"),
                    expected_params: ground_truth_parameters.keys().cloned().collect(),
                }),
            };
        };
        if !value_matches_list(value, ground_truth_parameter_values) {
            return EvaluationResultEntry {
                id,
                valid: false,
                error: Some(EvaluationError::InvalidParamValue {
                    param: param.clone(),
                    actual_value: value.clone(),
                    expected_values: ground_truth_parameter_values.clone(),
                    decoded_output: serde_json::to_string(functions)
                        .expect("Should serialize correctly"),
                }),
            };
        }
    }
    EvaluationResultEntry {
        id,
        valid: true,
        error: None,
    }
}

fn value_matches_list(value: &serde_json::Value, expected_list: &Vec<serde_json::Value>) -> bool {
    for expected in expected_list {
        if value_matches_any(value, expected) {
            return true;
        }
    }
    false
}

fn value_matches_any(value: &serde_json::Value, expected: &serde_json::Value) -> bool {
    if value == expected {
        return true;
    }
    if let serde_json::Value::Array(expected_arry) = expected {
        for expected_item in expected_arry {
            if value_matches_any(value, expected_item) {
                return true;
            }
        }
    }
    return false;
}
