use std::{
    sync::{Arc, atomic::AtomicUsize},
};

use atomic_refcell::AtomicRefCell;
use pyo3::{Python, types::PyAnyMethods};

use crate::{
    config::{ApiModel, Model},
    models::{
        backend::{BackendType, ModelBackend, WhichBackend, get_or_create_backend},
    },
    tool::category_cache::CategoryCache,
    tool::error_analysis::{EvaluationError, ToolErrorCategory},
};

pub async fn categorize_entry(
    evaluation_error: &EvaluationError,
    category_cache: Arc<AtomicRefCell<CategoryCache>>,
    cache_hits: Arc<AtomicUsize>,
) -> ToolErrorCategory {
    match evaluation_error {
        EvaluationError::NoFunctionCallsFound { .. }
        | EvaluationError::JsonDecodeError { .. }
        | EvaluationError::ParsingError { .. } => ToolErrorCategory::SyntaxError,
        EvaluationError::InvalidEntryCount { .. }
        | EvaluationError::WrongFuncName { .. }
        | EvaluationError::MissingRequiredParam { .. }
        | EvaluationError::UnexpectedParam { .. } => ToolErrorCategory::MiscError,
        EvaluationError::InvalidParamValue {
            param,
            actual_value,
            expected_values,
            decoded_output: _,
        } => {
            let category_cache_guard = category_cache.borrow();
            let actual_value_str =
                serde_json::to_string(actual_value).expect("Should serialize actual value");
            let expected_values_str: Vec<String> = expected_values
                .iter()
                .map(|v| serde_json::to_string(v).expect("Should serialize expected value"))
                .collect();
            let cache_key = (actual_value_str, expected_values_str);
            if let Some(cached_category) = category_cache_guard.0.get(&cache_key) {
                cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                cached_category.clone()
            } else {
                drop(category_cache_guard); // release the borrow before awaiting
                let category =
                    categorize_parameter_value_async(param, actual_value, expected_values).await;
                let mut category_cache_guard = category_cache.borrow_mut();
                category_cache_guard.0.insert(cache_key, category.clone());
                category
            }
        }
    }
}

pub async fn categorize_parameter_value_async(
    param_name: &str,
    actual_value: &serde_json::Value,
    expected_values: &Vec<serde_json::Value>,
) -> ToolErrorCategory {
    // The backend type must be ApiOrHuggingFace because an ApiBackend will return ApiOrHuggingFace as its backend type.
    let categorize_backend =
        get_or_create_backend(Model::Api(ApiModel::Gpt5), BackendType::ApiOrHuggingFace,  WhichBackend::Assist, 1).await;
    let categorize_backend = categorize_backend.as_ref().expect("Should get backend");
    // let categorize_api_backend = (categorize_backend.as_ref() as &dyn Any)
    //     .downcast_ref::<ApiBackend>()
    //     .expect("Should be able to downcast");
    let ModelBackend::Api(api_backend) = categorize_backend.as_ref() else {
        panic!("Categorize backend should be ApiBackend");
    };
    let client = &api_backend.client;
    let param_name = param_name.to_string();
    let actual_value = serde_json::to_string(actual_value).expect("Should serialize actual value");
    let expected_values =
        serde_json::to_string(expected_values).expect("Should serialize expected values");
    // call python to categorize
    let fut = Python::attach(|py| {
        let gpt5_backend_module = py
            .import("src_py.gpt5_backend")
            .expect("Failed to import src_py.gpt5_backend module");
        let categorize_async_fn = gpt5_backend_module
            .getattr("categorize_parameter_value_async")
            .expect("Failed to get categorize_parameter_value_async function");
        let model_name = api_backend.model.to_string();
        let arguments = (
            model_name,
            client,
            param_name,
            actual_value,
            expected_values,
        );
        let py_future = categorize_async_fn
            .call1(arguments)
            .expect("Failed to call categorize_parameter_value_async function");
        pyo3_async_runtimes::tokio::into_future(py_future)
            .expect("Failed to convert to Rust future")
    });
    let category_str = fut.await.expect("Should get category string from python");
    let category_str = Python::attach(|py| {
        category_str
            .extract::<String>(py)
            .expect("Should extract string")
    });
    // needs to be consistent with src_py/gpt5_backend.py's prompt design
    match category_str.as_str() {
        "wrong_value" => ToolErrorCategory::WrongValue,
        "relevant_but_incorrect" => ToolErrorCategory::RelevantButIncorrect,
        "exactly_same_meaning" => ToolErrorCategory::ExactlySameMeaning,
        "language_mismatch_wrong_value" => ToolErrorCategory::LanguageMismatchWrongValue,
        "language_mismatch_relevant_but_incorrect" => {
            ToolErrorCategory::LanguageMismatchRelevantButIncorrect
        }
        "language_mismatch_exactly_same_meaning" => {
            ToolErrorCategory::LanguageMismatchExactlySameMeaning
        }
        _ => {
            println!(
                "Warning: categorize LLM returned an unknown category: {}",
                category_str
            );
            ToolErrorCategory::OtherError
        }
    }
}
