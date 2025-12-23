use std::sync::Arc;

use crate::{
    config::{ApiModel, LocalModel, Model},
    models::{
        deepseek_interface::DeepSeekInterface,
        function_name_mapper::{FunctionNameMapper},
        gpt5_interface::Gpt5Interface,
        llama3_1_interface::Llama3_1Interface,
        qwen3_interface::Qwen3Interface,
        granite4_interface::Granite4Interface,
    },
    tool::{
        bfcl_formats::{BfclFunctionDef, BfclOutputFunctionCall},
        error_analysis::EvaluationError,
    },
};

#[async_trait::async_trait]
pub trait ModelInterface: Send + Sync {
    // async fn generate_tool_call_async(
    //     &self,
    //     backend: Arc<ModelBackend>,
    //     raw_functions: Vec<BfclFunctionDef>,
    //     user_question: String,
    //     prompt_passing_in_english: bool,
    //     name_mapper: Arc<AtomicRefCell<FunctionNameMapper>>,
    // ) -> String;

    // async fn translate_tool_question_async(
    //     &self,
    //     backend: Arc<ModelBackend>,
    //     user_question: String,
    // ) -> String;

    // async fn translate_tool_answer_async(
    //     &self,
    //     backend: Arc<ModelBackend>,
    //     parameter_value: String,
    // ) -> String;

    fn parse_tool_calls(
        &self,
        raw_output: &str,
        name_mapper: &FunctionNameMapper,
    ) -> Result<Vec<BfclOutputFunctionCall>, EvaluationError>;

    fn generate_tool_definitions(
        &self,
        bfcl_tool_definitions: &Vec<BfclFunctionDef>,
        function_name_mapper: &FunctionNameMapper,
    ) -> serde_json::Value;
}

pub fn get_model_interface(model: Model) -> Arc<dyn ModelInterface> {
    match model {
        Model::Api(api_model) => match api_model {
            ApiModel::Gpt5 | ApiModel::Gpt5Mini | ApiModel::Gpt5Nano => Arc::new(Gpt5Interface),
            ApiModel::DeepSeek => Arc::new(DeepSeekInterface),
            _ => {
                unimplemented!(
                    "API model interfaces other than GPT-5 and DeepSeek are not implemented yet."
                )
            }
        },
        Model::Local(local_model) => match local_model {
            LocalModel::Llama3_1_8B | LocalModel::Llama3_1_70B => Arc::new(Llama3_1Interface),
            LocalModel::Qwen3_8B | LocalModel::Qwen3_14B | LocalModel::Qwen3_30bA3b | LocalModel::Qwen3_32B | LocalModel::Qwen3Next80bA3b => Arc::new(Qwen3Interface),
            LocalModel::Granite4_0HTiny | LocalModel::Granite4_0HSmall => Arc::new(Granite4Interface),
            _ => {
                unimplemented!(
                    "Local model interfaces other than Llama 3.1, Qwen3, and Granite 4.0 are not implemented yet."
                )
            }
        },
    }
}
