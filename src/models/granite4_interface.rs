
use crate::{
    models::{function_name_mapper::FunctionNameMapper,
        model_interface::ModelInterface,
    },
    one_entry_map::KeyValuePair,
    tool::bfcl_formats::{BfclFunctionDef, BfclOutputFunctionCall, BfclParameter},
    tool::error_analysis::EvaluationError,
};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// Granite 4.0 tool format following OpenAI's function definition schema
/// https://huggingface.co/ibm-granite/granite-4.0-micro
///
/// Granite uses special XML-like tags for tool calls:
/// <tool_call>
/// {"name": "function_name", "arguments": {"param": "value"}}
/// </tool_call>
#[derive(Serialize)]
pub struct Granite4Tool {
    #[serde(rename = "type")]
    pub ty: String,
    pub function: Granite4Function,
}

#[derive(Serialize)]
pub struct Granite4Function {
    pub name: String,
    pub description: String,
    pub parameters: Granite4Parameter,
}

/// JSON Schema for Granite 4 function parameters
#[derive(Serialize)]
pub struct Granite4Parameter {
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<IndexMap<String, Granite4Parameter>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<Granite4Parameter>>,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub r#enum: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<serde_json::Value>,
}

#[derive(Deserialize, Clone)]
pub struct Granite4OutputFunctionCall {
    name: String,
    arguments: IndexMap<String, serde_json::Value>, // Direct JSON object
}

#[derive(Copy, Clone)]
pub struct Granite4Interface;

impl Granite4Interface {
    pub fn map_type_hint(ty: &str) -> String {
        match ty {
            "dict" => "object".to_string(),
            "float" => "number".to_string(),
            "tuple" => "array".to_string(),
            _ => ty.to_string(),
        }
    }
}

fn bfcl_param_to_granite4_param(bfcl_parameter: &BfclParameter) -> Granite4Parameter {
    let BfclParameter {
        ty: bfcl_type,
        properties: bfcl_properties,
        items: bfcl_items,
        r#enum: bfcl_enum,
        description: bfcl_description,
        format: bfcl_format,
        required: bfcl_required,
        default: bfcl_default,
        optional: _,
        maximum: bfcl_maximum,
    } = bfcl_parameter;

    let granite4_type = Granite4Interface::map_type_hint(bfcl_type);
    let bfcl_required = bfcl_required.as_ref();

    let granite4_properties = bfcl_properties.as_ref().map(|props| {
        let mut granite4_props = IndexMap::new();
        for (prop_name, prop_value) in props.iter() {
            let granite4_prop_value = bfcl_param_to_granite4_param(prop_value);
            granite4_props.insert(prop_name.clone(), granite4_prop_value);
        }
        granite4_props
    });

    let granite4_items = bfcl_items.as_ref().map(|item| {
        let granite4_item = bfcl_param_to_granite4_param(item);
        Box::new(granite4_item)
    });

    let granite4_enum = bfcl_enum.clone();
    let granite4_required: Option<Vec<String>> = bfcl_required.map(|reqs| reqs.to_vec());

    Granite4Parameter {
        ty: granite4_type,
        properties: granite4_properties,
        description: bfcl_description.clone(),
        items: granite4_items,
        r#enum: granite4_enum,
        required: granite4_required,
        default: bfcl_default.clone(),
        format: bfcl_format.clone(),
        maximum: bfcl_maximum.clone(),
    }
}

#[async_trait::async_trait]
impl ModelInterface for Granite4Interface {
    fn generate_tool_definitions(
        &self,
        bfcl_functions: &Vec<BfclFunctionDef>,
        name_mapper: &FunctionNameMapper,
    ) -> serde_json::Value {
        let mut granite4_tools = Vec::new();
        for bfcl_func in bfcl_functions.iter() {
            let sanitized_name = name_mapper
                .original_to_sanitized
                .get(&bfcl_func.name)
                .expect("Function name mapper does not contain key")
                .clone();
            let bfcl_param = &bfcl_func.parameters;
            let granite4_params = bfcl_param_to_granite4_param(bfcl_param);
            let description = bfcl_func.description.clone();
            granite4_tools.push(Granite4Tool {
                ty: "function".to_string(),
                function: Granite4Function {
                    name: sanitized_name,
                    description,
                    parameters: granite4_params,
                },
            });
        }
        serde_json::to_value(granite4_tools).expect("Failed to serialize Granite 4 tools")
    }

    fn parse_tool_calls(
        &self,
        raw_output: &str,
        name_mapper: &FunctionNameMapper,
    ) -> Result<Vec<BfclOutputFunctionCall>, EvaluationError> {
        // Granite 4 outputs in format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        let mut model_result_raw = raw_output.trim().to_string();

        // Extract content from <tool_call> tags if present
        let mut tool_calls_list = Vec::new();

        if model_result_raw.contains("<tool_call>") {
            // Extract all tool calls between tags
            let mut remaining = model_result_raw.as_str();

            while let Some(start_idx) = remaining.find("<tool_call>") {
                if let Some(end_idx) = remaining.find("</tool_call>") {
                    let tool_call_content = &remaining[start_idx + "<tool_call>".len()..end_idx];
                    let trimmed = tool_call_content.trim();
                    if !trimmed.is_empty() {
                        tool_calls_list.push(trimmed.to_string());
                    }
                    remaining = &remaining[end_idx + "</tool_call>".len()..];
                } else {
                    break;
                }
            }

            // If we found tool calls, join them as a JSON array
            if !tool_calls_list.is_empty() {
                model_result_raw = format!("[{}]", tool_calls_list.join(","));
            }
        }

        // Strip backticks and whitespace
        model_result_raw = model_result_raw.trim_matches(|c| c == '`' || c == '\n' || c == ' ').to_string();

        // Add brackets if needed (for single objects)
        if !model_result_raw.starts_with('[') && model_result_raw.starts_with('{') {
            model_result_raw = format!("[{}]", model_result_raw);
        }

        // Parse the JSON
        let granite4_calls: Vec<Granite4OutputFunctionCall> = serde_json::from_str(&model_result_raw)
            .map_err(|e| EvaluationError::JsonDecodeError {
                error_message: e.to_string(),
                raw_output: raw_output.to_string(),
            })?;

        // Convert Granite 4 format to BFCL format
        let mut bfcl_calls = Vec::new();
        for granite4_call in granite4_calls {
            // Map the function name back to original
            let original_name = name_mapper
                .sanitized_to_original
                .get(&granite4_call.name)
                .expect("Function name mapper does not contain key")
                .clone();

            bfcl_calls.push(BfclOutputFunctionCall(KeyValuePair {
                key: original_name,
                value: granite4_call.arguments,
            }));
        }

        Ok(bfcl_calls)
    }
}
