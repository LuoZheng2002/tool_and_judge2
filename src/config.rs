use pyo3::{basic::CompareOp, prelude::*};
use strum_macros::{Display, EnumString};
#[pyclass]
#[derive(Clone, EnumString, Display, PartialEq, Eq, Copy)]
pub enum ApiModel {
    #[strum(serialize = "gpt-5")]
    Gpt5,
    #[strum(serialize = "gpt-5-mini")]
    Gpt5Mini,
    #[strum(serialize = "gpt-5-nano")]
    Gpt5Nano,
    #[strum(serialize = "gpt-4.1")]
    Gpt4_1,
    #[strum(serialize = "deepseek-chat")]
    DeepSeek,
    #[strum(serialize = "meta.llama3-1-8b-instruct-v1:0")]
    Llama3_1_8B,
    #[strum(serialize = "meta.llama3-1-70b-instruct-v1:0")]
    Llama3_1_70B,
}

#[pymethods]
impl ApiModel {
    pub fn to_string(&self) -> String {
        format!("{}", self)
    }
    fn __richcmp__(&self, other: PyRef<ApiModel>, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self == &*other),
            CompareOp::Ne => Ok(self != &*other),
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Only equality comparisons are supported",
            )),
        }
    }
    pub fn api_key_name(&self) -> String {
        match self {
            ApiModel::Gpt5 | ApiModel::Gpt5Mini | ApiModel::Gpt5Nano | ApiModel::Gpt4_1 => {
                "OPENAI_API_KEY".to_string()
            }
            ApiModel::DeepSeek => "DEEPSEEK_API_KEY".to_string(),
            ApiModel::Llama3_1_8B | ApiModel::Llama3_1_70B => {
                panic!("Llama 3 does not use an API key")
            }
        }
    }
    pub fn base_url(&self) -> String {
        match self {
            ApiModel::Gpt5 | ApiModel::Gpt5Mini | ApiModel::Gpt5Nano | ApiModel::Gpt4_1 => {
                "https://api.openai.com/v1".to_string()
            }
            ApiModel::DeepSeek => "https://api.deepseek.com".to_string(),
            ApiModel::Llama3_1_8B | ApiModel::Llama3_1_70B => {
                panic!("Llama 3 does not use a base URL")
            }
        }
    }
}

impl std::fmt::Debug for ApiModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

#[pyclass]
#[derive(Clone, EnumString, Display, PartialEq, Eq, Copy)]
pub enum LocalModel {
    #[strum(serialize = "ibm-granite/granite-4.0-h-tiny")]
    Granite4_0HTiny,
    #[strum(serialize = "ibm-granite/granite-4.0-h-small")]
    Granite4_0HSmall,
    #[strum(serialize = "Qwen/Qwen3-8B")]
    Qwen3_8B,
    #[strum(serialize = "Qwen/Qwen3-14B")]
    Qwen3_14B,
    #[strum(serialize = "Qwen/Qwen3-30B-A3B")]
    Qwen3_30bA3b,
    #[strum(serialize = "Qwen/Qwen3-32B")]
    Qwen3_32B,
    #[strum(serialize = "Qwen/Qwen3-Next-80B-A3B-Instruct")]
    Qwen3Next80bA3b,
    #[strum(serialize = "Qwen/Qwen3-235B")]
    Qwen3_235bA22b,
    #[strum(serialize = "meta-llama/Llama-3.1-8B-Instruct")]
    Llama3_1_8B,
    #[strum(serialize = "meta-llama/Llama-3.1-70B-Instruct")]
    Llama3_1_70B,
    #[strum(serialize = "meta-llama/Llama-3.3-70B-Instruct")]
    Llama3_3_70B,
    #[strum(serialize = "CohereLabs/aya-expanse-32b")]
    AyaExpanse32B,
    #[strum(serialize = "Unbabel/M-Prometheus-14B")]
    UnbabelMPrometheus14B,
    #[strum(serialize = "prometheus-eval/prometheus-8x7b-v2.0")]
    Prometheus8x7bV2,
}
#[pymethods]
impl LocalModel {
    pub fn to_string(&self) -> String {
        format!("{}", self)
    }
    pub fn size_in_billion_parameters(&self) -> f32 {
        match self {
            LocalModel::Granite4_0HTiny => 0.3,
            LocalModel::Granite4_0HSmall => 1.3,
            LocalModel::Qwen3_8B => 8.0,
            LocalModel::Qwen3_14B => 14.0,
            LocalModel::Qwen3_30bA3b => 30.0,
            LocalModel::Qwen3_32B => 32.0,
            LocalModel::Qwen3Next80bA3b => 80.0,
            LocalModel::Qwen3_235bA22b => 235.0,
            LocalModel::Llama3_1_8B => 8.0,
            LocalModel::Llama3_1_70B => 70.0,
            LocalModel::Llama3_3_70B => 70.0,
            LocalModel::AyaExpanse32B => 32.0,
            LocalModel::UnbabelMPrometheus14B => 14.0,
            LocalModel::Prometheus8x7bV2 => 47.0,
        }
    }
    fn __richcmp__(&self, other: PyRef<LocalModel>, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self == &*other),
            CompareOp::Ne => Ok(self != &*other),
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Only equality comparisons are supported",
            )),
        }
    }
}

impl std::fmt::Debug for LocalModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub enum Model {
    Api(ApiModel),
    Local(LocalModel),
}

#[pymethods]
impl Model {
    pub fn to_string(&self) -> String {
        match self {
            Model::Api(api_model) => api_model.to_string(),
            Model::Local(local_model) => local_model.to_string(),
        }
    }
    fn __richcmp__(&self, other: PyRef<Model>, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self == &*other),
            CompareOp::Ne => Ok(self != &*other),
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Only equality comparisons are supported",
            )),
        }
    }
}

/* ---------------------------------------------------------------------------------------------------- */
/* Tool Project Configuration                                                                           */
/* ---------------------------------------------------------------------------------------------------- */

pub fn requires_name_sanitization(model: Model) -> bool {
    match model {
        Model::Api(api_model) => match api_model {
            ApiModel::Gpt5 | ApiModel::Gpt5Mini | ApiModel::Gpt5Nano => true,
            _ => false,
        },
        Model::Local(_) => false,
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub enum Language {
    Chinese,
    Hindi,
    Igbo,
}

#[pyclass]
#[derive(Clone, Debug)]
pub enum TranslateOption {
    FullyTranslated,
    FullyTranslatedPromptTranslate,
    PartiallyTranslated,
    FullyTranslatedPreTranslate,
    FullyTranslatedPostTranslate,
}

#[pyclass]
#[derive(Clone, Debug)]
pub enum AddNoiseMode {
    NoNoise,
    Synonym,
    Paraphrase,
}

#[pyclass]
#[derive(Clone, Debug)]
pub enum TranslateMode {
    Translated {
        language: Language,
        option: TranslateOption,
    },
    NotTranslated {},
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct ToolExperiment {
    pub translate_mode: TranslateMode,
    pub add_noise_mode: AddNoiseMode,
}

#[pymethods]
impl ToolExperiment {
    #[new]
    fn new(translate_mode: TranslateMode, add_noise_mode: AddNoiseMode) -> Self {
        ToolExperiment {
            translate_mode,
            add_noise_mode,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct ToolConfig {
    #[pyo3(get)]
    pub model: Model,
    #[pyo3(get)]
    pub experiments: Vec<ToolExperiment>,
}

#[pymethods]
impl ToolConfig {
    #[new]
    fn new(model: Model, experiments: Vec<ToolExperiment>) -> Self {
        ToolConfig { model, experiments }
    }
}

/* ---------------------------------------------------------------------------------------------------- */
/* Judge Project Configuration                                                                          */
/* ---------------------------------------------------------------------------------------------------- */

// #[pyclass]
// #[derive(Clone, Debug)]
// pub enum JudgeExperiment {
//     PreferenceDirect { lang1: String, lang2: String },
//     Perplexity { lang: String },
// }


#[pyclass]
#[derive(Clone, Debug)]
pub enum JudgeExperiments {
    HuggingFace{
        perplexity_experiments: Vec<PerplexityExperiment>,
    },
    Vllm{
        preference_experiments: Vec<PreferenceExperiment>,
        perplexity_experiments: Vec<PerplexityExperiment>,
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PerplexityExperiment {
    pub lang: String,
}
#[pymethods]
impl PerplexityExperiment {
    #[new]
    fn new(lang: String) -> Self {
        PerplexityExperiment { lang }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PreferenceExperiment {
    pub lang1: String,
    pub lang2: String,
}

#[pymethods]
impl PreferenceExperiment {
    #[new]
    fn new(lang1: String, lang2: String) -> Self {
        PreferenceExperiment { lang1, lang2 }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct JudgeConfig {
    #[pyo3(get)]
    pub model: LocalModel,
    #[pyo3(get)]
    pub experiments: JudgeExperiments,
}

#[pymethods]
impl JudgeConfig {
    #[new]
    fn new(model: LocalModel, experiments: JudgeExperiments) -> Self {
        JudgeConfig { model, experiments }
    }
}
