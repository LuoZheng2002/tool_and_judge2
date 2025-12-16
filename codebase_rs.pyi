    



from enum import Enum


class AddNoiseMode(Enum):
    NoNoise: ...
    Synonym: ...
    Paraphrase: ...

class ApiModel(Enum):
    Gpt5: ...
    Gpt5Mini: ...
    Gpt5Nano: ...
    DeepSeek: ...
    Llama3_1_8B: ...
    Llama3_1_70B: ...

class Language(Enum):
    Chinese: ...
    Hindi: ...

class LocalModel(Enum):
    Granite4_0HTiny: ...
    Granite4_0HSmall: ...
    Qwen3_8B: ...
    Qwen3_14B: ...
    Qwen3_30bA3b: ...
    Qwen3_32B: ...
    Qwen3Next80bA3b: ...
    Llama3_1_8B: ...
    Llama3_1_70B: ...

class Model:
    Api = ApiModel
    Local = LocalModel

class TranslateOption(Enum):
    FullyTranslated: ...
    FullyTranslatedPromptTranslate: ...
    PartiallyTranslated: ...
    FullyTranslatedPreTranslate: ...
    FullyTranslatedPostTranslate: ...

class TranslateMode_Translated:
    def __new__(cls, *, language: Language, option: TranslateOption) -> TranslateMode: ...

class TranslateMode_NotTranslated:
    def __new__(cls) -> TranslateMode: ...

class TranslateMode:
    Translated = TranslateMode_Translated
    NotTranslated = TranslateMode_NotTranslated

class ToolConfig:
    def __new__(cls, model: Model, translate_mode: TranslateMode, add_noise_mode: AddNoiseMode) -> ToolConfig: ...

# #[pyclass]
# #[derive(Clone, EnumString, Display)]
# pub enum ApiModel {
#     #[strum(serialize = "gpt-5")]
#     Gpt5,
#     #[strum(serialize = "gpt-5-mini")]
#     Gpt5Mini,
#     #[strum(serialize = "gpt-5-nano")]
#     Gpt5Nano,
#     #[strum(serialize = "deepseek-chat")]
#     DeepSeek,
#     #[strum(serialize = "meta.llama3-1-8b-instruct-v1:0")]
#     Llama3_1_8B,
#     #[strum(serialize = "meta.llama3-1-70b-instruct-v1:0")]
#     Llama3_1_70B,
# }

# impl std::fmt::Debug for ApiModel {
#     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
#         write!(f, "{}", self)
#     }
# }

# #[pyclass]
# #[derive(Clone, EnumString, Display)]
# pub enum LocalModel {
#     #[strum(serialize = "ibm-granite/granite-4.0-h-tiny")]
#     Granite4_0HTiny,    
#     #[strum(serialize = "ibm-granite/granite-4.0-h-small")]
#     Granite4_0HSmall,
#     #[strum(serialize = "Qwen/Qwen3-8B")]
#     Qwen3_8B,
#     #[strum(serialize = "Qwen/Qwen3-14B")]
#     Qwen3_14B,
#     #[strum(serialize = "Qwen/Qwen3-30B-A3B")]
#     Qwen3_30bA3b,
#     #[strum(serialize = "Qwen/Qwen3-32B-A3B")]
#     Qwen3_32B,
#     #[strum(serialize = "Qwen/Qwen3-Next-80B-A3B-Instruct")]
#     Qwen3Next80bA3b,
# }

# impl std::fmt::Debug for LocalModel {
#     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
#         write!(f, "{}", self)
#     }
# }


# #[pyclass]
# #[derive(Clone, Debug)]
# pub enum Model {
#     Api(ApiModel),
#     Local(LocalModel),
# }

# /* ---------------------------------------------------------------------------------------------------- */
# /* Tool Project Configuration                                                                           */
# /* ---------------------------------------------------------------------------------------------------- */

# pub fn requires_name_sanitization(model: Model) -> bool {
#     match model {
#         Model::Api(api_model) => match api_model {
#             ApiModel::Gpt5 | ApiModel::Gpt5Mini | ApiModel::Gpt5Nano => true,
#             _ => false,
#         },
#         Model::Local(_) => false,
#     }
# }

# #[pyclass]
# #[derive(Clone, Debug)]
# pub enum Language {
#     Chinese,
#     Hindi,
# }

# #[pyclass]
# #[derive(Clone, Debug)]
# pub enum TranslateOption {
#     FullyTranslated,
#     FullyTranslatedPromptTranslate,
#     PartiallyTranslated,
#     FullyTranslatedPreTranslate,
#     FullyTranslatedPostTranslate,
# }

# #[pyclass]
# #[derive(Clone, Debug)]
# pub enum AddNoiseMode {
#     NoNoise,
#     Synonym,
#     Paraphrase,
# }

# #[pyclass]
# #[derive(Clone, Debug)]
# pub enum TranslateMode {
#     Translated {
#         language: Language,
#         option: TranslateOption,
#     },
#     NotTranslated {},
# }

# #[pyclass]
# #[derive(Clone)]
# pub struct ToolConfig {
#     pub model: Model,
#     pub translate_mode: TranslateMode,
#     pub add_noise_mode: AddNoiseMode,
# }

# #[pymethods]
# impl ToolConfig {
#     #[new]
#     fn new(model: Model, translate_mode: TranslateMode, add_noise_mode: AddNoiseMode) -> Self {
#         ToolConfig {
#             model,
#             translate_mode,
#             add_noise_mode,
#         }
#     }
# }