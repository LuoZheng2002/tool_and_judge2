use strum_macros::{Display, EnumString};

#[derive(Clone, EnumString, Display)]
pub enum ToolErrorCategory {
    #[strum(serialize = "syntax_error")]
    SyntaxError,
    #[strum(serialize = "misc_errors")]
    MiscErrors,
    #[strum(serialize = "wrong_values")]
    WrongValues,
    #[strum(serialize = "relevant_but_incorrect")]
    RelevantButIncorrect,
    #[strum(serialize = "exactly_same_meaning")]
    ExactlySameMeaning,
    #[strum(serialize = "language_mismatch_wrong_values")]
    LanguageMismatchWrongValues,
    #[strum(serialize = "language_mismatch_relevant_but_incorrect")]
    LanguageMismatchRelevantButIncorrect,
    #[strum(serialize = "language_mismatch_exactly_same_meaning")]
    LanguageMismatchExactlySameMeaning,
    #[strum(serialize = "other_errors")]
    OtherErrors,
}

#[derive(Clone, EnumString, Display)]
pub enum EvaluationError {
    #[strum(serialize = "no_function_calls_found")]
    NoFunctionCallsFound,
    #[strum(serialize = "json_decode_error")]
    JsonDecodeError,
    #[strum(serialize = "parsing_error")]
    ParsingError,
    #[strum(serialize = "invalid_entry_count")]
    InvalidEntryCount,
    #[strum(serialize = "wrong_func_name")]
    WrongFuncName,
    #[strum(serialize = "missing_required_param")]
    MissingRequiredParam,
    #[strum(serialize = "unexpected_param")]
    UnexpectedParam,
    #[strum(serialize = "invalid_param_value")]
    InvalidParamValue,
}
