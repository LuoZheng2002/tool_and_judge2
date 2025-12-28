use std::{path::PathBuf, sync::LazyLock};

pub static TOOL_BASE_DATASET_PATH: LazyLock<PathBuf> =
    LazyLock::new(|| PathBuf::from("tool/dataset"));
pub static TOOL_BASE_RESULT_PATH: LazyLock<PathBuf> =
    LazyLock::new(|| PathBuf::from("tool/result"));
