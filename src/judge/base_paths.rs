use std::{path::PathBuf, sync::LazyLock};

pub static JUDGE_BASE_DATASET_PATH: LazyLock<PathBuf> =
    LazyLock::new(|| PathBuf::from("judge/datasets"));
pub static JUDGE_BASE_RESULT_PATH: LazyLock<PathBuf> =
    LazyLock::new(|| PathBuf::from("judge/result"));