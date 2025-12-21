use serde::{Deserialize, Serialize};

use crate::tool::error_analysis::{EvaluationError, ToolErrorCategory};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EvaluationSummary {
    pub accuracy: f32,
    pub total_cases: usize,
    pub correct_cases: usize,
}

