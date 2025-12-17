use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct SingleAnswerDatasetEntry {
    pub index: usize,
    pub question: String,
    pub answer: String,
    pub is_correct: bool,
    pub subject: String,
}
