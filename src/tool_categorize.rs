use std::sync::Arc;

use atomic_refcell::AtomicRefCell;

use crate::{models::{backend::ModelBackend, model_interface::ModelInterface}, tool_category_cache::CategoryCache, tool_error_analysis::{EvaluationError, ToolErrorCategory}, tool_file_models::CategorizedEntry};




pub async fn categorize_entry(
    id: &str,
    evaluation_error: &EvaluationError,
    model_interface: Arc<dyn ModelInterface>,
    backend: Arc<dyn ModelBackend>,
    category_cache: Arc<AtomicRefCell<CategoryCache>>,
) -> CategorizedEntry {
    
    
    todo!()
}