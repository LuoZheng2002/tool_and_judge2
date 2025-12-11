

use pyo3::{Bound, types::{PyAnyMethods, PyList, PyListMethods}};

use crate::config::ToolConfig;



pub fn tool_run_impl<'py>(configs: &Bound<'py, PyList>, num_gpus: usize) {
    println!("Tool run implementation called with {} configs and {} GPUs.", configs.len(), num_gpus);   
    let extracted_configs: Vec<ToolConfig> = configs
        .iter()
        .map(|config| config.extract().expect("Failed to extract ToolConfig from Python object"))
        .collect();
    
}