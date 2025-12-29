use pyo3::prelude::*;

pub mod config;
pub mod judge;
pub mod models;
pub mod one_entry_map;
pub mod single_or_list;
pub mod tool;
pub mod utils;

#[pymodule]
pub mod codebase_rs {
    // use pyo3::{Py, pyfunction, types::PyList};
    // #[pyfunction] // Inline definition of a pyfunction, also made available to Python
    // async fn tool_run_async(configs: Py<PyList>, num_gpus: usize) {
    //     println!("Running tool from Rust!\n");
    //     unimplemented!();
    //     // tool_run::tool_run_async(configs, num_gpus).await;
    // }

    // use crate::tool_run;

    #[pymodule_export]
    use super::{
        config::{
            AddNoiseMode, ApiModel, JudgeConfig, JudgeExperiments, PerplexityExperiment, PreferenceExperiment, Language, LocalModel, Model,
            ToolConfig, ToolExperiment, TranslateMode, TranslateOption,
        },
        judge::preference::preference_prepare_aggregated_input,
        judge::{
            perplexity::{
                perplexity_dispatch_generate_perplexity_results,
                perplexity_dispatch_response_results,
                perplexity_dispatch_styled_answers_results,
                perplexity_generate_perplexity_aggregated_input_file_path,
                perplexity_generate_perplexity_aggregated_output_file_path,
                perplexity_generate_response_input_file_path,
                perplexity_generate_response_output_file_path,
                perplexity_generate_styled_answers_input_file_path,
                perplexity_generate_styled_answers_output_file_path,
                perplexity_prepare_generate_perplexity_aggregated_input,
                perplexity_prepare_generate_styled_answers_input,
                perplexity_prepare_response_input,
            }
        },
        models::backend::GenerationResult,
        tool::{
            passes::pass_categorize::{
                pass_categorize_aggregated_input_file_path,
                pass_categorize_aggregated_output_file_path, pass_categorize_dispatch_results,
                pass_categorize_prepare_aggregated_input,
            },
            passes::pass_evaluate::pass_evaluate,
            passes::pass_generate_raw::{
                pass_generate_raw_aggregated_input_file_path,
                pass_generate_raw_aggregated_output_file_path, pass_generate_raw_dispatch_results,
                pass_generate_raw_prepare_aggregated_input,
            },
            passes::pass_parse_output::pass_parse_output,
            passes::pass_post_translate::{
                pass_post_translate_aggregated_input_file_path,
                pass_post_translate_aggregated_output_file_path,
                pass_post_translate_dispatch_results, pass_post_translate_prepare_aggregated_input,
            },
            passes::pass_pre_translate::{
                pass_pre_translate_aggregated_questions_input_file_path,
                pass_pre_translate_aggregated_questions_output_file_path,
                pass_pre_translate_dispatch_results,
                pass_pre_translate_prepare_aggregated_questions,
            },
            passes::pass_statistics::pass_statistics,
        },
    };
}