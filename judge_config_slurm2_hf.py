from codebase_rs import *

config = JudgeConfig(LocalModel.Qwen3_235bA22b, JudgeExperiments.HuggingFace(
    perplexity_experiments=[
        PerplexityExperiment("en"), 
        PerplexityExperiment("zh_cn"),
        PerplexityExperiment("fr_fr"), 
        PerplexityExperiment("ar_xy"), 
        PerplexityExperiment("sw_ke"), 
    ]
))