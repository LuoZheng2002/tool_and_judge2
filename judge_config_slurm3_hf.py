from codebase_rs import *

config = JudgeConfig(LocalModel.Llama3_3_70B, JudgeExperiments.HuggingFace(
    perplexity_experiments=[
        PerplexityExperiment("en"), 
        PerplexityExperiment("zh_cn"),
        PerplexityExperiment("fr_fr"), 
        PerplexityExperiment("ar_xy"), 
        PerplexityExperiment("sw_ke"), 
    ]
))