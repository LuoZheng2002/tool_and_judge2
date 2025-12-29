from codebase_rs import *

config = JudgeConfig(LocalModel.Qwen3_8B, JudgeExperiments.HuggingFace(
    perplexity_experiments=[PerplexityExperiment("en"), PerplexityExperiment("zh_cn")]
))
