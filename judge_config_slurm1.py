from codebase_rs import *

config = JudgeConfig(LocalModel.Qwen3_8B, JudgeExperiments.Vllm(
    preference_experiments=[PreferenceExperiment("en", "zh_cn")],
    perplexity_experiments=[PerplexityExperiment("en"), PerplexityExperiment("zh_cn")]
))



