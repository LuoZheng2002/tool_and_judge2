from codebase_rs import *

config = JudgeConfig(LocalModel.Llama3_3_70B, JudgeExperiments.Vllm(
    preference_experiments=[        
        PreferenceExperiment("en", "zh_cn"), 
        PreferenceExperiment("en", "fr_fr"),
        PreferenceExperiment("en", "ar_xy"), 
        PreferenceExperiment("en", "sw_ke"),
        # PreferenceExperiment("en", "en"),
        # PreferenceExperiment("zh_cn", "zh_cn"),
        # PreferenceExperiment("fr_fr", "fr_fr"),
        # PreferenceExperiment("ar_xy", "ar_xy"),
        # PreferenceExperiment("sw_ke", "sw_ke"),
        ],
    perplexity_experiments=[
        PerplexityExperiment("en"), 
        PerplexityExperiment("zh_cn"),
        PerplexityExperiment("fr_fr"), 
        PerplexityExperiment("ar_xy"), 
        PerplexityExperiment("sw_ke"), 
    ]
))
