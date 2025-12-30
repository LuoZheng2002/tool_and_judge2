from codebase_rs import *

config = JudgeConfig(LocalModel.Llama3_3_70B, JudgeExperiments.HuggingFace(
    perplexity_experiments=[
        PerplexityExperiment("en"), 
        PerplexityExperiment("zh_cn"),
        PerplexityExperiment("fr_fr"), 
        PerplexityExperiment("de_de"), 
        PerplexityExperiment("ja_jp"), 
        PerplexityExperiment("ko_kr"), 
        PerplexityExperiment("ar_xy"), 
        PerplexityExperiment("bn_bd"), 
        PerplexityExperiment("hi_in"), 
        PerplexityExperiment("id_id"), 
        PerplexityExperiment("it_it"), 
        PerplexityExperiment("pt_br"), 
        PerplexityExperiment("es_la"), 
        PerplexityExperiment("sw_ke"), 
        PerplexityExperiment("yo_ng"),
    ]
))