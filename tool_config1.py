from codebase_rs import *


configs = [
    ToolConfig(Model.Api(ApiModel.Gpt5Nano), TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslatedPreTranslate), AddNoiseMode.Paraphrase),
    # ToolConfig(Model.Api(ApiModel.Gpt5Mini), TranslateMode.Translated(language=Language.Hindi, option=TranslateOption.PartiallyTranslated), AddNoiseMode.Synonym),
]