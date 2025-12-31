from codebase_rs import *

experiments = []


# Igbo - only NoNoise available
for translate in [
    TranslateMode.Translated(language=Language.Igbo, option=TranslateOption.FullyTranslated),
    TranslateMode.Translated(language=Language.Igbo, option=TranslateOption.PartiallyTranslated),
    TranslateMode.Translated(language=Language.Igbo, option=TranslateOption.FullyTranslatedPromptTranslate),
    TranslateMode.Translated(language=Language.Igbo, option=TranslateOption.FullyTranslatedPreTranslate),
    TranslateMode.Translated(language=Language.Igbo, option=TranslateOption.FullyTranslatedPostTranslate),
]:
    experiments.append(ToolExperiment(translate, AddNoiseMode.NoNoise))

config = ToolConfig(
    Model.Api(ApiModel.Gpt5Nano),
    experiments
)
