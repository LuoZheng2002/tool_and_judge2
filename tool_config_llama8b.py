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
    Model.Local(LocalModel.Llama3_1_8B),  # 0.3B parameters - very fast
    experiments
)

