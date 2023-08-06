from pydantic import BaseSettings

class GeneralSettings(BaseSettings):
    DATASET_PATH = "copa-processed-ds.json"

class LLMExperimentSettings(BaseSettings):

    # list of LLMs to use. Can be either links to huggingface or local checkpoints
    MODELS = [
        "databricks/dolly-v2-3b",
        "databricks/dolly-v2-7b",
        "databricks/dolly-v2-12b"
    ]

    # Path to the directory, where to save the results for the MODELS.
    # Will be created or overwritten automatically
    # For each model its subdirectory will be created
    SAVE_DIR = "LLM-results"

class MNLIExperimentSettings(BaseSettings):

    # MNLI model configs. These models, by default output the labels in the "labels_dict".
    # Each dict in the list contains a specific config to run the corresponding model
    MODELS_CONFIGS = [
        {
            "modelname": "facebook/bart-large-mnli",
            "device": "cuda",
            "labels_dict": {
                0: "contradiction",
                1: "neutral",
                2: "entailment"
            }
        },
        {
            "modelname": "MoritzLaurer/xlm-v-base-mnli-xnli",
            "device": "cuda",
            "labels_dict": {
                0: "contradiction",
                1: "neutral",
                2: "entailment"
            }
        },
        {
            "modelname": "MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli",
            "device": "cuda",
            "labels_dict": {
                0: "contradiction",
                1: "neutral",
                2: "entailment"
            }
        },
        {
            "modelname": "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
            "device": "cuda",
            "labels_dict": {
                0: "contradiction",
                1: "neutral",
                2: "entailment"
            }
        }
    ]

    # Path to the directory, where to save the results for the MODELS.
    # Will be created or overwritten automatically
    # For each model its subdirectory will be created
    SAVE_DIR = "MNLI-results"


general_settings = GeneralSettings()
llm_experiment_settings = LLMExperimentSettings()
mnli_experiment_settings = MNLIExperimentSettings()