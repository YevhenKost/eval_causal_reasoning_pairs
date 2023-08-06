from mnli_utils import load_model_tokenizer, get_logits_dict
from data_utils import save_sample, get_cause_neutral_effect_sentences

from settings import mnli_experiment_settings, general_settings

import torch
import os, json
from tqdm import tqdm

from typing import List, Union, Dict, Any

def predict_save_model_results(
        dataset: List[Dict[str, Union[str, int]]],
        device: str,
        labels_dict: Dict[int, str],
        nli_model: Any,
        tokenizer: Any,
        save_dir: str) -> None:
    """
    Predicting score and labels for the dataset
    :param dataset: list of dicts, COPA dataset
    :param device: str, cuda or cpu, device to inference model on
    :param labels_dict: dict: {int: str}, decoding dict of labels for the model
    :param nli_model: transformers model, MNLI model to run experiment on
    :param tokenizer: transformers tokenizer for the MNLI model
    :param save_dir: str, path to the save dir, where to store the results
    :return: None,

    Saves into the save_dir the results with the "{idx}.json" name, where idx is the index of the dataset sample
    """
    with torch.no_grad():
        for i, sample_dict in tqdm(enumerate(dataset)):
            cause, neutral, effect = get_cause_neutral_effect_sentences(sample_dict=sample_dict)
            prompt_dict = {
                "cause": cause,
                "neutral": neutral,
                "effect": effect,
            }

            # run through model pre-trained on MNLI
            positive_output_dict = get_logits_dict(
                premise=cause,
                hypothesis=effect,
                device=device,
                labels_dict=labels_dict,
                nli_model=nli_model,
                tokenizer=tokenizer)
            negative_output_dict = get_logits_dict(
                premise=cause,
                hypothesis=neutral,
                device=device,
                labels_dict=labels_dict,
                nli_model=nli_model,
                tokenizer=tokenizer)

            prompt_dict["cause_effect_logits"] = positive_output_dict
            prompt_dict["cause_neutral_logits"] = positive_output_dict

            save_sample(
                output=prompt_dict,
                idx=i,
                savedir=save_dir)

def run():
    root_save_dir = mnli_experiment_settings.SAVE_DIR
    os.makedirs(root_save_dir, exist_ok=True)

    dataset = json.load(open(
        general_settings.DATASET_PATH, "r"
    ))

    for model_config in mnli_experiment_settings.MODELS_CONFIGS:

        mnli_model, mnli_tokenizer = load_model_tokenizer(
            modelname=model_config["modelname"],
            device=model_config["device"]
        )

        model_results_save_dir = os.path.join(
            root_save_dir,
            model_config["modelname"].replace("/", "-")
        )
        os.makedirs(model_results_save_dir, exist_ok=True)

        predict_save_model_results(
            dataset=dataset,
            device=model_config["device"],
            labels_dict=model_config["labels_dict"],
            nli_model=mnli_model,
            tokenizer=mnli_tokenizer,
            save_dir=model_results_save_dir
        )

if __name__ == '__main__':
    run()
