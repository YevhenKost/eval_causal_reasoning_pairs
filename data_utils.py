import os, json

from typing import Dict, Union, Tuple, Any

def get_cause_neutral_effect_sentences(sample_dict: Dict[str, Union[str, int]]) -> Tuple[str, str, str]:
    """
    Parsing cause-effect-neutral based on the dict from COPA.

    :rtype: object
    :param sample_dict: dict, dictionary with the keys: label, premise, choice1, choice2, question
    :return: tuple of str:
        cause sentence, neutral sentence, effect sentence
    """
    cause = sample_dict["premise"]
    effect = sample_dict[f"""choice{sample_dict["label"] + 1}"""]
    neutral = sample_dict["choice1"] if sample_dict["label"] == 1 else sample_dict["choice2"]

    return cause, neutral, effect

def save_sample(output: Dict[str, Any], idx: Union[str, int], savedir: str) -> None:
    """
    Saving output to the dir with the name of {idx}.json

    :param output: dict that should be saved
    :param idx: str or int, the id of the output and a name identifier to use
    :param savedir: str, path to the save dir, where to save output
    :return: None
    """
    with open(
            os.path.join(savedir, f"{str(idx)}.json"),
            "w",
            encoding="utf-8"
    ) as f:
        json.dump(output, f, ensure_ascii=False)
