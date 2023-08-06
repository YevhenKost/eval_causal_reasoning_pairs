from llm_utils import load_text_generator, LLMDataPrompter
from settings import llm_experiment_settings, general_settings

from data_utils import save_sample

import os, json
from tqdm import tqdm

from typing import List, Dict, Union, Any, Generator, Tuple

def gen_outputs(
        dataset: List[Dict[str, Union[str, int]]],
        func_generate_text: Any) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
    """
    Generating outputs one-by-one from the LLM model
    :param dataset: list of dicts, COPA dataset
    :param func_generate_text: trasnformers pileine, function that takes as an input text and outputs the
        dict in transformers pipeline format
    :return: generates a tuple: (int, dict), where:
        i: int, index of the sample in the dataset
        dict: generated output with the inputs
            To the output of LLMDataPrompter.gen_prompts_dicts added "outputs_positive" and
                "outputs_negative" fields with the generated results for the "prompt_positive"
                and "prompt_negative" fields respectively.
    """


    for i, prompt_dict in enumerate(
            LLMDataPrompter.gen_prompts_dicts(
                dataset=dataset
            )
    ):
        prompt_dict["outputs_positive"] = func_generate_text(
            prompt_dict["prompt_positive"]
        )
        prompt_dict["outputs_negative"] = func_generate_text(
            prompt_dict["prompt_negative"]
        )

        yield i, prompt_dict

def run():

    root_save_dir = llm_experiment_settings.SAVE_DIR
    os.makedirs(root_save_dir, exist_ok=True)

    dataset = json.load(open(
        general_settings.DATASET_PATH, "r"
    ))

    for model_name in llm_experiment_settings.MODELS:
        model_results_save_dir = os.path.join(
            root_save_dir,
            model_name.replace("/", "-")
        )
        os.makedirs(model_results_save_dir, exist_ok=True)

        llm_generator = load_text_generator(modelname=model_name)

        for idx, output in tqdm(gen_outputs(
            dataset=dataset,
            func_generate_text=llm_generator
        )):

            save_sample(
                output=output,
                idx=idx,
                savedir=model_results_save_dir
            )

if __name__ == '__main__':
    run()
