import torch
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import get_cause_neutral_effect_sentences

from typing import Dict, Union, List, Generator

def load_text_generator(modelname: str) -> InstructionTextGenerationPipeline:
    """
    Loading text generation pipeline
    :param modelname: str, huggingface link to the model or local path to the checkpoint
    :return: InstructionTextGenerationPipeline, loaded pipeline
    """
    tokenizer = AutoTokenizer.from_pretrained(
        modelname,
        padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        modelname,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    func_generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
    return func_generate_text

class LLMDataPrompter:

    """
    Utils to build prompt for the COPA dataset
    """

    @classmethod
    def make_prompt(cls, sentence_one: str, sentence_two: str) -> str:
        """
        Create a prompt with given sentences for pair of sentences
        """
        return f"""Is "{sentence_one}" cause "{sentence_two}"? Yes or No"""

    @classmethod
    def _get_prompts(cls, sample_dict: Dict[str, Union[str, int]]) -> Dict[str, str]:
        """
        Creating prompts for the sample from COPA dataset.
        :param sample_dict: dict, sample from COPA dataset
        :return: dict of the format:
            {
                "cause": str, cause sentence
                "neutral": str, neutral to the cause sentence
                "effect": str, effect of the cause sentence
                "prompt_positive": str, prompt for cause-effect LLM input
                "prompt_negative": str, prompt for cause-neutral LLM input
            }
        """

        cause, neutral, effect = get_cause_neutral_effect_sentences(sample_dict=sample_dict)

        prompt_dict = {
            "cause": cause,
            "neutral": neutral,
            "effect": effect,
            "prompt_positive": cls.make_prompt(cause, effect),
            "prompt_negative": cls.make_prompt(cause, neutral)
        }

        return prompt_dict

    @classmethod
    def gen_prompts_dicts(self, dataset: List[Dict[str, Union[str, int]]]) -> Generator[Dict[str, str], None, None]:
        """
        Generating LLM inputs
        :param dataset: list of dicts, COPA dataset
        :return: generates dicts, see ._get_prompts output for more details
        """

        for sample_dict in dataset:
            yield self._get_prompts(
                sample_dict=sample_dict,
            )
