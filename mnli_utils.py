from transformers import AutoModelForSequenceClassification, AutoTokenizer

from typing import Dict, Any, Tuple

def get_logits_dict(
        premise: str,
        hypothesis: str,
        device: str,
        labels_dict: Dict[int, str],
        nli_model: Any, tokenizer: Any) -> Dict[str, float]:
    """
    Generating model outputs for the given premise and hypothesis. Not the distribution, but scores.

        :param premise: str
        :param hypothesis: str
        :param device: str, cuda or cpu
        :param labels_dict: dict of format: {int: str}, label decocing dict of the model
        :param nli_model: model (transformers) to use for prediction
        :param tokenizer: tokenizer for the model
        :return: dict: {label: float}, label with the predicted corresponding score
    """



    x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                        truncation_strategy='only_first')
    logits = nli_model(x.to(device))[0].detach().cpu().numpy().reshape(-1).tolist()
    output_dict = {label_name: logits[idx_label] for idx_label, label_name in labels_dict.items()}
    return output_dict


def load_model_tokenizer(modelname: str, device: str) -> Tuple[Any, Any]:
    """
    Loading transformers model from the huggingface or the local checkpoint
    :param modelname: str, link to the huggingface model or path to the local checkpoint
    :param device: str, cuda or cpu, device to use
    :return: tuple: loaded transformers model and tokenizer
    """
    model = AutoModelForSequenceClassification.from_pretrained(modelname).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(modelname)

    return model, tokenizer