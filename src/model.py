
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from typing import Tuple


def load_model(
    model_name: str, tokenizer_name: str = None, device: str = "cuda"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a base model from Hugging Face's model hub.
    Args:
        model_name (str): The name of the model to load.
        tokenizer_name (str): The name of the tokenizer to load.
        device (str): The device to load the model on (default: "cuda").
    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.
    """
    if os.path.isdir(model_name) and os.path.isfile(os.path.join(model_name, "adapter.pth")):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
