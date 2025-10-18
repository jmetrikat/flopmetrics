import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from metriwatt.evaluation import ModelEnergyEvaluator, EnergyEvaluationArguments

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_SHORT_NAME = "llama3.2_1b"
INPUT_LENGTH = 100
VOCAB_SAMPLE_SIZE = 1000
NUM_SAMPLES = 110
NUM_WARMUP_SAMPLES = 10
NVIDIA_QUERY_INTERVAL = 5

config_name = input("Enter configuration name (e.g., a40_fp16): ").strip()
if not config_name:
    raise ValueError("Configuration name cannot be empty")

JSONL_PATH = f"results/eff_{config_name}_{MODEL_SHORT_NAME}_energy_results.jsonl"

def evaluate_model():
    model, tokenizer = load_model()
    eval_args = EnergyEvaluationArguments(
        num_samples=NUM_SAMPLES,
        input_length=INPUT_LENGTH,
        num_warmup_samples=NUM_WARMUP_SAMPLES,
        nvidia_query_interval=NVIDIA_QUERY_INTERVAL
    )

    evaluator = ModelEnergyEvaluator(
        model, tokenizer, args=eval_args, verbose=True
    )
    metrics = evaluator.evaluate(execute=["forward", "backward"])

    energy_forward = metrics["forward_energy_mean"]
    energy_backward = metrics["backward_energy_mean"]
    flops_forward = metrics["forward_flops_sum"]
    flops_backward = metrics["backward_flops_sum"]
    print(f"average energy fwd (Ws) = {energy_forward}, FLOPs fwd = {flops_forward}, average energy bwd (Ws) = {energy_backward}, FLOPs bwd = {flops_backward}")

    return metrics


def load_model():
    """
    Load the model with the specified number of layers.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)  # when 3b doesn't fit in mem: torch_dtype=torch.bfloat16
    return model.to("cuda").eval(), tokenizer


results = evaluate_model()
with open(JSONL_PATH, "w", encoding="utf-8") as f:
    sanitized = {k: v.item() if hasattr(v, "item") else v for k, v in results.items()}
    f.write(json.dumps(sanitized) + "\n")
print(f"Saved energy results to {JSONL_PATH}")
