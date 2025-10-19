import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from metriwatt.evaluation import ModelEnergyEvaluator, EnergyEvaluationArguments
import time
import gc

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_SHORT_NAME = "llama3.2_1b"

# Optimized parameters for better GPU utilization
OPTIMIZED_CONFIGS = {
    "small_batch": {
        "batch_size": 32,
        "input_length": 512,  # Longer sequences for better utilization
        "num_samples": 200,
        "num_warmup_samples": 20,
        "nvidia_query_interval": 10,  # Less frequent queries to reduce overhead
        "gradient_accumulation_steps": 4,
    },
    "medium_batch": {
        "batch_size": 64,
        "input_length": 1024,
        "num_samples": 150,
        "num_warmup_samples": 15,
        "nvidia_query_interval": 15,
        "gradient_accumulation_steps": 2,
    },
    "large_batch": {
        "batch_size": 128,
        "input_length": 2048,
        "num_samples": 100,
        "num_warmup_samples": 10,
        "nvidia_query_interval": 20,
        "gradient_accumulation_steps": 1,
    }
}

def get_optimal_batch_size(model, device, dtype=torch.bfloat16):
    """Automatically determine optimal batch size based on GPU memory"""
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())

    # Get GPU memory info
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(device).total_memory
        available_memory = gpu_memory * 0.8  # Use 80% of GPU memory

        # Estimate memory per sample (rough approximation)
        memory_per_sample = model_size * 4  # Forward + backward + gradients

        optimal_batch_size = int(available_memory / memory_per_sample)
        optimal_batch_size = max(16, min(optimal_batch_size, 256))  # Clamp between 16-256

        print(f"GPU Memory: {gpu_memory / 1e9:.1f}GB")
        print(f"Model Size: {model_size / 1e9:.1f}GB")
        print(f"Estimated optimal batch size: {optimal_batch_size}")

        return optimal_batch_size
    return 32

def load_model_optimized():
    """Load model with optimizations for better performance"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Use bfloat16 for better performance on modern GPUs
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Automatic device placement
        low_cpu_mem_usage=True
    )

    # Enable optimizations
    model = model.eval()

    print("Model loaded successfully (no compilation)")

    return model, tokenizer

def evaluate_model_optimized(config_name="medium_batch"):
    """Evaluate model with optimized settings"""
    print(f"Using configuration: {config_name}")
    config = OPTIMIZED_CONFIGS[config_name]

    model, tokenizer = load_model_optimized()

    # Auto-determine optimal batch size
    optimal_batch_size = get_optimal_batch_size(model, model.device)
    config["batch_size"] = min(config["batch_size"], optimal_batch_size)

    print(f"Final configuration: {config}")

    eval_args = EnergyEvaluationArguments(
        num_samples=config["num_samples"],
        input_length=config["input_length"],
        batch_size=config["batch_size"],
        num_warmup_samples=config["num_warmup_samples"],
        nvidia_query_interval=config["nvidia_query_interval"]
    )

    evaluator = ModelEnergyEvaluator(
        model, tokenizer, args=eval_args, verbose=True
    )

    # Clear GPU cache before measurement
    torch.cuda.empty_cache()
    gc.collect()

    print("Starting optimized evaluation...")
    start_time = time.time()

    metrics = evaluator.evaluate(execute=["forward", "backward"])

    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")

    # Calculate efficiency metrics
    energy_forward = metrics["forward_energy_mean"]
    energy_backward = metrics["backward_energy_mean"]
    flops_forward = metrics["forward_flops_sum"]
    flops_backward = metrics["backward_flops_sum"]

    gflops_per_joule_forward = (flops_forward / energy_forward) / 1e9
    gflops_per_joule_backward = (flops_backward / energy_backward) / 1e9

    print(f"\n=== OPTIMIZED RESULTS ===")
    print(f"Batch size: {config['batch_size']}")
    print(f"Input length: {config['input_length']}")
    print(f"Forward - Energy: {energy_forward:.3f}J, FLOPs: {flops_forward/1e9:.1f}G, Efficiency: {gflops_per_joule_forward:.2f} GFLOPs/J")
    print(f"Backward - Energy: {energy_backward:.3f}J, FLOPs: {flops_backward/1e9:.1f}G, Efficiency: {gflops_per_joule_backward:.2f} GFLOPs/J")

    # Add configuration info to results
    metrics["config_name"] = config_name
    metrics["batch_size"] = config["batch_size"]
    metrics["input_length"] = config["input_length"]
    metrics["gflops_per_joule_forward"] = gflops_per_joule_forward
    metrics["gflops_per_joule_backward"] = gflops_per_joule_backward

    return metrics

def benchmark_all_configs():
    """Benchmark all configurations to find the best one"""
    config_name = input("Enter configuration name (small_batch/medium_batch/large_batch) or 'all' for benchmarking: ").strip()

    if config_name == "all":
        results = {}
        for config in OPTIMIZED_CONFIGS.keys():
            print(f"\n{'='*50}")
            print(f"Benchmarking {config}")
            print(f"{'='*50}")

            try:
                metrics = evaluate_model_optimized(config)
                results[config] = metrics

                # Save individual results
                jsonl_path = f"results/optimized_{config}_{MODEL_SHORT_NAME}_energy_results.jsonl"
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    sanitized = {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
                    f.write(json.dumps(sanitized) + "\n")
                print(f"Saved results to {jsonl_path}")

            except Exception as e:
                print(f"Error benchmarking {config}: {e}")
                continue

        # Find best configuration
        best_config = None
        best_efficiency = 0
        for config, metrics in results.items():
            total_efficiency = metrics["gflops_per_joule_forward"] + metrics["gflops_per_joule_backward"]
            if total_efficiency > best_efficiency:
                best_efficiency = total_efficiency
                best_config = config

        print(f"\n{'='*50}")
        print(f"BEST CONFIGURATION: {best_config}")
        print(f"Total Efficiency: {best_efficiency:.2f} GFLOPs/J")
        print(f"{'='*50}")

        return results[best_config] if best_config else None

    else:
        if config_name not in OPTIMIZED_CONFIGS:
            print(f"Invalid configuration. Available: {list(OPTIMIZED_CONFIGS.keys())}")
            return None

        return evaluate_model_optimized(config_name)

if __name__ == "__main__":
    results = benchmark_all_configs()

    if results:
        config_name = results.get("config_name", "unknown")
        jsonl_path = f"results/optimized_{config_name}_{MODEL_SHORT_NAME}_energy_results.jsonl"

        with open(jsonl_path, "w", encoding="utf-8") as f:
            sanitized = {k: v.item() if hasattr(v, "item") else v for k, v in results.items()}
            f.write(json.dumps(sanitized) + "\n")
        print(f"Final results saved to {jsonl_path}")
