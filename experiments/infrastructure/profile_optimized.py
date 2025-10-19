import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from metriwatt.evaluation import ModelEnergyEvaluator, EnergyEvaluationArguments
import time
import gc

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_SHORT_NAME = "llama3.2_1b"

# Optimized configurations focusing on batch size and sequence length
CONFIGS = {
    "small_batch": {
        "batch_size": 32,
        "input_length": 512,  # Longer sequences for better utilization
        "num_samples": 100,
        "num_warmup_samples": 10,
        "nvidia_query_interval": 10,  # Consistent across all configs for fair comparison
        "description": "Small batch optimization with longer sequences"
    },
    "medium_batch": {
        "batch_size": 64,
        "input_length": 1024,
        "num_samples": 100,
        "num_warmup_samples": 10,
        "nvidia_query_interval": 10,  # Consistent across all configs for fair comparison
        "description": "Medium batch optimization for balanced performance"
    },
    "large_batch": {
        "batch_size": 128,
        "input_length": 2048,
        "num_samples": 100,
        "num_warmup_samples": 10,
        "nvidia_query_interval": 10,  # Consistent across all configs for fair comparison
        "description": "Large batch optimization for maximum throughput"
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

def load_model():
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

    print("Model loaded successfully")
    return model, tokenizer

def evaluate_config(model, tokenizer, config_name, config):
    """Evaluate model with specific configuration"""
    print(f"\n{'='*60}")
    print(f"Evaluating configuration: {config_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}")

    # Auto-determine optimal batch size based on GPU memory
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

    print("Starting evaluation...")
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

    print(f"\n=== {config_name.upper()} RESULTS ===")
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
    metrics["description"] = config["description"]

    return metrics

def benchmark_machine():
    """Benchmark all configurations for a specific machine"""
    gpu_config = input("Enter GPU configuration name (e.g., a40_fp16, l40_fp32): ").strip()
    if not gpu_config:
        raise ValueError("GPU configuration name cannot be empty")

    print(f"\n{'='*80}")
    print(f"BENCHMARKING MACHINE: {gpu_config.upper()}")
    print(f"{'='*80}")

    # Load model once for all configurations
    model, tokenizer = load_model()

    # Evaluate all configurations
    all_results = {}
    for config_name, config in CONFIGS.items():
        try:
            metrics = evaluate_config(model, tokenizer, config_name, config.copy())
            all_results[config_name] = metrics
        except Exception as e:
            print(f"Error evaluating {config_name}: {e}")
            continue

    # Find best configuration
    best_config = None
    best_efficiency = 0
    for config_name, metrics in all_results.items():
        total_efficiency = metrics["gflops_per_joule_forward"] + metrics["gflops_per_joule_backward"]
        if total_efficiency > best_efficiency:
            best_efficiency = total_efficiency
            best_config = config_name

    print(f"\n{'='*80}")
    print(f"BENCHMARK SUMMARY FOR {gpu_config.upper()}")
    print(f"{'='*80}")

    # Sort results by efficiency
    sorted_results = sorted(all_results.items(),
                           key=lambda x: x[1]["gflops_per_joule_forward"] + x[1]["gflops_per_joule_backward"],
                           reverse=True)

    for i, (config_name, metrics) in enumerate(sorted_results):
        total_eff = metrics["gflops_per_joule_forward"] + metrics["gflops_per_joule_backward"]
        print(f"{i+1}. {config_name.upper()}: {total_eff:.2f} GFLOPs/J")
        print(f"   Batch: {metrics['batch_size']}, Length: {metrics['input_length']}")
        print(f"   Forward: {metrics['gflops_per_joule_forward']:.2f}, Backward: {metrics['gflops_per_joule_backward']:.2f}")

    if best_config:
        print(f"\n🏆 BEST CONFIGURATION: {best_config.upper()}")
        print(f"   Total Efficiency: {best_efficiency:.2f} GFLOPs/J")

        # Calculate improvement over baseline
        if "baseline" in all_results:
            baseline_eff = all_results["baseline"]["gflops_per_joule_forward"] + all_results["baseline"]["gflops_per_joule_backward"]
            improvement = best_efficiency / baseline_eff
            print(f"   Improvement over baseline: {improvement:.1f}x")

    # Save unified results
    jsonl_path = f"results_optimized/{gpu_config}_{MODEL_SHORT_NAME}_all_configs_results.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for config_name, metrics in all_results.items():
            sanitized = {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
            sanitized["gpu_config"] = gpu_config
            f.write(json.dumps(sanitized) + "\n")

    print(f"\n📁 All results saved to: {jsonl_path}")
    return all_results

if __name__ == "__main__":
    benchmark_machine()
