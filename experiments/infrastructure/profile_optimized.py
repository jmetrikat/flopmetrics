import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from metriwatt.evaluation import ModelEnergyEvaluator, EnergyEvaluationArguments
import time
import gc
import os
import argparse

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_SHORT_NAME = "llama3.2_1b"

CONFIGS = {
    "small": {
        "batch_size": 8,
        "input_length": 512,
        "num_samples": 100,
        "num_warmup_samples": 10,
        "nvidia_query_interval": 10,
        "precision": "bf16",
    },
    "mid": {
        "batch_size": 8,
        "input_length": 1024,
        "num_samples": 100,
        "num_warmup_samples": 10,
        "nvidia_query_interval": 10,
        "precision": "bf16",
    },
    "large": {
        "batch_size": 8,
        "input_length": 2048,
        "num_samples": 100,
        "num_warmup_samples": 10,
        "nvidia_query_interval": 10,
        "precision": "bf16",
    }
}

def load_model(precision="bf16"):
    """Load model with optimizations for better performance"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Choose precision based on parameter
    if precision == "fp32":
        torch_dtype = torch.float32
        print("Loading model with FP32 precision")
    elif precision == "fp16":
        torch_dtype = torch.float16
        print("Loading model with FP16 precision")
    elif precision == "bf16":
        torch_dtype = torch.bfloat16
        print("Loading model with BF16 precision")
    else:
        raise ValueError(f"Unsupported precision: {precision}. Use 'fp32', 'fp16', or 'bf16'")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map="auto",  # Automatic device placement
        low_cpu_mem_usage=True
    )

    # Enable optimizations
    model = model.eval()

    print(f"Model loaded successfully with {precision} precision")
    return model, tokenizer

def evaluate_config(model, tokenizer, config_name, config, gpu_config):
    """Evaluate model with specific configuration"""
    print(f"\n{'='*60}")
    print(f"Evaluating configuration: {config_name}")
    print(f"{'='*60}")

    print(f"Final configuration: {config}")
    print(f"Precision: {config.get('precision', 'bf16')}")

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

    try:
        metrics = evaluator.evaluate(execute=["forward", "backward"])
    except Exception as e:
        print(f"Error during evaluation: {e}")
        # Return a minimal metrics dict to avoid crashes
        return {
            "config_name": config_name,
            "batch_size": config["batch_size"],
            "input_length": config["input_length"],
            "precision": config.get("precision", "bf16"),
            "error": str(e),
            "gflops_per_joule_forward": 0,
            "gflops_per_joule_backward": 0
        }

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
    metrics["precision"] = config.get("precision", "fp16")
    metrics["gflops_per_joule_forward"] = gflops_per_joule_forward
    metrics["gflops_per_joule_backward"] = gflops_per_joule_backward
    metrics["gpu_config"] = gpu_config

    # Save individual result immediately
    save_individual_result(metrics, config_name, gpu_config, config.get('precision', 'bf16'))

    return metrics

def save_individual_result(metrics, config_name, gpu_config, precision="bf16"):
    """Save individual configuration result to file"""
    # Ensure results_optimized directory exists
    os.makedirs("results_optimized", exist_ok=True)

    # Create individual result file with precision in name
    individual_path = f"results_optimized/{gpu_config}_{precision}_{MODEL_SHORT_NAME}_{config_name}_result.jsonl"

    with open(individual_path, "w", encoding="utf-8") as f:
        sanitized = {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
        f.write(json.dumps(sanitized) + "\n")

    print(f"📁 Individual result saved to: {individual_path}")

def benchmark_machine(configs_to_run=None, precision="bf16"):
    """Benchmark all configurations for a specific machine"""
    gpu_config = input("Enter GPU configuration name (e.g., a40_fp16, l40_fp32): ").strip()
    if not gpu_config:
        raise ValueError("GPU configuration name cannot be empty")

    print(f"\n{'='*80}")
    print(f"BENCHMARKING MACHINE: {gpu_config.upper()}")
    print(f"PRECISION: {precision.upper()}")
    print(f"{'='*80}")

    # Determine which configurations to run
    if configs_to_run is None:
        configs_to_run = list(CONFIGS.keys())
    else:
        # Validate config names
        invalid_configs = [c for c in configs_to_run if c not in CONFIGS]
        if invalid_configs:
            print(f"Warning: Invalid configuration names: {invalid_configs}")
            configs_to_run = [c for c in configs_to_run if c in CONFIGS]

    print(f"Running configurations: {configs_to_run}")

    # Load model once for all configurations with specified precision
    model, tokenizer = load_model(precision)

    # Evaluate specified configurations
    all_results = {}
    for config_name in configs_to_run:
        config = CONFIGS[config_name].copy()
        # Override precision with command line argument
        config["precision"] = precision
        try:
            metrics = evaluate_config(model, tokenizer, config_name, config, gpu_config)
            all_results[config_name] = metrics
        except Exception as e:
            print(f"Error evaluating {config_name}: {e}")
            # Create error result
            all_results[config_name] = {
                "config_name": config_name,
                "batch_size": config["batch_size"],
                "input_length": config["input_length"],
                "precision": precision,
                "error": str(e),
                "gflops_per_joule_forward": 0,
                "gflops_per_joule_backward": 0,
                "gpu_config": gpu_config
            }
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
    os.makedirs("results_optimized", exist_ok=True)
    jsonl_path = f"results_optimized/{gpu_config}_{precision}_{MODEL_SHORT_NAME}_all_configs_results.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for config_name, metrics in all_results.items():
            sanitized = {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
            sanitized["gpu_config"] = gpu_config
            f.write(json.dumps(sanitized) + "\n")

    print(f"\n📁 All results saved to: {jsonl_path}")
    return all_results

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Benchmark model configurations")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS.keys()),
        help="Specific configurations to run (e.g., --configs small mid large)"
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configurations and exit"
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Precision to use for model loading (default: bf16)"
    )

    args = parser.parse_args()

    if args.list_configs:
        print("Available configurations:")
        for name, config in CONFIGS.items():
            print(f"  {name}")
        print("\nAvailable precisions:")
        print("  fp32 - Single precision (32-bit)")
        print("  fp16 - Half precision (16-bit)")
        print("  bf16 - Brain floating point (16-bit)")
        return

    configs_to_run = args.configs
    benchmark_machine(configs_to_run, args.precision)

if __name__ == "__main__":
    main()
