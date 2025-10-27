import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from flopmetrics.evaluation import ModelEnergyEvaluator, EnergyEvaluationArguments
import time
import gc
import os
import argparse
import glob

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_SHORT_NAME = "llama3.2_1b"

CONFIGS = {
    "small": {
        "batch_size": 8,
        "input_length": 256,
        "num_samples": 100,
        "num_warmup_samples": 10,
        "nvidia_query_interval": 10,
        "precision": "bf16",
    },
    "mid": {
        "batch_size": 8,
        "input_length": 512,
        "num_samples": 100,
        "num_warmup_samples": 10,
        "nvidia_query_interval": 10,
        "precision": "bf16",
    },
    "large": {
        "batch_size": 8,
        "input_length": 1024,
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
        device_map="auto",
        low_cpu_mem_usage=True
    )

    model = model.eval()

    print(f"Model loaded successfully with {precision} precision")
    return model, tokenizer

def evaluate_config(model, tokenizer, config_name, config, gpu_config, forward_only=False):
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

    torch.cuda.empty_cache()
    gc.collect()

    print("Starting evaluation...")
    start_time = time.time()

    try:
        if forward_only:
            metrics = evaluator.evaluate(execute=["forward"])
            print("Running forward pass only (inference mode)")
        else:
            metrics = evaluator.evaluate(execute=["forward", "backward"])
            print("Running forward and backward passes (training mode)")
    except Exception as e:
        print(f"Error during evaluation: {e}")
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
    flops_forward = metrics["forward_flops_sum"]
    gflops_per_joule_forward = (flops_forward / energy_forward) / 1e9

    if forward_only:
        energy_backward = 0.0
        flops_backward = 0.0
        gflops_per_joule_backward = 0.0
        print(f"\n=== {config_name.upper()} RESULTS (FORWARD ONLY) ===")
        print(f"Batch size: {config['batch_size']}")
        print(f"Input length: {config['input_length']}")
        print(f"Forward - Energy: {energy_forward:.3f}J, FLOPs: {flops_forward/1e9:.1f}G, Efficiency: {gflops_per_joule_forward:.2f} GFLOPs/J")
    else:
        energy_backward = metrics["backward_energy_mean"]
        flops_backward = metrics["backward_flops_sum"]
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
    metrics["precision"] = config.get("precision", "bf16")
    metrics["gflops_per_joule_forward"] = gflops_per_joule_forward
    metrics["gflops_per_joule_backward"] = gflops_per_joule_backward
    metrics["gpu_config"] = gpu_config

    save_individual_result(metrics, config_name, gpu_config, config.get('precision', 'bf16'))

    return metrics

def save_individual_result(metrics, config_name, gpu_config, precision="bf16"):
    """Save individual configuration result to file"""
    os.makedirs("results_optimized", exist_ok=True)

    individual_path = f"results_optimized/{gpu_config}_{precision}_{MODEL_SHORT_NAME}_{config_name}_result.jsonl"

    with open(individual_path, "w", encoding="utf-8") as f:
        sanitized = {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
        f.write(json.dumps(sanitized) + "\n")

    print(f"📁 Individual result saved to: {individual_path}")

def unify_results(gpu_config, precision="bf16"):
    """Unify existing individual result files into a single file"""
    results_dir = "results_optimized"
    if not os.path.exists(results_dir):
        print(f"❌ Results directory '{results_dir}' does not exist")
        return

    pattern = f"{results_dir}/{gpu_config}_{precision}_{MODEL_SHORT_NAME}_*_result.jsonl"
    individual_files = glob.glob(pattern)

    if not individual_files:
        print(f"❌ No individual result files found matching pattern: {pattern}")
        return

    print(f"📁 Found {len(individual_files)} individual result files:")
    for file in individual_files:
        print(f"   - {os.path.basename(file)}")

    unified_path = f"{results_dir}/{gpu_config}_{precision}_{MODEL_SHORT_NAME}_all_configs_results.jsonl"

    with open(unified_path, "w", encoding="utf-8") as unified_file:
        for individual_file in sorted(individual_files):
            try:
                with open(individual_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        unified_file.write(content + "\n")
                        print(f"✅ Added: {os.path.basename(individual_file)}")
            except Exception as e:
                print(f"❌ Error reading {individual_file}: {e}")

    print(f"\n🎉 Unified results saved to: {unified_path}")

def benchmark_machine(configs_to_run=None, precision="bf16", forward_only=False):
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
            metrics = evaluate_config(model, tokenizer, config_name, config, gpu_config, forward_only)
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
        if forward_only:
            total_efficiency = metrics["gflops_per_joule_forward"]
        else:
            total_efficiency = metrics["gflops_per_joule_forward"] + metrics["gflops_per_joule_backward"]
        if total_efficiency > best_efficiency:
            best_efficiency = total_efficiency
            best_config = config_name

    print(f"\n{'='*80}")
    print(f"BENCHMARK SUMMARY FOR {gpu_config.upper()}")
    print(f"{'='*80}")

    # Sort results by efficiency
    if forward_only:
        sorted_results = sorted(all_results.items(),
                               key=lambda x: x[1]["gflops_per_joule_forward"],
                               reverse=True)
    else:
        sorted_results = sorted(all_results.items(),
                               key=lambda x: x[1]["gflops_per_joule_forward"] + x[1]["gflops_per_joule_backward"],
                               reverse=True)

    for i, (config_name, metrics) in enumerate(sorted_results):
        if forward_only:
            total_eff = metrics["gflops_per_joule_forward"]
            print(f"{i+1}. {config_name.upper()}: {total_eff:.2f} GFLOPs/J (Forward Only)")
            print(f"   Batch: {metrics['batch_size']}, Length: {metrics['input_length']}")
            print(f"   Forward: {metrics['gflops_per_joule_forward']:.2f}")
        else:
            total_eff = metrics["gflops_per_joule_forward"] + metrics["gflops_per_joule_backward"]
            print(f"{i+1}. {config_name.upper()}: {total_eff:.2f} GFLOPs/J")
            print(f"   Batch: {metrics['batch_size']}, Length: {metrics['input_length']}")
            print(f"   Forward: {metrics['gflops_per_joule_forward']:.2f}, Backward: {metrics['gflops_per_joule_backward']:.2f}")

    if best_config:
        print(f"\n🏆 BEST CONFIGURATION: {best_config.upper()}")
        print(f"   Total Efficiency: {best_efficiency:.2f} GFLOPs/J")


    # Save unified results (all configurations in one file for easy comparison)
    os.makedirs("results_optimized", exist_ok=True)
    jsonl_path = f"results_optimized/{gpu_config}_{precision}_{MODEL_SHORT_NAME}_all_configs_results.jsonl"

    with open(jsonl_path, "a", encoding="utf-8") as f:
        for config_name, metrics in all_results.items():
            sanitized = {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
            f.write(json.dumps(sanitized) + "\n")

    print(f"\n📁 Unified results appended to: {jsonl_path}")
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
    parser.add_argument(
        "--unify",
        action="store_true",
        help="Unify existing individual result files into a single file"
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Run only forward pass (no backward pass)"
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

    if args.unify:
        gpu_config = input("Enter GPU configuration name to unify (e.g., a40, l40): ").strip()
        if not gpu_config:
            raise ValueError("GPU configuration name cannot be empty")
        unify_results(gpu_config, args.precision)
        return

    configs_to_run = args.configs
    benchmark_machine(configs_to_run, args.precision, args.forward_only)

if __name__ == "__main__":
    main()
