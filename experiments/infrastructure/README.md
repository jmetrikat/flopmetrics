# Benchmarking Architecture Efficiency

## Prerequisites

- CUDA-capable machines with NVIDIA drivers and nvidia-smi available
- Python environment with project dependencies installed (poetry/pip)
- Access to the target GPUs (e.g., V100, A40, L40, A100, H100)

## Steps

1. Profile on each machine
   Run the optimized benchmark on every target machine. This collects per-config JSONL results and appends a unified file per GPU/precision.

```bash
# List available configs and precisions
python profile_optimized.py --list-configs

# Run all configs for FP16 (recommended) and append unified results
python profile_optimized.py --precision fp16

# Forward-only inference runs (optional)
python profile_optimized.py --precision fp16 --forward-only

# Unify previously saved individual result files into one JSONL
python profile_optimized.py --precision fp16 --unify
```

Outputs are written under `experiments/infrastructure/results_optimized/`:

- `<gpu>_<precision>_llama3.2_1b_<config>_result.jsonl` (per config)
- `<gpu>_<precision>_llama3.2_1b_all_configs_results.jsonl` (unified)

2. Aggregate and visualize
   After you have run profiling on all GPUs/precisions and copied the `*_all_configs_results.jsonl` files into `results_optimized/`, run:

```bash
python analyze_optimized.py
```

This will:

- Create plots comparing TFLOPs/J across GPUs/configs/precisions
- Draw forward vs backward efficiency with clusters and Pareto fronts
- Save a CSV with errors at `results_optimized/gpu_efficiency_comparison.csv`

## Notes

- All efficiencies in analysis are reported as TFLOPs/J
- Error bars use SEM-based propagation from measured energy/time SEM
