# Assessing Energy Efficiency of Large Language Model Training Using FLOP-Based Metrics

This repository accompanies my Bachelor's thesis: "Assessing Energy Efficiency of Large Language Model Training Using FLOP-Based Metrics" at the chair of Artificial Intelligence and Sustainability at the Hasso-Plattner-Institute. It contains a compact toolkit to profile, analyze, and compare energy efficiency during LLM training and evaluation.

## Structure

- **`flopmetrics/`**: Core library for profiling, metrics, model/network definitions, and evaluation helpers.
- **`experiments/metrics/`**: FLOP counting validation: comparing analytical, PyTorch profiler, and NCU measurements on MLP networks with hyper-parameter surface analyses.
- **`experiments/architecture/`**: GPU efficiency benchmarking: profiling Llama models across GPU architectures (V100, A40, L40, A100, H100) to measure energy consumption and compute efficiency.

## Quickstart

1. Install dependencies (requires Poetry):

```bash
./install.sh
```

2. Activate the environment:

```bash
$(poetry env activate)
```

3. Explore metrics experiments and power profiling:

- See [`experiments/metrics/README.md`](experiments/metrics/README.md) for metrics workflows and notebooks.
- See [`experiments/architecture/README.md`](experiments/architecture/README.md) for energy profiling instructions.

## Reproducibility

1. Connect to the appropriate system:

   a. For standard experiments, use the HPC cluster:

   ```bash
   ssh <firstname>.<lastname>@hpc.sci.hpi.de
   ```

   b. For NCU experiments, use the VM with root access:

   ```bash
   ssh <firstname>.<lastname>@bp2024rh1.cloud.sci.hpi.de
   ```

2. Request a GPU node:

```bash
srun --account=<account> --gpus=1 --mem=64G --cpus-per-task=10 --partition=gpu-interactive --nodelist=<nodelist> --pty bash
```

3. Enter a shell in container with CUDA dependencies installed.

4. In the container, clone the repository:

```bash
git clone https://github.com/jmetrikat/flopmetrics.git
```

5. Install the dependencies:

```bash
./install.sh
```

6. Activate the environment:

```bash
$(poetry env activate)
```

7. Run the experiments:

```bash
python experiments/metrics/hyper_profile.py new --n 2 4 8 16 --d 256 512 1024 --m 32 64 128
```

## Thesis Abstract

Artificial intelligence (AI) systems, particularly large language models (LLMs), have reached remarkable levels of capability, yet their training demands immense computational power and energy. Estimating and comparing these costs often relies on floating-point operations (FLOPs), a convenient, model-intrinsic measure of neural network scale. However, identical FLOP counts do not necessarily correspond to equal energy consumption. This thesis explores how reliably FLOPs reflect the true energy cost of training and how they can be transformed into predictive tools.

We combine (i) closed-form FLOP budgets for forward and backward passes in a controlled MLP setting, (ii) GPU-level validation with PyTorch Profiler and NVIDIA Nsight Compute, and (iii) device-inclusive efficiency measurements (FLOPs per Joule) under a fixed protocol using board-power integration. The analysis cleanly separates analytic FLOPs from the arithmetic operations actually realized on GPUs and reconstructs their difference with three interpretable terms: GEMM epilogues/algorithmic overheads, elementwise activations, and split-K reductions.

Across a broad range of layer, batch, and sequence configurations, measured on-GPU FLOPs exceed analytic lower bounds by single- to double-digit percentages. These discrepancies diminish with increasing arithmetic intensity and tile-aligned sequence lengths. Building on this factorization, we introduce GPU-adjusted FLOP formulas that accurately reproduce instruction counts and pair them with pass-specific efficiency to map FLOPs to Joules via a minimal calibration: $\widehat{E}=F_{\bullet}/\widehat{EE}$.

Empirically, we quantify FLOPs/J across five NVIDIA architectures (V100, A40, A100, H100, L40S) and three precisions (FP16, BF16, FP32), reporting a persistent forward–backward asymmetry: forward efficiency rises with sequence length as larger GEMMs better amortize epilogues, while backward efficiency falls due to added reductions, gating, and memory pressure. The L40S sets the forward (inference-like) efficiency frontier (up to 6.24 TFLOPs/J at FP16, $m{=}1024$), whereas the H100 leads in backward (training-like) efficiency (up to 0.94 TFLOPs/J at FP16, $m{=}256$). BF16 nears FP16 where natively supported, while upcasting erases gains.

The result is a lightweight, calibration-aware recipe for predicting training energy from FLOPs, together with practical guidance - choose tile-friendly shapes, prefer the lowest stable precision, and match devices to workloads - that reduces overheads and enables transparent $CO_2e$ conversion under explicit grid and PUE assumptions.
