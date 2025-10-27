# Assessing Energy Efficiency of Large Language Model Training Using FLOP-Based Metrics

This repository accompanies my Bachelor's thesis: "Assessing Energy Efficiency of Large Language Model Training Using FLOP-Based Metrics" at the chair of Artificial Intelligence and Sustainability at the Hasso-Plattner-Institute. It contains a compact toolkit to profile, analyze, and compare energy efficiency during LLM training and evaluation.

## Repository layout

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

- See `experiments/metrics/README.md` for metrics workflows and notebooks.
- See `experiments/architecture/README.md` for energy profiling instructions.
