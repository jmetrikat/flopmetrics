# Infrastructures and Metrics for Assessing Energy-Efficient Large Language Model Training

This repository accompanies my Bachelor's thesis: _"Assessing Energy Efficiency of Large Language Model Training Using FLOP-Based Metrics"_ at the chair of Artificial Intelligence and Sustainability at the Hasso-Plattner-Institute. It contains a compact toolkit to profile, analyze, and compare energy efficiency during LLM training and evaluation.

## Repository layout

- **`flopmetrics/`**: Core library for profiling, metrics, model/network definitions, and evaluation helpers.
- **`experiments/metrics/`**: Notebooks, scripts, and results for FLOPs/throughput/hyper-surface analyses.
- **`experiments/infrastructure/`**: Power/energy profiling utilities and result summaries across GPU types/precisions.

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
- See `experiments/infrastructure/README.md` for energy profiling instructions.
