### Infrastructures and Metrics for Assessing Energy-Efficient Large Language Model Training

A compact toolkit to profile, analyze, and compare energy efficiency during LLM training and evaluation. It includes reusable metrics, simple experimental setups, and integrations for benchmarking.

### Repository layout

- **`metriwatt/`**: Core library for profiling, metrics, model/network definitions, and evaluation helpers.
- **`experiments/metrics/`**: Notebooks, scripts, and results for FLOPs/throughput/hyper-surface analyses.
- **`experiments/infrastructure/`**: Power/energy profiling utilities and result summaries across GPU types/precisions.

### Quickstart

1. Install dependencies (requires Poetry):

```bash
poetry install
```

2. Activate the environment:

```bash
poetry env activate
```

3. Explore metrics experiments and power profiling:

- See `experiments/metrics/README.md` for metrics workflows and notebooks.
- See `experiments/infrastructure/README.md` for energy profiling instructions.
