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

- See `experiments/metrics/README.md` for metrics workflows and notebooks.
- See `experiments/architecture/README.md` for energy profiling instructions.


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
