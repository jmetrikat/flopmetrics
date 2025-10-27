# MLP Profiling and Validation

## Prerequisites

- CUDA-capable machine with NVIDIA drivers
- Python environment with project dependencies installed
- Access to NCU (NVIDIA Nsight Compute) for detailed profiling

## Notebooks

### mlp.ipynb

Provides baseline profiling and validation of FLOPs measurements:

1. **Single Configuration Analysis**

   - Profiles MLP with fixed dimensions (n=10, d=1024, m=128)
   - Compares PyTorch profiler, NCU profiler, and theoretical calculations
   - Validates measurement accuracy across different methods

2. **Multi-Dimension Comparison**
   - Sweeps through different dimensions (d from 128 to 2048)
   - Compares PyTorch vs analytical FLOP counts
   - Saves results to `results/analytical_vs_pytorch_comparison.csv`

### analysis.ipynb

Analyzes detailed NCU profiling data:

- Processes raw NCU measurements for setup, forward, and backward passes
- Separates FLOPs by kernel type (GEMM, KSplit, Activation)
- Compares NCU measurements against theoretical calculations
- Visualizes overhead metrics (omega, rho, kappa)
- Generates plots for forward and backward pass efficiency analysis

## Scripts

### hyper_profile.py

Profiles MLP networks across hyperparameter sweeps using command-line flags:

#### Start New Profiling Run

```bash
# Run with default parameters
python hyper_profile.py new

# Run with custom parameters
python hyper_profile.py new --n 2 4 8 16 --d 256 512 1024 --m 32 64 128

# Run without plot generation
python hyper_profile.py new --no-plots
```

#### Resume Interrupted Profiling Run

```bash
# Resume profiling (skips already-profiled configurations)
python hyper_profile.py resume <run_directory> --n 2 4 8 --d 128 256 512 --m 16 32 64

# Example:
python hyper_profile.py resume 20250922-120007_n2-32_d128-2048_m16-256
```

#### Analyze and Plot Existing Run

```bash
# Analyze and generate all plots
python hyper_profile.py analyze <run_directory>

# Example:
python hyper_profile.py analyze 20250922-120007_n2-32_d128-2048_m16-256

# Analyze without generating plots
python hyper_profile.py analyze <run_directory> --no-plots
```

**Default Configuration:**

- n_list: [2, 4, 8, 16, 32] layers
- d_list: 128 to 2048, step 128
- m_list: 16 to 256, step 16

**Output Structure:**

- `results_hyper_surface/<timestamp>_n<range>_d<range>_m<range>/`
  - `raw/`: NCU CSV files for each (n, d, m) configuration
  - `processed/`: Aggregated metrics and overhead calculations
  - `plots/`: 3D surface plots for each metric

## Results

All experimental results are stored in `results/`:

- **mlp\_\*.csv**: Raw and processed NCU profiling data
- **analytical_vs_pytorch_comparison.csv**: Validation of measurement accuracy
- **hyper_surface/**: Timestamped experimental runs with raw, processed, and plotted data

## Notes

- MLP networks use sigmoid activation functions
- Setup phase FLOPs are measured separately and subtracted from forward/backward passes
- NCU provides detailed kernel-level FLOP measurements
- Theoretical calculations assume standard matrix multiplication formulas
