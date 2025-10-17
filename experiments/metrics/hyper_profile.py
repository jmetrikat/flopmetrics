""" Profiles a toy network for various configurations of N, D, and M,"""
import itertools
import os
from datetime import datetime
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
from metriwatt.ncu import NCUProfiler
from metriwatt.network import run_toy_network_forward_ncu, run_toy_network_forward_backward_ncu, construct_toy_network_and_input_for_ncu

def _format_range_label(values):
    """Return a min-max compact label for a list of ints, e.g., [2,4,8] -> "2-8"."""
    if not values:
        return ""
    try:
        vmin = min(values)
        vmax = max(values)
    except Exception:
        return ""
    return f"{vmin}-{vmax}"

def _prepare_run_dirs(n_list, d_list, m_list, root_dir="results_hyper_surface"):
    """Create a timestamped run directory with raw/processed/plots subfolders.

    The directory name encodes the ranges for n, d, m.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    n_label = _format_range_label(n_list)
    d_label = _format_range_label(d_list)
    m_label = _format_range_label(m_list)
    run_dir = os.path.join(root_dir, f"{timestamp}_n{n_label}_d{d_label}_m{m_label}")
    raw_dir = os.path.join(run_dir, "raw")
    processed_dir = os.path.join(run_dir, "processed")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return run_dir, raw_dir, processed_dir, plots_dir

def _extract_param_sets_from_raw(raw_dir):
    """Parse CSV filenames in raw_dir to infer unique sorted n, d, m lists.

    Expected filenames like: hyper_surface_n{n}_d{d}_m{m}_setup*.csv
    """
    pattern = re.compile(r"hyper_surface_n(\d+)_d(\d+)_m(\d+)_")
    n_values = set()
    d_values = set()
    m_values = set()
    try:
        for name in os.listdir(raw_dir):
            match = pattern.match(name)
            if match:
                n_values.add(int(match.group(1)))
                d_values.add(int(match.group(2)))
                m_values.add(int(match.group(3)))
    except FileNotFoundError:
        pass
    return sorted(n_values), sorted(d_values), sorted(m_values)

def run_analysis_for_existing_run(run_subdir, root_dir="results_hyper_surface", do_plots=True):
    """Run only analysis and plotting for an existing run directory.

    Args:
        run_subdir: The subfolder name inside root_dir (e.g., "20250922-111810_n2-32_d128-1024_m16-256").
        root_dir: Parent directory that contains run folders.
        do_plots: Whether to generate plots into the run's plots directory.
    """
    run_dir = os.path.join(root_dir, run_subdir)
    raw_dir = os.path.join(run_dir, "raw")
    processed_dir = os.path.join(run_dir, "processed")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    n_list, d_list, m_list = _extract_param_sets_from_raw(raw_dir)
    if not (n_list and d_list and m_list):
        print(f"No parameters inferred from {raw_dir}; nothing to analyze.")
        return

    print(f"Analyzing existing run: {run_dir}")
    analyze_hyper_surface_metrics(n_list, d_list, m_list, raw_dir=raw_dir, processed_dir=processed_dir)

    if do_plots:
        metrics = ["omega_fwd", "rho_fwd", "kappa_fwd", "omega_bwd", "rho_bwd", "kappa_bwd", "percent_diff_fwd", "percent_diff_bwd"]
        for n_val in n_list:
            for metric in metrics:
                print(f"Plotting {metric} for n={n_val}")
                plot_metric_surface_for_fixed_n(processed_dir=processed_dir, plots_dir=plots_dir, metric=metric, n_fixed=n_val)
                # Also create log-scale version for better small value visibility
                plot_metric_surface_for_fixed_n_log(processed_dir=processed_dir, plots_dir=plots_dir, metric=metric, n_fixed=n_val)
                # Create filtered version focusing on larger values
                plot_metric_surface_for_fixed_n_filtered(processed_dir=processed_dir, plots_dir=plots_dir, metric=metric, n_fixed=n_val)

def run_profile_for_existing_run(run_subdir, n_list, d_list, m_list, root_dir="results_hyper_surface"):
    """Continue profiling into an existing run's raw directory.

    Skips any (n,d,m) that already have their CSVs present.
    """
    run_dir = os.path.join(root_dir, run_subdir)
    raw_dir = os.path.join(run_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    print(f"Resuming profiling into: {raw_dir}")
    profile_and_save_ncu_metrics(n_list, d_list, m_list, result_dir=raw_dir)

def profile_and_save_ncu_metrics(n_list, d_list, m_list, result_dir="results_hyper_surface/raw"):
    """
    Profiles the toy network for all (N, D, M) configs, saving all NCU metrics as CSV files.
    """
    os.makedirs(result_dir, exist_ok=True)
    total = len(n_list) * len(d_list) * len(m_list)
    print(f"Profiling {total} configurations...")
    for i, (n, d, m) in enumerate(itertools.product(n_list, d_list, m_list), 1):
        print(f"Progress: [{i}/{total}] n={n}, d={d}, m={m}")
        base = os.path.join(result_dir, f"hyper_surface_n{n}_d{d}_m{m}")
        setup_csv = base + "_setup.csv"
        fwd_csv = base + "_setup_forward.csv"
        bwd_csv = base + "_setup_forward_backward.csv"
        if all(os.path.exists(f) for f in [fwd_csv, bwd_csv, setup_csv]):
            print(f"Skipping existing files for n={n}, d={d}, m={m}")
            continue
        try:
            # Setup FLOPs
            ncu_setup = NCUProfiler()
            ncu_setup.profile_function(construct_toy_network_and_input_for_ncu, {"dim": d, "n_layers": n, "n_tokens": m})
            ncu_setup.result.to_csv(setup_csv, index=True)

            # Forward pass profiling
            ncu_fwd = NCUProfiler()
            ncu_fwd.profile_function(run_toy_network_forward_ncu, {"dim": d, "n_layers": n, "n_tokens": m})
            ncu_fwd.result.to_csv(fwd_csv, index=True)

            # Backward pass profiling
            ncu_bwd = NCUProfiler()
            ncu_bwd.profile_function(run_toy_network_forward_backward_ncu, {"dim": d, "n_layers": n, "n_tokens": m})
            ncu_bwd.result.to_csv(bwd_csv, index=True)

            print(f"Saved: {fwd_csv}, {bwd_csv}, {setup_csv}")
        except Exception as e:
            print(f"Failed for n={n}, d={d}, m={m}: {e}")

def analyze_hyper_surface_metrics(n_list, d_list, m_list, raw_dir="results_hyper_surface/raw", processed_dir="results_hyper_surface/processed", summary_csv="hyper_surface_summary.csv"):
    """
    Analyzes the saved NCU CSVs for all (n, d, m) configs and computes overhead metrics.
    """
    os.makedirs(processed_dir, exist_ok=True)
    records = []
    for n, d, m in itertools.product(n_list, d_list, m_list):
        base = os.path.join(raw_dir, f"hyper_surface_n{n}_d{d}_m{m}")
        setup_csv = base + "_setup.csv"
        fwd_csv = base + "_setup_forward.csv"
        bwd_csv = base + "_setup_forward_backward.csv"
        if not all(os.path.exists(f) for f in [fwd_csv, bwd_csv, setup_csv]):
            continue

        # Load CSVs
        df_setup = pd.read_csv(setup_csv)
        df_fwd = pd.read_csv(fwd_csv)
        df_bwd = pd.read_csv(bwd_csv)

        # Drop setup rows from forward
        max_setup_index = df_setup.index.max()
        df_fwd_filtered = df_fwd[df_fwd.index > max_setup_index].copy()

        # Drop setup and forward rows from backward
        max_fwd_index = df_fwd.index.max()
        df_bwd_filtered = df_bwd[df_bwd.index > max_fwd_index].copy()

        def sum_metric_by_kernel(df, metric, kernel_pattern):
            return df[
                (df["Metric Name"].str.contains(metric, case=False, na=False)) &
                (df["Kernel Name"].str.contains(kernel_pattern, case=False, na=False))
            ]["Metric Value"].sum()

        # Forward metrics by component
        # Forward GEMM metrics
        ffma_gemm_fwd = sum_metric_by_kernel(df_fwd_filtered, "ffma", "sgemm")
        fadd_gemm_fwd = sum_metric_by_kernel(df_fwd_filtered, "fadd", "sgemm")
        fmul_gemm_fwd = sum_metric_by_kernel(df_fwd_filtered, "fmul", "sgemm")
        ncu_total_flops_gemm_fwd = ffma_gemm_fwd * 2 + fadd_gemm_fwd + fmul_gemm_fwd

        # Forward KSplit metrics
        ffma_ksplit_fwd = sum_metric_by_kernel(df_fwd_filtered, "ffma", "splitK|reduce_kernel")
        fadd_ksplit_fwd = sum_metric_by_kernel(df_fwd_filtered, "fadd", "splitK|reduce_kernel")
        fmul_ksplit_fwd = sum_metric_by_kernel(df_fwd_filtered, "fmul", "splitK|reduce_kernel")
        ncu_total_flops_ksplit_fwd = ffma_ksplit_fwd * 2 + fadd_ksplit_fwd + fmul_ksplit_fwd

        # Forward Activation metrics
        ffma_activation_fwd = sum_metric_by_kernel(df_fwd_filtered, "ffma", "elementwise")
        fadd_activation_fwd = sum_metric_by_kernel(df_fwd_filtered, "fadd", "elementwise")
        fmul_activation_fwd = sum_metric_by_kernel(df_fwd_filtered, "fmul", "elementwise")
        ncu_total_flops_activation_fwd = ffma_activation_fwd * 2 + fadd_activation_fwd + fmul_activation_fwd

        ncu_total_flops_fwd = ncu_total_flops_gemm_fwd + ncu_total_flops_ksplit_fwd + ncu_total_flops_activation_fwd

        # Backward metrics by component
        # Backward GEMM metrics
        ffma_gemm_bwd = sum_metric_by_kernel(df_bwd_filtered, "ffma", "sgemm")
        fadd_gemm_bwd = sum_metric_by_kernel(df_bwd_filtered, "fadd", "sgemm")
        fmul_gemm_bwd = sum_metric_by_kernel(df_bwd_filtered, "fmul", "sgemm")
        ncu_total_flops_gemm_bwd = ffma_gemm_bwd * 2 + fadd_gemm_bwd + fmul_gemm_bwd

        # Backward KSplit metrics
        ffma_ksplit_bwd = sum_metric_by_kernel(df_bwd_filtered, "ffma", "splitK|reduce_kernel")
        fadd_ksplit_bwd = sum_metric_by_kernel(df_bwd_filtered, "fadd", "splitK|reduce_kernel")
        fmul_ksplit_bwd = sum_metric_by_kernel(df_bwd_filtered, "fmul", "splitK|reduce_kernel")
        ncu_total_flops_ksplit_bwd = ffma_ksplit_bwd * 2 + fadd_ksplit_bwd + fmul_ksplit_bwd

        # Backward Activation metrics
        ffma_activation_bwd = sum_metric_by_kernel(df_bwd_filtered, "ffma", "elementwise")
        fadd_activation_bwd = sum_metric_by_kernel(df_bwd_filtered, "fadd", "elementwise")
        fmul_activation_bwd = sum_metric_by_kernel(df_bwd_filtered, "fmul", "elementwise")
        ncu_total_flops_activation_bwd = ffma_activation_bwd * 2 + fadd_activation_bwd + fmul_activation_bwd

        ncu_total_flops_bwd = ncu_total_flops_gemm_bwd + ncu_total_flops_ksplit_bwd + ncu_total_flops_activation_bwd

        # Theoretical FLOPs
        theoretical_gemm_flops = n * 2 * d * d * m
        theoretical_activation_flops = n * d * m
        theoretical_total_flops = theoretical_gemm_flops + theoretical_activation_flops

        theoretical_gemm_flops_bwd = (n - 1) * 4 * d * d * m + 2 * d * d * m
        theoretical_activation_flops_bwd = (n - 1) * 2 * d * m + d * m
        theoretical_total_flops_bwd = theoretical_gemm_flops_bwd + theoretical_activation_flops_bwd

        # Overhead metrics
        # Forward pass overheads
        omega_fwd = ncu_total_flops_gemm_fwd / theoretical_gemm_flops - 1 if theoretical_gemm_flops else float('nan')
        # rho_fwd as absolute FLOPs per D*M operation (not overhead factor)
        rho_fwd = ncu_total_flops_ksplit_fwd / (n * d * m) if n * d * m != 0 else float('nan')
        # kappa_fwd as overhead factor like epsilon
        kappa_fwd = ncu_total_flops_activation_fwd / theoretical_activation_flops - 1 if theoretical_activation_flops else float('nan')

        # Backward pass overheads
        omega_bwd = ncu_total_flops_gemm_bwd / theoretical_gemm_flops_bwd - 1 if theoretical_gemm_flops_bwd else float('nan')
        # rho_bwd as absolute FLOPs per D*M operation (not overhead factor)
        rho_bwd = ncu_total_flops_ksplit_bwd / ((2 * n - 1) * d * m) if ((2 * n - 1) * d * m) != 0 else float('nan')
        # kappa_bwd as overhead factor like epsilon
        kappa_bwd = ncu_total_flops_activation_bwd / theoretical_activation_flops_bwd - 1 if theoretical_activation_flops_bwd else float('nan')

        # Percentage differences (total), expressed in percent
        percent_diff_fwd = ((ncu_total_flops_fwd - theoretical_total_flops) / theoretical_total_flops * 100.0) if theoretical_total_flops else float('nan')
        percent_diff_bwd = ((ncu_total_flops_bwd - theoretical_total_flops_bwd) / theoretical_total_flops_bwd * 100.0) if theoretical_total_flops_bwd else float('nan')

        records.append({
            "n": n, "d": d, "m": m,
            "ncu_total_flops_fwd": ncu_total_flops_fwd,
            "theoretical_total_flops_fwd": theoretical_total_flops,
            "percent_diff_fwd": percent_diff_fwd,
            "ncu_total_flops_bwd": ncu_total_flops_bwd,
            "theoretical_total_flops_bwd": theoretical_total_flops_bwd,
            "percent_diff_bwd": percent_diff_bwd,
            "omega_fwd": omega_fwd,
            "rho_fwd": rho_fwd,
            "kappa_fwd": kappa_fwd,
            "omega_bwd": omega_bwd,
            "rho_bwd": rho_bwd,
            "kappa_bwd": kappa_bwd,
        })
    df_summary = pd.DataFrame(records)
    summary_path = os.path.join(processed_dir, summary_csv)
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

def plot_metric_surface_for_fixed_n(processed_dir="results_hyper_surface/processed", plots_dir="results_hyper_surface/plots", summary_csv="hyper_surface_summary.csv",
                                    metric="omega_fwd", n_fixed=4):
    """
    Plots a 3D surface for a given metric over m (x-axis) and d (y-axis), fixing n. Saves as HTML.
    """
    os.makedirs(plots_dir, exist_ok=True)
    summary_path = os.path.join(processed_dir, summary_csv)
    df = pd.read_csv(summary_path)
    df = df[df["n"] == n_fixed]
    if df.empty:
        print(f"No data for n={n_fixed}")
        return

    surface = df.pivot(index="d", columns="m", values=metric)
    x = surface.columns.values  # m
    y = surface.index.values    # d
    z = surface.values

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
    fig.update_layout(
        title=f"{metric} surface for n={n_fixed}",
        scene=dict(
            xaxis_title="m (sequence length)",
            yaxis_title="d (hidden dim)",
            zaxis_title=metric
        ),
        autosize=True
    )
    plot_path_html = os.path.join(plots_dir, f"{metric}_n{n_fixed}.html")
    fig.write_html(plot_path_html)
    print(f"Saved plots to {plot_path_html}")

def plot_metric_surface_for_fixed_n_log(processed_dir="results_hyper_surface/processed", plots_dir="results_hyper_surface/plots", summary_csv="hyper_surface_summary.csv",
                                        metric="omega_fwd", n_fixed=4):
    """
    Plots a 3D surface for a given metric over m (x-axis) and d (y-axis), fixing n, using log scale for better small value visibility. Saves as HTML.
    """
    os.makedirs(plots_dir, exist_ok=True)
    summary_path = os.path.join(processed_dir, summary_csv)
    df = pd.read_csv(summary_path)
    df = df[df["n"] == n_fixed]
    if df.empty:
        print(f"No data for n={n_fixed}")
        return

    surface = df.pivot(index="d", columns="m", values=metric)
    x = surface.columns.values  # m
    y = surface.index.values    # d
    z = surface.values

    # Apply log transformation to z values for better small value visibility
    # Add small epsilon to avoid log(0)
    z_log = np.log10(z + 1e-10)

    fig = go.Figure(data=[go.Surface(z=z_log, x=x, y=y, colorscale='Viridis')])
    fig.update_layout(
        title=f"{metric} surface for n={n_fixed} (log scale)",
        scene=dict(
            xaxis_title="m (sequence length)",
            yaxis_title="d (hidden dim)",
            zaxis_title=f"log10({metric})"
        ),
        autosize=True
    )
    plot_path_html = os.path.join(plots_dir, f"{metric}_n{n_fixed}_log.html")
    fig.write_html(plot_path_html)
    print(f"Saved log-scale plots to {plot_path_html}")

def plot_metric_surface_for_fixed_n_filtered(processed_dir="results_hyper_surface/processed", plots_dir="results_hyper_surface/plots", summary_csv="hyper_surface_summary.csv",
                                           metric="omega_fwd", n_fixed=None, d_min=512, m_min=128):
    """
    Plots a 3D surface for a given metric over m (x-axis) and d (y-axis), fixing n,
    but filtering to only show larger values (d >= d_min, m >= m_min). If n_fixed is None, uses the largest available n.
    """
    os.makedirs(plots_dir, exist_ok=True)
    summary_path = os.path.join(processed_dir, summary_csv)
    df = pd.read_csv(summary_path)

    # If n_fixed is None, use the largest available n
    if n_fixed is None:
        n_fixed = df["n"].max()

    df = df[df["n"] == n_fixed]
    if df.empty:
        print(f"No data for n={n_fixed}")
        return

    # Filter data to focus on larger values
    df_filtered = df[(df["d"] >= d_min) & (df["m"] >= m_min)]
    if df_filtered.empty:
        print(f"No data for n={n_fixed} with d>={d_min} and m>={m_min}")
        return

    surface = df_filtered.pivot(index="d", columns="m", values=metric)
    x = surface.columns.values  # m
    y = surface.index.values    # d
    z = surface.values

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
    fig.update_layout(
        title=f"{metric} surface for n={n_fixed} (filtered: d>={d_min}, m>={m_min})",
        scene=dict(
            xaxis_title="m (sequence length)",
            yaxis_title="d (hidden dim)",
            zaxis_title=metric
        ),
        autosize=True
    )
    plot_path_html = os.path.join(plots_dir, f"{metric}_n{n_fixed}_filtered.html")
    fig.write_html(plot_path_html)
    print(f"Saved filtered plots to {plot_path_html}")

def main():
    n_list = [2, 4, 8, 16, 32]              # number of layers
    d_list = list(range(128, 2049, 128))    # hidden dimension: start=128, stop=2048, step=128
    m_list = list(range(16, 257, 16))       # sequence length: start=16, stop=256, step=16
    run_dir, raw_dir, processed_dir, plots_dir = _prepare_run_dirs(n_list, d_list, m_list, root_dir="results_hyper_surface")

    # Step 1: Profile and save raw metrics
    print(f"Starting hyper-surface profiling...\nRun directory: {run_dir}\nRaw: {raw_dir}\nProcessed: {processed_dir}\nPlots: {plots_dir}")
    profile_and_save_ncu_metrics(n_list, d_list, m_list, result_dir=raw_dir)

    # Step 2: Analyze and save summary
    print("Analyzing hyper-surface metrics...")
    analyze_hyper_surface_metrics(n_list, d_list, m_list, raw_dir=raw_dir, processed_dir=processed_dir)

    # Step 3: Plot all metrics for each n
    print("Plotting hyper-surface metrics...")
    metrics = ["omega_fwd", "rho_fwd", "kappa_fwd", "omega_bwd", "rho_bwd", "kappa_bwd"]
    for n_val in n_list:
        for metric in metrics:
            print(f"Plotting {metric} for n={n_val}")
            plot_metric_surface_for_fixed_n(processed_dir=processed_dir, plots_dir=plots_dir, metric=metric, n_fixed=n_val)
            # Also create log-scale version for better small value visibility
            plot_metric_surface_for_fixed_n_log(processed_dir=processed_dir, plots_dir=plots_dir, metric=metric, n_fixed=n_val)
            # Create filtered version focusing on larger values
            plot_metric_surface_for_fixed_n_filtered(processed_dir=processed_dir, plots_dir=plots_dir, metric=metric, n_fixed=n_val)


if __name__ == "__main__":
    # OPTION 1: Start new hypersurface run from scratch
    # main()

    # OPTION 2: Resume an interrupted profiling run
    # run_profile_for_existing_run("20250922-120007_n2-32_d128-2048_m16-256", n_list, d_list, m_list)

    # OPTION 3: Analysis and plots only (for existing completed runs)
    run_analysis_for_existing_run("20250922-120007_n2-32_d128-2048_m16-256")

    # To get data from remote machine to your local machine:
    # rsync -avz --progress jannis.metrikat@bp2024rh1.cloud.sci.hpi.de:/home/jannis.metrikat/lora-bp/repo/experiments/toy_network/results_hyper_surface/20250922-120007_n2-32_d128-2048_m16-256 /Users/jmetrikat/Library/CloudStorage/OneDrive-UniversitätPotsdam/hpi/bachelor/25-ss/ba/research
