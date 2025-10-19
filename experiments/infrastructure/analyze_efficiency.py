import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import os

def load_results_from_jsonl(filepath):
    """Load results from JSONL file"""
    with open(filepath, 'r') as f:
        line = f.readline()
        return json.loads(line)

def analyze_gpu_efficiency():
    """Comprehensive analysis of GPU efficiency improvements"""

    # Find all result files
    baseline_files = glob.glob("results/eff_*_llama3.2_1b_energy_results.jsonl")
    optimized_files = glob.glob("results/optimized_*_llama3.2_1b_energy_results.jsonl")
    ultra_optimized_files = glob.glob("results/ultra_optimized_*_llama3.2_1b_energy_results.jsonl")

    print("Found result files:")
    print(f"Baseline: {len(baseline_files)} files")
    print(f"Optimized: {len(optimized_files)} files")
    print(f"Ultra Optimized: {len(ultra_optimized_files)} files")

    # Load baseline results
    baseline_results = {}
    for filepath in baseline_files:
        filename = os.path.basename(filepath)
        # Extract GPU and precision from filename
        parts = filename.replace("eff_", "").replace("_llama3.2_1b_energy_results.jsonl", "").split("_")
        gpu = parts[0].upper()
        precision = parts[1].upper()

        key = f"{gpu}_{precision}"
        baseline_results[key] = load_results_from_jsonl(filepath)
        baseline_results[key]["gpu"] = gpu
        baseline_results[key]["precision"] = precision
        baseline_results[key]["type"] = "baseline"

    # Load optimized results
    optimized_results = {}
    for filepath in optimized_files:
        filename = os.path.basename(filepath)
        parts = filename.replace("optimized_", "").replace("_llama3.2_1b_energy_results.jsonl", "").split("_")
        config = parts[0]

        key = f"optimized_{config}"
        optimized_results[key] = load_results_from_jsonl(filepath)
        optimized_results[key]["config"] = config
        optimized_results[key]["type"] = "optimized"

    # Load ultra optimized results
    ultra_results = {}
    for filepath in ultra_optimized_files:
        filename = os.path.basename(filepath)
        parts = filename.replace("ultra_optimized_", "").replace("_llama3.2_1b_energy_results.jsonl", "").split("_")
        config = parts[0]

        key = f"ultra_{config}"
        ultra_results[key] = load_results_from_jsonl(filepath)
        ultra_results[key]["config"] = config
        ultra_results[key]["type"] = "ultra_optimized"

    # Combine all results
    all_results = {**baseline_results, **optimized_results, **ultra_results}

    if not all_results:
        print("No results found. Please run the profiling scripts first.")
        return

    # Create comparison dataframe
    comparison_data = []
    for key, data in all_results.items():
        forward_gflops_per_joule = data.get("gflops_per_joule_forward",
                                          data["forward_flops_sum"] / data["forward_energy_sum"] / 1e9)
        backward_gflops_per_joule = data.get("gflops_per_joule_backward",
                                           data["backward_flops_sum"] / data["backward_energy_sum"] / 1e9)

        comparison_data.append({
            "key": key,
            "gpu": data.get("gpu", "Unknown"),
            "precision": data.get("precision", "Unknown"),
            "config": data.get("config", "baseline"),
            "type": data["type"],
            "batch_size": data.get("batch_size", 16),
            "effective_batch_size": data.get("effective_batch_size", data.get("batch_size", 16)),
            "input_length": data.get("input_length", 100),
            "forward_gflops_per_joule": forward_gflops_per_joule,
            "backward_gflops_per_joule": backward_gflops_per_joule,
            "total_gflops_per_joule": forward_gflops_per_joule + backward_gflops_per_joule,
            "forward_energy": data["forward_energy_mean"],
            "backward_energy": data["backward_energy_mean"],
            "forward_gpu_time": data["forward_gpu_time_mean"] / 1000,  # Convert to ms
            "backward_gpu_time": data["backward_gpu_time_mean"] / 1000,
        })

    df = pd.DataFrame(comparison_data)

    # Sort by total efficiency
    df = df.sort_values("total_gflops_per_joule", ascending=False)

    print("\n=== GPU EFFICIENCY COMPARISON ===")
    print(df[["key", "gpu", "precision", "config", "effective_batch_size",
             "input_length", "total_gflops_per_joule", "forward_gflops_per_joule",
             "backward_gflops_per_joule"]].to_string(index=False))

    # Calculate improvements
    if len(baseline_results) > 0:
        print("\n=== IMPROVEMENT ANALYSIS ===")

        # Find best baseline for each GPU/precision combination
        baseline_best = df[df["type"] == "baseline"].groupby(["gpu", "precision"]).first()

        for _, baseline in baseline_best.iterrows():
            gpu = baseline["gpu"]
            precision = baseline["precision"]

            print(f"\n{gpu} {precision} Baseline: {baseline['total_gflops_per_joule']:.2f} GFLOPs/J")

            # Find optimized versions
            optimized = df[(df["gpu"] == gpu) & (df["precision"] == precision) & (df["type"] != "baseline")]
            if len(optimized) > 0:
                best_optimized = optimized.iloc[0]
                improvement = best_optimized["total_gflops_per_joule"] / baseline["total_gflops_per_joule"]
                print(f"Best Optimized: {best_optimized['total_gflops_per_joule']:.2f} GFLOPs/J ({improvement:.1f}x improvement)")
                print(f"  Config: {best_optimized['config']}")
                print(f"  Batch size: {best_optimized['effective_batch_size']}")
                print(f"  Input length: {best_optimized['input_length']}")

    # Create visualization
    create_efficiency_plots(df)

    # Save detailed results
    df.to_csv("results/gpu_efficiency_comparison.csv", index=False)
    print(f"\nDetailed comparison saved to: results/gpu_efficiency_comparison.csv")

    return df

def create_efficiency_plots(df):
    """Create visualization plots for efficiency comparison"""

    # Plot 1: Total GFLOPs/J by configuration type
    fig1 = go.Figure()

    colors = {"baseline": "red", "optimized": "blue", "ultra_optimized": "green"}

    for config_type in df["type"].unique():
        subset = df[df["type"] == config_type]
        fig1.add_trace(go.Scatter(
            x=subset["effective_batch_size"],
            y=subset["total_gflops_per_joule"],
            mode='markers',
            name=config_type.replace("_", " ").title(),
            marker=dict(color=colors.get(config_type, "gray"), size=10),
            text=subset["key"],
            hovertemplate="<b>%{text}</b><br>" +
                         "Batch Size: %{x}<br>" +
                         "GFLOPs/J: %{y:.2f}<br>" +
                         "<extra></extra>"
        ))

    fig1.update_layout(
        title="GPU Efficiency: Total GFLOPs per Joule vs Effective Batch Size",
        xaxis_title="Effective Batch Size",
        yaxis_title="Total GFLOPs per Joule",
        hovermode='closest'
    )

    fig1.show()

    # Plot 2: Forward vs Backward efficiency
    fig2 = go.Figure()

    for config_type in df["type"].unique():
        subset = df[df["type"] == config_type]
        fig2.add_trace(go.Scatter(
            x=subset["forward_gflops_per_joule"],
            y=subset["backward_gflops_per_joule"],
            mode='markers',
            name=config_type.replace("_", " ").title(),
            marker=dict(color=colors.get(config_type, "gray"), size=10),
            text=subset["key"],
            hovertemplate="<b>%{text}</b><br>" +
                         "Forward: %{x:.2f} GFLOPs/J<br>" +
                         "Backward: %{y:.2f} GFLOPs/J<br>" +
                         "<extra></extra>"
        ))

    fig2.update_layout(
        title="Forward vs Backward Efficiency",
        xaxis_title="Forward GFLOPs per Joule",
        yaxis_title="Backward GFLOPs per Joule",
        hovermode='closest'
    )

    fig2.show()

    # Plot 3: Energy vs Time efficiency
    fig3 = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Forward Pass", "Backward Pass"),
        horizontal_spacing=0.1
    )

    for config_type in df["type"].unique():
        subset = df[df["type"] == config_type]

        fig3.add_trace(go.Scatter(
            x=subset["forward_gpu_time"],
            y=subset["forward_gflops_per_joule"],
            mode='markers',
            name=f"{config_type.replace('_', ' ').title()} (Forward)",
            marker=dict(color=colors.get(config_type, "gray"), size=8),
            text=subset["key"],
            showlegend=False
        ), row=1, col=1)

        fig3.add_trace(go.Scatter(
            x=subset["backward_gpu_time"],
            y=subset["backward_gflops_per_joule"],
            mode='markers',
            name=f"{config_type.replace('_', ' ').title()} (Backward)",
            marker=dict(color=colors.get(config_type, "gray"), size=8),
            text=subset["key"],
            showlegend=False
        ), row=1, col=2)

    fig3.update_xaxes(title_text="GPU Time (ms)", row=1, col=1)
    fig3.update_xaxes(title_text="GPU Time (ms)", row=1, col=2)
    fig3.update_yaxes(title_text="GFLOPs per Joule", row=1, col=1)
    fig3.update_yaxes(title_text="GFLOPs per Joule", row=1, col=2)

    fig3.update_layout(
        title="GPU Time vs Efficiency",
        height=400
    )

    fig3.show()

def print_optimization_recommendations():
    """Print specific recommendations for improving GPU utilization"""

    print("\n" + "="*60)
    print("GPU UTILIZATION OPTIMIZATION RECOMMENDATIONS")
    print("="*60)

    print("\n1. BATCH SIZE OPTIMIZATION:")
    print("   - Current: 16 (too small)")
    print("   - Recommended: 64-256 depending on GPU memory")
    print("   - Expected improvement: 2-4x GFLOPs/J")

    print("\n2. GRADIENT ACCUMULATION:")
    print("   - Use gradient accumulation to simulate larger batch sizes")
    print("   - Reduces memory usage while maintaining efficiency")
    print("   - Expected improvement: 1.5-2x GFLOPs/J")

    print("\n3. SEQUENCE LENGTH:")
    print("   - Current: 100 tokens (too short)")
    print("   - Recommended: 512-2048 tokens")
    print("   - Better GPU utilization for attention mechanisms")
    print("   - Expected improvement: 1.2-1.8x GFLOPs/J")

    print("\n4. PROFILING OVERHEAD REDUCTION:")
    print("   - Current: 5ms query interval (too frequent)")
    print("   - Recommended: 15-30ms query interval")
    print("   - Reduces measurement overhead")
    print("   - Expected improvement: 1.1-1.3x GFLOPs/J")

    print("\n5. MEMORY OPTIMIZATIONS:")
    print("   - Use torch.compile() for kernel fusion")
    print("   - Enable mixed precision training")
    print("   - Use memory-efficient attention")
    print("   - Expected improvement: 1.2-1.5x GFLOPs/J")

    print("\n6. DATA LOADING OPTIMIZATIONS:")
    print("   - Use more diverse token patterns")
    print("   - Pre-generate datasets to avoid tokenization overhead")
    print("   - Use pinned memory for faster transfers")
    print("   - Expected improvement: 1.1-1.2x GFLOPs/J")

    print("\nTOTAL EXPECTED IMPROVEMENT: 4-12x better GFLOPs/J")
    print("\nTo achieve these improvements:")
    print("1. Run: python profile_optimized.py")
    print("2. Run: python profile_ultra_optimized.py")
    print("3. Compare results with this analysis script")

if __name__ == "__main__":
    print("GPU Efficiency Analysis Tool")
    print("="*40)

    # Print recommendations first
    print_optimization_recommendations()

    # Run analysis
    try:
        df = analyze_gpu_efficiency()
        if df is not None and len(df) > 0:
            print("\nAnalysis completed successfully!")
        else:
            print("\nNo results to analyze. Please run the profiling scripts first.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please ensure you have run the profiling scripts and have result files in the results/ directory.")
