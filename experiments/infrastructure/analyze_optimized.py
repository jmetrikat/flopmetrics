import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import os

def load_results_from_jsonl(filepath):
    """Load results from JSONL file"""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def analyze_unified_results():
    """Comprehensive analysis of unified GPU efficiency results"""

    # Find all unified result files
    unified_files = glob.glob("results_optimized/*_llama3.2_1b_all_configs_results.jsonl")

    print(f"Found {len(unified_files)} result files")
    if not unified_files:
        print("No results found. Run: python profile_optimized.py")
        return

    # Load all results
    all_results = []
    for filepath in unified_files:
        filename = os.path.basename(filepath)
        # Extract GPU config and precision from filename (e.g., "a40_bf16_llama3.2_1b_all_configs_results.jsonl")
        # Remove the suffix to get "a40_bf16"
        base_name = filename.replace("_llama3.2_1b_all_configs_results.jsonl", "")
        # Split by last underscore to separate GPU config and precision
        parts = base_name.rsplit("_", 1)
        if len(parts) == 2:
            gpu_config, precision = parts
        else:
            gpu_config = base_name
            precision = "unknown"

        config_results = load_results_from_jsonl(filepath)
        for result in config_results:
            result["gpu_config"] = gpu_config
            result["precision"] = precision
            all_results.append(result)

    if not all_results:
        print("No results found in unified files.")
        return

    # Create comparison dataframe
    comparison_data = []
    for data in all_results:
        # Handle both forward-only and full training modes
        forward_gflops_per_joule = data.get("gflops_per_joule_forward", 0.0)
        backward_gflops_per_joule = data.get("gflops_per_joule_backward", 0.0)

        # Check if this is forward-only mode (backward metrics are 0)
        is_forward_only = backward_gflops_per_joule == 0.0

        # Calculate total efficiency
        if is_forward_only:
            total_gflops_per_joule = forward_gflops_per_joule
        else:
            total_gflops_per_joule = forward_gflops_per_joule + backward_gflops_per_joule

        comparison_data.append({
            "gpu_config": data.get("gpu_config", "Unknown"),
            "precision": data.get("precision", "unknown"),
            "config_name": data.get("config_name", "Unknown"),
            "batch_size": data.get("batch_size", 8),
            "input_length": data.get("input_length", 256),
            "forward_gflops_per_joule": forward_gflops_per_joule,
            "backward_gflops_per_joule": backward_gflops_per_joule,
            "total_gflops_per_joule": total_gflops_per_joule,
            "is_forward_only": is_forward_only,
            "forward_energy": data.get("forward_energy_mean", 0.0),
            "backward_energy": data.get("backward_energy_mean", 0.0),
            "forward_gpu_time": data.get("forward_gpu_time_mean", 0.0) / 1000,  # Convert to ms
            "backward_gpu_time": data.get("backward_gpu_time_mean", 0.0) / 1000,
        })

    df = pd.DataFrame(comparison_data)

    # Sort by total efficiency
    df = df.sort_values("total_gflops_per_joule", ascending=False)

    print("\n=== EFFICIENCY RESULTS ===")
    print(df[["gpu_config", "precision", "config_name", "batch_size", "input_length",
             "total_gflops_per_joule", "is_forward_only"]].to_string(index=False))

    # Show best per GPU and precision combination
    print("\n=== BEST CONFIGURATION PER GPU/PRECISION ===")
    for gpu in df["gpu_config"].unique():
        gpu_df = df[df["gpu_config"] == gpu]
        for precision in gpu_df["precision"].unique():
            precision_df = gpu_df[gpu_df["precision"] == precision].sort_values("total_gflops_per_joule", ascending=False)
            if len(precision_df) > 0:
                best = precision_df.iloc[0]
                mode = "Forward Only" if best['is_forward_only'] else "Full Training"
                print(f"{gpu.upper()}_{precision.upper()}: {best['config_name']} - {best['total_gflops_per_joule']:.1f} GFLOPs/J ({mode})")

    # Global best
    global_best = df.iloc[0]
    mode = "Forward Only" if global_best['is_forward_only'] else "Full Training"
    print(f"\n🏆 GLOBAL BEST:")
    print(f"{global_best['gpu_config'].upper()}_{global_best['precision'].upper()} {global_best['config_name']} - {global_best['total_gflops_per_joule']:.1f} GFLOPs/J ({mode})")

    # Create visualization
    create_unified_plots(df)

    # Save detailed results
    df.to_csv("results_optimized/efficiency_comparison.csv", index=False)
    print(f"\nResults saved to: results_optimized/efficiency_comparison.csv")

    return df

def create_unified_plots(df):
    """Create visualization plots for unified efficiency comparison"""

    # Plot 1: Efficiency by GPU and Configuration
    fig1 = go.Figure()

    colors = {"small": "blue", "mid": "green", "large": "purple"}

    for config_name in df["config_name"].unique():
        subset = df[df["config_name"] == config_name]
        fig1.add_trace(go.Scatter(
            x=subset["gpu_config"] + "_" + subset["precision"],
            y=subset["total_gflops_per_joule"],
            mode='markers',
            name=config_name.title(),
            marker=dict(color=colors.get(config_name, "gray"), size=12),
            text=subset["config_name"] + " (" + subset["precision"] + ")",
            hovertemplate="<b>%{text}</b><br>" +
                         "GPU: %{x}<br>" +
                         "GFLOPs/J: %{y:.2f}<br>" +
                         "<extra></extra>"
        ))

    fig1.update_layout(
        title="GPU Efficiency: Total GFLOPs per Joule by GPU and Configuration",
        xaxis_title="GPU Configuration",
        yaxis_title="Total GFLOPs per Joule",
        hovermode='closest',
        xaxis={'categoryorder': 'total descending'}
    )

    fig1.show()

    # Plot 2: Forward vs Backward efficiency
    fig2 = go.Figure()

    for config_name in df["config_name"].unique():
        subset = df[df["config_name"] == config_name]
        fig2.add_trace(go.Scatter(
            x=subset["forward_gflops_per_joule"],
            y=subset["backward_gflops_per_joule"],
            mode='markers',
            name=config_name.replace("_", " ").title(),
            marker=dict(color=colors.get(config_name, "gray"), size=10),
            text=subset["gpu_config"],
            hovertemplate="<b>%{text}</b><br>" +
                         "Forward: %{x:.2f} GFLOPs/J<br>" +
                         "Backward: %{y:.2f} GFLOPs/J<br>" +
                         "<extra></extra>"
        ))

    fig2.update_layout(
        title="Forward vs Backward Efficiency by Configuration",
        xaxis_title="Forward GFLOPs per Joule",
        yaxis_title="Backward GFLOPs per Joule",
        hovermode='closest'
    )

    fig2.show()

    # Plot 3: Batch size vs Efficiency
    fig3 = go.Figure()

    for config_name in df["config_name"].unique():
        subset = df[df["config_name"] == config_name]
        fig3.add_trace(go.Scatter(
            x=subset["batch_size"],
            y=subset["total_gflops_per_joule"],
            mode='markers',
            name=config_name.replace("_", " ").title(),
            marker=dict(color=colors.get(config_name, "gray"), size=10),
            text=subset["gpu_config"],
            hovertemplate="<b>%{text}</b><br>" +
                         "Batch Size: %{x}<br>" +
                         "GFLOPs/J: %{y:.2f}<br>" +
                         "<extra></extra>"
        ))

    fig3.update_layout(
        title="Batch Size vs Total Efficiency",
        xaxis_title="Batch Size",
        yaxis_title="Total GFLOPs per Joule",
        hovermode='closest'
    )

    fig3.show()

def print_optimization_summary():
    """Print essential optimization information"""
    print("\n=== GPU OPTIMIZATION CONFIGURATIONS ===")
    print("Small:  8 batch, 256 tokens")
    print("Mid:    8 batch, 512 tokens")
    print("Large:  8 batch, 1024 tokens")
    print("\nAll configs use: 100 samples, 10 warmup, 10ms query interval")
    print("Default precision: BF16")
    print("Supports: --forward-only flag for inference benchmarking")

if __name__ == "__main__":
    print("GPU Efficiency Analysis")
    print_optimization_summary()

    try:
        df = analyze_unified_results()
        if df is not None and len(df) > 0:
            print("\n✅ Analysis complete!")
        else:
            print("\n❌ No results found. Run: python profile_optimized.py")
    except Exception as e:
        print(f"❌ Error: {e}")
