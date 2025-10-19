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
        # Extract GPU config from filename (e.g., "a40_fp16_llama3.2_1b_all_configs_results.jsonl")
        gpu_config = filename.replace("_llama3.2_1b_all_configs_results.jsonl", "")

        config_results = load_results_from_jsonl(filepath)
        for result in config_results:
            result["gpu_config"] = gpu_config
            all_results.append(result)

    if not all_results:
        print("No results found in unified files.")
        return

    # Create comparison dataframe
    comparison_data = []
    for data in all_results:
        forward_gflops_per_joule = data.get("gflops_per_joule_forward",
                                          data["forward_flops_sum"] / data["forward_energy_sum"] / 1e9)
        backward_gflops_per_joule = data.get("gflops_per_joule_backward",
                                           data["backward_flops_sum"] / data["backward_energy_sum"] / 1e9)

        comparison_data.append({
            "gpu_config": data.get("gpu_config", "Unknown"),
            "config_name": data.get("config_name", "Unknown"),
            "description": data.get("description", ""),
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

    print("\n=== EFFICIENCY RESULTS ===")
    print(df[["gpu_config", "config_name", "batch_size", "input_length",
             "total_gflops_per_joule"]].to_string(index=False))

    # Show best per GPU
    print("\n=== BEST CONFIGURATION PER GPU ===")
    for gpu in df["gpu_config"].unique():
        gpu_df = df[df["gpu_config"] == gpu].sort_values("total_gflops_per_joule", ascending=False)
        best = gpu_df.iloc[0]
        print(f"{gpu.upper()}: {best['config_name']} - {best['total_gflops_per_joule']:.1f} GFLOPs/J")

    # Global best
    global_best = df.iloc[0]
    print(f"\n🏆 GLOBAL BEST:")
    print(f"{global_best['gpu_config'].upper()} {global_best['config_name']} - {global_best['total_gflops_per_joule']:.1f} GFLOPs/J")

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

    colors = {"baseline": "red", "small_batch": "blue", "medium_batch": "green", "large_batch": "purple"}

    for config_name in df["config_name"].unique():
        subset = df[df["config_name"] == config_name]
        fig1.add_trace(go.Scatter(
            x=subset["gpu_config"],
            y=subset["total_gflops_per_joule"],
            mode='markers',
            name=config_name.replace("_", " ").title(),
            marker=dict(color=colors.get(config_name, "gray"), size=12),
            text=subset["description"],
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
    print("Small batch:  32 batch, 512 tokens")
    print("Medium batch: 64 batch, 1024 tokens")
    print("Large batch:  128 batch, 2048 tokens")
    print("\nAll configs use: 100 samples, 10 warmup, 10ms query interval")

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
