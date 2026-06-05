import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import os

def load_json(filepath: str) -> dict:
    """Load JSON results file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_cdf(data: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute CDF from data points.

    Returns:
        (x_values, y_values) where y_values are cumulative probabilities
    """
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, y


def extract_metric(data: dict, metric: str) -> List[float]:
    """Extract metric values from results JSON.

    Args:
        data: Loaded JSON data
        metric: One of 'ttft', 'itl', 'latency', 'tpot'

    Returns:
        List of metric values (in milliseconds)
    """
    if metric == 'ttft':
        # TTFTs are already in list format
        values = data.get('ttfts', [])
        # Convert to ms if needed
        if values and values[0] < 1:  # Likely in seconds
            values = [v * 1000 for v in values]
        return values[5:] # Skip first 5 tokens to avoid startup overhead 

    elif metric == 'itl':
        # ITLs are list of lists (one list per request)
        itls = data.get('itls', [])
        # Get the first 5
        itls = itls[5:]
        # Flatten all ITLs into single list
        all_itls = []
        for itl_list in itls:
            if itl_list:  # Skip empty lists
                all_itls.extend(itl_list)
        # Convert to ms if needed
        if all_itls and all_itls[0] < 1:
            all_itls = [v * 1000 for v in all_itls]
        return all_itls

    elif metric == 'latency' or metric == 'e2e':
        # End-to-end latency
        # Check if we have per-request latencies
        if 'latencies' in data:
            values = data['latencies']
        else:
            # Compute from start_times if available
            # For now, return empty
            return []
        if values and values[0] < 1:
            values = [v * 1000 for v in values]
        return values

    elif metric == 'tpot':
        # Time per output token
        # Similar to ITL but different calculation
        return []

    else:
        raise ValueError(f"Unknown metric: {metric}")


def plot_cdf_comparison(
    dooly_file: str,
    vllm_file: str,
    metric: str,
    output_file: str = None,
    title: str = None,
    style: str = 'smooth',
):
    """Plot CDF comparison between DoolySim and vLLM.

    Args:
        dooly_file: Path to DoolySim JSON results
        vllm_file: Path to vLLM JSON results
        metric: Metric to plot ('ttft', 'itl', 'latency')
        output_file: Optional output file path (if None, display interactively)
        title: Optional custom title
        style: 'smooth' (default), 'stepped', or 'markers'
    """
    # Load data
    dooly_data = load_json(dooly_file)
    vllm_data = load_json(vllm_file)

    # Extract metric values
    dooly_values = extract_metric(dooly_data, metric)
    vllm_values = extract_metric(vllm_data, metric)

    if not dooly_values:
        print(f"WARNING: No {metric} data found in DoolySim results")
    if not vllm_values:
        print(f"WARNING: No {metric} data found in vLLM results")

    if not dooly_values and not vllm_values:
        print("ERROR: No data to plot!")
        return

    # Use publication-quality style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create figure with compact aspect ratio (more square, less wide)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Colorblind-friendly palette
    colors = {'dooly': '#0173b2', 'vllm': '#de8f05'}

    if dooly_values:
        x_dooly, y_dooly = compute_cdf(dooly_values)

        if style == 'smooth':
            ax.plot(x_dooly, y_dooly, label='DoolySim', linewidth=3,
                   color=colors['dooly'], alpha=0.85)
        elif style == 'stepped':
            ax.step(x_dooly, y_dooly, label='DoolySim', linewidth=2.5,
                   color=colors['dooly'], alpha=0.85, where='post')
        else:  # markers
            ax.plot(x_dooly, y_dooly, label='DoolySim', linewidth=2,
                   color=colors['dooly'], marker='o', markersize=5,
                   markevery=max(1, len(x_dooly)//15), alpha=0.85,
                   markerfacecolor='white', markeredgewidth=2)

    if vllm_values:
        x_vllm, y_vllm = compute_cdf(vllm_values)

        if style == 'smooth':
            ax.plot(x_vllm, y_vllm, label='vLLM', linewidth=3,
                   color=colors['vllm'], alpha=0.85)
        elif style == 'stepped':
            ax.step(x_vllm, y_vllm, label='vLLM', linewidth=2.5,
                   color=colors['vllm'], alpha=0.85, where='post')
        else:  # markers
            ax.plot(x_vllm, y_vllm, label='vLLM', linewidth=2,
                   color=colors['vllm'], marker='s', markersize=5,
                   markevery=max(1, len(x_vllm)//15), alpha=0.85,
                   markerfacecolor='white', markeredgewidth=2)

    # Formatting
    metric_names = {
        'ttft': 'Time to First Token (TTFT)',
        'itl': 'Inter-Token Latency (ITL)',
        'latency': 'End-to-End Latency',
        'e2e': 'End-to-End Latency',
        'tpot': 'Time Per Output Token (TPOT)',
    }

    metric_name = metric_names.get(metric, metric.upper())

    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    else:
        ax.set_title(f'{metric_name} - CDF Comparison', fontsize=16, fontweight='bold', pad=20)

    ax.set_xlabel(f'{metric_name} (ms)', fontsize=14, fontweight='semibold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='semibold')

    # Enhanced grid
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Set y-axis to percentage first
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

    # Add percentile reference lines
    for percentile in [0.5, 0.95, 0.99]:
        ax.axhline(y=percentile, color='gray', linestyle=':',
                  linewidth=1, alpha=0.4, zorder=0)
        # Put labels on the right side
        ax.text(ax.get_xlim()[1], percentile, f'  P{int(percentile*100)}',
               fontsize=9, va='center', ha='left', color='gray', alpha=0.7)

    # Compute statistics and update legend labels
    if dooly_values and vllm_values:
        dooly_p50 = np.percentile(dooly_values, 50)
        dooly_p95 = np.percentile(dooly_values, 95)
        vllm_p50 = np.percentile(vllm_values, 50)
        vllm_p95 = np.percentile(vllm_values, 95)

        # Update line labels to include statistics
        for line in ax.get_lines():
            if line.get_label() == 'DoolySim':
                line.set_label(f'DoolySim (P50: {dooly_p50:.1f}ms, P95: {dooly_p95:.1f}ms)')
            elif line.get_label() == 'vLLM':
                line.set_label(f'vLLM (P50: {vllm_p50:.1f}ms, P95: {vllm_p95:.1f}ms)')

    # Compact legend in lower right with integrated statistics
    legend = ax.legend(fontsize=10, loc='lower right', framealpha=0.95,
                      edgecolor='gray', fancybox=True, shadow=True)
    legend.get_frame().set_linewidth(1.5)
    legend.set_zorder(100)

    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {output_file}")
    else:
        plt.show()

def calculate_stats(dooly_file, vllm_file):
    # calculate p50, p95, p99 for both dooly and vllm and print them out
    dooly_data = load_json(dooly_file)
    vllm_data = load_json(vllm_file)
    
    dooly_ttft_values = extract_metric(dooly_data, 'ttft')
    dooly_itl_values = extract_metric(dooly_data, 'itl')
    vllm_ttft_values = extract_metric(vllm_data, 'ttft')
    vllm_itl_values = extract_metric(vllm_data, 'itl')
    
    dooly_values={
        'ttft': dooly_ttft_values,
        'itl': dooly_itl_values,
        'duration': dooly_data.get('duration', 0),
        'output_throughput': dooly_data.get('output_throughput', 0)
    }
    
    vllm_values={
        'ttft': vllm_ttft_values,
        'itl': vllm_itl_values,
        'duration': vllm_data.get('duration', 0),
        'output_throughput': vllm_data.get('output_throughput', 0)
    }
    
    summary_values ={}
    for metric in ['ttft', 'itl', 'duration', 'output_throughput']:
        summary_values[metric] = {}
        for model in ['dooly', 'vllm']:
            if model == 'dooly':
                source = dooly_values
            else:
                source = vllm_values

            if metric in ['ttft', 'itl']:
                summary_values[metric][model] = {}
                # Check if the metric has any data before calculating percentiles
                if source[metric] and len(source[metric]) > 0:
                    summary_values[metric][model]['p50'] = np.percentile(source[metric], 50)
                    summary_values[metric][model]['p95'] = np.percentile(source[metric], 95)
                    summary_values[metric][model]['p99'] = np.percentile(source[metric], 99)
                else:
                    # Handle empty data case
                    summary_values[metric][model]['p50'] = 0
                    summary_values[metric][model]['p95'] = 0
                    summary_values[metric][model]['p99'] = 0
            else:
                summary_values[metric][model] = source[metric]

        if metric in ['ttft', 'itl']:
            # Only calculate percentage differences if both models have data
            if (summary_values[metric]['vllm']['p50'] > 0 and
                summary_values[metric]['dooly']['p50'] > 0):
                summary_values[metric]['p50_perc_diff'] = (summary_values[metric]['dooly']['p50'] - summary_values[metric]['vllm']['p50'])/summary_values[metric]['vllm']['p50'] * 100
                summary_values[metric]['p95_perc_diff'] = (summary_values[metric]['dooly']['p95'] - summary_values[metric]['vllm']['p95'])/summary_values[metric]['vllm']['p95'] * 100
                summary_values[metric]['p99_perc_diff'] = (summary_values[metric]['dooly']['p99'] - summary_values[metric]['vllm']['p99'])/summary_values[metric]['vllm']['p99'] * 100
            else:
                # Set to 0 if no data to compare
                summary_values[metric]['p50_perc_diff'] = 0
                summary_values[metric]['p95_perc_diff'] = 0
                summary_values[metric]['p99_perc_diff'] = 0
        else:
            # Handle division by zero for other metrics
            if summary_values[metric]['vllm'] > 0:
                summary_values[metric]['perc_diff'] = (summary_values[metric]['dooly'] - summary_values[metric]['vllm'])/summary_values[metric]['vllm'] * 100
            else:
                summary_values[metric]['perc_diff'] = 0
    return summary_values

def plot_stats(stats: dict, output_dir: str = "."):
    """Plot summary statistics (bar charts) comparing DoolySim and vLLM."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = {'dooly': '#0173b2', 'vllm': '#de8f05'}
    
    # --- 1. TTFT Percentiles (Grouped Bar Chart) ---
    ax = axes[0,0]  # Top Left
    labels = ['P50', 'P95', 'P99']
    x = np.arange(len(labels))
    width = 0.35
    
    dooly_ttft = [stats['ttft']['dooly']['p50'], stats['ttft']['dooly']['p95'], stats['ttft']['dooly']['p99']]
    vllm_ttft = [stats['ttft']['vllm']['p50'], stats['ttft']['vllm']['p95'], stats['ttft']['vllm']['p99']]
    
    ax.bar(x - width/2, dooly_ttft, width, label='DoolySim', color=colors['dooly'])
    ax.bar(x + width/2, vllm_ttft, width, label='vLLM', color=colors['vllm'])
    ax.set_ylabel('Latency (ms)', fontweight='semibold')
    ax.set_title('TTFT Percentiles', fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='semibold')
    ax.legend()

    # --- 2. ITL Percentiles (Grouped Bar Chart) ---
    ax = axes[0,1]  # Top Right
    dooly_itl = [stats['itl']['dooly']['p50'], stats['itl']['dooly']['p95'], stats['itl']['dooly']['p99']]
    vllm_itl = [stats['itl']['vllm']['p50'], stats['itl']['vllm']['p95'], stats['itl']['vllm']['p99']]
    
    ax.bar(x - width/2, dooly_itl, width, label='DoolySim', color=colors['dooly'])
    ax.bar(x + width/2, vllm_itl, width, label='vLLM', color=colors['vllm'])
    ax.set_ylabel('Latency (ms)', fontweight='semibold')
    ax.set_title('ITL Percentiles', fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='semibold')
    ax.legend()

    # --- 3. Total Duration (Standard Bar Chart) ---
    ax = axes[1,0]  # Bottom Left
    models = ['DoolySim', 'vLLM']
    durations = [stats['duration']['dooly'], stats['duration']['vllm']]
    bars = ax.bar(models, durations, color=[colors['dooly'], colors['vllm']])
    ax.set_ylabel('Time (seconds)', fontweight='semibold')
    ax.set_title('Total Generation Duration', fontsize=14, fontweight='bold', pad=10)
    
    # --- 4. Output Throughput (Standard Bar Chart) ---
    ax = axes[1,1]  # Bottom Right
    throughputs = [stats['output_throughput']['dooly'], stats['output_throughput']['vllm']]
    bars = ax.bar(models, throughputs, color=[colors['dooly'], colors['vllm']])
    ax.set_ylabel('Tokens / second', fontweight='semibold')
    ax.set_title('Output Throughput', fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout(pad=3.0)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'summary_stats.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved summary statistics plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Plot CDF comparison between DoolySim and vLLM results"
    )

    parser.add_argument("--dooly", type=str, required=True,
                       help="Path to DoolySim JSON results")
    parser.add_argument("--vllm", type=str, required=True,
                       help="Path to vLLM JSON results")
    parser.add_argument("--metric", type=str, required=True,
                       choices=['ttft', 'itl', 'latency', 'e2e', 'tpot'],
                       help="Metric to plot")
    parser.add_argument("--output", "-o", type=str,
                       help="Output file path (e.g., comparison.png)")
    parser.add_argument("--title", type=str,
                       help="Custom plot title")
    parser.add_argument("--style", type=str, default='smooth',
                       choices=['smooth', 'stepped', 'markers'],
                       help="Plot style: smooth (default), stepped, or markers")

    args = parser.parse_args()

    plot_cdf_comparison(
        dooly_file=args.dooly,
        vllm_file=args.vllm,
        metric=args.metric,
        output_file=args.output,
        title=args.title,
        style=args.style,
    )
    
    stats = calculate_stats(args.dooly, args.vllm)

    import json
    json_output = json.dumps(stats, indent=4)
    print(json_output)

    plot_stats(stats, output_dir=os.path.dirname(args.output) if args.output else ".")

if __name__ == "__main__":
    main()
