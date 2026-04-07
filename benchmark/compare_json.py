"""Compare DoolySim and vLLM results from JSON files.

This script compares two JSON files in vLLM benchmark format and produces
detailed accuracy metrics.
"""

import argparse
import json
import numpy as np
from typing import Dict, List, Any


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def calculate_percentiles(values: List[float], percentiles: List[int] = [50, 75, 90, 95, 99]) -> Dict[str, float]:
    """Calculate percentiles for a list of values."""
    if not values:
        return {f"p{p}": 0.0 for p in percentiles}

    arr = np.array(values)
    result = {}
    for p in percentiles:
        result[f"p{p}"] = float(np.percentile(arr, p))
    return result


def calculate_mape(predicted: List[float], actual: List[float]) -> float:
    """Calculate Mean Absolute Percentage Error."""
    if len(predicted) != len(actual) or len(actual) == 0:
        return 0.0

    pred = np.array(predicted)
    act = np.array(actual)

    # Avoid division by zero
    mask = act != 0
    if not mask.any():
        return 0.0

    return float(np.mean(np.abs((pred[mask] - act[mask]) / act[mask])) * 100)


def compare_metrics(dooly_json: Dict, vllm_json: Dict) -> Dict[str, Any]:
    """Compare metrics between DoolySim and vLLM results.

    Returns:
        Dictionary with comparison results
    """
    results = {}

    # ========================================================================
    # Aggregate Metrics
    # ========================================================================

    results["duration"] = {
        "dooly": dooly_json["duration"],
        "vllm": vllm_json["duration"],
        "diff": dooly_json["duration"] - vllm_json["duration"],
        "diff_pct": ((dooly_json["duration"] - vllm_json["duration"]) / vllm_json["duration"] * 100) if vllm_json["duration"] > 0 else 0,
    }

    results["output_throughput"] = {
        "dooly": dooly_json["output_throughput"],
        "vllm": vllm_json["output_throughput"],
        "diff": dooly_json["output_throughput"] - vllm_json["output_throughput"],
        "diff_pct": ((dooly_json["output_throughput"] - vllm_json["output_throughput"]) / vllm_json["output_throughput"] * 100) if vllm_json["output_throughput"] > 0 else 0,
    }

    results["total_token_throughput"] = {
        "dooly": dooly_json["total_token_throughput"],
        "vllm": vllm_json["total_token_throughput"],
        "diff": dooly_json["total_token_throughput"] - vllm_json["total_token_throughput"],
        "diff_pct": ((dooly_json["total_token_throughput"] - vllm_json["total_token_throughput"]) / vllm_json["total_token_throughput"] * 100) if vllm_json["total_token_throughput"] > 0 else 0,
    }

    # ========================================================================
    # TTFT Metrics
    # ========================================================================

    dooly_ttfts = [t * 1000 for t in dooly_json["ttfts"]]  # Convert to ms
    vllm_ttfts = [t * 1000 for t in vllm_json["ttfts"]]    # Convert to ms

    dooly_ttft_percentiles = calculate_percentiles(dooly_ttfts)
    vllm_ttft_percentiles = calculate_percentiles(vllm_ttfts)

    results["ttft"] = {
        "avg": {
            "dooly": np.mean(dooly_ttfts),
            "vllm": np.mean(vllm_ttfts),
            "diff": np.mean(dooly_ttfts) - np.mean(vllm_ttfts),
            "diff_pct": ((np.mean(dooly_ttfts) - np.mean(vllm_ttfts)) / np.mean(vllm_ttfts) * 100) if np.mean(vllm_ttfts) > 0 else 0,
        },
        "mape": calculate_mape(dooly_ttfts, vllm_ttfts),
    }

    for p in [50, 75, 90, 95, 99]:
        key = f"p{p}"
        dooly_val = dooly_ttft_percentiles[key]
        vllm_val = vllm_ttft_percentiles[key]
        results["ttft"][key] = {
            "dooly": dooly_val,
            "vllm": vllm_val,
            "diff": dooly_val - vllm_val,
            "diff_pct": ((dooly_val - vllm_val) / vllm_val * 100) if vllm_val > 0 else 0,
        }

    # ========================================================================
    # ITL Metrics
    # ========================================================================

    # Flatten ITL arrays and convert to ms
    dooly_itls_flat = [itl * 1000 for request_itls in dooly_json["itls"] for itl in request_itls]
    vllm_itls_flat = [itl * 1000 for request_itls in vllm_json["itls"] for itl in request_itls]

    dooly_itl_percentiles = calculate_percentiles(dooly_itls_flat)
    vllm_itl_percentiles = calculate_percentiles(vllm_itls_flat)

    results["itl"] = {
        "avg": {
            "dooly": np.mean(dooly_itls_flat) if dooly_itls_flat else 0,
            "vllm": np.mean(vllm_itls_flat) if vllm_itls_flat else 0,
            "diff": (np.mean(dooly_itls_flat) if dooly_itls_flat else 0) - (np.mean(vllm_itls_flat) if vllm_itls_flat else 0),
            "diff_pct": (((np.mean(dooly_itls_flat) if dooly_itls_flat else 0) - (np.mean(vllm_itls_flat) if vllm_itls_flat else 0)) / (np.mean(vllm_itls_flat) if vllm_itls_flat else 1) * 100) if vllm_itls_flat and np.mean(vllm_itls_flat) > 0 else 0,
        },
        "mape": calculate_mape(dooly_itls_flat, vllm_itls_flat) if dooly_itls_flat and vllm_itls_flat else 0,
    }

    for p in [50, 75, 90, 95, 99]:
        key = f"p{p}"
        dooly_val = dooly_itl_percentiles[key]
        vllm_val = vllm_itl_percentiles[key]
        results["itl"][key] = {
            "dooly": dooly_val,
            "vllm": vllm_val,
            "diff": dooly_val - vllm_val,
            "diff_pct": ((dooly_val - vllm_val) / vllm_val * 100) if vllm_val > 0 else 0,
        }

    return results


def print_comparison_report(results: Dict[str, Any]) -> None:
    """Print formatted comparison report."""

    print()
    print("=" * 80)
    print("DoolySim vs vLLM Comparison Report")
    print("=" * 80)
    print()

    # Duration
    print("Duration (seconds)")
    print(f"  DoolySim:  {results['duration']['dooly']:>10.3f}s")
    print(f"  vLLM:      {results['duration']['vllm']:>10.3f}s")
    print(f"  Diff:      {results['duration']['diff']:>10.3f}s ({results['duration']['diff_pct']:+.1f}%)")
    print()

    # Output Throughput
    print("Output Throughput (tokens/sec)")
    print(f"  DoolySim:  {results['output_throughput']['dooly']:>10.2f}")
    print(f"  vLLM:      {results['output_throughput']['vllm']:>10.2f}")
    print(f"  Diff:      {results['output_throughput']['diff']:>10.2f} ({results['output_throughput']['diff_pct']:+.1f}%)")
    print()

    # Total Token Throughput
    print("Total Token Throughput (tokens/sec)")
    print(f"  DoolySim:  {results['total_token_throughput']['dooly']:>10.2f}")
    print(f"  vLLM:      {results['total_token_throughput']['vllm']:>10.2f}")
    print(f"  Diff:      {results['total_token_throughput']['diff']:>10.2f} ({results['total_token_throughput']['diff_pct']:+.1f}%)")
    print()

    # TTFT
    print("-" * 80)
    print("Time to First Token (TTFT) in milliseconds")
    print("-" * 80)
    print(f"{'Metric':<10} {'DoolySim':>12} {'vLLM':>12} {'Diff':>12} {'Diff %':>12} {'MAPE':>12}")
    print("-" * 80)

    ttft = results["ttft"]
    print(f"{'Avg':<10} {ttft['avg']['dooly']:>12.2f} {ttft['avg']['vllm']:>12.2f} "
          f"{ttft['avg']['diff']:>12.2f} {ttft['avg']['diff_pct']:>11.1f}% {ttft['mape']:>11.2f}%")

    for p in [50, 75, 90, 95, 99]:
        key = f"p{p}"
        metric = ttft[key]
        print(f"{key.upper():<10} {metric['dooly']:>12.2f} {metric['vllm']:>12.2f} "
              f"{metric['diff']:>12.2f} {metric['diff_pct']:>11.1f}%")

    print()

    # ITL
    print("-" * 80)
    print("Inter-Token Latency (ITL) in milliseconds")
    print("-" * 80)
    print(f"{'Metric':<10} {'DoolySim':>12} {'vLLM':>12} {'Diff':>12} {'Diff %':>12} {'MAPE':>12}")
    print("-" * 80)

    itl = results["itl"]
    print(f"{'Avg':<10} {itl['avg']['dooly']:>12.2f} {itl['avg']['vllm']:>12.2f} "
          f"{itl['avg']['diff']:>12.2f} {itl['avg']['diff_pct']:>11.1f}% {itl['mape']:>11.2f}%")

    for p in [50, 75, 90, 95, 99]:
        key = f"p{p}"
        metric = itl[key]
        print(f"{key.upper():<10} {metric['dooly']:>12.2f} {metric['vllm']:>12.2f} "
              f"{metric['diff']:>12.2f} {metric['diff_pct']:>11.1f}%")

    print()
    print("=" * 80)
    print()

    # Summary
    print("Summary:")
    print(f"  TTFT MAPE: {ttft['mape']:.2f}%")
    print(f"  ITL MAPE:  {itl['mape']:.2f}%")
    print()

    # Accuracy assessment
    if ttft['mape'] < 10 and itl['mape'] < 10:
        print("  ✓ Excellent accuracy (MAPE < 10%)")
    elif ttft['mape'] < 20 and itl['mape'] < 20:
        print("  ✓ Good accuracy (MAPE < 20%)")
    elif ttft['mape'] < 50 and itl['mape'] < 50:
        print("  ⚠ Moderate accuracy (MAPE < 50%)")
    else:
        print("  ✗ Poor accuracy (MAPE >= 50%)")
    print()


def export_comparison_json(results: Dict[str, Any], output_path: str) -> None:
    """Export comparison results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Exported comparison results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare DoolySim and vLLM JSON results"
    )

    parser.add_argument("--dooly", type=str, required=True,
                       help="Path to DoolySim JSON file")
    parser.add_argument("--vllm", type=str, required=True,
                       help="Path to vLLM JSON file")
    parser.add_argument("--output", type=str,
                       help="Path to save comparison JSON (optional)")

    args = parser.parse_args()

    # Load JSON files
    print(f"Loading DoolySim results from: {args.dooly}")
    dooly_json = load_json(args.dooly)

    print(f"Loading vLLM results from: {args.vllm}")
    vllm_json = load_json(args.vllm)

    print(f"Loaded {dooly_json['num_prompts']} DoolySim requests")
    print(f"Loaded {vllm_json['num_prompts']} vLLM requests")

    # Compare
    results = compare_metrics(dooly_json, vllm_json)

    # Print report
    print_comparison_report(results)

    # Export if requested
    if args.output:
        export_comparison_json(results, args.output)


if __name__ == "__main__":
    main()
