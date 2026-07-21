"""Grouped bar chart comparing DoolySim vs vLLM end-to-end throughput (or duration).

One group per model, two bars per group (vLLM + DoolySim), normalized so the
vLLM bar is 1.0. A percentage-error tag (DoolySim vs vLLM) sits on top of the
DoolySim bar so it's easy to spot simulator accuracy per model.

Usage:
    python -m doolysim.tools.plot_throughput_bars \\
        --configs comparison/openchat-0.5rps/Qwen/Qwen3-4B \\
                  comparison/openchat-0.5rps/meta-llama/Llama-3.1-8B \\
                  comparison/openchat-0.5rps/Qwen/Qwen3-32B \\
                  comparison/openchat-0.5rps/meta-llama/Llama-3.1-70B \\
        --labels "Qwen3-4B" "Llama-3.1-8B" "Qwen3-32B" "Llama-3.1-70B" \\
        --metric request_throughput \\
        --output ./comparison/throughput.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from doolysim.tools.plot_comparison import load_json
from doolysim.tools.plot_configs_cdf import (
    CONFIG_COLORS, SOURCE_LABELS, find_result_file, read_config_model_name,
)


# Scalar-per-run metrics and their display names / units.
SCALAR_METRICS = {
    "request_throughput": ("Request Throughput", "req/s"),
    "output_throughput":  ("Output Throughput",  "tok/s"),
    "total_throughput":   ("Total Throughput",   "tok/s"),
    "duration":           ("Run Duration",       "s"),
}

# Metrics where "lower is better" (so error sign is inverted for interpretation).
LOWER_IS_BETTER = {"duration"}


def _load_scalar(filepath: str, metric: str) -> Optional[float]:
    data = load_json(filepath)
    if metric not in data:
        return None
    try:
        return float(data[metric])
    except (TypeError, ValueError):
        return None


def _gather(
    config_dirs: List[str],
    metric: str,
    use_batch: bool,
) -> Dict[str, Dict[str, float]]:
    """Return {config_dir: {'dooly': v, 'vllm': v}} with missing values omitted."""
    out: Dict[str, Dict[str, float]] = {}
    for cfg in config_dirs:
        per_source: Dict[str, float] = {}
        for source in ("vllm", "dooly"):
            path = find_result_file(cfg, source, use_batch)
            if path is None:
                print(f"WARNING: no {source} file in {cfg}")
                continue
            val = _load_scalar(path, metric)
            if val is None:
                print(f"WARNING: no '{metric}' field in {path}")
                continue
            per_source[source] = val
        if per_source:
            out[cfg] = per_source
    return out


def plot_throughput_bars(
    config_dirs: List[str],
    metric: str,
    output_file: Optional[str],
    title: Optional[str],
    use_batch: bool,
    normalize: bool,
    config_labels: Optional[Dict[str, str]] = None,
) -> None:
    if metric not in SCALAR_METRICS:
        raise SystemExit(
            f"Unknown metric '{metric}'. Choose from: {sorted(SCALAR_METRICS)}"
        )

    config_labels = config_labels or {}
    plt.style.use("seaborn-v0_8-whitegrid")

    data = _gather(config_dirs, metric, use_batch)
    kept_cfgs = [c for c in config_dirs if c in data]
    if not kept_cfgs:
        raise SystemExit("No usable data found across configs.")

    metric_label, unit = SCALAR_METRICS[metric]
    n = len(kept_cfgs)

    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * n + 2.0), 4.5))

    bar_width = 0.36
    x = np.arange(n, dtype=float)
    vllm_vals, dooly_vals = [], []
    plot_vllm, plot_dooly = [], []
    labels: List[str] = []

    for i, cfg in enumerate(kept_cfgs):
        per = data[cfg]
        v_vllm = per.get("vllm")
        v_dooly = per.get("dooly")
        vllm_vals.append(v_vllm)
        dooly_vals.append(v_dooly)

        # Normalize by vLLM value (per-model) so the vLLM bar is 1.0.
        denom = v_vllm if (normalize and v_vllm not in (None, 0)) else 1.0
        plot_vllm.append(v_vllm / denom if v_vllm is not None else 0.0)
        plot_dooly.append(v_dooly / denom if v_dooly is not None else 0.0)

        labels.append(
            config_labels.get(cfg)
            or read_config_model_name(cfg, use_batch)
            or os.path.basename(cfg.rstrip("/"))
        )

    colors = [CONFIG_COLORS[i % len(CONFIG_COLORS)] for i in range(n)]

    bars_vllm = ax.bar(
        x - bar_width / 2, plot_vllm, bar_width,
        color=colors, edgecolor="black", linewidth=1.0, alpha=0.95,
        label=SOURCE_LABELS["vllm"],
    )
    bars_dooly = ax.bar(
        x + bar_width / 2, plot_dooly, bar_width,
        color=colors, edgecolor="black", linewidth=1.0, alpha=0.95,
        hatch="//", label=SOURCE_LABELS["dooly"],
    )

    # Error tag above each DoolySim bar: (dooly - vllm) / vllm * 100%.
    for i, bar in enumerate(bars_dooly):
        v, d = vllm_vals[i], dooly_vals[i]
        if v in (None, 0) or d is None:
            continue
        err_pct = (d - v) / v * 100.0
        if metric in LOWER_IS_BETTER:
            err_pct = -err_pct  # invert so positive = sim underestimates goodness
        y_top = max(bar.get_height(), bars_vllm[i].get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2, y_top + 0.02,
            f"{err_pct:+.1f}%",
            ha="center", va="bottom",
            fontsize=12, fontweight="semibold",
            color="#b00020" if abs(err_pct) >= 5 else "#1a1a1a",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    y_label = (
        f"Normalized {metric_label}  (vLLM = 1.0)"
        if normalize else f"{metric_label} ({unit})"
    )
    ax.set_ylabel(y_label, fontsize=14, fontweight="semibold")
    if title:
        ax.set_title(title, fontsize=14, fontweight="semibold")
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=1)
    ax.set_axisbelow(True)
    ax.margins(y=0.18)

    if normalize:
        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5, alpha=0.5, zorder=0)

    # ax.legend(loc="upper right", fontsize=12, frameon=True)

    # Print summary.
    print(f"\n{metric} summary:")
    print(f"  {'config':<40s} {'vLLM':>12s} {'DoolySim':>12s} {'err %':>9s}")
    for cfg, label in zip(kept_cfgs, labels):
        per = data[cfg]
        v = per.get("vllm")
        d = per.get("dooly")
        if v is None or d is None:
            err_str = "--"
        else:
            err = (d - v) / v * 100.0 if v else 0.0
            if metric in LOWER_IS_BETTER:
                err = -err
            err_str = f"{err:+.2f}"
        v_str = f"{v:.4g}" if v is not None else "--"
        d_str = f"{d:.4g}" if d is not None else "--"
        print(f"  {label:<40s} {v_str:>12s} {d_str:>12s} {err_str:>9s}")

    plt.tight_layout()
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.05)
        print(f"\n✓ Saved plot to {output_file}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grouped bar chart of DoolySim vs vLLM end-to-end scalars."
    )
    parser.add_argument("input_dir", nargs="?",
                        help="Base dir; entries in --configs without '/' resolve here.")
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Config dirs (per-model). Same resolution as plot_configs_cdf.")
    parser.add_argument("--labels", nargs="+",
                        help="Optional per-config labels; must match --configs count.")
    parser.add_argument("--metric", default="request_throughput",
                        choices=sorted(SCALAR_METRICS.keys()))
    parser.add_argument("--normalize", dest="normalize", action="store_true", default=True,
                        help="Normalize bars by vLLM value per model (default on).")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false",
                        help="Plot raw values with real units instead of normalized.")
    parser.add_argument("--batch", action="store_true",
                        help="Pick up *_batch_*.json instead of streaming files.")
    parser.add_argument("--output", "-o")
    parser.add_argument("--title")
    args = parser.parse_args()

    def _is_pathlike(entry: str) -> bool:
        return os.path.isabs(entry) or ("/" in entry)

    resolved: List[str] = []
    for entry in args.configs:
        if _is_pathlike(entry):
            resolved.append(os.path.abspath(entry))
        else:
            if not args.input_dir:
                parser.error(f"'{entry}' is not a path; provide input_dir to resolve it")
            resolved.append(os.path.abspath(os.path.join(args.input_dir, entry)))

    missing = [p for p in resolved if not os.path.isdir(p)]
    if missing:
        parser.error(f"config dirs not found: {missing}")

    labels_map: Dict[str, str] = {}
    if args.labels:
        if len(args.labels) != len(resolved):
            parser.error(
                f"--labels count ({len(args.labels)}) must match --configs count ({len(resolved)})"
            )
        labels_map = dict(zip(resolved, args.labels))

    plot_throughput_bars(
        config_dirs=resolved,
        metric=args.metric,
        output_file=args.output,
        title=args.title,
        use_batch=args.batch,
        normalize=args.normalize,
        config_labels=labels_map,
    )


if __name__ == "__main__":
    main()
