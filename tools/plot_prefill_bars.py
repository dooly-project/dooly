import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory


import argparse

arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("--results-dir", type=Path, required=True)
arg_parse.add_argument("--workload", type=str, default="decode", help="workload name for output file naming")
arg_parse.add_argument("--split-per-model", action="store_true")
arg_parse.add_argument("--no-title", action="store_true")
args = arg_parse.parse_args()
workload = args.workload

# HERE = Path(__file__).resolve().parent
HERE = args.results_dir
# SUMMARY = HERE / f"{workload}_validation_summary.json"
SUMMARY = HERE / f"validation_summary.json"
OUT_STEM        = HERE / f"{workload}_validation_bars"
LEGEND_OUT_STEM = HERE / f"{workload}_validation_bars_legend"

LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
LENGTH_LABELS = ["512", "1K", "2K", "4K", "8K", "16K"]

BACKENDS = [
    ("FLASHINFER",  "#f4a8a8"),  # pastel red
    ("FLASH_ATTN",  "#a8c8f0"),  # pastel blue
    ("TRITON_ATTN", "#a8e0a8"),  # pastel green
]

MODEL_MAP = {
    "CohereLabs/c4ai-command-r7b-12-2024": "Command-R7B",
    "meta-llama/Llama-3.1-8B": "Llama-3.1-8B",
}

GAP_BETWEEN_MODELS = 0.3  # x-units of empty space between the two model sections

rows = json.loads(SUMMARY.read_text())
rows = [row for row in rows if row["workload"] == workload]
all_models = sorted({r["model"] for r in rows}, reverse=True)  # Llama first, then Cohere

n_groups = len(LENGTHS)
n_bars   = len(BACKENDS) * 2
group_w  = 0.85
bar_w    = group_w / n_bars
section_stride = n_groups + GAP_BETWEEN_MODELS

# Figure dimensions for the combined (all-models) figure. When --split-per-model
# is used, the per-model figure width is scaled by the fraction of x-range that
# one model takes inside the combined figure, so each bar keeps the same
# visual width whether plotted standalone or alongside other models.
COMBINED_FIG_W = 14
COMBINED_FIG_H = 3.0
EDGE_MARGIN_LEFT  = 0.1
EDGE_MARGIN_RIGHT = 0.1


def _xrange_units(n_models):
    """Total x-axis range (in data units) for `n_models` side-by-side."""
    last_x = (n_models - 1) * section_stride + n_groups - 1
    return last_x + group_w + EDGE_MARGIN_LEFT + EDGE_MARGIN_RIGHT


def render(models_to_plot, out_stem, ylim_top=None):
    """Render the bar chart for the given models and save to <out_stem>.{png,pdf}.

    ``ylim_top``: when given, force the y-axis upper bound to this value so
    multiple per-model figures share the y-scale of the combined figure. When
    None, autoscale and apply the workload-specific headroom multiplier.
    """
    width_scale = _xrange_units(len(models_to_plot)) / _xrange_units(len(all_models))
    fig, ax = plt.subplots(1, 1, figsize=(COMBINED_FIG_W * width_scale, COMBINED_FIG_H))

    all_xticks  = []
    all_xlabels = []
    separators  = []

    for j, model in enumerate(models_to_plot):
        base = j * section_stride
        x = np.arange(n_groups) + base

        for i, (backend, color) in enumerate(BACKENDS):
            pred = [next((r["predicted_ms"] for r in rows
                          if r["model"] == model and r["backend"] == backend
                          and r["length"] == L), np.nan)
                    for L in LENGTHS]
            meas = [next((r["measured_ms"] for r in rows
                          if r["model"] == model and r["backend"] == backend
                          and r["length"] == L), np.nan)
                    for L in LENGTHS]
            pred_off = (2 * i     - n_bars / 2 + 0.5) * bar_w
            meas_off = (2 * i + 1 - n_bars / 2 + 0.5) * bar_w
            ax.bar(x + pred_off, pred, bar_w, color=color,
                   edgecolor="black", hatch="///")
            ax.bar(x + meas_off, meas, bar_w, color=color,
                   edgecolor="black")
            for k, (xi, p, m) in enumerate(zip(x, pred, meas)):
                if not (np.isfinite(p) and np.isfinite(m) and m > 0):
                    continue
                gap = (p - m) / m * 100
                side_annot = ((model == "meta-llama/Llama-3.1-8B" or model == "CohereLabs/c4ai-command-r7b-12-2024")
                              and backend == "TRITON_ATTN"
                              and k == n_groups - 1 and workload == "prefill")
                if side_annot:
                    if model == "meta-llama/Llama-3.1-8B":
                        ax.text(xi + meas_off + bar_w / 2 - 1.0, max(p, m),
                                f"{gap:.1f}%",
                                ha="left", va="top", fontsize=16, rotation=0)
                    elif model == "CohereLabs/c4ai-command-r7b-12-2024":
                        ax.text(xi - 0.05, max(p, m) + 300,
                                f"{gap:.1f}%",
                                ha="left", va="top", fontsize=16, rotation=0)
                else:
                    ax.text((xi + pred_off + xi + meas_off) / 2,
                            max(p, m), f" {gap:.1f}%",
                            ha="center", va="bottom", fontsize=16, rotation=90)

        all_xticks.extend(x.tolist())
        all_xlabels.extend(LENGTH_LABELS)
        if j < len(models_to_plot) - 1:
            separators.append(base + n_groups - 0.5 + GAP_BETWEEN_MODELS / 2)

    # Vertical separator(s) between model sections
    for sx in separators:
        ax.axvline(sx, color="gray", linestyle="--", linewidth=1)

    # Model name section labels, placed just above the axes box
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    if not args.no_title:
        for j, model in enumerate(models_to_plot):
            center = j * section_stride + (n_groups - 1) / 2
            ax.text(center, 1.02, MODEL_MAP.get(model, model),
                    ha="center", va="bottom", transform=trans,
                    fontsize=20, fontweight="bold")

    ax.set_xticks(all_xticks)
    ax.set_xticklabels(all_xlabels, fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_xlabel("Token Count", fontsize=20)
    # ax.set_ylabel("Latency [s]", fontsize=20)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y/1000:g}"))
    ax.grid(axis="y", alpha=0.3)

    # Headroom for the rotated annotations on the tallest bars
    if ylim_top is not None:
        ax.set_ylim(top=ylim_top)
    else:
        ax.set_ylim(top=ax.get_ylim()[1] * (1.0 if workload == "prefill" else 1.75))

    # Trim left/right whitespace: hug the outer bars with a small margin
    EDGE_MARGIN_LEFT  = 0.1
    EDGE_MARGIN_RIGHT = 0.1   # extra room for the side-placed TRITON_ATTN annotation
    last_x = (len(models_to_plot) - 1) * section_stride + n_groups - 1
    ax.set_xlim(-group_w / 2 - EDGE_MARGIN_LEFT, last_x + group_w / 2 + EDGE_MARGIN_RIGHT)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = out_stem.parent / f"{out_stem.name}.{ext}"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"saved: {out}")
    plt.close(fig)


if args.split_per_model:
    # Compute the y-limit the combined figure would have used, so each
    # per-model figure shares the same y-scale.
    data_max = max(
        v for r in rows
        for v in (r.get("predicted_ms"), r.get("measured_ms"))
        if isinstance(v, (int, float))
    )
    # Match matplotlib's default 5% autoscale margin, then apply the
    # workload-specific headroom multiplier.
    shared_ylim = data_max * 1.05 * (1.0 if workload == "prefill" else 1.35)
    for model in all_models:
        safe = MODEL_MAP.get(model, model).replace("/", "_").replace(" ", "_")
        out_stem = HERE / f"{workload}_validation_bars_{safe}"
        render([model], out_stem, ylim_top=shared_ylim)
else:
    render(all_models, OUT_STEM)

# --- Standalone legend figure ----------------------------------------------
legend_handles = [Patch(facecolor=color, edgecolor="black", label=backend)
                  for backend, color in BACKENDS]
legend_handles += [
    Patch(facecolor="white", edgecolor="black", label="measured"),
    Patch(facecolor="white", edgecolor="black", hatch="///", label="predicted"),
]
legend_fig, legend_ax = plt.subplots(figsize=(10, 0.6))
legend_ax.axis("off")
legend_ax.legend(handles=legend_handles, loc="center", ncol=5,
                 fontsize=20, frameon=False)
for ext in ("png", "pdf"):
    out = LEGEND_OUT_STEM.parent / f"{LEGEND_OUT_STEM.name}.{ext}"
    legend_fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved: {out}")
