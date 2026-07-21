#!/usr/bin/env python3
"""Chess-style heatmap of validate_predictions.py results.

For each (workload, model) panel, rows are the *decision-point* axis
(prefill length for prefill-heavy workloads; batch size for decode-heavy),
columns are the 6 (source × attention-backend) options:

    vLLM measured                        Dooly predicted
    ┌──────┬──────┬──────┐  ┃  ┌──────┬──────┬──────┐
    │  FA  │  FI  │  TRI │  ┃  │  FA  │  FI  │  TRI │
    └──────┴──────┴──────┘     └──────┴──────┴──────┘

Each cell is colored by % slower than the best (fastest) cell in its row.
A red ★ marks the row winner; arrows mark "winner inversions" — places
where the fastest configuration changes between adjacent rows.

Data source: a `validation_summary.json` produced by
`doolysim.tools.validate_predictions`. Each row has fields:
    model, backend, tp, length, predicted_ms, measured_ms, ...

Decode-heavy data is not yet present in the summary file (validate_predictions
only runs single-prefill batches at this point); the `--workload` flag still
lets you select between {prefill, decode, both}. Decode panels are emitted
empty (with a notice in the title) until a future run populates them.

Usage
-----
    python -m doolysim.tools.plot_validation_chess \
        --summary /home/jkim38/dooly/results/scenario_a_validation_locked_1700_v2_dense/validation_summary.json \
        --output-dir /home/jkim38/dooly/results/scenario_a_validation_locked_1700_v2_dense \
        --workload prefill                # or 'decode' or 'both' (default: prefill)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch


# Canonical short-name mapping for the column header (matches the original chess plot).
_BACKEND_SHORT = {
    "FLASHINFER":  "FI",
    "FLASH_ATTN":  "FA",
    "TRITON_ATTN": "TRI",
}
_BACKEND_ORDER = ["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"]   # left → right within each half

_SOURCE_LABEL = {"measured": "vLLM measured", "predicted": "Dooly predicted"}

# Aliases the user might type on --model-order that aren't substrings of the
# display-name. Map: alias substring (lowercased) → display name in the data.
_MODEL_ALIASES: dict[str, str] = {
    "cohere":  "Command-R7B",
    "command": "Command-R7B",
    "c4ai":    "Command-R7B",
    "r7b":     "Command-R7B",
    "llama":   "Llama-3.1-8B",
    "8b":      "Llama-3.1-8B",
}

_Y_AXIS_MAPPING ={
    512: "512",
    1024: "1K",
    2048: "2K",
    4096: "4K",
    8192: "8K",
    16384: "16K",
}


def model_short(name: str) -> str:
    n = name.split("/")[-1]
    if "c4ai" in n:
        n = "Command-R7B"
    return n


# ────────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────────

def detect_workload(row: dict) -> str:
    """Best-effort workload classification from a validation_summary row.

    Priority:
      1. Explicit `workload` field if present.
      2. Heuristic: `length` and `out_length` keys → prefill if length >= out_length.
      3. Default fallback: 'prefill' (validate_predictions currently emits prefill-only
         single-request configs).
    """
    if "workload" in row and row["workload"] in ("prefill", "decode"):
        return row["workload"]
    in_len = row.get("length") or row.get("input_length")
    out_len = row.get("output_length") or row.get("out_length")
    if isinstance(in_len, int) and isinstance(out_len, int):
        return "prefill" if in_len >= out_len else "decode"
    return "prefill"


def detect_axis_value(row: dict, workload: str) -> int | None:
    """Return the row-axis value for this workload (length for prefill,
    batch size for decode)."""
    if workload == "prefill":
        return row.get("length")
    if workload == "decode":
        # validate_predictions doesn't yet write decode rows, but if it does we
        # expect a `batch_size` field. Fall back to `length` so the panel
        # builds something even from prefill-shaped rows.
        return row.get("batch_size") or row.get("length")
    return None


def load_summary(summary_path: Path) -> list[dict]:
    with open(summary_path) as f:
        return json.load(f)


def organize(rows: list[dict], wanted_workloads: set[str]) -> dict:
    """Reshape summary rows into the structure expected by the chess plotter:

        {(workload, model_short): {axis_value: {(source, backend): latency_ms}}}

    Only rows whose detected workload is in `wanted_workloads` are kept.
    """
    out: dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        wl = detect_workload(r)
        if wl not in wanted_workloads:
            continue
        ax = detect_axis_value(r, wl)
        if ax is None:
            continue
        ms = model_short(r["model"])
        backend = r["backend"]
        if "measured_ms" in r:
            out[(wl, ms)][ax][("measured",  backend)] = r["measured_ms"]
        if "predicted_ms" in r:
            out[(wl, ms)][ax][("predicted", backend)] = r["predicted_ms"]
    return out


# ────────────────────────────────────────────────────────────────────────────────
# Matrix construction
# ────────────────────────────────────────────────────────────────────────────────

def build_matrix(panel_data: dict, axis_values: list[int],
                 sources: list[str]) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Per-row gap-from-best matrix for one panel.

    Columns are `sources` outer, backends inner — e.g. for sources=["predicted"]
    you get [(predicted, FA), (predicted, FI), (predicted, TRI)]; for both
    sources you get a 6-column matrix with measured-FA…measured-TRI on the
    left, predicted-FA…predicted-TRI on the right.

    Each row is normalized by its own minimum so cell value = % slower than
    the row's best cell. NaN rows (no data) stay NaN.
    """
    cols = [(s, b) for s in sources for b in _BACKEND_ORDER]
    n_rows = len(axis_values)
    n_cols = len(cols)
    raw = np.full((n_rows, n_cols), np.nan)

    for i, ax in enumerate(axis_values):
        per = panel_data.get(ax, {})
        for j, (s, b) in enumerate(cols):
            if (s, b) in per:
                raw[i, j] = per[(s, b)]

    # Normalize each row by its min (% slower than best in row).
    norm = np.full_like(raw, np.nan)
    for i in range(n_rows):
        row = raw[i, :]
        valid = ~np.isnan(row)
        if np.any(valid):
            best = np.nanmin(row)
            if best > 0:
                norm[i, :] = (row - best) / best * 100.0

    return norm, cols


def build_combined_matrix(per_model_panels: dict[str, dict],
                          axis_values: list[int],
                          sources: list[str],
                          model_order: list[str]
                          ) -> tuple[np.ndarray, list[tuple[str, str, str]]]:
    """Combined matrix where each row is normalized across *all* models.

    Columns are model outer × source inner × backend innermost — e.g. for
    model_order=['Command-R7B','Llama-3.1-8B'] and sources=['predicted'] you
    get 6 cols: [(Command-R7B,predicted,FA), (Command-R7B,predicted,FI),
    (Command-R7B,predicted,TRI), (Llama-3.1-8B,predicted,FA), …, (Llama-3.1-8B,predicted,TRI)].

    The per-row min is taken across ALL columns, so the ★ winner identifies
    the fastest (model × source × backend) combination at that L — and
    inversions across rows show when the best-overall configuration migrates
    between models.
    """
    cols = [(m, s, b) for m in model_order for s in sources for b in _BACKEND_ORDER]
    n_rows = len(axis_values)
    n_cols = len(cols)
    raw = np.full((n_rows, n_cols), np.nan)

    for i, ax in enumerate(axis_values):
        for j, (m, s, b) in enumerate(cols):
            per = per_model_panels.get(m, {}).get(ax, {})
            if (s, b) in per:
                raw[i, j] = per[(s, b)]

    norm = np.full_like(raw, np.nan)
    for i in range(n_rows):
        row = raw[i, :]
        valid = ~np.isnan(row)
        if np.any(valid):
            best = np.nanmin(row)
            if best > 0:
                norm[i, :] = (row - best) / best * 100.0

    return norm, cols


# ────────────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────────────

def _format_axis_label(workload: str, v: int) -> str:
    """Pretty label for the row axis (length → '4K', batch size → '4')."""
    if workload == "prefill":
        if v >= 1024 and v % 1024 == 0:
            return f"{v // 1024}K"
        return str(v)
    return str(v)


def plot_chess(data: dict, axis_values_by_workload: dict, output_dir: Path,
               workloads: list[str], sources: list[str], vmax: float = 45.0,
               tie_threshold: float = 0.5,
               combine_models: bool = True,
               model_order: list[str] | None = None,
               write_standalone_legend: bool = True,
               no_figure_legend: bool = True,
               source: str = "measured",) -> None:
    """Draw the chess heatmap.

    With ``combine_models=True`` (default), all models share a single panel
    per workload — columns become model × source × backend, and the ★ winner
    identifies the fastest (model, source, backend) combination at each row.
    Inversion arrows then show *cross-model* migrations of the optimum
    (which is the SWA crossover story).

    With ``combine_models=False`` it falls back to the original layout: one
    panel per (workload, model).
    """
    cmap = LinearSegmentedColormap.from_list("gap", [
        "#f5f5f2", "#d4e4d4", "#8fc09a", "#4d9466", "#2a6e44", "#0f3d22",
    ])
    BEST_COLOR = "#d62728"
    PATH_COLOR = "#d62728"

    # Discover models present in the data, applying the requested order.
    models_in_data = sorted({m for (_, m) in data.keys()})
    if model_order is None:
        # Default: Llama first (full-attention baseline), then Cohere/Command,
        # then the rest alphabetically. Tells the SWA-crossover story
        # left-to-right.
        def _key(m: str) -> tuple[int, str]:
            if "Llama" in m or "llama" in m:
                return (0, m)
            if "Command" in m or "Cohere" in m:
                return (1, m)
            return (2, m)
        model_order = sorted(models_in_data, key=_key)
    else:
        # User-supplied order. Match each requested name against models in
        # the data either exactly or as a substring (so 'Llama' matches
        # 'Llama-3.1-8B', 'Command' matches 'Command-R7B', etc.).
        resolved: list[str] = []
        used: set[str] = set()
        for requested in model_order:
            req_lower = requested.lower()
            # Resolve via alias table first: if the requested string contains
            # any alias key, prefer the alias's mapped display name.
            alias_target: str | None = None
            for alias_sub, target in _MODEL_ALIASES.items():
                if alias_sub in req_lower:
                    alias_target = target
                    break
            match = next(
                (m for m in models_in_data
                 if m not in used
                 and (m == requested
                      or req_lower in m.lower()
                      or m == alias_target)),
                None,
            )
            if match is not None:
                resolved.append(match)
                used.add(match)
            else:
                print(f"[plot_chess] warning: --model-order entry "
                      f"{requested!r} did not match any model in the data; "
                      f"skipping. Available: {sorted(models_in_data)}")
        # Append any unrequested models at the end so they aren't dropped.
        for m in sorted(models_in_data):
            if m not in used:
                resolved.append(m)
        model_order = resolved

    workload_priority = {wl: i for i, wl in enumerate(workloads)}
    if combine_models:
        # One panel per workload that has at least one model.
        panels: list = sorted(
            {wl for (wl, _) in data.keys()},
            key=lambda wl: workload_priority.get(wl, 99),
        )
    else:
        panels = sorted(
            data.keys(),
            key=lambda k: (workload_priority.get(k[0], 99), k[1])
        )

    if not panels:
        print("[plot_chess] no panels to draw — empty data dict; aborting.")
        return

    # Figure size scales with row count so dense panels stay readable.
    max_axis = max(
        (len(axis_values_by_workload.get(p if combine_models else p[0], []))
         for p in panels),
        default=8,
    )
    fig_h = max(5.5, 0.42 * max_axis + 2.5)
    n_panels = len(panels)
    n_cols_per_panel = (
        len(model_order) * len(sources) * len(_BACKEND_ORDER)
        if combine_models else len(sources) * len(_BACKEND_ORDER)
    )
    panel_w = max(3.6, 0.55 * n_cols_per_panel + 1.0)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(panel_w * n_panels + 1.0, fig_h),
        sharey=False,
    )
    if n_panels == 1:
        axes = [axes]

    fig.subplots_adjust(wspace=0.18, top=0.84, bottom=0.10, left=0.07, right=0.97)

    # Per-panel rendering. With combine_models=True, each panel is just the
    # workload (rows = axis values, cols = model × source × backend). With
    # combine_models=False, each panel is (workload, model) as before.
    for idx, panel_key in enumerate(panels):
        ax = axes[idx]

        if combine_models:
            workload = panel_key
            axis_values = axis_values_by_workload.get(workload, [])
            # axis_values=[]
            if not axis_values:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"no {workload} data\n(in summary)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=20, color="#888888")
                ax.set_title(f"({workload})", fontsize=20, pad=8)
                continue
            per_model_panels = {
                m: data[(workload, m)] for m in model_order
                if (workload, m) in data
            }
            mat, cols = build_combined_matrix(
                per_model_panels, axis_values, sources, model_order
            )
        else:
            workload, model = panel_key
            axis_values = axis_values_by_workload.get(workload, [])
            if not axis_values:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"no {workload} data\n(in summary)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=20, color="#888888")
                ax.set_title(f"{model}\n({workload})", fontsize=20, pad=8)
                continue

            mat, cols = build_matrix(data[(workload, model)], axis_values, sources)
        n_rows, n_cols = mat.shape

        ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax,
                  aspect="auto", interpolation="nearest")

        # Winner stars + cells
        winner_cols: list[int] = []
        for r in range(n_rows):
            row = mat[r]
            valid = ~np.isnan(row)
            if not np.any(valid):
                winner_cols.append(0)
                continue
            row_min = np.nanmin(row)
            best_cols = np.where(np.abs(row - row_min) <= tie_threshold)[0]
            is_unique = (len(best_cols) == 1)
            winner_cols.append(int(best_cols[0]))

            for c in range(n_cols):
                v = mat[r, c]
                if np.isnan(v):
                    continue
                is_best = (np.abs(v - row_min) <= tie_threshold)
                if is_best:
                    rect = FancyBboxPatch(
                        (c - 0.44, r - 0.44), 0.88, 0.88,
                        boxstyle="round,pad=0.02", facecolor="none",
                        edgecolor=BEST_COLOR, linewidth=2.2, zorder=5,
                    )
                    ax.add_patch(rect)
                    ax.text(c, r, "★", ha="center", va="center",
                            fontsize=18, fontweight="bold",
                            color=BEST_COLOR, zorder=6)
                else:
                    text_color = "white" if v > 22 else "#3a3a3a"
                    ax.text(c, r, f"{int(round(v))}%",
                            ha="center", va="center",
                            fontsize=18, color=text_color, alpha=0.7)

        # Winner-inversion arrows between adjacent rows
        for r in range(len(winner_cols) - 1):
            c1, c2 = winner_cols[r], winner_cols[r + 1]
            if c1 != c2:
                ax.annotate(
                    "", xy=(c2, r + 1), xytext=(c1, r),
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.25,head_length=0.15",
                        color=PATH_COLOR, linewidth=2.0,
                        connectionstyle="arc3,rad=0.15",
                        shrinkA=6, shrinkB=6,
                    ),
                    zorder=8,
                )

        # ----- Grid + dividers -----
        for rr in range(n_rows + 1):
            ax.axhline(rr - 0.5, color="white", linewidth=0.8)
        for cc in range(n_cols + 1):
            ax.axvline(cc - 0.5, color="white", linewidth=0.8)

        # Backend group is the inner-most axis. With multiple sources OR
        # multiple models combined, draw thicker dividers at every group
        # boundary (every `len(_BACKEND_ORDER)` columns, except trailing).
        group_w = len(_BACKEND_ORDER)
        if combine_models:
            # Heaviest divider between models, lighter between sources.
            for c_boundary in range(group_w, n_cols, group_w):
                # Is this a model boundary or just a source boundary?
                model_boundary = (c_boundary % (group_w * len(sources))) == 0
                lw = 3.0 if model_boundary else 2.0
                ax.axvline(c_boundary - 0.5, color="white", linewidth=lw)
        elif len(sources) > 1:
            for s_idx in range(1, len(sources)):
                ax.axvline(s_idx * group_w - 0.5, color="white", linewidth=2.6)

        # ----- Column ticks: backend short names (always inner-most axis) -----
        ax.set_xticks(range(n_cols))
        if combine_models:
            ax.set_xticklabels(
                [_BACKEND_SHORT[c[2]] for c in cols], fontsize=22
            )
        else:
            ax.set_xticklabels(
                [_BACKEND_SHORT[c[1]] for c in cols], fontsize=22
            )
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(
            [_Y_AXIS_MAPPING.get(v, _format_axis_label(workload, v))
             for v in axis_values],
            fontsize=22, 
        )
        ax.tick_params(length=0)

        # ----- Title / panel header -----
        # if combine_models:
        #     ax.set_title(
        #         f"Predicted across all models" if sources == ["predicted"]
        #         else f"All models" if len(sources) > 1
        #         else f"Measured across all models",
        #         fontsize=14, pad=8,
        #     )
        # else:
        #     ax.set_title(model, fontsize=14, pad=8)

        # ----- Group labels above the panel -----
        # combine_models: label each model group (and within it, each source group).
        # !combine_models: label each source half if more than one source.
        if combine_models:
            cols_per_model = group_w * len(sources)
            for m_idx, m in enumerate(model_order):
                if not any((workload, m) == k for k in data.keys()):
                    continue
                center_col = m_idx * cols_per_model + (cols_per_model - 1) / 2
                ax.text(center_col, -0.55, m, ha="center", va="bottom",
                        fontsize=18, fontweight="bold", clip_on=False)
                # If we also have multiple sources, label them under the model.
                if len(sources) > 1:
                    for s_idx, src in enumerate(sources):
                        sub_center = m_idx * cols_per_model + s_idx * group_w + (group_w - 1) / 2
                        ax.text(sub_center, -0.9, _SOURCE_LABEL[src],
                                ha="center", va="bottom",
                                fontsize=10, color="#444444",
                                clip_on=False)
        elif len(sources) > 1:
            for s_idx, src in enumerate(sources):
                center_col = s_idx * group_w + (group_w - 1) / 2
                ax.text(center_col, -0.95, _SOURCE_LABEL[src],
                        ha="center", va="bottom", fontsize=18,
                        fontweight="bold", clip_on=False)

        # Y-label only on left-most panel.
        if source == "measured":
            ax.set_ylabel(
                "Total Token Count",
                fontsize=20, labelpad=1, fontweight="bold"
            )

    # Workload-section labels above the panels (centered across each
    # workload's group of panels).
    by_workload: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(panels):
        wl = p if combine_models else p[0]
        by_workload[wl].append(i)
    fig_w_per_panel = 1.0 / n_panels
    for wl, idxs in by_workload.items():
        if not idxs:
            continue
        center = (min(idxs) + max(idxs) + 1) / 2 / n_panels   # in figure x-fraction
        # Map "fig fraction" through the actual subplot extents using transAxes.
        # Easier: write the label above each panel center in figure coords
        # via fig.text; the subplot positions are nearly uniform.
        x_fig = (sum(idxs) / len(idxs) + 0.5) / n_panels
        x_fig = 0.06 + x_fig * (0.96 - 0.06)
        # fig.text(x_fig, 0.9,
        #          "Prefill Heavy" if wl == "prefill" else "Decode Heavy",
        #          ha="center", va="bottom", fontsize=16, fontweight="bold")

    # Vertical light divider between workload groups in the figure (if both).
    workload_groups = list(by_workload.keys())
    if len(workload_groups) > 1:
        # Place divider between the last panel of group 0 and first panel of group 1.
        last_of_first = max(by_workload[workload_groups[0]])
        x_div = (last_of_first + 1) / n_panels
        x_div = 0.06 + x_div * (0.96 - 0.06)
        fig.add_artist(plt.Line2D(
            [x_div, x_div], [0.10, 0.85],
            transform=fig.transFigure, color="#bbbbbb",
            linewidth=1.0, linestyle=":",
        ))

    # Legend handles (used both for the embedded figure-legend and the
    # standalone legend file if requested).
    legend_elements = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
                   markeredgecolor=BEST_COLOR, markeredgewidth=2,
                   markersize=11, label="Best (row winner)"),
        plt.Line2D([0], [0], color=PATH_COLOR, linewidth=2.2,
                   marker=">", markersize=6, label="Winner inversion"),
    ]
    if not no_figure_legend:
        fig.legend(
            handles=legend_elements, loc="lower center",
            bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=20,
            frameon=False, fancybox=False, edgecolor="#cccccc",
            handlelength=2.2, columnspacing=1.5,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = output_dir / f"h100_{workloads[0]}_{source}.pdf"
    # out_png = output_dir / "inversion_heatmap_v2.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    # fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    # print(f"Saved {out_png}")

    # Standalone legend — separate, axis-less figure that can sit alongside
    # other figures in a paper / slide deck without copying the heatmap legend.
    if write_standalone_legend:
        save_standalone_legend(legend_elements, output_dir)


def save_standalone_legend(legend_elements: list, output_dir: Path,
                           filename_stem: str = "inversion_heatmap_legend",
                           ncol: int | None = None,
                           fontsize: int = 12) -> None:
    """Save just the legend block as its own PDF + PNG, no axes, no data."""
    if ncol is None:
        ncol = len(legend_elements)
    fig_legend = plt.figure(figsize=(0.9 * len(legend_elements) + 1.6, 0.8))
    fig_legend.legend(
        handles=legend_elements, loc="center", ncol=ncol,
        fontsize=20, frameon=False, fancybox=False,
        edgecolor="#cccccc", handlelength=2.2, columnspacing=1.5,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = output_dir / f"{filename_stem}.pdf"
    out_png = output_dir / f"{filename_stem}.png"
    fig_legend.savefig(out_pdf, bbox_inches="tight", dpi=300, transparent=True)
    fig_legend.savefig(out_png, bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig_legend)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

def _parse_workload(s: str) -> list[str]:
    if s == "both":
        return ["prefill", "decode"]
    if s in ("prefill", "decode"):
        return [s]
    raise argparse.ArgumentTypeError(f"--workload must be prefill, decode, or both (got {s!r})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Chess-style heatmap of vLLM-vs-Dooly across (model, backend, length).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--summary", "-s", required=True, type=Path,
                    help="validate_predictions.py validation_summary.json")
    ap.add_argument("--output-dir", "-o", default=None, type=Path,
                    help="Where to save the heatmap PNG/PDF. Defaults to "
                         "<summary's parent directory>.")
    ap.add_argument("--workload", "-w", type=_parse_workload, default=["prefill"],
                    help="Which workload(s) to plot: 'prefill', 'decode', or "
                         "'both'. (default: prefill)")
    ap.add_argument("--source", choices=["predicted", "measured", "both"],
                    default="predicted",
                    help="Which latency source to render. 'predicted' = "
                         "Dooly only (default), 'measured' = vLLM only, "
                         "'both' = side-by-side halves.")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[512, 1024, 2048, 4096, 8192, 16384],
                    help="Subset of prefill lengths to plot as rows. Default "
                         "is the doubling sweep 512, 1K, 2K, 4K, 8K, 16K. "
                         "Pass --lengths with no extra args (or 'all') to use "
                         "every length in the summary.")
    ap.add_argument("--decode-batch-sizes", type=int, nargs="+", default=None,
                    help="Subset of batch sizes to plot as rows for decode "
                         "panels (default: all unique batch sizes seen).")
    ap.add_argument("--separate-models", action="store_true",
                    help="One panel per (workload, model). By default, all "
                         "models share a single per-workload panel so the "
                         "winner-inversion arrow can migrate ACROSS models.")
    ap.add_argument("--model-order", "-M", nargs="+", default=None,
                    help="Explicit left-to-right model order. Each entry "
                         "matches a model name in the summary either exactly "
                         "(e.g. 'meta-llama/Llama-3.1-8B') or as a "
                         "case-insensitive substring (e.g. 'Llama' or "
                         "'Cohere'). Models not listed are appended at the "
                         "end. Default: Llama → Cohere → others.")
    ap.add_argument("--vmax", type=float, default=45.0,
                    help="Color scale upper bound (% slower than best). "
                         "(default: 45)")
    ap.add_argument("--tie-threshold", type=float, default=0.5,
                    help="Cells within this many %% of best are tied (each "
                         "gets a star). (default: 0.5)")
    ap.add_argument("--no-standalone-legend", action="store_true",
                    help="Don't write the separate inversion_heatmap_legend "
                         "PDF/PNG. (Default: it IS written, so you can place "
                         "the legend once next to multiple figures.)")
    ap.add_argument("--no-figure-legend", action="store_true",
                    help="Strip the legend from the heatmap figure itself "
                         "(useful when you'll place the standalone legend "
                         "alongside it in your paper / slide).")
    args = ap.parse_args()

    rows = load_summary(args.summary)
    out_dir = args.output_dir or args.summary.parent

    wanted = set(args.workload)
    organized = organize(rows, wanted)

    if not organized:
        print(f"[plot_validation_chess] no rows match workloads={args.workload}; "
              f"summary contains "
              f"{sorted({detect_workload(r) for r in rows})}.")
        return 1

    # Per-workload axis values: union of axes seen in that workload's rows,
    # sorted ascending. Apply user overrides if provided.
    axis_by_workload: dict[str, list[int]] = {}
    for wl in args.workload:
        seen = sorted({
            ax for (w, _), per_ax in organized.items() if w == wl
            for ax in per_ax
        })
        if wl == "prefill" and args.lengths is not None:
            seen = [v for v in args.lengths if v in seen]
        if wl == "decode" and args.decode_batch_sizes is not None:
            seen = [v for v in args.decode_batch_sizes if v in seen]
        axis_by_workload[wl] = seen

    # Drop panels whose axis ended up empty.
    organized = {
        k: v for k, v in organized.items()
        if axis_by_workload.get(k[0])
    }

    sources = (["measured", "predicted"] if args.source == "both"
               else [args.source])

    plot_chess(organized, axis_by_workload, out_dir,
               workloads=args.workload, sources=sources,
               vmax=args.vmax, tie_threshold=args.tie_threshold,
               combine_models=not args.separate_models,
               model_order=args.model_order,
               write_standalone_legend=not args.no_standalone_legend,
               no_figure_legend=args.no_figure_legend,
               source=args.source,
               )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
