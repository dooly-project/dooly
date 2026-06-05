"""Plot Dooly-predicted vs vLLM-measured latency for every (model, backend) combo
in a validate_predictions.py summary.

One subplot per model. Each backend gets a color; dashed = Dooly predicted,
thick solid = real vLLM measurement.

Usage:
    python -m simulator.tools.plot_validation \
        --summary /home/jkim38/dooly/results/scenario_a_validation_locked_1700_v2_dense/validation_summary.json \
        --output  /home/jkim38/dooly/results/scenario_a_validation_locked_1700_v2_dense/latency_curves.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FixedLocator


# Color encoding: each model gets a single solid color (eye groups same model
# together at a glance — no shade variation). Backend is encoded by marker
# shape instead. Pred vs measured stays in the line style.
_MODEL_COLOR = {
    "meta-llama/Llama-3.1-8B":              "#1f77b4",  # blue
    "Command-R7B":  "#2ca02c",  # green
}
# Fallback colors for unseen models (cycled in order of appearance).
_FALLBACK_COLORS = ["#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
_BACKEND_MARKER = {
    "FLASHINFER":  "o",   # circle
    "FLASH_ATTN":  "s",   # square
    "TRITON_ATTN": "^",   # triangle up
}
_FALLBACK_MARKERS = ["D", "v", "P", "X", "*"]


def _model_color(model: str) -> str:
    if "c4ai" in model:
        model = "Command-R7B"
    if model not in _MODEL_COLOR:
        used = set(_MODEL_COLOR.values())
        for c in _FALLBACK_COLORS:
            if c not in used:
                _MODEL_COLOR[model] = c
                break
        else:
            _MODEL_COLOR[model] = _FALLBACK_COLORS[
                len(_MODEL_COLOR) % len(_FALLBACK_COLORS)
            ]
    return _MODEL_COLOR[model]


def _backend_marker(backend: str) -> str:
    if backend not in _BACKEND_MARKER:
        used = set(_BACKEND_MARKER.values())
        for m in _FALLBACK_MARKERS:
            if m not in used:
                _BACKEND_MARKER[backend] = m
                break
        else:
            _BACKEND_MARKER[backend] = "o"
    return _BACKEND_MARKER[backend]


def _build_compact_legend_handles(
    models: list[str], backends: list[str]
) -> tuple[list, list]:
    """Build a 5-entry legend: 2 model colors + 3 backend markers.

    Color entries use a neutral solid line of the model's color (so the eye
    learns "blue line = Llama, green line = Cohere"). Marker entries use a
    neutral gray marker on a faint line (so the eye learns "circle = FLASHINFER,
    square = FLASH_ATTN, triangle = TRITON_ATTN"). Pred/measured line-style
    convention is shown in the title.
    """
    handles, labels = [], []
    # Model color entries
    for model in models:
        if "c4ai" in model:
            model = "Command-R7B"
        handles.append(
            Line2D([], [], color=_model_color(model), linewidth=4, marker=None)
        )
        labels.append(model_short(model))
    # Backend marker entries (neutral gray so the marker shape is what reads)
    for backend in backends:
        handles.append(
            Line2D([], [], color="#666666", linewidth=0,
                   marker=_backend_marker(backend), markersize=9,
                   markerfacecolor="#666666", markeredgecolor="#222222")
        )
        labels.append(backend)
    return handles, labels

# Line styling: dashed for predictions, thick solid for measured.
PRED_STYLE = dict(linestyle="--", linewidth=1.6, alpha=0.95)
MEAS_STYLE = dict(linestyle="-",  linewidth=3.0, alpha=0.85)


def _token_label(n: float, _pos=None) -> str:
    """Format token counts as K notation: 1024 -> '1K', 16384 -> '16K', 768 -> '768'."""
    n = int(round(n))
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    if n >= 1000:
        # non-power-of-2 thousands (e.g. 1280 -> '1.25K') stay numeric for clarity.
        return f"{n}"
    return f"{n}"


def _apply_token_axis(ax, token_values: list[int]) -> None:
    """Linear x-axis with K-style labels.

    On a linear axis the sub-1K powers-of-2 sit near the origin and overlap
    each other, so we only label powers-of-2 ≥ 1K. Smaller data points still
    show as minor ticks (faint grid, no text).
    """
    sampled = sorted(set(token_values))
    major = [v for v in sampled if v >= 1024 and (v & (v - 1)) == 0]
    minor = [v for v in sampled if v not in major]
    if not major:
        major = sampled
    ax.xaxis.set_major_locator(FixedLocator(major))
    ax.xaxis.set_major_formatter(FuncFormatter(_token_label))
    ax.xaxis.set_minor_locator(FixedLocator(minor))


def model_short(name: str) -> str:
    return name.split("/")[-1]


def load_summary(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def group_by_model_backend(rows: list[dict]) -> dict:
    """{(model, backend): [(length, predicted_ms, measured_ms), ...] sorted by length}"""
    g: dict = defaultdict(list)
    for r in rows:
        g[(r["model"], r["backend"])].append(
            (r["length"], r["predicted_ms"], r["measured_ms"])
        )
    for k in g:
        g[k].sort()
    return g


def _draw_curves(ax, grouped, models, backends) -> None:
    """Plot every (model, backend) pred + meas pair into the given axes.

    Shared between the main axis and the zoom inset so they stay in sync.
    """
    for model in models:
        color = _model_color(model)
        for backend in backends:
            pts = grouped.get((model, backend), [])
            if not pts:
                continue
            xs = [L for L, _, _ in pts]
            preds = [p for _, p, _ in pts]
            meas  = [m for _, _, m in pts]
            marker = _backend_marker(backend)
            ax.plot(xs, meas,  color=color, marker=marker, markersize=6,
                    markerfacecolor=color, markeredgecolor=color, **MEAS_STYLE)
            ax.plot(xs, preds, color=color, marker=marker, markersize=6,
                    markerfacecolor="white", markeredgecolor=color, **PRED_STYLE)


def _all_zero_crossings(xs: list[int], ys: list[float]) -> list[float]:
    """Linearly-interpolated x values where y crosses 0, in increasing-x order."""
    crossings = []
    for i in range(1, len(xs)):
        y0, y1 = ys[i - 1], ys[i]
        if (y0 > 0 and y1 <= 0) or (y0 < 0 and y1 >= 0):
            x0, x1 = xs[i - 1], xs[i]
            t = y0 / (y0 - y1) if (y0 - y1) != 0 else 0
            crossings.append(x0 + t * (x1 - x0))
    return crossings


def _interp_inversion(xs: list[int], ys: list[float],
                      method: str = "median") -> float | None:
    """Summary x where y == 0 (linear interp), or None if no crossing.

    method:
        - 'first'  : earliest sign change (sensitive to noise)
        - 'last'   : latest sign change (sensitive to noise on the back end)
        - 'median' : median of all sign changes — robust to single-point
                     noise that produces a spurious extra crossing
                     (default; what we recommend when the curve hovers
                     near zero in the crossover region).
    """
    crossings = _all_zero_crossings(xs, ys)
    if not crossings:
        return None
    if method == "first":
        return crossings[0]
    if method == "last":
        return crossings[-1]
    if method == "median":
        # Even count → halfway between the two middle crossings.
        n = len(crossings)
        if n % 2 == 1:
            return crossings[n // 2]
        return 0.5 * (crossings[n // 2 - 1] + crossings[n // 2])
    raise ValueError(f"unknown method {method!r}")


def _crossover_band(xs: list[int], ys: list[float],
                    floor: float) -> tuple[float, float] | None:
    """Return (lo, hi) — the x-range where |y| < floor (linearly interpolated
    at the band edges). Spans the first entry into the band through the last
    exit out of it; if the curve dips out and back in, the whole stretch is
    treated as one "uncertain" region. Returns None if the curve never enters
    the band.
    """
    in_band = [abs(y) < floor for y in ys]
    if not any(in_band):
        return None

    first = next(i for i, b in enumerate(in_band) if b)
    last = max(i for i, b in enumerate(in_band) if b)

    # Entry edge: interpolate where |y| crosses `floor` on segment (first-1, first)
    if first == 0:
        lo = float(xs[0])
    else:
        y0, y1 = ys[first - 1], ys[first]
        target = floor if y0 > 0 else -floor
        t = (target - y0) / (y1 - y0) if y1 != y0 else 0.0
        lo = xs[first - 1] + t * (xs[first] - xs[first - 1])

    # Exit edge: interpolate where |y| crosses `floor` on segment (last, last+1)
    if last == len(xs) - 1:
        hi = float(xs[-1])
    else:
        y0, y1 = ys[last], ys[last + 1]
        target = floor if y1 > 0 else -floor
        t = (target - y0) / (y1 - y0) if y1 != y0 else 0.0
        hi = xs[last] + t * (xs[last + 1] - xs[last])

    return (lo, hi)


def _draw_diff_inset(inset, grouped, models, backends, baseline_model: str,
                     source: str = "measured",
                     noise_floor: float | None = None,
                     inversion_method: str = "median") -> None:
    """Plot relative latency `(other - baseline)/baseline × 100 [%]` per backend.

    The zero-line marks the inversion (where `other` overtakes `baseline`).
    Color: the *other* model's color (consistent with the main legend).
    Marker shape: backend, same as the main plot.

    `source` selects which latency to use:
        - 'measured'  : the vLLM-direct measurement (default)
        - 'predicted' : the Dooly db_predictor prediction

    `noise_floor` (in % units): if set, shade per-backend axvspan over the
    L-range where |y| < noise_floor — making the "crossover band" explicit
    instead of pretending the inversion is a single L value.
    """
    if source not in ("measured", "predicted"):
        raise ValueError(f"source must be 'measured' or 'predicted', got {source!r}")
    others = [m for m in models if m != baseline_model]
    if not others:
        inset.text(0.5, 0.5, "(need >1 model for diff inset)",
                   transform=inset.transAxes, ha="center", va="center", fontsize=9)
        return
    other = others[0]
    accent = _model_color(other)

    # Horizontal "noise band" — anything inside this band is statistically
    # indistinguishable from zero given typical per-model prediction noise.
    if noise_floor is not None and noise_floor > 0:
        inset.axhspan(-noise_floor, +noise_floor, color="#e0e0e0", alpha=0.55,
                      zorder=0)

    inset.axhline(0.0, color="black", linewidth=0.9, alpha=0.55)

    # Collect all crossings first so we can dodge overlapping label positions.
    crossings: list[tuple[str, float]] = []
    bands: list[tuple[str, float, float]] = []   # (backend, lo, hi)
    backend_lines: dict[str, tuple[list[int], list[float]]] = {}
    for backend in backends:
        per_L: dict[int, dict[str, float]] = defaultdict(dict)
        for (m, b), pts in grouped.items():
            if b != backend:
                continue
            for L, pred, meas in pts:
                per_L[L][m] = pred if source == "predicted" else meas
        xs = sorted(L for L, vals in per_L.items()
                    if baseline_model in vals and other in vals)
        if not xs:
            continue
        ys = [(per_L[L][other] - per_L[L][baseline_model])
              / per_L[L][baseline_model] * 100.0 for L in xs]
        backend_lines[backend] = (xs, ys)
        inset.plot(xs, ys, color=accent, marker=_backend_marker(backend),
                   markersize=6, markerfacecolor=accent, markeredgecolor=accent,
                   linewidth=1.8, alpha=0.95)
        inv_x = _interp_inversion(xs, ys, method=inversion_method)
        if inv_x is not None:
            crossings.append((backend, inv_x))
        if noise_floor is not None and noise_floor > 0:
            band = _crossover_band(xs, ys, noise_floor)
            if band is not None:
                bands.append((backend, band[0], band[1]))

    # Per-backend crossover bands (vertical shading on the L axis).
    # Each band uses the model's accent color but with low alpha so the
    # underlying noise band still reads through.
    for backend, lo, hi in bands:
        inset.axvspan(lo, hi, color=accent, alpha=0.10, zorder=0)
        # Draw thin edge marks at the band boundaries to make the range read.
        for edge in (lo, hi):
            inset.axvline(edge, color=accent, alpha=0.4, linewidth=0.8,
                          linestyle=":", zorder=1)

    # Sort crossings by x and dodge labels that are within ~10% of x-range.
    crossings.sort(key=lambda kv: kv[1])
    if crossings:
        all_xs_inset = [x for xs, _ in backend_lines.values() for x in xs]
        x_range = max(all_xs_inset) - min(all_xs_inset)
        last_x = -float("inf")
        offset_step = 0
        for backend, inv_x in crossings:
            inset.scatter([inv_x], [0.0], color="white",
                          edgecolors="black", linewidth=1.4, s=70, zorder=5)
            # Dodge label vertically when crossings are too close in x.
            if inv_x - last_x < 0.10 * x_range:
                offset_step = 1 - offset_step  # alternate up/down
            else:
                offset_step = 0
            yoff = (12 if offset_step == 0 else -22)
            inset.annotate(f"{backend}\nL={int(round(inv_x))}",
                           xy=(inv_x, 0), xytext=(0, yoff),
                           textcoords="offset points",
                           fontsize=8, ha="center", color="black",
                           bbox=dict(boxstyle="round,pad=0.25",
                                     facecolor="white", edgecolor="#888888",
                                     alpha=0.92))
            last_x = inv_x

    other_short = model_short(other)
    base_short = model_short(baseline_model)
    src_tag = "vLLM measured" if source == "measured" else "Dooly predicted"
    inset.set_xlabel("prefill length (tokens)", fontsize=9)
    inset.set_ylabel(f"({other_short} − {base_short}) / {base_short}  [%]",
                     fontsize=9)
    inset.set_title(
        f"relative latency ({src_tag}): {other_short} vs {base_short}    "
        f"(↓ = {other_short} faster)", fontsize=10, pad=3)
    all_xs = sorted({L for pts in grouped.values() for L, _, _ in pts})
    _apply_token_axis(inset, all_xs)
    inset.tick_params(axis="both", labelsize=8)
    inset.grid(True, which="major", alpha=0.3)
    inset.grid(True, which="minor", alpha=0.1)


def plot_inversion(rows: list[dict], output: Path,
                   baseline_model: str | None = None,
                   noise_floor: float | None = 0.5,
                   inversion_method: str = "median",
                   title: str | None = None) -> None:
    """Standalone inversion figure: relative latency curves per backend.

    Overlays:
      - Solid line + filled marker = vLLM measured
      - Dotted line + hollow marker = Dooly predicted

    Color: the *other* model's color (consistent with the latency-figure legend).
    Marker shape: backend (same convention as the latency figure).
    """
    grouped = group_by_model_backend(rows)
    models = sorted({m for m, _ in grouped})
    backends = sorted({b for _, b in grouped})
    if len(models) < 2:
        print("[plot_inversion] need >= 2 models, skipping")
        return

    baseline = baseline_model or next(
        (m for m in models if "Llama" in m), models[0]
    )
    other = next(m for m in models if m != baseline)
    accent = _model_color(other)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Horizontal noise band — anything in here is statistically indistinguishable from 0.
    if noise_floor is not None and noise_floor > 0:
        ax.axhspan(-noise_floor, +noise_floor, color="#e0e0e0",
                   alpha=0.55, zorder=0)
    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.55)

    # Per-backend: plot measured (solid) and predicted (dotted) curves and
    # collect their median crossings for annotation.
    crossings: list[tuple[str, str, float]] = []   # (backend, source, x)
    bands: list[tuple[str, str, float, float]] = []  # (backend, source, lo, hi)

    for backend in backends:
        per_L_meas: dict[int, dict[str, float]] = defaultdict(dict)
        per_L_pred: dict[int, dict[str, float]] = defaultdict(dict)
        for (m, b), pts in grouped.items():
            if b != backend:
                continue
            for L, pred, meas in pts:
                per_L_meas[L][m] = meas
                per_L_pred[L][m] = pred

        for source, per_L, line_kwargs in [
            ("measured", per_L_meas,
             dict(linestyle="-",  linewidth=2.4,
                  markerfacecolor=accent, markeredgecolor=accent, alpha=0.95)),
            ("predicted", per_L_pred,
             dict(linestyle=":",  linewidth=1.8,
                  markerfacecolor="white", markeredgecolor=accent, alpha=0.95)),
        ]:
            xs = sorted(L for L, vals in per_L.items()
                        if baseline in vals and other in vals)
            if not xs:
                continue
            ys = [(per_L[L][other] - per_L[L][baseline])
                  / per_L[L][baseline] * 100.0 for L in xs]
            ax.plot(xs, ys, color=accent, marker=_backend_marker(backend),
                    markersize=6, **line_kwargs)

            inv = _interp_inversion(xs, ys, method=inversion_method)
            if inv is not None:
                crossings.append((backend, source, inv))
            if noise_floor is not None and noise_floor > 0:
                band = _crossover_band(xs, ys, noise_floor)
                if band is not None:
                    bands.append((backend, source, band[0], band[1]))

    # Mark each crossing with a contrasting dot.
    crossings.sort(key=lambda t: t[2])
    last_x = -float("inf")
    offset_step = 0
    all_xs_main = sorted({L for pts in grouped.values() for L, _, _ in pts})
    x_range = max(all_xs_main) - min(all_xs_main)
    for backend, source, inv_x in crossings:
        face = "white" if source == "predicted" else accent
        ax.scatter([inv_x], [0.0], color=face, edgecolors="black",
                   linewidth=1.4, s=70, zorder=5)
        # Dodge labels that crowd horizontally.
        if inv_x - last_x < 0.07 * x_range:
            offset_step = 1 - offset_step
        else:
            offset_step = 0
        yoff = (14 if offset_step == 0 else -22)
        ax.annotate(f"{backend}\n{source}: {int(round(inv_x))}",
                    xy=(inv_x, 0), xytext=(0, yoff),
                    textcoords="offset points",
                    fontsize=8, ha="center", color="black",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", edgecolor="#888888",
                              alpha=0.92))
        last_x = inv_x

    other_short = model_short(other)
    base_short = model_short(baseline)
    ax.set_xlabel("prefill length (tokens)")
    ax.set_ylabel(f"({other_short} − {base_short}) / {base_short}  [%]")
    # if title is None:
        # title = (f"Cohere-vs-Llama inversion ({other_short} − {base_short}) / {base_short} × 100"
        #          f"\n(solid + filled = vLLM measured; dotted + hollow = Dooly predicted; "
        #          f"shaded = ±{noise_floor:.1f}% noise band)")
    # ax.set_title(title, fontsize=11)
    _apply_token_axis(ax, all_xs_main)
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)

    # Legend: 3 backend shapes + 2 line-style entries (measured / predicted).
    handles, labels = [], []
    for backend in backends:
        handles.append(Line2D([], [], color="#666666", linewidth=0,
                              marker=_backend_marker(backend), markersize=9,
                              markerfacecolor="#666666",
                              markeredgecolor="#222222"))
        labels.append(backend)
    handles.append(Line2D([], [], color=accent, linewidth=2.4, linestyle="-"))
    labels.append("measured (vLLM)")
    handles.append(Line2D([], [], color=accent, linewidth=1.8, linestyle=":"))
    labels.append("predicted (Dooly)")
    ax.legend(handles, labels, loc="lower left", fontsize=10, framealpha=0.95,
              title="shape → backend     line → source", title_fontsize=9)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def plot(rows: list[dict], output: Path, log: bool, title: str | None = None,
         inset_zoom_min: int | None = None,
         inset_zoom_max: int | None = 4096,
         inset_mode: str = "zoom",
         diff_baseline: str | None = None,
         diff_source: str = "measured",
         diff_noise_floor: float | None = 0.5,
         inversion_method: str = "median") -> None:
    """All (model, backend) curves on a single axis.

    Color encodes the model (one solid color per model). Marker shape
    encodes the backend (circle / square / triangle for FLASHINFER /
    FLASH_ATTN / TRITON_ATTN). Line style encodes predicted vs measured
    (dashed + hollow markers = Dooly, thick solid + filled markers = vLLM).

    Inset modes:
      - 'zoom': inset axis shows window [inset_zoom_min, inset_zoom_max] of
        the same latency curves.
      - 'diff': inset shows relative latency (other_model - baseline) /
        baseline × 100 per backend, exposing the inversion point as a
        zero-crossing. Baseline defaults to whichever model has 'Llama'
        in its name; override with `diff_baseline`.
    """
    grouped = group_by_model_backend(rows)
    models = sorted({m for m, _ in grouped})
    backends = sorted({b for _, b in grouped})

    fig, ax = plt.subplots(figsize=(13, 8.5))

    _draw_curves(ax, grouped, models, backends)

    ax.set_xlabel("prefill length (tokens)")
    ax.set_ylabel("latency (ms)")
    if title:
        ax.set_title(f"{title}\n(thick solid + filled = vLLM measured;  dashed + hollow = Dooly predicted)",
                 fontsize=11)
    if log:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
    else:
        all_xs = sorted({L for pts in grouped.values() for L, _, _ in pts})
        _apply_token_axis(ax, all_xs)
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)

    handles, labels = _build_compact_legend_handles(models, backends)
    ax.legend(handles, labels, loc="lower right", fontsize=15, framealpha=0.95, title_fontsize=9)

    # ----- Inset -----
    if inset_mode == "diff" and not log and len(models) >= 2:
        baseline = diff_baseline or next(
            (m for m in models if "Llama" in m), models[0]
        )
        inset = ax.inset_axes([0.05, 0.5, 0.5, 0.42])
        _draw_diff_inset(inset, grouped, models, backends, baseline,
                         source=diff_source, noise_floor=diff_noise_floor,
                         inversion_method=inversion_method)
    elif inset_mode == "zoom" and inset_zoom_max is not None and not log:
        inset = ax.inset_axes([0.05, 0.5, 0.5, 0.42])
        _draw_curves(inset, grouped, models, backends)
        # Limit to small-L window
        all_xs = sorted({L for pts in grouped.values() for L, _, _ in pts})
        lo = inset_zoom_min if inset_zoom_min is not None else min(all_xs)
        zoom_xs = [x for x in all_xs if lo <= x <= inset_zoom_max]
        if not zoom_xs:
            print(f"[warn] no data in inset window [{lo}, {inset_zoom_max}]; skipping inset")
        else:
            zoom_ys = [
                v for pts in grouped.values()
                for L, p, m in pts if lo <= L <= inset_zoom_max
                for v in (p, m)
            ]
            zoom_min_y, zoom_max_y = min(zoom_ys), max(zoom_ys)
            x_pad = max(50, (inset_zoom_max - lo) * 0.02)
            y_pad = (zoom_max_y - zoom_min_y) * 0.08 if zoom_max_y > zoom_min_y else 1.0
            y_lo = 0 if lo <= min(all_xs) else max(0.0, zoom_min_y - y_pad)
            inset.set_xlim(lo - x_pad, inset_zoom_max + x_pad)
            inset.set_ylim(y_lo, zoom_max_y + y_pad)
            _apply_token_axis(inset, zoom_xs)
            inset.tick_params(axis="both", labelsize=12)
            inset.grid(True, which="major", alpha=0.3)
            inset.grid(True, which="minor", alpha=0.1)
            lo_label = _token_label(lo)
            hi_label = _token_label(inset_zoom_max)
            inset.set_title(f"zoom: {lo_label} ≤ L ≤ {hi_label}", fontsize=15, pad=2)
            ax.indicate_inset_zoom(inset, edgecolor="#444444", alpha=0.6, linewidth=1.0)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True, type=Path,
                    help="validate_predictions.py summary JSON.")
    ap.add_argument("--output", default=None, type=Path,
                    help="Output path for the latency curve PNG. "
                         "Defaults to <summary_dir>/latency_curves.png")
    ap.add_argument("--gap-output", default=None, type=Path,
                    help="Output path for the gap%% PNG. "
                         "Defaults to <summary_dir>/gap_curves.png")
    ap.add_argument("--log", action="store_true",
                    help="Log-log axes (default: linear).")
    ap.add_argument("--inset-zoom-min", type=int, default=None,
                    help="Lower bound of the inset zoom window in tokens. "
                         "Defaults to the smallest sampled L.")
    ap.add_argument("--inset-zoom-max", type=int, default=4096,
                    help="Upper bound of the inset zoom window. "
                         "Set to 0 to disable the inset entirely. "
                         "(default: 4096)")
    ap.add_argument("--inset-mode", choices=["zoom", "diff", "none"],
                    default="none",
                    help="Inset on the main latency figure. "
                         "zoom = zoomed copy of latency curves; "
                         "diff = relative latency inset; "
                         "none = no inset; the inversion plot is written as "
                         "a separate figure instead. (default: none)")
    ap.add_argument("--inversion-output", default=None, type=Path,
                    help="Output path for the standalone inversion figure. "
                         "Defaults to <summary_dir>/inversion_curves.png.")
    ap.add_argument("--skip-inversion", action="store_true",
                    help="Don't write a standalone inversion figure.")
    ap.add_argument("--diff-baseline", default=None,
                    help="Baseline model for inset-mode=diff (default: the "
                         "model whose name contains 'Llama').")
    ap.add_argument("--diff-source", choices=["measured", "predicted"],
                    default="measured",
                    help="Which latency to use for the diff inset: "
                         "'measured' (vLLM-direct) or 'predicted' (Dooly). "
                         "(default: measured)")
    ap.add_argument("--diff-noise-floor", type=float, default=0.5,
                    help="In percent. Shade per-backend axvspan over the "
                         "L-range where |D| < this value, plus a horizontal "
                         "axhspan inside ±this value, to make the 'crossover "
                         "band' explicit. Set to 0 to disable shading. "
                         "(default: 0.5)")
    ap.add_argument("--inversion-method",
                    choices=["first", "last", "median"], default="median",
                    help="Which sign-change to label when the diff curve "
                         "crosses zero multiple times. 'median' is robust to "
                         "single-point noise; 'first' / 'last' are the "
                         "earliest / latest crossings respectively. "
                         "(default: median)")
    ap.add_argument("--title-suffix", default="",
                    help="Appended to the plot title.")
    args = ap.parse_args()

    rows = load_summary(args.summary)

    out = args.output or args.summary.parent / "latency_curves.png"
    inset_max = args.inset_zoom_max if args.inset_zoom_max > 0 else None
    plot(rows, out, log=args.log,
        #  title=f"Predicted vs measured{args.title_suffix}",
        inset_zoom_min=args.inset_zoom_min,
        inset_zoom_max=inset_max,
        inset_mode=args.inset_mode,
        diff_baseline=args.diff_baseline,
        diff_source=args.diff_source,
        diff_noise_floor=(args.diff_noise_floor if args.diff_noise_floor > 0 else None),
        inversion_method=args.inversion_method)

    # Standalone inversion figure (overlay of measured + predicted diff curves).
    if not args.skip_inversion:
        inv_out = args.inversion_output or args.summary.parent / "inversion_curves.png"
        plot_inversion(
            rows, inv_out,
            baseline_model=args.diff_baseline,
            noise_floor=(args.diff_noise_floor if args.diff_noise_floor > 0 else None),
            inversion_method=args.inversion_method,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
