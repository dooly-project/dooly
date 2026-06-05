"""Scenario A v2 — overlap & dedup benefit analysis.

Produces just two plots:
  (1) breadth_vs_depth.png
        Per-op-class dedup efficiency (% of naive time saved, bars, left axis)
        vs absolute GPU-hours saved (red diamonds, right axis, log).
        Includes a headline banner with total savings.
  (2) cumulative_coverage.png
        Greedy model-ordering coverage curves + marginal contribution bars.

Naive-aggregation modes (--naive-mode):
  max          (default, legacy): per-(sig, backend), use max wall_time across
                 measurements; multiply by len(models_using_sig). Treats every
                 model's contribution as the worst-of-all-shards measurement.
  extrapolated: per-(model, sig, backend), use the model's own measurement when
                 present in the CSV; otherwise extrapolate by per-config rate
                 from same-sig donors at the model's native max-position cap.
                 Designed to recover what a model WOULD have measured at its
                 own cap when it deduped to a shorter-cap rep at profile time.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PANEL_CHOICES = ["coverage", "cost", "marginal"]

# Module-level font-size scaler. Set by --pdf to shrink all explicit
# fontsize=N / labelsize=N values proportionally for paper-column figures.
_FS_SCALE = 1.0
PNG_SCALE = 0.8


def fs(n):
    """Scaled font size. Returns n unless --pdf is active."""
    return max(5.0, n * _FS_SCALE)


def apply_paper_rc():
    """Paper-submission friendly defaults: embedded TrueType fonts, serif."""
    mpl.rcParams.update({
        "axes.labelsize":   15,
        "axes.titlesize":  15,
        "xtick.labelsize":  15,
        "ytick.labelsize":  15,
        "legend.fontsize":  10,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.02,
    })



parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--csv-path", type=Path, default=None)
parser.add_argument("--db-path", type=Path, default=None)
parser.add_argument("--out-dir", type=Path, default=None)
parser.add_argument("--panels", nargs="+", choices=PANEL_CHOICES, default=None, help="Subset of cumulative_coverage panels to plot. Default: all three.")
parser.add_argument("--skip-breadth", action="store_true", help="Skip generating breadth_vs_depth.png")
parser.add_argument("--pdf", action="store_true", help="Emit PDF figures sized for paper submission (7in wide, Times serif, Type-42 embedded).")
parser.add_argument("--naive-mode", choices=["max", "direct", "extrapolated"], default="direct",
                    help="How to compute naive time. "
                         "'direct' (default) sums each (model, sig, backend) "
                         "row's own wall_time from the CSV — correct when "
                         "every model directly measured each of its sigs "
                         "(e.g. v2 multi-shard runs). Cells with no CSV row "
                         "contribute 0 (and are reported in the coverage line). "
                         "'max' (legacy) multiplies max-shard wall_time by "
                         "n_models_using_sig — over-bounds. "
                         "'extrapolated' uses direct lookup where available "
                         "and rate-extrapolates missing cells via attention-"
                         "anchored ratios (use only when coverage is partial).")

# Use parse_known_args() so importing this module from another script (e.g.,
# export_op_savings.py with its own CLI flags) doesn't crash here. The unknown
# flags are forwarded to whatever argparse the importing script runs.
_modlevel_args, _ = parser.parse_known_args()
DB = Path("/home/jkim38/dooly-project/doolyprof-dev/database/dooly-eval-v2.db") if not _modlevel_args.db_path else _modlevel_args.db_path
CSV_V2 = Path("/home/jkim38/dooly/comparison/overlap_v2/profile_timings_v2.csv") if not _modlevel_args.csv_path else _modlevel_args.csv_path
OUT_DIR = Path("/home/jkim38/dooly/comparison/overlap_v2") if not _modlevel_args.out_dir else _modlevel_args.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_model_sigs(db_path: Path):
    """{(model, backend): set(sig_hash)}"""
    out = defaultdict(set)
    with sqlite3.connect(str(db_path)) as c:
        for model, backend, sig in c.execute("""
            SELECT pm.model_name, pm.backend, s.signature_hash
            FROM profiled_models pm
            JOIN model_operations mo ON mo.model_id = pm.model_id
            JOIN signatures s ON s.operation_id = mo.operation_id
        """):
            out[(model, backend)].add(sig)
    return out


def load_sig_op_name(db_path: Path):
    """{sig_hash: operation_name}"""
    out = {}
    with sqlite3.connect(str(db_path)) as c:
        for sig, op in c.execute("SELECT signature_hash, operation_name FROM signatures"):
            out[sig] = op
    return out


def load_sig_times(csv_path: Path):
    """{(sig_hash, backend): max wall_time_s seen} (max for safety on re-runs)."""
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            sig = row.get("signature_hash", "")
            backend = row.get("backend", "")
            if not sig:
                continue
            try:
                t = float(row["wall_time_s"])
            except (ValueError, KeyError):
                continue
            k = (sig, backend)
            if k not in out or t > out[k]:
                out[k] = t
    return out


def load_naive_direct(csv_path: Path, model_sigs):
    """{(model, backend, sig): wall_time_s} — direct CSV lookup, latest row wins.

    For each `(model, backend, sig)` tuple in `model_sigs`, look up the matching
    CSV row's wall_time. Cells with no CSV row are simply omitted from the
    returned dict (so naive sums skip them; their absence is reported via the
    coverage breakdown).

    Used by --naive-mode=direct, which is correct for runs where every model
    measured each of its sigs in its own shard (e.g. v2 multi-shard runs with
    --overwrite). No extrapolation; no rate model; just direct CSV lookup.
    """
    direct = {}
    csv_rows = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            sig = row.get("signature_hash", "")
            backend = row.get("backend", "")
            model = row.get("model_name", "")
            if not (sig and model):
                continue
            try:
                t = float(row["wall_time_s"])
            except (ValueError, KeyError):
                continue
            if t <= 0:
                continue
            csv_rows += 1
            key = (model, backend, sig)
            # Latest CSV row wins (later runs overwrite earlier in the file).
            direct[key] = t

    out = {}
    n_direct = 0
    n_missing = 0
    for (m, b), sigs in model_sigs.items():
        for s in sigs:
            if (m, b, s) in direct:
                out[(m, b, s)] = direct[(m, b, s)]
                n_direct += 1
            else:
                n_missing += 1
    report = {"direct": n_direct, "missing": n_missing, "csv_rows_seen": csv_rows}
    return out, report


def load_csv_donors(csv_path: Path):
    """{(sig_hash, backend): [(model, n_configs, wall_time_s), ...]}.

    Multi-shard CSVs (e.g. v2 vLLM run) emit one row per (shard, model, sig,
    backend) pair, so a given (sig, backend) can have several donor rows from
    different models at different sweep caps. Used by the extrapolated-naive
    path to estimate what each model would have measured at its own cap.
    """
    out = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            sig = row.get("signature_hash", "")
            backend = row.get("backend", "")
            model = row.get("model_name", "")
            if not sig or not model:
                continue
            try:
                ncfg = int(row["n_configs"])
                t = float(row["wall_time_s"])
            except (ValueError, KeyError):
                continue
            if ncfg <= 0 or t <= 0:
                continue
            out[(sig, backend)].append((model, ncfg, t))
    return out


def load_model_caps(model_names):
    """{model_name: max_position_embeddings}.

    Read from the local HuggingFace hub cache. Models without a cached
    config.json fall back to a conservative default of 8192. Same lookup the
    profiler itself does at sweep-config time.
    """
    hf = os.path.expanduser("~/.cache/huggingface/hub")
    out = {}
    for m in model_names:
        safe = "models--" + m.replace("/", "--")
        cfg_paths = glob.glob(f"{hf}/{safe}/snapshots/*/config.json")
        cap = 8192
        if cfg_paths:
            try:
                with open(cfg_paths[0]) as f:
                    cfg = json.load(f)
                cap = (cfg.get("max_position_embeddings")
                       or cfg.get("max_sequence_length")
                       or cap)
            except (OSError, json.JSONDecodeError):
                pass
        out[m] = int(cap)
    return out


def predict_n_configs_at_cap(donors, target_cap):
    """Predict n_configs at `target_cap` from a list of (model, n_configs, t) donors.

    Each donor's (cap, n_configs) is implicit — the donor measured the same
    sig at that donor's own cap. n_configs grows roughly logarithmically with
    cap because the sweep enumerates seq_len bucket-wise. We use linear
    interpolation on the ((cap, n_configs)) curve, with closest-match fallback
    when target_cap is outside the donor range.

    Caller passes donor caps via the donor list; we infer cap from each donor
    by mapping model_name through `model_caps` (closure-captured by caller).
    """
    # caller supplies (cap, n_configs) pairs already resolved
    if not donors:
        return 0
    points = sorted(set((cap, n) for (cap, n) in donors), key=lambda x: x[0])
    if len(points) == 1:
        # only one donor; assume same n_configs as that donor regardless of cap
        return points[0][1]
    # Below the smallest cap → use smallest's n_configs
    if target_cap <= points[0][0]:
        return points[0][1]
    # Above the largest cap → use largest's n_configs
    if target_cap >= points[-1][0]:
        return points[-1][1]
    # Linear interpolate in (cap, n_configs)
    for (c0, n0), (c1, n1) in zip(points, points[1:]):
        if c0 <= target_cap <= c1:
            frac = (target_cap - c0) / (c1 - c0) if c1 > c0 else 0
            return int(round(n0 + frac * (n1 - n0)))
    return points[-1][1]


def build_model_ncfg_reference(donors, sig_op, model_caps=None, target_models=None):
    """{(model, backend): typical n_configs for attention sigs at this model's cap}.

    For each (model, backend) appearing anywhere in the CSV, take the median
    n_configs across attention-class donor rows. This serves as the per-model
    "sweep size fingerprint at its native cap" — empirically grounded, not
    extrapolated. Used to scale donor wall_time to a target model via
    ratio of (target_ref_ncfg / donor_ref_ncfg).

    Why attention specifically: attention sigs sweep over (batch, kv_size)
    so their n_configs is ~uniform across sigs of the same arch class at the
    same cap, making them the most reliable per-model reference.

    If `model_caps` and `target_models` are provided, ALSO derive a fallback
    reference for any model that lacks an attention measurement, using
    interpolation in (cap, n_configs)-space across the observed data. This
    handles models that were dedup-skipped from attention profiling entirely.
    """
    by_mb = defaultdict(list)
    for (sig, backend), rows in donors.items():
        if "Attention" not in (sig_op.get(sig) or ""):
            continue
        for (m, n, t) in rows:
            if n > 0:
                by_mb[(m, backend)].append(n)
    direct_ref = {k: median(vs) for k, vs in by_mb.items() if vs}

    if not (model_caps and target_models):
        return direct_ref

    # Per-backend (cap, n_configs) curve from direct refs to interpolate from.
    by_b = defaultdict(list)  # backend -> [(cap, ncfg)]
    for (m, b), nc in direct_ref.items():
        cap = model_caps.get(m)
        if cap:
            by_b[b].append((cap, nc))

    out = dict(direct_ref)
    backends = {b for (_, b) in by_mb}
    for m in target_models:
        for b in backends:
            if (m, b) in out:
                continue
            cap = model_caps.get(m)
            curve = by_b.get(b, [])
            if not cap or not curve:
                continue
            out[(m, b)] = predict_n_configs_at_cap(curve, cap)
    return out


def estimate_naive_per_msb(model_sigs, donors, model_caps, sig_op):
    """Per-(model, backend, sig) wall_time estimate for the extrapolated naive.

    Two-step algorithm:
      1. If donors has a row for THIS (model, sig, backend) → use it directly.
      2. Else: scale a donor measurement by the inter-model n_configs ratio:
           predicted_t = donor.t × (target_ref_ncfg / donor_ref_ncfg)
         where ref_ncfg is each model's typical attention n_configs at its
         native cap (computed from direct measurements anywhere in the CSV).

    This works correctly in BOTH directions:
      - Target cap > donor caps (UP-extrap, e.g., c4ai using Meta-Llama-3 sig):
        target_ref_ncfg > donor_ref_ncfg → predicted_t scales UP.
      - Target cap < donor caps (DOWN-extrap, e.g., Meta-Llama-3 using c4ai sig):
        target_ref_ncfg < donor_ref_ncfg → predicted_t scales DOWN.

    Median across multiple donors filters per-shard noise.

    Returns: {(model, backend, sig): wall_time_s}, plus a coverage report.
    """
    target_models = sorted({m for (m, _) in model_sigs})
    ref_ncfg = build_model_ncfg_reference(
        donors, sig_op, model_caps=model_caps, target_models=target_models)

    out = {}
    direct = 0
    same_cap = 0
    extrapolated = 0
    no_donor = 0
    no_ref = 0
    for (model, backend), sigs in model_sigs.items():
        target_cap = model_caps.get(model)
        for sig in sigs:
            ds = donors.get((sig, backend), [])
            if not ds:
                no_donor += 1
                continue
            # Direct match: this model's own measurement of this (sig, backend)
            self_rows = [(n, t) for (mname, n, t) in ds if mname == model]
            if self_rows:
                out[(model, backend, sig)] = max(t for _, t in self_rows)
                direct += 1
                continue

            # Same-cap peer: another model with identical native cap measured
            # this exact (sig, backend). Since the sig hash matches, the kernel
            # inputs are bit-identical; sweep range is set by min(cap, cli),
            # so a same-cap peer's wall_time is ~equal to what this model would
            # have measured (within ~5% noise). Prefer this over multi-donor
            # ratio scaling, which can over-shoot when donors span cap classes.
            if target_cap is not None:
                peer_rows = [(n, t) for (mname, n, t) in ds
                             if mname != model and model_caps.get(mname) == target_cap]
                if peer_rows:
                    out[(model, backend, sig)] = max(t for _, t in peer_rows)
                    same_cap += 1
                    continue

            # Need ref_ncfg for the target model on this backend
            target_ref = ref_ncfg.get((model, backend))
            if target_ref is None:
                # Target model has no attention reference on this backend — fall
                # back to legacy rate × predicted_ncfg (less reliable but covers
                # the gap).
                rates = [t / n for (_, n, t) in ds if n > 0]
                if not rates:
                    no_donor += 1
                    continue
                rate = median(rates)
                m_cap = model_caps.get(model, 8192)
                donor_pts = [(model_caps.get(mn, 8192), n) for (mn, n, _) in ds]
                pred_ncfg = predict_n_configs_at_cap(donor_pts, m_cap)
                out[(model, backend, sig)] = rate * pred_ncfg
                no_ref += 1
                continue

            # Ratio-scale each donor's wall_time using the reference n_configs
            # that empirically captures each model's effective sweep size.
            preds = []
            for (donor_model, n_donor, t_donor) in ds:
                donor_ref = ref_ncfg.get((donor_model, backend), n_donor)
                if donor_ref <= 0:
                    continue
                ratio = target_ref / donor_ref
                preds.append(t_donor * ratio)
            if not preds:
                no_donor += 1
                continue
            out[(model, backend, sig)] = median(preds)
            extrapolated += 1

    report = {
        "direct": direct,
        "same_cap_peer": same_cap,
        "extrapolated": extrapolated,
        "no_donor": no_donor,
        "no_ref_fallback": no_ref,
    }
    return out, report


# SHORT_NAME_MAP = {
#     "meta-llama/Meta-Llama-3-8B":                 "L3-8B",
#     "meta-llama/Llama-3.1-8B":                    "L3.1-8B",
#     "meta-llama/Llama-2-7b-hf":                   "L2-7B",
#     "Qwen/Qwen2.5-7B":                            "Q2.5-7B",
#     "Qwen/Qwen3-8B":                              "Q3-8B",
#     "CohereLabs/aya-expanse-8b":                  "C-8B",
#     "CohereLabs/c4ai-command-r7b-12-2024":        "C-7B",
#     "mistralai/Mistral-7B-v0.3":                  "M-7B",
#     "mistralai/Ministral-8B-Instruct-2410":       "M-8B",
#     "deepseek-ai/deepseek-llm-7b-base":           "D-7B",
#     "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":   "D-L8B",
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":    "D-Q7B",
# }


SHORT_NAME_MAP = {
    "meta-llama/Meta-Llama-3-8B":                 "L3-8B",
    "meta-llama/Llama-3.1-8B":                    "L3.1-8B",
    "meta-llama/Llama-2-7b-hf":                   "L2-7B",
    "Qwen/Qwen2.5-7B":                            "Q2.5-7B",
    "Qwen/Qwen3-8B":                              "Q3-8B",
    "CohereLabs/aya-expanse-8b":                  "C-8B",
    "CohereLabs/c4ai-command-r7b-12-2024":        "C-7B",
    "mistralai/Mistral-7B-v0.3":                  "M-7B",
    "mistralai/Ministral-8B-Instruct-2410":       "M-8B",
    "deepseek-ai/deepseek-llm-7b-base":           "D-L7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":   "D-L8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":    "D-Q7B",
}


# Family-based color coding (reused across panels for model-tick coloring).
FAMILY_COLOR = {
    "meta-llama":   "#1f77b4",   # blue
    "Qwen":         "#d62728",   # red
    "CohereLabs":   "#ff7f0e",   # orange
    "mistralai":    "#2ca02c",   # green
    "deepseek-ai":  "#9467bd",   # purple
}


def short_name(full: str) -> str:
    if full in SHORT_NAME_MAP:
        return SHORT_NAME_MAP[full]
    base = full.split("/")[-1]
    return (base
        .replace("-Instruct-2410", "-Inst")
        .replace("-Distill-", "-D-")
        .replace("c4ai-command-", "cmd-")
    )


def family_color(full: str) -> str:
    org = full.split("/")[0] if "/" in full else full
    return FAMILY_COLOR.get(org, "#555555")


def op_class(op_name: str) -> str:
    if op_name == "Attention":
        return "Attention"
    l = op_name.lower()
    if "linear" in l:                return "Linear (aten::linear)"
    if "rms_norm" in l or "layer_norm" in l: return "Norm (RMSNorm)"
    if "rotary" in l:                return "RoPE"
    if "silu" in l or "act" in l:    return "Activation"
    if "embedding" in l:             return "Embedding"
    if "argmax" in l or "topk" in l: return "Sampler"
    return "Other"


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------

def _render_breadth(class_naive, class_dedup, class_saved, class_reduct,
                    class_sigs, backend_dedup, backend_naive, oc_color,
                    ext: str = "png"):
    classes_sorted = sorted(class_naive.keys(), key=lambda c: -class_saved[c])
    reduct_vals = [class_reduct[c] for c in classes_sorted]
    saved_h     = [class_saved[c] / 3600.0 for c in classes_sorted]
    count_vals  = [class_sigs[c]  for c in classes_sorted]

    figsize = (7.0, 3.8) if ext == "pdf" else (12, 5.8)
    fig, ax_left = plt.subplots(figsize=figsize)
    x = np.arange(len(classes_sorted))
    colors_b = [oc_color.get(c, "#888") for c in classes_sorted]

    ax_left.bar(x, reduct_vals, 0.55, color=colors_b, edgecolor="white", linewidth=0.6)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(classes_sorted, rotation=20, ha="right", fontsize=fs(10))
    ax_left.set_ylabel("Profiling time saved by dedup  (% of naive time)",
                       fontsize=fs(11), color="#004477", fontweight="semibold")
    ax_left.tick_params(axis="y", labelcolor="#004477", labelsize=fs(12))
    ax_left.set_ylim(0, 105)
    ax_left.axhline(y=50, color="#004477", linestyle=":", alpha=0.4)
    ax_left.grid(axis="y", linestyle="--", alpha=0.3)
    for xi, rv, cv in zip(x, reduct_vals, count_vals):
        ax_left.text(xi, rv + 1.5, f"{rv:.0f}%\n({cv} sigs)",
                     ha="center", va="bottom", fontsize=fs(9), color="#333")

    ax_right = ax_left.twinx()
    ax_right.plot(x, saved_h, marker="D", linestyle="", markersize=13,
                  markerfacecolor="#cc3333", markeredgecolor="white", zorder=5)
    ax_right.set_ylabel("Wall-clock profiling time saved by dedup (GPU-hours)",
                        fontsize=fs(11), color="#cc3333", fontweight="semibold")
    ax_right.tick_params(axis="y", labelcolor="#cc3333", labelsize=fs(12))
    ax_right.set_yscale("symlog", linthresh=0.01)
    ax_right.set_ylim(0, max(saved_h) * 1.4)
    for xi, hv in zip(x, saved_h):
        lbl = (f"{hv:.1f}h" if hv >= 1 else
               (f"{hv*60:.0f}min" if hv >= 1/60 else f"{hv*3600:.0f}s"))
        ax_right.annotate(lbl, (xi, hv),
                          textcoords="offset points", xytext=(0, -18),
                          ha="center", fontsize=fs(9), color="#cc3333")

    lbars = plt.Line2D([], [], color="#666", marker="s", linestyle="",
                       markersize=11, label="% of naive time saved (left axis)")
    ldots = plt.Line2D([], [], color="#cc3333", marker="D", linestyle="",
                       markersize=11, markerfacecolor="#cc3333",
                       label="GPU-hours saved (right axis, log)")
    ax_left.legend(handles=[lbars, ldots], loc="upper right", fontsize=fs(12))

    # ax_left.set_title(
    #     "Profiling Overhead Reduction Per Operation\n"
    #     "(12 models × 3 backends)",
    #     fontsize=fs(15))
    fig.tight_layout()
    out = OUT_DIR / f"breadth_vs_depth.{ext}"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parser.parse_args()
    panels = args.panels if args.panels else list(PANEL_CHOICES)
    panels_set = set(panels)

    ext = "pdf" if args.pdf else "png"
    # if args.pdf:
    #     apply_paper_rc()
    #     global _FS_SCALE
    #     _FS_SCALE = 0.60   # 12 → 7.2, 13 → 7.8, 15 → 9 etc.
    
    print(f"Loading DB : {DB}")
    model_sigs = load_model_sigs(DB)
    sig_op     = load_sig_op_name(DB)
    sig_times  = load_sig_times(CSV_V2)
    print(f"  {len(model_sigs)} (model, backend) entries, "
          f"{len(sig_op)} sigs, {len(sig_times)} timing rows")

    backends = sorted({b for (m, b) in model_sigs})

    # --- cross-backend aggregation (union of sigs per model; sum T_sig across backends)
    merged_sigs: dict[str, set[str]] = defaultdict(set)
    for (model, backend), sigs in model_sigs.items():
        merged_sigs[model] |= sigs
    models_all = sorted(merged_sigs.keys())
    N_MODELS = len(models_all)

    sig_models: dict[str, set[str]] = defaultdict(set)
    for (model, backend), sigs in model_sigs.items():
        for s in sigs:
            sig_models[s].add(model)

    sig_total_time: dict[str, float] = {}
    for s in sig_models:
        sig_total_time[s] = sum(sig_times.get((s, b), 0.0) for b in backends)

    # If --naive-mode=extrapolated, build per-(model, backend, sig) estimates
    # so naive can be summed cell-by-cell (each model gets its own measurement
    # at its own native cap, or a rate-extrapolated proxy when it deduped at
    # profile time to a shorter-cap rep).
    naive_per_msb: dict[tuple[str, str, str], float] = {}
    extrapolation_report: dict[str, int] = {}
    if args.naive_mode == "direct":
        naive_per_msb, direct_report = load_naive_direct(CSV_V2, model_sigs)
        cov_total = direct_report['direct'] + direct_report['missing']
        print(f"  naive-mode=direct: "
              f"{direct_report['direct']} direct, "
              f"{direct_report['missing']} no-CSV-row "
              f"(out of {cov_total} cells; CSV had {direct_report['csv_rows_seen']} usable rows)")
    elif args.naive_mode == "extrapolated":
        donors = load_csv_donors(CSV_V2)
        model_caps = load_model_caps(models_all)
        naive_per_msb, extrapolation_report = estimate_naive_per_msb(
            model_sigs, donors, model_caps, sig_op)
        cov_total = sum(extrapolation_report.values())
        print(f"  naive-mode=extrapolated: "
              f"{extrapolation_report['direct']} direct, "
              f"{extrapolation_report.get('same_cap_peer', 0)} same-cap-peer, "
              f"{extrapolation_report['extrapolated']} ratio-extrapolated, "
              f"{extrapolation_report['no_ref_fallback']} rate-fallback, "
              f"{extrapolation_report['no_donor']} no-donor "
              f"(out of {cov_total} cells)")
        cap_str = ", ".join(f"{short_name(m)}={model_caps[m]}" for m in models_all)
        print(f"  model native caps: {cap_str}")

    # Colors shared across the two plots, keyed by op class
    op_classes = sorted({op_class(sig_op.get(s, "Other")) for s in sig_models})
    palette = plt.get_cmap("tab10")
    oc_color = {oc: palette(i) for i, oc in enumerate(op_classes)}

    # Per-class dedup / naive bookkeeping (for breadth_vs_depth)
    class_naive = defaultdict(float)
    class_dedup = defaultdict(float)
    class_sigs  = defaultdict(int)
    for s, ms in sig_models.items():
        oc = op_class(sig_op.get(s, "Other"))
        t = sig_total_time[s]
        class_dedup[oc] += t
        if args.naive_mode == "max":
            # Legacy: every model in `ms` charged the max-shard wall_time for sig s
            class_naive[oc] += t * len(ms)
        # else: extrapolated naive is summed in the per-(model, backend, sig)
        # loop below so we don't double-count here.
        class_sigs[oc]  += 1

    if args.naive_mode in ("direct", "extrapolated"):
        # Sum naive_per_msb into per-class buckets, summing across backends.
        for (model, backend, sig), t in naive_per_msb.items():
            oc = op_class(sig_op.get(sig, "Other"))
            class_naive[oc] += t

    class_saved  = {c: class_naive[c] - class_dedup[c] for c in class_naive}
    class_reduct = {c: (100 * class_saved[c] / class_naive[c]) if class_naive[c] else 0
                    for c in class_naive}

    # Per-backend totals (for the headline banner and coverage plot annotations)
    backend_dedup = defaultdict(float)
    backend_naive = defaultdict(float)
    # Dedup per-backend: sum over unique (sig, backend) of measured T_sig.
    # Same in both modes — dedup is "what we actually paid in our run".
    for (s, b), t in sig_times.items():
        backend_dedup[b] += t
    # Naive per-backend: depends on mode.
    if args.naive_mode == "max":
        # Legacy: each (model, backend, sig) charged the max-shard wall_time.
        for (m, b), sigs in model_sigs.items():
            for s in sigs:
                backend_naive[b] += sig_times.get((s, b), 0.0)
    else:
        # direct or extrapolated: each (model, backend, sig) charged its own
        # measurement (direct CSV row, or rate-extrapolated estimate when
        # missing in extrapolated mode).
        for (m, b, s), t in naive_per_msb.items():
            backend_naive[b] += t

    # ---------------------------------------------------------------------
    # (1) BREADTH vs DEPTH
    # ---------------------------------------------------------------------
    if args.skip_breadth:
        print(f"\n(1) breadth_vs_depth.{ext}  [SKIPPED]")
    else:
        print(f"\n(1) breadth_vs_depth.{ext}")
        _render_breadth(class_naive, class_dedup, class_saved, class_reduct,
                        class_sigs, backend_dedup, backend_naive, oc_color,
                        ext=ext)

    # ---------------------------------------------------------------------
    # (2) CUMULATIVE COVERAGE
    # ---------------------------------------------------------------------
    print(f"(2) cumulative_coverage.{ext}")
    all_sigs_universe = set(sig_models.keys())
    all_time_universe = sum(sig_total_time.values())

    # Aggregate greedy coverage
    covered_sigs: set[str] = set()
    covered_time: float = 0.0
    pick_order: list[tuple[str, int, float]] = []
    remaining = set(models_all)
    xs = [0]; ys_sigs_pct = [0.0]; ys_time_pct = [0.0]
    while remaining:
        best = max(remaining,
                   key=lambda m: len(merged_sigs[m] - covered_sigs))
        new_sigs = merged_sigs[best] - covered_sigs
        new_time = sum(sig_total_time[s] for s in new_sigs)
        covered_sigs |= new_sigs
        covered_time += new_time
        pick_order.append((best, len(new_sigs), new_time))
        remaining.remove(best)
        xs.append(len(pick_order))
        ys_sigs_pct.append(100.0 * len(covered_sigs) / len(all_sigs_universe))
        ys_time_pct.append(100.0 * covered_time / all_time_universe if all_time_universe else 0)

    # Also compute per-backend greedy curves (in the SAME model ordering
    # as the aggregate, so each step is comparable across the three lines).
    per_backend_sig_curves = {}   # backend -> [pct covered over steps]
    per_backend_time_curves = {}
    for bk in backends:
        # universe for this backend
        bk_sigs = {s for s, ms in sig_models.items()
                   if any((m, bk) in model_sigs and s in model_sigs[(m, bk)]
                          for m in ms)}
        bk_time_total = sum(sig_times.get((s, bk), 0.0) for s in bk_sigs)
        cov_s = set()
        cov_t = 0.0
        pct_s = [0.0]
        pct_t = [0.0]
        for (m, _, _) in pick_order:
            new_s = {s for s in model_sigs.get((m, bk), set()) if s not in cov_s}
            new_t = sum(sig_times.get((s, bk), 0.0) for s in new_s)
            cov_s |= new_s
            cov_t += new_t
            pct_s.append(100.0 * len(cov_s) / len(bk_sigs) if bk_sigs else 0)
            pct_t.append(100.0 * cov_t / bk_time_total if bk_time_total else 0)
        per_backend_sig_curves[bk] = pct_s
        per_backend_time_curves[bk] = pct_t

    model_lbls = [short_name(m) for m, _, _ in pick_order]
    model_colors = [family_color(m) for m, _, _ in pick_order]
    marg_sigs = [ns for _, ns, _ in pick_order]

    # Also break the marginal contribution down by backend so the bars can
    # be stacked per-backend. For each step, count the NEW sigs / time that
    # this model adds per backend (aggregate "new sigs" across backends sums
    # to marg_sigs[i], and likewise for time).
    #
    # Time contribution: in 'direct'/'extrapolated' modes we charge the FIRST
    # encountering model its OWN measurement (via naive_per_msb). This makes
    # dedup_cum_h[i] <= naive_cum_h[i] at every step, because dedup only
    # avoids re-profiling shared sigs — it doesn't change a single model's
    # cost. In 'max' mode (legacy) we keep using sig_times (max-shard) for
    # backward-compatible numbers.
    per_backend_marg_sigs = {b: [] for b in backends}
    per_backend_marg_time = {b: [] for b in backends}
    covered_bk: dict[str, set[str]] = {b: set() for b in backends}
    for (m, _, _) in pick_order:
        for b in backends:
            model_bk_sigs = model_sigs.get((m, b), set())
            new_bk = model_bk_sigs - covered_bk[b]
            if args.naive_mode == "max":
                new_bk_time = sum(sig_times.get((s, b), 0.0) for s in new_bk)
            else:
                # direct or extrapolated: charge the first encountering model
                # its own measurement, matching how naive_per_model is built.
                new_bk_time = sum(naive_per_msb.get((m, b, s), 0.0) for s in new_bk)
            per_backend_marg_sigs[b].append(len(new_bk))
            per_backend_marg_time[b].append(new_bk_time / 3600.0)
            covered_bk[b] |= new_bk

    # marg_time: per-pick dedup contribution in hours, summed across backends.
    # Computed from per_backend_marg_time (which tracks (sig, backend) coverage
    # correctly). Earlier this used pick_order's nt = sig_total_time-based sum,
    # which over-counted cross-backend contributions for the first pick of any
    # sig (it charged a model for backend measurements it doesn't even use).
    marg_time = [sum(per_backend_marg_time[b][i] for b in backends)
                 for i in range(len(pick_order))]

    # --- Cost curves: naive (linear) vs dedup (concave saturation) ---
    # Computed unconditionally so the "cost" panel can be plotted alone.
    # Dedup cum-cost: sum the per-pick contributions at (sig, backend)
    # granularity. Guarantees dedup_cum_h[i] <= naive_cum_h[i] for all i.
    dedup_cum_h = [0.0]
    for i in range(len(pick_order)):
        dedup_cum_h.append(dedup_cum_h[-1] + marg_time[i])
    naive_per_model = {}
    for m in models_all:
        total = 0.0
        for b in backends:
            for s in model_sigs.get((m, b), set()):
                if args.naive_mode == "max":
                    total += sig_times.get((s, b), 0.0)
                else:  # direct or extrapolated
                    total += naive_per_msb.get((m, b, s), 0.0)
        naive_per_model[m] = total / 3600.0
    naive_cum_h = [0.0]
    for (m, _, _) in pick_order:
        naive_cum_h.append(naive_cum_h[-1] + naive_per_model[m])

    xbar = np.arange(1, len(pick_order) + 1)
    backend_styles = {"FLASHINFER": ":", "flashinfer": ":", "FLASH_ATTN": "--", "fa3": "--","TRITON_ATTN": "-.", "triton": "-."}
    blue_shades  = {"FLASHINFER": "#08519c", "flashinfer": "#08519c", "FLASH_ATTN": "#3182bd", "fa3": "#3182bd", "TRITON_ATTN": "#6baed6", "triton": "#6baed6"}
    orange_shades = {"FLASHINFER": "#8c2d04", "flashinfer": "#8c2d04", "FLASH_ATTN": "#d94801", "fa3": "#d94801", "TRITON_ATTN": "#fd8d3c", "triton": "#fd8d3c"}

    # Allocate subplots for the requested panels
    n_panels = len(panels)
    # if args.pdf:
    #     # Paper-ready: 7in wide for 1x3 two-column span, ~3.3in for single panel
    #     per_panel_w = 7.0 / 3 if n_panels > 1 else 3.3
    #     fig_height  = 3.0 if n_panels > 1 else 2.6
    # else:
    per_panel_w = 6.0 if n_panels > 1 else 6.5 * PNG_SCALE
    fig_height  = 4.6 if n_panels > 1 else 4.0 * PNG_SCALE
    fig, axes = plt.subplots(
        1, n_panels, figsize=(per_panel_w * n_panels, fig_height),
        gridspec_kw={"wspace": 0.12} if n_panels > 1 else None,
        squeeze=False,
    )
    axes = axes.flatten().tolist()
    ax_by_name = {name: axes[i] for i, name in enumerate(panels)}

    # ---- Coverage panel ----
    if "coverage" in panels_set:
        ax_cov = ax_by_name["coverage"]
        # for bk in backends:
        #     ax_cov.plot(xs, per_backend_sig_curves[bk],
        #                 color="#0173b2", alpha=0.45, linewidth=1.0,
        #                 linestyle=backend_styles.get(bk, ":"))
        #     ax_cov.plot(xs, per_backend_time_curves[bk],
        #                 color="#d55e00", alpha=0.45, linewidth=1.0,
        #                 linestyle=backend_styles.get(bk, ":"))
        ax_cov.fill_between(xs, ys_sigs_pct, alpha=0.15, color="#0173b2")
        ax_cov.plot(xs, ys_sigs_pct, marker="o", color="#0173b2",
                    linewidth=2.5, label="Unique signatures")
        ax_cov.fill_between(xs, ys_time_pct, alpha=0.15, color="#d55e00")
        ax_cov.plot(xs, ys_time_pct, marker="s", color="#d55e00",
                    linewidth=2.5, linestyle="--", label="Total profiling time")
        style_handles = [
            # plt.Line2D([], [], color="#888", linestyle=backend_styles[bk],
            #            linewidth=1.2, label=bk)
            # for bk in backends
        ]
        for level in (50, 90, 99):
            ax_cov.axhline(y=level, color="gray", linestyle=":",
                           linewidth=0.8, alpha=0.6)
        ax_cov.set_ylim(0, 101)
        ax_cov.set_ylabel("Cumulative Coverage (%)", fontsize=fs(15), fontweight="semibold")
        # ax_cov.set_title("Coverage (Greedy Ordering)", fontsize=fs(13))
        ax_cov.tick_params(axis="y", labelsize=fs(12))
        agg_handles, agg_labels = ax_cov.get_legend_handles_labels()
        merged_handles = agg_handles + style_handles
        merged_labels  = agg_labels + [h.get_label() for h in style_handles]
        ax_cov.legend(merged_handles, merged_labels,
                      loc="lower right", fontsize=fs(15), framealpha=0.9, ncol=1)
        ax_cov.grid(axis="y", linestyle="--", alpha=0.3)
        ax_cov.set_xticks(xbar)
        ax_cov.set_xticklabels(model_lbls, rotation=35, ha="right", fontsize=fs(15))
        for tick_lbl, col in zip(ax_cov.get_xticklabels(), model_colors):
            tick_lbl.set_color(col)

    # ---- Cost panel ----
    if "cost" in panels_set:
        ax_cost = ax_by_name["cost"]
        # ax_cost.fill_between(xs, dedup_cum_h, naive_cum_h, alpha=0.22,
        #                      color="#81c784", label="Profiling Overhead Reduction")
        ax_cost.fill_between(xs, dedup_cum_h, naive_cum_h, alpha=0.22,
                             color="#81c784")
        ax_cost.plot(xs, naive_cum_h, marker="s", color="#d55e00",
                     linewidth=2.3, linestyle="-",
                     label=f"Naive: {naive_cum_h[-1]:.1f}h")
        ax_cost.plot(xs, dedup_cum_h, marker="o", color="#0173b2",
                     linewidth=2.3, linestyle="-",
                     label=f"Dooly: {dedup_cum_h[-1]:.1f}h")
        total_saved_abs = naive_cum_h[-1] - dedup_cum_h[-1]
        ax_cost.annotate(
            f"Saved {total_saved_abs:.1f} hr",
            xy=(xs[-1], (naive_cum_h[-1] + dedup_cum_h[-1]) / 2),
            xytext=(xs[-1] - 5.3, (naive_cum_h[-1] + dedup_cum_h[-1]) / 2 - 5),
            fontsize=fs(15), color="#2e7d32", fontweight="semibold")
            # arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.3))
        ax_cost.set_ylabel("GPU Hours", fontsize=fs(15), fontweight="semibold")
        # ax_cost.set_title("Dooly vs Naive Profiling", fontsize=fs(13))
        ax_cost.tick_params(axis="y", labelsize=fs(12))
        ax_cost.legend(loc="upper left", fontsize=fs(15), framealpha=0.9)
        ax_cost.grid(axis="y", linestyle="--", alpha=0.3)
        ax_cost.set_xticks(xbar)
        ax_cost.set_xticklabels(model_lbls, rotation=35, ha="right", fontsize=fs(15))
        for tick_lbl, col in zip(ax_cost.get_xticklabels(), model_colors):
            tick_lbl.set_color(col)

    # ---- Marginal panel ----
    if "marginal" in panels_set:
        ax_marg = ax_by_name["marginal"]
        width = 0.38
        bot = np.zeros(len(pick_order))
        for b in backends:
            heights = np.array(per_backend_marg_sigs[b])
            ax_marg.bar(xbar - width/2, heights, width, bottom=bot,
                        color=blue_shades.get(b, "#0173b2"),
                        edgecolor="white", linewidth=0.5)
            bot += heights
        ax_marg2 = ax_marg.twinx()
        bot = np.zeros(len(pick_order))
        for b in backends:
            heights = np.array(per_backend_marg_time[b])
            ax_marg2.bar(xbar + width/2, heights, width, bottom=bot,
                         color=orange_shades.get(b, "#d55e00"),
                         edgecolor="white", linewidth=0.5)
            bot += heights
        ax_marg.set_xticks(xbar)
        ax_marg.set_xticklabels(model_lbls, rotation=35, ha="right", fontsize=fs(12))
        for tick_lbl, col in zip(ax_marg.get_xticklabels(), model_colors):
            tick_lbl.set_color(col)
        ax_marg.set_ylabel("New Op Count", fontsize=fs(12), color="#08519c", fontweight="semibold")
        ax_marg.tick_params(axis="y", labelcolor="#08519c", labelsize=fs(12))
        ax_marg2.set_ylabel("New Cost (h)", fontsize=fs(12), color="#d94801", fontweight="semibold")
        ax_marg2.tick_params(axis="y", labelcolor="#d94801", labelsize=fs(12))
        # ax_marg.set_title("Marginal Contribution per Model", fontsize=fs(13))
        ax_marg.grid(axis="y", linestyle="--", alpha=0.3)

        from matplotlib.patches import Rectangle
        class _PairedSwatch:
            def __init__(self, c1, c2):
                self.c1, self.c2 = c1, c2
        class _PairedHandler:
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                x0, y0 = handlebox.xdescent, handlebox.ydescent
                w, h = handlebox.width, handlebox.height
                r1 = Rectangle((x0, y0), w / 2, h,
                               facecolor=orig_handle.c1, edgecolor="white")
                r2 = Rectangle((x0 + w / 2, y0), w / 2, h,
                               facecolor=orig_handle.c2, edgecolor="white")
                handlebox.add_artist(r1)
                handlebox.add_artist(r2)
                return r1
        legend_handles = [_PairedSwatch(blue_shades[b], orange_shades[b]) for b in backends]
        legend_labels  = [b for b in backends]
        ax_marg.legend(legend_handles, legend_labels,
                       handler_map={_PairedSwatch: _PairedHandler()},
                       loc="upper right", fontsize=fs(12), framealpha=0.9)

        for i, (ns, nt) in enumerate(zip(marg_sigs, marg_time)):
            if ns == 0 and nt == 0:
                ax_marg.text(i + 1, 0.5, "~",
                             ha="center", va="bottom", fontsize=fs(15), color="black")

    fig.tight_layout()
    if n_panels > 1:
        fig.subplots_adjust(wspace=0.10)

    if panels == list(PANEL_CHOICES):
        out_name = f"cumulative_coverage.{ext}"
    else:
        out_name = f"cumulative_coverage_{'_'.join(panels)}.{ext}"
    fig.savefig(OUT_DIR / out_name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_name}")

    print(f"\nDone. Plots in {OUT_DIR}/")


if __name__ == "__main__":
    main()
