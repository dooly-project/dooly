"""Export per-op-class dedup savings to CSV.

Naive accounting modes (--naive-mode):
  max         (legacy):  naive[op] = Σ_sig (T_sig × |models_using_sig|)
                         dedup[op] = Σ_sig T_sig
                         where T_sig = Σ_b max(wall_time_s for (s, b) in CSV)
                         Over-bounds when donors span cap classes.
  direct                 naive[op] = Σ_(m,b,s) wall_time_s with direct CSV row
                         Skips dedup-skipped cells (correct for uniform-cap runs).
  extrapolated           Same as direct, plus same-cap-peer / ratio-extrapolation
                         to fill in dedup-skipped cells (correct for multi-shard).

For 'direct' / 'extrapolated', dedup[op] uses greedy first-encountered ordering
(same as analyze_overlap.py cost-panel dedup_cum_h): for each unique (sig, b),
attribute the wall_time of the model that first encounters it in the greedy
pick order, and sum those within the op-class bucket.

Output columns (one row per bucket):
    group, operation, n_sigs, n_configs, n_reuses, avg_sharing,
    pct_op_reduction, pct_of_total_saved,
    gpu_hours_saved, gpu_hours_profiled
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from analyze_overlap import (
    DB, CSV_V2, SHORT_NAME_MAP,
    load_model_sigs, load_sig_op_name, load_sig_times,
    load_naive_direct, load_csv_donors, load_model_caps,
    estimate_naive_per_msb,
    op_class,
)


def _short(model: str) -> str:
    return SHORT_NAME_MAP.get(model, model.split("/")[-1])


def load_sig_module(csv_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            sig = row.get("signature_hash", "")
            mod = row.get("module_name", "")
            if sig and mod and sig not in out:
                out[sig] = mod
    return out


# ---------------------------------------------------------------------------
# Attention architecture labels (derived from user set)
# ---------------------------------------------------------------------------

_CR7B = "CohereLabs/c4ai-command-r7b-12-2024"
_MIN8 = "mistralai/Ministral-8B-Instruct-2410"
_L2   = "meta-llama/Llama-2-7b-hf"
_DSLM = "deepseek-ai/deepseek-llm-7b-base"
_Q25  = "Qwen/Qwen2.5-7B"
_DSQ  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def attention_arch_label(users: frozenset[str]) -> str:
    """Architecture-aware label for an Attention user-set group."""
    n = len(users)
    if n == 1 and _CR7B in users:
        return "Attention (SWA window=4096, CR-7B)"
    if n == 1 and _MIN8 in users:
        return "Attention (SWA window=32K, Min-8B)"
    if {_L2, _DSLM}.issubset(users) and n == 2:
        return "Attention (MHA 32/32/128, Llama-2/DS-LLM)"
    if {_Q25, _DSQ}.issubset(users) and n == 2:
        return "Attention (GQA 28/4/128, Qwen family)"
    if n >= 5:
        return f"Attention (GQA 32/8/128, {n} models)"
    # Fallback: auto-generated short-name list
    short = sorted(_short(u) for u in users)
    if n == 1:
        return f"Attention (unique: {short[0]})"
    return f"Attention (×{n}: {', '.join(short)})"


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def _parent_of(oc: str) -> str:
    """Parent op-class grouping. Used only for sectioned output."""
    if oc == "Linear (aten::linear)":
        return "Linear (aten::linear)"
    return oc


def _bucket_for(sig, sig_op_map, sig_module_map, sig_users_map,
                split_linear, split_attention, split_all):
    """Return (parent_op_class, bucket_key, label) for this signature."""
    oc = op_class(sig_op_map.get(sig, "Other"))
    mod = sig_module_map.get(sig, "")
    users = sig_users_map[sig]
    parent = _parent_of(oc)

    if split_all:
        if oc == "Attention":
            label = attention_arch_label(frozenset(users))
            key = ("Attention", frozenset(users))
        else:
            short_users = ", ".join(sorted(_short(u) for u in users))
            display_users = short_users if len(short_users) <= 40 else short_users[:37] + "…"
            tag = mod if mod else oc
            label = f"{tag} [×{len(users)}: {display_users}]"
            key = (oc, mod, frozenset(users))
        return parent, key, label

    if split_attention and oc == "Attention":
        key = ("Attention", frozenset(users))
        return parent, key, attention_arch_label(frozenset(users))

    if split_linear and oc == "Linear (aten::linear)" and mod:
        key = ("Linear", mod)
        return parent, key, mod

    # No split for this sig: bucket is just op-class.
    return parent, (oc,), oc


def _build_naive_per_msb(naive_mode: str, model_sigs, sig_op_map,
                         csv_path: Path, db_path: Path):
    """Build the {(model, backend, sig): wall_time} dict for the chosen mode.

    For 'max' returns None (caller falls back to legacy T_sig × |users| formula).
    """
    if naive_mode == "max":
        return None, {}
    if naive_mode == "direct":
        out, rep = load_naive_direct(csv_path, model_sigs)
        return out, rep
    if naive_mode == "extrapolated":
        donors = load_csv_donors(csv_path)
        all_models = sorted({m for (m, _) in model_sigs})
        caps = load_model_caps(all_models)
        out, rep = estimate_naive_per_msb(model_sigs, donors, caps, sig_op_map)
        return out, rep
    raise ValueError(f"unknown naive_mode: {naive_mode}")


def _build_dedup_per_sb(model_sigs, naive_per_msb, backends):
    """Greedy first-encountered dedup: for each unique (sig, backend), attribute
    the wall_time of the model that first encounters it in greedy pick order.

    Mirrors analyze_overlap.py cost-panel dedup_cum_h logic. Used in
    'direct' / 'extrapolated' modes.
    """
    merged_sigs: dict[str, set[str]] = defaultdict(set)
    for (m, b), sigs in model_sigs.items():
        merged_sigs[m] |= sigs
    all_models = sorted(merged_sigs.keys())

    covered_sigs: set[str] = set()
    pick_order: list[str] = []
    remaining = set(all_models)
    while remaining:
        best = max(remaining, key=lambda m: len(merged_sigs[m] - covered_sigs))
        new_sigs = merged_sigs[best] - covered_sigs
        covered_sigs |= new_sigs
        pick_order.append(best)
        remaining.remove(best)

    dedup_per_sb: dict[tuple[str, str], float] = {}
    covered_bk: dict[str, set[str]] = {b: set() for b in backends}
    for m in pick_order:
        for b in backends:
            new_bk = model_sigs.get((m, b), set()) - covered_bk[b]
            for s in new_bk:
                dedup_per_sb[(s, b)] = naive_per_msb.get((m, b, s), 0.0)
                covered_bk[b].add(s)
    return dedup_per_sb


def compute_op_savings(db_path: Path, csv_path: Path, *,
                       split_linear: bool = False,
                       split_attention: bool = False,
                       split_all: bool = False,
                       naive_mode: str = "direct"):
    model_sigs = load_model_sigs(db_path)
    sig_op_map = load_sig_op_name(db_path)
    sig_times  = load_sig_times(csv_path)
    sig_module_map = load_sig_module(csv_path)

    backends = sorted({b for (_, b) in model_sigs})

    sig_users_map: dict[str, set[str]] = defaultdict(set)
    # sig_configs_map[s] = set of (model, backend) pairs using sig s.
    # Different from sig_users_map because many sigs are shared across backends
    # (aten::linear, _C::rms_norm, etc. have backend-independent kernels).
    sig_configs_map: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for (model, backend), sigs in model_sigs.items():
        for s in sigs:
            sig_users_map[s].add(model)
            sig_configs_map[s].add((model, backend))

    sig_total_time: dict[str, float] = {}
    for s in sig_users_map:
        sig_total_time[s] = sum(sig_times.get((s, b), 0.0) for b in backends)

    # Build per-(m, b, s) wall_time dict and per-(s, b) dedup dict for
    # 'direct' / 'extrapolated' modes. 'max' mode falls back to T_sig × |users|.
    naive_per_msb, naive_rep = _build_naive_per_msb(
        naive_mode, model_sigs, sig_op_map, csv_path, db_path)
    if naive_per_msb is not None:
        dedup_per_sb = _build_dedup_per_sb(model_sigs, naive_per_msb, backends)
        print(f"[export_op_savings] naive-mode={naive_mode} report: {naive_rep}")
    else:
        dedup_per_sb = None
        print(f"[export_op_savings] naive-mode={naive_mode} (legacy T_sig × |users|)")

    # parent -> list of (bucket_key, label, list_of_sigs)
    parents: dict[str, dict[tuple, tuple[str, list[str]]]] = defaultdict(dict)
    for sig in sig_users_map:
        parent, key, label = _bucket_for(
            sig, sig_op_map, sig_module_map, sig_users_map,
            split_linear, split_attention, split_all,
        )
        entry = parents[parent].setdefault(key, (label, []))
        entry[1].append(sig)

    def _agg(sigs: list[str]):
        if naive_per_msb is not None:
            # direct / extrapolated mode: per-(model, backend, sig) accounting.
            # naive: sum over (m, b) cells using each sig of THIS model's measurement.
            # dedup: sum over (sig, backend) cells of greedy first-encountered cost.
            naive = 0.0
            for s in sigs:
                for (m, b) in sig_configs_map[s]:
                    naive += naive_per_msb.get((m, b, s), 0.0)
            dedup = 0.0
            for s in sigs:
                for b in backends:
                    if (s, b) in dedup_per_sb:
                        dedup += dedup_per_sb[(s, b)]
        else:
            # 'max' mode (legacy): T_sig × |models_using_sig|.
            naive = sum(sig_total_time[s] * len(sig_users_map[s]) for s in sigs)
            dedup = sum(sig_total_time[s] for s in sigs)
        saved = naive - dedup
        n_sigs = len(sigs)
        avg_sh = sum(len(sig_users_map[s]) for s in sigs) / n_sigs if n_sigs else 0.0
        # n_configs: total (model, backend) configurations that invoke any sig
        # in the bucket. Accounts for cross-backend sig reuse (a single sig
        # hash can be used by the same model across multiple backends when the
        # kernel is backend-independent — e.g., aten::linear).
        n_configs = sum(len(sig_configs_map[s]) for s in sigs)
        n_reuses = max(0, n_configs - n_sigs)
        return dict(naive=naive, dedup=dedup, saved=saved,
                    n_sigs=n_sigs, avg_sharing=avg_sh,
                    n_configs=n_configs, n_reuses=n_reuses)

    # Total saved across everything, for pct_of_total calculation
    all_saved = sum(_agg(sigs)["saved"]
                    for buckets in parents.values()
                    for (_label, sigs) in buckets.values())

    # Emit: per parent, aggregate row first (if it has > 1 child), then children.
    rows = []
    # Order parents by aggregate saved time, descending.
    parent_order = sorted(
        parents.keys(),
        key=lambda p: -sum(_agg(sigs)["saved"]
                           for (_label, sigs) in parents[p].values()),
    )
    for parent in parent_order:
        buckets = parents[parent]
        # Aggregate over all children of this parent.
        all_sigs_in_parent = [s for (_label, sigs) in buckets.values() for s in sigs]
        agg = _agg(all_sigs_in_parent)
        multi_child = len(buckets) > 1

        def _to_row(label: str, data: dict, is_group: bool) -> dict:
            pct_op    = (100.0 * data["saved"] / data["naive"]) if data["naive"] else 0.0
            pct_total = (100.0 * data["saved"] / all_saved) if all_saved else 0.0
            return {
                "group":              parent,
                "operation":          label,
                "n_sigs":             data["n_sigs"],
                "n_configs":          data["n_configs"],
                "n_reuses":           data["n_reuses"],
                "avg_sharing":        round(data["avg_sharing"], 2),
                "pct_op_reduction":   round(pct_op, 2),
                "pct_of_total_saved": round(pct_total, 2),
                "gpu_hours_saved":    round(data["saved"] / 3600.0, 4),
                "gpu_hours_profiled": round(data["dedup"] / 3600.0, 4),
            }

        if multi_child:
            # Group header row with aggregate stats.
            rows.append(_to_row(parent, agg, is_group=True))
            # Children sorted by saved time, descending.
            child_items = sorted(
                buckets.items(),
                key=lambda kv: -_agg(kv[1][1])["saved"],
            )
            for _key, (label, sigs) in child_items:
                child_agg = _agg(sigs)
                # Indent child label by two spaces for visual grouping.
                rows.append(_to_row("  " + label, child_agg, is_group=False))
        else:
            # Single-bucket parent: just emit the one row (no indent).
            (_key, (label, sigs)), = buckets.items()
            rows.append(_to_row(label, _agg(sigs), is_group=False))

    return rows


def _collapse_for_compact(rows: list[dict]) -> list[dict]:
    """Collapse small non-Attention / non-Linear classes into 'Other operators'."""
    keep_groups = {"Attention", "Linear (aten::linear)"}
    kept = [r for r in rows if r["group"] in keep_groups]
    others = [r for r in rows if r["group"] not in keep_groups]
    if not others:
        return kept

    # Sum across the collapsed rows. Average-sharing is weight-by-sigs.
    total_sigs     = sum(r["n_sigs"] for r in others)
    total_configs  = sum(r["n_configs"] for r in others)
    total_reuses   = sum(r["n_reuses"] for r in others)
    total_prof     = sum(r["gpu_hours_profiled"] for r in others)
    total_saved    = sum(r["gpu_hours_saved"] for r in others)
    total_pct_tot  = sum(r["pct_of_total_saved"] for r in others)
    # Reduction % recomputed from the totals (avoid averaging percentages).
    naive_total    = total_prof + total_saved
    pct_op         = 100.0 * total_saved / naive_total if naive_total else 0.0
    avg_sh         = (sum(r["avg_sharing"] * r["n_sigs"] for r in others)
                      / total_sigs) if total_sigs else 0.0

    merged = {
        "group":              "Other operators",
        "operation":          "Other operators (Norm, RoPE, Embedding, Activation, Sampler, …)",
        "n_sigs":             total_sigs,
        "n_configs":          total_configs,
        "n_reuses":           total_reuses,
        "avg_sharing":        round(avg_sh, 2),
        "pct_op_reduction":   round(pct_op, 2),
        "pct_of_total_saved": round(total_pct_tot, 2),
        "gpu_hours_saved":    round(total_saved, 4),
        "gpu_hours_profiled": round(total_prof, 4),
    }
    return kept + [merged]


def _tex_escape(s: str) -> str:
    return (s.replace("&", r"\&")
             .replace("%", r"\%")
             .replace("_", r"\_")
             .replace("×", r"$\times$"))


def _render_latex(rows: list[dict], compact: bool = False) -> str:
    """Render a booktabs tabular with merged group column via \\multirow.

    When `compact` is True, drops the '% of saved' column for a tighter table.
    """
    # Partition rows by group; detect which groups have multiple children.
    from collections import OrderedDict
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in rows:
        groups.setdefault(r["group"], []).append(r)

    # Totals (sum over bottom-most rows: children where present, else parent)
    total_reuses  = 0
    total_prof    = 0.0
    total_saved   = 0.0
    for gname, grows in groups.items():
        children = [r for r in grows if r["operation"].startswith("  ")]
        if children:
            for c in children:
                total_reuses += c["n_reuses"]
                total_prof   += c["gpu_hours_profiled"]
                total_saved  += c["gpu_hours_saved"]
        else:
            r = grows[0]
            total_reuses += r["n_reuses"]
            total_prof   += r["gpu_hours_profiled"]
            total_saved  += r["gpu_hours_saved"]

    lines = []
    col_spec = r"@{}llrrrr@{}" if compact else r"@{}llrrrrr@{}"
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    if compact:
        lines.append(
            r"Group & Variant & \# reuses & Profile (h) & Saved (h) & "
            r"Reduction \\"
        )
    else:
        lines.append(
            r"Group & Variant & \# reuses & Profile (h) & Saved (h) & "
            r"Reduction & \% of saved \\"
        )
    lines.append(r"\midrule")

    def _strip_group_prefix(label: str, gname: str) -> str:
        """Remove redundant 'Attention (' / 'Linear' prefix from child labels
        since the Group column already carries that context."""
        if gname == "Attention" and label.startswith("Attention ("):
            return label[len("Attention ("):].rstrip(")")
        return label

    # Partition: multi-child groups first, single-row groups collected at the end.
    multi_groups = [(g, rs) for g, rs in groups.items() if len(rs) > 1]
    single_groups = [(g, rs) for g, rs in groups.items() if len(rs) == 1]

    first = True
    for gname, grows in multi_groups:
        if not first:
            lines.append(r"\midrule")
        first = False
        header = grows[0]
        children = grows[1:]
        tail_agg = "" if compact else rf" & {header['pct_of_total_saved']:.1f}\%"
        lines.append(
            rf"\multirow{{{len(children) + 1}}}{{*}}{{"
            rf"\textbf{{{_tex_escape(gname)}}}}} "
            rf"& \textit{{(aggregate)}} & {header['n_reuses']} & "
            rf"{header['gpu_hours_profiled']:.2f} & "
            rf"{header['gpu_hours_saved']:.2f} & "
            rf"{header['pct_op_reduction']:.1f}\%{tail_agg} \\"
        )
        for c in children:
            label = _strip_group_prefix(c["operation"].lstrip(), gname)
            tail = "" if compact else rf" & {c['pct_of_total_saved']:.1f}\%"
            lines.append(
                rf" & {_tex_escape(label)} & {c['n_reuses']} & "
                rf"{c['gpu_hours_profiled']:.2f} & "
                rf"{c['gpu_hours_saved']:.2f} & "
                rf"{c['pct_op_reduction']:.1f}\%{tail} \\"
            )

    # Single-row groups: one \midrule to open the block, no rules between.
    if single_groups:
        if multi_groups:
            lines.append(r"\midrule")
        for gname, grows in single_groups:
            r = grows[0]
            tail = "" if compact else rf" & {r['pct_of_total_saved']:.1f}\%"
            lines.append(
                rf"\multicolumn{{2}}{{@{{}}l}}{{{_tex_escape(gname)}}} & "
                rf"{r['n_reuses']} & {r['gpu_hours_profiled']:.2f} & "
                rf"{r['gpu_hours_saved']:.2f} & "
                rf"{r['pct_op_reduction']:.1f}\%{tail} \\"
            )

    lines.append(r"\midrule")
    total_red = 100.0 * total_saved / (total_prof + total_saved) if (total_prof + total_saved) else 0
    total_tail = "" if compact else r" & \textbf{100.0\%}"
    lines.append(
        rf"\multicolumn{{2}}{{@{{}}l}}{{\textbf{{Total}}}} & "
        rf"\textbf{{{total_reuses}}} & "
        rf"\textbf{{{total_prof:.2f}}} & "
        rf"\textbf{{{total_saved:.2f}}} & "
        rf"\textbf{{{total_red:.1f}\%}}{total_tail} \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def _is_group_header(row: dict, groups: dict) -> bool:
    return row["operation"] == row["group"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--timing-csv", type=Path, default=CSV_V2)
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("/home/jkim38/dooly/comparison/overlap_v2/op_savings.csv"),
                        help="Output CSV path")
    parser.add_argument("--split-linear", action="store_true",
                        help="Break Linear into per-module subclasses, "
                             "with an aggregate Linear row above them.")
    parser.add_argument("--split-attention", action="store_true",
                        help="Break Attention into per-architecture groups, "
                             "with an aggregate Attention row above them.")
    parser.add_argument("--split-all", action="store_true",
                        help="Fully expand every (op, module, user-set) combination.")
    parser.add_argument("--latex", action="store_true",
                        help="Emit a paper-ready LaTeX tabular (booktabs + "
                             "multirow) instead of CSV. Writes to <output>.tex "
                             "if --output still ends in .csv.")
    parser.add_argument("--compact", action="store_true",
                        help="Compact paper table: Attention variants split, "
                             "Linear kept as one aggregate row, all other ops "
                             "collapsed into a single 'Other operators' row. "
                             "Drops the '%% of saved' column. Implies "
                             "--split-attention.")
    parser.add_argument("--naive-mode", choices=["max", "direct", "extrapolated"],
                        default="direct",
                        help="How to compute naive time per cell. "
                             "'direct' (default): sum per-(model, backend, sig) "
                             "wall_time from CSV; cells without a CSV row "
                             "contribute 0. Correct for uniform-cap runs. "
                             "'max' (legacy): naive[op] = Σ T_sig × |users|; "
                             "over-bounds when donors span cap classes. "
                             "'extrapolated': direct + same-cap-peer fallback "
                             "+ ratio scaling for missing cells. Correct for "
                             "multi-shard runs with within-shard dedup.")
    args = parser.parse_args()

    if args.compact:
        # Compact forces: attention split, linear aggregated, others merged.
        args.split_attention = True
        args.split_linear = False
        args.split_all = False

    rows = compute_op_savings(
        args.db, args.timing_csv,
        split_linear=args.split_linear,
        split_attention=args.split_attention,
        split_all=args.split_all,
        naive_mode=args.naive_mode,
    )

    if args.compact:
        rows = _collapse_for_compact(rows)

    fieldnames = ["group", "operation", "n_sigs", "n_configs", "n_reuses",
                  "avg_sharing", "pct_op_reduction", "pct_of_total_saved",
                  "gpu_hours_saved", "gpu_hours_profiled"]

    if args.latex:
        out_path = args.output.with_suffix(".tex")
        tex = _render_latex(rows, compact=args.compact)
        with open(out_path, "w") as f:
            f.write(tex)
        print(f"Wrote LaTeX to {out_path}")
    else:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.output}")
    print()
    print(f"  {'operation':<48s}  {'#sig':>4s}  {'avgShr':>6s}  "
          f"{'%op':>6s}  {'%tot':>6s}  {'prof h':>7s}  {'saved h':>8s}")
    for r in rows:
        op = r["operation"]
        if len(op) > 48:
            op = op[:45] + "..."
        print(f"  {op:<48s}  {r['n_sigs']:>4d}  {r['avg_sharing']:>6.2f}  "
              f"{r['pct_op_reduction']:>5.1f}%  {r['pct_of_total_saved']:>5.1f}%  "
              f"{r['gpu_hours_profiled']:>7.3f}  {r['gpu_hours_saved']:>8.3f}")


if __name__ == "__main__":
    main()
