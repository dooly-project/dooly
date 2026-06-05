"""Plot TTFT or ITL CDFs across multiple configurations.

Given a directory containing per-configuration subdirectories (e.g. tp1/, tp4/),
each holding dooly_*.json and vllm_*.json result files, overlay the CDFs for
the selected configurations on a single plot.
"""

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from doolysim.tools.plot_comparison import extract_metric, load_json, compute_cdf


# Colorblind-friendly palette; extend if more configs are needed.
CONFIG_COLORS = [
    '#0173b2', '#de8f05', '#029e73', '#d55e00',
    '#cc78bc', '#ca9161', '#fbafe4', '#949494',
]

SOURCE_LINESTYLES = {'vllm': '-', 'dooly': '--'}
SOURCE_LABELS = {'vllm': 'vLLM', 'dooly': 'DoolySim'}

METRIC_NAMES = {
    'ttft': 'Time to First Token',
    'itl': 'Time per Output Token',
}

# Batch-trace metrics: user-facing key -> (JSON field, y-axis label)
BATCH_METRICS = {
    'batch_size': ('batch_size', 'Scheduled Requests     '),
    'running': ('running_count', 'Running requests'),
    'waiting': ('waiting_count', 'Waiting requests'),
    'tokens': ('total_tokens_scheduled', 'Tokens scheduled / step'),
}


def find_result_file(config_dir: str, source: str, use_batch: bool) -> Optional[str]:
    """Find the dooly/vllm JSON file inside a configuration directory.

    Streaming results (non-batch) are preferred unless use_batch is True.
    """
    batch_pattern = os.path.join(config_dir, f'{source}_batch_*.json')
    stream_candidates = sorted(
        p for p in glob.glob(os.path.join(config_dir, f'{source}_*.json'))
        if '_batch_' not in os.path.basename(p)
    )
    batch_candidates = sorted(glob.glob(batch_pattern))

    candidates = batch_candidates if use_batch else stream_candidates
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"WARNING: multiple {source} files in {config_dir}, using {candidates[0]}")
    return candidates[0]


def read_config_model_name(config_dir: str, use_batch: bool) -> Optional[str]:
    """Read the model name from a result JSON in config_dir.

    Returns the last '/'-separated segment of the JSON's 'model' field,
    e.g. 'meta-llama/Llama-3.1-70B' -> 'Llama-3.1-70B'. Returns None if no
    result file is found or the field is missing.
    """
    for source in ('dooly', 'vllm'):
        filepath = find_result_file(config_dir, source, use_batch)
        if filepath is None:
            continue
        data = load_json(filepath)
        model = data.get('model') or data.get('model_id') or data.get('model_name')
        if model:
            return str(model).split('/')[-1]
    return None


def discover_configs(input_dir: str) -> List[str]:
    """List subdirectory names of input_dir, sorted."""
    return sorted(
        name for name in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, name))
    )


def _draw_cdf_on_ax(
    ax,
    config_dirs: List[str],
    metric: str,
    sources: List[str],
    use_batch: bool,
    style: str,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], List[str]]:
    """Draw CDFs + annotations for one metric onto the given axis.

    Returns (pcts, plotted_configs) so callers can print summaries.
    """
    pcts: Dict[str, Dict[str, Dict[str, float]]] = {}
    plotted_configs: List[str] = []

    for idx, config_dir in enumerate(config_dirs):
        config = config_dir
        if not os.path.isdir(config_dir):
            print(f"WARNING: config dir not found: {config_dir}")
            continue

        color = CONFIG_COLORS[idx % len(CONFIG_COLORS)]

        for source in sources:
            filepath = find_result_file(config_dir, source, use_batch)
            if filepath is None:
                print(f"WARNING: no {source} file in {config_dir}")
                continue

            values = extract_metric(load_json(filepath), metric)
            if not values:
                print(f"WARNING: no {metric} values in {filepath}")
                continue

            x, y = compute_cdf(values)
            x = x / 1000.0  # ms -> s for x-axis display
            p50 = float(np.percentile(values, 50))
            p95 = float(np.percentile(values, 95))
            p99 = float(np.percentile(values, 99))
            label = f'{config} {SOURCE_LABELS[source]}'
            linestyle = SOURCE_LINESTYLES[source]

            if style == 'stepped':
                ax.step(
                    x, y, label=label, linewidth=2.4, color=color,
                    linestyle=linestyle, alpha=0.9, where='post',
                )
            elif style == 'markers':
                marker = 'o' if source == 'vllm' else 's'
                ax.plot(
                    x, y, label=label, linewidth=2, color=color,
                    linestyle=linestyle, marker=marker, markersize=5,
                    markevery=max(1, len(x) // 15), alpha=0.6,
                    markerfacecolor='white', markeredgewidth=1.8,
                )
            else:  # smooth
                ax.plot(
                    x, y, label=label, linewidth=4, color=color,
                    linestyle=linestyle, alpha=0.9,
                )

            pcts.setdefault(config, {})[source] = {'p50': p50, 'p95': p95, 'p99': p99}
            if config not in plotted_configs:
                plotted_configs.append(config)

    metric_name = METRIC_NAMES.get(metric, metric.upper())
    ax.set_xlabel(f'{metric_name} (s)', fontsize=20, fontweight='semibold')
    ax.set_ylabel('CDF', fontsize=22, fontweight='semibold')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='both', labelsize=22)
    ax.margins(x=0.01)

    # ITL spans ~2 orders of magnitude across models, so a linear axis leaves
    # either the head or the tail visually empty. Log-scale keeps both readable.
    # TTFT stays linear.
    if metric == 'itl':
        from matplotlib.ticker import ScalarFormatter, LogLocator
        p99_values = [
            row['p99'] for cfg in pcts.values() for row in cfg.values()
        ]
    #     ax.set_xscale('log')
    #     # Render tick labels as plain numbers (10, 100, 1000) instead of 10^n.
    #     ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    #     ax.xaxis.set_major_formatter(ScalarFormatter())
    #     ax.xaxis.set_minor_formatter(plt.NullFormatter())
    #     if p99_values:
    #         # ax.set_xlim(left=0.005, right=max(p99_values) / 1000.0 * 1.1)
        ax.set_xscale('function', functions=(np.sqrt, np.square)) 
        ax.set_xlim(left=0.005, right=max(p99_values)/1000.0 * 2.5)

    for percentile in [0.5, 0.95, 0.99]:
        ax.axhline(y=percentile, color='gray', linestyle=':',
                   linewidth=2, alpha=0.4, zorder=0)

    diff_text = _build_diff_text(pcts, plotted_configs, sources)
    # if diff_text:
    #     ax.text(
    #         0.95, 0.1, diff_text,
    #         transform=ax.transAxes, fontsize=12, family='monospace',
    #         va='top', ha='right', zorder=99,
    #         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
    #                   edgecolor='gray', alpha=0.92, linewidth=1.0),
    #     )

    return pcts, plotted_configs


def _first_dispatch_time_ms(trace: List[dict]) -> Optional[float]:
    """Return sim_time_ms of the first step where a request is scheduled."""
    for row in trace:
        if row.get('batch_size', 0) > 0 or row.get('scheduled_req_ids'):
            return float(row.get('sim_time_ms', 0.0))
    return None


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='same')


def _compute_batch_series(
    trace: List[dict],
    field: str,
    smooth_window: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (times_since_first_dispatch_s, values) for a batch trace.

    Single source of truth for batch temporal data — used by both the
    matplotlib temporal plot and the CSV export so they cannot diverge.
    Returns None if the trace has no dispatched step.
    """
    if not isinstance(trace, list) or not trace:
        return None
    t0 = _first_dispatch_time_ms(trace)
    if t0 is None:
        return None
    times = np.array(
        [float(r.get('sim_time_ms', 0.0)) for r in trace]
    ) - t0
    values = np.array(
        [float(r.get(field, 0)) for r in trace], dtype=float,
    )
    mask = times >= 0
    times, values = times[mask], values[mask]
    if times.size == 0:
        return None
    if smooth_window > 1:
        values = _rolling_mean(values, smooth_window)
    return times / 1000.0, values


def _draw_batch_on_ax(
    ax,
    config_dirs: List[str],
    metric: str,
    sources: List[str],
    smooth_window: int,
) -> List[str]:
    """Plot a batch-trace field vs time for each config/source."""
    field, y_label = BATCH_METRICS[metric]
    plotted_configs: List[str] = []

    for idx, config_dir in enumerate(config_dirs):
        config = config_dir
        if not os.path.isdir(config_dir):
            print(f"WARNING: config dir not found: {config_dir}")
            continue

        color = CONFIG_COLORS[idx % len(CONFIG_COLORS)]

        for source in sources:
            filepath = find_result_file(config_dir, source, use_batch=True)
            if filepath is None:
                print(f"WARNING: no {source}_batch_*.json in {config_dir}")
                continue

            trace = load_json(filepath)
            if not isinstance(trace, list) or not trace:
                print(f"WARNING: empty/invalid batch trace {filepath}")
                continue

            t0 = _first_dispatch_time_ms(trace)
            if t0 is None:
                print(f"WARNING: no dispatched step in {filepath}")
                continue

            times = np.array(
                [float(r.get('sim_time_ms', 0.0)) for r in trace]
            ) - t0
            values = np.array(
                [float(r.get(field, 0)) for r in trace], dtype=float
            )
            mask = times >= 0
            times, values = times[mask], values[mask]
            if times.size == 0:
                continue

            if smooth_window > 1:
                values = _rolling_mean(values, smooth_window)

            times_s = times / 1000.0
            linestyle = SOURCE_LINESTYLES[source]
            ax.plot(
                times_s, values, color=color, linestyle=linestyle,
                linewidth=2.5, alpha=0.9,
            )

            if config not in plotted_configs:
                plotted_configs.append(config)

    ax.set_xlabel('Time Since First Dispatch (s)', fontsize=20, fontweight='semibold')
    ax.set_ylabel(y_label, fontsize=16, fontweight='semibold')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=22)
    ax.margins(x=0.01)
    ax.set_ylim(bottom=0)

    return plotted_configs


def _draw_batch_cdf_on_ax(
    ax,
    config_dirs: List[str],
    metric: str,
    sources: List[str],
) -> List[str]:
    """Draw CDF of per-step batch-trace values (across all scheduler steps)."""
    field, value_label = BATCH_METRICS[metric]
    plotted_configs: List[str] = []

    for idx, config_dir in enumerate(config_dirs):
        config = config_dir
        if not os.path.isdir(config_dir):
            print(f"WARNING: config dir not found: {config_dir}")
            continue

        color = CONFIG_COLORS[idx % len(CONFIG_COLORS)]

        for source in sources:
            filepath = find_result_file(config_dir, source, use_batch=True)
            if filepath is None:
                print(f"WARNING: no {source}_batch_*.json in {config_dir}")
                continue

            trace = load_json(filepath)
            if not isinstance(trace, list) or not trace:
                print(f"WARNING: empty/invalid batch trace {filepath}")
                continue

            # Only count steps where something was actually scheduled.
            values = [
                float(r.get(field, 0)) for r in trace
                if r.get('batch_size', 0) > 0 or r.get('scheduled_req_ids')
            ]
            if not values:
                continue

            x, y = compute_cdf(values)
            linestyle = SOURCE_LINESTYLES[source]
            ax.plot(
                x, y, color=color, linestyle=linestyle,
                linewidth=4, alpha=0.9,
            )

            if config not in plotted_configs:
                plotted_configs.append(config)

    ax.set_xlabel(value_label, fontsize=22, fontweight='semibold')
    ax.set_ylabel('CDF', fontsize=22, fontweight='semibold')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='both', labelsize=22)
    ax.margins(x=0.01)

    for percentile in [0.5, 0.95, 0.99]:
        ax.axhline(y=percentile, color='gray', linestyle=':',
                   linewidth=2, alpha=0.4, zorder=0)

    return plotted_configs


def _print_summary(
    metric: str,
    pcts: Dict[str, Dict[str, Dict[str, float]]],
    plotted_configs: List[str],
    sources: List[str],
) -> None:
    print(f"\nPercentile summary [{metric}]:")
    print(f"  {'config':<10} {'source':<8} {'P50 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10}")
    for config in plotted_configs:
        for source in sources:
            row = pcts.get(config, {}).get(source)
            if row is None:
                continue
            print(
                f"  {config:<10} {source:<8} "
                f"{row['p50']:>10.2f} {row['p95']:>10.2f} {row['p99']:>10.2f}"
            )


def _draw_one_metric(
    ax,
    metric: str,
    config_dirs: List[str],
    sources: List[str],
    use_batch: bool,
    style: str,
    batch_view: str,
    smooth_window: int,
) -> List[str]:
    """Dispatch to the right drawer for one metric; return plotted config dirs."""
    if metric in BATCH_METRICS:
        if batch_view == 'cdf':
            return _draw_batch_cdf_on_ax(ax, config_dirs, metric, sources)
        return _draw_batch_on_ax(ax, config_dirs, metric, sources, smooth_window)
    pcts, plotted_configs = _draw_cdf_on_ax(
        ax, config_dirs, metric, sources, use_batch, style,
    )
    if plotted_configs:
        _print_summary(metric, pcts, plotted_configs, sources)
    return plotted_configs


def _build_legend_handles(
    config_dirs: List[str],
    plotted_configs: List[str],
    sources: List[str],
    use_batch: bool,
    config_labels: Dict[str, str],
) -> List[Line2D]:
    config_handles = [
        Line2D(
            [0], [0],
            marker='o', linestyle='None',
            markerfacecolor=CONFIG_COLORS[config_dirs.index(c) % len(CONFIG_COLORS)],
            markeredgecolor=CONFIG_COLORS[config_dirs.index(c) % len(CONFIG_COLORS)],
            markersize=12,
            label=config_labels.get(c)
                  or read_config_model_name(c, use_batch)
                  or os.path.basename(c.rstrip('/')).upper(),
        )
        for c in plotted_configs
    ]
    source_handles = [
        Line2D(
            [0], [0], color='black', linestyle=SOURCE_LINESTYLES[s],
            linewidth=3, label=SOURCE_LABELS[s],
        )
        for s in sources
    ]
    return config_handles + source_handles


def _save_split_outputs(
    split_dir: str,
    metrics: List[str],
    config_dirs: List[str],
    sources: List[str],
    use_batch: bool,
    style: str,
    batch_view: str,
    smooth_window: int,
    config_labels: Dict[str, str],
) -> None:
    """Save each subplot and the legend as separate PDFs in split_dir."""
    os.makedirs(split_dir, exist_ok=True)
    all_plotted_configs: List[str] = []
    any_plotted = False

    for i, m in enumerate(metrics, start=1):
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        plotted_configs = _draw_one_metric(
            ax, m, config_dirs, sources, use_batch, style,
            batch_view, smooth_window,
        )
        if plotted_configs:
            any_plotted = True
            for c in plotted_configs:
                if c not in all_plotted_configs:
                    all_plotted_configs.append(c)

        plt.tight_layout(pad=0.6)
        out_path = os.path.join(split_dir, f'plot{i}.pdf')
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
        fig.savefig(out_path.replace('.pdf', '.png'), dpi=300,
                    bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"✓ Saved {out_path}")

    if not any_plotted:
        print("ERROR: no data plotted")
        sys.exit(1)

    handles = _build_legend_handles(
        config_dirs, all_plotted_configs, sources, use_batch, config_labels,
    )
    legend_fig = plt.figure(figsize=(max(6, 1.6 * len(handles)), 0.5))
    legend_fig.legend(
        handles=handles, loc='center', ncol=len(handles),
        frameon=False, fontsize=24,
        handlelength=1.2, handletextpad=0.1, columnspacing=0.4,
        borderaxespad=0.2,
    )
    legend_path = os.path.join(split_dir, 'legend.pdf')
    legend_fig.savefig(legend_path, bbox_inches='tight', pad_inches=0.05)
    legend_fig.savefig(legend_path.replace('.pdf', '.png'), dpi=300,
                       bbox_inches='tight', pad_inches=0.05)
    plt.close(legend_fig)
    print(f"✓ Saved {legend_path}")


def _compress_constant_runs(
    x: np.ndarray, y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop interior points of horizontal segments (lossless for line plots).

    A point i is "interior" iff y[i-1] == y[i] == y[i+1]. Removing such
    points doesn't change the line drawn between consecutive samples, so
    the rendered figure is bit-for-bit identical to the uncompressed
    series.
    """
    n = x.size
    if n <= 2:
        return x, y
    keep = np.ones(n, dtype=bool)
    interior = (y[:-2] == y[1:-1]) & (y[1:-1] == y[2:])
    keep[1:-1] = ~interior
    return x[keep], y[keep]


def _resample_cdf(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample a CDF to ~max_points while preserving tail accuracy.

    Strategy: linear spacing in [0, 0.99] for the body plus geometric
    spacing in [0.99, 1.0] so p95/p99/p999 land on real sample indices.
    Empirical CDFs are step functions, so for each target y we pick the
    nearest underlying sample (no interpolation between points).
    """
    n = x.size
    if max_points <= 0 or n <= max_points:
        return x, y

    # ~70% body / ~30% tail; dial down if the user picked a small budget.
    n_body = max(int(max_points * 0.7), 1)
    n_tail = max(max_points - n_body, 1)

    body = np.linspace(1.0 / n, 0.99, n_body, endpoint=False)
    # geomspace in distance-from-1: 0.99, 0.999, 0.9999, ...
    tail_gap = np.geomspace(0.01, max(1.0 / n, 1e-9), n_tail)
    tail = 1.0 - tail_gap

    y_targets = np.unique(np.concatenate([body, tail, [1.0]]))
    # Each y_target maps to a real index in the sorted array (1-based / n).
    indices = np.clip(np.ceil(y_targets * n).astype(int) - 1, 0, n - 1)
    indices = np.unique(indices)
    return x[indices], y[indices]


def _collect_cdfs_for_config(
    config_dir: str,
    metrics: List[str],
    sources: List[str],
    use_batch: bool,
    max_points: int = 0,
    smooth_window: int = 1,
) -> "Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]":
    """Compute (x, y) arrays for each (metric, source) pair in one config.

    TTFT/ITL: empirical CDF, x in seconds, y in [0, 1].
    Batch metrics: temporal trace, x = time-since-first-dispatch (s),
    y = the metric value at each scheduler step, with the same warm-up
    filter and optional smoothing as the on-screen temporal plot.
    """
    out: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for metric in metrics:
        for source in sources:
            filepath = find_result_file(
                config_dir, source,
                use_batch=use_batch or metric in BATCH_METRICS,
            )
            if filepath is None:
                continue

            if metric in BATCH_METRICS:
                field, _ = BATCH_METRICS[metric]
                trace = load_json(filepath)
                if not isinstance(trace, list) or not trace:
                    continue
                t0 = _first_dispatch_time_ms(trace)
                if t0 is None:
                    continue
                times = np.array(
                    [float(r.get('sim_time_ms', 0.0)) for r in trace]
                ) - t0
                values = np.array(
                    [float(r.get(field, 0)) for r in trace], dtype=float
                )
                mask = times >= 0
                times, values = times[mask], values[mask]
                if times.size == 0:
                    continue
                if smooth_window > 1:
                    values = _rolling_mean(values, smooth_window)
                x, y = times / 1000.0, values
                # Drop interior points of horizontal runs. The rendered
                # line is unchanged but the CSV shrinks dramatically when
                # the trace has long idle / steady-state stretches.
                x, y = _compress_constant_runs(x, y)
            else:
                values = extract_metric(load_json(filepath), metric)
                if not values:
                    continue
                x, y = compute_cdf(values)
                x = x / 1000.0  # ms -> s, matches the plot's x-axis units
                x, y = _resample_cdf(x, y, max_points)

            out[(metric, source)] = (x, y)
    return out


def _csv_axis_names(metric: str) -> Tuple[str, str]:
    """Column-name suffixes for (x, y) of this metric in the CSV.

    Batch metrics export a temporal trace; others export an empirical CDF.
    """
    if metric in BATCH_METRICS:
        return 'time_s', 'value'
    return 'x_s', 'cdf'


def _export_cdf_csv(
    config_dir: str,
    metrics: List[str],
    sources: List[str],
    use_batch: bool,
    csv_path: str,
    max_points: int = 0,
    smooth_window: int = 1,
) -> bool:
    """Write one wide-format CSV with paired x/cdf columns per (metric, source).

    Each series is independently sorted, so shorter series are right-padded
    with empty cells. Returns True if any data was written.
    """
    cdfs = _collect_cdfs_for_config(
        config_dir, metrics, sources, use_batch,
        max_points=max_points, smooth_window=smooth_window,
    )
    if not cdfs:
        return False

    # Stable column order: metrics in user-supplied order, sources nested.
    columns: List[Tuple[str, str]] = [
        (m, s) for m in metrics for s in sources if (m, s) in cdfs
    ]
    max_len = max(cdfs[c][0].size for c in columns)

    header: List[str] = []
    for metric, source in columns:
        x_name, y_name = _csv_axis_names(metric)
        header.append(f'{metric}_{source}_{x_name}')
        header.append(f'{metric}_{source}_{y_name}')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(max_len):
            row: List[str] = []
            for metric, source in columns:
                x, y = cdfs[(metric, source)]
                if i < x.size:
                    row.append(f'{float(x[i]):.6g}')
                    row.append(f'{float(y[i]):.6g}')
                else:
                    row.extend(['', ''])
            writer.writerow(row)
    return True


def _export_all_cdf_csvs(
    config_dirs: List[str],
    metrics: List[str],
    sources: List[str],
    use_batch: bool,
    csv_output_dir: str,
    config_labels: Dict[str, str],
    max_points: int = 0,
    smooth_window: int = 1,
) -> None:
    """Write one CSV per config_dir into csv_output_dir.

    TTFT/ITL columns hold an empirical CDF; batch-metric columns hold the
    temporal trace (x = time-since-first-dispatch in seconds, y = value).
    """
    os.makedirs(csv_output_dir, exist_ok=True)
    for config_dir in config_dirs:
        if not os.path.isdir(config_dir):
            print(f"WARNING: config dir not found: {config_dir}")
            continue

        label = (
            config_labels.get(config_dir)
            or read_config_model_name(config_dir, use_batch)
            or os.path.basename(config_dir.rstrip('/'))
        )
        # Sanitize for use as a filename.
        safe = ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_'
                       for c in label)
        csv_path = os.path.join(csv_output_dir, f'{safe}_cdf.csv')
        if _export_cdf_csv(
            config_dir, metrics, sources, use_batch, csv_path,
            max_points=max_points, smooth_window=smooth_window,
        ):
            print(f"✓ Saved CDF CSV to {csv_path}")
        else:
            print(f"WARNING: no CDF data exported for {config_dir}")


def plot_configs_cdf(
    config_dirs: List[str],
    metric: str,
    sources: List[str],
    use_batch: bool,
    output_file: Optional[str],
    title: Optional[str],
    style: str,
    batch_view: str = 'temporal',
    smooth_window: int = 1,
    config_labels: Optional[Dict[str, str]] = None,
    split_output: Optional[str] = None,
    csv_output: Optional[str] = None,
    max_cdf_points: int = 0,
) -> None:
    config_labels = config_labels or {}
    plt.style.use('seaborn-v0_8-whitegrid')

    if metric == 'both':
        metrics = ['ttft', 'itl']
    elif metric == 'batch_both':
        metrics = ['batch_size', 'running']
    elif metric == 'all':
        metrics = ['ttft', 'itl', 'batch_size']
    else:
        metrics = [metric]

    if csv_output:
        _export_all_cdf_csvs(
            config_dirs, metrics, sources, use_batch, csv_output, config_labels,
            max_points=max_cdf_points, smooth_window=smooth_window,
        )

    if split_output:
        _save_split_outputs(
            split_output, metrics, config_dirs, sources, use_batch,
            style, batch_view, smooth_window, config_labels,
        )
        return

    ncols = len(metrics)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), squeeze=False)

    any_plotted = False
    all_plotted_configs: List[str] = []
    for ax, m in zip(axes[0], metrics):
        plotted_configs = _draw_one_metric(
            ax, m, config_dirs, sources, use_batch, style,
            batch_view, smooth_window,
        )
        if plotted_configs:
            any_plotted = True
            for c in plotted_configs:
                if c not in all_plotted_configs:
                    all_plotted_configs.append(c)

    if not any_plotted:
        print("ERROR: no data plotted")
        sys.exit(1)

    # Combined horizontal legend at the top: colors = configs, styles = sources.
    handles = _build_legend_handles(
        config_dirs, all_plotted_configs, sources, use_batch, config_labels,
    )
    fig.legend(
        handles=handles, loc='upper center', ncol=len(handles),
        bbox_to_anchor=(0.5, 1.005), frameon=False, fontsize=24,
        handlelength=1.2, handletextpad=0.3, columnspacing=0.6,
        borderaxespad=0.2,
    )

    plt.tight_layout(pad=0.6, rect=(0, 0, 1, 0.93))

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.05)
        print(f"✓ Saved plot to {output_file}")
    else:
        plt.show()


def _build_diff_text(
    pcts: Dict[str, Dict[str, Dict[str, float]]],
    configs: List[str],
    sources: List[str],
) -> str:
    """Build a compact sim-vs-real error table for annotation.

    Returns '' when we don't have both dooly and vllm for at least one config.
    """
    if 'dooly' not in sources or 'vllm' not in sources:
        return ''
    rows: List[str] = []
    for config in configs:
        entry = pcts.get(config, {})
        if 'dooly' not in entry or 'vllm' not in entry:
            continue
        d, v = entry['dooly'], entry['vllm']
        parts = [f'{config:<5}']
        for pct_key, pct_label in [('p95', 'P95'), ('p99', 'P99')]:
            abs_diff = d[pct_key] - v[pct_key]
            rel_diff = (abs_diff / v[pct_key] * 100.0) if v[pct_key] else 0.0
            parts.append(
                f'{pct_label} {abs_diff:+6.1f}ms ({rel_diff:+5.1f}%)'
            )
        rows.append('  '.join(parts))
    if not rows:
        return ''
    # header = 'Sim − Real (DoolySim − vLLM)'
    return '\n'.join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot TTFT/ITL CDFs across selected configurations.",
    )
    parser.add_argument(
        "input_dir", nargs='?',
        help=(
            "Base dir containing config subdirs (e.g. comparison/sharegpt-500reqs-2rps). "
            "Optional if every --configs entry is an absolute path or contains '/'."
        ),
    )
    parser.add_argument(
        "--configs", nargs='+',
        help=(
            "Configurations to plot. Entries without '/' are resolved against input_dir "
            "(e.g. tp1 tp4). Entries containing '/' or starting with '/' are used as "
            "full paths (e.g. comparison/run-1024/tp1 /abs/path/run-8192/tp4). "
            "Default: all subdirs of input_dir."
        ),
    )
    parser.add_argument(
        "--labels", nargs='+',
        help="Legend labels, one per --configs entry. Overrides the model-name auto label.",
    )
    parser.add_argument(
        "--metric",
        choices=['ttft', 'itl', 'both',
                 'batch_size', 'running', 'waiting', 'tokens', 'batch_both',
                 'all'],
        default='ttft',
        help=(
            "Metric to plot (default: ttft). 'both' = TTFT+ITL side-by-side. "
            "Batch-trace metrics: batch_size, running, waiting, tokens; "
            "'batch_both' = batch_size+running side-by-side. "
            "'all' = TTFT + ITL + batch_size side-by-side."
        ),
    )
    parser.add_argument(
        "--batch-view", choices=['temporal', 'cdf'], default='temporal',
        help="For batch-trace metrics, plot values over time ('temporal') or as a CDF ('cdf')",
    )
    parser.add_argument(
        "--smooth", type=int, default=1,
        help="Rolling-mean window (in steps) for temporal batch plots (default: 1 = raw)",
    )
    parser.add_argument(
        "--sources", nargs='+', choices=['dooly', 'vllm'], default=['dooly', 'vllm'],
        help="Which result sources to plot per config (default: both)",
    )
    parser.add_argument(
        "--batch", action='store_true',
        help="Use *_batch_*.json files instead of the streaming results",
    )
    parser.add_argument("--output", "-o", help="Output image path (e.g. cdf.png)")
    parser.add_argument(
        "--split-output",
        help=(
            "Directory to save each subplot and the legend as separate PDFs "
            "(legend.pdf, plot1.pdf, plot2.pdf, ...). Use for paper subfigures."
        ),
    )
    parser.add_argument(
        "--csv-output",
        help=(
            "Directory to save one CDF CSV per config (e.g. {model}_cdf.csv). "
            "Wide format with paired x/cdf columns per (metric, source) so the "
            "same figure can be plotted in Excel."
        ),
    )
    parser.add_argument(
        "--max-cdf-points", type=int, default=1000,
        help=(
            "Downsample each CDF series to ~N points before writing CSV "
            "(default: 1000; pass 0 to disable). Uses a dense-tail grid so "
            "p95/p99/p999 percentile accuracy is preserved."
        ),
    )
    parser.add_argument("--title", help="Custom plot title")
    parser.add_argument(
        "--style", choices=['smooth', 'stepped', 'markers'], default='smooth',
        help="Line style (default: smooth)",
    )
    parser.add_argument(
        "--list-configs", action='store_true',
        help="Print the configurations found under input_dir and exit",
    )

    args = parser.parse_args()

    def _is_pathlike(entry: str) -> bool:
        return os.path.isabs(entry) or ('/' in entry)

    if args.list_configs:
        if not args.input_dir or not os.path.isdir(args.input_dir):
            parser.error("--list-configs requires a valid input_dir")
        names = discover_configs(args.input_dir)
        print('\n'.join(names) if names else '(no subdirectories found)')
        return

    raw_configs: List[str] = args.configs or []
    if not raw_configs:
        if not args.input_dir or not os.path.isdir(args.input_dir):
            parser.error("provide --configs or a valid input_dir to discover configs from")
        names = discover_configs(args.input_dir)
        if not names:
            parser.error(f"no configuration subdirectories under {args.input_dir}")
        raw_configs = names

    # Resolve each entry: full path if path-like, else input_dir-relative.
    resolved: List[str] = []
    for entry in raw_configs:
        if _is_pathlike(entry):
            resolved.append(os.path.abspath(entry))
        else:
            if not args.input_dir:
                parser.error(
                    f"'{entry}' is not a path; input_dir is required to resolve it"
                )
            resolved.append(os.path.abspath(os.path.join(args.input_dir, entry)))

    missing = [p for p in resolved if not os.path.isdir(p)]
    if missing:
        parser.error(f"config dirs not found: {missing}")

    # Build optional labels map from --labels.
    labels_map: Dict[str, str] = {}
    if args.labels:
        if len(args.labels) != len(resolved):
            parser.error(
                f"--labels count ({len(args.labels)}) must match --configs count ({len(resolved)})"
            )
        labels_map = dict(zip(resolved, args.labels))

    plot_configs_cdf(
        config_dirs=resolved,
        metric=args.metric,
        sources=args.sources,
        use_batch=args.batch,
        output_file=args.output,
        title=args.title,
        style=args.style,
        batch_view=args.batch_view,
        smooth_window=max(1, args.smooth),
        config_labels=labels_map,
        split_output=args.split_output,
        csv_output=args.csv_output,
        max_cdf_points=max(0, args.max_cdf_points),
    )


if __name__ == "__main__":
    main()
