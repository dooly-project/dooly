#!/usr/bin/env python3
"""Validate Dooly taint labels inside the largest model call in a trace."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


UNKNOWN_RE = re.compile(r"\?\((\d+)\)")
LABEL_RE = re.compile(
    r"\b("
    r"MODEL_CONFIG|"
    r"NUM_TOKS|NUM_TOKEN|NUM_TOKENS|"
    r"NUM_REQS|NUM_REQUEST|NUM_REQUESTS|"
    r"MAX_NUM_TOKS|MAX_NUM_TOKEN|MAX_NUM_TOKENS|MAX_TOTAL_NUM_TOKENS|"
    r"MAX_NUM_REQS|MAX_NUM_REQUEST|MAX_NUM_REQUESTS|MAX_RUNNING_REQUESTS|"
    r"SINGLETON|MIX"
    r")\((\d+)\)"
)
ARG_LABEL_RE = re.compile(
    r"\b("
    r"MODEL_CONFIG|"
    r"NUM_TOKS|NUM_TOKEN|NUM_TOKENS|"
    r"NUM_REQS|NUM_REQUEST|NUM_REQUESTS|"
    r"MAX_NUM_TOKS|MAX_NUM_TOKEN|MAX_NUM_TOKENS|MAX_TOTAL_NUM_TOKENS|"
    r"MAX_NUM_REQS|MAX_NUM_REQUEST|MAX_NUM_REQUESTS|MAX_RUNNING_REQUESTS"
    r")=(\d+)"
)
MIX_HIST_RE = re.compile(r"MIX\((\d+)\)\[hist:([^\]]+)\]")


@dataclass(frozen=True)
class Problem:
    kind: str
    event_name: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check Dooly trace taint labels inside the largest MODULE: *Model "
            "call block. The check fails on unknown ?() labels and on runtime "
            "values mislabeled as model config or vice versa."
        )
    )
    parser.add_argument(
        "trace",
        type=Path,
        help="Trace JSON file, .trace.json.gz file, or directory containing traces.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        required=True,
        help=(
            "Expected NUM_TOKS value(s) for the model call being validated. "
            "Pass multiple values when the trace contains helper token lengths."
        ),
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        nargs="+",
        required=True,
        help=(
            "Expected NUM_REQS value(s) for the model call being validated. "
            "SGLang traces can include helper lengths such as num_reqs + 1."
        ),
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        nargs="+",
        default=None,
        help="Expected MAX_NUM_TOKS value, if that label appears in the trace.",
    )
    parser.add_argument(
        "--max-num-requests",
        type=int,
        nargs="+",
        default=None,
        help="Expected MAX_NUM_REQS value, if that label appears in the trace.",
    )
    parser.add_argument(
        "--model-event-regex",
        default=r"^MODULE: .*Model\b",
        help=(
            "Regex used to find top-level model module events before the IN: "
            "section is stripped. Defaults to '^MODULE: .*Model\\b'."
        ),
    )
    parser.add_argument(
        "--scan-all-events",
        action="store_true",
        help="Validate the whole trace instead of the largest model module interval.",
    )
    parser.add_argument(
        "--strict-singleton-runtime",
        action="store_true",
        help=(
            "Treat SINGLETON(1) as an error when NUM_TOKS or NUM_REQS is 1. "
            "By default SINGLETON(1) is display-only and allowed."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Maximum failing event examples to print per problem class.",
    )
    return parser.parse_args()


def resolve_trace_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise SystemExit(f"trace path does not exist: {path}")

    candidates = sorted(
        list(path.glob("*.trace.json")) + list(path.glob("*.trace.json.gz")),
        key=lambda item: item.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit(f"no *.trace.json or *.trace.json.gz files found in {path}")
    return candidates[-1]


def load_trace(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def event_name(event: dict) -> str:
    name = event.get("name")
    return name if isinstance(name, str) else ""


def event_end(event: dict) -> float:
    return float(event.get("ts", 0.0)) + float(event.get("dur", 0.0) or 0.0)


def model_event_prefix(name: str) -> str:
    return name.split(" IN:", 1)[0]


def choose_largest_model_event(events: list[dict], model_regex: re.Pattern[str]) -> dict:
    model_events = [
        event
        for event in events
        if model_regex.search(model_event_prefix(event_name(event)))
        and "ts" in event
        and "dur" in event
    ]
    if not model_events:
        raise SystemExit(
            "no model module event matched --model-event-regex; "
            "use --scan-all-events or adjust --model-event-regex"
        )
    return max(model_events, key=lambda event: float(event.get("dur", 0.0) or 0.0))


def overlaps_interval(event: dict, start: float, end: float) -> bool:
    if "ts" not in event:
        return False
    ts = float(event["ts"])
    dur = float(event.get("dur", 0.0) or 0.0)
    if dur <= 0:
        return start <= ts <= end
    return ts < end and ts + dur > start


def scoped_events(
    events: list[dict],
    *,
    scan_all_events: bool,
    model_regex: re.Pattern[str],
) -> tuple[list[dict], str]:
    if scan_all_events:
        return events, "whole trace"

    model_event = choose_largest_model_event(events, model_regex)
    start = float(model_event["ts"])
    end = event_end(model_event)
    scope = (
        f"{model_event_prefix(event_name(model_event))} "
        f"ts={start:.3f} dur={float(model_event.get('dur', 0.0)):.3f}"
    )
    return [
        event for event in events if overlaps_interval(event, start, end)
    ], scope


def runtime_value_sets(args: argparse.Namespace) -> dict[str, set[int]]:
    values = {
        "NUM_TOKS": set(args.num_tokens),
        "NUM_REQS": set(args.num_requests),
        "MAX_NUM_TOKS": set(),
        "MAX_NUM_REQS": set(),
    }
    if args.max_num_tokens is not None:
        values["MAX_NUM_TOKS"].update(args.max_num_tokens)
    if args.max_num_requests is not None:
        values["MAX_NUM_REQS"].update(args.max_num_requests)
    return values


def canonical_label(label: str) -> str:
    if label in {"NUM_TOKEN", "NUM_TOKENS"}:
        return "NUM_TOKS"
    if label in {"NUM_REQUEST", "NUM_REQUESTS"}:
        return "NUM_REQS"
    if label in {
        "MAX_NUM_TOKEN",
        "MAX_NUM_TOKENS",
        "MAX_TOTAL_NUM_TOKENS",
    }:
        return "MAX_NUM_TOKS"
    if label in {
        "MAX_NUM_REQUEST",
        "MAX_NUM_REQUESTS",
        "MAX_RUNNING_REQUESTS",
    }:
        return "MAX_NUM_REQS"
    return label


def sample(problems: Iterable[Problem], limit: int) -> dict[str, list[Problem]]:
    grouped: dict[str, list[Problem]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for problem in problems:
        key = (problem.kind, problem.detail, problem.event_name)
        if key in seen:
            continue
        seen.add(key)
        if len(grouped[problem.kind]) < limit:
            grouped[problem.kind].append(problem)
    return grouped


def iter_labels(name: str) -> Iterable[tuple[str, int]]:
    for label, raw_value in LABEL_RE.findall(name):
        yield canonical_label(label), int(raw_value)
    for label, raw_value in ARG_LABEL_RE.findall(name):
        yield canonical_label(label), int(raw_value)


def iter_mix_components(name: str) -> Iterable[tuple[str, int]]:
    for _, history in MIX_HIST_RE.findall(name):
        for component in history.split(","):
            parts = [part.strip() for part in component.split(":")]
            if len(parts) != 2:
                continue
            left, right = parts
            if left.isdigit():
                label = canonical_label(right)
                value = int(left)
            elif right.isdigit():
                label = canonical_label(left)
                value = int(right)
            else:
                continue
            yield label, value


def validate_events(events: list[dict], args: argparse.Namespace) -> tuple[list[Problem], Counter]:
    problems: list[Problem] = []
    label_counts: Counter = Counter()
    expected = runtime_value_sets(args)
    token_values = expected["NUM_TOKS"]
    request_values = expected["NUM_REQS"]
    max_token_values = expected["MAX_NUM_TOKS"]
    max_request_values = expected["MAX_NUM_REQS"]
    runtime_values = token_values | request_values | max_token_values | max_request_values

    for event in events:
        name = event_name(event)
        if not name:
            continue
        for label, value in iter_labels(name):
            label_counts[(label, value)] += 1

    observed_expected_runtime_values = {
        value
        for (label, value), count in label_counts.items()
        if count > 0 and label in expected and value in expected[label]
    }

    for event in events:
        name = event_name(event)
        if not name:
            continue

        for match in UNKNOWN_RE.finditer(name):
            problems.append(
                Problem(
                    kind="unknown",
                    event_name=name,
                    detail=f"unknown label ?({match.group(1)})",
                )
            )

        for label, value in iter_labels(name):
            if label == "SINGLETON":
                if (
                    args.strict_singleton_runtime
                    and value in token_values | request_values
                ):
                    problems.append(
                        Problem(
                            kind="singleton-runtime",
                            event_name=name,
                            detail=(
                                f"SINGLETON({value}) collides with expected "
                                "runtime value"
                            ),
                        )
                    )
                continue

            if label == "MIX":
                continue

            if label == "MODEL_CONFIG":
                if (
                    value in runtime_values
                    and value not in observed_expected_runtime_values
                ):
                    problems.append(
                        Problem(
                            kind="model-config-runtime-value",
                            event_name=name,
                            detail=(
                                f"MODEL_CONFIG({value}) uses a known runtime "
                                "dimension value"
                            ),
                        )
                    )
                continue

            if label == "NUM_TOKS" and value not in token_values:
                problems.append(
                    Problem(
                        kind="wrong-num-toks-value",
                        event_name=name,
                        detail=f"NUM_TOKS({value}) != expected {sorted(token_values)}",
                    )
                )
            elif label == "NUM_REQS" and value not in request_values:
                problems.append(
                    Problem(
                        kind="wrong-num-reqs-value",
                        event_name=name,
                        detail=f"NUM_REQS({value}) != expected {sorted(request_values)}",
                    )
                )
            elif max_token_values and label == "MAX_NUM_TOKS" and value not in max_token_values:
                problems.append(
                    Problem(
                        kind="wrong-max-num-toks-value",
                        event_name=name,
                        detail=(
                            f"MAX_NUM_TOKS({value}) != expected "
                            f"{sorted(max_token_values)}"
                        ),
                    )
                )
            elif max_request_values and label == "MAX_NUM_REQS" and value not in max_request_values:
                problems.append(
                    Problem(
                        kind="wrong-max-num-reqs-value",
                        event_name=name,
                        detail=(
                            f"MAX_NUM_REQS({value}) != expected "
                            f"{sorted(max_request_values)}"
                        ),
                    )
                )

        for label, value in iter_mix_components(name):
            if label == "NUM_TOKS" and value not in token_values:
                problems.append(
                    Problem(
                        kind="wrong-mix-num-toks-value",
                        event_name=name,
                        detail=(
                            f"MIX history NUM_TOKS:{value} != expected "
                            f"{sorted(token_values)}"
                        ),
                    )
                )
            elif label == "NUM_REQS" and value not in request_values:
                problems.append(
                    Problem(
                        kind="wrong-mix-num-reqs-value",
                        event_name=name,
                        detail=(
                            f"MIX history NUM_REQS:{value} != expected "
                            f"{sorted(request_values)}"
                        ),
                    )
                )
            elif (
                label == "MODEL_CONFIG"
                and value in runtime_values
                and value not in observed_expected_runtime_values
            ):
                problems.append(
                    Problem(
                        kind="mix-model-config-runtime-value",
                        event_name=name,
                        detail=(
                            f"MIX history MODEL_CONFIG:{value} uses a known "
                            "runtime dimension value"
                        ),
                    )
                )

    return problems, label_counts


def print_summary(
    *,
    trace_path: Path,
    scope: str,
    events: list[dict],
    problems: list[Problem],
    label_counts: Counter,
    max_examples: int,
) -> None:
    print(f"trace: {trace_path}")
    print(f"scope: {scope}")
    print(f"events checked: {len(events)}")
    print(f"labels seen: {sum(label_counts.values())}")

    if label_counts:
        print("top labels:")
        for (label, value), count in label_counts.most_common(12):
            print(f"  {label}({value}): {count}")

    if not problems:
        print("result: PASS")
        return

    problem_counts = Counter(problem.kind for problem in problems)
    print("result: FAIL")
    print("problems:")
    for kind, count in problem_counts.most_common():
        print(f"  {kind}: {count}")

    print("examples:")
    for kind, entries in sample(problems, max_examples).items():
        print(f"  {kind}:")
        for entry in entries:
            print(f"    - {entry.detail}: {entry.event_name}")


def main() -> int:
    args = parse_args()
    trace_path = resolve_trace_path(args.trace)
    trace_data = load_trace(trace_path)
    events = trace_data.get("traceEvents", [])
    if not isinstance(events, list):
        raise SystemExit("traceEvents is missing or is not a list")

    scoped, scope = scoped_events(
        events,
        scan_all_events=args.scan_all_events,
        model_regex=re.compile(args.model_event_regex),
    )
    problems, label_counts = validate_events(scoped, args)
    print_summary(
        trace_path=trace_path,
        scope=scope,
        events=scoped,
        problems=problems,
        label_counts=label_counts,
        max_examples=args.max_examples,
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
