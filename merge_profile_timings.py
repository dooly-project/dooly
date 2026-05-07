#!/usr/bin/env python
"""Merge per-shard profile_timings CSVs into one file.

Each shard writes profile_timings_gpu{N}.csv; this tool concatenates them
into profile_timings.csv (keeping a single header). Rows are tagged with
the source shard so you can still separate them later.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path
import re
import sys


SHARD_RE = re.compile(r"^profile_timings_(gpu\d+)\.csv$")


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge per-shard profile_timings CSVs")
    ap.add_argument("--dir", type=Path, required=True,
                    help="Directory containing profile_timings_gpuN.csv files.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Merged output path. Defaults to {dir}/profile_timings.csv.")
    ap.add_argument("--delete-shards", action="store_true",
                    help="Delete per-shard files after merging. Default: keep them. "
                         "Backups like *_pre8192.csv or *_orig.csv are NEVER touched "
                         "since the shard regex requires the suffix to be 'gpuN'.")
    args = ap.parse_args()

    d: Path = args.dir
    if not d.is_dir():
        print(f"[ERROR] Not a directory: {d}", file=sys.stderr)
        return 2

    # Only match strict 'profile_timings_gpuN.csv' — explicitly skip backups
    # like profile_timings_gpu0_orig.csv, profile_timings_gpu0_pre8192.csv,
    # and the merged-output file profile_timings_v2.csv.
    shard_files = sorted(
        p for p in d.glob("profile_timings_*.csv")
        if SHARD_RE.match(p.name)
    )
    if not shard_files:
        print(f"[WARN] No profile_timings_*.csv files found in {d}")
        return 0

    out_path = args.output or (d / "profile_timings.csv")
    header = None
    total_rows = 0

    with open(out_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        for shard_path in shard_files:
            m = SHARD_RE.search(shard_path.name)
            shard_tag = m.group(1) if m else shard_path.stem
            with open(shard_path, newline="") as in_f:
                reader = csv.reader(in_f)
                try:
                    this_header = next(reader)
                except StopIteration:
                    continue
                if header is None:
                    header = this_header + ["shard"]
                    writer.writerow(header)
                elif this_header != header[:-1]:
                    print(f"[WARN] Header mismatch in {shard_path.name}; skipping.")
                    continue
                for row in reader:
                    writer.writerow(row + [shard_tag])
                    total_rows += 1
            print(f"  merged {shard_path.name}")

    print(f"\n[OK] Wrote {total_rows} rows from {len(shard_files)} shards -> {out_path}")

    if args.delete_shards:
        for p in shard_files:
            p.unlink()
        print(f"[OK] Removed {len(shard_files)} shard files.")
    else:
        print(f"[OK] Per-shard files kept (pass --delete-shards to remove them).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
