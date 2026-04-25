#!/usr/bin/env python
"""Merge per-GPU SQLite DBs produced by run_scenario_a.py into one DB.

Each worker writes to dooly-eval_gpu{N}.db with AUTOINCREMENT ids that
collide across shards. This tool:

  1. Creates an empty target DB using the profiler's schema (copied from
     one of the shards so it exactly matches whatever version produced it).
  2. For each shard, copies rows table-by-table, remapping AUTOINCREMENT
     foreign keys. Signatures and coll_ops are deduplicated on
     signature_hash (content-addressed); other tables concatenate.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict


# Tables with content-addressed UNIQUE key, deduplicated across shards.
# Maps table -> (autoincrement PK column, dedup key column).
DEDUP_TABLES = {
    "signatures":  ("operation_id", "signature_hash"),
    "coll_ops":    ("collective_id", "signature_hash"),
}


def copy_schema(src_db: Path, dst_db: Path) -> None:
    """Create dst_db with the same schema as src_db (tables + indexes)."""
    src = sqlite3.connect(str(src_db))
    dst = sqlite3.connect(str(dst_db))
    try:
        ddl_rows = src.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND type IN ('table','index') "
            "ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END"
        ).fetchall()
        for (ddl,) in ddl_rows:
            if ddl.strip().lower().startswith(("create table sqlite_", "create index sqlite_")):
                continue
            dst.execute(ddl)
        dst.commit()
    finally:
        src.close()
        dst.close()


def fetch_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone())


def merge_dedup_table(
    src: sqlite3.Connection, dst: sqlite3.Connection, table: str,
    pk_col: str, dedup_col: str,
) -> Dict[int, int]:
    """Copy src.table -> dst.table, dedup'd on `dedup_col`. Returns old_pk -> new_pk."""
    if not table_exists(src, table):
        return {}
    cols = fetch_columns(src, table)
    insert_cols = [c for c in cols if c != pk_col]
    placeholders = ",".join("?" for _ in insert_cols)
    insert_sql = (
        f"INSERT OR IGNORE INTO {table} ({','.join(insert_cols)}) VALUES ({placeholders})"
    )
    id_map: Dict[int, int] = {}
    src_cur = src.execute(f"SELECT {','.join(cols)} FROM {table}")
    for row in src_cur:
        row_dict = dict(zip(cols, row))
        old_pk = row_dict[pk_col]
        dedup_val = row_dict[dedup_col]
        dst.execute(insert_sql, [row_dict[c] for c in insert_cols])
        (new_pk,) = dst.execute(
            f"SELECT {pk_col} FROM {table} WHERE {dedup_col}=?", (dedup_val,)
        ).fetchone()
        id_map[old_pk] = new_pk
    return id_map


def merge_profiled_models(src, dst) -> Dict[int, int]:
    """Copy profiled_models rows. Shards profile disjoint (model, backend) sets,
    but we still INSERT OR IGNORE on (model_name, backend, tp_degree) to stay safe."""
    if not table_exists(src, "profiled_models"):
        return {}
    cols = fetch_columns(src, "profiled_models")
    insert_cols = [c for c in cols if c != "model_id"]
    placeholders = ",".join("?" for _ in insert_cols)
    id_map: Dict[int, int] = {}
    rows = src.execute(f"SELECT {','.join(cols)} FROM profiled_models").fetchall()
    for row in rows:
        row_dict = dict(zip(cols, row))
        old_id = row_dict["model_id"]
        # Check for existing row first.
        existing = dst.execute(
            "SELECT model_id FROM profiled_models "
            "WHERE model_name=? AND backend=? AND tp_degree=?",
            (row_dict["model_name"], row_dict["backend"], row_dict["tp_degree"]),
        ).fetchone()
        if existing:
            id_map[old_id] = existing[0]
            continue
        cur = dst.execute(
            f"INSERT INTO profiled_models ({','.join(insert_cols)}) VALUES ({placeholders})",
            [row_dict[c] for c in insert_cols],
        )
        id_map[old_id] = cur.lastrowid
    return id_map


def merge_with_fk_remap(
    src: sqlite3.Connection, dst: sqlite3.Connection, table: str,
    pk_col: str | None, fk_remaps: Dict[str, Dict[int, int]],
    unique_cols: tuple[str, ...] = (),
) -> None:
    """Copy src.table rows to dst, remapping foreign keys and dropping duplicates
    that would violate `unique_cols` (if provided)."""
    if not table_exists(src, table):
        return
    cols = fetch_columns(src, table)
    insert_cols = [c for c in cols if c != pk_col] if pk_col else cols
    placeholders = ",".join("?" for _ in insert_cols)
    verb = "INSERT OR IGNORE" if unique_cols else "INSERT"
    insert_sql = f"{verb} INTO {table} ({','.join(insert_cols)}) VALUES ({placeholders})"

    rows = src.execute(f"SELECT {','.join(cols)} FROM {table}").fetchall()
    for row in rows:
        row_dict = dict(zip(cols, row))
        for fk_col, mapping in fk_remaps.items():
            if fk_col in row_dict and row_dict[fk_col] in mapping:
                row_dict[fk_col] = mapping[row_dict[fk_col]]
        dst.execute(insert_sql, [row_dict[c] for c in insert_cols])


def scan_signature_result_stats(shard_path: Path) -> Dict[str, tuple[int, int]]:
    """For one shard, return {signature_hash: (row_count, max_num_tokens)}.

    Used by pick_result_owners to decide which shard "owns" each signature.
    """
    conn = sqlite3.connect(str(shard_path))
    try:
        return {
            sh: (cnt, int(max_nt or 0))
            for sh, cnt, max_nt in conn.execute(
                "SELECT signature_hash, COUNT(*), COALESCE(MAX(num_tokens), 0) "
                "FROM results GROUP BY signature_hash"
            )
        }
    finally:
        conn.close()


def pick_result_owners(shards: list[Path]) -> Dict[str, Path]:
    """For each signature_hash, pick the shard that profiled it most widely.

    Ranking key is (max_num_tokens, row_count) descending: a shard that swept
    the signature up to a larger num_tokens is a superset of one that stopped
    earlier (e.g. because that model had a smaller max_seq_len). Row count
    breaks ties.
    """
    best: Dict[str, tuple[tuple[int, int], Path]] = {}
    per_shard_counts: Dict[Path, int] = {}
    for shard in shards:
        stats = scan_signature_result_stats(shard)
        per_shard_counts[shard] = sum(cnt for cnt, _ in stats.values())
        for sh, (cnt, max_nt) in stats.items():
            key = (max_nt, cnt)
            if sh not in best or key > best[sh][0]:
                best[sh] = (key, shard)
    owners = {sh: path for sh, (_, path) in best.items()}
    print(f"[MERGE] Result ownership: {len(owners)} signatures assigned to shards")
    for shard in shards:
        owned = sum(1 for p in owners.values() if p == shard)
        print(f"  {shard.name}: owns {owned} signatures out of {len(scan_signature_result_stats(shard))} present")
    return owners


def merge_results_owned(
    src: sqlite3.Connection, dst: sqlite3.Connection, owned_sigs: set[str]
) -> None:
    """Copy only the `results` rows whose signature_hash this shard owns."""
    if not table_exists(src, "results") or not owned_sigs:
        return
    cols = fetch_columns(src, "results")
    insert_cols = [c for c in cols if c != "id"]
    placeholders = ",".join("?" for _ in insert_cols)
    insert_sql = f"INSERT INTO results ({','.join(insert_cols)}) VALUES ({placeholders})"

    # Chunk to avoid SQLite's 500-var IN-clause limit.
    sigs = list(owned_sigs)
    copied = 0
    CHUNK = 400
    for i in range(0, len(sigs), CHUNK):
        chunk = sigs[i:i + CHUNK]
        qmarks = ",".join("?" for _ in chunk)
        rows = src.execute(
            f"SELECT {','.join(cols)} FROM results WHERE signature_hash IN ({qmarks})",
            chunk,
        ).fetchall()
        for row in rows:
            row_dict = dict(zip(cols, row))
            dst.execute(insert_sql, [row_dict[c] for c in insert_cols])
            copied += 1
    print(f"    results: copied {copied} rows for {len(owned_sigs)} owned signatures")


def merge_shard(
    shard_db: Path,
    dst: sqlite3.Connection,
    owned_sigs_for_this_shard: set[str],
) -> None:
    """Merge one shard DB into the open destination connection.

    `owned_sigs_for_this_shard` filters which results rows get copied — only
    signatures this shard "owns" are imported from its results table.
    Signatures, profiled_models, model_operations, and coll_ops are merged
    normally regardless.
    """
    src = sqlite3.connect(str(shard_db))
    try:
        # 1. Content-addressed dedup: signatures, coll_ops. Builds remap tables.
        sig_map   = merge_dedup_table(src, dst, "signatures", "operation_id", "signature_hash")
        coll_map  = merge_dedup_table(src, dst, "coll_ops",   "collective_id", "signature_hash")
        # 2. profiled_models: disjoint across shards, dedup on (name, backend, tp).
        model_map = merge_profiled_models(src, dst)
        # 3. model_operations: FK-remap; UNIQUE(model, op, count) dedups.
        #    Multiple models mapping to the same operation_id is the whole
        #    point — signatures dedup'd above ensures exactly one operation_id
        #    per signature_hash, so every (shard,model)'s link lands there.
        merge_with_fk_remap(src, dst, "model_operations",
                            pk_col="id",
                            fk_remaps={"model_id": model_map, "operation_id": sig_map},
                            unique_cols=("model_id", "operation_id", "count"))
        # 4. results: only import signatures THIS shard owns. Other shards'
        #    rows for the same signature are dropped (a smaller-max_seq_len
        #    model's sweep is covered by the wider sweep of the owner).
        merge_results_owned(src, dst, owned_sigs_for_this_shard)
        # 5. Collective tables.
        merge_with_fk_remap(src, dst, "model_collectives",
                            pk_col="id",
                            fk_remaps={"model_id": model_map, "collective_id": coll_map},
                            unique_cols=("model_id", "collective_id"))
        merge_with_fk_remap(src, dst, "model_collective_configs",
                            pk_col="id",
                            fk_remaps={"model_id": model_map, "collective_id": coll_map},
                            unique_cols=("model_id", "collective_id"))
        merge_with_fk_remap(src, dst, "coll_op_results",
                            pk_col="id",
                            fk_remaps={},
                            unique_cols=("operation", "topology", "backend", "dtype",
                                         "tp_size", "size_bytes", "dim_param", "num_dims"))
        dst.commit()
    finally:
        src.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge per-GPU profiler DBs into one")
    ap.add_argument("--input", nargs="+", type=Path, required=True,
                    help="Paths of shard DBs (or a glob pattern).")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output merged DB path. Created fresh; refuses to overwrite.")
    ap.add_argument("--force", action="store_true",
                    help="Allow overwriting an existing output DB.")
    args = ap.parse_args()

    shards = []
    for p in args.input:
        if p.is_file():
            shards.append(p)
        else:
            # Allow glob-like patterns via the shell; if not expanded, try here.
            matches = list(p.parent.glob(p.name))
            shards.extend(m for m in matches if m.is_file())
    shards = sorted(set(shards))
    if not shards:
        print("[ERROR] No shard DB files found.", file=sys.stderr)
        return 2

    if args.output.exists():
        if not args.force:
            print(f"[ERROR] Output DB exists: {args.output} (use --force to overwrite)")
            return 2
        args.output.unlink()

    print(f"[MERGE] Target:  {args.output}")
    print(f"[MERGE] Shards:  {len(shards)}")
    for s in shards:
        print(f"          - {s}")

    # Start the output with a schema copied from the first shard.
    copy_schema(shards[0], args.output)
    # First pass: decide which shard "owns" each signature's results.
    result_owners = pick_result_owners(shards)
    dst = sqlite3.connect(str(args.output))
    try:
        for shard in shards:
            owned = {sh for sh, p in result_owners.items() if p == shard}
            print(f"[MERGE] Merging {shard.name} (owns {len(owned)} signatures for results) ...")
            merge_shard(shard, dst, owned)

        # Final stats.
        rows_models  = dst.execute("SELECT COUNT(*) FROM profiled_models").fetchone()[0]
        rows_sigs    = dst.execute("SELECT COUNT(*) FROM signatures").fetchone()[0]
        rows_results = dst.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        print(f"[MERGE] Done. profiled_models={rows_models}  signatures={rows_sigs}  results={rows_results}")
    finally:
        dst.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
