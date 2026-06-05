#!/usr/bin/env bash
#
# publish.sh — one-way publish of the doolyprof package from the private dev
# repo (doolyprof-dev) into the public release repo (doolyprof-public).
#
# Publishing is a content copy only. This script NEVER commits or pushes: it
# syncs the publishable paths into the public working tree and then stops, so
# you can review `git diff` and commit/push yourself.
#
# Only the paths in PUBLISH_PATHS are touched. Everything else in the public
# repo (LICENSE, example_traces/, README.md, requirements.txt, .gitignore) is
# left untouched — those are public-only and intentionally diverge from dev.
#
# Usage:
#   ./publish.sh                       # dry-run: show what WOULD change (default)
#   ./publish.sh --apply               # actually sync into the public working tree
#   ./publish.sh --apply --allow-dirty # sync even if dev has uncommitted changes
#
set -euo pipefail

# ---- config --------------------------------------------------------------
SRC="${SRC:-$HOME/dooly-project/doolyprof-dev}"      # private source of truth
DST="${DST:-$HOME/dooly-project/doolyprof-public}"   # public release repo

# Paths (relative to repo root, trailing slash) to publish. Each is synced
# with --delete so files removed in dev are also removed in public.
PUBLISH_PATHS=(
  "doolyprof/"
)

# Always-on excludes (build/cache noise).
EXCLUDES=(
  "__pycache__/"
  "*.pyc"
  ".DS_Store"
)

# Project-specific "publish gitignore": dev-only paths to keep out of the
# public repo. Uses rsync filter syntax (one pattern per line). See .publishignore.
PUBLISHIGNORE="$SRC/.publishignore"
# --------------------------------------------------------------------------

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }

APPLY=0
ALLOW_DIRTY=0
for arg in "$@"; do
  case "$arg" in
    --apply)       APPLY=1 ;;
    --allow-dirty) ALLOW_DIRTY=1 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "unknown arg: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# ---- preflight -----------------------------------------------------------
[ -d "$SRC/.git" ] || { echo "ERROR: source is not a git repo: $SRC" >&2; exit 1; }
[ -d "$DST/.git" ] || { echo "ERROR: dest is not a git repo: $DST"   >&2; exit 1; }

# You publish what is committed, not stray working-tree edits. Refuse a dirty
# source under the published paths unless explicitly overridden.
for p in "${PUBLISH_PATHS[@]}"; do
  if [ -n "$(git -C "$SRC" status --porcelain -- "$p")" ]; then
    if [ "$ALLOW_DIRTY" -eq 0 ]; then
      echo "ERROR: '$p' has uncommitted changes in $SRC." >&2
      echo "       Commit them first, or re-run with --allow-dirty." >&2
      exit 1
    fi
    echo "WARNING: publishing with uncommitted changes in '$p' (--allow-dirty)."
  fi
done

# ---- build rsync args ----------------------------------------------------
# --checksum: decide by file CONTENT, not size+mtime. Identical files are
# skipped entirely, so the itemized output matches exactly what git will see
# as changed (mtime-only differences are ignored).
RSYNC_OPTS=(-a -c --delete --itemize-changes)
[ "$APPLY" -eq 1 ] || RSYNC_OPTS+=(--dry-run)
for e in "${EXCLUDES[@]}"; do RSYNC_OPTS+=(--exclude="$e"); done
[ -f "$PUBLISHIGNORE" ] && RSYNC_OPTS+=(--exclude-from="$PUBLISHIGNORE")

if [ "$APPLY" -eq 1 ]; then
  echo "==> PUBLISH (apply):  $SRC  ->  $DST"
else
  echo "==> DRY-RUN (no changes written). Re-run with --apply to sync."
fi
echo "    source commit: $(git -C "$SRC" rev-parse --short HEAD) on $(git -C "$SRC" rev-parse --abbrev-ref HEAD)"
echo

for p in "${PUBLISH_PATHS[@]}"; do
  echo "--- $p ---"
  rsync "${RSYNC_OPTS[@]}" "$SRC/$p" "$DST/$p"
done

# ---- review --------------------------------------------------------------
echo
if [ "$APPLY" -eq 1 ]; then
  echo "==> Synced. Review and publish from the public repo:"
  git -C "$DST" status -s
  echo
  echo "    git -C $DST diff                              # review changes"
  echo "    git -C $DST add -A && git -C $DST commit -m 'Sync from doolyprof-dev'"
  echo "    git -C $DST push origin main"
else
  echo "(dry-run only — nothing changed. Lines above starting with '>f' / '*deleting' show what --apply would write.)"
fi
