#!/usr/bin/env bash
#
# publish.sh — one-way, channel-aware publish of the doolyprof package from the
# private dev repo (doolyprof-dev) into the public release repo (doolyprof-public).
#
# A "channel" is a (dev-branch -> public-branch) pair. The script publishes the
# channel matching whatever branch doolyprof-dev is currently on, and REFUSES to
# run unless doolyprof-public is on the matching branch — so you can't cross
# wires (e.g. publish sglang content onto public main).
#
#   Channels:   main                   -> main
#               sglang-port-from-main  -> sglang-branch
#
# Publishing is a content copy only. This script NEVER commits or pushes: it
# syncs the publishable paths into the public working tree and then stops, so
# you can review `git diff` and commit/push yourself. Public-only files
# (LICENSE, README.md, requirements.txt, .gitignore, example_traces/) are never
# touched; only PUBLISH_PATHS are synced.
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

# dev-branch -> public-branch channel map
declare -A CHANNELS=(
  [main]="main"
  [sglang-port-from-main]="sglang-branch"
)

# Paths (relative to repo root, trailing slash) to publish. Synced with --delete.
PUBLISH_PATHS=(
  "doolyprof/"
)

EXCLUDES=(
  "__pycache__/"
  "*.pyc"
  ".DS_Store"
)

# Project-specific "publish gitignore": dev-only paths to keep out of public.
PUBLISHIGNORE="$SRC/.publishignore"
# --------------------------------------------------------------------------

usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; }

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

# Resolve the channel from dev's current branch, and verify public matches.
SRC_BRANCH="$(git -C "$SRC" rev-parse --abbrev-ref HEAD)"
DST_BRANCH="${CHANNELS[$SRC_BRANCH]:-}"
if [ -z "$DST_BRANCH" ]; then
  echo "ERROR: no publish channel configured for dev branch '$SRC_BRANCH'." >&2
  echo "       Known channels: ${!CHANNELS[*]}" >&2
  exit 1
fi
CUR_DST_BRANCH="$(git -C "$DST" rev-parse --abbrev-ref HEAD)"
if [ "$CUR_DST_BRANCH" != "$DST_BRANCH" ]; then
  echo "ERROR: channel '$SRC_BRANCH' publishes to public branch '$DST_BRANCH'," >&2
  echo "       but $DST is on '$CUR_DST_BRANCH'." >&2
  echo "       Switch it first:  git -C $DST checkout $DST_BRANCH" >&2
  exit 1
fi

# You publish what is committed, not stray working-tree edits.
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
# --checksum: decide by CONTENT, not size+mtime, so output matches git's view.
RSYNC_OPTS=(-a -c --delete --itemize-changes)
[ "$APPLY" -eq 1 ] || RSYNC_OPTS+=(--dry-run)
for e in "${EXCLUDES[@]}"; do RSYNC_OPTS+=(--exclude="$e"); done
[ -f "$PUBLISHIGNORE" ] && RSYNC_OPTS+=(--exclude-from="$PUBLISHIGNORE")

if [ "$APPLY" -eq 1 ]; then
  echo "==> PUBLISH (apply)   channel: $SRC_BRANCH -> $DST_BRANCH"
else
  echo "==> DRY-RUN (no changes written)   channel: $SRC_BRANCH -> $DST_BRANCH"
fi
echo "    source commit: $(git -C "$SRC" rev-parse --short HEAD)"
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
  echo "    git -C $DST diff                                    # review changes"
  echo "    git -C $DST add -A && git -C $DST commit -m 'Sync from doolyprof-dev ($SRC_BRANCH)'"
  echo "    git -C $DST push origin $DST_BRANCH"
else
  echo "(dry-run only — nothing changed. Lines above starting with '>f' / '*deleting' show what --apply would write.)"
fi
