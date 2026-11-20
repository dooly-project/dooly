#!/bin/bash
# example.sh [dense] [moe] [hetero] [-- RUN-OPTIONS]: trace and profile three example models with
# docker/dooly-docker.sh, one after another (all three when none is named):
#
#   dense   meta-llama/Llama-3.1-8B   dense, tensor parallel 2, FLASH_ATTN. With two or more GPUs
#                                     it also measures the all-reduces between the two ranks on
#                                     two linked GPUs (--profile-comm); with one GPU, one rank only.
#   moe     Qwen/Qwen3-30B-A3B        mixture of experts (128 experts, 8 per token), FLASHINFER
#   hetero  openai/gpt-oss-20b        layers alternating sliding-window and full attention (also a
#                                     mixture of experts), TRITON_ATTN: vLLM runs gpt-oss's
#                                     attention sinks with Triton on A100 and Blackwell alike
#
# Each model's config and tokenizer are downloaded first unless already in HF_CACHE (no weights).
# Llama is gated: accept its license on huggingface.co and set HF_TOKEN for the download. The
# others need no login. Computation is profiled on one GPU (Qwen3-30B-A3B needs an 80 GB one).
#
# Results go to $DOOLY_OUT/example/NAME/ (default ./dooly-out/example/), each run's output to
# $DOOLY_OUT/example/NAME.log. Each example starts from an empty folder and replaces its previous
# results (a run into an existing profile.db skips every operation already in it). The script first
# says whether Docker Hub has a newer image than this machine. At the end a summary shows, per model, whether the run worked, its
# operations per forward pass and its collectives. Operations with one name but different shapes
# or attention settings are profiled separately and listed as kinds: gpt-oss shows two kinds of
# Attention, 12 calls each.
#
# The settings of dooly-docker.sh apply (DOCKER, DOOLY_OUT, HF_CACHE, DOOLY_CACHE, DOOLY_IMAGE,
# HF_TOKEN); options after "--" go to every run and win over the ones above, e.g.
#   docker/example.sh moe -- --attention-backend FLASH_ATTN --max-batch-size 4
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN="$HERE/dooly-docker.sh"
export DOOLY_OUT=${DOOLY_OUT:-$PWD/dooly-out}
HF=${HF_CACHE:-$HOME/.cache/huggingface}
IMAGE=${DOOLY_IMAGE:-jhkjames1110/doolyprof:latest}
DOCKER=${DOCKER:-docker}

declare -A MODEL=([dense]="meta-llama/Llama-3.1-8B" [moe]="Qwen/Qwen3-30B-A3B" [hetero]="openai/gpt-oss-20b")
declare -A WHAT=([dense]="dense, tensor parallel 2" [moe]="mixture of experts"
                 [hetero]="alternating sliding-window and full attention")
declare -A GPUS=([dense]=1 [moe]=1 [hetero]=1)
declare -A ARGS=([dense]="--tp 2 --attention-backend FLASH_ATTN"
                 [moe]="--attention-backend FLASHINFER --gpu-memory-cap 0.95"
                 [hetero]="--attention-backend TRITON_ATTN")
if [ "$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')" -ge 2 ]; then
  GPUS[dense]=2 ARGS[dense]+=" --profile-comm"
fi

names=() extra=()
while [ $# -gt 0 ]; do
  case $1 in
    --) shift; extra=("$@"); break ;;
    dense|moe|hetero) names+=("$1"); shift ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "example.sh: unknown example '$1' (dense, moe, hetero)" >&2; exit 2 ;;
  esac
done
[ ${#names[@]} -gt 0 ] || names=(dense moe hetero)
mkdir -p "$DOOLY_OUT/example"

"$RUN" check-update            # once here, not before each of the runs below
export DOOLY_UPDATE_CHECK=0
declare -A STATUS SECS
for n in "${names[@]}"; do
  m=${MODEL[$n]}
  [ -d "$HF/hub/models--${m//\//--}" ] && continue
  if ! "$RUN" download "$m"; then
    STATUS[$n]="download failed"
    [[ $m == meta-llama/* ]] && STATUS[$n]+=" (gated: accept the license on huggingface.co/$m, then set HF_TOKEN)"
  fi
done
for n in "${names[@]}"; do
  [ -n "${STATUS[$n]:-}" ] && continue
  echo
  echo "=== $n: ${MODEL[$n]} (${WHAT[$n]}) on ${GPUS[$n]} GPU(s): ${ARGS[$n]} ${extra[*]:-}"
  t0=$(date +%s)
  rm -rf "${DOOLY_OUT:?}/example/$n"   # this example's previous results, so the summary is this run's
  # shellcheck disable=SC2086  # ARGS holds several words on purpose
  "$RUN" --gpus "${GPUS[$n]}" run --model "${MODEL[$n]}" --out "/out/example/$n" ${ARGS[$n]} \
    ${extra[@]+"${extra[@]}"} 2>&1 | tee "$DOOLY_OUT/example/$n.log"
  rc=${PIPESTATUS[0]}
  SECS[$n]=$(( $(date +%s) - t0 ))
  if [ "$rc" = 0 ]; then STATUS[$n]=ok; else STATUS[$n]="failed (exit $rc, see $DOOLY_OUT/example/$n.log)"; fi
done

echo
echo "=== summary ($DOOLY_OUT/example)"
failed=0
for n in "${names[@]}"; do
  printf '%-7s %-24s %-6s %s\n' "$n" "${MODEL[$n]}" "${SECS[$n]:+${SECS[$n]}s}" "${STATUS[$n]}"
  [ "${STATUS[$n]}" = ok ] || failed=1
done
# operations and collectives per forward pass, read from each profile.db with the image's Python
$DOCKER run --rm -i -v "$DOOLY_OUT/example":/ex:ro --entrypoint python "$IMAGE" - "${names[@]}" <<'EOF'
import collections, os, sqlite3, sys
for name in sys.argv[1:]:
    db = f"/ex/{name}/profile.db"
    if not os.path.exists(db):
        continue
    con = sqlite3.connect(db)
    ops = con.execute("SELECT s.operation_name, mo.count FROM model_operations mo "
                      "JOIN signatures s ON s.operation_id = mo.operation_id").fetchall()
    rows = con.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    kinds = collections.defaultdict(list)
    for op, count in ops:
        kinds[op].append(count)
    print(f"\n{name}: {len(ops)} operations to profile, {rows} result rows; calls per forward pass:")
    for op in sorted(kinds, key=lambda o: -sum(kinds[o])):
        counts = sorted(kinds[op], reverse=True)
        more = f"   {len(counts)} kinds: {' + '.join(map(str, counts))}" if len(counts) > 1 else ""
        print(f"    {op:42s} {sum(counts):5d}{more}")
    colls = con.execute("SELECT c.operation_name, c.tp_degree, SUM(mc.count) FROM model_collectives mc "
                        "JOIN coll_ops c USING (collective_id) GROUP BY 1, 2").fetchall()
    for op, group, count in colls:
        print(f"    {op + f' (group of {group})':42s} {count:5d}   collective")
    measured = con.execute("SELECT topology, tp_size, COUNT(*) FROM coll_op_results GROUP BY 1, 2").fetchall()
    for topology, group, n in measured:
        print(f"    collective latencies measured: {n} on {topology}, groups of {group}")
EOF
exit $failed
