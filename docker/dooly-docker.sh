#!/bin/bash
# dooly-docker.sh [--gpus N] <dooly command> [args...]: run the DoolyProf image on N GPUs (default 1)
# that no other user is on, e.g.
#
#   docker/dooly-docker.sh run --model meta-llama/Llama-3.1-8B --tp 2 --out /out/llama
#   docker/dooly-docker.sh test
#
# Inside the container the chosen GPUs are 0..N-1. Paths under /out are $DOOLY_OUT on the host.
# A container cannot see other users' processes, so the free-GPU check runs here: a GPU is taken
# when it has under 4 GB in use, no process of another user, and the lock dir /tmp/jk_gpu<i>.lock
# can be created (the same lock the queue scripts use).
#
#   DOOLY_IMAGE  image (doolyprof:vllm0.33)       DOOLY_OUT    host folder mounted at /out ($PWD/dooly-out)
#   HF_CACHE     host Hugging Face cache, at /hf  DOOLY_CACHE  host folder for FlashInfer / Triton / vLLM
#                ($HOME/.cache/huggingface)                    caches, the container's HOME ($HOME/.cache/dooly-docker)
#   DOCKER       docker command (docker); "sudo docker" where your account is not in the docker group
set -u
IMAGE=${DOOLY_IMAGE:-doolyprof:vllm0.33}
DOCKER=${DOCKER:-docker}
OUT=${DOOLY_OUT:-$PWD/dooly-out}
HF=${HF_CACHE:-$HOME/.cache/huggingface}
CACHE=${DOOLY_CACHE:-$HOME/.cache/dooly-docker}
N=1
if [ "${1:-}" = "--gpus" ]; then N=$2; shift 2; fi
mkdir -p "$OUT" "$CACHE"

claimed=()
release() { for g in "${claimed[@]}"; do rmdir /tmp/jk_gpu$g.lock 2>/dev/null; done; }
trap release EXIT
claim() {
  local count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  for ((g = 0; g < count; g++)); do
    [ ${#claimed[@]} -ge $N ] && return 0
    local uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
    local mem=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    local others=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" |
                   while IFS=, read u p; do [ "$(ps -o user= -p $p | tr -d ' ')" != "$USER" ] && echo $p; done)
    if [ "$mem" -lt 4000 ] && [ -z "$others" ] && mkdir /tmp/jk_gpu$g.lock 2>/dev/null; then claimed+=($g); fi
  done
  [ ${#claimed[@]} -ge $N ]
}
until claim; do release; claimed=(); echo "[dooly-docker] waiting for $N free GPU(s)" >&2; sleep 60; done
devices=$(IFS=,; echo "${claimed[*]}")
echo "[dooly-docker] GPUs $devices -> $IMAGE $*" >&2

$DOCKER run --rm --gpus "\"device=$devices\"" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" -e HOME=/cache -e HF_HOME=/hf \
  -v "$HF":/hf -v "$OUT":/out -v "$CACHE":/cache \
  "$IMAGE" "$@"
