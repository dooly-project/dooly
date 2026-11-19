#!/bin/bash
# dooly-docker.sh [--gpus N | --devices LIST] <dooly command> [args...]: run the DoolyProf image on
# N GPUs (default 1) that no other user is on, or on the GPUs LIST names (host indices as nvidia-smi
# shows them, e.g. 2,3), e.g.
#
#   docker/dooly-docker.sh run --model meta-llama/Llama-3.1-8B --out /out/llama
#   docker/dooly-docker.sh --gpus 2 run --model Qwen/Qwen3-8B --tp 2 --out /out/qwen3 --profile-comm
#   docker/dooly-docker.sh --devices 2,3 run --model Qwen/Qwen3-8B --tp 2 --out /out/qwen3 --profile-comm
#   docker/dooly-docker.sh test
#
# Inside the container the chosen GPUs are 0..N-1 in the order chosen (CUDA_VISIBLE_DEVICES; the
# first k GPUs form a k-rank group). Paths under /out are $DOOLY_OUT on the host.
# A container cannot see other users' processes, so the free-GPU check runs here: a GPU is free
# when it has under 4 GB in use, no process of another user, and the lock dir /tmp/jk_gpu<i>.lock
# can be created (the same lock the queue scripts use).
#
# Collectives are measured on the GPUs the container gets, so of the free GPUs the wrapper takes
# the N whose weakest link (nvidia-smi topo -m) is the strongest, ordered so that the first k of
# them are as well linked as they can be (a 2-rank group runs on GPUs 0 and 1 of the container).
# When the free GPUs are linked worse than the machine's best N (e.g. two GPUs of different NVLink
# pairs), it waits for better ones; DOOLY_ANY_LINK=1 takes them as they are. The choice is made by
# doolyprof/profiler/utils/topology.py next to this script (python3, standard library only);
# without it the first free GPUs are taken. --devices takes the GPUs named, in that order: it waits
# while one of them is in use or locked, and says when they are linked worse than the machine's best.
#
#   DOOLY_IMAGE  image (doolyprof:vllm0.33)       DOOLY_OUT    host folder mounted at /out ($PWD/dooly-out)
#   HF_CACHE     host Hugging Face cache, at /hf  DOOLY_CACHE  host folder for FlashInfer / Triton / vLLM
#                ($HOME/.cache/huggingface)                    caches, the container's HOME ($HOME/.cache/dooly-docker)
#   DOCKER       docker command (docker); "sudo docker" where your account is not in the docker group
#   DOOLY_*      every other DOOLY_ variable is passed into the container (e.g. DOOLY_TOPOLOGY, the
#                label collective measurements are stored under; DOOLY_CAP_MAX)
set -u
IMAGE=${DOOLY_IMAGE:-doolyprof:vllm0.33}
DOCKER=${DOCKER:-docker}
OUT=${DOOLY_OUT:-$PWD/dooly-out}
HF=${HF_CACHE:-$HOME/.cache/huggingface}
CACHE=${DOOLY_CACHE:-$HOME/.cache/dooly-docker}
TOPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../doolyprof/profiler/utils/topology.py"
N="" DEVICES=""
while [ $# -gt 0 ]; do
  case $1 in
    --gpus) N=$2; shift 2 ;;
    --devices) DEVICES=$2; shift 2 ;;
    *) break ;;
  esac
done
if [ -n "$DEVICES" ]; then
  count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  seen=,
  for g in ${DEVICES//,/ }; do
    if ! [[ $g =~ ^[0-9]+$ ]] || [ "$g" -ge "$count" ] || [[ $seen == *,$g,* ]]; then
      echo "[dooly-docker] --devices $DEVICES: '$g' is not one of GPUs 0-$((count - 1)) or is named twice" >&2
      exit 2
    fi
    seen+="$g,"
  done
  DEVICES=${seen#,}; DEVICES=${DEVICES%,}
  given=$(echo "$DEVICES" | tr , '\n' | grep -c .)
  if [ -n "$N" ] && [ "$N" != "$given" ]; then
    echo "[dooly-docker] --gpus $N and --devices $DEVICES ($given GPUs) disagree" >&2
    exit 2
  fi
  N=$given
fi
N=${N:-1}
mkdir -p "$OUT" "$CACHE"

free_gpus() {  # comma-separated indices of the GPUs nobody is on
  local free=() g uuid mem others
  local count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  for ((g = 0; g < count; g++)); do
    uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
    mem=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    others=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" |
             while IFS=, read u p; do [ "$(ps -o user= -p $p | tr -d ' ')" != "$USER" ] && echo $p; done)
    [ "$mem" -lt 4000 ] && [ -z "$others" ] && [ ! -e /tmp/jk_gpu$g.lock ] && free+=($g)
  done
  (IFS=,; echo "${free[*]}")
}

picker=1
if ! command -v python3 >/dev/null || [ ! -f "$TOPO" ]; then
  picker=0
  echo "[dooly-docker] no python3 or $TOPO: taking the first free GPUs, whatever their links" >&2
fi

claimed=()
release() { for g in ${claimed[@]+"${claimed[@]}"}; do rmdir /tmp/jk_gpu$g.lock 2>/dev/null; done; claimed=(); }
trap release EXIT
claim() {  # lock the chosen GPUs, all or none
  local g
  for g in ${1//,/ }; do
    if mkdir /tmp/jk_gpu$g.lock 2>/dev/null; then claimed+=($g); else release; return 1; fi
  done
}

said=""
while :; do
  free=$(free_gpus)
  chosen="" link="" best="" same="" busy=""
  if [ -n "$DEVICES" ]; then
    chosen=$DEVICES
    busy=$(for g in ${DEVICES//,/ }; do [[ ",$free," == *",$g,"* ]] || echo $g; done | paste -sd, -)
    [ $picker = 1 ] && read -r link best same < <(nvidia-smi topo -m | python3 -I "$TOPO" link "$DEVICES")
  elif [ $picker = 1 ]; then
    read -r chosen link best same < <(nvidia-smi topo -m | python3 -I "$TOPO" pick "$N" "$free")
  else
    chosen=$(echo "$free" | tr , '\n' | head -n "$N" | paste -sd, -)
    [ "$(echo "$chosen" | tr , '\n' | grep -c .)" -lt "$N" ] && chosen=""
  fi
  same=${same:-1}
  if [ -n "$busy" ]; then
    msg="waiting for GPU(s) $busy of --devices $DEVICES (in use or locked)"
  elif [ -z "$chosen" ]; then
    msg="waiting for $N free GPU(s) (free: ${free:-none})"
  elif [ "$same" != 1 ] && [ -z "$DEVICES" ] && [ "${DOOLY_ANY_LINK:-0}" != 1 ]; then
    msg="waiting for $N free GPUs linked by $best like the machine's best: free GPUs $chosen are linked by $link (DOOLY_ANY_LINK=1 takes them)"
  elif claim "$chosen"; then
    break
  else
    continue  # another run locked one of them first
  fi
  [ "$msg" != "$said" ] && echo "[dooly-docker] $msg" >&2 && said=$msg
  sleep 60
done
devices=$chosen
[ -n "$link" ] && [ "$N" -gt 1 ] && echo "[dooly-docker] GPUs $devices linked by $link (the machine's best $N: $best)" >&2
[ "$same" != 1 ] && echo "[dooly-docker] these GPUs are linked worse than the machine's best $N;" \
  "collectives measured on them are stored under the label of their own links" >&2
echo "[dooly-docker] GPUs $devices -> $IMAGE $*" >&2

# The container numbers its GPUs by PCI bus (the host's nvidia-smi order), whatever order docker
# is given: CUDA_VISIBLE_DEVICES puts them back in the order chosen.
sorted=$(echo "$devices" | tr , '\n' | sort -n | paste -sd, -)
order=$(for g in ${devices//,/ }; do echo "$sorted" | tr , '\n' | grep -nx "$g" | cut -d: -f1; done |
        awk '{print $1 - 1}' | paste -sd, -)
envs=(-e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="$order")
for name in $(compgen -e); do
  case $name in
    DOOLY_IMAGE|DOOLY_OUT|DOOLY_CACHE|DOOLY_ANY_LINK) ;;
    DOOLY_*) envs+=(-e "$name") ;;
  esac
done

$DOCKER run --rm --gpus "\"device=$devices\"" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" -e HOME=/cache -e HF_HOME=/hf ${envs[@]+"${envs[@]}"} \
  -v "$HF":/hf -v "$OUT":/out -v "$CACHE":/cache \
  "$IMAGE" "$@"
