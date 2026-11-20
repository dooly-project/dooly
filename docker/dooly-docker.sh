#!/bin/bash
# dooly-docker.sh [--gpus N | --devices LIST] <dooly command> [args...]: run the DoolyProf image on
# N GPUs (default 1) that no other user is on, or on the GPUs LIST names (host indices as nvidia-smi
# shows them, e.g. 2,3). See docker/README.md. e.g.
#
#   docker/dooly-docker.sh download Qwen/Qwen3-0.6B         (config and tokenizer files, no GPU)
#   docker/dooly-docker.sh run --model Qwen/Qwen3-0.6B --out /out/qwen3-0.6b
#   docker/dooly-docker.sh --gpus 2 run --model Qwen/Qwen3-8B --tp 2 --out /out/qwen3 --profile-comm
#   docker/dooly-docker.sh --devices 2,3 run --model Qwen/Qwen3-8B --tp 2 --out /out/qwen3 --profile-comm
#   docker/dooly-docker.sh test
#   docker/dooly-docker.sh check-update                     (is Docker Hub's image newer than yours?)
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
#   DOOLY_IMAGE  image (jhkjames1110/doolyprof:latest)  DOOLY_OUT  host folder mounted at /out ($PWD/dooly-out)
#   HF_CACHE     host Hugging Face cache, at /hf  DOOLY_CACHE  host folder for FlashInfer / Triton / vLLM
#                ($HOME/.cache/huggingface)                    caches, the container's HOME ($HOME/.cache/dooly-docker)
#   DOCKER       docker command (docker); "sudo docker" where your account is not in the docker group
#   HF_TOKEN     passed to download, for gated models (or a token saved by `hf auth login` in HF_CACHE)
#   DOOLY_UPDATE_CHECK  0: do not compare the image with Docker Hub's before each command (it only
#                warns when Docker Hub has a newer one; the copy on this machine is what runs)
#   DOOLY_*      every other DOOLY_ variable is passed into the container (e.g. DOOLY_TOPOLOGY, the
#                label collective measurements are stored under)
#
# The image works offline (HF_HUB_OFFLINE=1): `download` fetches a model's config and tokenizer files
# into HF_CACHE first. Weights are never needed: the tracer and profiler load dummy weights.
set -u
IMAGE=${DOOLY_IMAGE:-jhkjames1110/doolyprof:latest}
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

image_update_check() {  # 0: same as Docker Hub, 1: Docker Hub has a newer one (warned), 2: not checked
  [ "${DOOLY_UPDATE_CHECK:-1}" = 0 ] && return 2
  command -v curl > /dev/null || return 2
  local ref=${IMAGE#docker.io/} repo tag last digests token remote rev
  [[ $ref == *@* ]] && return 2                                   # pinned by digest
  last=${ref##*/}
  if [[ $last == *:* ]]; then tag=${last##*:}; repo=${ref%:*}; else tag=latest; repo=$ref; fi
  if [[ $repo == */* ]]; then
    case ${repo%%/*} in *.*|*:*|localhost) return 2 ;; esac       # another registry
  else
    repo=library/$repo
  fi
  # this machine's copy, as the digests Docker recorded when it was pulled or pushed (none if built here)
  digests=$($DOCKER image inspect --format '{{join .RepoDigests "\n"}}' "$IMAGE" 2> /dev/null) || return 2
  digests=$(sed -n "s|^\(docker.io/\)\{0,1\}${repo#library/}@||p" <<< "$digests")
  [ -n "$digests" ] || return 2
  # Docker Hub's: a HEAD request for the manifest (does not count as a pull)
  token=$(curl -fsS --max-time 4 "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$repo:pull" \
            2> /dev/null | sed -n 's/.*"token" *: *"\([^"]*\)".*/\1/p')
  [ -n "$token" ] || return 2
  remote=$(curl -fsSI --max-time 4 -H "Authorization: Bearer $token" \
             -H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json" \
             "https://registry-1.docker.io/v2/$repo/manifests/$tag" 2> /dev/null |
           tr -d '\r' | grep -i '^docker-content-digest:' | sed 's/^[^:]*: *//')
  [ -n "$remote" ] || return 2
  grep -qxF "$remote" <<< "$digests" && return 0
  rev=$($DOCKER image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMAGE" 2> /dev/null)
  echo "[dooly-docker] Docker Hub has a newer $IMAGE than this machine${rev:+ (yours was built from commit ${rev:0:7})}. To update:" >&2
  echo "  $DOCKER pull $IMAGE        (this run uses the copy here; DOOLY_UPDATE_CHECK=0 skips the check)" >&2
  return 1
}

if [ "${1:-}" = "check-update" ]; then
  DOOLY_UPDATE_CHECK=1 image_update_check
  case $? in
    0) echo "[dooly-docker] $IMAGE is up to date" >&2 ;;
    2) echo "[dooly-docker] could not compare $IMAGE with Docker Hub (not on this machine yet, built here, another registry, or no network)" >&2 ;;
  esac
  exit 0
fi
image_update_check

if [ "${1:-}" = "download" ]; then  # model config and tokenizer files into $HF; needs no GPU
  shift
  [ $# -gt 0 ] || { echo "usage: dooly-docker.sh download MODEL [MODEL ...]" >&2; exit 2; }
  mkdir -p "$HF" "$CACHE"
  token=(); [ -n "${HF_TOKEN:-}" ] && token=(-e HF_TOKEN)
  rc=0
  for model in "$@"; do
    echo "[dooly-docker] downloading $model into $HF (config and tokenizer files; no weights)" >&2
    $DOCKER run --rm --user "$(id -u):$(id -g)" -e HOME=/cache -e HF_HOME=/hf -e HF_HUB_OFFLINE=0 \
      ${token[@]+"${token[@]}"} -v "$HF":/hf -v "$CACHE":/cache --entrypoint hf "$IMAGE" download "$model" \
      --exclude "*.safetensors" --exclude "*.bin" --exclude "*.pt" --exclude "*.pth" --exclude "*.ckpt" \
      --exclude "*.gguf" --exclude "*.h5" --exclude "*.msgpack" --exclude "*.onnx" >&2 || rc=1
  done
  [ $rc = 0 ] && echo "[dooly-docker] done" >&2
  exit $rc
fi

if [ "${1:-}" = "run" ]; then  # the image is offline: say so before a GPU is claimed for nothing
  model="" prev=""
  for a in "$@"; do
    case $a in --model=*) model=${a#--model=} ;; esac
    [ "$prev" = "--model" ] && model=$a
    prev=$a
  done
  if [[ $model =~ ^[^/.][^/]*/[^/]+$ ]] && [ ! -d "$HF/hub/models--${model//\//--}" ]; then
    echo "[dooly-docker] $model is not in $HF (the image works offline): run" >&2
    echo "  docker/dooly-docker.sh download $model" >&2
    exit 2
  fi
fi

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
    DOOLY_IMAGE|DOOLY_OUT|DOOLY_CACHE|DOOLY_ANY_LINK|DOOLY_UPDATE_CHECK) ;;
    DOOLY_*) envs+=(-e "$name") ;;
  esac
done

$DOCKER run --rm --gpus "\"device=$devices\"" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" -e HOME=/cache -e HF_HOME=/hf ${envs[@]+"${envs[@]}"} \
  -v "$HF":/hf -v "$OUT":/out -v "$CACHE":/cache \
  "$IMAGE" "$@"
