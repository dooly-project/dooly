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
#   docker/dooly-docker.sh export --db /out/qwen3/profile.db --out /out/profiles   (no GPU; CSV tree)
#   docker/dooly-docker.sh import /out/profiles --db /out/qwen3/profile.db         (no GPU)
#   docker/dooly-docker.sh verify /out/profiles                                     (no GPU)
#   docker/dooly-docker.sh download-db /out/DoolyDB [--revision refs/pr/N]   (DoolyDB, or a pull request)
#   docker/dooly-docker.sh upload /out/DoolyDB "ORG/MODEL on GPU"            (a pull request on DoolyDB)
#   docker/dooly-docker.sh check-update                     (is Docker Hub's image newer than yours?)
#
# Every command needs a Hugging Face token: HF_TOKEN_FILE, the path of a file that holds it
# (preferred), or HF_TOKEN, the token itself. Only download, download-db and upload hand it to a
# container (the file mounted read-only, or the token in an env file only you can read), and they
# use no other: never a login saved on this machine (`hf auth login`).
#
# Inside the container the chosen GPUs are 0..N-1 in the order chosen (CUDA_VISIBLE_DEVICES; the
# first k GPUs form a k-rank group). Paths under /out are $DOOLY_OUT on the host.
# A container cannot see other users' processes, so the free-GPU check runs here: a GPU is free
# when it has under 4 GB in use, runs no kernels (utilization under 5% in each of three samples a
# second apart), has no process of another user, and no lock dir /tmp/dooly_gpu<i>.lock. On an
# account several people share, their processes look like yours: memory, utilization and the
# locks are what tell their GPUs apart.
# The script creates that lock for each GPU it takes and removes it when the run ends, so two runs
# never share a GPU, not even while one of them is starting and uses no memory yet.
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
#   HF_TOKEN_FILE  file holding your Hugging Face token   HF_TOKEN  the token itself (one is required)
#   DOOLY_DB_REPO  the dataset download-db and upload use (dooly-project/DoolyDB)
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
LOCK=/tmp/dooly_gpu   # $LOCK<i>.lock: GPU i is taken by a run of this script
TOPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../doolyprof/profiler/utils/topology.py"
DB_REPO=${DOOLY_DB_REPO:-dooly-project/DoolyDB}
N="" DEVICES=""
while [ $# -gt 0 ]; do
  case $1 in
    --gpus) N=$2; shift 2 ;;
    --devices) DEVICES=$2; shift 2 ;;
    *) break ;;
  esac
done

# The Hugging Face token, required by every command (see the top of this file).
if [ -n "${HF_TOKEN_FILE:-}" ]; then
  if [ ! -f "$HF_TOKEN_FILE" ] || [ ! -r "$HF_TOKEN_FILE" ] || [ ! -s "$HF_TOKEN_FILE" ]; then
    echo "[dooly-docker] HF_TOKEN_FILE=$HF_TOKEN_FILE is not a readable, non-empty file" >&2
    exit 2
  fi
  HF_TOKEN_FILE="$(cd "$(dirname -- "$HF_TOKEN_FILE")" && pwd -P)/$(basename -- "$HF_TOKEN_FILE")"
elif [ -z "${HF_TOKEN:-}" ]; then
  echo "[dooly-docker] a Hugging Face token is required: HF_TOKEN_FILE=/path/to/token.txt (a file that holds" >&2
  echo "  it; preferred) or HF_TOKEN=<token>. The commands that reach Hugging Face use it, and no saved login." >&2
  exit 2
fi
# The image's huggingface_hub is pinned: no hints to update it or to install its `hf` skill for agents.
HF_QUIET="-e HF_HUB_DISABLE_UPDATE_CHECK=1"
token=()   # docker run options handing the token to a container (token_options)
token_options() {  # the file read-only (HF_TOKEN_PATH), or the token in an env file only this user can read:
  # `-e HF_TOKEN` alone passes nothing through `sudo docker`, and `-e HF_TOKEN=...` would show it in the
  # process list. Either one is read instead of a login saved in HF_CACHE.
  if [ -n "${HF_TOKEN_FILE:-}" ]; then
    token=(-v "$HF_TOKEN_FILE":/run/dooly/hf_token:ro -e HF_TOKEN_PATH=/run/dooly/hf_token)
  else
    tokfile=$(mktemp) && chmod 600 "$tokfile" && printf 'HF_TOKEN=%s\n' "$HF_TOKEN" > "$tokfile"
    trap 'rm -f "$tokfile"' EXIT
    token=(--env-file "$tokfile")
  fi
}

MANIFESTS="application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"

hub_image() {  # hub_image REPO DIGEST TOKEN: "<created> <commit>" of that image on Docker Hub, from its
  # manifest and config (a few KB; not a pull). Prints nothing without python3 or when they cannot be read.
  command -v python3 > /dev/null || return 0
  python3 - "$@" "$MANIFESTS" <<'EOF' 2> /dev/null
import json, platform, subprocess, sys
repo, ref, token, manifests = sys.argv[1:]
def get(path, accept=manifests):  # curl -L does not send the token on to the blob store it redirects to
    out = subprocess.run(["curl", "-fsSL", "--max-time", "6", "-H", "Authorization: Bearer " + token,
                          "-H", "Accept: " + accept, f"https://registry-1.docker.io/v2/{repo}/{path}"],
                         capture_output=True, check=True).stdout
    return json.loads(out)
m = get("manifests/" + ref)
if "manifests" in m:  # one image per platform: this machine's
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine(), platform.machine())
    m = get("manifests/" + next(e["digest"] for e in m["manifests"]
                                if e.get("platform", {}).get("os") == "linux"
                                and e["platform"].get("architecture") == arch))
config = get("blobs/" + m["config"]["digest"], accept="*/*")
labels = (config.get("config") or {}).get("Labels") or {}
print(config.get("created") or "-", labels.get("org.opencontainers.image.revision") or "")
EOF
}

built() {  # built CREATED COMMIT: "built 2026-11-24 14:38 UTC from commit cf974cf"
  echo "built ${1:0:10} ${1:11:5} UTC${2:+ from commit ${2:0:7}}"
}

UPDATE_NOTE=""
image_update_check() {  # 0: same as Docker Hub's, 1: Docker Hub's is newer (warned), 2: not checked,
                        # 3: this machine's is not older than Docker Hub's, or another build of the
                        #    same commit (UPDATE_NOTE says which)
  [ "${DOOLY_UPDATE_CHECK:-1}" = 0 ] && return 2
  command -v curl > /dev/null || return 2
  local ref=${IMAGE#docker.io/} repo tag last created rev digests token remote hub hub_created hub_rev
  [[ $ref == *@* ]] && return 2                                   # pinned by digest
  last=${ref##*/}
  if [[ $last == *:* ]]; then tag=${last##*:}; repo=${ref%:*}; else tag=latest; repo=$ref; fi
  if [[ $repo == */* ]]; then
    case ${repo%%/*} in *.*|*:*|localhost) return 2 ;; esac       # another registry
  else
    repo=library/$repo
  fi
  # this machine's copy: when it was built, from which commit, and the digests Docker recorded
  # when it was pulled or pushed (none, or only a local one, if it was built here)
  created=$($DOCKER image inspect --format '{{.Created}}' "$IMAGE" 2> /dev/null) || return 2
  rev=$($DOCKER image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMAGE" 2> /dev/null)
  digests=$($DOCKER image inspect --format '{{join .RepoDigests "\n"}}' "$IMAGE" 2> /dev/null)
  digests=$(sed -n "s|^\(docker.io/\)\{0,1\}${repo#library/}@||p" <<< "$digests")
  # Docker Hub's: a HEAD request for the manifest (does not count as a pull)
  token=$(curl -fsS --max-time 4 "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$repo:pull" \
            2> /dev/null | sed -n 's/.*"token" *: *"\([^"]*\)".*/\1/p')
  [ -n "$token" ] || return 2
  remote=$(curl -fsSI --max-time 4 -H "Authorization: Bearer $token" -H "Accept: $MANIFESTS" \
             "https://registry-1.docker.io/v2/$repo/manifests/$tag" 2> /dev/null |
           tr -d '\r' | grep -i '^docker-content-digest:' | sed 's/^[^:]*: *//')
  [ -n "$remote" ] || return 2
  grep -qxF "$remote" <<< "$digests" && return 0
  # A different image: newer only if Docker Hub's was built later (both times are UTC, ISO 8601)
  hub=$(hub_image "$repo" "$remote" "$token")
  read -r hub_created hub_rev <<< "$hub"
  if [ -n "$rev" ] && [ "$rev" = "${hub_rev:-}" ]; then  # e.g. built here from the commit Docker Hub's is
    UPDATE_NOTE="this machine's $IMAGE ($(built "$created" "$rev")) and Docker Hub's ($(built "$hub_created" "$hub_rev")) are builds of the same commit"
    return 3
  fi
  if [[ ${hub_created:-} == [0-9]* ]]; then
    if [[ ! ${hub_created:0:19} > ${created:0:19} ]]; then
      UPDATE_NOTE="this machine's $IMAGE ($(built "$created" "$rev")) is not older than Docker Hub's ($(built "$hub_created" "${hub_rev:-}"))"
      return 3
    fi
    echo "[dooly-docker] Docker Hub has a newer $IMAGE ($(built "$hub_created" "${hub_rev:-}")) than this machine ($(built "$created" "$rev")). To update:" >&2
  else
    echo "[dooly-docker] Docker Hub's $IMAGE is not the one on this machine ($(built "$created" "$rev")) and may be newer. To update:" >&2
  fi
  echo "  $DOCKER pull $IMAGE        (this run uses the copy here; DOOLY_UPDATE_CHECK=0 skips the check)" >&2
  return 1
}

if [ "${1:-}" = "check-update" ]; then
  DOOLY_UPDATE_CHECK=1 image_update_check
  case $? in
    0) echo "[dooly-docker] $IMAGE is up to date" >&2 ;;
    2) echo "[dooly-docker] could not compare $IMAGE with Docker Hub (not on this machine yet, another registry, or no network)" >&2 ;;
    3) echo "[dooly-docker] $UPDATE_NOTE: nothing to update" >&2 ;;
  esac
  exit 0
fi
image_update_check

if [ "${1:-}" = "download" ]; then  # model config and tokenizer files into $HF; needs no GPU
  shift
  [ $# -gt 0 ] || { echo "usage: dooly-docker.sh download MODEL [MODEL ...]" >&2; exit 2; }
  mkdir -p "$HF" "$CACHE"
  token_options
  rc=0
  for model in "$@"; do
    echo "[dooly-docker] downloading $model into $HF (config and tokenizer files; no weights)" >&2
    $DOCKER run --rm --user "$(id -u):$(id -g)" -e HOME=/cache -e HF_HOME=/hf -e HF_HUB_OFFLINE=0 $HF_QUIET \
      ${token[@]+"${token[@]}"} -v "$HF":/hf -v "$CACHE":/cache --entrypoint hf "$IMAGE" download "$model" \
      --exclude "*.safetensors" --exclude "*.bin" --exclude "*.pt" --exclude "*.pth" --exclude "*.ckpt" \
      --exclude "*.gguf" --exclude "*.h5" --exclude "*.msgpack" --exclude "*.onnx" >&2 || rc=1
  done
  [ $rc = 0 ] && echo "[dooly-docker] done" >&2
  exit $rc
fi

under_out() {  # $1 must be a path under /out, the container's view of $DOOLY_OUT
  case $1 in /out/?*) return 0 ;; esac
  echo "[dooly-docker] $2: '$1' must be a folder under /out/ (\$DOOLY_OUT, here $OUT)" >&2
  exit 2
}

if [ "${1:-}" = "download-db" ]; then  # DoolyDB (or a revision of it, e.g. refs/pr/N) into a folder; no GPU
  shift
  [ $# -ge 1 ] || { echo "usage: dooly-docker.sh download-db /out/DIR [--revision REV]" >&2; exit 2; }
  dir=$1; shift
  under_out "$dir" download-db
  mkdir -p "$OUT"
  token_options
  echo "[dooly-docker] downloading $DB_REPO${*:+ ($*)} into $dir" >&2
  $DOCKER run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=0 $HF_QUIET \
    "${token[@]}" -v "$OUT":/out --entrypoint hf "$IMAGE" download "$DB_REPO" --repo-type dataset \
    --local-dir "$dir" "$@" >&2
  exit $?
fi

if [ "${1:-}" = "upload" ]; then  # a folder as a pull request on DoolyDB, after dooly verify; no GPU
  shift
  [ $# -eq 2 ] || { echo "usage: dooly-docker.sh upload /out/DIR \"ORG/MODEL on GPU\"" >&2; exit 2; }
  dir=$1 message=$2
  under_out "$dir" upload
  if ! $DOCKER run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$OUT":/out:ro "$IMAGE" verify "$dir"; then
    echo "[dooly-docker] not uploaded: dooly verify found problems (docs/how-to-contribute.md, If verify fails)" >&2
    exit 1
  fi
  token_options
  $DOCKER run --rm -i --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=0 $HF_QUIET \
    "${token[@]}" -v "$OUT":/out:ro --entrypoint python "$IMAGE" - "$DB_REPO" "$dir" "$message" <<'EOF'
# Always a pull request, never a commit to main; then what it changes compared with main.
import sys
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile
repo, folder, message = sys.argv[1:]
api = HfApi()
try:
    who = api.whoami()["name"]
except Exception as e:  # noqa: BLE001
    sys.exit(f"[dooly-docker] the Hugging Face token was not accepted: {e}")
print(f"[dooly-docker] uploading {folder} to {repo} as {who}, as a pull request", file=sys.stderr, flush=True)
info = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=folder, commit_message=message,
                         create_pr=True)
if not info.pr_url:
    sys.exit(f"[dooly-docker] nothing uploaded: {folder} has no file that {repo} does not already have")

def files(rev):
    return {f.path: f.blob_id for f in api.list_repo_tree(repo, repo_type="dataset", revision=rev, recursive=True)
            if isinstance(f, RepoFile)}
main, pr = files("main"), files(info.pr_revision)
added = sorted(set(pr) - set(main))
changed = sorted(p for p in set(pr) & set(main) if pr[p] != main[p])
removed = sorted(set(main) - set(pr))
folders = sorted({p.rsplit("/", 1)[0] for p in added})
print(f"[dooly-docker] pull request #{info.pr_num}: {info.pr_url}")
print(f"[dooly-docker] compared with main it adds {len(added)} file(s), changes {len(changed)}, removes {len(removed)}")
for f in folders:
    print(f"    + {f}/")
for p in changed:
    print(f"    ~ {p}")
for p in removed:
    print(f"    - {p}")
print("[dooly-docker] it is a draft: check it, then publish it on its page", flush=True)
EOF
  exit $?
fi

if [ "${1:-}" = "export" ] || [ "${1:-}" = "import" ] || [ "${1:-}" = "verify" ]; then  # database <-> CSV tree; no GPU
  mkdir -p "$OUT"
  exec $DOCKER run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$OUT":/out "$IMAGE" "$@"
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
mkdir -p "$OUT" "$CACHE" "$HF"   # before docker run -v, which would make them root-owned

free_gpus() {  # comma-separated indices of the GPUs nobody is on
  local free=() g uuid mem others active s
  local count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  # GPUs running kernels: 5% or more in any of three samples a second apart (one sample can fall
  # between a job's kernels). This also catches jobs whose processes the user check below cannot
  # tell from ours (an account several people share).
  active=$(for s in 1 2 3; do
             [ $s = 1 ] || sleep 1
             nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits
           done | awk -F', *' '$2 + 0 >= 5 { print $1 }' | sort -un | paste -sd, -)
  for ((g = 0; g < count; g++)); do
    uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
    mem=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    others=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" |
             while IFS=, read u p; do [ "$(ps -o user= -p $p | tr -d ' ')" != "$USER" ] && echo $p; done)
    [ "$mem" -lt 4000 ] && [ -z "$others" ] && [[ ",$active," != *",$g,"* ]] && [ ! -e $LOCK$g.lock ] &&
      free+=($g)
  done
  (IFS=,; echo "${free[*]}")
}

picker=1
if ! command -v python3 >/dev/null || [ ! -f "$TOPO" ]; then
  picker=0
  echo "[dooly-docker] no python3 or $TOPO: taking the first free GPUs, whatever their links" >&2
fi

claimed=()
release() { for g in ${claimed[@]+"${claimed[@]}"}; do rmdir $LOCK$g.lock 2>/dev/null; done; claimed=(); }
trap release EXIT
claim() {  # lock the chosen GPUs, all or none
  local g
  for g in ${1//,/ }; do
    if mkdir $LOCK$g.lock 2>/dev/null; then claimed+=($g); else release; return 1; fi
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
    DOOLY_IMAGE|DOOLY_OUT|DOOLY_CACHE|DOOLY_ANY_LINK|DOOLY_UPDATE_CHECK|DOOLY_DB_REPO) ;;
    DOOLY_*) envs+=(-e "$name=${!name}") ;;  # with its value: `-e NAME` alone passes nothing through sudo
  esac
done

$DOCKER run --rm --gpus "\"device=$devices\"" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" -e HOME=/cache -e HF_HOME=/hf ${envs[@]+"${envs[@]}"} \
  -v "$HF":/hf -v "$OUT":/out -v "$CACHE":/cache \
  "$IMAGE" "$@"
