#!/bin/bash
# gcp.sh [STEP] [-- ARGS]: run DoolyProf on a Google Cloud GPU VM from your own machine (macOS or
# Linux; needs only gcloud, logged in to a project, and git here).
#
# Steps (default: all):
#   up      create the VM (MACHINE, Spot by default) from Google's Deep Learning VM image with
#           driver 580, in the first zone that has the machine type, a subnet of NETWORK in its
#           region, and capacity
#   setup   on the VM: wait for the driver, add Docker and the NVIDIA Container Toolkit if the
#           image lacks them, pull the DoolyProf image, copy docker/ and doolyprof/ of HEAD
#           (committed files only); with LOCAL_IMAGE=1 also build a thin image of HEAD on top of it
#   run     `dooly-docker.sh test`, then `example.sh ARGS`, detached on the VM; waits for them
#   exec    run the command ARGS in the VM's copy of the repository (e.g. `exec -- docker/dooly-docker.sh
#           --gpus 2 run --model M --tp 2 --out /out/m --profile-comm`), detached; waits for it
#   fetch   copy the VM's dooly-out/ and its logs to ./dooly-out/gcp/VM/
#   ssh     open a shell on the VM
#   down    delete the VM
#   all     up, setup, run, fetch, down (down also on failure, unless KEEP=1)
#
# run and exec copy the Hugging Face token file to the VM first (stdin, mode 600, never printed)
# and set HF_TOKEN_FILE there, and DOOLY_IMAGE=doolyprof:local with LOCAL_IMAGE=1.
#
# Settings (environment):
#   HF_TOKEN_FILE  file that holds your Hugging Face token (required for run and exec)
#   MACHINE        machine type (default a3-highgpu-2g). Two GPUs let example.sh also measure the
#                  dense model's collectives. E.g. a3-highgpu-2g (2x H100 80GB, NVLink; Spot or
#                  Flex-start only), a2-ultragpu-2g (2x A100 80GB SXM, NVLink), g4-standard-96
#                  (2x RTX PRO 6000 Blackwell), g2-standard-24 (2x L4), a4-highgpu-8g (8x B200)
#   GCP_PROJECT    project (default: gcloud's configured one)
#   GCP_ZONES      zones to try, in order (default: every zone that offers MACHINE, US first)
#   NETWORK        VPC network (default "default"); zones whose region has no subnet are skipped
#   PROVISIONING   SPOT (default), STANDARD (on demand) or FLEX_START
#   MAX_RUN        the VM deletes itself after this long, whatever happens (default 3h)
#   VM             instance name (default dooly-MACHINE, so VMs of different machines can coexist)
#   DISK_GB        boot disk size (default 200); DISK_TYPE (default pd-balanced, hyperdisk-balanced
#                  on machine families that only take Hyperdisk: g4, a4, a3-ultragpu)
#   LOCAL_IMAGE=1  build and use doolyprof:local from HEAD (to try a fix; never pushed)
#   KEEP=1         with `all`: keep the VM at the end (delete it later with `down`)
#
# Examples:
#   HF_TOKEN_FILE=~/.hf-token.txt docker/gcp.sh                                  # H100, all examples
#   MACHINE=a2-ultragpu-2g HF_TOKEN_FILE=~/.hf-token.txt docker/gcp.sh all -- moe hetero
#   MACHINE=g4-standard-96 LOCAL_IMAGE=1 docker/gcp.sh up; ... setup; ... exec -- CMD; ... down
set -euo pipefail

case ${1:-} in -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;; esac

MACHINE=${MACHINE:-a3-highgpu-2g}
VM=${VM:-dooly-$MACHINE}
PROVISIONING=${PROVISIONING:-SPOT}
MAX_RUN=${MAX_RUN:-3h}
DISK_GB=${DISK_GB:-200}
case $MACHINE in
  g4-*|a4-*|a4x-*|a3-ultragpu-*) DISK_TYPE=${DISK_TYPE:-hyperdisk-balanced} ;;
  *) DISK_TYPE=${DISK_TYPE:-pd-balanced} ;;
esac
NETWORK=${NETWORK:-default}
IMAGE_FAMILY=${IMAGE_FAMILY:-common-cu129-ubuntu-2404-nvidia-580}
DOOLY_IMAGE=${DOOLY_IMAGE:-jhkjames1110/doolyprof:latest}
REMOTE=dooly                       # folder in the VM user's home
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
LOCAL_OUT=${LOCAL_OUT:-$ROOT/dooly-out/gcp/$VM}

die() { echo "gcp: $*" >&2; exit 1; }
log() { echo "[gcp $VM] $*" >&2; }

command -v gcloud >/dev/null || die "gcloud not found: install the Google Cloud SDK, then gcloud auth login"
PROJECT=${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}
[ -n "$PROJECT" ] || die "no project: set GCP_PROJECT or run gcloud config set project PROJECT"

zone_of() {  # the zone the VM is in, or nothing
  gcloud compute instances list --project "$PROJECT" --filter="name=$VM" \
    --format='value(zone.basename())' 2>/dev/null | head -1
}
gssh() {  # gssh COMMAND: run COMMAND on the VM (stdin passes through)
  local zone; zone=$(zone_of); [ -n "$zone" ] || die "no VM $VM in project $PROJECT (preempted, or not up yet?)"
  gcloud compute ssh "$VM" --project "$PROJECT" --zone "$zone" --quiet --command "$1"
}

step_up() {
  if [ -n "$(zone_of)" ]; then log "already exists in $(zone_of)"; return; fi
  local zones=${GCP_ZONES:-}
  if [ -z "$zones" ]; then
    zones=$(gcloud compute machine-types list --project "$PROJECT" --filter="name=$MACHINE" \
      --format='value(zone)' | sort | awk '/^us-/ { print; next } { rest = rest $0 "\n" } END { printf "%s", rest }' \
      | tr '\n' ' ')   # US zones first
    [ -n "$zones" ] || die "no zone offers machine type $MACHINE"
  fi
  local prov=(--provisioning-model="$PROVISIONING" --max-run-duration="$MAX_RUN"
              --instance-termination-action=DELETE)
  if [ "$PROVISIONING" = FLEX_START ]; then
    prov+=(--reservation-affinity=none --request-valid-for-duration=2h)
  fi
  local tried=0 zone subnet
  for zone in $zones; do
    subnet=$(gcloud compute networks subnets list --project "$PROJECT" --network "$NETWORK" \
      --regions "${zone%-*}" --format='value(name)' 2>/dev/null | head -1)
    [ -n "$subnet" ] || continue    # no subnet of NETWORK in that region
    tried=$((tried + 1))
    log "creating ($MACHINE, $PROVISIONING) in $zone"
    if gcloud compute instances create "$VM" --project "$PROJECT" --zone "$zone" \
        --machine-type "$MACHINE" --maintenance-policy TERMINATE "${prov[@]}" \
        --image-family "$IMAGE_FAMILY" --image-project deeplearning-platform-release \
        --boot-disk-size "${DISK_GB}GB" --boot-disk-type "$DISK_TYPE" \
        --network-interface="nic-type=GVNIC,subnet=$subnet" \
        --metadata install-nvidia-driver=True; then
      log "up in $zone; it deletes itself after $MAX_RUN"
      log "waiting for SSH"
      for _ in $(seq 30); do gssh true 2>/dev/null && return; sleep 10; done
      die "VM is up but SSH does not answer; try: $0 ssh"
    fi
    log "no luck in $zone (capacity, quota or disk type; gcloud says why above)"
  done
  [ $tried -gt 0 ] || die "none of the zones that offer $MACHINE is in a region with a subnet of network $NETWORK"
  die "could not create $VM in any of its $tried zones (check the regional GPU quota for $PROVISIONING)"
}

step_setup() {
  log "copying docker/, doolyprof/ and doolysim/ of $(git -C "$ROOT" rev-parse --short HEAD) (committed files only)"
  git -C "$ROOT" archive --format=tar HEAD docker doolyprof doolysim \
    | gssh "rm -rf ~/$REMOTE/docker ~/$REMOTE/doolyprof ~/$REMOTE/doolysim && mkdir -p ~/$REMOTE && tar -x -C ~/$REMOTE"
  # shellcheck disable=SC2087  # expanded here on purpose: DOOLY_IMAGE, LOCAL_IMAGE and the revision
  gssh "bash -s" <<EOF
set -euo pipefail
mkdir -p ~/.cache/huggingface ~/.cache/dooly-docker   # ours, not created root-owned by docker -v
echo "[vm] waiting for the NVIDIA driver"
for i in \$(seq 60); do nvidia-smi -L >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
major=\$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "\$major" -ge 580 ] || { echo "[vm] driver \$major < 580 (the image needs CUDA 13.0)"; exit 1; }
nvidia-smi topo -m
if ! command -v docker >/dev/null; then
  echo "[vm] installing Docker"; sudo apt-get update -q && sudo apt-get install -yq docker.io
fi
if ! command -v nvidia-ctk >/dev/null; then
  echo "[vm] installing the NVIDIA Container Toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -q && sudo apt-get install -yq nvidia-container-toolkit
fi
sudo nvidia-ctk runtime configure --runtime=docker >/dev/null && sudo systemctl restart docker
sudo usermod -aG docker "\$USER"      # takes effect on the next SSH session
echo "[vm] pulling $DOOLY_IMAGE (8 GB)"
sudo docker pull -q $DOOLY_IMAGE
if [ "${LOCAL_IMAGE:-}" = 1 ]; then  # thin image of HEAD, as in docker/AGENTS.md
  echo "[vm] building doolyprof:local from $(git -C "$ROOT" rev-parse --short HEAD)"
  cd ~/$REMOTE && sudo docker build -q -t doolyprof:local --build-arg DOOLY_REVISION=$(git -C "$ROOT" rev-parse HEAD) -f - . <<'DOCKERFILE'
FROM $DOOLY_IMAGE
RUN rm -rf /opt/dooly/doolyprof /opt/dooly/doolysim
COPY doolyprof /opt/dooly/doolyprof
COPY doolysim /opt/dooly/doolysim
ARG DOOLY_REVISION=
RUN if [ -n "\$DOOLY_REVISION" ]; then echo "\$DOOLY_REVISION" > /opt/dooly/doolyprof/REVISION; fi
LABEL org.opencontainers.image.revision=\$DOOLY_REVISION
DOCKERFILE
fi
sudo docker run --rm --gpus all --entrypoint nvidia-smi $DOOLY_IMAGE -L
EOF
}

start_detached() {  # start_detached NAME COMMAND: run COMMAND in ~/REMOTE on the VM, detached, and wait
  local name=$1 cmd=$2
  [ -n "${HF_TOKEN_FILE:-}" ] && [ -r "$HF_TOKEN_FILE" ] && [ -s "$HF_TOKEN_FILE" ] \
    || die "HF_TOKEN_FILE must name a readable, non-empty file that holds your Hugging Face token"
  log "copying the token to ~/.hf-token.txt on the VM (mode 600)"
  gssh "umask 077; cat > ~/.hf-token.txt" < "$HF_TOKEN_FILE"
  local image_env=""
  [ "${LOCAL_IMAGE:-}" = 1 ] && image_env="export DOOLY_IMAGE=doolyprof:local && "
  log "starting on the VM: $cmd"
  printf '%s\n' "$cmd" | gssh "cat > ~/$name.cmd && cd ~/$REMOTE && rm -f ~/$name.rc && \
    export HF_TOKEN_FILE=\$HOME/.hf-token.txt && $image_env\
    { nohup bash -c 'bash ~/$name.cmd; echo \$? > ~/$name.rc' > ~/$name.log 2>&1 < /dev/null & }"
  # only nohup goes to the background: a background job's stdin is /dev/null, so cat must not be in it
  log "running; progress below, log on the VM: ~/$name.log"
  local last="" line
  while :; do
    sleep 60
    [ -n "$(zone_of)" ] || die "the VM is gone (Spot preemption or MAX_RUN reached); results are lost"
    line=$(gssh "tail -n 1 ~/$name.log; cat ~/$name.rc 2>/dev/null | sed 's/^/RC=/'" 2>/dev/null || true)
    case $line in *RC=*) RC=${line##*RC=}; break ;; esac
    [ "$line" != "$last" ] && { echo "  $line" | cut -c1-160; last=$line; }
  done
  log "finished with exit $RC"
}

quoted() { local out="" a; for a in "$@"; do out+=" $(printf %q "$a")"; done; echo "${out# }"; }

step_fetch() {
  local zone; zone=$(zone_of); [ -n "$zone" ] || die "no VM to fetch from"
  mkdir -p "$LOCAL_OUT"
  log "copying dooly-out/ and the logs to $LOCAL_OUT"
  gssh "cd ~ && tar -c --ignore-failed-read --transform 's,^$REMOTE/,,' $REMOTE/dooly-out *.log 2>/dev/null" | tar -x -C "$LOCAL_OUT" \
    || log "could not fetch everything"
  if [ -f "$LOCAL_OUT/run.log" ]; then sed -n '/=== summary/,$p' "$LOCAL_OUT/run.log"; fi
}

step_down() {
  local zone; zone=$(zone_of)
  [ -n "$zone" ] || { log "no VM to delete"; return 0; }
  log "deleting it in $zone"
  gcloud compute instances delete "$VM" --project "$PROJECT" --zone "$zone" --quiet
}

STEP=all
case ${1:-} in up|setup|run|exec|fetch|down|ssh|all) STEP=$1; shift ;; --|"") ;;
  *) die "unknown step '$1' (up, setup, run, exec, fetch, down, ssh, all)" ;; esac
[ "${1:-}" = -- ] && shift
RC=1
case $STEP in
  up) step_up ;;
  setup) step_setup ;;
  run) start_detached run "docker/dooly-docker.sh test && docker/example.sh $(quoted "$@")"; exit "$RC" ;;
  exec) [ $# -gt 0 ] || die "usage: $0 exec -- COMMAND [ARGS]"
        start_detached exec "$(quoted "$@")"; exit "$RC" ;;
  fetch) step_fetch ;;
  down) step_down ;;
  ssh) zone=$(zone_of); [ -n "$zone" ] || die "no VM $VM"
       exec gcloud compute ssh "$VM" --project "$PROJECT" --zone "$zone" ;;
  all)
    if [ "${KEEP:-}" != 1 ]; then
      trap 'rc=$?; [ -n "$(zone_of)" ] && { step_fetch || true; step_down; }; exit $rc' EXIT
    fi
    step_up; step_setup
    start_detached run "docker/dooly-docker.sh test && docker/example.sh $(quoted "$@")"
    [ "${KEEP:-}" = 1 ] && step_fetch && log "VM kept: $0 ssh / $0 down"
    exit "$RC" ;;
esac
