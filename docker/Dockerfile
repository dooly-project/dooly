# DoolyProf and DoolySim on vLLM 0.33. Published as jhkjames1110/doolyprof:latest (docker/README.md);
# to build it:
#
#   docker build -f docker/Dockerfile --build-arg DOOLY_REVISION=$(git rev-parse HEAD) -t doolyprof:vllm0.33 .
#                                                                    (from the repository root)
#   DOOLY_IMAGE=doolyprof:vllm0.33 docker/dooly-docker.sh test       (to use it)
#
# - CUDA 13.0 toolkit, not only the runtime: FlashInfer compiles kernels the first time a shape
#   needs them and needs nvcc and the CUDA headers; Triton builds its launcher with gcc.
# - Python 3.11 and the exact package set DoolyProf was tested with on vLLM 0.33
#   (docker/requirements.txt: `pip freeze` of that environment, vllm==0.33.0, torch==2.14.0), plus
#   DoolySim's pandas and scikit-learn.
# - DoolyProf and DoolySim are pure Python and are copied in; `dooly` (doolyprof/cli.py) is the
#   entry point. DoolySim runs with `--entrypoint python IMAGE -m doolysim.cli ...`.
# - --build-arg DOOLY_REVISION=$(git rev-parse HEAD) records the commit: in the label
#   org.opencontainers.image.revision (docker/dooly-docker.sh check-update shows it) and in
#   doolyprof/REVISION, which every profiling run records (the copy here is not a git checkout).
# Model weights are not needed (both stages load dummy weights); mount the Hugging Face cache for
# the config and tokenizer files (see docker/dooly-docker.sh).
ARG CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
FROM ghcr.io/astral-sh/uv:0.9.7 AS uv

FROM ${CUDA_IMAGE}
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates libnuma1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN uv python install 3.11 && uv venv --python 3.11 /opt/venv
COPY docker/requirements.txt /tmp/requirements.txt
RUN uv pip install --python /opt/venv/bin/python --no-cache -r /tmp/requirements.txt

COPY doolyprof /opt/dooly/doolyprof
COPY doolysim /opt/dooly/doolysim
ARG DOOLY_REVISION=
RUN if [ -n "$DOOLY_REVISION" ]; then echo "$DOOLY_REVISION" > /opt/dooly/doolyprof/REVISION; fi
LABEL org.opencontainers.image.revision=$DOOLY_REVISION
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/dooly \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    HF_HUB_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1
WORKDIR /opt/dooly
ENTRYPOINT ["python", "-m", "doolyprof.cli"]
CMD ["--help"]
