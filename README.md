# Dooly

This repository is the official implementation of [Dooly](). It holds two packages:

- [`doolyprof/`](doolyprof/README.md), the Dooly Profiler: traces a vLLM model, finds its distinct operations and
  profiles each of them into a database (`profile.db`). `docker/` runs it in a container.
- [`doolysim/`](doolysim/README.md), the Dooly Simulator: predicts vLLM serving performance (TTFT, TPOT, latency) by
  running vLLM's own scheduler on the CPU and predicting each step's latency from a
  `profile.db`.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Documentation

- [doolyprof/README.md](doolyprof/README.md): tracing and profiling a model, with Docker or the
  installed packages; [docker/README.md](docker/README.md) covers the image and its scripts.
- [doolysim/README.md](doolysim/README.md): training DoolySim's models and simulating a workload.

Both are published to `main` with their packages.

## Development

`dev-branch` holds the development history and `main` the release. `publish.sh` copies
`doolyprof/`, `doolysim/` and `docker/` (minus `.publishignore`) from a worktree of
`dev-branch` into a worktree of `main`, where the result is committed:

```publish
git worktree add ../dooly-dev dev-branch
SRC=../dooly-dev DST=. bash ../dooly-dev/publish.sh            # dry run; add --apply to copy
```

Dev-only tools (not published): measure real vLLM on a workload, with each step's GPU kernel time
(`--kernel-trace-dir`, the time DoolySim's step predictions are compared against), and plot it
against a simulation:

```vllm
python -m doolysim.benchmark.measure_vllm_streaming --model meta-llama/Llama-3.1-8B --backend FLASH_ATTN --tp 1 --load-dataset [WORKLOAD.jsonl] --output ./comparison/vllm.json [--kernel-trace-dir /abs/path]
```

```comparison
python -m doolysim.tools.plot_comparison --dooly ./comparison/dooly.json --vllm ./comparison/vllm.json --metric ttft -o ./comparison/ttft.png
```
