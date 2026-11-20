# Dooly

This repository is the official implementation of [Dooly](). It holds two packages:

- `doolyprof/`, the Dooly Profiler: traces a vLLM model, finds its distinct operations and
  profiles each of them into a database (`profile.db`). `docker/` runs it in a container.
- `doolysim/`, the Dooly Simulator: predicts vLLM serving performance (TTFT, TPOT, latency) by
  running vLLM's own scheduler on the CPU and predicting each step's latency from a
  `profile.db`.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Dooly Profiler

Scripts to run Dooly Profiler:

```tracer
python -m doolyprof.run.run-tracer --model meta-llama/Llama-3.1-8B
```

```profiler
python -m doolyprof.run.run-profiler --model meta-llama/Llama-3.1-8B --traces [PATH_TO_TRACE] --attention-backend FLASHINFER --max-batch-size 32 --max-seq-len 1024 --tp 1 --db-path [PATH_TO_DB]
```

We can also use the `--profie-only` flag to only profile given there is a trace available.

```e2e
python -m doolyprof.run.run-e2e --model meta-llama/Llama-3.1-8B --attention-backed FLASHINFER --max-batch-size 32 --max-seq-len 1024 --tp 1 --db-path [PATH_TO_DB]
```

## Dooly Simulator

DoolySim predicts each step from a DoolyProf `profile.db`. Its latency models do not extrapolate
beyond the profiled batch sizes and sequence lengths, so profile at the scale you will simulate
(for example `--max-batch-size 32 --max-seq-len 8192`). Then train the models into a prediction
cache:

```estimator
python -m doolysim.estimator --model meta-llama/Llama-3.1-8B --backend FLASH_ATTN --tp-size 1 --db-path [PATH_TO_DB] --cache-dir [PATH_TO_CACHE]
```

Simulate a workload, a JSONL file of requests with arrival times:

```simulator
python -m doolysim.cli --model meta-llama/Llama-3.1-8B --backend FLASH_ATTN --tp 1 --db-path [PATH_TO_DB] --cache-dir [PATH_TO_CACHE] --load-dataset [WORKLOAD.jsonl] --json-output ./comparison/dooly.json
```

Measure real vLLM on the same workload (on a GPU). `--kernel-trace-dir` also traces each step's
GPU kernel time, the time DoolySim's step predictions are compared against:

```vllm
python -m doolysim.benchmark.measure_vllm_streaming --model meta-llama/Llama-3.1-8B --backend FLASH_ATTN --tp 1 --load-dataset [WORKLOAD.jsonl] --output ./comparison/vllm.json [--kernel-trace-dir /abs/path]
```

Workloads can also be generated: `--num-requests 20 --arrival-rate 1.0 --input-len 512
--output-len 32 --export-requests ./dataset/synthetic.jsonl` (synthetic), or from ShareGPT
(download `ShareGPT_V3_unfiltered_cleaned_split.json` from
https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered, then
`--use-sharegpt [PATH] --num-requests 100 --arrival-rate 1.0 --export-requests
./dataset/sharegpt_100req.jsonl`), with the same measurement command.

Compare the two:

```comparison
python -m doolysim.tools.plot_comparison --dooly ./comparison/dooly.json --vllm ./comparison/vllm.json --metric ttft -o ./comparison/ttft.png
```
