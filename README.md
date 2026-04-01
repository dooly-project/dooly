# Dooly

This repository is the official implementation of [Dooly](). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Dooly Profiler

Scripts to run Dooly Profiler:

```tracer
python -m dooly.run.run-tracer --model meta-llama/Llama-3.1-8B
```

```profiler
python -m dooly.run.run-profiler --model meta-llama/Llama-3.1-8B --traces [PATH_TO_TRACE] --attention-backend FLASHINFER --max-batch-size 32 --max-seq-len 1024 --tp 1 --db-path [PATH_TO_DB]
```

We can also use the `--profie-only` flag to only profile given there is a trace available.

```e2e
python -m dooly.run.run-e2e --model meta-llama/Llama-3.1-8B --attention-backed FLASHINFER --max-batch-size 32 --max-seq-len 1024 --tp 1 --db-path [PATH_TO_DB]
```

## Dooly Simulator
TODO