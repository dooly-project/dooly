# DoolySim
simulator implementation for end to end evaluation of Dooly

# Commonly Used Scripts:

## Offline vLLM Baseline

### Using Synthetic Data:

```
python -m doolysim.benchmark.measure_vllm_streaming --num-requests 20 --arrival-rate 1.0 --verbose --output-len 32 --input-len 512 --export-requests ./dataset/test_mixed_uniform_rate1.jsonl
```

### Using ShareGPT Data

First download the dataset:

```
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

After that, we can generate the vLLM formatted dataset and run using this single script. 
Generating the vLLM formatted dataset can be done with another script in our `/tools` directory.

```
python -m doolysim.benchmark.measure_vllm_streaming --num-requests 100 --arrival-rate 1.0 --verbose --export-requests ./dataset/sharegpt_100req_1-0.jsonl --use-sharegpt ./dataset/ShareGPT_V3_unfiltered_cleaned_split.json [--output-len 1]
```

### Loading datasets:
```
python -m doolysim.benchmark.measure_vllm_streaming --num-requests 100 --arrival-rate 1.0 --verbose --load-dataset ./dataset/sharegpt_100req_original_rate1-0.jsonl
```

## DoolySim

We can use the same generation input parameters in DoolySim when testing. 

```
python -m doolysim.cli --model meta-llama/Llama-3.1-8B --seed 42 --max-num-batched-tokens 8192 --gpu-mem-util 0.9 --json-output ./comparison/dooly_raw.json --load-dataset ./dataset/sharegpt_100req_original_rate1-0.jsonl --verbose --cpu-overhead-ms 8.0
```

## Comparison Plot

```CDF comparison
python -m doolysim.tools.plot_comparison --dooly ./comparison/dooly_sharegpt_all_rate0-5.json --vllm ./comparison/vllm_sharegpt_all_rate0-5.json --metric ttft -o ./comparison/temp.png
```