import argparse
import os
import sys
from pathlib import Path
from time import time

# Add project root to path for dooly imports
# run-profiler.py -> run/ -> dooly/ -> dooly (repo root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Parse --tp early and set VLLM_FAKE_TP env var BEFORE importing vLLM modules.
# vllm_layer_profiler.py applies fake TP patches at module level based on this env var.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--tp", type=int, default=1)
_pre_args, _ = _pre_parser.parse_known_args()
os.environ["VLLM_FAKE_TP"] = str(_pre_args.tp)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import multiprocessing as mp

from dooly.profiler.importer import OpImporter
from dooly.profiler.resolver import ModuleResolver
from dooly.profiler.vllm_layer_profiler import VLLMLayerProfiler

from dooly.profiler.model import ModelInfo
from dooly.profiler.profiler import Profiler 

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["meta-llama/Llama-3.1-8B", "BioMistral/BioMistral-7B", 
                 "Qwen/Qwen2.5-7B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"],
    )
    parser.add_argument(
        "--traces",
        nargs="+",
        default=[
            "./utns_sim/data/vllm_profile/profiler_traces/llama-trace.json",
            "./utns_sim/data/vllm_profile/profiler_traces/mistral-trace.json",
            "./utns_sim/data/vllm_profile/profiler_traces/qwen-trace.json",
            "./utns_sim/data/vllm_profile/profiler_traces/deepseek-trace.json",
        ],
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_MAP.keys(),
        default="bfloat16",
    )
    parser.add_argument(
        "--profile-output",
        default="./vllm_profile/per-model-profile.csv",
    )
    parser.add_argument("--attention-backend", default="FLASHINFER")
    parser.add_argument("--workload-mode", default="vidur")
    parser.add_argument("--max-batch-size", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--test-counts", type=int, default=1)
    parser.add_argument("--force-prefill-kernel", action="store_true", default=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to SQLite database for persistent storage and deduplication")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device ID to use (default: 0)")
    return parser.parse_args()


def resolve_model(model, dtype, attention_backend, max_model_len, result_queue, gpu):
    importer=OpImporter()
    vlp = VLLMLayerProfiler(
        model_name=model.name,
        dtype=str(model.dtype).split(".")[-1],
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
        attn_backend=attention_backend,
        gpu=gpu,
    )
    resolver = ModuleResolver(
        vlp=vlp,
        importer=importer,
        dtype=dtype
    ) 

    resolved_ops = resolver.resolve(model.used_ops)

    result_queue.put([(module, None) for module, _ in resolved_ops])    

    print("\nClosing vLLM instance...")
    vlp.close()
    print(f"=== Resolution complete for {model.name} ({dtype}) ({attention_backend}) ===\n")


def main() -> None:
    args = parse_args()
    if len(args.models) != len(args.traces):
        raise SystemExit("--models and --traces must have the same length")

    dtype = DTYPE_MAP[args.dtype]
    models = [
        ModelInfo(name, trace_path, dtype)
        for name, trace_path in zip(args.models, args.traces)
    ]

    resolved_modules = []

    mp.set_start_method("spawn", force=True)

    result_queue = mp.Queue()
    importer = OpImporter()
    for i, model in enumerate(models):
        print(f"[RUNNER] Starting RESOLVER for {model.name}...")
        p = mp.Process(target=resolve_model, args=(model, dtype, args.attention_backend, args.max_seq_len, result_queue, args.gpu))
        p.start()

        # we initialize the to_profile_resolved with just the module names for now
        # and we will add the callables later during profiling
        model.to_profile_resolved = result_queue.get()

        p.join()

        if p.exitcode != 0:
            print(f"Run {i + 1} failed with exit code {p.exitcode}")
    # Compare modules across models
    models[0].compare_with_many(models[1:], db_path=args.db_path)
    
    print("Profiling Plan: ")
    from dooly.profiler.module import ModuleComparator
    comparator = ModuleComparator()
    for model in models:
        print(f"Model: {model.name}\n Modules to Profile: {len(model.to_profile)}")
        for module in model.to_profile:
            print(f"  - {module.module_name}, {module.operation_name}")
            # print the parameter and values

            # signature = comparator._module_signature(module)
            # print(f"      Signature: {signature}")

            # for _, child in module.hierarchy.anchor.children.items():
            #     print(f"      Child: {child.name}")
            #     print(f"      Child Params: {child.params}")

        print()

    profile_output = Path(args.profile_output)
    profile_output.parent.mkdir(parents=True, exist_ok=True)

    profiler = Profiler(
        output_path=str(profile_output),
        workload_mode=args.workload_mode,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        test_counts=args.test_counts,
        force_prefill_kernel=args.force_prefill_kernel,
        attention_backend=args.attention_backend,
        world_config={"tp": args.tp, "pp": 1},
        dtype=dtype,
        db_path=args.db_path,
        gpu=args.gpu,
    )
    
    start_time = time()
    profiler.profile(models)
    elapsed_s = time() - start_time
    print(f"Profiling completed in {elapsed_s:.2f}s")

if __name__ == "__main__":
    main()
