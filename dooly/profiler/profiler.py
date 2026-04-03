import os
import socket
import sqlite3
import json
from pathlib import Path
import torch
import torch.distributed as dist
import traceback
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from tqdm import tqdm
from vllm.distributed import parallel_state
import multiprocessing as mp

def find_free_port() -> int:
    """Find a free port by binding to port 0 and letting the OS assign one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

from dooly.profiler.vllm_layer_profiler import VLLMLayerProfiler, make_dict 
from dooly.profiler.model import ModelInfo as ProfileModel
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from dooly.profiler.importer import OpImporter
from dooly.profiler.module import ModuleInfo
from dooly.profiler.input_generator import InputGenerator 
from dooly.profiler.resolver import ModuleResolver
from dooly.profiler.utils.generate_batch import AttentionBatchConfig


def wrap_list_args(op_packet, op_name: str, inputs: List) -> List:
    """Wrap inputs that should be lists according to PyTorch schema."""
    from torch._ops import OpOverloadPacket

    if not isinstance(op_packet, OpOverloadPacket):
        return inputs

    # Get specific overload — same best-match logic as prepare_op_args.
    overload_name = op_name.split('.')[-1] if '.' in op_name else None
    if overload_name and hasattr(op_packet, overload_name):
        op_overload = getattr(op_packet, overload_name)
    else:
        best_overload = None
        best_score = float('inf')
        for attr_name in dir(op_packet):
            try:
                candidate = getattr(op_packet, attr_name)
                if not hasattr(candidate, '_schema'):
                    continue
                positional_count = sum(
                    1 for a in candidate._schema.arguments if not a.kwarg_only
                )
                score = abs(positional_count - len(inputs))
                if score < best_score:
                    best_score = score
                    best_overload = candidate
            except Exception:
                continue
        if best_overload is None:
            return inputs
        op_overload = best_overload

    schema = op_overload._schema
    wrapped = list(inputs)

    for i, arg in enumerate(schema.arguments):
        if i >= len(wrapped):
            break
        arg_type = str(arg.type)
        # Wrap list arguments (List[...] or type[] patterns)
        if ('List[' in arg_type or '[]' in arg_type) and not isinstance(wrapped[i], list):
            wrapped[i] = [wrapped[i]]

    return wrapped


def prepare_op_args(op_packet, op_name: str, inputs: List) -> Tuple[List, Dict]:
    """
    Prepare inputs for PyTorch op call by:
    1. Wrapping inputs that should be lists according to schema
    2. Separating keyword-only arguments from positional arguments

    Returns:
        Tuple of (positional_args, keyword_args)
    """
    from torch._ops import OpOverloadPacket

    if not isinstance(op_packet, OpOverloadPacket):
        return inputs, {}

    # Get specific overload (e.g., aten::add.Tensor).
    # When the op name includes an explicit overload suffix (e.g. "aten::mean.dim"),
    # use it directly.  When there is no suffix (e.g. "aten::mean"), try every
    # available overload and pick the one whose number of positional (non-kwarg-only)
    # arguments best matches the number of inputs we have.  This prevents
    # misrouting inputs through the wrong schema (e.g. feeding [-1] into dtype).
    overload_name = op_name.split('.')[-1] if '.' in op_name else None

    if overload_name and hasattr(op_packet, overload_name):
        op_overload = getattr(op_packet, overload_name)
    else:
        # No explicit overload; score every overload by how closely its
        # positional-arg count matches len(inputs).
        best_overload = None
        best_score = float('inf')
        for attr_name in dir(op_packet):
            try:
                candidate = getattr(op_packet, attr_name)
                if not hasattr(candidate, '_schema'):
                    continue
                positional_count = sum(
                    1 for a in candidate._schema.arguments if not a.kwarg_only
                )
                score = abs(positional_count - len(inputs))
                if score < best_score:
                    best_score = score
                    best_overload = candidate
            except Exception:
                continue
        if best_overload is None:
            return inputs, {}
        op_overload = best_overload

    schema = op_overload._schema
    positional_args = []
    keyword_args = {}

    for i, arg in enumerate(schema.arguments):
        if i >= len(inputs):
            break

        value = inputs[i]
        arg_type = str(arg.type)

        # Wrap list arguments (List[...] or type[] patterns) - but not if already a list
        if ('List[' in arg_type or '[]' in arg_type) and not isinstance(value, list):
            value = [value]

        # Separate keyword-only arguments
        if arg.kwarg_only:
            keyword_args[arg.name] = value
        else:
            positional_args.append(value)

    # Fix shape-list arguments (e.g. reshape/view second arg) whose concrete
    # values captured from the trace don't match the dummy-run tensor size.
    # When the primary input is a Tensor, check each list arg that looks like
    # a shape spec (all ints, possibly with -1).  If its concrete product
    # doesn't divide the tensor's total elements evenly, replace offending
    # values with -1 so the op can infer the right size itself.
    primary_tensor = next(
        (a for a in positional_args if isinstance(a, torch.Tensor)), None
    )
    if primary_tensor is not None:
        total_elements = primary_tensor.numel()
        for idx, val in enumerate(positional_args):
            if not isinstance(val, list):
                continue
            if not all(isinstance(v, int) for v in val):
                continue
            concrete_vals = [v for v in val if v != -1]
            if not concrete_vals:
                continue
            product = 1
            for v in concrete_vals:
                product *= abs(v)
            if product == 0 or total_elements % product != 0:
                # Replace positive concrete values that are incompatible with -1
                positional_args[idx] = [-1 if v > 0 else v for v in val]

    return positional_args, keyword_args


class Profiler:
    def __init__(
        self,
        output_path: str,
        workload_mode: str = "vidur",
        attention_backend: str = "TRITON_ATTN",
        max_batch_size: int = 5,
        max_seq_len: int = 512,
        test_counts: int = 10,
        force_prefill_kernel: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        world_config: Optional[Dict[str, int]] = None,
        db_path: Optional[str] = None,
        gpu: str = "0",
    ):
        self.importer = OpImporter()
        self.module_caller = None
        self.output_path = output_path
        self.db_path = db_path
        self.world_config = world_config
        self.tp_config = world_config.get("tp", 1)
        self.pp_config = world_config.get("pp", 1)
        self.gpu = gpu

        # Initialize database if db_path provided
        if self.db_path:
            self._init_db()

        self.dtype = dtype

        self.workload_mode = workload_mode
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.attention_backend = attention_backend

        output_dir = str(Path(output_path).parent)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.attention_config = {
            "batch_size": max_batch_size,
            "mode": workload_mode,
            "test_counts": test_counts,
            "output_dir": output_dir,
            "use_decode_kernel": False,
            "req_size": max_seq_len,
            "quiet_mode": True,
            "force_prefill_kernel": force_prefill_kernel,
            "num_runs": 5,
            "decode_repeat": 10,
            "random_pages": True,
        }

        self.attention_profiler: Optional[FlashInferProfile] = None
        # Shape: [num_blocks, 2, block_size, num_kv_heads, head_dim]
        self._global_kv_cache: Optional[torch.Tensor] = None
        self._global_kv_cache_spec: Optional[Tuple[int, int, int, int]] = None

    def _init_db(self) -> None:
        """Initialize SQLite database with required tables."""
        # Create parent directory if needed
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            Path(db_dir).mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signatures (
                operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                signature_hash TEXT UNIQUE NOT NULL,
                operation_name TEXT NOT NULL,
                signature_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signature_hash TEXT NOT NULL,
                mean_latency_ms REAL,
                median_latency_ms REAL,
                max_latency_ms REAL,
                min_latency_ms REAL,
                std_latency_ms REAL,
                num_tokens INTEGER,
                num_requests INTEGER,
                is_context BOOLEAN NULL,
                is_prefill BOOLEAN NULL,
                kv_cache_size INTEGER NULL,
                prefill_chunk_size INTEGER NULL,
                num_tensor_parallel_workers INTEGER,
                input_shapes TEXT, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signature_hash) REFERENCES signatures(signature_hash)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiled_models (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_family TEXT NOT NULL,
                backend TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                operation_id INTEGER NOT NULL,
                count INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES profiled_models(model_id),
                FOREIGN KEY (operation_id) REFERENCES signatures(operation_id),
                UNIQUE(model_id, operation_id, count)
            )
        """) 
        conn.commit()
        conn.close()
        print(f"[PROFILER] Initialized database at {self.db_path}")

    # main profiling function
    def profile(self, models_to_profile: List[ProfileModel]) -> None:
        results, attention_results, attention_kernel_infos = self._profile_models(models_to_profile)
    
        if self.db_path:
            self.write_to_models_db(models_to_profile)
            self.write_to_results_db(results)
        else:
            self.write_csv(results, attention_results, attention_kernel_infos)

    def _profile_models(self, models: List[ProfileModel]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        all_results: List[Dict[str, Any]] = []
        attention_results: List[Dict[str, Any]] = []
        attention_kernel_infos: List[Dict[str, Any]] = []

        result_queue = mp.Queue()

        for model in models:
            modules = model.to_profile or []
            if not modules:
                continue
            print(f"\n[PROFILER] Profiling model: {model.name}")

            # Use minimum of user-specified max_seq_len and model's max_model_len
            # This respects user input as upper bound while also respecting model limits
            # (following Vidur's approach)
            model_max = model.get_max_model_len()
            if model_max:
                self.max_model_len = min(self.max_seq_len, model_max)
                print(f"[PROFILER] Using max_model_len={self.max_model_len} (min of user's {self.max_seq_len} and model's {model_max})")
            else:
                self.max_model_len = self.max_seq_len
                print(f"[PROFILER] Using max_model_len={self.max_model_len} from user config")

            print(f"\n[PROFILER] Profiling at TP={self.tp_config}")

            # Partition modules into raw ops vs vLLM modules
            # to_profile_resolved contains 4-tuples: (module, callable, sig_hash, sig_json)
            raw_ops = [(m, c, h, j) for m, c, h, j in model.to_profile_resolved if not m.is_module]
            vllm_modules = [(m, c, h, j) for m, c, h, j in model.to_profile_resolved if m.is_module]

            # Phase 1: Profile raw ops in main process (no vLLM needed - maximum GPU memory)
            if raw_ops:
                print(f"\n[PROFILER] Phase 1: Profiling {len(raw_ops)} raw ops (no vLLM)")
                raw_results = self._profile_raw_ops(model, raw_ops)
                all_results.extend(raw_results)

            # Phase 2: Profile vLLM modules in subprocess (needs vLLM context)
            if vllm_modules:
                print(f"\n[PROFILER] Phase 2: Profiling {len(vllm_modules)} vLLM modules")
                p = mp.Process(
                    target=self._profile_vllm_modules_subprocess,
                    args=(model, vllm_modules, result_queue)
                )
                p.start()

                # Get results BEFORE join to avoid deadlock
                vllm_results = result_queue.get()
                all_results.extend(vllm_results)

                p.join()

                if p.exitcode != 0:
                    print(f"[PROFILER] WARNING: Subprocess exited with code {p.exitcode}")

            print("="*20)

        return all_results, attention_results, attention_kernel_infos

    def _profile_raw_ops(
        self,
        model: ProfileModel,
        raw_ops: List[Tuple[ModuleInfo, Callable, str, str]]
    ) -> List[Dict[str, Any]]:
        """Profile raw ops (is_module=False) in main process without vLLM.

        Raw ops don't need VLLMLayerProfiler since they just call PyTorch/vLLM
        ops directly via OpImporter. This gives them maximum GPU memory.

        Note: Callables are None from resolution phase (run-profiler.py line 89)
        because they can't cross process boundaries. We re-import them here.
        """
        all_results = []

        pbar = tqdm(raw_ops, desc=f"Profiling raw ops (TP={self.tp_config})", unit="op")

        for i, (module, _, sig_hash, sig_json) in enumerate(pbar):
            pbar.set_description(f"Raw ops [{i+1}/{len(raw_ops)}]")
            tqdm.write(f"[{i+1}/{len(raw_ops)}] Profiling raw op: {module.operation_name} for {module.module_name}")

            # Import the callable (resolution phase sets it to None)
            callable_obj, msg = self.importer.import_op(op_full_name=module.operation_name)
            if callable_obj is None:
                tqdm.write(f"  Could not import: {msg}")
                continue

            torch.cuda.empty_cache()

            module_results = self._profile_single_raw_op(
                module, model, callable_obj,
                max_model_len=self.max_model_len,
                sig_hash=sig_hash,
                sig_json=sig_json,
            )

            if module_results:
                all_results.extend(module_results)
                tqdm.write(f"  Profiled {len(module_results)} configs")

        print(f"[PROFILER] Finished profiling {len(all_results)} raw op configs")
        return all_results

    def _profile_vllm_modules_subprocess(
        self,
        model: ProfileModel,
        vllm_modules: List[Tuple[ModuleInfo, Callable, str, str]],
        results: mp.Queue
    ):
        """Profile vLLM modules (is_module=True) with full vLLM context.

        This runs in a subprocess to ensure GPU memory is fully freed when done.
        """
        # Initialize VLLMLayerProfiler
        self.module_caller = VLLMLayerProfiler(
            model_name=model.name,
            dtype=str(self.dtype).split(".")[-1],
            enforce_eager=True,
            gpu_memory_utilization=0.9,
            max_model_len=self.max_model_len,
            attn_backend=self.attention_backend,
            gpu=self.gpu,
        )

        # Re-resolve callables with loaded model
        resolver = ModuleResolver(
            vlp=self.module_caller,
            importer=self.importer,
            dtype=self.dtype,
        )

        resolved_modules = []
        for module, _, sig_hash, sig_json in vllm_modules:
            callable_obj = resolver.try_import(module)
            resolved_modules.append((module, callable_obj, sig_hash, sig_json))

        all_results = []

        pbar = tqdm(resolved_modules, desc=f"Profiling modules (TP={self.tp_config})", unit="module")

        for i, (module, callable_obj, sig_hash, sig_json) in enumerate(pbar):
            pbar.set_description(f"Modules [{i+1}/{len(resolved_modules)}]")
            tqdm.write(f"[{i+1}/{len(resolved_modules)}] Profiling {module.module_name} - ({module.operation_name})")

            torch.cuda.empty_cache()

            module_results = self._profile_single_module_for_model(
                module, model, callable_obj,
                max_model_len=self.max_model_len,
                sig_hash=sig_hash,
                sig_json=sig_json,
            )

            if module_results:
                all_results.extend(module_results)
                tqdm.write(f"  Profiled {len(module_results)} configs")

        self.module_caller.close()

        # Put all results at once to avoid queue deadlock
        results.put(all_results)
        print(f"[PROFILER] Finished profiling {len(all_results)} module configs")

    def _profile_single_raw_op(
        self,
        module: ModuleInfo,
        model: ProfileModel,
        callable_obj: Callable,
        max_model_len: Optional[int] = None,
        sig_hash: Optional[str] = None,
        sig_json: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Profile a single raw op (is_module=False) without vLLM context."""
        try:
            limit = max_model_len if max_model_len is not None else self.max_seq_len

            generator = InputGenerator(
                module=module,
                max_num_request=self.max_batch_size,
                max_num_token=limit,
                test_counts=self.attention_config["test_counts"],
                dtype=self.dtype,
            )

            workload_configs = generator.prepare_inputs(
                mode=self.workload_mode,
                use_batch_config=False  # Raw ops don't use batch config
            )

            # Sort by size descending so large configs fail early if OOM
            def get_config_size(config):
                return config.get('num_tokens', 0) * config.get('num_requests', 1)
            workload_configs = sorted(workload_configs, key=get_config_size, reverse=True)

            results = []
            pbar_workload_configs = tqdm(
                workload_configs,
                desc=f"  Configs for {module.operation_name[:30]}",
                unit="config",
                leave=False,
            )

            for workload_config in pbar_workload_configs:
                # Update progress bar with current config info
                pbar_workload_configs.set_description(
                    f"  Configs for {module.operation_name[:30]} "
                    f"(tokens: {workload_config.get('num_tokens', '?')}, reqs: {workload_config.get('num_requests', '?')})"
                )

                inputs, workload_params = generator._create_tensor_inputs_from_params(workload_config)

                if not inputs:
                    continue

                args, kwargs = prepare_op_args(callable_obj, module.operation_name, inputs)

                def run_callable():
                    # Reconstruct kwargs from arg_name if present (for modules called with keyword args)
                    has_named = any(
                        getattr(pi, 'arg_name', None) is not None
                        for pi in module.inputs
                    )
                    if has_named and module.is_module:
                        # Re-split: positional inputs have no arg_name
                        positional_tensors = [
                            tensor for pi, tensor in zip(module.inputs, inputs)
                            if getattr(pi, 'arg_name', None) is None
                        ]
                        kwarg_tensors = {
                            pi.arg_name: tensor
                            for pi, tensor in zip(module.inputs, inputs)
                            if getattr(pi, 'arg_name', None) is not None
                        }
                        callable_obj(*positional_tensors, **kwarg_tensors)
                    else:
                        callable_obj(*args, **kwargs)

                try:
                    # Warmup
                    for _ in range(5):
                        run_callable()

                    torch.cuda.synchronize()

                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    run_results = []
                    for _ in range(20):
                        start_event.record()
                        run_callable()
                        end_event.record()
                        torch.cuda.synchronize()
                        latency_ms = start_event.elapsed_time(end_event)
                        run_results.append(latency_ms)

                    # Use original inputs for shape tracking (not args which may have list wrapping)
                    all_inputs = inputs
                    result = {
                        "signature_hash": sig_hash,
                        "signature_json": sig_json,  # for signatures table only
                        "module_name": module.module_name,
                        "operation_name": module.operation_name,
                        "mean_latency_ms": sum(run_results) / len(run_results),
                        "median_latency_ms": sorted(run_results)[len(run_results)//2],
                        "max_latency_ms": max(run_results),
                        "min_latency_ms": min(run_results),
                        "std_latency_ms": (sum((x - sum(run_results)/len(run_results))**2 for x in run_results) / len(run_results))**0.5,
                        "input_shapes": [list(i.shape) if hasattr(i, "shape") else str(i) for i in all_inputs],
                        "workload_params": workload_params,
                        "num_tokens": workload_params.get('num_tokens'),
                        "num_requests": workload_params.get('num_requests'),
                        "num_tensor_parallel_workers": self.tp_config,
                    }
                    results.append(result)

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" in str(e).lower() or "CUDA" in str(e):
                        print(f"[PROFILER] OOM for {module.operation_name}, skipping...")
                    else:
                        print(f"[PROFILER] Error: {e}")
                    torch.cuda.empty_cache()
                    continue

                del inputs, args, kwargs
                torch.cuda.empty_cache()

            pbar_workload_configs.close()
            return results if results else None

        except Exception as e:
            print(f"[PROFILER] Error profiling {module.operation_name}: {e}")
            return None

    def _profile_single_module_for_model(
        self,
        module: ModuleInfo,
        model: ProfileModel,
        callable_obj: Callable,
        # model_params: Dict[str, Any],
        max_model_len: Optional[int] = None,
        sig_hash: Optional[str] = None,
        sig_json: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        try:
            # Adjust model params for tensor parallelism
            num_kv_heads = model.num_kv_heads
            head_dim = model.head_dim

            # Use max_model_len as the limit for token counts
            # (Vidur also limits to model's max_position_embeddings, not batch * seq_len)
            limit = max_model_len if max_model_len is not None else self.max_seq_len

            layer_name = None
            if module.is_module and self.module_caller is not None:
                name_to_module = make_dict(self.module_caller.model)
                layer_name = self.module_caller.get_layer_name_by_module(module.operation_name, name_to_module)

            generator = InputGenerator(
                module=module,
                max_num_request=self.max_batch_size,
                max_num_token=limit,
                test_counts=self.attention_config["test_counts"],
                dtype=self.dtype,
            )

            workload_configs = generator.prepare_inputs(
                mode=self.workload_mode,
                use_batch_config=(layer_name is not None)
            )

            # Sort workload configs by size descending so large configs fail early if OOM
            def get_config_size(config):
                if isinstance(config, AttentionBatchConfig):
                    return config.batch_size * (config.prefill_chunk_size + config.kv_cache_size)
                else:
                    return config.get('num_tokens', 0) * config.get('num_requests', 1)

            workload_configs = sorted(workload_configs, key=get_config_size, reverse=True)

            results = []
            config_idx = 0
            pbar_workload_configs = tqdm(
                workload_configs,
                desc=f"  Configs for {module.module_name[:25]}",
                unit="config",
                leave=False,
            )

            for workload_config in pbar_workload_configs:
                # workload_config is AttentionBatchConfig for attention, Dict[str, int] for non-attention
                if isinstance(workload_config, AttentionBatchConfig):
                    pbar_workload_configs.set_description(
                        f"  Configs for {module.module_name[:25]} "
                        f"(Batch: {workload_config.batch_size}, Prefill: {workload_config.prefill_chunk_size}, KV: {workload_config.kv_cache_size})"
                    )
                else:
                    pbar_workload_configs.set_description(
                        f"  Configs for {module.module_name[:25]} "
                        f"(tokens: {workload_config.get('num_tokens', '?')}, reqs: {workload_config.get('num_requests', '?')})"
                    )

                # _create_tensor_inputs_from_params now accepts AttentionBatchConfig or Dict[str, int]
                # and returns (inputs, workload_params)
                inputs, workload_params = generator._create_tensor_inputs_from_params(workload_config)

                if not inputs:
                    # Invalid config, skip
                    continue

                # Prepare inputs: wrap list args and separate keyword-only args
                args, kwargs = prepare_op_args(callable_obj, module.operation_name, inputs)

                forward_context = None
                if layer_name and isinstance(workload_config, AttentionBatchConfig):
                    batch_config = AttentionBatchConfig(
                        batch_size=workload_config.batch_size,
                        prefill_chunk_size=workload_config.prefill_chunk_size,
                        kv_cache_size=workload_config.kv_cache_size,
                        is_prefill=workload_config.is_prefill,
                    )
                    try:
                        result = self.module_caller.build_layer_metadata(layer_name, batch_config)
                        if result is None:
                            # Skip this config - exceeds available KV cache blocks
                            continue
                        layer_metadata, _ = result
                        forward_context = {layer_name: layer_metadata}
                    except Exception as e:
                        print(f"[PROFILER] Warning: Could not build metadata: {e}")
                        forward_context = None

                def run_callable():
                    # Reconstruct kwargs from arg_name if present (for modules called with keyword args)
                    has_named = any(
                        getattr(pi, 'arg_name', None) is not None
                        for pi in module.inputs
                    )
                    if has_named and module.is_module:
                        # Re-split: positional inputs have no arg_name
                        positional_tensors = [
                            tensor for pi, tensor in zip(module.inputs, inputs)
                            if getattr(pi, 'arg_name', None) is None
                        ]
                        kwarg_tensors = {
                            pi.arg_name: tensor
                            for pi, tensor in zip(module.inputs, inputs)
                            if getattr(pi, 'arg_name', None) is not None
                        }
                        callable_obj(*positional_tensors, **kwarg_tensors)
                    else:
                        callable_obj(*args, **kwargs)

                def run_callable_with_context():
                    # Reconstruct kwargs from arg_name if present (for modules called with keyword args)
                    has_named = any(
                        getattr(pi, 'arg_name', None) is not None
                        for pi in module.inputs
                    )
                    if has_named and module.is_module:
                        # Re-split: positional inputs have no arg_name
                        positional_tensors = [
                            tensor for pi, tensor in zip(module.inputs, inputs)
                            if getattr(pi, 'arg_name', None) is None
                        ]
                        kwarg_tensors = {
                            pi.arg_name: tensor
                            for pi, tensor in zip(module.inputs, inputs)
                            if getattr(pi, 'arg_name', None) is not None
                        }
                        with set_current_vllm_config(self.module_caller.vllm_config):
                            with set_forward_context(forward_context, self.module_caller.vllm_config, virtual_engine=0):
                                callable_obj(*positional_tensors, **kwarg_tensors)
                    else:
                        with set_current_vllm_config(self.module_caller.vllm_config):
                            with set_forward_context(forward_context, self.module_caller.vllm_config, virtual_engine=0):
                                callable_obj(*args, **kwargs)

                run_fn = run_callable_with_context if forward_context else run_callable

                try:
                    # warm up
                    for _ in range(5):
                        run_fn()

                    torch.cuda.synchronize()

                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    # 20 runs
                    run_results = []
                    for _ in range(20):
                        start_event.record()
                        run_fn()
                        end_event.record()
                        torch.cuda.synchronize()
                        latency_ms = start_event.elapsed_time(end_event)
                        run_results.append(latency_ms)

                    # Use original inputs for shape tracking (before prepare_op_args wraps list args)
                    all_inputs = inputs
                    result = {
                        "signature_hash": sig_hash,
                        "signature_json": sig_json,  # for signatures table only
                        "module_name": module.module_name,
                        "operation_name": module.operation_name,
                        "num_tensor_parallel_workers": self.tp_config,
                        "mean_latency_ms": sum(run_results) / len(run_results),
                        "median_latency_ms": sorted(run_results)[len(run_results)//2],
                        "max_latency_ms": max(run_results),
                        "min_latency_ms": min(run_results),
                        "std_latency_ms": (sum((x - sum(run_results)/len(run_results))**2 for x in run_results) / len(run_results))**0.5,
                        "input_shapes": [
                            list(i.shape) if hasattr(i, "shape") else str(i) for i in all_inputs
                        ],
                        "workload_params": workload_params,
                        "num_tokens": workload_params.get('num_tokens'),
                        "num_requests": workload_params.get('num_requests'),
                    }
                    results.append(result)

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" in str(e).lower() or "CUDA" in str(e):
                        print(f"[PROFILER] OOM for {module.module_name} with config {workload_params}, skipping...")
                    else:
                        print(f"[PROFILER] Error for {module.module_name}: {e}")
                    # Clean up and continue
                    torch.cuda.empty_cache()
                    continue

                # Clean up after successful profiling
                del inputs, args, kwargs
                torch.cuda.empty_cache()
                config_idx += 1

            pbar_workload_configs.close()

            if results:
                print(f"[PROFILER] Successfully profiled {module.module_name}.")
                return results

            return None

        except Exception as e:
            print(f"[PROFILER] Error profiling {module.module_name}: {e}")
            traceback.print_exc()
            return None
        
    def write_csv(self, results: List[Dict[str, Any]], attention_results: List[Dict[str, Any]], attention_kernel_infos: List[Dict[str, Any]]) -> None:
        with open(self.output_path, "w") as f:
            f.write(
                "module_name,operation_name,num_tensor_parallel_workers,"
                "num_tokens,num_requests,"
                "mean_latency_ms,median_latency_ms,max_latency_ms,min_latency_ms,std_latency_ms,"
                "input_shapes,workload_params\n"
            )
            for result in results:
                workload_params_str = str(result.get("workload_params", {}))
                num_tokens = result.get("num_tokens", "")
                num_requests = result.get("num_requests", "")
                f.write(
                    f"{result['module_name']},{result['operation_name']},{self.tp_config},"
                    f"{num_tokens},{num_requests},"
                    f"{result['mean_latency_ms']:.4f},{result['median_latency_ms']:.4f},"
                    f"{result['max_latency_ms']:.4f},{result['min_latency_ms']:.4f},{result['std_latency_ms']:.4f},"
                    f"\"{result['input_shapes']}\",\"{workload_params_str}\"\n"
                )
        if attention_results:
            self.attention_profiler._export_estimator_features(
                attention_results,
                attention_kernel_infos,
                csv_filename=os.path.join(self.attention_config["output_dir"], "attention_profile.csv"),
            )
        print(f"[PROFILER] Profiling complete. Results saved to {self.output_path}")

    def write_to_models_db(self, models: List[ProfileModel]) -> None:
        """Write model names to SQLite database."""
        if not self.db_path:
            print("[PROFILER] No database path specified, skipping models database write")
            return

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("BEGIN")

            for model in models:
                # Insert model and get the auto-generated model_id
                cursor = conn.execute(
                    """INSERT INTO profiled_models (model_name, model_family, backend) VALUES (?, ?, ?)""",
                    (model.name, model.model_family, self.attention_backend)
                )
                model_id = cursor.lastrowid

                # Use all_resolved_ops which contains ALL operations this model uses
                # (not just the ones that need profiling)
                for module, callable_obj, sig_hash, sig_json in model.all_resolved_ops:
                    if sig_hash:
                        # First check if signature already exists
                        cursor = conn.execute(
                            "SELECT operation_id FROM signatures WHERE signature_hash = ?",
                            (sig_hash,)
                        )
                        existing = cursor.fetchone()

                        if existing:
                            operation_id = existing[0]
                        else:
                            # Only insert if it doesn't exist (avoids incrementing counter)
                            cursor = conn.execute(
                                """INSERT INTO signatures (signature_hash, operation_name, signature_json)
                                   VALUES (?, ?, ?)""",
                                (sig_hash, module.operation_name, sig_json)
                            )
                            operation_id = cursor.lastrowid

                        # Create relationship in model_operations table
                        count = getattr(module, 'count', 1)
                        conn.execute(
                            """INSERT OR IGNORE INTO model_operations (model_id, operation_id, count)
                               VALUES (?, ?, ?)""",
                            (model_id, operation_id, count)
                        )

            conn.execute("COMMIT")
            print(f"[PROFILER] Wrote {len(models)} models to database at {self.db_path}")

        except Exception as e:
            conn.execute("ROLLBACK")
            print(f"[PROFILER] Error writing models to database: {e}")
            raise
        finally:
            conn.close()

    def write_to_results_db(self, results: List[Dict[str, Any]]) -> None:
        """Write profiling results to SQLite database."""
        if not self.db_path:
            print("[PROFILER] No database path specified, skipping database write")
            return

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("BEGIN")

            for result in results:
                sig_hash = result.get('signature_hash')
                if not sig_hash:
                    continue

                # Check if signature exists before inserting (to avoid incrementing autoincrement)
                cursor = conn.execute(
                    "SELECT operation_id FROM signatures WHERE signature_hash = ?",
                    (sig_hash,)
                )
                if not cursor.fetchone():
                    # Only insert if it doesn't exist
                    conn.execute(
                        """INSERT INTO signatures (signature_hash, operation_name, signature_json)
                           VALUES (?, ?, ?)""",
                        (sig_hash, result['operation_name'], result.get('signature_json'))
                    )

                workload_params = result.get('workload_params', {})
                num_tokens = workload_params.get('num_tokens')
                num_requests = workload_params.get('num_requests')
                is_prefill = workload_params.get('is_prefill', None)
                prefill_chunk_size = workload_params.get('prefill_chunk_size', None)
                kv_cache_size = workload_params.get('kv_cache_size', None)
                is_context = is_prefill is not None and kv_cache_size is not None and prefill_chunk_size is not None

                # Insert result with individual columns for workload params
                conn.execute(
                    """INSERT INTO results (
                        signature_hash,
                        mean_latency_ms, median_latency_ms, max_latency_ms, min_latency_ms, std_latency_ms,
                        num_tokens, num_requests,
                        is_context, is_prefill, kv_cache_size, prefill_chunk_size,
                        num_tensor_parallel_workers,
                        input_shapes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        sig_hash,
                        result.get('mean_latency_ms'),
                        result.get('median_latency_ms'),
                        result.get('max_latency_ms'),
                        result.get('min_latency_ms'),
                        result.get('std_latency_ms'),
                        num_tokens,
                        num_requests,
                        is_context,
                        is_prefill,
                        kv_cache_size,
                        prefill_chunk_size,
                        result.get('num_tensor_parallel_workers', self.tp_config),
                        json.dumps(result.get('input_shapes', [])),
                    )
                )

            conn.execute("COMMIT")
            print(f"[PROFILER] Wrote {len(results)} results to database at {self.db_path}")

        except Exception as e:
            conn.execute("ROLLBACK")
            print(f"[PROFILER] Error writing to database: {e}")
            raise
        finally:
            conn.close()

    def _init_vllm_distributed_environment(self) -> None:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["VLLM_USE_V1"] = "0"

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")

        parallel_state.init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="env://",
        )

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    def _get_module_callable(self, module: ModuleInfo) -> Optional[Any]:
        callable_obj = None
        
        if module.is_module:
            name_to_module = make_dict(self.module_caller.model)
            anchor_name = module.anchor.name
            lookup_name = anchor_name
            instance_idx = 0
            if '_' in anchor_name and anchor_name.rsplit('_', 1)[-1].isdigit():
                lookup_name = anchor_name.rsplit('_', 1)[0]
                instance_idx = int(anchor_name.rsplit('_', 1)[1])
            paths = name_to_module.get(lookup_name)
            callable_path = paths[min(instance_idx, len(paths) - 1)] if paths else None

            if callable_path:
                print(f"[PROFILER] Module path: {callable_path} (from anchor '{anchor_name}' -> lookup '{lookup_name}[{instance_idx}]')")
                callable_obj = self.module_caller.model.get_submodule(callable_path)
            else:
                print(f"[PROFILER] Module not found. Anchor: '{anchor_name}', Lookup: '{lookup_name}', Keys: {list(name_to_module.keys())}")
                return None
        
        else:
            callable_obj, _ = self.importer.import_op(op_full_name=module.operation_name)

            if callable_obj is None:
                print(f"[PROFILER] Could not import module: {module.module_name} for operation {module.operation_name}")
                return None

        return callable_obj
