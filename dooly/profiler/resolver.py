"""
Module Resolver: validates ModuleInfo objects before profiling.

Sits between trace.py and profiler.py:
  trace.py (parse → ModuleInfos) → resolver.py (test & merge) → profiler.py (just profile)

For each ModuleInfo, tests import and dummy execution. On failure, walks up
the trace hierarchy to find a parent that works. When a parent absorbs children,
all sibling ModuleInfos under that parent are merged to avoid redundant profiling.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from dooly.profiler.module import ModuleInfo
from dooly.profiler.finput import extract_inputs_from_taint
from dooly.profiler.input_generator import InputGenerator, AttentionBatchConfig
from dooly.profiler.parser import Event, OperationHierarchy
from dooly.profiler.importer import OpImporter
from dooly.profiler.vllm_layer_profiler import make_dict, VLLMLayerProfiler


class ModuleResolver:
    """Validates and merges ModuleInfo objects before profiling.
    
    For each module, tries to:
    1. Import the callable (op or submodule)
    2. Run a single dummy forward pass
    
    On failure, walks up the hierarchy to find a working parent,
    absorbing sibling ops that share that parent.
    """

    def __init__(
        self,
        vlp: VLLMLayerProfiler,               # VLLMLayerProfiler instance (has .model)
        importer: OpImporter,
        dtype: torch.dtype = torch.bfloat16,
        max_batch_size: int = 4,
        max_seq_len: int = 128,
    ):
        self.vlp = vlp
        self.importer = importer
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        # Build module name → path dict once
        self._name_to_module = make_dict(vlp.model) if vlp else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self, modules: List[ModuleInfo]
    ) -> List[Tuple[ModuleInfo, Callable]]:
        """Validate each ModuleInfo and return (module, callable) pairs.

        When a module can't be imported/executed, walks up its hierarchy
        to find a working parent. Siblings under that parent are absorbed.

        Returns:
            List of (ModuleInfo, callable) tuples ready for profiling.
            The ModuleInfo may be modified (is_module set, absorbed_ops populated).
        """
        resolved: List[Tuple[ModuleInfo, Callable]] = []
        resolved_indices: set = set()  # indices of modules already resolved successfully
        absorbed_indices: set = set()  # indices of modules already absorbed by a parent

        for idx, module in enumerate(modules):
            if idx in absorbed_indices:
                continue

            print(f"\n[RESOLVER] Trying: {module.operation_name} (module={module.is_module})")

            if module.module_name == "None":
                print(f"[RESOLVER]   Skipping module with name 'None'")
                continue
            
            # Try importing the op/module as-is
            callable_obj = self.try_import(module)

            if callable_obj is not None:
                # Dummy run to verify it works
                success = self._try_dummy_run(callable_obj, module)
                if success:
                    print(f"[RESOLVER] ✓ {module.operation_name} — import + dummy OK")
                    resolved.append((module, callable_obj))
                    resolved_indices.add(idx)
                    continue
                else:
                    print(f"[RESOLVER] ✗ {module.operation_name} — dummy run failed, walking up")
            else:
                print(f"[RESOLVER] ✗ {module.operation_name} — import failed, walking up")

            # Walk up hierarchy to find a working parent
            if module.hierarchy is None:
                print(f"[RESOLVER] ✗ {module.operation_name} — no hierarchy, skipping")
                continue

            found_parent = False
            curr_event = module.hierarchy.anchor.parent

            while curr_event:
                parent_module = self._build_module_info_from_event(curr_event, module.hierarchy, module)
                if parent_module is None:
                    curr_event = curr_event.parent
                    continue

                # Try import + dummy for the parent
                parent_callable = self.try_import(parent_module)
                if parent_callable is None:
                    curr_event = curr_event.parent
                    continue
                success = self._try_dummy_run(parent_callable, parent_module)
                if not success:
                    curr_event = curr_event.parent
                    continue

                # absorb siblings that share this parent
                absorbed_names, absorbed_from_resolved = self._absorb_siblings(curr_event, idx, modules, absorbed_indices, resolved_indices)
                parent_module.absorbed_ops = absorbed_names

                # Remove already-resolved modules that got absorbed by this parent
                if absorbed_from_resolved:
                    resolved[:] = [(m, c) for m, c in resolved if m.operation_name not in absorbed_from_resolved]
                    print(f"[RESOLVER]   Removed from resolved: {absorbed_from_resolved}")

                print(f"[RESOLVER] ✓ Resolved to parent: {parent_module.operation_name}")
                if absorbed_names:
                    print(f"[RESOLVER]   Absorbed: {absorbed_names}")

                resolved.append((parent_module, parent_callable))
                found_parent = True
                break

            if not found_parent:
                print(f"[RESOLVER] ✗ {module.operation_name} — no working parent found, skipping")

        print(f"\n[RESOLVER] Resolved {len(resolved)} modules for profiling")
        return resolved

    # ------------------------------------------------------------------
    # Import helpers
    # ------------------------------------------------------------------

    def try_import(self, module: ModuleInfo) -> Optional[Callable]:
        """Try to import the callable for a ModuleInfo.
        
        For modules (is_module=True): look up in model via make_dict + get_submodule.
        For ops: use OpImporter.import_op.
        """
        if module.is_module:
            return self._import_module(module)
        else:
            return self._import_op(module)

    def _import_op(self, module: ModuleInfo) -> Optional[Callable]:
        """Try importing a cpu_op via OpImporter."""
        try:
            callable_obj, msg = self.importer.import_op(op_full_name=module.operation_name)
            if callable_obj is None:
                print(f"[RESOLVER]   Import failed for {module.operation_name}: {msg}")
            return callable_obj
        except Exception as e:
            print(f"[RESOLVER]   Import error for {module.operation_name}: {e}")
            return None

    def _import_module(self, module: ModuleInfo) -> Optional[Callable]:
        """Try getting a module callable via make_dict + get_submodule.

        When multiple paths exist for the same class name (e.g. GemmaRMSNorm
        appears as input_layernorm, post_attention_layernorm, q_norm, k_norm),
        walk the hierarchy anchor's parent chain to find an ancestor whose
        class name maps to a known set of paths, then filter the candidates to
        those that start with one of those ancestor paths.
        """
        if not self._name_to_module:
            print("[RESOLVER]   No model loaded, can't import module")
            return None

        anchor_name = module.operation_name
        lookup_name = anchor_name
        if '_' in anchor_name and anchor_name.rsplit('_', 1)[-1].isdigit():
            lookup_name = anchor_name.rsplit('_', 1)[0]

        paths = self._name_to_module.get(lookup_name)
        if not paths:
            print(f"[RESOLVER]   Module '{lookup_name}' not found in model dict")
            return None

        callable_path = paths[0]  # default: first path

        # If there are multiple candidate paths, use the ancestor event chain
        # to narrow down to paths that live under the correct parent submodule.
        if len(paths) > 1 and module.hierarchy is not None:
            ancestor = module.hierarchy.anchor.parent  # parent of the GemmaRMSNorm event
            while ancestor is not None:
                if ancestor.cat == 'module':
                    ancestor_class = ancestor.name
                    if '_' in ancestor_class and ancestor_class.rsplit('_', 1)[-1].isdigit():
                        ancestor_class = ancestor_class.rsplit('_', 1)[0]
                    ancestor_paths = self._name_to_module.get(ancestor_class, [])
                    # Filter candidates to those under any of the ancestor's paths
                    filtered = [p for p in paths
                                if any(p.startswith(ap + '.') or p == ap
                                       for ap in ancestor_paths)]
                    if filtered:
                        callable_path = filtered[0]
                        break
                ancestor = ancestor.parent

        try:
            submodule = self.vlp.model.get_submodule(callable_path)
            print(f"[RESOLVER]   Found module at: {callable_path}")
            print(f"[RESOLVER]   Module type: {type(submodule).__name__}")
            return submodule
        except Exception as e:
            print(f"[RESOLVER]   get_submodule failed for '{callable_path}': {e}")
            return None

    # ------------------------------------------------------------------
    # Dummy run
    # ------------------------------------------------------------------

    def _try_dummy_run(self, callable_obj: Callable, module: ModuleInfo) -> bool:
        """Run one forward pass to verify the callable works with the given inputs.

        Uses a small workload to minimize overhead. Catches CUDA errors
        that would otherwise corrupt the GPU context.

        For modules that require forward context (Attention, Mamba), uses
        VLLMLayerProfiler's infrastructure to set up proper metadata.
        """
        try:
            # Use get_layer_name_by_module to determine if forward context is needed
            layer_name = None
            if module.is_module and self.vlp is not None:
                layer_name = self.vlp.get_layer_name_by_module(module.operation_name, self._name_to_module)
                print(f"[RESOLVER DEBUG] layer_name for {module.operation_name}: {layer_name}")

            generator = InputGenerator(
                module=module,
                max_num_request=self.max_batch_size,
                max_num_token=self.max_seq_len,
                test_counts=1,
                dtype=self.dtype,
            )

            workload_configs = generator.prepare_inputs(
                mode="vidur",
                use_batch_config=(layer_name is not None)
            )
            assert(workload_configs is not None)

            workload_batch = workload_configs[0]
            print(f"[RESOLVER DEBUG] workload_batch type: {type(workload_batch).__name__}")
            if isinstance(workload_batch, AttentionBatchConfig):
                print(f"[RESOLVER DEBUG] AttentionBatchConfig: batch_size={workload_batch.batch_size}, "
                      f"is_prefill={workload_batch.is_prefill}, kv_cache_size={workload_batch.kv_cache_size}")
                if workload_batch.is_prefill:
                    # Prefill: num_tokens = prefill_chunk_size
                    tensor_workload = [workload_batch.prefill_chunk_size + workload_batch.kv_cache_size]
                else:
                    # Decode: num_tokens = batch_size (1 token per request) 
                    tensor_workload = [workload_batch.kv_cache_size] * workload_batch.batch_size
            else:
                tensor_workload = workload_batch

            inputs, workload_params = generator._create_tensor_inputs_from_params(tensor_workload)
            if not inputs:
                print("[RESOLVER]   No inputs created")
                return False

            # Debug: Show what inputs were generated and how they'll be called
            print(f"[RESOLVER DEBUG] Module: {module.operation_name}")
            print(f"[RESOLVER DEBUG] Generated {len(inputs)} inputs:")
            for i, inp in enumerate(inputs):
                if hasattr(inp, 'shape'):
                    pi = module.inputs[i] if i < len(module.inputs) else None
                    pi_arg_name = getattr(pi, 'arg_name', None) if pi else None
                    print(f"[RESOLVER DEBUG]   [{i}] shape={inp.shape}, arg_name={pi_arg_name}")
                else:
                    print(f"[RESOLVER DEBUG]   [{i}] {type(inp).__name__}: {inp}")

            from dooly.profiler.profiler import prepare_op_args
            args, kwargs = prepare_op_args(callable_obj, module.operation_name, inputs)

            forward_context = None
            if layer_name and isinstance(workload_batch, AttentionBatchConfig):
                print(f"[RESOLVER DEBUG] Setting up forward_context for layer: {layer_name}")
                batch_config = AttentionBatchConfig(
                    batch_size=workload_batch.batch_size,
                    prefill_chunk_size=workload_batch.prefill_chunk_size,
                    kv_cache_size=workload_batch.kv_cache_size,
                    is_prefill=workload_batch.is_prefill,
                )
                try:
                    result = self.vlp.build_layer_metadata(layer_name, batch_config)
                    if result is None:
                        # Skip this config - exceeds available KV cache blocks
                        print(f"[RESOLVER]   Skipping config (exceeds KV cache): {batch_config}")
                        return False
                    layer_metadata, _ = result
                    forward_context = {layer_name: layer_metadata}
                    print(f"[RESOLVER DEBUG] forward_context set successfully")
                except Exception as e:
                    print(f"[RESOLVER]   Could not build metadata for {layer_name}: {e}")
                    forward_context = None
            else:
                print(f"[RESOLVER DEBUG] No forward_context needed (layer_name={layer_name}, is_AttentionBatchConfig={isinstance(workload_batch, AttentionBatchConfig)})")

            def run_callable():
                with torch.no_grad():
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
                        
                        # print(f"[RESOLVER DEBUG] Calling {module.operation_name} with:")
                        # print(f"[RESOLVER DEBUG]   Positional: {len(positional_tensors)} args")
                        # print(f"[RESOLVER DEBUG]   Kwargs: {list(kwarg_tensors.keys())}")
                        
                        for k, v in kwarg_tensors.items():
                            if hasattr(v, 'shape'):
                                print(f"[RESOLVER DEBUG]     {k}={v.shape}")
                        callable_obj(*positional_tensors, **kwarg_tensors)
                    else:
                        callable_obj(*args, **kwargs)

            def run_callable_with_context():
                from vllm.config import set_current_vllm_config
                from vllm.forward_context import set_forward_context
                with torch.no_grad():
                    # Same kwarg reconstruction as run_callable()
                    has_named = any(
                        getattr(pi, 'arg_name', None) is not None
                        for pi in module.inputs
                    )
                    if has_named and module.is_module:
                        positional_tensors = [
                            tensor for pi, tensor in zip(module.inputs, inputs)
                            if getattr(pi, 'arg_name', None) is None
                        ]
                        kwarg_tensors = {
                            pi.arg_name: tensor
                            for pi, tensor in zip(module.inputs, inputs)
                            if getattr(pi, 'arg_name', None) is not None
                        }
                        
                        # print(f"[RESOLVER DEBUG] Calling {module.operation_name} WITH CONTEXT:")
                        # print(f"[RESOLVER DEBUG]   Positional: {len(positional_tensors)} args")
                        # print(f"[RESOLVER DEBUG]   Kwargs: {list(kwarg_tensors.keys())}")
                        
                        for k, v in kwarg_tensors.items():
                            if hasattr(v, 'shape'):
                                print(f"[RESOLVER DEBUG]     {k}={v.shape}")
                        with set_current_vllm_config(self.vlp.vllm_config):
                            with set_forward_context(forward_context, self.vlp.vllm_config, virtual_engine=0):
                                callable_obj(*positional_tensors, **kwarg_tensors)
                    else:
                        with set_current_vllm_config(self.vlp.vllm_config):
                            with set_forward_context(forward_context, self.vlp.vllm_config, virtual_engine=0):
                                callable_obj(*args, **kwargs)

            run_fn = run_callable_with_context if forward_context else run_callable

            # Single forward pass
            run_fn()
            
            torch.cuda.synchronize()

            # Clean up
            del inputs, args, kwargs
            torch.cuda.empty_cache()

            return True

        except Exception as e:
            print(f"[RESOLVER]   Dummy run failed: {e}")
            # Clean up CUDA state
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return False



    # ------------------------------------------------------------------
    # Hierarchy walking & merging
    # ------------------------------------------------------------------

    def _build_module_info_from_event(
        self, parent_event: Event, hierarchy: OperationHierarchy, original: ModuleInfo
    ) -> Optional[ModuleInfo]:
        """Create a new ModuleInfo for a parent event in the hierarchy.
        
        For module parents: extracts inputs from the parent's MODULE taint annotation.
        For cpu_op parents: tries to find taint info from associated taints.
        """
        parent_name = parent_event.name
        is_module = (parent_event.cat == 'module')

        if is_module:
            shapes = parent_event.args.get("Input Dims", [])
            dtypes = parent_event.args.get("Input type", [])
            concrete_inputs = parent_event.args.get("Concrete Inputs", [])

            # MODULE: user_annotation events never have "Input Dims" in args — shape info
            # is encoded in event.params (the IN:[...] taint string). Derive shapes from it.
            if not shapes and parent_event.params:
                from dooly.profiler.taint import parse_taint_string
                taint_inputs = parse_taint_string(parent_event.params)
                shapes = [[dim.value for dim in ti.dimensions] for ti in taint_inputs]

            # print(f"[DEBUG @ resolver] parent_event.name: {parent_event.name}, parent_event.params: {parent_event.params}")
            # print(f"[DEBUG @ resolver] shapes: {shapes}")
            # print(f"[DEBUG @ resolver] concrete_inputs: {concrete_inputs}")
            try:
                inputs = extract_inputs_from_taint(parent_event.params, shapes, dtypes, concrete_inputs)
            except Exception:
                inputs = []
            if not inputs:
                # Fall back: use original module's inputs (may not be accurate, but better than nothing)
                print(f"[RESOLVER]   Using original inputs for parent {parent_name}")
                inputs = original.inputs

            # Update anchor to parent (original failed, so safe to mutate)
            hierarchy.anchor = parent_event

            return ModuleInfo(
                module_name=parent_name,
                operation_name=parent_name,
                inputs=inputs,
                is_module=True,
                backend=original.backend,
                dtype=original.dtype,
                hierarchy=hierarchy,
                count=original.count,  # inherit count from original module
            )
        else:
            # cpu_op parent: use the same import path, just different op name
            # Check if it has a namespace:: format (importable)
            if '::' not in parent_name:
                return None

            # Reuse original inputs — cpu_op siblings at the same level 
            # typically share the same input signature
            return ModuleInfo(
                module_name=original.module_name,
                operation_name=parent_name,
                inputs=original.inputs,
                is_module=False,
                backend=original.backend,
                dtype=original.dtype,
                hierarchy=hierarchy,
                count=original.count,
            )

    def _absorb_siblings(
        self,
        parent_event: Event,
        current_idx: int,
        all_modules: List[ModuleInfo],
        absorbed_indices: set,
        resolved_indices: set,
    ) -> Tuple[List[str], List[str]]:
        """Find and mark all sibling modules whose ancestry passes through parent_event.

        Any module whose anchor.parent chain includes parent_event gets absorbed.
        Returns:
            - absorbed_names: list of all absorbed operation names
            - absorbed_from_resolved: list of names that were already in resolved
        """
        absorbed_names = []
        absorbed_from_resolved = []

        for i, module in enumerate(all_modules):
            if i == current_idx or i in absorbed_indices:
                continue
            if module.hierarchy is None:
                continue

            # Check if this module's anchor is a descendant of parent_event
            current = module.hierarchy.anchor
            while current:
                if current is parent_event:
                    absorbed_indices.add(i)
                    absorbed_names.append(module.operation_name)
                    # Track if this was already resolved
                    if i in resolved_indices:
                        absorbed_from_resolved.append(module.operation_name)
                        resolved_indices.discard(i)
                    break
                current = current.parent

        return absorbed_names, absorbed_from_resolved

# ---------------------------------------------------------------------------
# Manual test: python -m dooly.profiler.resolver <trace.json> <model>
# Loads the real vLLM model, then shows which callable each module resolves to.
# ---------------------------------------------------------------------------

def main():
    import argparse, json
    from dooly.profiler.trace import TraceParser
    from dooly.profiler.vllm_layer_profiler import VLLMLayerProfiler, make_dict
    from dooly.profiler.importer import OpImporter

    parser = argparse.ArgumentParser(description="Test module path disambiguation with real model")
    parser.add_argument("trace", help="Path to .pt.trace.json file")
    parser.add_argument("model", help="HuggingFace model name (e.g. Qwen/Qwen3.5-2B)")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    # 1. Parse trace → ModuleInfos
    print(f"Parsing trace: {args.trace}")
    trace_parser = TraceParser()
    with open(args.trace) as f:
        data = json.load(f)
    trace_events = data.get('traceEvents', data)
    module_infos = trace_parser.extract_module_infos(trace_events)
    print(f"\nGot {len(module_infos)} module infos")

    # 2. Load vLLM model
    print(f"\nLoading model: {args.model}")
    vlp = VLLMLayerProfiler(
        model_name=args.model,
        dtype=args.dtype,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        gpu=args.gpu,
    )

    # 3. Build ModuleResolver and call _import_module for each module info
    resolver = ModuleResolver(vlp=vlp, importer=OpImporter())

    print(f"\n{'Operation':<35} {'Resolved path / result':<60} {'Type'}")
    print("-" * 115)
    for mi in module_infos:
        if mi.is_module:
            callable_obj = resolver._import_module(mi)
            if callable_obj is not None:
                # Show the actual submodule path used
                paths = resolver._name_to_module.get(mi.operation_name, [])
                result = type(callable_obj).__name__
            else:
                result = "FAILED"
        else:
            callable_obj = resolver._import_op(mi)
            result = str(callable_obj)[:50] if callable_obj else "FAILED"
        typ = "module" if mi.is_module else "op"
        print(f"{mi.operation_name:<35} {result:<60} {typ}")

    vlp.close()


if __name__ == "__main__":
    main()
