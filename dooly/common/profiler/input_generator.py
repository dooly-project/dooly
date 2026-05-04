from .module import ModuleInfo
from typing import List, Dict, Any, Tuple, Optional, Union
import torch
import sys

from .utils.generate_batch import (generate_gradient_inputs, generate_single_max_inputs,
                                                         generate_vidur_inputs, generate_multi_max_inputs,
                                                         generate_vidur_num_tokens, AttentionBatchConfig,
                                                         get_attention_batch_sizes_to_profile)


def is_attention_operation(module: ModuleInfo) -> bool:
    attention_ops = [
        "attention", "Attention", "atten", "Atten", "attn", "Attn"
    ]
    for op in attention_ops:
        if op in module.module_name:
            return True
    return False


def classify_op(module: ModuleInfo) -> str:
    """Classify an op so the profiler can choose an input sweep strategy."""
    name = module.module_name or ""
    if is_attention_operation(module):
        return "attention"
    for tok in ("FusedMoE", "SharedFusedMoE", "MoE"):
        if tok in name:
            return "moe"
    for tok in ("Mamba", "MambaMixer"):
        if tok in name:
            return "mamba"
    if "SSM" in name:
        return "ssm"
    return "stateless"



class InputGenerator:
    # for a given ModuleInfo, generate the inputs needed to profile
    def __init__(self, module: ModuleInfo, max_num_request: int=16,
                 max_num_token: int=1024, min_num_requests:int=1, test_counts: int=10,
                 dtype: torch.dtype = torch.bfloat16):
        self.module = module
        self.max_num_request = max_num_request
        self.max_num_token = max_num_token
        self.test_counts = test_counts
        self.batches = []
        self.seen_batches = set()
        self.dtype = dtype
        self.min_num_requests = min_num_requests

        self.dtype_map = {
            "float32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "c10::BFloat16": torch.bfloat16,
            "long": torch.long,
            "long int": torch.long,
            "int": torch.int32,
            "int32": torch.int32,
            "int64": torch.long,
            "bool": torch.bool,
            "double": torch.float64,
            "trace_dtype": None,
            # torch.* prefixed variants (from MODULE DTYPES section)
            "torch.float32": torch.float32,
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.int32": torch.int32,
            "torch.int64": torch.long,
            "torch.long": torch.long,
            "torch.bool": torch.bool,
            "torch.float64": torch.float64,
        }

    def _parse_scalar_value(self, scalar_str: str) -> Any:
        """
        Parse a scalar value string into its appropriate Python type.
        Handles: floats, ints, bools, lists (e.g., "[4096]"), and strings.
        """
        import ast

        # Try parsing as a Python literal (handles lists, ints, floats, bools, tuples)
        try:
            value = ast.literal_eval(scalar_str)
            return value
        except (ValueError, SyntaxError):
            pass

        # Try float (handles scientific notation like "1.0e-05")
        try:
            value = float(scalar_str)
            # Check if it's actually an int
            if value == int(value) and 'e' not in scalar_str.lower() and '.' not in scalar_str:
                return int(value)
            return value
        except ValueError:
            pass

        # Try bool
        if scalar_str.lower() in ('true', 'false'):
            return scalar_str.lower() == 'true'

        # Keep as string
        return scalar_str

    def _create_tensor_inputs_from_params(
        self, workload_config: Union[AttentionBatchConfig, Dict[str, int], List[int]]
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Create tensor inputs using ProfileInput structure.

        Args:
            workload_config: Either an AttentionBatchConfig, a dict with 'num_tokens'/'num_requests',
                           or a List[int] (legacy format from resolver)

        Returns:
            Tuple of (input_tensors, workload_params) where workload_params contains the
            actual num_tokens and num_requests values used
        """
        inputs = []
        trace_dtype = self.dtype

        from .taint import DimensionType

        # Extract num_tokens and num_requests from workload_config
        if isinstance(workload_config, AttentionBatchConfig):
            num_tokens = workload_config.num_tokens
            num_requests = workload_config.batch_size
            workload_params = {
                'num_tokens': num_tokens,
                'num_requests': num_requests,
                'is_prefill': workload_config.is_prefill,
                'prefill_chunk_size': workload_config.prefill_chunk_size,
                'kv_cache_size': workload_config.kv_cache_size
            }
        elif isinstance(workload_config, list):
            # Legacy List[int] format from resolver
            # First element is num_tokens, length is num_requests
            num_tokens = workload_config[0] if workload_config else 1
            num_requests = len(workload_config)
            workload_params = {
                'num_tokens': num_tokens,
                'num_requests': num_requests
            }
        else:
            # Dict format from non-attention operations
            num_tokens = workload_config.get('num_tokens', 1)
            num_requests = workload_config.get('num_requests', 1)
            workload_params = {
                'num_tokens': num_tokens,
                'num_requests': num_requests
            }

        for input_idx, profile_input in enumerate(self.module.inputs):
            if profile_input.type == "Tensor":
                # Build shape: replace workload dimensions with sweep values
                shape = []
                num_tokens_dim_indices = profile_input.get_toks_dim()
                num_requests_dim_indices = profile_input.get_reqs_dim()
                mix_dim_indices = profile_input.get_mix_dims()

                # Debug: print dimension info for first input
                # if input_idx == 0:
                #     print(f"[DEBUG InputGenerator] First input dimensions: {profile_input.dimensions}")
                #     print(f"[DEBUG InputGenerator] NUM_TOKS dims: {num_tokens_dim_indices}")
                #     print(f"[DEBUG InputGenerator] MIX dims: {mix_dim_indices}")
                #     print(f"[DEBUG InputGenerator] Components: {profile_input.dimension_components}")

                for dim_idx, actual_dim_value in enumerate(profile_input.actual_shape):
                    if dim_idx in num_tokens_dim_indices:
                        # NUM_TOKENS dimension - use num_tokens from workload config
                        shape.append(num_tokens)
                    elif dim_idx in num_requests_dim_indices:
                        # NUM_REQUESTS dimension - use num_requests from workload config
                        shape.append(num_requests)
                    elif dim_idx in mix_dim_indices:
                        # MIX dimension - calculate size from components
                        # e.g., NUM_TOKS(269) × MODEL_CONFIG(40) = 10760
                        # When sweeping with num_tokens=100, becomes 100 × 40 = 4000
                        components = profile_input.dimension_components[dim_idx] if profile_input.dimension_components else None
                        if components:
                            new_size = 1
                            # if input_idx == 0 and dim_idx == 0:
                            #     print(f"[DEBUG MIX] Processing MIX dim: components={components}, num_tokens={num_tokens}, num_requests={num_requests}")
                            for taint_name, qty in components.items():
                                if 'NUM_TOK' in taint_name:
                                    new_size *= num_tokens
                                    # if input_idx == 0 and dim_idx == 0:
                                    #     print(f"[DEBUG MIX]   {taint_name}: {qty} → multiply by num_tokens={num_tokens}, new_size={new_size}")
                                elif 'NUM_REQ' in taint_name:
                                    new_size *= num_requests
                                    # if input_idx == 0 and dim_idx == 0:
                                    #     print(f"[DEBUG MIX]   {taint_name}: {qty} → multiply by num_requests={num_requests}, new_size={new_size}")
                                else:
                                    # MODEL_CONFIG or other - use original quantity
                                    new_size *= qty
                                    # if input_idx == 0 and dim_idx == 0:
                                    #     print(f"[DEBUG MIX]   {taint_name}: {qty} → multiply by qty={qty}, new_size={new_size}")
                            # if input_idx == 0 and dim_idx == 0:
                            #     print(f"[DEBUG MIX] Final new_size for MIX dim: {new_size}")
                            shape.append(new_size)
                        else:
                            # MIX without components - fallback to fixed value (shouldn't happen)
                            shape.append(actual_dim_value)
                    else:
                        # MODEL_CONFIG dimension - keep fixed value from trace
                        shape.append(actual_dim_value)

                # A zero dim usually indicates an optional Tensor arg that was
                # None in the traced call. Preserve positional alignment by
                # emitting None instead of failing the whole config.
                if not all(isinstance(d, int) and d > 0 for d in shape):
                    if any(isinstance(d, int) and d == 0 for d in shape):
                        print(f"[InputGenerator] Shape {shape} has zero dim — treating as None")
                        inputs.append(None)
                        continue
                    print(f"[InputGenerator] Invalid shape {shape}, skipping input")
                    print(f"[InputGenerator]   actual_shape={profile_input.actual_shape}")
                    print(f"[InputGenerator]   dimensions={[d.value for d in profile_input.dimensions]}")
                    print(f"[InputGenerator]   num_requests_dim_indices={num_requests_dim_indices}")
                    print(f"[InputGenerator]   num_tokens_dim_indices={num_tokens_dim_indices}")
                    print(f"[InputGenerator]   num_tokens={num_tokens}, num_requests={num_requests}")
                    return [], workload_params

                # Get dtype
                input_dtype_str = profile_input.dtype
                if input_dtype_str == "trace_dtype":
                    input_dtype = trace_dtype
                else:
                    input_dtype = self.dtype_map.get(input_dtype_str, trace_dtype)

                shape_tuple = tuple(shape)

                # Create tensor
                if input_dtype in (torch.long, torch.int32, torch.int64):
                    if len(shape) == 1:
                        max_val = shape[0] if shape[0] < 10000 else 1000
                    else:
                        max_val = 1000
                    tensor = torch.randint(0, max_val, shape_tuple, dtype=input_dtype, device='cuda')
                else:
                    tensor = torch.randn(shape_tuple, dtype=input_dtype, device='cuda')
                inputs.append(tensor)

            elif profile_input.type == "Scalar":
                # Use scalar_value directly
                if profile_input.scalar_value:
                    value = self._parse_scalar_value(profile_input.scalar_value)
                    inputs.append(value)
                else:
                    # Provide default based on dtype
                    if profile_input.dtype == "string":
                        inputs.append("auto")
                    elif profile_input.dtype == "bool":
                        inputs.append(False)
                    elif profile_input.dtype in ("int", "long"):
                        inputs.append(0)
                    elif profile_input.dtype in ("float", "double"):
                        inputs.append(0.0)
                    # else:
                    #     # Empty scalar with unknown dtype (e.g., "Scalar") represents None
                    #     # This handles Tensor? arguments that were None in the trace
                    #     inputs.append(None)

        # print(f"[InputGenerator] Generated inputs: {[i.shape if isinstance(i, torch.Tensor) else i for i in inputs]}")
        return inputs, workload_params

    def _generate_num_requests_sweep(self) -> List[int]:
       return get_attention_batch_sizes_to_profile(min_batch_size=self.min_num_requests, max_batch_size=self.max_num_request) 

    def _generate_num_tokens_sweep(self) -> List[int]:
        """Generate sweep values for num_tokens based on max_num_token."""
        return generate_vidur_num_tokens(max_size=self.max_num_token)

    def prepare_inputs(self, mode: str, use_batch_config: bool = None) -> Union[List[AttentionBatchConfig], List[Dict[str, int]]]:
        if use_batch_config is True:
            return self._prepare_attention_inputs(mode)
        if use_batch_config is False:
            return self._prepare_stateless_inputs()

        op_class = classify_op(self.module)
        if op_class == "attention":
            return self._prepare_attention_inputs(mode)
        return self._prepare_stateless_inputs()

    def _prepare_attention_inputs(self, mode: str) -> List[AttentionBatchConfig]:
        if mode != "vidur":
            raise NotImplementedError(f"Mode {mode} is not implemented yet")
        self.batches = generate_vidur_inputs(
            max_seq_len=self.max_num_token,
            min_batch_size=1,
            max_batch_size=self.max_num_request,
            profile_only_prefill=False,
            profile_only_decode=False,
        )
        return self.batches

    def _prepare_stateless_inputs(self) -> List[Dict[str, int]]:
        num_tokens_sweep = self._generate_num_tokens_sweep()
        num_requests_sweep = self._generate_num_requests_sweep()

        has_num_tokens_dim = any(
            inp.get_toks_dim() for inp in self.module.inputs if inp.type == "Tensor"
        )
        has_num_requests_dim = any(
            inp.get_reqs_dim() for inp in self.module.inputs if inp.type == "Tensor"
        )

        for inp in self.module.inputs:
            if inp.type == "Tensor" and inp.dimension_components:
                for dim_idx in inp.get_mix_dims():
                    components = inp.dimension_components[dim_idx]
                    if components:
                        for taint_name in components.keys():
                            if 'NUM_TOK' in taint_name:
                                has_num_tokens_dim = True
                            if 'NUM_REQ' in taint_name:
                                has_num_requests_dim = True

        workload_configs = []
        if has_num_tokens_dim and has_num_requests_dim:
            for num_tokens in num_tokens_sweep:
                for num_requests in num_requests_sweep:
                    if num_tokens >= num_requests:
                        workload_configs.append({
                            'num_tokens': num_tokens,
                            'num_requests': num_requests
                        })
        elif has_num_tokens_dim:
            for num_tokens in num_tokens_sweep:
                workload_configs.append({
                    'num_tokens': num_tokens,
                    'num_requests': 1
                })
        elif has_num_requests_dim:
            for num_requests in num_requests_sweep:
                workload_configs.append({
                    'num_tokens': num_requests,
                    'num_requests': num_requests
                })
        else:
            workload_configs.append({
                'num_tokens': 1,
                'num_requests': 1
            })

        self.batches = workload_configs
        return self.batches
