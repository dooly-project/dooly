from typing import Dict, List, Any, Optional, Tuple, Callable
import torch
import json

from doolyprof.profiler.module import ModuleInfo, ModuleComparator
from doolyprof.profiler.trace import TraceParser
from vllm.transformers_utils.config import get_config

class ModelInfo:
    def __init__(
        self,
        name: str,
        trace_dir: str,
        dtype: torch.dtype
    ):
        # load model configuration
        self.config = get_config(name, trust_remote_code=True)
        self.name = name
        self.dtype = dtype
        self.model_family = self.get_model_family()

        # Every shape the profiler uses comes from the trace's taints; nothing
        # here derives sizes from config attributes (a head_dim computed as
        # hidden_size // num_heads, asserted against the config, was wrong for
        # every model whose head size is set on its own: MLA, Qwen3, Gemma).
        self.max_model_len = self.get_max_model_len()

        parser = TraceParser()
        self.all_ops: List[ModuleInfo] = parser.parse_trace(
            trace_path=trace_dir,
            verbose=False,
            metadata_path=None
        )
        # Preserve the tracer-time taint registry so the resolver can do
        # inverse lookups (e.g. which value carries NUM_REQS?) without having
        # to rescan the trace file.
        self.taint_registry = getattr(parser, 'taint_registry', {}) or {}

        self.used_ops: List[ModuleInfo] = [op for op in self.all_ops if not op.is_collective_op]
        self.collective_ops: List[ModuleInfo] = [op for op in self.all_ops if op.is_collective_op]
        self.to_profile: List[ModuleInfo] = self.used_ops
        
        self.all_resolved_ops: List[Tuple[ModuleInfo, Callable, str, str]] = []  # All ops with signatures
        self.to_profile_resolved: List[Tuple[ModuleInfo, Callable]] = []
        
    def compare_with_many(self, models: List['ModelInfo'], db_path: Optional[str] = None, overwrite=False) -> None:
        all_models = list(models)
        if self not in all_models:
            all_models.insert(0, self)

        comparator = ModuleComparator()
        # list of model names
        model_names = [model.name for model in all_models]
        # list of [list of (ModuleInfo, Callable)]
        model_modules = [model.to_profile_resolved for model in all_models]
        pruned, all_ops = comparator.prune_overlaps(model_modules, model_names, db_path=db_path, overwrite=overwrite)

        # update all models
        for model in all_models:
            # to_profile_resolved: operations that need profiling
            model.to_profile_resolved = pruned.get(model.name, [])
            model.to_profile = [module for module, _, _, _ in model.to_profile_resolved]
            # all_resolved_ops: ALL operations with signatures (for model_operations table)
            model.all_resolved_ops = all_ops.get(model.name, [])


    # common values across model configs
    def get_model_family(self) -> str:
        """Get model family from config's model_type attribute."""
        return getattr(self.config, "model_type", "unknown")

    def get_max_model_len(self) -> Optional[int]:
        candidates = [
            "max_position_embeddings",
            "max_sequence_length",
            "seq_length",
            "n_positions",
        ]
        for attr in candidates:
            value = getattr(self.config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    # save a json file with Dict[(model_name), (List[Dict[(module_name), (operation_name), (params)]])]
    def to_json(self, models: List['ModelInfo'], json_output: str) -> Dict[str, List[Dict[str, Any]]]: 
        print(f"Outputting profiling results to {json_output}")
        all_models = list(models)
        if self not in all_models:
            all_models.insert(0, self)

        output_data: Dict[str, Dict[str, List[str]]] = {}
        for model in all_models:
            output_data[model.name] = {}
            for module in model.to_profile:
                if module.module_name not in output_data[model.name]:
                    output_data[model.name][module.module_name] = []
                output_data[model.name][module.module_name].append(module.operation_name)

        with open(json_output, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        return output_data
