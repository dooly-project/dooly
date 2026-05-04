from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Set, Optional, Union, Callable
import json
import hashlib
import sqlite3
import os
from enum import Enum

from .finput import ProfileInput
from .parser import Event, OperationHierarchy

def enum_to_string(enum_value) -> str:
    """Convert an enum to its string value for JSON serialization."""
    if hasattr(enum_value, 'value'):
        return enum_value.value
    return str(enum_value)


def enum_list_to_strings(enum_list: List) -> List[str]:
    """Convert a list of enums to their string values for JSON serialization."""
    return [enum_to_string(e) for e in enum_list]

@dataclass
class ModuleInfo:
    module_name: str
    operation_name: str
    inputs: List[ProfileInput]
    is_module: bool = False
    call_chain: List[str] = field(default_factory=list)
    backend: str = ""
    dtype: str = ""
    is_collective_op: bool = False
    count: int = 0 # count of occurance (before deduplication)
    hierarchy: Optional[OperationHierarchy] = field(default=None, repr=False)
    absorbed_ops: List[str] = field(default_factory=list)  # ops merged into this module by resolver
    
    # Convenience properties for accessing hierarchy info
    @property
    def anchor(self) -> Optional[Event]:
        """The profiling anchor (nearest tainted ancestor)."""
        return self.hierarchy.anchor if self.hierarchy else None

    @property
    def root_module(self) -> Optional[Event]:
        """The outermost nn.Module containing this operation."""
        return self.hierarchy.root_module if self.hierarchy else None

    @property
    def nearest_module(self) -> Optional[Event]:
        """The nearest nn.Module parent of the anchor."""
        return self.hierarchy.nearest_module if self.hierarchy else None

    @property
    def kernels(self) -> List[Event]:
        """GPU kernels that use this anchor."""
        return self.hierarchy.kernels if self.hierarchy else []

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of this ModuleInfo
        """
        result = {
            'module_name': self.module_name,
            'operation_name': self.operation_name,
            'inputs': [inp.to_dict() for inp in self.inputs],
            'call_chain': self.call_chain,
            'backend': self.backend,
            'dtype': self.dtype,
        }

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModuleInfo':
        """
        Create ModuleInfo from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ModuleInfo instance
        """
        # Convert list of lists back to set of tuples
        return cls(
            module_name=data['module_name'],
            operation_name=data['operation_name'],
            inputs=[ProfileInput.from_dict(inp) for inp in data.get('inputs', [])],
            call_chain=data.get('call_chain', []),
            backend=data.get('backend', ''),
            dtype=data.get('dtype', ''),
        )

    def _semantic_params_list(self) -> List[Dict[str, Any]]:
        """
        Extracts MODEL_CONFIG dimensions and scalar values as semantic params from ProfileInput.
        """
        # Extract MODEL_CONFIG dimension values and scalars from ProfileInput
        params_dict = {}
        for i, inp in enumerate(self.inputs):
            if inp.type == "Tensor":
                # Extract MODEL_CONFIG dimensions
                from .taint import DimensionType
                for j, (dim_type, dim_val) in enumerate(zip(inp.dimensions, inp.actual_shape)):
                    if dim_type == DimensionType.MODEL_CONFIG:
                        # Use a generic name for MODEL_CONFIG dimensions
                        param_name = f"input{i}_dim{j}"
                        params_dict[param_name] = dim_val
            elif inp.type == "Scalar" and inp.scalar_value:
                # Extract scalar value
                param_name = f"input{i}_scalar"
                try:
                    # Try to parse as float
                    params_dict[param_name] = float(inp.scalar_value)
                except (ValueError, TypeError):
                    # Keep as string if not a number
                    params_dict[param_name] = inp.scalar_value

        params_list = [params_dict]

        # if self.hierarchy:
        #     print(self.hierarchy.anchor.name, "params_list :", params_list)
        return params_list


class ModuleComparator:
    """
    Compare module infos across models using semantic parameters.
    """
    def __init__(self):
        pass

    def _extract_params_from_event(self, event: Event) -> Dict[str, Any]:
        """Extract MODEL_CONFIG params from an Event's taint string and args.

        Used when the event doesn't have module_info attached.
        """
        from .finput import extract_inputs_from_taint
        from .taint import DimensionType

        params = {}

        if event.params:  # Has taint string
            shapes = event.args.get("Input Dims", [])
            dtypes = event.args.get("Input type", [])
            concrete_inputs = event.args.get("Concrete Inputs", [])

            try:
                inputs = extract_inputs_from_taint(
                    taint_string=event.params,
                    trace_input_dims=shapes,
                    trace_input_dtypes=dtypes,
                    concrete_inputs=concrete_inputs
                )
                if inputs:
                    for i, inp in enumerate(inputs):
                        if inp.type == "Tensor":
                            for j, (dim_type, dim_val) in enumerate(zip(inp.dimensions, inp.actual_shape)):
                                if dim_type == DimensionType.MODEL_CONFIG:
                                    params[f"input{i}_dim{j}"] = dim_val
                        elif inp.type == "Scalar" and inp.scalar_value:
                            try:
                                params[f"input{i}_scalar"] = float(inp.scalar_value)
                            except (ValueError, TypeError):
                                params[f"input{i}_scalar"] = inp.scalar_value
            except Exception:
                pass  # If extraction fails, return empty params

        return params

    def _collect_descendant_signatures(self, event: Event, depth: int = 0) -> List[Tuple[str, int, Tuple]]:
        """Recursively collect signatures from all descendants.

        Returns list of (name, depth, params_tuple) for each descendant.
        This captures the full operation tree structure.
        """
        signatures = []

        for idx, child in event.children.items():
            child_name = child.name

            # Extract params: prefer module_info if available, else parse event directly
            if child.module_info:
                params_list = child.module_info._semantic_params_list()
                params = params_list[0] if params_list else {}
            else:
                params = self._extract_params_from_event(child)

            params_tuple = tuple(sorted(params.items()))
            signatures.append((child_name, depth, params_tuple))

            # Recurse into grandchildren
            child_signatures = self._collect_descendant_signatures(child, depth + 1)
            signatures.extend(child_signatures)

        return signatures

    def _get_model_param_signature(self, module: ModuleInfo) -> Tuple[Tuple[str, Any], ...]:
        # Get anchor's own params first
        params_list = module._semantic_params_list()
        anchor_params = params_list[0] if params_list else {}
        anchor_params_tuple = tuple(sorted(anchor_params.items()))

        # Collect descendant signatures if children exist
        if module.hierarchy and module.hierarchy.anchor.children:
            descendant_sigs = self._collect_descendant_signatures(module.hierarchy.anchor)
            # Sort by (name, depth) for consistent ordering
            sorted_sigs = sorted(descendant_sigs, key=lambda x: (x[0], x[1]))
            # Combine anchor params with descendant signatures
            descendant_tuple = tuple((name, params) for name, depth, params in sorted_sigs)
            return (anchor_params_tuple, descendant_tuple)
        else:
            # No children: return just anchor's params
            return anchor_params_tuple

    def _module_signature(
        self, module: ModuleInfo
    ) -> Tuple[str, str, Tuple[Tuple[str, Any], ...]]:
        return (module.operation_name, self._get_model_param_signature(module))

    def _hash_signature(self, signature: Tuple) -> str:
        """Convert a signature tuple to a stable hash string."""
        canonical = json.dumps(signature, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _load_existing_signatures(self, db_path: str) -> Set[str]:
        """Load existing signature hashes from the database."""
        if not os.path.exists(db_path):
            return set()

        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute("SELECT signature_hash FROM signatures")
            return {row[0] for row in cursor}
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return set()
        finally:
            conn.close()

    def prune_overlaps(
        self,
        model_module_infos: List[List[Tuple[ModuleInfo, Callable]]],
        model_names: Optional[List[str]] = None,
        db_path: Optional[str] = None,
    ) -> Dict[str, List[Tuple[ModuleInfo, Callable, str, str]]]:
        if model_names is None:
            model_names = [f"Model{i+1}" for i in range(len(model_module_infos))]

        if len(model_module_infos) != len(model_names):
            raise ValueError(
                f"Number of model info lists ({len(model_module_infos)}) must match "
                f"number of names ({len(model_names)})"
            )

        # Load existing signatures from database
        existing_hashes = set()
        if db_path:
            existing_hashes = self._load_existing_signatures(db_path)
            if existing_hashes:
                print(f"[COMPARATOR] Loaded {len(existing_hashes)} existing signatures from database")

        pruned: Dict[str, List[Tuple[ModuleInfo, Callable, str]]] = {name: [] for name in model_names}
        all_ops: Dict[str, List[Tuple[ModuleInfo, Callable, str, str]]] = {name: [] for name in model_names}
        seen_this_session: Set[str] = set()

        for model_name, mod_call_tuple in zip(model_names, model_module_infos):
            for module, callable in mod_call_tuple:
                signature = self._module_signature(module)
                sig_hash = self._hash_signature(signature)
                sig_json = json.dumps(signature, sort_keys=True, default=str)

                # print(f"Model Name: {model_name}, Module Name: {module.module_name}, Hash: {sig_hash[:16]}...")

                # Always add to all_ops (for model_operations tracking)
                all_ops[model_name].append((module, callable, sig_hash, sig_json))

                # Skip if already in database (for profiling)
                if sig_hash in existing_hashes:
                    print(f" - {module.module_name} with hash {sig_hash[:16]}... already in database, skipping profiling")
                    continue

                # Skip if already seen this session (for profiling)
                if sig_hash in seen_this_session:
                    print(f" - {module.module_name} with hash {sig_hash[:16]}... already seen this session, skipping profiling")
                    continue

                seen_this_session.add(sig_hash)
                pruned[model_name].append((module, callable, sig_hash, sig_json))

        return pruned, all_ops
