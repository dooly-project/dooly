from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Set, Optional, Union, Callable
import functools
import json
import hashlib
import sqlite3
import os
from enum import Enum

from doolyprof.profiler.finput import ProfileInput
from doolyprof.profiler.parser import Event, OperationHierarchy

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
    # Short hash of the module instance's primitive attrs, captured by the
    # tracer via _extract_module_state_hash. Populated only for modules
    # (is_module=True) whose MODULE annotation carried a STATE: trailer.
    # Used by _module_signature so two layer instances that differ only in
    # runtime-branching config (e.g. sliding_window=None vs 4096) hash to
    # distinct signatures even when their kernel symbols match.
    state_hash: str = ""
    # Dotted paths (in the served model) of the module instances this info
    # stands for, from the tracer's PATH: field. After dedup, one per member of
    # the group. The resolver fetches the instance by path and verifies that
    # all members are configured and shaped identically. Never hashed.
    instance_paths: List[str] = field(default_factory=list)
    # The trace anchor of every member of the dedup group (the template's
    # first): the resolver takes a group's calls that run inside a parent it
    # profiles as one unit off the group's count, call by call. Never hashed.
    instance_anchors: List[Any] = field(default_factory=list, repr=False, compare=False)
    # For a Triton launch scope (tracer/triton_launch.py): how to launch the
    # kernel again from the generated tensor inputs (profiler/utils/triton_replay.py).
    launch_spec: Optional[dict] = None

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
            if inp.type in ("Tensor", "TensorList"):
                # Extract MODEL_CONFIG dimensions (per element for a TensorList)
                from doolyprof.profiler.taint import DimensionType
                for prefix, part in _tensor_part_prefixes(i, inp):
                    for j, (dim_type, dim_val) in enumerate(zip(part.dimensions, part.actual_shape)):
                        if dim_type == DimensionType.MODEL_CONFIG:
                            # Use a generic name for MODEL_CONFIG dimensions
                            params_dict[f"{prefix}_dim{j}"] = dim_val
                        elif part.dim_offset(j):
                            # a swept dim at sweep value + k (NUM_REQS+1) is a
                            # different input from one at the sweep value
                            params_dict[f"{prefix}_dim{j}_offset"] = part.dim_offset(j)
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

    def input_dtypes(self) -> Tuple[str, ...]:
        """The dtype of every input, by position (a signature field).

        ``dtype`` is only the first floating-point tensor's dtype, the one the
        kernel computes in. An op whose other inputs differ in dtype (an fp8
        weight with an fp32 or bf16 scale, int32 or int64 indices, a bf16 or
        fp32 residual) is a different op. A tensor's dtype is named by torch,
        as ``dtype`` is; a TensorList by its recorded dtype; any other
        argument is ''. Trailing '' entries are dropped, so an omitted
        trailing scalar argument does not count as a difference.
        """
        from doolyprof.profiler.input_generator import trace_dtype_to_torch, canonical_dtype_name

        out: List[str] = []
        for inp in self.inputs:
            if inp.type == "Tensor":
                td = trace_dtype_to_torch(inp.dtype) if inp.dtype else None
                out.append(canonical_dtype_name(td) if td is not None else (inp.dtype or ""))
            elif inp.type == "TensorList":
                out.append(inp.dtype or "")
            else:
                out.append("")
        while out and not out[-1]:
            out.pop()
        return tuple(out)


def _tensor_part_prefixes(i, inp):
    """(param-name prefix, Tensor ProfileInput) pairs for input ``i``:
    ``input{i}`` for a Tensor, ``input{i}_t{k}`` for element k of a TensorList."""
    if inp.type == "TensorList":
        return [(f"input{i}_t{k}", el) for k, el in enumerate(inp.tensor_parts())]
    return [(f"input{i}", inp)]


@functools.lru_cache(maxsize=None)
def is_builtin_operator(op_name: str) -> bool:
    """Whether ``op_name`` (``aten::pad``, ``aten::mul.Tensor``) is an operator
    PyTorch itself defines, by torch's own test
    (``torch._library.utils.is_builtin``).

    A name that does not resolve to a registered operator (a custom op whose
    library is not loaded, a module name, a Triton kernel), or a torch
    without that test, counts as not builtin, which keeps the parent's state
    in the signature (``parent_state_hash``).
    """
    ns, sep, rest = op_name.partition("::")
    if not sep or not ns or not rest:
        return False
    name, _, overload = rest.partition(".")
    try:
        import torch
        from torch._library.utils import is_builtin
        packet = getattr(getattr(torch.ops, ns), name)
        return bool(is_builtin(getattr(packet, overload or packet.overloads()[0])))
    except Exception:
        return False


def parent_state_hash(info: "ModuleInfo") -> str:
    """The state hash a raw-op anchor's signature takes from its nearest
    containing module, or '' when the op cannot see that module's state.

    A builtin PyTorch operator computes from its arguments and process-wide
    settings only, so the module it runs under can change its work only
    through the arguments, which the signature records (MODEL_CONFIG dims,
    scalar arguments, every input's dtype). Carrying the parent's hash would
    only split identical calls by where they were made (Kimi-K3's
    sequence-parallel pad under 16 decoder-layer configurations).

    Any other operator can read state its arguments do not show: vLLM's
    attention ops look up their layer, and with it the sliding window, by the
    layer name through the forward context. Those keep the parent's hash, so
    calls under differently configured layers stay apart.
    """
    if is_builtin_operator(info.operation_name):
        return ""
    nearest = info.hierarchy.nearest_module if info.hierarchy else None
    if nearest is None:
        return ""
    return (getattr(nearest, "args", {}) or {}).get("state_hash", "") or ""


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
        from doolyprof.profiler.finput import extract_inputs_from_taint
        from doolyprof.profiler.taint import DimensionType

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
                        if inp.type in ("Tensor", "TensorList"):
                            for prefix, part in _tensor_part_prefixes(i, inp):
                                for j, (dim_type, dim_val) in enumerate(zip(part.dimensions, part.actual_shape)):
                                    if dim_type == DimensionType.MODEL_CONFIG:
                                        params[f"{prefix}_dim{j}"] = dim_val
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

        Returns list of (name, depth, params_tuple) for each SEMANTIC
        descendant (torch-op / nn.Module / cpu_op / comm nodes, with their
        tainted input dims). Raw device-kernel nodes are excluded structurally
        by ``child.cat == 'kernel'`` (the Kineto category set in parser.py),
        not by a name heuristic, because their autotuner-chosen
        tile/prefix/format specialization varies with the traced token count
        and falsely splits identical ops.
        """
        signatures = []

        for idx, child in event.children.items():
            child_name = child.name

            if child.cat != 'kernel':
                # Extract params: prefer module_info if available, else parse event directly
                if child.module_info:
                    params_list = child.module_info._semantic_params_list()
                    params = params_list[0] if params_list else {}
                else:
                    params = self._extract_params_from_event(child)

                params_tuple = tuple(sorted(params.items()))
                signatures.append((child_name, depth, params_tuple))

            # Recurse into grandchildren (descend even through a dropped raw
            # kernel node so any semantic descendants are preserved).
            child_signatures = self._collect_descendant_signatures(child, depth + 1)
            signatures.extend(child_signatures)

        return signatures

    def _get_model_param_signature(self, module: ModuleInfo) -> Tuple[Tuple[str, Any], ...]:
        """The anchor's own MODEL_CONFIG dims and scalar args, sorted."""
        params_list = module._semantic_params_list()
        anchor_params = params_list[0] if params_list else {}
        return tuple(sorted(anchor_params.items()))

    def _descendant_signature(self, module: ModuleInfo) -> Tuple[Tuple[str, Any], ...]:
        """Semantic ops executed under the anchor, as (name, tainted dims),
        sorted. Recorded in signature_json for inspection; NOT part of the
        hash (see _module_signature)."""
        if module.hierarchy and module.hierarchy.anchor.children:
            descendant_sigs = self._collect_descendant_signatures(module.hierarchy.anchor)
            sorted_sigs = sorted(descendant_sigs, key=lambda x: (x[0], x[1]))
            return tuple((name, params) for name, depth, params in sorted_sigs)
        return ()

    def _module_signature(
        self, module: ModuleInfo
    ) -> Tuple:
        # Signature = operation_name
        #           + the anchor's own MODEL_CONFIG dims / scalar args
        #           + state_hash (own for modules, else nearest parent
        #             module's unless the op is a builtin PyTorch operator,
        #             see parent_state_hash — the module's configuration,
        #             including the attention backend's implementation class
        #             and its constructor contract, cache dtypes, sliding
        #             window ...)
        #           + dtype (derived from the op's own traced inputs)
        #           + every input's dtype (ModuleInfo.input_dtypes).
        #
        # Deliberately NOT hashed: the ops executed under the anchor
        # (descendants) and GPU kernel names. Both describe how the
        # engine happened to execute this instance, not what it computes:
        # the descendant list differed between instances of one layer class
        # only through one-time work (Triton autotuning, per-forward cached
        # metadata, tracer dispatch records), and the kernel symbols change
        # with the traced token count. Over every database produced so far
        # the descendant list never separated two anchors that agreed on the
        # first four hashed fields, except the attention backend, which the
        # state hash now carries explicitly. The descendants are still written
        # to signature_json (6th element) for inspection.
        if module.is_module or module.launch_spec:
            # a Triton launch scope hashes its own fixed arguments: one kernel
            # launched with different constexprs (Kimi-K3's attention residual
            # mixes a layer-dependent number of blocks) is a different kernel
            state_hash = module.state_hash or ""
        else:
            # cpu_op (non-module) anchor: the nearest containing module's
            # state hash (the tracer stashed it on the module event's args via
            # _extract_module_state_hash) when the op can read that module's
            # configuration; '' for a builtin PyTorch operator, whose work
            # its own arguments fix (parent_state_hash).
            state_hash = parent_state_hash(module)

        parts = (
            module.operation_name,
            self._get_model_param_signature(module),
            state_hash,
            module.dtype or "",
            module.input_dtypes(),
        )
        return parts

    def _hash_signature(self, signature: Tuple) -> str:
        """Convert a signature tuple to a stable hash string."""
        canonical = json.dumps(signature, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _load_existing_signatures(self, db_path: str, device: Optional[str] = None,
                                  engine: Optional[str] = None) -> Set[str]:
        """Signature hashes the database already has measurements of on ``device`` with
        ``engine`` (doolyprof.profiler.db). Rows of other devices or engines, and rows written
        before devices were recorded, are not reused."""
        if not os.path.exists(db_path):
            return set()

        conn = sqlite3.connect(db_path)
        try:
            measured = {row[0] for row in conn.execute(
                "SELECT DISTINCT signature_hash FROM results WHERE device = ? AND engine = ?", (device, engine))}
            elsewhere = conn.execute(
                "SELECT COUNT(DISTINCT signature_hash) FROM results WHERE device IS NOT ? OR engine IS NOT ?",
                (device, engine)).fetchone()[0]
            if elsewhere:
                print(f"[COMPARATOR] {elsewhere} operation(s) in the database were measured on another device or "
                      f"engine (or before devices were recorded); they are measured again for {device}, {engine}")
            return measured
        except sqlite3.OperationalError:
            # No results table yet, or one from before devices were recorded
            try:
                n = conn.execute("SELECT COUNT(DISTINCT signature_hash) FROM results").fetchone()[0]
                if n:
                    print(f"[COMPARATOR] the database predates GPU and engine labels; its {n} operation(s) are "
                          f"measured again for {device}, {engine}")
            except sqlite3.OperationalError:
                pass
            return set()
        finally:
            conn.close()

    def prune_overlaps(
        self,
        model_module_infos: List[List[Tuple[ModuleInfo, Callable]]],
        model_names: Optional[List[str]] = None,
        db_path: Optional[str] = None,
        overwrite: bool = False,
        device: Optional[str] = None,
        engine: Optional[str] = None,
    ) -> Dict[str, List[Tuple[ModuleInfo, Callable, str, str]]]:
        if model_names is None:
            model_names = [f"Model{i+1}" for i in range(len(model_module_infos))]

        if len(model_module_infos) != len(model_names):
            raise ValueError(
                f"Number of model info lists ({len(model_module_infos)}) must match "
                f"number of names ({len(model_names)})"
            )

        # Load existing signatures from database (unless overwrite is True)
        existing_hashes = set()
        if db_path and not overwrite:
            existing_hashes = self._load_existing_signatures(db_path, device, engine)
            if existing_hashes:
                print(f"[COMPARATOR] Loaded {len(existing_hashes)} signatures already measured on {device} with {engine}")
        elif overwrite:
            print(f"[COMPARATOR] Overwrite mode enabled - ignoring existing signatures in database")

        pruned: Dict[str, List[Tuple[ModuleInfo, Callable, str]]] = {name: [] for name in model_names}
        all_ops: Dict[str, List[Tuple[ModuleInfo, Callable, str, str]]] = {name: [] for name in model_names}
        seen_this_session: Set[str] = set()

        for model_name, mod_call_tuple in zip(model_names, model_module_infos):
            for module, callable in mod_call_tuple:
                signature = self._module_signature(module)
                sig_hash = self._hash_signature(signature)
                sig_json = json.dumps(signature + (self._descendant_signature(module),),
                                      sort_keys=True, default=str)

                # print(f"Model Name: {model_name}, Module Name: {module.module_name}, Hash: {sig_hash[:16]}...")

                # Always add to all_ops (for model_operations tracking)
                all_ops[model_name].append((module, callable, sig_hash, sig_json))

                # Skip if already in database (for profiling)
                if sig_hash in existing_hashes:
                    print(f" - {module.module_name} with hash {sig_hash[:16]}... already measured on this device and engine, skipping profiling")
                    continue

                # Skip if already seen this session (for profiling)
                if sig_hash in seen_this_session:
                    print(f" - {module.module_name} with hash {sig_hash[:16]}... already seen this session, skipping profiling")
                    continue

                seen_this_session.add(sig_hash)
                pruned[model_name].append((module, callable, sig_hash, sig_json))

        return pruned, all_ops
