import importlib
import inspect
from typing import Dict, Callable, Optional, Set
import os
import torch, torch._C
import torch.nn.functional as F

class OpImporter:
    def __init__(self):
        self._imported_vllm_ops = set()
        self._imported_sglang_ops = set()
        # Track which ops require forward context (auto-detected)
        self._context_required_ops: Set[str] = set()
        # Map op name to its module for context checking
        self._op_to_module: Dict[str, str] = {}

    def _get_registered_ops(self) -> set:
        """Get current registered ops (fresh, not cached)."""
        return set(torch._C._dispatch_get_all_op_names())

    def _check_requires_forward_context(self, module_path: str, op_name: str) -> bool:
        """
        Check if an operation requires forward context by inspecting its module source.

        Operations that use get_forward_context() need special setup for profiling.
        """
        try:
            module = importlib.import_module(module_path)
            source = inspect.getsource(module)

            # Check if the module uses get_forward_context
            # This is the pattern used by mamba_mixer, fused_moe, kda, linear_attn, etc.
            if 'get_forward_context()' in source:
                return True

            # Also check for static_forward_context registration pattern
            if 'static_forward_context[' in source:
                return True

            return False
        except Exception:
            return False

    def requires_forward_context(self, op_full_name: str) -> bool:
        """Check if an operation requires forward context setup."""
        return op_full_name in self._context_required_ops

    def get_context_required_ops(self) -> Set[str]:
        """Get all operations that require forward context."""
        return self._context_required_ops.copy()

    def _try_import_vllm_custom_op(self, op_name: str) -> bool:
        """
        Dynamically find and import the vLLM module that contains a custom op.
        Searches vLLM's model_executor.layers directory structure.
        """
        if op_name in self._imported_vllm_ops:
            return False
        self._imported_vllm_ops.add(op_name)

        try:
            import vllm.model_executor.layers as layers_pkg
        except ImportError:
            return False

        layers_path = os.path.dirname(layers_pkg.__file__)
        base_path = os.path.dirname(layers_path)

        for root, dirs, files in os.walk(layers_path):
            for file in files:
                if not file.endswith('.py') or file.startswith('_'):
                    continue

                module_name = file[:-3]
                if op_name == module_name or op_name in module_name:
                    rel_from_vllm = os.path.relpath(root, os.path.dirname(base_path))
                    module_path = rel_from_vllm.replace(os.sep, '.') + '.' + module_name
                    full_module = f'vllm.{module_path}'

                    try:
                        importlib.import_module(full_module)
                        op_full_name = f'vllm::{op_name}'
                        if op_full_name in torch._C._dispatch_get_all_op_names():
                            self._op_to_module[op_full_name] = full_module
                            if self._check_requires_forward_context(full_module, op_name):
                                self._context_required_ops.add(op_full_name)
                            return True
                    except ImportError:
                        continue

        return False

    def _try_import_sglang_custom_op(self, op_name: str) -> bool:
        """
        Dynamically find and import the sglang module that contains a custom op.
        Searches sglang.jit_kernel and sglang.srt.layers directory structures.
        sglang ops register under torch.ops.sglang when their module is imported
        (eager=True is the default in register_custom_op).
        """
        if op_name in self._imported_sglang_ops:
            return False
        self._imported_sglang_ops.add(op_name)

        search_pkgs = []
        try:
            import sglang.jit_kernel as jit_pkg
            search_pkgs.append(jit_pkg)
        except ImportError:
            pass
        try:
            import sglang.srt.layers as layers_pkg
            search_pkgs.append(layers_pkg)
        except ImportError:
            pass

        if not search_pkgs:
            return False

        op_full_name = f'sglang::{op_name}'

        for pkg in search_pkgs:
            if pkg.__file__ is None:
                continue  # namespace package — no filesystem root to walk
            pkg_path = os.path.dirname(pkg.__file__)
            pkg_root = os.path.dirname(pkg_path)

            for root, dirs, files in os.walk(pkg_path):
                # skip test/benchmark subtrees
                if any(part in ('tests', 'benchmark', 'benchmarks') for part in root.split(os.sep)):
                    continue
                for file in files:
                    if not file.endswith('.py') or file.startswith('_'):
                        continue

                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path) as fh:
                            src = fh.read()
                    except OSError:
                        continue

                    if op_name not in src:
                        continue

                    module_name = file[:-3]
                    rel = os.path.relpath(root, os.path.dirname(pkg_root))
                    full_module = rel.replace(os.sep, '.') + '.' + module_name

                    try:
                        importlib.import_module(full_module)
                    except ImportError:
                        continue

                    if op_full_name in torch._C._dispatch_get_all_op_names():
                        self._op_to_module[op_full_name] = full_module
                        if self._check_requires_forward_context(full_module, op_name):
                            self._context_required_ops.add(op_full_name)
                        return True

        return False

    def import_op(self, op_full_name: str) -> tuple[Optional[Callable], str]:
        namespace, full_op_name = op_full_name.split("::")
        try:
            # handle overloads (e.g., 'add.Tensor' -> 'add')
            if "." in full_op_name:
                op_name = full_op_name.split(".")[0]
            else:
                op_name = full_op_name

            # For custom ops, import the module first so the op registers itself
            if namespace == "vllm":
                self._try_import_vllm_custom_op(op_name)
            elif namespace == "sglang":
                self._try_import_sglang_custom_op(op_name)

            # check if torch ops has namespace
            if not hasattr(torch.ops, namespace):
                return None, f"Namespace '{namespace}' not found"

            # Get the operator library
            lib = getattr(torch.ops, namespace)
            if not hasattr(lib, op_name):
                return None, f"Operator '{op_name}' not found in {namespace}"

            # Return OpOverloadPacket - handles overload dispatch automatically
            # e.g., torch.ops.aten.index dispatches to aten::index.Tensor
            op_packet = getattr(lib, op_name)
            return op_packet, "Success"

        except Exception as e:
            return None, f"Error: {e}"
