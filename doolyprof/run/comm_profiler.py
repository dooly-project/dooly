import argparse
import os
import sys
import torch
import time
import numpy as np
from typing import List, Dict, Tuple, Any, Literal
import sqlite3
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM
from doolyprof.profiler.utils.gpu_memory import engine_memory_fraction
from doolyprof.profiler.utils.topology import group_topology
from vllm.distributed.parallel_state import get_tp_group

# Dtype to bytes mapping
DTYPE_TO_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int8: 1,
}


GRAPH_OPS = 8  # collectives per captured graph in "graph" timing (a second graph holds 2x)


def _graph_replay_ms(graph, warmup_iters, measure_iters):
    for _ in range(max(1, warmup_iters // 2)):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(measure_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        graph.replay()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    return float(np.median(times))


def _time_graph(op_fn, warmup_iters, measure_iters, capture_ctx=None, ops=GRAPH_OPS):
    """GPU time of one collective, from CUDA-graph replays (no CPU launch time).

    Two graphs are captured, with ``ops`` and ``2 * ops`` back-to-back calls, and
    the result is (T(2*ops) - T(ops)) / ops: the graph launch and the wait for the
    slower rank at the first collective are the same in both and cancel.
    ``capture_ctx`` is the engine's own capture context (GroupCoordinator.graph_capture),
    so communicators that keep graph-specific state (custom all-reduce registers
    the captured buffers) behave as in the engine's CUDA-graph mode."""
    import contextlib
    for _ in range(warmup_iters):
        op_fn()
    torch.cuda.synchronize()
    medians = []
    for n in (ops, 2 * ops):
        graph = torch.cuda.CUDAGraph()
        ctx = capture_ctx() if capture_ctx is not None else contextlib.nullcontext()
        with ctx as cap:
            stream = getattr(cap, "stream", None) or torch.cuda.Stream()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(n):
                    op_fn()
        torch.cuda.synchronize()
        medians.append(_graph_replay_ms(graph, warmup_iters, measure_iters))
        del graph
    return max(0.0, (medians[1] - medians[0]) / ops)


def _time_collective(op_fn, warmup_iters: int, measure_iters: int,
                     comm_timing: str = "isolated", capture_ctx=None) -> float:
    """Time a collective ``op_fn()`` (no-arg callable) and return latency in ms.

    Modes (default preserves the original per-call ISOLATED behavior):

    - ``"graph"``: GPU time only, from CUDA-graph replays (``_time_graph``).
      Excludes the CPU time to launch each call, which ISOLATED counts as GPU
      idle time between its events and which dominates small messages.

    - ``"isolated"``: each iteration does sync -> record start -> ONE collective
      -> record end -> sync; returns the MEDIAN per-call GPU time. Every call
      pays full NCCL kernel startup-from-idle, which overestimates the cost of
      collectives that actually run back-to-back inside the model forward.

    - ``"amortized"``: record start -> run ``measure_iters`` collectives
      back-to-back -> record end -> ONE sync at the end; returns total / iters.
      Kernels queue back-to-back (no per-call idle gap), closer to how the model
      forward issues them. Additive experiment-only option.
    """
    if comm_timing == "graph":
        return _time_graph(op_fn, warmup_iters, measure_iters, capture_ctx)

    # Warmup (identical for both modes)
    for _ in range(warmup_iters):
        op_fn()
        torch.cuda.synchronize()

    if comm_timing == "amortized":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(measure_iters):
            op_fn()
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / measure_iters

    # Default: isolated per-call timing, median across iters.
    latencies = []
    for _ in range(measure_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        op_fn()
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    return float(np.median(latencies))


# Point-to-point needs a matching side on the peer rank: the engine's send is
# matched by its recv. Every other recorded call is a collective and is
# replayed by all ranks with the recorded arguments.
_P2P_PEER = {"send": "recv", "recv": "send"}


def _variable_labels(entry):
    return set(entry.get("variable_labels", []))


def _rebuild(value, workload, tensors, labels, device):
    """Recorded argument (comm_args JSON) -> live value at this sweep point."""
    from doolyprof.profiler.comm_args import is_tensor_ref
    if is_tensor_ref(value):
        return tensors[value["tensor"]]
    if isinstance(value, dict):
        if "int" in value:
            return workload if value.get("taint") in labels else value["int"]
        if "shape" in value:
            return torch.Size([_rebuild(d, workload, tensors, labels, device) for d in value["shape"]])
        if "dtype" in value:
            return getattr(torch, value["dtype"])
        if "device" in value:
            return device
        return {k: _rebuild(v, workload, tensors, labels, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_rebuild(v, workload, tensors, labels, device) for v in value]
    return value


def _swept_lists(meta, labels):
    """Traced values of list arguments made of swept integers (e.g. per-rank
    ``sizes`` of all_gatherv / reduce_scatterv)."""
    lists = []

    def visit(v):
        if isinstance(v, list) and v and all(isinstance(x, dict) and x.get("taint") in labels for x in v):
            lists.append([x["int"] for x in v])
        elif isinstance(v, list):
            for x in v:
                visit(x)
        elif isinstance(v, dict) and "int" not in v and "shape" not in v:
            for x in v.values():
                visit(x)

    for v in meta.get("args", {}).values():
        visit(v)
    return lists


def _swept_dim(traced, workload, lists):
    """A swept payload dim at this sweep point. Every swept integer argument
    becomes ``workload`` (so a per-rank list becomes [workload] * len); a dim
    that was traced as the sum of such a list keeps being its sum (e.g. the
    reduce_scatterv input), any other swept dim is ``workload``."""
    for traced_list in lists:
        if traced == sum(traced_list) and len(traced_list) > 1 and traced not in traced_list:
            return workload * len(traced_list)
    return workload


def _payload_tensors(entry, workload, dtype, device):
    """Inputs at the traced shapes with every swept (NUM_TOKS / NUM_REQS) dim
    rebuilt for ``workload`` (_swept_dim), in their traced dtypes, as compute ops
    are profiled (a DP+EP dispatch moves fp32 router weights next to the bf16
    hidden states); the profiled dtype only where no dtype was recorded."""
    traced = entry["meta"].get("dtypes", [])
    lists = _swept_lists(entry["meta"], _variable_labels(entry))
    tensors = []
    for i, inp in enumerate(entry["inputs"]):
        shape = [_swept_dim(size, workload, lists) if var else size
                 for size, var in zip(inp["shape"], inp["variable"])]
        tdt = getattr(torch, traced[i], None) if i < len(traced) else None
        if tdt is None:
            tdt = dtype
        if tdt.is_floating_point:
            tensors.append(torch.randn(shape, dtype=tdt, device=device))
        else:
            tensors.append(torch.zeros(shape, dtype=tdt, device=device))
    return tensors


def _payload_bytes(tensors, kwargs, dtype, recv_only=()):
    """Bytes the call sends: its tensor arguments, except those it only
    receives into (``recv_only``, e.g. reduce_scatterv_into_output's output)."""
    if tensors:
        return sum(t.numel() * t.element_size() for i, t in enumerate(tensors) if i not in recv_only)
    # Receiving side: the payload is the shape it will produce (e.g. recv(size, dtype)).
    shapes = [v for v in kwargs.values() if isinstance(v, torch.Size)]
    dtypes = [v for v in kwargs.values() if isinstance(v, torch.dtype)] or [dtype]
    return sum(int(np.prod(s)) * torch.empty((), dtype=dtypes[0]).element_size() for s in shapes)


def _build_call(entry, workload, comm, tp_rank, dtype, device):
    """(no-arg callable replaying the recorded call on the real group, payload bytes)."""
    import inspect
    op = entry["operation"]
    meta = entry["meta"]
    labels = _variable_labels(entry)
    tensors = _payload_tensors(entry, workload, dtype, device)
    kwargs = {k: _rebuild(v, workload, tensors, labels, device) for k, v in meta.get("args", {}).items()}
    size_bytes = _payload_bytes(tensors, kwargs, dtype, set(entry.get("recv_only") or ()))

    if op in _P2P_PEER:
        # Rank 0 sends the recorded payload to rank 1, which receives it, through
        # the engine's own send/recv (dst/src are local ranks).
        if tensors:
            payload = tensors[0]
            shape, pdtype = payload.shape, payload.dtype
        else:
            shape = next(v for v in kwargs.values() if isinstance(v, torch.Size))
            pdtype = next((v for v in kwargs.values() if isinstance(v, torch.dtype)), dtype)
            payload = torch.randn(shape, dtype=pdtype, device=device) if pdtype.is_floating_point \
                else torch.zeros(shape, dtype=pdtype, device=device)

        def _call():
            if tp_rank == 0:
                comm.send(payload, dst=1)
            elif tp_rank == 1:
                comm.recv(shape, pdtype, src=0)
        return _call, size_bytes

    if meta.get("api") == "torch.distributed":
        fn = getattr(torch.distributed, op)
        kwargs["group"] = comm.device_group
    else:
        fn = getattr(comm, op)
    params = inspect.signature(fn).parameters
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    call_kwargs = {k: v for k, v in kwargs.items() if accepts_any or k in params}
    return (lambda: fn(**call_kwargs)), size_bytes


def _profile_recorded(entry, sizes, comm, tp_rank, tp_world_size, dtype, device,
                      warmup_iters, measure_iters, comm_timing, tp_size, backend, topology, dtype_str,
                      group=None):
    op = entry["operation"]
    tag = f"{op} (dim={entry['dim_param']}, ndims={entry['num_dims']})"
    recorded_world = entry["meta"].get("world_size")
    if recorded_world and int(recorded_world) != tp_world_size:
        # Replayed on a group of another size it would be stored under the
        # wrong group size (the profiler keys the collective by its own group).
        print(f"[WARN] Skipping {tag}: recorded on a {recorded_world}-rank group, replay group has {tp_world_size}")
        return []
    if op in _P2P_PEER and tp_world_size < 2:
        print(f"[WARN] Skipping {tag}: point-to-point needs >= 2 ranks")
        return []
    if entry["meta"].get("api") != "torch.distributed" and not callable(getattr(comm, op, None)):
        print(f"[WARN] Skipping {tag}: the real group's device communicator has no '{op}'")
        return []
    model_dims = max(1, int(entry.get("model_config_dims") or 1))
    workloads = sorted({s // model_dims for s in sizes} - {0})
    if not workloads:
        return []
    # Calls that moved no data on the emulated group were already dropped at
    # trace time (COMM_NOOP), so every planned entry is a real transfer.
    timing = comm_timing
    capture_ctx = group.graph_capture if group is not None else None
    if timing == "graph":
        # Every rank tries the same capture; if any rank cannot capture this call,
        # all of them time it with the isolated method instead (one method per entry).
        probe, _ = _build_call(entry, workloads[0], comm, tp_rank, dtype, device)
        ok = 1
        try:
            _time_collective(probe, 1, 1, "graph", capture_ctx)
        except Exception as e:  # noqa: BLE001
            ok = 0
            print(f"[WARN] {tag}: CUDA-graph capture failed ({type(e).__name__}: {str(e)[:120]})")
        flag = torch.tensor([ok], dtype=torch.int32)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN,
                                     group=group.cpu_group if group is not None else None)
        if not flag.item():
            timing = "isolated"
            print(f"[WARN] {tag}: timed with the isolated method instead of graph")
    print(f"Profiling {tag} over {len(workloads)} sizes ({timing} timing)...")
    results = []
    for workload in workloads:
        call, size_bytes = _build_call(entry, workload, comm, tp_rank, dtype, device)
        latency = _time_collective(call, warmup_iters, measure_iters, timing, capture_ctx)
        results.append({
            'operation': op,
            'topology': topology,
            'backend': backend,
            'tp_size': tp_size,
            'size_bytes': size_bytes,
            'dtype': dtype_str,
            'dim_param': entry['dim_param'],
            'num_dims': entry['num_dims'],
            'latency_ms': latency,
        })
    print(f"✓ Completed {tag}")
    return results


# ============================================================================
# STANDALONE WORKER FUNCTION (called via collective_rpc)
# ============================================================================
def _run_profiling_on_worker_standalone(
    worker,
    config_dict: Dict[str, Any],  # CommProfilingConfig as dict
    plan_info: Dict[str, Dict[str, Any]],
    profile_flags: Dict[str, bool],
    tp_size: int,
    backend: str,
    topology: str,
) -> List[Dict]:
    """
    Standalone worker function for profiling (not a class method).
    This runs inside each worker process via collective_rpc.

    Every recorded collective variant in ``plan_info`` (CommProfiler.plan) is
    replayed through the same engine method it was recorded from, on this
    real TP group, with its recorded arguments and every swept dimension set
    to the per-rank workload of each sweep size.
    """
    from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
    tp_group = get_tp_group()
    tp_world_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    device = torch.device("cuda")
    comm = tp_group.device_communicator

    dtype = config_dict['dtype']
    warmup_iters = config_dict['warmup_iters']
    measure_iters = config_dict['measure_iters']
    max_size = config_dict['max_size']
    # Timing mode: "isolated" (default, per-call) or "amortized" (back-to-back).
    comm_timing = config_dict.get('comm_timing', 'isolated')
    print(f"[DEBUG Worker] comm_timing mode = {comm_timing}")
    print(f"[DEBUG Worker] TP rank={tp_rank}/{tp_world_size}, dtype={dtype}")

    # Generate sweep sizes
    COLLECTIVE_SIZE_SPACE = (
        list(range(1024, 512 * 1024 + 1, 4 * 1024))
        + list(range(512 * 1024, 8 * 1024 * 1024 + 1, 16 * 1024))
        + list(range(8 * 1024 * 1024, 64 * 1024 * 1024 + 1, 64 * 1024))
        + list(range(64 * 1024 * 1024 + 1, 512 * 1024 * 1024 + 1, 256 * 1024))
    )
    sizes = [s for s in COLLECTIVE_SIZE_SPACE if s <= max_size]

    results = []
    dtype_str = str(dtype).split('.')[-1]

    for key in sorted(plan_info):
        results.extend(_profile_recorded(
            plan_info[key], sizes, comm, tp_rank, tp_world_size, dtype, device,
            warmup_iters, measure_iters, comm_timing, tp_size, backend, topology, dtype_str,
            group=tp_group))

    # Profile send/recv (pipeline-parallel stage boundary), through vLLM's group
    # coordinator -- same path as all_reduce above, identical to what PP pays.
    if profile_flags.get('send_recv', False):
        if tp_world_size != 2:
            print(f"[WARN] Skipping send_recv: requires exactly 2 ranks, got {tp_world_size}")
        else:
            print("Profiling send_recv...")
            for size in sizes:
                try:
                    tensor = torch.randn(size, dtype=dtype, device=device)
                    size_bytes = tensor.numel() * DTYPE_TO_BYTES[dtype]

                    def _p2p():
                        # dst/src are LOCAL ranks within the 2-rank group
                        if tp_rank == 0:
                            tp_group.send(tensor, dst=1)
                        else:
                            tp_group.recv(tensor.size(), dtype, src=0)

                    latency = _time_collective(
                        _p2p, warmup_iters, measure_iters, comm_timing, tp_group.graph_capture,
                    )

                    results.append({
                        'operation': 'send_recv',
                        'topology': topology,
                        'backend': backend,
                        'tp_size': tp_size,
                        'size_bytes': size_bytes,
                        'dtype': dtype_str,
                        'dim_param': None,
                        'num_dims': 1,
                        'latency_ms': latency
                    })
                except Exception as e:
                    raise ValueError(f"Error profiling send_recv (size={size}): {e}")

            print(f"✓ Completed send_recv profiling")

    return results


@dataclass
class CommProfilingConfig:
    """Configuration for communication profiling sweep (following Vidur's approach)"""
    dtype: torch.dtype = torch.bfloat16
    warmup_iters: int = 10
    measure_iters: int = 10
    max_req_count: int = 256
    max_seq_len: int = 8192
    max_size: int = max_seq_len * 1024 * 1024
    # Timing mode for collectives: "isolated" (default, per-call synced median)
    # or "amortized" (measure_iters back-to-back between two events, single
    # sync, divided by iters). Additive experiment-only option.
    comm_timing: str = "isolated"
    
    def __post_init__(self):
        if self.dtype is None:
            self.dtype = torch.bfloat16

    def get_collective_sizes_to_profile(self) -> List[int]:
        COLLECTIVE_SIZE_SPACE = (
            list(range(1024, 512 * 1024 + 1, 4 * 1024))                                 # 1K to 512K, step 4K
            + list(range(512 * 1024, 8 * 1024 * 1024 + 1, 16 * 1024))                   # 512K to 8M, step 16K
            + list(range(8 * 1024 * 1024, 64 * 1024 * 1024 + 1, 64 * 1024))             # 8M to 64M, step 64K
            + list(range(64 * 1024 * 1024 + 1, 512 * 1024 * 1024 + 1, 256 * 1024))      # 64M to 512M, step 256K
        )
        sizes_to_profile = []
        for size in COLLECTIVE_SIZE_SPACE:
            if size <= self.max_size:
                sizes_to_profile.append(size)
            else:
                break
        return sizes_to_profile

class CommProfiler:
    def __init__(self, config: CommProfilingConfig, db_path):
        self.config = config
        self.db_path = db_path

    def make_db(self):
        # create table in db
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coll_op_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT NOT NULL,
                topology TEXT NOT NULL,
                backend TEXT NOT NULL,
                tp_size INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                dtype TEXT,
                dim_param INTEGER,
                num_dims INTEGER,
                latency_ms REAL NOT NULL,
                UNIQUE(operation, topology, backend, dtype, tp_size, size_bytes, dim_param, num_dims)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coll_ops (
                collective_id INTEGER PRIMARY KEY AUTOINCREMENT,
                signature_hash TEXT UNIQUE NOT NULL,
                operation_name TEXT NOT NULL,  
                topology TEXT NOT NULL,        
                tp_degree INTEGER NOT NULL,   
                num_dims INTEGER,
                dim_param INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_collectives (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                collective_id INTEGER NOT NULL,
                count INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (model_id) REFERENCES profiled_models(model_id),
                FOREIGN KEY (collective_id) REFERENCES coll_ops(collective_id),
                UNIQUE(model_id, collective_id)
               ) 
            """)
        
        conn.commit()
        conn.close()
        
        print(f"✓ Database initialized at {self.db_path}")

    def should_profile(self, tp_size, overwrite=False, topology=None):
        """Check if we should profile based on existing data and overwrite flag.

        ``topology`` is the label the measurements are stored under (group_topology(tp_size) by default)."""
        topology = topology or group_topology(tp_size)
        if overwrite:
            print("[COMM] Overwrite mode enabled - will re-profile all operations")
            return True

        if not hasattr(self, 'plan_info') or not self.plan_info:
            return False

        # Check every planned (operation, dim_param, num_dims) variant
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        needs_profiling = False
        for entry in self.plan_info.values():
            existing_count = cursor.execute("""
                SELECT COUNT(*) FROM coll_op_results
                WHERE operation = ? AND topology = ? AND tp_size = ?
                AND dim_param IS ? AND num_dims = ?
            """, (entry["operation"], topology, tp_size, entry["dim_param"], entry["num_dims"])).fetchone()[0]
            tag = f"{entry['operation']} (dim={entry['dim_param']}, ndims={entry['num_dims']})"
            if existing_count == 0:
                needs_profiling = True
                print(f"[COMM] {tag} needs profiling (no existing data)")
            else:
                print(f"[COMM] {tag} already profiled ({existing_count} measurements), skipping")

        conn.close()
        return needs_profiling

    def plan(self, collectives: List[Any]):
        """One entry per recorded collective variant, keyed like coll_ops
        (operation, dim_param, num_dims): the traced input shapes, which of
        their dimensions are swept, the product of the first input's fixed
        dims, and the recorded call (comm_args).

        A dim is swept when it carries a NUM_TOKS / NUM_REQS taint, or when it
        equals a per-rank size of a swept list argument or their sum: an EP
        dispatch runs inside the MoE custom op, where the tracer sees plain
        tensors, but its per-rank sizes keep the token taint. Every other dim
        is fixed, labelled MODEL_CONFIG or not, so the element sweep divides by
        all of them."""
        from doolyprof.profiler.comm_args import parse_comm_args, identity_int, swept_payload_dims
        from doolyprof.profiler.taint import DimensionType

        swept = {DimensionType.NUM_TOKENS, DimensionType.NUM_REQUESTS}
        self.plan_info: Dict[str, Dict[str, Any]] = {}
        for collective in collectives:
            tensors = collective.inputs
            if not tensors:
                continue
            meta = parse_comm_args(collective.hierarchy.anchor.args.get('comm_args'))
            shapes = [list(t.actual_shape) for t in tensors]
            dim_param = identity_int(meta)
            num_dims = len(shapes[0])
            key = f"{collective.operation_name}|{dim_param}|{num_dims}"
            if key in self.plan_info:
                continue
            labels = {d.value for d in swept}
            variable = [swept_payload_dims(meta, shape, t.dimensions, swept, labels)
                        for shape, t in zip(shapes, tensors)]
            model_dims = 1
            for size, var in zip(shapes[0], variable[0]):
                if not var:
                    model_dims *= size
            self.plan_info[key] = {
                'operation': collective.operation_name,
                'dim_param': dim_param,
                'num_dims': num_dims,
                'model_config_dims': model_dims,
                'inputs': [{'shape': shape, 'variable': var} for shape, var in zip(shapes, variable)],
                'variable_labels': sorted(d.value for d in swept),
                'meta': meta,
                # inputs the call only received into (the parser's COMM_RECV_ONLY):
                # replayed, but not part of the payload size
                'recv_only': list(collective.hierarchy.anchor.args.get('recv_only', [])),
            }
            print(f"[COMM] planned {key}: inputs={shapes}, model_config_dims={model_dims}")
        self.profile_send_recv_bool: bool = False  # enabled by the worker for PP profiling

    def profile(
        self,
        model: str,
        dtype: Literal["auto", "half", "float16", "bfloat16", "float", "float32"],
        config: CommProfilingConfig,
        tp_size: int,
        backend: str,
        topology: str
    ) -> List[Dict]:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        llm = LLM(
            model=model,
            dtype=dtype,  # Model dtype for initialization
            enforce_eager=True,
            load_format="dummy",  # Fast initialization without loading real weights
            # what is free on the TP GPUs now, at most 0.5 (utils/gpu_memory.py)
            gpu_memory_utilization=engine_memory_fraction(cap=0.5, devices=tp_size, who="comm profiler"),
            max_model_len=2048,  # Small model_len sufficient for comm profiling
            tensor_parallel_size=tp_size,  # REAL tensor parallelism
            language_model_only=True,  # the traced language model, no multimodal encoders
            skip_tokenizer_init=True,  # collectives only, no text
        )

        print(f"\n=== Communication Profiling Sweep ===")
        print(f"Model: {model}")
        print(f"TP Size: {tp_size}")
        print(f"Backend: {backend}")
        print(f"Topology: {topology}")
        print(f"Max size: {config.max_size} elements")

        # Generate sweep sizes for display
        sizes = config.get_collective_sizes_to_profile()
        print(f"Size range: {sizes[0]} to {sizes[-1]} elements")
        print(f"Example sizes: {sizes[:5]}... {sizes[-5:]}")
        print()

        # Convert config to dict for serialization across RPC boundary
        config_dict = {
            'dtype': config.dtype,
            'warmup_iters': config.warmup_iters,
            'measure_iters': config.measure_iters,
            'max_size': config.max_size,
            'comm_timing': getattr(config, 'comm_timing', 'isolated'),
        }

        # Prepare profile flags
        profile_flags = {
            'send_recv': getattr(self, 'profile_send_recv_bool', False),
        }

        # Call standalone function (not instance method) to avoid serialization issues
        results = llm.llm_engine.collective_rpc(
            method=_run_profiling_on_worker_standalone,
            timeout=None,
            args=(config_dict, self.plan_info, profile_flags, tp_size, backend, topology),
            kwargs=None
        )

        # collective_rpc returns results from all workers, we just need results from one worker
        # (all workers run the same profiling code and should return identical results)
        if results:
            self.results = results[0]  # Take results from first worker
            
        print(f"✓ Completed {len(self.results)} profiling measurements")

    def save_to_db(self, overwrite=True):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        unique_ops = set()

        for result in self.results:
            # Check if this configuration already exists
            existing = cursor.execute("""
                SELECT id FROM coll_op_results
                WHERE operation = ? AND topology = ? AND backend = ?
                AND tp_size = ? AND size_bytes = ? AND dtype = ?
                AND dim_param = ? AND num_dims = ?
            """, (
                result['operation'], result['topology'], result['backend'],
                result['tp_size'], result['size_bytes'], result['dtype'],
                result['dim_param'], result['num_dims']
            )).fetchone()

            if existing and overwrite:
                # Update existing result
                cursor.execute("""
                    UPDATE coll_op_results
                    SET latency_ms = ?
                    WHERE id = ?
                """, (result['latency_ms'], existing[0]))
            elif not existing:
                # Insert new result
                cursor.execute("""
                    INSERT INTO coll_op_results
                    (operation, topology, backend, tp_size, size_bytes, dtype, dim_param, num_dims, latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                result['operation'],
                result['topology'],
                result['backend'],
                result['tp_size'],
                result['size_bytes'],
                result['dtype'],
                result['dim_param'],
                result['num_dims'],
                result['latency_ms']
            ))
            
            # track unique collective operations for coll_ops table
            unique_ops.add((
                result['operation'],
                result['topology'], 
                result['tp_size'],
                result['dim_param'],
                result['num_dims']
            ))
        
        # insert unique collective operations into coll_ops table
        for op_name, topology, tp_degree, dim_param, num_dims in unique_ops:
            # Generate signature
            import hashlib
            sig_data = f"{op_name}|{topology}|{tp_degree}|{dim_param}|{num_dims}"
            sig_hash = hashlib.sha256(sig_data.encode()).hexdigest()
            
            cursor.execute("""
                INSERT OR IGNORE INTO coll_ops 
                (signature_hash, operation_name, topology, tp_degree, dim_param, num_dims)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (sig_hash, op_name, topology, tp_degree, dim_param, num_dims))
        
        conn.commit()
        conn.close()