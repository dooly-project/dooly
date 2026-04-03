"""Latency estimation for simulation steps."""

from typing import Dict, Tuple, Optional
import sqlite3
import os
import sys
from pathlib import Path

from ..request import BatchConfig


def is_verbose() -> bool:
    """Check if verbose mode is enabled."""
    try:
        from .. import cli
        return cli.VERBOSE
    except (ImportError, AttributeError):
        return False


class LatencyEstimator:
    """Handles latency prediction using regression models."""

    def __init__(
        self,
        model_name: str,
        backend: str,
        vllm_config,
        cache_dir: str,
        db_path: str,
        fallback_duration_ms: float = 10.0
    ):
        """Initialize latency estimator.

        Args:
            model_name: Name of the model being simulated
            backend: Backend type (e.g., "FLASHINFER")
            vllm_config: vLLM configuration object
            cache_dir: Directory containing cached regression models
            db_path: Path to profiling database
            fallback_duration_ms: Fallback step duration if models fail to load
        """
        self.model_name = model_name
        self.backend = backend
        self.vllm_config = vllm_config
        self.fallback_duration_ms = fallback_duration_ms

        # Initialize predictor and load operations
        self.latency_predictor = self._init_predictor(cache_dir, db_path)
        self.model_operations = self._load_model_operations(db_path)

    def _init_predictor(self, cache_dir: str, db_path: str):
        """Initialize latency predictor from cached models."""
        try:
            # Add simulator directory to path if not already there
            simulator_dir = Path(__file__).parent.parent
            if str(simulator_dir) not in sys.path:
                sys.path.insert(0, str(simulator_dir))

            from simulator.estimator import EstimatorConfig, SimplifiedExecutionTimeEstimator

            # Infer backend from vllm_config if available
            backend = self.backend
            if hasattr(self.vllm_config, 'attention_config'):
                backend_enum = getattr(self.vllm_config.attention_config, 'backend', None)
                if backend_enum:
                    backend = str(backend_enum).split('.')[-1].upper()

            config = EstimatorConfig(
                cache_dir=cache_dir,
                model_name=self.model_name,
                backend=backend,
                db_path=db_path,
                no_cache=False,  # Use cached models
            )

            print(f"Loading regression models from {cache_dir}...")
            estimator = SimplifiedExecutionTimeEstimator(config)
            estimator.train_all_models()  # Loads from cache

            predictor = estimator.get_predictor()
            print(f"Loaded {len(predictor.models)} operation models")
            return predictor

        except Exception as e:
            print(f"Warning: Failed to initialize latency predictor: {e}")
            print(f"Falling back to fixed step duration ({self.fallback_duration_ms}ms)")
            return None

    def _load_model_operations(self, db_path: str) -> Dict[Tuple[str, str], int]:
        """Load operation counts from model_operations table.

        Returns:
            Dict mapping (operation_name, signature_hash) -> count
        """
        if not os.path.exists(db_path):
            print(f"Warning: Database not found at {db_path}")
            return {}

        try:
            conn = sqlite3.connect(db_path)
            query = """
                SELECT s.operation_name, s.signature_hash, mo.count
                FROM model_operations mo
                JOIN signatures s ON mo.operation_id = s.operation_id
                JOIN profiled_models pm ON mo.model_id = pm.model_id
                WHERE pm.model_name = ? AND pm.backend = ?
            """

            cursor = conn.execute(query, (self.model_name, self.backend))
            operations = {}
            for row in cursor:
                op_name, sig_hash, count = row
                operations[(op_name, sig_hash)] = count

            conn.close()
            print(f"Loaded {len(operations)} operation signatures from database (backend={self.backend})")
            return operations

        except Exception as e:
            print(f"Warning: Failed to load model operations: {e}")
            return {}

    def estimate_step_latency(self, sched_output, step_id: int = -1) -> float:
        """Estimate step latency using regression models.

        Args:
            sched_output: SchedulerOutput from scheduler.schedule()
            step_id: Current step ID for debug output
        Returns:
            Estimated step latency in milliseconds
        """
        if self.latency_predictor is None or not self.model_operations:
            return self.fallback_duration_ms

        # Debug: Show available models on first call
        if is_verbose() and step_id == 0:
            print(f"[DEBUG] Total models loaded: {len(self.latency_predictor.models)}")
            attention_models = sorted([k for k in self.latency_predictor.models.keys() if 'Attention' in k])
            print(f"[DEBUG] Attention models available: {attention_models}")

        try:
            # Extract batch information
            batch = BatchConfig(sched_output)

            # If no tokens scheduled, return 0
            if batch.total_tokens == 0:
                return 0.0

            total_latency_ms = 0.0

            # Iterate through all operations for this model
            for (op_name, sig_hash), count in self.model_operations.items():
                sig_short = sig_hash[:8]
                model_key = f"{op_name}_{sig_short}"

                # For Attention operations, check for _prefill and _decode variants
                if op_name == "Attention":
                    prefill_key = f"{op_name}_prefill_{sig_short}"
                    decode_key = f"{op_name}_decode_{sig_short}"
                    has_model = (prefill_key in self.latency_predictor.models or
                                decode_key in self.latency_predictor.models)
                    if not has_model:
                        if is_verbose() and step_id < 3:
                            print(f"[DEBUG Step {step_id}] Skipping {op_name}_{sig_short} - no prefill/decode models found")
                        continue
                else:
                    # Standard check for non-Attention operations
                    if model_key not in self.latency_predictor.models:
                        if is_verbose() and step_id < 3:
                            print(f"[DEBUG Step {step_id}] Skipping {model_key} - no model found")
                        continue

                # Predict latency based on operation type
                if op_name == "Attention":
                    op_latency = self._estimate_attention_latency(batch, sig_short, op_name)
                    total_latency_ms += op_latency * count
                else:
                    # Non-attention operations (linear, etc)
                    try:
                        pred = self.latency_predictor.predict_simple(
                            model_key,
                            num_tokens=batch.total_tokens
                        )
                        total_latency_ms += pred * count
                    except Exception:
                        pass

            # Return latency in milliseconds
            return total_latency_ms

        except Exception as e:
            # Fallback to fixed duration on any error
            print(f"Warning: Latency estimation failed: {e}, using fixed duration")
            return self.fallback_duration_ms

    def _estimate_attention_latency(self, batch, sig_short: str, op_name: str) -> float:
        """Estimate attention operation latency for prefill and decode phases.

        Args:
            batch: BatchConfig containing prefill and decode request info
            sig_short: Short signature hash
            op_name: Operation name ("Attention")

        Returns:
            Estimated latency in milliseconds
        """
        op_latency = 0.0

        # Prefill phase - aggregate using L2 norm
        if batch.prefill_requests:
            # Extract kv_cache_sizes and prefill_chunk_sizes
            # Note: tuple format is (num_tokens, kv_cache_size)
            kv_cache_sizes = [kv_cache_size for num_tokens, kv_cache_size in batch.prefill_requests]
            prefill_chunk_sizes = [num_tokens for num_tokens, kv_cache_size in batch.prefill_requests]

            # Aggregate using sum for kv_cache and L2 norm for prefill_chunk_size
            agg_kv_cache_size = sum(kv_cache_sizes)
            agg_prefill_chunk_size = sum([x**2 for x in prefill_chunk_sizes]) ** 0.5  # L2 norm

            if is_verbose():
                print(f"      [Attention Prefill] num_reqs={len(batch.prefill_requests)}, "
                        f"agg_kv_cache={agg_kv_cache_size}, "
                        f"agg_prefill_chunk={agg_prefill_chunk_size:.1f} (L2 norm of {prefill_chunk_sizes})")

            # Use Attention_prefill_{sig_short} model key
            prefill_model_key = f"{op_name}_prefill_{sig_short}"

            try:
                # Make ONE prediction for the entire prefill batch
                pred = self.latency_predictor.predict_simple(
                    prefill_model_key,
                    kv_cache_size=agg_kv_cache_size,
                    prefill_chunk_size=int(round(agg_prefill_chunk_size))
                )
                op_latency += pred

                if is_verbose():
                    print(f"      [Attention Prefill] PREDICTION: model={prefill_model_key}, "
                        f"kv_cache={agg_kv_cache_size}, chunk={int(round(agg_prefill_chunk_size))}, "
                        f"predicted_latency={pred:.3f}ms")

            except Exception as e:
                    raise ValueError(f"    WARNING: Prefill prediction failed - agg_kv={agg_kv_cache_size}, "
                          f"agg_chunk={int(round(agg_prefill_chunk_size))}, error: {e}")

        # Decode phase - use decode batch size only
        if batch.decode_requests:
            # Decode batch size = number of decode requests ONLY
            decode_batch_size = len(batch.decode_requests)
            # Average KV cache size across decode requests
            avg_kv_cache = sum(batch.decode_requests) / decode_batch_size

            if is_verbose():
                print(f"      [Attention Decode] batch={decode_batch_size}, avg_kv_cache={int(avg_kv_cache)}, "
                        f"decode_kv_values={batch.decode_requests[:5]}")

            # Use Attention_decode_{sig_short} model key
            decode_model_key = f"{op_name}_decode_{sig_short}"

            try:
                pred = self.latency_predictor.predict_simple(
                    decode_model_key,
                    batch_size=decode_batch_size,
                    kv_cache_size=int(avg_kv_cache)
                )
                op_latency += pred

                if is_verbose():
                    print(f"      [Attention Decode] PREDICTION: model={decode_model_key}, "
                        f"batch_size={decode_batch_size}, kv_cache={int(avg_kv_cache)}, "
                        f"predicted_latency={pred:.3f}ms")
            except Exception as e:
                    raise ValueError(f"    WARNING: Decode prediction failed - batch={decode_batch_size}, "
                          f"avg_kv={int(avg_kv_cache)}, error: {e}")
        else:
            print(f"      [Attention] NO DECODE REQUESTS - prefill_reqs={len(batch.prefill_requests)}, "
                f"decode_reqs={len(batch.decode_requests)}")

        return op_latency
