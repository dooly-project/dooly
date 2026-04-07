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
        fallback_duration_ms: float = 10.0,
        use_csv_cache: bool = True,
    ):
        """Initialize latency estimator.

        Args:
            model_name: Name of the model being simulated
            backend: Backend type (e.g., "FLASHINFER")
            vllm_config: vLLM configuration object
            cache_dir: Directory containing cached prediction CSV files
            db_path: Path to profiling database
            fallback_duration_ms: Fallback step duration if models fail to load
            use_csv_cache: If True, try to load from CSV cache first (faster)
        """
        self.model_name = model_name
        self.backend = backend
        self.vllm_config = vllm_config
        self.fallback_duration_ms = fallback_duration_ms
        self.use_csv_cache = use_csv_cache

        # Initialize predictor and load operations
        self.latency_predictor = self._init_predictor(cache_dir, db_path)
        self.model_operations = self._load_model_operations(db_path)

    def _init_predictor(self, cache_dir: str, db_path: str):
        """Initialize latency predictor from cached models or CSV files."""
        try:
            # Add simulator directory to path if not already there
            simulator_dir = Path(__file__).parent.parent
            if str(simulator_dir) not in sys.path:
                sys.path.insert(0, str(simulator_dir))

            # Try to load from CSV cache first (much faster)
            if self.use_csv_cache:
                try:
                    from simulator.core.prediction_cache import load_all_predictions
                    from simulator.operation_time_predictor import OperationTimePredictor

                    print(f"Attempting to load CSV prediction cache from {cache_dir}...")
                    cache = load_all_predictions(cache_dir)

                    if len(cache.predictions) > 0:
                        # Create a simple predictor from CSV cache
                        # We need to infer operation_is_context from operation names
                        operation_is_context = {}
                        for op_name in cache.predictions.keys():
                            # Attention operations have _prefill or _decode suffix
                            base_name = op_name.replace('_prefill', '').replace('_decode', '')
                            # Remove signature hash
                            import re
                            base_name = re.sub(r'_[0-9a-f]{8}$', '', base_name)
                            operation_is_context[base_name] = 'Attention' in base_name or 'attention' in base_name.lower()

                        predictor = OperationTimePredictor(
                            models={},  # No sklearn models needed for CSV cache
                            predictions=cache.predictions,
                            operation_is_context=operation_is_context,
                            kv_cache_granularity=64,
                        )
                        print(f"Successfully loaded {len(cache.predictions)} operations from CSV cache")
                        return predictor
                    else:
                        print("CSV cache is empty, falling back to model loading...")
                except Exception as e:
                    print(f"CSV cache loading failed: {e}")
                    print("Falling back to model loading...")

            # Fall back to loading regression models
            from simulator.estimator import EstimatorConfig, SimplifiedExecutionTimeEstimator

            # Infer backend from vllm_config if available
            backend = self.backend
            if hasattr(self.vllm_config, 'attention_config'):
                backend_enum = getattr(self.vllm_config.attention_config, 'backend', None)
                if backend_enum:
                    backend = str(backend_enum).split('.')[-1].upper()

            # Try to extract model name from vllm_config if needed
            model_name = self.model_name
            if hasattr(self.vllm_config, 'model_config') and hasattr(self.vllm_config.model_config, 'model'):
                config_model = self.vllm_config.model_config.model
                if config_model:
                    model_name = config_model
                    print(f"Using model name from vllm_config: {model_name}")

            config = EstimatorConfig(
                cache_dir=cache_dir,
                model_name=model_name,
                backend=backend,
                db_path=db_path,
                no_cache=False,  # Use cached models
            )

            print(f"Loading regression models...")
            estimator = SimplifiedExecutionTimeEstimator(config)
            estimator.train_all_models()  # Loads from cache

            predictor = estimator.get_predictor()
            print(f"Loaded {len(predictor.models)} operation models")
            return predictor

        except Exception as e:
            print(f"Warning: Failed to initialize latency predictor: {e}")
            print(f"Falling back to fixed step duration ({self.fallback_duration_ms}ms)")
            import traceback
            traceback.print_exc()
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
            print(f"\n[Step {step_id}] Estimating latency for batch: total_tokens={batch.total_tokens}, {batch.batch_size} requests (prefill={len(batch.prefill_requests)}, decode={len(batch.decode_requests)})")

            # DETECT MIXED BATCHES (prefill + decode together)
            if len(batch.prefill_requests) > 0 and len(batch.decode_requests) > 0:
                print(f"\n{'='*80}")
                print(f"MIXED BATCH DETECTED at Step {step_id}!")
                print(f"  Prefill requests: {len(batch.prefill_requests)}")
                print(f"  Decode requests: {len(batch.decode_requests)}")
                print(f"  This requires special handling for latency prediction!")
                print(f"{'='*80}\n")

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
                    # Check both models (for sklearn) and predictions (for CSV cache)
                    has_model = (prefill_key in self.latency_predictor.models or
                                decode_key in self.latency_predictor.models or
                                prefill_key in self.latency_predictor.predictions or
                                decode_key in self.latency_predictor.predictions)
                    if not has_model:
                        if is_verbose() and step_id < 3:
                            print(f"[DEBUG Step {step_id}] Skipping {op_name}_{sig_short} - no prefill/decode models found")
                        continue
                else:
                    # Standard check for non-Attention operations
                    # Check both models (for sklearn) and predictions (for CSV cache)
                    if model_key not in self.latency_predictor.models and model_key not in self.latency_predictor.predictions:
                        if is_verbose() and step_id < 3:
                            print(f"[DEBUG Step {step_id}] Skipping {model_key} - no model found")
                        continue

                # Predict latency based on operation type
                if op_name == "Attention":
                    op_latency = self._estimate_attention_latency(batch, sig_short, op_name, step_id)
                    contribution = op_latency * count
                    total_latency_ms += contribution
                    # print(f"  [{op_name}_{sig_short}] op_latency={op_latency:.6f} ms × count={count} = {contribution:.6f} ms")
                else:
                    # Non-attention operations (linear, etc)
                    try:
                        pred = self.latency_predictor.predict_simple(
                            model_key,
                            num_tokens=batch.total_tokens
                        )
                        contribution = pred * count
                        total_latency_ms += contribution

                        # DEBUG: Show predictions for decode steps
                        if len(batch.decode_requests) > 0 and step_id >= 0:
                            print(f"  [{model_key}] pred={pred:.6f} ms × count={count} = {contribution:.6f} ms (num_tokens={batch.total_tokens})")
                    except Exception as e:
                        print(f"  [{model_key}] FAILED: {e}")
                        pass

            # Return latency in milliseconds
            # DEBUG: Show total for decode steps
            if len(batch.decode_requests) > 0 and step_id >= 0:
                print(f"\n[Step {step_id}] TOTAL PREDICTED LATENCY: {total_latency_ms:.6f} ms")
                print(f"  (This is for {len(batch.decode_requests)} decode request(s) generating {batch.total_tokens} token(s))")

            return total_latency_ms

        except Exception as e:
            # Fallback to fixed duration on any error
            print(f"Warning: Latency estimation failed: {e}, using fixed duration")
            return self.fallback_duration_ms

    def _estimate_attention_latency(self, batch, sig_short: str, op_name: str, step_id: int = -1) -> float:
        """Estimate attention operation latency for prefill and decode phases.

        Args:
            batch: BatchConfig containing prefill and decode request info
            sig_short: Short signature hash
            op_name: Operation name ("Attention")
            step_id: Current step ID for debug output

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

            # if is_verbose():
                # print(f"      [Attention Prefill] num_reqs={len(batch.prefill_requests)}, "
                #         f"agg_kv_cache={agg_kv_cache_size}, "
                #         f"agg_prefill_chunk={agg_prefill_chunk_size:.1f} (L2 norm of {prefill_chunk_sizes})")

            # Use Attention_prefill_{sig_short} model key
            prefill_model_key = f"{op_name}_prefill_{sig_short}"

            try:
                # Make ONE prediction for the entire prefill batch
                pred = self.latency_predictor.predict_simple(
                    prefill_model_key,
                    kv_cache_size=agg_kv_cache_size,
                    prefill_chunk_size=int(round(agg_prefill_chunk_size))
                )

                # Apply batching overhead (Vidur's approach for GQA models)
                # Llama-3.1-8B uses GQA: num_q_heads=32, num_kv_heads=8
                ATTENTION_PREFILL_BATCHING_OVERHEAD = 0.1
                num_prefill_reqs = len(batch.prefill_requests)
                overhead_multiplier = 1.0 + ATTENTION_PREFILL_BATCHING_OVERHEAD * int(num_prefill_reqs > 1)

                pred_with_overhead = pred * overhead_multiplier
                op_latency += pred_with_overhead

                if num_prefill_reqs > 1:
                    print(f"  [Attention Prefill] base={pred:.6f}ms, overhead={ATTENTION_PREFILL_BATCHING_OVERHEAD*100:.0f}%, final={pred_with_overhead:.6f}ms ({num_prefill_reqs} reqs)")

            except Exception as e:
                    raise ValueError(f"    WARNING: Prefill prediction failed - agg_kv={agg_kv_cache_size}, "
                          f"agg_chunk={int(round(agg_prefill_chunk_size))}, error: {e}")

        # Decode phase - use decode batch size only
        if batch.decode_requests:
            # Decode batch size = number of decode requests ONLY
            decode_batch_size = len(batch.decode_requests)
            # Average KV cache size across decode requests
            avg_kv_cache = sum(batch.decode_requests) / decode_batch_size

            # DEBUGGING: Check if batch_size exceeds profiled range
            MAX_PROFILED_BATCH_SIZE = 16
            MAX_PROFILED_KV_CACHE = 16128

            print(f"\n[DECODE DEBUG] Step {step_id}:")
            print(f"  decode_batch_size: {decode_batch_size}")
            print(f"  decode_kv_values (first 5): {batch.decode_requests[:5]}")
            print(f"  avg_kv_cache: {int(avg_kv_cache)}")

            if decode_batch_size > MAX_PROFILED_BATCH_SIZE:
                print(f"\n{'='*80}")
                print(f"ERROR: Decode batch_size={decode_batch_size} EXCEEDS profiled range!")
                print(f"  Max profiled batch_size: {MAX_PROFILED_BATCH_SIZE}")
                print(f"  Current batch_size: {decode_batch_size}")
                print(f"  This will cause EXTRAPOLATION errors in regression model!")
                print(f"{'='*80}\n")
                import sys
                sys.exit(1)

            if avg_kv_cache > MAX_PROFILED_KV_CACHE:
                print(f"\n{'='*80}")
                print(f"WARNING: avg_kv_cache={int(avg_kv_cache)} exceeds profiled range!")
                print(f"  Max profiled kv_cache: {MAX_PROFILED_KV_CACHE}")
                print(f"  Current avg_kv_cache: {int(avg_kv_cache)}")
                print(f"{'='*80}\n")

            # Use Attention_decode_{sig_short} model key
            decode_model_key = f"{op_name}_decode_{sig_short}"

            try:
                pred = self.latency_predictor.predict_simple(
                    decode_model_key,
                    batch_size=decode_batch_size,
                    kv_cache_size=int(avg_kv_cache)
                )

                # Apply batching overhead (Vidur's approach for GQA models)
                ATTENTION_DECODE_BATCHING_OVERHEAD = 0.1
                overhead_multiplier = 1.0 + ATTENTION_DECODE_BATCHING_OVERHEAD * int(decode_batch_size > 1)

                pred_with_overhead = pred * overhead_multiplier
                op_latency += pred_with_overhead

                print(f"  Base prediction: {pred:.6f} ms (for ONE layer)")
                if decode_batch_size > 1:
                    print(f"  Batching overhead: {ATTENTION_DECODE_BATCHING_OVERHEAD*100:.0f}% (batch_size={decode_batch_size})")
                print(f"  Final prediction: {pred_with_overhead:.6f} ms (for ONE layer)")
                print(f"  Total for 32 layers: {pred_with_overhead * 32:.6f} ms")
            except Exception as e:
                    raise ValueError(f"    WARNING: Decode prediction failed - batch={decode_batch_size}, "
                          f"avg_kv={int(avg_kv_cache)}, error: {e}")
        else:
            print(f"      [Attention] NO DECODE REQUESTS - prefill_reqs={len(batch.prefill_requests)}, "
                f"decode_reqs={len(batch.decode_requests)}")

        return op_latency
