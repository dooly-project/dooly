"""
Simplified Execution Time Estimator using is_context flag.
Uses is_context to determine whether to use attention features or just sequence length.
"""

import hashlib
import os
import pickle
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


@dataclass
class EstimatorConfig:
    """Configuration for the execution time estimator."""

    # Database paths
    db_path: str = "/home/jkim38/dooly/profiler/database/results.db"
    cache_dir: str = "/home/jkim38/dooly/simulator/cache"

    # Model/Backend filtering
    model_name: Optional[str] = None  # e.g., "meta-llama/Llama-3.1-8B"
    backend: Optional[str] = None  # e.g., "FLASHINFER" or "FLASH_ATTN"

    # Model training parameters (matching Vidur)
    k_fold_cv_splits: int = 10  # Vidur default: 10
    num_training_job_threads: int = 4
    no_cache: bool = False

    # Random Forest parameters for grid search (matching Vidur)
    num_estimators: List[int] = None
    max_depth: List[int] = None
    min_samples_split: List[int] = None

    # Prediction parameters
    prediction_max_tokens: int = 16384
    prediction_max_batch_size: int = 16
    prediction_max_prefill_chunk_size: int = 10240  # Max prefill chunk size for predictions
    kv_cache_granularity: int = 64  # Granularity for KV cache size predictions

    def __post_init__(self):
        """Set default values for mutable parameters (matching Vidur)."""
        if self.num_estimators is None:
            self.num_estimators = [250, 500, 750]  # Vidur: [250, 500, 750]
        if self.max_depth is None:
            self.max_depth = [8, 16, 32]  # Vidur: [8, 16, 32]
        if self.min_samples_split is None:
            self.min_samples_split = [2, 5, 10]  # Vidur: [2, 5, 10]


class SimplifiedExecutionTimeEstimator:
    """
    Simplified estimator using is_context flag for feature selection.

    Feature selection logic:
    - is_context=False: Use only num_tokens (for MLP, linear, norm, etc.)
    - is_context=True: Use batch_size, kv_cache_size, prefill_chunk_size (for attention)
    """

    def __init__(self, config: EstimatorConfig = None):
        """Initialize the estimator with configuration."""
        self.config = config or EstimatorConfig()
        os.makedirs(self.config.cache_dir, exist_ok=True)

        # Will store trained models per operation signature
        self.models: Dict[str, BaseEstimator] = {}

        # Will store predictions for quick lookup
        self.predictions: Dict[str, Dict[Tuple, float]] = {}

        # Map operation names to is_context flag
        self.operation_is_context: Dict[str, bool] = {}

        # Load and prepare data
        self._load_data()

    def _load_data(self):
        """Load data from the results database."""
        conn = sqlite3.connect(self.config.db_path)

        # Build query with optional filtering
        # Note: Explicitly list r columns to avoid duplicate signature_hash
        query = """
            SELECT
                r.id, r.mean_latency_ms, r.median_latency_ms, r.max_latency_ms,
                r.min_latency_ms, r.std_latency_ms, r.num_tokens, r.num_requests,
                r.is_context, r.is_prefill, r.kv_cache_size, r.prefill_chunk_size,
                r.num_tensor_parallel_workers, r.input_shapes, r.created_at,
                s.operation_name, s.signature_hash,
                pm.model_name, pm.backend
            FROM results r
            JOIN signatures s ON r.signature_hash = s.signature_hash
            JOIN model_operations mo ON s.operation_id = mo.operation_id
            JOIN profiled_models pm ON mo.model_id = pm.model_id
        """

        where_conditions = []
        params = []

        if self.config.model_name:
            where_conditions.append("pm.model_name = ?")
            params.append(self.config.model_name)

        if self.config.backend:
            where_conditions.append("pm.backend = ?")
            params.append(self.config.backend)

        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)

        # Load results data with filtering
        self.results_df = pd.read_sql_query(query, conn, params=params)

        conn.close()

        # Print filtering information
        filter_info = []
        if self.config.model_name:
            filter_info.append(f"model={self.config.model_name}")
        if self.config.backend:
            filter_info.append(f"backend={self.config.backend}")

        filter_str = f" (filtered by {', '.join(filter_info)})" if filter_info else ""
        print(f"Loaded {len(self.results_df)} results{filter_str}")

        # Check if new columns exist, otherwise parse from workload_params
        self._ensure_columns()

        # Group by operation AND signature to train separate models per signature
        # This is important for operations like aten::linear which have multiple
        # signatures with different matrix dimensions
        self.operation_groups = self.results_df.groupby(['operation_name', 'signature_hash'])

        # Build operation_is_context mapping from the data
        self._build_operation_is_context_mapping()

    def _ensure_columns(self):
        """Ensure required columns exist, parsing from workload_params if needed."""

        # Check if new column structure exists
        if 'is_context' not in self.results_df.columns:
            print("Note: is_context column not found. Parsing from workload_params...")

            # Parse workload_params to create columns
            import json

            for idx, row in self.results_df.iterrows():
                if pd.notna(row.get('workload_params')):
                    try:
                        params = json.loads(row['workload_params'])

                        # Set is_context based on operation name (fallback method)
                        op_lower = row['operation_name'].lower() if 'operation_name' in row else ''
                        self.results_df.loc[idx, 'is_context'] = 'attention' in op_lower or 'attn' in op_lower

                        # Extract other parameters
                        self.results_df.loc[idx, 'num_tokens'] = params.get('num_tokens', 0)
                        self.results_df.loc[idx, 'batch_size'] = params.get('batch_size', params.get('num_requests', 1))
                        self.results_df.loc[idx, 'is_prefill'] = params.get('is_prefill', False)
                        self.results_df.loc[idx, 'kv_cache_size'] = params.get('kv_cache_size', 0)
                        self.results_df.loc[idx, 'prefill_chunk_size'] = params.get('prefill_chunk_size', 0)

                    except Exception as e:
                        print(f"Warning: Could not parse workload_params at index {idx}: {e}")

    def _build_operation_is_context_mapping(self):
        """Build a mapping of operation_name -> is_context flag."""
        if 'is_context' not in self.results_df.columns:
            print("Warning: is_context column not found, mapping will be empty")
            return

        # For each operation, determine if it's a context operation
        # Note: operation_groups now returns (operation_name, signature_hash), group_df
        for (op_name, sig_hash), group_df in self.operation_groups:
            # Take the first value (should be consistent across the operation)
            if len(group_df) > 0 and 'is_context' in group_df.columns:
                is_context_value = group_df['is_context'].iloc[0]
                # Handle different representations of boolean
                if pd.notna(is_context_value):
                    self.operation_is_context[op_name] = bool(is_context_value)
                else:
                    self.operation_is_context[op_name] = False
            else:
                self.operation_is_context[op_name] = False

        print(f"Built operation_is_context mapping for {len(self.operation_is_context)} unique operations")
        # Show which operations are context operations
        context_ops = [op for op, is_ctx in self.operation_is_context.items() if is_ctx]
        if context_ops:
            print(f"  Context operations: {context_ops}")

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Determine feature columns based on is_context flag.

        Simple rule:
        - is_context=False: Use only num_tokens
        - is_context=True: Use attention-specific features
        """

        if 'is_context' not in df.columns:
            # Fallback: assume non-context if column doesn't exist
            return ['num_tokens'] if 'num_tokens' in df.columns else []

        # Get the is_context value (should be same for all rows in this operation)
        is_context = df['is_context'].iloc[0] if len(df) > 0 else False

        if is_context:
            # Context-dependent operation (attention)
            # Check if prefill or decode
            is_prefill = df['is_prefill'].iloc[0] if 'is_prefill' in df.columns and len(df) > 0 else None

            if is_prefill == 1 or is_prefill == 1.0 or is_prefill is True:
                # Prefill attention: use kv_cache_size and prefill_chunk_size^2
                feature_cols = []
                if 'kv_cache_size' in df.columns:
                    feature_cols.append('kv_cache_size')
                if 'prefill_chunk_size' in df.columns:
                    df['prefill_chunk_size_squared'] = df['prefill_chunk_size'] ** 2
                    feature_cols.append('prefill_chunk_size_squared')
                return feature_cols if feature_cols else ['num_tokens']

            elif is_prefill == 0 or is_prefill == 0.0 or is_prefill is False:
                # Decode attention: use batch_size and kv_cache_size
                feature_cols = []

                # Use batch_size if available, otherwise num_requests as proxy
                if 'batch_size' in df.columns:
                    feature_cols.append('batch_size')
                elif 'num_requests' in df.columns:
                    # Use num_requests as batch_size for decode
                    df['batch_size'] = df['num_requests']
                    feature_cols.append('batch_size')

                if 'kv_cache_size' in df.columns:
                    feature_cols.append('kv_cache_size')

                return feature_cols if feature_cols else ['num_tokens']

            else:
                # Mixed or unknown: use all available context features
                feature_cols = []
                if 'batch_size' in df.columns:
                    feature_cols.append('batch_size')
                if 'num_tokens' in df.columns:
                    feature_cols.append('num_tokens')
                return feature_cols
        else:
            # Non-context operation (MLP, linear, norm, etc.)
            # Use only sequence length
            return ['num_tokens'] if 'num_tokens' in df.columns else []

    def train_all_models(self):
        """Train models for all operation signatures in the database."""

        print("\n=== Training or Loading Models for All Operation Signatures ===")

        # Note: operation_groups now returns (operation_name, signature_hash), group_df
        for (operation_name, sig_hash), group_df in self.operation_groups:
            # Create a unique model name that includes signature hash (first 8 chars)
            sig_short = sig_hash[:8]
            model_key = f"{operation_name}_{sig_short}"

            print(f"\nProcessing: {operation_name} (sig: {sig_short}, {len(group_df)} samples)")

            # Check if this is a context operation
            is_context = group_df['is_context'].iloc[0] if 'is_context' in group_df else False

            # Special handling for context operations - split prefill and decode
            if is_context and 'is_prefill' in group_df.columns:
                # Get unique values of is_prefill
                prefill_values = group_df['is_prefill'].unique()

                if len(prefill_values) > 1:
                    # We have both prefill and decode data
                    prefill_df = group_df[group_df['is_prefill'] == 1.0].copy()
                    decode_df = group_df[group_df['is_prefill'] == 0.0].copy()

                    # Train separate models with signature hash included
                    if len(prefill_df) >= 5:
                        prefill_model_name = f"{operation_name}_prefill_{sig_short}"
                        self._train_operation_model(prefill_model_name, prefill_df)
                        # Mark as context operation
                        self.operation_is_context[prefill_model_name] = True

                    if len(decode_df) >= 5:
                        decode_model_name = f"{operation_name}_decode_{sig_short}"
                        self._train_operation_model(decode_model_name, decode_df)
                        # Mark as context operation
                        self.operation_is_context[decode_model_name] = True

                    continue  # Skip the unified model

            # Train unified model for this operation signature
            self._train_operation_model(model_key, group_df)

        print(f"\n=== Training Complete: {len(self.models)} models trained ===")

    def _get_model_hash(self, model_name: str, df: pd.DataFrame) -> str:
        """
        Generate a hash for model caching based on config and data.

        Args:
            model_name: Name of the operation/model
            df: Training dataframe

        Returns:
            8-character hash string
        """
        # Create a config string with relevant parameters
        config_dict = {
            'model_name': self.config.model_name,
            'backend': self.config.backend,
            'num_estimators': self.config.num_estimators,
            'max_depth': self.config.max_depth,
            'min_samples_split': self.config.min_samples_split,
            'k_fold_cv_splits': self.config.k_fold_cv_splits,
        }
        config_str = str(config_dict)

        # Hash the data
        df_hash_str = hashlib.md5(df.to_json().encode("utf-8")).hexdigest()
        combined_str = f"{config_str}_{model_name}_{df_hash_str}"

        return hashlib.md5(combined_str.encode("utf-8")).hexdigest()[:8]

    def _load_model_from_cache(self, model_name: str, model_hash: str) -> Optional[BaseEstimator]:
        """
        Load a trained model from cache.

        Args:
            model_name: Name of the operation/model
            model_hash: Hash of the model configuration and data

        Returns:
            Loaded model or None if not found or caching disabled
        """
        if self.config.no_cache:
            return None

        cache_file = os.path.join(self.config.cache_dir, f"{model_name}_{model_hash}.pkl")

        if not os.path.exists(cache_file):
            return None

        try:
            with open(cache_file, "rb") as f:
                model = pickle.load(f)
            print(f"    Loaded {model_name} from cache")
            return model
        except Exception as e:
            print(f"    Warning: Could not load {model_name} from cache: {e}")
            return None

    def _store_model_in_cache(self, model_name: str, model_hash: str, model: BaseEstimator) -> None:
        """
        Store a trained model in cache.

        Args:
            model_name: Name of the operation/model
            model_hash: Hash of the model configuration and data
            model: Trained model to cache
        """
        if self.config.no_cache:
            return

        cache_file = os.path.join(self.config.cache_dir, f"{model_name}_{model_hash}.pkl")

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"    Warning: Could not cache {model_name}: {e}")

    def _train_operation_model(self, operation_name: str, df: pd.DataFrame):
        """Train a model for a specific operation."""

        # Get features based on is_context flag
        feature_cols = self._get_feature_columns(df)
        target_col = 'median_latency_ms'

        if not feature_cols:
            print(f"  Skipping {operation_name}: no valid features found")
            return

        # Prepare training data
        available_cols = [c for c in feature_cols + [target_col] if c in df.columns]
        train_df = df[available_cols].copy()
        train_df = train_df.dropna()

        # Update feature_cols to only include available columns
        feature_cols = [c for c in feature_cols if c in train_df.columns and c != target_col]

        if len(train_df) < 5:
            print(f"  Skipping {operation_name}: insufficient data ({len(train_df)} samples)")
            return

        # Generate hash for caching
        model_hash = self._get_model_hash(operation_name, train_df)

        # Try to load from cache
        cached_model = self._load_model_from_cache(operation_name, model_hash)
        if cached_model is not None:
            self.models[operation_name] = cached_model
            # Still generate predictions for the cached model
            self._generate_predictions(operation_name, cached_model, feature_cols,
                                      is_context='is_context' in df and df['is_context'].iloc[0])
            return

        print(f"  Training {operation_name} with features: {feature_cols}")

        try:
            model = self._train_model(
                operation_name=operation_name,
                df=train_df,
                feature_cols=feature_cols,
                target_col=target_col
            )
            self.models[operation_name] = model

            # Store in cache
            self._store_model_in_cache(operation_name, model_hash, model)

            # Generate predictions for common input ranges
            self._generate_predictions(operation_name, model, feature_cols,
                                      is_context='is_context' in df and df['is_context'].iloc[0])

        except Exception as e:
            print(f"  Error training model for {operation_name}: {e}")
            import traceback
            traceback.print_exc()

    def _train_model(
        self,
        operation_name: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str
    ) -> BaseEstimator:
        """Train a Random Forest model for specific operation and features."""

        # Prepare model and grid search
        model = RandomForestRegressor(random_state=42)

        grid_params = {
            'n_estimators': self.config.num_estimators,
            'max_depth': self.config.max_depth,
            'min_samples_split': self.config.min_samples_split,
        }

        scorer = make_scorer(
            self.mean_absolute_percentage_error,
            greater_is_better=False
        )

        cv_splits = min(self.config.k_fold_cv_splits, len(df))
        if cv_splits < 2:
            cv_splits = 2

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid_params,
            scoring=scorer,
            cv=cv_splits,
            n_jobs=self.config.num_training_job_threads,
            verbose=0
        )

        # Train
        X = df[feature_cols]
        y = df[target_col]

        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        score = -grid_search.best_score_

        print(f"    Best params: {grid_search.best_params_}")
        print(f"    MAPE: {score:.2f}%")

        return best_model

    @staticmethod
    def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
        """Calculate MAPE for model evaluation."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        # Handle zero values
        zero_mask = y_true == 0
        non_zero_mask = ~zero_mask

        error = np.zeros_like(y_true, dtype=float)
        error[non_zero_mask] = np.abs(
            (y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]
        ) * 100

        error[zero_mask] = np.where(y_pred[zero_mask] == 0, 0, 100)

        return np.mean(error)

    def _generate_predictions(
        self,
        operation_name: str,
        model: BaseEstimator,
        feature_cols: List[str],
        is_context: bool = False
    ):
        """
        Generate predictions for common input ranges.

        Matches Vidur's approach: generate predictions for EVERY value in range
        (not with gaps).
        """

        if not is_context:
            # Non-context: generate predictions for EVERY num_tokens (like Vidur)
            if feature_cols == ['num_tokens']:
                num_tokens_range = np.arange(1, self.config.prediction_max_tokens + 1)
                X = pd.DataFrame({'num_tokens': num_tokens_range})

                predictions = model.predict(X)
                pred_dict = {(tokens,): pred for tokens, pred in zip(num_tokens_range, predictions)}
                self.predictions[operation_name] = pred_dict
                print(f"    Generated {len(pred_dict)} predictions")

        else:
            # Context operation: generate appropriate prediction grids
            if 'prefill_chunk_size_squared' in feature_cols:
                # Prefill predictions (like Vidur)
                # kv_cache: with granularity, prefill_chunk: EVERY value from 1 to max
                kv_cache_range = np.arange(0, self.config.prediction_max_tokens + 1,
                                          self.config.kv_cache_granularity)
                prefill_chunk_range = np.arange(1, self.config.prediction_max_prefill_chunk_size + 1)

                kv_cache, prefill_chunk = np.meshgrid(kv_cache_range, prefill_chunk_range)
                X = pd.DataFrame({
                    'kv_cache_size': kv_cache.flatten(),
                    'prefill_chunk_size_squared': (prefill_chunk.flatten()) ** 2
                })

            elif set(feature_cols) == {'batch_size', 'kv_cache_size'}:
                # Decode predictions (like Vidur)
                # batch_size: EVERY value from 1 to max, kv_cache: with granularity
                batch_range = np.arange(1, self.config.prediction_max_batch_size + 1)
                kv_cache_range = np.arange(0, self.config.prediction_max_tokens + 1,
                                          self.config.kv_cache_granularity)

                batch, kv_cache = np.meshgrid(batch_range, kv_cache_range)
                X = pd.DataFrame({
                    'batch_size': batch.flatten(),
                    'kv_cache_size': kv_cache.flatten()
                })
            else:
                print(f"    Skipping prediction generation for {operation_name}: unusual features {feature_cols}")
                return

            predictions = model.predict(X)
            pred_dict = {}
            for idx, row in X.iterrows():
                key = tuple(row[col] for col in feature_cols)
                pred_dict[key] = predictions[idx]

            self.predictions[operation_name] = pred_dict
            print(f"    Generated {len(pred_dict)} predictions")

    def predict(
        self,
        operation_name: str,
        **kwargs
    ) -> float:
        """
        Predict execution time for an operation.

        Args:
            operation_name: Name of the operation (possibly with _prefill/_decode suffix)
            **kwargs: Feature values (e.g., num_tokens=128, batch_size=4, kv_cache_size=512)

        Returns:
            Predicted execution time in milliseconds
        """

        if operation_name not in self.models:
            raise ValueError(f"No model trained for operation: {operation_name}")

        model = self.models[operation_name]

        # Determine which features this model expects
        # This is stored when we train the model
        if operation_name in self.predictions and len(self.predictions[operation_name]) > 0:
            # Get feature names from a sample prediction key
            sample_key = next(iter(self.predictions[operation_name].keys()))
            num_features = len(sample_key)

            # Map common features based on number and operation name
            if '_prefill' in operation_name:
                feature_names = ['kv_cache_size', 'prefill_chunk_size_squared']
                # Handle prefill_chunk_size -> prefill_chunk_size_squared
                if 'prefill_chunk_size' in kwargs and 'prefill_chunk_size_squared' not in kwargs:
                    kwargs['prefill_chunk_size_squared'] = kwargs['prefill_chunk_size'] ** 2
            elif '_decode' in operation_name:
                feature_names = ['batch_size', 'kv_cache_size']
            else:
                feature_names = ['num_tokens']
        else:
            # Fallback
            feature_names = ['num_tokens']

        # Prepare input
        features = []
        for col in feature_names:
            if col not in kwargs:
                raise ValueError(f"Missing required feature '{col}' for operation {operation_name}")
            features.append(kwargs[col])

        # Check if we have pre-computed prediction
        feature_tuple = tuple(features)
        if operation_name in self.predictions and feature_tuple in self.predictions[operation_name]:
            return self.predictions[operation_name][feature_tuple]

        # Otherwise, predict on the fly
        X = pd.DataFrame([{col: kwargs[col] for col in feature_names}])
        prediction = model.predict(X)[0]

        return prediction

    def get_operation_stats(self) -> pd.DataFrame:
        """Get statistics about trained operations."""

        stats = []
        for (op_name, sig_hash), group_df in self.operation_groups:
            is_context = group_df['is_context'].iloc[0] if 'is_context' in group_df else False
            sig_short = sig_hash[:8]
            model_key = f"{op_name}_{sig_short}"

            stats.append({
                'operation': op_name,
                'signature': sig_short,
                'is_context': is_context,
                'num_samples': len(group_df),
                'min_tokens': group_df['num_tokens'].min() if 'num_tokens' in group_df else None,
                'max_tokens': group_df['num_tokens'].max() if 'num_tokens' in group_df else None,
                'mean_latency': group_df['median_latency_ms'].mean(),
                'std_latency': group_df['median_latency_ms'].std(),
                'has_model': model_key in self.models or
                           f"{op_name}_prefill" in self.models or
                           f"{op_name}_decode" in self.models
            })

        return pd.DataFrame(stats)

    def get_predictor(
        self,
        kv_cache_granularity: int = 64,
        attention_prefill_batching_overhead_fraction: float = 0.0,
        attention_decode_batching_overhead_fraction: float = 0.0,
    ):
        """
        Get an OperationTimePredictor instance using Vidur-style prediction.

        Args:
            kv_cache_granularity: Granularity for rounding KV cache sizes (default: 64)
            attention_prefill_batching_overhead_fraction: Overhead when batching prefill (default: 0.0)
            attention_decode_batching_overhead_fraction: Overhead when batching decode (default: 0.0)

        Returns:
            OperationTimePredictor instance
        """
        try:
            from simulator.operation_time_predictor import OperationTimePredictor
        except ImportError:
            from operation_time_predictor import OperationTimePredictor

        return OperationTimePredictor(
            models=self.models,
            predictions=self.predictions,
            operation_is_context=self.operation_is_context,
            kv_cache_granularity=kv_cache_granularity,
            attention_prefill_batching_overhead_fraction=attention_prefill_batching_overhead_fraction,
            attention_decode_batching_overhead_fraction=attention_decode_batching_overhead_fraction,
        )

    def validate_predictions(self, num_samples: int = 10):
        """
        Validate predictions against actual profiled data.

        Args:
            num_samples: Number of random samples to test per operation

        Prints comparison of predicted vs actual latencies.
        """
        print("\n=== Prediction Validation ===")

        for (operation_name, sig_hash), group_df in self.operation_groups:
            sig_short = sig_hash[:8]
            model_key = f"{operation_name}_{sig_short}"

            # Check if we have a model for this operation signature
            model_names = []
            if model_key in self.models:
                model_names.append(model_key)
            if f"{operation_name}_prefill" in self.models:
                model_names.append(f"{operation_name}_prefill")
            if f"{operation_name}_decode" in self.models:
                model_names.append(f"{operation_name}_decode")

            if not model_names:
                continue

            print(f"\n{operation_name} (sig: {sig_short}):")

            # For split operations (prefill/decode), test each separately
            for model_name in model_names:
                if '_prefill' in model_name:
                    test_df = group_df[group_df['is_prefill'] == 1.0].copy()
                elif '_decode' in model_name:
                    test_df = group_df[group_df['is_prefill'] == 0.0].copy()
                else:
                    test_df = group_df.copy()

                if len(test_df) == 0:
                    continue

                # Sample random rows
                sample_size = min(num_samples, len(test_df))
                samples = test_df.sample(n=sample_size, random_state=42)

                # Get feature columns
                feature_cols = self._get_feature_columns(samples)
                if not feature_cols:
                    continue

                # Prepare features
                X = samples[feature_cols].copy()
                y_actual = samples['median_latency_ms'].values

                # Get predictions
                model = self.models[model_name]
                y_pred = model.predict(X)

                # Calculate errors
                errors = np.abs((y_actual - y_pred) / y_actual) * 100
                mape = np.mean(errors)

                print(f"  {model_name}:")
                print(f"    Tested {sample_size} samples, MAPE: {mape:.2f}%")
                print(f"    Mean actual: {np.mean(y_actual):.4f} ms, Mean predicted: {np.mean(y_pred):.4f} ms")

                # Show a few examples
                for i in range(min(3, sample_size)):
                    features_str = ", ".join([f"{col}={X.iloc[i][col]:.0f}" for col in feature_cols])
                    print(f"      [{features_str}] → Actual: {y_actual[i]:.4f} ms, Predicted: {y_pred[i]:.4f} ms, Error: {errors[i]:.1f}%")


def main(model_name=None, backend=None, db_path=None, cache_dir=None):
    """
    Example usage of the simplified estimator.

    Args:
        model_name: Optional model name to filter (e.g., "meta-llama/Llama-3.1-8B")
        backend: Optional backend to filter (e.g., "FLASHINFER" or "FLASH_ATTN")
        db_path: Optional database path (defaults to dooly_data.db)
        cache_dir: Optional cache directory for models
    """

    # Initialize estimator with optional filtering
    config = EstimatorConfig(
        cache_dir=cache_dir if cache_dir else "/home/jkim38/dooly/simulator/model_cache_v2",
        model_name=model_name,
        backend=backend,
        db_path=db_path if db_path else "/home/jkim38/dooly/profiler/database/dooly_data.db",
        no_cache=False
    )

    print(f"\n=== Initializing Simplified Estimator ===")
    print(f"Using is_context flag for feature selection")
    if model_name:
        print(f"Model filter: {model_name}")
    if backend:
        print(f"Backend filter: {backend}")

    estimator = SimplifiedExecutionTimeEstimator(config)

    # Train all models
    estimator.train_all_models()

    # Show statistics
    print("\n=== Operation Statistics ===")
    stats = estimator.get_operation_stats()
    print(stats.to_string())

    # Validate predictions
    estimator.validate_predictions(num_samples=5)

    # Example predictions using OperationTimePredictor
    print("\n=== Example Predictions (using OperationTimePredictor) ===")

    # Get predictor instance
    predictor = estimator.get_predictor(
        kv_cache_granularity=64,
        attention_prefill_batching_overhead_fraction=0.0,
        attention_decode_batching_overhead_fraction=0.0,
    )

    # Try some simple predictions
    for op_name in list(estimator.models.keys())[:5]:  # Limit to first 5 for brevity
        print(f"\nOperation: {op_name}")

        try:
            if '_prefill' in op_name:
                # Prefill example
                pred = predictor.predict_simple(
                    op_name,
                    kv_cache_size=512,
                    prefill_chunk_size=128
                )
                print(f"  Prefill (kv=512, chunk=128): {pred:.4f} ms")

            elif '_decode' in op_name:
                # Decode example
                pred = predictor.predict_simple(
                    op_name,
                    batch_size=8,
                    kv_cache_size=1024
                )
                print(f"  Decode (batch=8, kv=1024): {pred:.4f} ms")

            else:
                # Regular operation
                pred = predictor.predict_simple(op_name, num_tokens=256)
                print(f"  With 256 tokens: {pred:.4f} ms")

        except Exception as e:
            print(f"  Could not predict: {e}")

    # Demonstrate batch-aware prediction (Vidur-style)
    print("\n=== Batch-aware Predictions (Vidur-style) ===")
    try:
        from operation_time_predictor import Batch, Request

        # Create a sample batch with decode requests
        batch = Batch(
            requests=[
                Request(num_processed_tokens=512, is_prefill_complete=True),
                Request(num_processed_tokens=1024, is_prefill_complete=True),
            ],
            num_tokens=[1, 1],
            size=2
        )

        # Find a decode attention operation
        decode_ops = [op for op in estimator.models.keys() if '_decode' in op]
        if decode_ops:
            op_name = decode_ops[0]
            pred = predictor.predict(op_name, batch)
            print(f"{op_name} with batch (2 decode requests, avg kv=768): {pred:.4f} ms")

        # Create a sample batch with prefill request
        batch_prefill = Batch(
            requests=[
                Request(num_processed_tokens=0, is_prefill_complete=False),
            ],
            num_tokens=[128],
            size=1
        )

        prefill_ops = [op for op in estimator.models.keys() if '_prefill' in op]
        if prefill_ops:
            op_name = prefill_ops[0]
            pred = predictor.predict(op_name, batch_prefill)
            print(f"{op_name} with batch (1 prefill request, chunk=128): {pred:.4f} ms")

    except Exception as e:
        print(f"Could not demonstrate batch-aware prediction: {e}")

    return estimator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train simplified execution time estimator")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name to filter (e.g., 'meta-llama/Llama-3.1-8B')")
    parser.add_argument("--backend", type=str, default=None,
                       help="Backend to filter (e.g., 'FLASHINFER' or 'FLASH_ATTN')")
    parser.add_argument("--db-path", type=str, default="/home/jkim38/dooly/profiler/database/results.db", help="Path to SQLite database for results")
    parser.add_argument("--cache-dir", type=str, default="/home/jkim38/dooly/simulator/cache", help="Directory for caching trained models") 

    args = parser.parse_args()
    estimator = main(model_name=args.model, backend=args.backend,
                    db_path=args.db_path, cache_dir=args.cache_dir)