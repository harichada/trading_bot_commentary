"""
ML Pipeline Module for Trading Bot

A comprehensive machine learning pipeline including:
- Feature Store with point-in-time correctness
- Model A/B Testing Framework
- Concept Drift Detection
- Model Versioning & Governance
- Online Learning Pipeline
- Prediction Confidence & Explanation

Author: Trading Bot Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union,
    TypeVar, Generic, Protocol, Set
)

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type variables for generic types
T = TypeVar('T')
ModelType = TypeVar('ModelType', bound=BaseEstimator)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class DriftSeverity(Enum):
    """Severity levels for detected drift."""
    NONE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class ModelStatus(Enum):
    """Model lifecycle status states."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    SHADOW = "shadow"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class TrafficSplitStrategy(Enum):
    """Traffic splitting strategies for A/B testing."""
    RANDOM = "random"
    STICKY = "sticky"  # Same user always gets same model
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"


class FeatureType(Enum):
    """Types of features in the feature store."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    TEMPORAL = "temporal"
    TEXT = "text"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeatureMetadata:
    """Metadata for a feature in the feature store."""
    name: str
    feature_type: FeatureType
    description: str = ""
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    source: str = ""
    transformation: str = ""
    dependencies: List[str] = field(default_factory=list)
    statistics: Dict[str, float] = field(default_factory=dict)
    is_active: bool = True
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['feature_type'] = self.feature_type.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureMetadata':
        """Create from dictionary."""
        data = data.copy()
        data['feature_type'] = FeatureType(data['feature_type'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class FeatureValue:
    """A feature value at a specific point in time."""
    feature_name: str
    value: Any
    timestamp: datetime
    version: int = 1
    entity_id: str = ""


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    model_id: str
    name: str
    version: str
    algorithm: str
    status: ModelStatus = ModelStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    description: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    parent_model_id: Optional[str] = None
    training_data_hash: str = ""
    artifact_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        data = data.copy()
        data['status'] = ModelStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class DriftReport:
    """Report of detected drift in data or model."""
    timestamp: datetime
    drift_type: str  # "data", "prediction", "feature_importance"
    severity: DriftSeverity
    affected_features: List[str]
    statistics: Dict[str, float]
    recommendation: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['severity'] = self.severity.name
        return data


@dataclass
class ABTestResult:
    """Results from an A/B test comparison."""
    test_id: str
    model_a_id: str
    model_b_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    model_a_metrics: Dict[str, float] = field(default_factory=dict)
    model_b_metrics: Dict[str, float] = field(default_factory=dict)
    sample_size_a: int = 0
    sample_size_b: int = 0
    p_value: Optional[float] = None
    is_significant: bool = False
    winner: Optional[str] = None
    confidence_level: float = 0.95


@dataclass
class PredictionResult:
    """Result of a model prediction with confidence and explanation."""
    prediction: Any
    probability: Optional[np.ndarray] = None
    confidence: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    model_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OnlineLearningState:
    """State for online learning pipeline."""
    model_id: str
    n_samples_seen: int = 0
    current_learning_rate: float = 0.01
    batch_buffer: List[Tuple[np.ndarray, int]] = field(default_factory=list)
    performance_history: List[Dict[str, float]] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)
    ensemble_weights: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# FEATURE STORE
# =============================================================================

class FeatureStore:
    """
    Point-in-time correct feature store with versioning and metadata tracking.

    Provides efficient storage using numpy arrays and proper cache management
    for high-performance feature retrieval in trading scenarios.
    """

    def __init__(
        self,
        storage_path: str = "feature_store",
        cache_size_mb: int = 100,
        enable_versioning: bool = True
    ):
        """
        Initialize the feature store.

        Args:
            storage_path: Directory for persistent storage
            cache_size_mb: Maximum cache size in megabytes
            enable_versioning: Whether to enable feature versioning
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.cache_size_mb = cache_size_mb
        self.enable_versioning = enable_versioning

        # In-memory storage
        self._features: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)
        self._timestamps: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)
        self._metadata: Dict[str, FeatureMetadata] = {}
        self._cache: Dict[str, Tuple[np.ndarray, datetime]] = {}
        self._cache_size_bytes: int = 0
        self._lock = threading.RLock()

        # Load existing metadata
        self._load_metadata()

        logger.info(f"FeatureStore initialized at {storage_path}")

    def _load_metadata(self) -> None:
        """Load feature metadata from disk."""
        metadata_file = self.storage_path / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    for name, meta_dict in data.items():
                        self._metadata[name] = FeatureMetadata.from_dict(meta_dict)
                logger.info(f"Loaded metadata for {len(self._metadata)} features")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")

    def _save_metadata(self) -> None:
        """Save feature metadata to disk."""
        metadata_file = self.storage_path / "metadata.json"
        try:
            data = {name: meta.to_dict() for name, meta in self._metadata.items()}
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def register_feature(
        self,
        name: str,
        feature_type: FeatureType,
        description: str = "",
        source: str = "",
        transformation: str = "",
        dependencies: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> FeatureMetadata:
        """
        Register a new feature in the store.

        Args:
            name: Unique feature name
            feature_type: Type of the feature
            description: Human-readable description
            source: Data source identifier
            transformation: Description of transformation applied
            dependencies: List of dependent feature names
            tags: Tags for categorization

        Returns:
            FeatureMetadata for the registered feature
        """
        with self._lock:
            if name in self._metadata:
                # Increment version for existing feature
                existing = self._metadata[name]
                if self.enable_versioning:
                    metadata = FeatureMetadata(
                        name=name,
                        feature_type=feature_type,
                        description=description,
                        version=existing.version + 1,
                        source=source,
                        transformation=transformation,
                        dependencies=dependencies or [],
                        tags=tags or []
                    )
                else:
                    existing.updated_at = datetime.now()
                    existing.description = description
                    existing.source = source
                    metadata = existing
            else:
                metadata = FeatureMetadata(
                    name=name,
                    feature_type=feature_type,
                    description=description,
                    source=source,
                    transformation=transformation,
                    dependencies=dependencies or [],
                    tags=tags or []
                )

            self._metadata[name] = metadata
            self._save_metadata()

            logger.info(f"Registered feature: {name} (v{metadata.version})")
            return metadata

    def store_feature_values(
        self,
        feature_name: str,
        values: np.ndarray,
        timestamps: np.ndarray,
        entity_ids: Optional[np.ndarray] = None,
        version: Optional[int] = None
    ) -> None:
        """
        Store feature values with timestamps for point-in-time retrieval.

        Args:
            feature_name: Name of the feature
            values: Numpy array of feature values
            timestamps: Numpy array of timestamps (Unix timestamps)
            entity_ids: Optional array of entity identifiers
            version: Feature version (uses latest if not specified)
        """
        if feature_name not in self._metadata:
            raise ValueError(f"Feature '{feature_name}' not registered")

        values = np.asarray(values)
        timestamps = np.asarray(timestamps)

        if len(values) != len(timestamps):
            raise ValueError("Values and timestamps must have same length")

        with self._lock:
            ver = version or self._metadata[feature_name].version

            # Sort by timestamp for efficient point-in-time retrieval
            sort_idx = np.argsort(timestamps)
            values = values[sort_idx]
            timestamps = timestamps[sort_idx]

            # Append to existing data or create new
            if ver in self._features[feature_name]:
                existing_values = self._features[feature_name][ver]
                existing_ts = self._timestamps[feature_name][ver]

                # Merge and sort
                all_values = np.concatenate([existing_values, values])
                all_ts = np.concatenate([existing_ts, timestamps])
                sort_idx = np.argsort(all_ts)

                self._features[feature_name][ver] = all_values[sort_idx]
                self._timestamps[feature_name][ver] = all_ts[sort_idx]
            else:
                self._features[feature_name][ver] = values
                self._timestamps[feature_name][ver] = timestamps

            # Update statistics
            self._update_feature_statistics(feature_name, values)

            logger.debug(f"Stored {len(values)} values for {feature_name}")

    def _update_feature_statistics(self, feature_name: str, values: np.ndarray) -> None:
        """Update feature statistics with new values."""
        meta = self._metadata[feature_name]

        if meta.feature_type in (FeatureType.NUMERICAL, FeatureType.TEMPORAL):
            valid_values = values[~np.isnan(values)] if np.issubdtype(values.dtype, np.floating) else values
            if len(valid_values) > 0:
                meta.statistics = {
                    'mean': float(np.mean(valid_values)),
                    'std': float(np.std(valid_values)),
                    'min': float(np.min(valid_values)),
                    'max': float(np.max(valid_values)),
                    'median': float(np.median(valid_values)),
                    'count': int(len(valid_values))
                }
        elif meta.feature_type == FeatureType.CATEGORICAL:
            unique, counts = np.unique(values, return_counts=True)
            meta.statistics = {
                'unique_count': int(len(unique)),
                'mode': str(unique[np.argmax(counts)]),
                'count': int(len(values))
            }

    def get_feature_at_time(
        self,
        feature_name: str,
        as_of_timestamp: float,
        version: Optional[int] = None
    ) -> Optional[Any]:
        """
        Get the feature value as of a specific timestamp (point-in-time correct).

        Args:
            feature_name: Name of the feature
            as_of_timestamp: Unix timestamp for point-in-time lookup
            version: Feature version (uses latest if not specified)

        Returns:
            Feature value as of the timestamp, or None if not available
        """
        if feature_name not in self._metadata:
            raise ValueError(f"Feature '{feature_name}' not registered")

        with self._lock:
            ver = version or self._metadata[feature_name].version

            if ver not in self._features[feature_name]:
                return None

            timestamps = self._timestamps[feature_name][ver]
            values = self._features[feature_name][ver]

            # Find the latest value before or at the given timestamp
            idx = np.searchsorted(timestamps, as_of_timestamp, side='right') - 1

            if idx < 0:
                return None

            return values[idx]

    def get_feature_batch(
        self,
        feature_names: List[str],
        as_of_timestamp: float,
        versions: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Get multiple features as of a specific timestamp.

        Args:
            feature_names: List of feature names
            as_of_timestamp: Unix timestamp for point-in-time lookup
            versions: Optional dict of feature versions

        Returns:
            Dictionary of feature name to value
        """
        versions = versions or {}
        result = {}

        for name in feature_names:
            try:
                value = self.get_feature_at_time(
                    name,
                    as_of_timestamp,
                    versions.get(name)
                )
                result[name] = value
            except ValueError as e:
                logger.warning(f"Error getting feature {name}: {e}")
                result[name] = None

        return result

    def get_feature_history(
        self,
        feature_name: str,
        start_timestamp: float,
        end_timestamp: float,
        version: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get feature values within a time range.

        Args:
            feature_name: Name of the feature
            start_timestamp: Start of time range
            end_timestamp: End of time range
            version: Feature version

        Returns:
            Tuple of (values, timestamps) arrays
        """
        if feature_name not in self._metadata:
            raise ValueError(f"Feature '{feature_name}' not registered")

        with self._lock:
            ver = version or self._metadata[feature_name].version

            if ver not in self._features[feature_name]:
                return np.array([]), np.array([])

            timestamps = self._timestamps[feature_name][ver]
            values = self._features[feature_name][ver]

            # Find indices within range
            start_idx = np.searchsorted(timestamps, start_timestamp, side='left')
            end_idx = np.searchsorted(timestamps, end_timestamp, side='right')

            return values[start_idx:end_idx], timestamps[start_idx:end_idx]

    def get_feature_metadata(self, feature_name: str) -> Optional[FeatureMetadata]:
        """Get metadata for a feature."""
        return self._metadata.get(feature_name)

    def list_features(
        self,
        tags: Optional[List[str]] = None,
        feature_type: Optional[FeatureType] = None,
        active_only: bool = True
    ) -> List[FeatureMetadata]:
        """
        List features matching criteria.

        Args:
            tags: Filter by tags (any match)
            feature_type: Filter by feature type
            active_only: Only return active features

        Returns:
            List of matching FeatureMetadata
        """
        results = []

        for meta in self._metadata.values():
            if active_only and not meta.is_active:
                continue
            if feature_type and meta.feature_type != feature_type:
                continue
            if tags and not any(t in meta.tags for t in tags):
                continue
            results.append(meta)

        return results

    def invalidate_cache(self, feature_name: Optional[str] = None) -> None:
        """
        Invalidate the cache for a feature or all features.

        Args:
            feature_name: Specific feature to invalidate, or None for all
        """
        with self._lock:
            if feature_name:
                keys_to_remove = [k for k in self._cache if k.startswith(feature_name)]
                for key in keys_to_remove:
                    if key in self._cache:
                        data, _ = self._cache[key]
                        self._cache_size_bytes -= data.nbytes
                        del self._cache[key]
            else:
                self._cache.clear()
                self._cache_size_bytes = 0

        logger.debug(f"Cache invalidated for: {feature_name or 'all features'}")

    def persist_features(self, feature_names: Optional[List[str]] = None) -> None:
        """
        Persist features to disk.

        Args:
            feature_names: List of features to persist, or None for all
        """
        features_to_save = feature_names or list(self._features.keys())

        for name in features_to_save:
            if name not in self._features:
                continue

            feature_dir = self.storage_path / name
            feature_dir.mkdir(exist_ok=True)

            for version, values in self._features[name].items():
                try:
                    np.save(feature_dir / f"values_v{version}.npy", values)
                    np.save(
                        feature_dir / f"timestamps_v{version}.npy",
                        self._timestamps[name][version]
                    )
                except Exception as e:
                    logger.error(f"Error persisting {name} v{version}: {e}")

        self._save_metadata()
        logger.info(f"Persisted {len(features_to_save)} features to disk")

    def load_features(self, feature_names: Optional[List[str]] = None) -> None:
        """
        Load features from disk.

        Args:
            feature_names: List of features to load, or None for all
        """
        if feature_names is None:
            feature_names = [d.name for d in self.storage_path.iterdir()
                          if d.is_dir() and d.name != "__pycache__"]

        for name in feature_names:
            feature_dir = self.storage_path / name
            if not feature_dir.exists():
                continue

            for values_file in feature_dir.glob("values_v*.npy"):
                try:
                    version = int(values_file.stem.split('_v')[1])
                    values = np.load(values_file)
                    timestamps = np.load(feature_dir / f"timestamps_v{version}.npy")

                    self._features[name][version] = values
                    self._timestamps[name][version] = timestamps
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}")

        logger.info(f"Loaded {len(feature_names)} features from disk")


# =============================================================================
# MODEL A/B TESTING FRAMEWORK
# =============================================================================

class ABTestingFramework:
    """
    Framework for A/B testing machine learning models with shadow deployment,
    traffic splitting, and statistical significance testing.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        min_sample_size: int = 100,
        auto_select_winner: bool = True
    ):
        """
        Initialize the A/B testing framework.

        Args:
            confidence_level: Statistical confidence level for significance
            min_sample_size: Minimum samples before significance testing
            auto_select_winner: Automatically select winner when significant
        """
        self.confidence_level = confidence_level
        self.min_sample_size = min_sample_size
        self.auto_select_winner = auto_select_winner

        self._active_tests: Dict[str, ABTestResult] = {}
        self._model_predictions: Dict[str, List[Tuple[Any, Any]]] = defaultdict(list)
        self._shadow_models: Dict[str, BaseEstimator] = {}
        self._traffic_counter: int = 0
        self._user_assignments: Dict[str, str] = {}
        self._lock = threading.RLock()

        logger.info("ABTestingFramework initialized")

    def create_test(
        self,
        test_id: str,
        model_a: BaseEstimator,
        model_b: BaseEstimator,
        model_a_id: str,
        model_b_id: str,
        traffic_split: float = 0.5,
        split_strategy: TrafficSplitStrategy = TrafficSplitStrategy.RANDOM
    ) -> ABTestResult:
        """
        Create a new A/B test.

        Args:
            test_id: Unique test identifier
            model_a: Champion model (control)
            model_b: Challenger model (treatment)
            model_a_id: ID of model A
            model_b_id: ID of model B
            traffic_split: Fraction of traffic to model B (0.0 to 1.0)
            split_strategy: Strategy for traffic splitting

        Returns:
            ABTestResult for the new test
        """
        with self._lock:
            if test_id in self._active_tests:
                raise ValueError(f"Test '{test_id}' already exists")

            result = ABTestResult(
                test_id=test_id,
                model_a_id=model_a_id,
                model_b_id=model_b_id,
                start_time=datetime.now()
            )

            self._active_tests[test_id] = result
            self._shadow_models[f"{test_id}_a"] = model_a
            self._shadow_models[f"{test_id}_b"] = model_b

            # Store test configuration
            result.model_a_metrics['traffic_split'] = 1.0 - traffic_split
            result.model_b_metrics['traffic_split'] = traffic_split

            logger.info(f"Created A/B test: {test_id} ({model_a_id} vs {model_b_id})")
            return result

    def get_model_for_prediction(
        self,
        test_id: str,
        user_id: Optional[str] = None,
        split_strategy: TrafficSplitStrategy = TrafficSplitStrategy.RANDOM
    ) -> Tuple[BaseEstimator, str]:
        """
        Get the appropriate model for a prediction based on traffic split.

        Args:
            test_id: Test identifier
            user_id: Optional user ID for sticky assignment
            split_strategy: Strategy for traffic splitting

        Returns:
            Tuple of (model, model_variant) where variant is 'a' or 'b'
        """
        if test_id not in self._active_tests:
            raise ValueError(f"Test '{test_id}' not found")

        result = self._active_tests[test_id]
        traffic_split_b = result.model_b_metrics.get('traffic_split', 0.5)

        with self._lock:
            if split_strategy == TrafficSplitStrategy.STICKY and user_id:
                if user_id in self._user_assignments:
                    variant = self._user_assignments[user_id]
                else:
                    # Assign based on hash for consistency
                    hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
                    variant = 'b' if (hash_val % 100) < (traffic_split_b * 100) else 'a'
                    self._user_assignments[user_id] = variant
            elif split_strategy == TrafficSplitStrategy.ROUND_ROBIN:
                variant = 'b' if self._traffic_counter % 2 == 1 else 'a'
                self._traffic_counter += 1
            else:  # Random or weighted
                variant = 'b' if np.random.random() < traffic_split_b else 'a'

        model_key = f"{test_id}_{variant}"
        return self._shadow_models[model_key], variant

    def record_prediction(
        self,
        test_id: str,
        variant: str,
        prediction: Any,
        actual: Any,
        confidence: float = 1.0
    ) -> None:
        """
        Record a prediction result for statistical analysis.

        Args:
            test_id: Test identifier
            variant: Model variant ('a' or 'b')
            prediction: Model's prediction
            actual: Actual outcome
            confidence: Prediction confidence
        """
        if test_id not in self._active_tests:
            raise ValueError(f"Test '{test_id}' not found")

        with self._lock:
            key = f"{test_id}_{variant}"
            self._model_predictions[key].append((prediction, actual, confidence))

            # Update sample counts
            result = self._active_tests[test_id]
            if variant == 'a':
                result.sample_size_a = len(self._model_predictions[f"{test_id}_a"])
            else:
                result.sample_size_b = len(self._model_predictions[f"{test_id}_b"])

            # Check for significance if enough samples
            if (result.sample_size_a >= self.min_sample_size and
                result.sample_size_b >= self.min_sample_size):
                self._update_test_statistics(test_id)

    def run_shadow_prediction(
        self,
        test_id: str,
        X: np.ndarray,
        y_actual: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Run predictions on both models (shadow mode) without affecting traffic.

        Args:
            test_id: Test identifier
            X: Input features
            y_actual: Optional actual outcomes for recording

        Returns:
            Dict with predictions from both models
        """
        if test_id not in self._active_tests:
            raise ValueError(f"Test '{test_id}' not found")

        model_a = self._shadow_models[f"{test_id}_a"]
        model_b = self._shadow_models[f"{test_id}_b"]

        try:
            pred_a = model_a.predict(X)
            pred_b = model_b.predict(X)

            # Get probabilities if available
            prob_a = model_a.predict_proba(X) if hasattr(model_a, 'predict_proba') else None
            prob_b = model_b.predict_proba(X) if hasattr(model_b, 'predict_proba') else None

            # Record if actual outcomes provided
            if y_actual is not None:
                for i in range(len(y_actual)):
                    self.record_prediction(test_id, 'a', pred_a[i], y_actual[i])
                    self.record_prediction(test_id, 'b', pred_b[i], y_actual[i])

            return {
                'model_a_predictions': pred_a,
                'model_b_predictions': pred_b,
                'model_a_probabilities': prob_a,
                'model_b_probabilities': prob_b
            }
        except Exception as e:
            logger.error(f"Error in shadow prediction: {e}")
            raise

    def _update_test_statistics(self, test_id: str) -> None:
        """Update statistical analysis for a test."""
        result = self._active_tests[test_id]

        preds_a = self._model_predictions[f"{test_id}_a"]
        preds_b = self._model_predictions[f"{test_id}_b"]

        if not preds_a or not preds_b:
            return

        # Calculate accuracy for each model
        correct_a = sum(1 for p, a, _ in preds_a if p == a)
        correct_b = sum(1 for p, a, _ in preds_b if p == a)

        acc_a = correct_a / len(preds_a)
        acc_b = correct_b / len(preds_b)

        result.model_a_metrics['accuracy'] = acc_a
        result.model_b_metrics['accuracy'] = acc_b

        # Perform statistical significance test (two-proportion z-test)
        n_a, n_b = len(preds_a), len(preds_b)
        p_pooled = (correct_a + correct_b) / (n_a + n_b)

        if p_pooled > 0 and p_pooled < 1:
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
            if se > 0:
                z_score = (acc_b - acc_a) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

                result.p_value = p_value
                result.is_significant = p_value < (1 - self.confidence_level)

                if result.is_significant and self.auto_select_winner:
                    result.winner = result.model_b_id if acc_b > acc_a else result.model_a_id
                    logger.info(
                        f"Test {test_id}: Winner selected - {result.winner} "
                        f"(p={p_value:.4f})"
                    )

    def get_test_results(self, test_id: str) -> ABTestResult:
        """Get current results for a test."""
        if test_id not in self._active_tests:
            raise ValueError(f"Test '{test_id}' not found")
        return self._active_tests[test_id]

    def end_test(self, test_id: str) -> ABTestResult:
        """
        End an A/B test and finalize results.

        Args:
            test_id: Test identifier

        Returns:
            Final ABTestResult
        """
        if test_id not in self._active_tests:
            raise ValueError(f"Test '{test_id}' not found")

        with self._lock:
            result = self._active_tests[test_id]
            result.end_time = datetime.now()

            # Final statistics update
            self._update_test_statistics(test_id)

            # Cleanup
            del self._shadow_models[f"{test_id}_a"]
            del self._shadow_models[f"{test_id}_b"]
            del self._model_predictions[f"{test_id}_a"]
            del self._model_predictions[f"{test_id}_b"]
            del self._active_tests[test_id]

            logger.info(f"Ended A/B test: {test_id}")
            return result

    def get_winner(self, test_id: str) -> Optional[str]:
        """Get the winning model ID if test is conclusive."""
        if test_id not in self._active_tests:
            raise ValueError(f"Test '{test_id}' not found")
        return self._active_tests[test_id].winner

    def list_active_tests(self) -> List[str]:
        """List all active test IDs."""
        return list(self._active_tests.keys())


# =============================================================================
# CONCEPT DRIFT DETECTION
# =============================================================================

class DriftDetector:
    """
    Detects concept drift in data distributions, predictions, and feature importance.

    Implements multiple drift detection methods including KS test, PSI,
    and custom monitors for prediction and feature importance drift.
    """

    # Thresholds for drift severity classification
    PSI_THRESHOLDS = {
        DriftSeverity.LOW: 0.1,
        DriftSeverity.MEDIUM: 0.2,
        DriftSeverity.HIGH: 0.3,
        DriftSeverity.CRITICAL: 0.5
    }

    KS_THRESHOLDS = {
        DriftSeverity.LOW: 0.1,
        DriftSeverity.MEDIUM: 0.2,
        DriftSeverity.HIGH: 0.3,
        DriftSeverity.CRITICAL: 0.4
    }

    def __init__(
        self,
        reference_window_size: int = 1000,
        detection_window_size: int = 100,
        psi_bins: int = 10,
        ks_significance: float = 0.05,
        auto_retrain_threshold: DriftSeverity = DriftSeverity.HIGH
    ):
        """
        Initialize the drift detector.

        Args:
            reference_window_size: Size of reference data window
            detection_window_size: Size of detection window
            psi_bins: Number of bins for PSI calculation
            ks_significance: Significance level for KS test
            auto_retrain_threshold: Severity level to trigger retraining
        """
        self.reference_window_size = reference_window_size
        self.detection_window_size = detection_window_size
        self.psi_bins = psi_bins
        self.ks_significance = ks_significance
        self.auto_retrain_threshold = auto_retrain_threshold

        self._reference_data: Dict[str, np.ndarray] = {}
        self._current_data: Dict[str, List[float]] = defaultdict(list)
        self._reference_predictions: np.ndarray = np.array([])
        self._current_predictions: List[float] = []
        self._reference_feature_importance: Dict[str, float] = {}
        self._drift_history: List[DriftReport] = []
        self._retrain_callbacks: List[Callable[[], None]] = []
        self._lock = threading.RLock()

        logger.info("DriftDetector initialized")

    def set_reference_data(
        self,
        feature_name: str,
        data: np.ndarray
    ) -> None:
        """
        Set reference data for a feature.

        Args:
            feature_name: Name of the feature
            data: Reference data array
        """
        with self._lock:
            # Keep only the most recent reference_window_size samples
            if len(data) > self.reference_window_size:
                data = data[-self.reference_window_size:]
            self._reference_data[feature_name] = np.asarray(data)

        logger.debug(f"Set reference data for {feature_name}: {len(data)} samples")

    def set_reference_predictions(self, predictions: np.ndarray) -> None:
        """Set reference predictions for prediction drift detection."""
        with self._lock:
            if len(predictions) > self.reference_window_size:
                predictions = predictions[-self.reference_window_size:]
            self._reference_predictions = np.asarray(predictions)

    def set_reference_feature_importance(
        self,
        importance: Dict[str, float]
    ) -> None:
        """Set reference feature importance values."""
        with self._lock:
            self._reference_feature_importance = importance.copy()

    def add_data_point(
        self,
        feature_values: Dict[str, float],
        prediction: Optional[float] = None
    ) -> Optional[DriftReport]:
        """
        Add a new data point and check for drift.

        Args:
            feature_values: Dict of feature name to value
            prediction: Optional model prediction

        Returns:
            DriftReport if drift detected, None otherwise
        """
        with self._lock:
            # Add to current windows
            for name, value in feature_values.items():
                self._current_data[name].append(value)
                # Keep window size limited
                if len(self._current_data[name]) > self.detection_window_size:
                    self._current_data[name] = self._current_data[name][-self.detection_window_size:]

            if prediction is not None:
                self._current_predictions.append(prediction)
                if len(self._current_predictions) > self.detection_window_size:
                    self._current_predictions = self._current_predictions[-self.detection_window_size:]

            # Check if we have enough data
            if len(self._current_predictions) < self.detection_window_size:
                return None

            # Check for drift
            return self._check_all_drift()

    def _check_all_drift(self) -> Optional[DriftReport]:
        """Check for all types of drift."""
        reports = []

        # Data drift (per feature)
        data_drift = self.detect_data_drift()
        if data_drift and data_drift.severity != DriftSeverity.NONE:
            reports.append(data_drift)

        # Prediction drift
        if len(self._reference_predictions) > 0 and len(self._current_predictions) > 0:
            pred_drift = self.detect_prediction_drift()
            if pred_drift and pred_drift.severity != DriftSeverity.NONE:
                reports.append(pred_drift)

        # Return the most severe drift
        if reports:
            most_severe = max(reports, key=lambda r: r.severity.value)
            self._drift_history.append(most_severe)

            # Trigger retraining if needed
            if most_severe.severity.value >= self.auto_retrain_threshold.value:
                self._trigger_retraining(most_severe)

            return most_severe

        return None

    def detect_data_drift(self) -> Optional[DriftReport]:
        """
        Detect data drift using KS test and PSI.

        Returns:
            DriftReport with drift information
        """
        affected_features = []
        all_stats = {}
        max_severity = DriftSeverity.NONE

        for feature_name, current in self._current_data.items():
            if feature_name not in self._reference_data:
                continue

            reference = self._reference_data[feature_name]
            current_arr = np.array(current)

            if len(current_arr) < 10 or len(reference) < 10:
                continue

            # KS Test
            ks_stat, ks_pvalue = stats.ks_2samp(reference, current_arr)

            # PSI
            psi_value = self._calculate_psi(reference, current_arr)

            # Determine severity
            severity = self._classify_severity_combined(ks_stat, psi_value)

            if severity != DriftSeverity.NONE:
                affected_features.append(feature_name)
                all_stats[feature_name] = {
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(ks_pvalue),
                    'psi': float(psi_value)
                }

                if severity.value > max_severity.value:
                    max_severity = severity

        if not affected_features:
            return DriftReport(
                timestamp=datetime.now(),
                drift_type="data",
                severity=DriftSeverity.NONE,
                affected_features=[],
                statistics={},
                recommendation="No data drift detected"
            )

        recommendation = self._generate_recommendation(
            "data", max_severity, affected_features
        )

        return DriftReport(
            timestamp=datetime.now(),
            drift_type="data",
            severity=max_severity,
            affected_features=affected_features,
            statistics=all_stats,
            recommendation=recommendation
        )

    def detect_prediction_drift(self) -> Optional[DriftReport]:
        """
        Detect drift in model predictions.

        Returns:
            DriftReport with prediction drift information
        """
        if len(self._reference_predictions) == 0 or len(self._current_predictions) == 0:
            return None

        current_arr = np.array(self._current_predictions)

        # KS test on prediction distributions
        ks_stat, ks_pvalue = stats.ks_2samp(
            self._reference_predictions, current_arr
        )

        # PSI for predictions
        psi_value = self._calculate_psi(
            self._reference_predictions, current_arr
        )

        severity = self._classify_severity_combined(ks_stat, psi_value)

        recommendation = self._generate_recommendation(
            "prediction", severity, ["predictions"]
        )

        return DriftReport(
            timestamp=datetime.now(),
            drift_type="prediction",
            severity=severity,
            affected_features=["predictions"],
            statistics={
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                'psi': float(psi_value),
                'reference_mean': float(np.mean(self._reference_predictions)),
                'current_mean': float(np.mean(current_arr)),
                'reference_std': float(np.std(self._reference_predictions)),
                'current_std': float(np.std(current_arr))
            },
            recommendation=recommendation
        )

    def detect_feature_importance_drift(
        self,
        current_importance: Dict[str, float]
    ) -> DriftReport:
        """
        Detect drift in feature importance.

        Args:
            current_importance: Current feature importance values

        Returns:
            DriftReport with feature importance drift information
        """
        if not self._reference_feature_importance:
            return DriftReport(
                timestamp=datetime.now(),
                drift_type="feature_importance",
                severity=DriftSeverity.NONE,
                affected_features=[],
                statistics={},
                recommendation="No reference feature importance set"
            )

        affected_features = []
        importance_changes = {}
        max_change = 0.0

        for feature, ref_importance in self._reference_feature_importance.items():
            if feature not in current_importance:
                continue

            current_imp = current_importance[feature]

            # Calculate relative change
            if ref_importance > 0:
                relative_change = abs(current_imp - ref_importance) / ref_importance
            else:
                relative_change = abs(current_imp - ref_importance)

            importance_changes[feature] = {
                'reference': ref_importance,
                'current': current_imp,
                'relative_change': relative_change
            }

            if relative_change > 0.3:  # 30% change threshold
                affected_features.append(feature)
                max_change = max(max_change, relative_change)

        # Classify severity based on maximum change
        if max_change >= 1.0:
            severity = DriftSeverity.CRITICAL
        elif max_change >= 0.7:
            severity = DriftSeverity.HIGH
        elif max_change >= 0.5:
            severity = DriftSeverity.MEDIUM
        elif max_change >= 0.3:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE

        recommendation = self._generate_recommendation(
            "feature_importance", severity, affected_features
        )

        report = DriftReport(
            timestamp=datetime.now(),
            drift_type="feature_importance",
            severity=severity,
            affected_features=affected_features,
            statistics={'max_relative_change': max_change},
            recommendation=recommendation,
            details=importance_changes
        )

        if severity != DriftSeverity.NONE:
            self._drift_history.append(report)

        return report

    def _calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        Args:
            reference: Reference data array
            current: Current data array

        Returns:
            PSI value
        """
        # Create bins from reference data
        _, bin_edges = np.histogram(reference, bins=self.psi_bins)

        # Calculate proportions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Add small epsilon to avoid division by zero
        eps = 1e-10
        ref_props = (ref_counts + eps) / (len(reference) + eps * self.psi_bins)
        cur_props = (cur_counts + eps) / (len(current) + eps * self.psi_bins)

        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

        return psi

    def _classify_severity_combined(
        self,
        ks_stat: float,
        psi_value: float
    ) -> DriftSeverity:
        """Classify severity based on KS statistic and PSI."""
        # Use the more severe of the two
        ks_severity = DriftSeverity.NONE
        for severity, threshold in sorted(
            self.KS_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if ks_stat >= threshold:
                ks_severity = severity
                break

        psi_severity = DriftSeverity.NONE
        for severity, threshold in sorted(
            self.PSI_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if psi_value >= threshold:
                psi_severity = severity
                break

        return max(ks_severity, psi_severity, key=lambda x: x.value)

    def _generate_recommendation(
        self,
        drift_type: str,
        severity: DriftSeverity,
        affected_features: List[str]
    ) -> str:
        """Generate a recommendation based on drift detection."""
        if severity == DriftSeverity.NONE:
            return "No action needed - no significant drift detected."

        recommendations = {
            DriftSeverity.LOW: (
                f"Low {drift_type} drift detected in {len(affected_features)} feature(s). "
                "Monitor closely and consider investigating root cause."
            ),
            DriftSeverity.MEDIUM: (
                f"Medium {drift_type} drift detected in {len(affected_features)} feature(s). "
                "Investigate data sources and consider partial model retraining."
            ),
            DriftSeverity.HIGH: (
                f"High {drift_type} drift detected in {len(affected_features)} feature(s). "
                "Model retraining recommended. Check for data pipeline issues."
            ),
            DriftSeverity.CRITICAL: (
                f"CRITICAL {drift_type} drift in {len(affected_features)} feature(s)! "
                "Immediate action required. Consider model rollback and emergency retraining."
            )
        }

        return recommendations.get(severity, "Unknown severity level")

    def register_retrain_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when retraining is triggered."""
        self._retrain_callbacks.append(callback)

    def _trigger_retraining(self, report: DriftReport) -> None:
        """Trigger model retraining callbacks."""
        logger.warning(
            f"Drift threshold exceeded ({report.severity.name}). "
            f"Triggering retraining for {len(self._retrain_callbacks)} callbacks."
        )

        for callback in self._retrain_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in retrain callback: {e}")

    def get_drift_history(
        self,
        limit: int = 100,
        drift_type: Optional[str] = None,
        min_severity: DriftSeverity = DriftSeverity.NONE
    ) -> List[DriftReport]:
        """Get drift detection history."""
        history = self._drift_history[-limit:]

        if drift_type:
            history = [r for r in history if r.drift_type == drift_type]

        if min_severity != DriftSeverity.NONE:
            history = [r for r in history if r.severity.value >= min_severity.value]

        return history

    def reset(self) -> None:
        """Reset all data windows and history."""
        with self._lock:
            self._current_data.clear()
            self._current_predictions.clear()
            self._drift_history.clear()
        logger.info("DriftDetector reset")


# =============================================================================
# MODEL VERSIONING & GOVERNANCE
# =============================================================================

class ModelRegistry:
    """
    Model registry for versioning, governance, and lifecycle management.

    Provides model registration, approval workflows, lineage tracking,
    and rollback capabilities.
    """

    def __init__(
        self,
        storage_path: str = "model_registry",
        require_approval: bool = True
    ):
        """
        Initialize the model registry.

        Args:
            storage_path: Directory for model storage
            require_approval: Whether approval is required for deployment
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.require_approval = require_approval

        self._models: Dict[str, ModelMetadata] = {}
        self._model_artifacts: Dict[str, BaseEstimator] = {}
        self._performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._deployed_model_id: Optional[str] = None
        self._lock = threading.RLock()

        # Load existing registry
        self._load_registry()

        logger.info(f"ModelRegistry initialized at {storage_path}")

    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = self.storage_path / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    for model_id, meta_dict in data.get('models', {}).items():
                        self._models[model_id] = ModelMetadata.from_dict(meta_dict)
                    self._deployed_model_id = data.get('deployed_model_id')
                logger.info(f"Loaded {len(self._models)} models from registry")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")

    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self.storage_path / "registry.json"
        try:
            data = {
                'models': {k: v.to_dict() for k, v in self._models.items()},
                'deployed_model_id': self._deployed_model_id
            }
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")

    def register_model(
        self,
        model: BaseEstimator,
        name: str,
        version: str,
        algorithm: str,
        description: str = "",
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        parent_model_id: Optional[str] = None,
        training_data_hash: str = ""
    ) -> ModelMetadata:
        """
        Register a new model in the registry.

        Args:
            model: Trained model object
            name: Model name
            version: Version string
            algorithm: Algorithm name
            description: Model description
            hyperparameters: Model hyperparameters
            feature_names: List of feature names used
            metrics: Training/validation metrics
            tags: Tags for categorization
            parent_model_id: ID of parent model (for lineage)
            training_data_hash: Hash of training data

        Returns:
            ModelMetadata for the registered model
        """
        model_id = str(uuid.uuid4())

        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            algorithm=algorithm,
            description=description,
            hyperparameters=hyperparameters or {},
            feature_names=feature_names or [],
            metrics=metrics or {},
            tags=tags or [],
            parent_model_id=parent_model_id,
            training_data_hash=training_data_hash
        )

        with self._lock:
            # Save model artifact
            artifact_path = self.storage_path / f"{model_id}.pkl"
            with open(artifact_path, 'wb') as f:
                pickle.dump(model, f)
            metadata.artifact_path = str(artifact_path)

            # Store in registry
            self._models[model_id] = metadata
            self._model_artifacts[model_id] = model
            self._save_registry()

        logger.info(f"Registered model: {name} v{version} (ID: {model_id})")
        return metadata

    def get_model(self, model_id: str) -> Optional[BaseEstimator]:
        """
        Get a model by ID.

        Args:
            model_id: Model identifier

        Returns:
            Model object or None
        """
        if model_id in self._model_artifacts:
            return self._model_artifacts[model_id]

        # Load from disk
        metadata = self._models.get(model_id)
        if metadata and metadata.artifact_path:
            try:
                with open(metadata.artifact_path, 'rb') as f:
                    model = pickle.load(f)
                    self._model_artifacts[model_id] = model
                    return model
            except Exception as e:
                logger.error(f"Error loading model {model_id}: {e}")

        return None

    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a model."""
        return self._models.get(model_id)

    def update_status(
        self,
        model_id: str,
        new_status: ModelStatus,
        approved_by: str = "system"
    ) -> bool:
        """
        Update model status (approval workflow).

        Args:
            model_id: Model identifier
            new_status: New status to set
            approved_by: Approver identifier

        Returns:
            True if status was updated
        """
        if model_id not in self._models:
            logger.warning(f"Model {model_id} not found")
            return False

        with self._lock:
            metadata = self._models[model_id]
            old_status = metadata.status

            # Validate status transitions
            valid_transitions = {
                ModelStatus.DRAFT: [ModelStatus.PENDING_REVIEW, ModelStatus.ARCHIVED],
                ModelStatus.PENDING_REVIEW: [ModelStatus.APPROVED, ModelStatus.DRAFT],
                ModelStatus.APPROVED: [ModelStatus.DEPLOYED, ModelStatus.SHADOW, ModelStatus.DEPRECATED],
                ModelStatus.DEPLOYED: [ModelStatus.DEPRECATED, ModelStatus.SHADOW],
                ModelStatus.SHADOW: [ModelStatus.DEPLOYED, ModelStatus.DEPRECATED],
                ModelStatus.DEPRECATED: [ModelStatus.ARCHIVED],
                ModelStatus.ARCHIVED: []
            }

            if new_status not in valid_transitions.get(old_status, []):
                logger.warning(
                    f"Invalid status transition: {old_status.value} -> {new_status.value}"
                )
                return False

            metadata.status = new_status
            metadata.updated_at = datetime.now()
            self._save_registry()

            logger.info(
                f"Model {model_id} status: {old_status.value} -> {new_status.value} "
                f"(by {approved_by})"
            )
            return True

    def deploy_model(self, model_id: str) -> bool:
        """
        Deploy a model to production.

        Args:
            model_id: Model identifier

        Returns:
            True if deployment successful
        """
        if model_id not in self._models:
            logger.warning(f"Model {model_id} not found")
            return False

        metadata = self._models[model_id]

        # Check if approval required
        if self.require_approval and metadata.status != ModelStatus.APPROVED:
            logger.warning(f"Model {model_id} not approved for deployment")
            return False

        with self._lock:
            # Deprecate current deployed model
            if self._deployed_model_id and self._deployed_model_id != model_id:
                old_model = self._models.get(self._deployed_model_id)
                if old_model:
                    old_model.status = ModelStatus.DEPRECATED
                    old_model.updated_at = datetime.now()

            # Deploy new model
            metadata.status = ModelStatus.DEPLOYED
            metadata.updated_at = datetime.now()
            self._deployed_model_id = model_id
            self._save_registry()

        logger.info(f"Deployed model: {model_id}")
        return True

    def rollback(self, target_model_id: str) -> bool:
        """
        Rollback to a previous model version.

        Args:
            target_model_id: Model ID to rollback to

        Returns:
            True if rollback successful
        """
        if target_model_id not in self._models:
            logger.warning(f"Target model {target_model_id} not found")
            return False

        target_metadata = self._models[target_model_id]

        # Allow rollback to approved or deprecated models
        if target_metadata.status not in (
            ModelStatus.APPROVED,
            ModelStatus.DEPRECATED,
            ModelStatus.SHADOW
        ):
            logger.warning(f"Cannot rollback to model in {target_metadata.status.value} status")
            return False

        with self._lock:
            # Deprecate current deployed model
            if self._deployed_model_id:
                current = self._models.get(self._deployed_model_id)
                if current:
                    current.status = ModelStatus.DEPRECATED
                    current.updated_at = datetime.now()

            # Restore target model
            target_metadata.status = ModelStatus.DEPLOYED
            target_metadata.updated_at = datetime.now()
            self._deployed_model_id = target_model_id
            self._save_registry()

        logger.info(f"Rolled back to model: {target_model_id}")
        return True

    def get_deployed_model(self) -> Optional[Tuple[BaseEstimator, ModelMetadata]]:
        """Get the currently deployed model and its metadata."""
        if not self._deployed_model_id:
            return None

        model = self.get_model(self._deployed_model_id)
        metadata = self._models.get(self._deployed_model_id)

        if model and metadata:
            return (model, metadata)
        return None

    def record_performance(
        self,
        model_id: str,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record performance metrics for a model.

        Args:
            model_id: Model identifier
            metrics: Performance metrics
            timestamp: Optional timestamp
        """
        with self._lock:
            self._performance_history[model_id].append({
                'timestamp': (timestamp or datetime.now()).isoformat(),
                'metrics': metrics
            })

            # Update model metadata
            if model_id in self._models:
                self._models[model_id].metrics.update(metrics)
                self._save_registry()

    def get_performance_history(
        self,
        model_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get performance history for a model."""
        return self._performance_history[model_id][-limit:]

    def get_lineage(self, model_id: str) -> List[ModelMetadata]:
        """
        Get the lineage (ancestry) of a model.

        Args:
            model_id: Model identifier

        Returns:
            List of ancestor models
        """
        lineage = []
        current_id = model_id

        while current_id:
            metadata = self._models.get(current_id)
            if not metadata:
                break
            lineage.append(metadata)
            current_id = metadata.parent_model_id

        return lineage

    def list_models(
        self,
        status: Optional[ModelStatus] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelMetadata]:
        """
        List models matching criteria.

        Args:
            status: Filter by status
            name: Filter by name
            tags: Filter by tags (any match)

        Returns:
            List of matching models
        """
        results = []

        for metadata in self._models.values():
            if status and metadata.status != status:
                continue
            if name and metadata.name != name:
                continue
            if tags and not any(t in metadata.tags for t in tags):
                continue
            results.append(metadata)

        return sorted(results, key=lambda m: m.created_at, reverse=True)

    def delete_model(self, model_id: str, force: bool = False) -> bool:
        """
        Delete a model from the registry.

        Args:
            model_id: Model identifier
            force: Force delete even if deployed

        Returns:
            True if deleted
        """
        if model_id not in self._models:
            return False

        metadata = self._models[model_id]

        if metadata.status == ModelStatus.DEPLOYED and not force:
            logger.warning("Cannot delete deployed model without force=True")
            return False

        with self._lock:
            # Remove artifact
            if metadata.artifact_path and os.path.exists(metadata.artifact_path):
                os.remove(metadata.artifact_path)

            # Remove from registry
            del self._models[model_id]
            if model_id in self._model_artifacts:
                del self._model_artifacts[model_id]

            if self._deployed_model_id == model_id:
                self._deployed_model_id = None

            self._save_registry()

        logger.info(f"Deleted model: {model_id}")
        return True


# =============================================================================
# ONLINE LEARNING PIPELINE
# =============================================================================

class OnlineLearningPipeline:
    """
    Online learning pipeline with incremental updates, mini-batch training,
    learning rate scheduling, and catastrophic forgetting prevention.
    """

    def __init__(
        self,
        base_model: Optional[BaseEstimator] = None,
        initial_learning_rate: float = 0.01,
        min_learning_rate: float = 0.0001,
        batch_size: int = 32,
        forgetting_factor: float = 0.99,
        ensemble_enabled: bool = True,
        max_ensemble_size: int = 5
    ):
        """
        Initialize the online learning pipeline.

        Args:
            base_model: Base model (must support partial_fit)
            initial_learning_rate: Initial learning rate
            min_learning_rate: Minimum learning rate
            batch_size: Mini-batch size
            forgetting_factor: Factor for exponential forgetting (0-1)
            ensemble_enabled: Enable model ensembling
            max_ensemble_size: Maximum number of ensemble members
        """
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.batch_size = batch_size
        self.forgetting_factor = forgetting_factor
        self.ensemble_enabled = ensemble_enabled
        self.max_ensemble_size = max_ensemble_size

        # Initialize base model
        if base_model is None:
            self.base_model = SGDClassifier(
                loss='log_loss',
                learning_rate='adaptive',
                eta0=initial_learning_rate,
                warm_start=True
            )
        else:
            self.base_model = base_model

        # State
        self._state = OnlineLearningState(
            model_id=str(uuid.uuid4()),
            current_learning_rate=initial_learning_rate
        )
        self._ensemble: List[Tuple[BaseEstimator, float]] = []  # (model, weight)
        self._scaler = StandardScaler()
        self._scaler_fitted = False
        self._feature_importance_history: List[Dict[str, float]] = []
        self._lock = threading.RLock()

        logger.info("OnlineLearningPipeline initialized")

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Perform incremental update with new data.

        Args:
            X: Feature matrix
            y: Target labels
            classes: Array of all possible classes

        Returns:
            Dict of training metrics
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if len(X) == 0:
            return {}

        with self._lock:
            # Update scaler incrementally
            if not self._scaler_fitted:
                self._scaler.fit(X)
                self._scaler_fitted = True
            else:
                self._scaler.partial_fit(X)

            X_scaled = self._scaler.transform(X)

            # Add to batch buffer
            for i in range(len(X)):
                self._state.batch_buffer.append((X_scaled[i], y[i]))

            # Train when batch is full
            metrics = {}
            if len(self._state.batch_buffer) >= self.batch_size:
                metrics = self._train_batch(classes)

            return metrics

    def _train_batch(
        self,
        classes: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Train on accumulated batch."""
        if not self._state.batch_buffer:
            return {}

        # Prepare batch data
        X_batch = np.array([x for x, _ in self._state.batch_buffer])
        y_batch = np.array([y for _, y in self._state.batch_buffer])

        # Update learning rate
        self._update_learning_rate()

        # Train base model
        try:
            if hasattr(self.base_model, 'partial_fit'):
                if classes is not None:
                    self.base_model.partial_fit(X_batch, y_batch, classes=classes)
                else:
                    self.base_model.partial_fit(X_batch, y_batch)
            else:
                self.base_model.fit(X_batch, y_batch)
        except Exception as e:
            logger.error(f"Error in batch training: {e}")
            self._state.batch_buffer.clear()
            return {}

        # Calculate metrics
        predictions = self.base_model.predict(X_batch)
        metrics = {
            'batch_accuracy': float(accuracy_score(y_batch, predictions)),
            'batch_size': len(y_batch),
            'learning_rate': self._state.current_learning_rate
        }

        # Update ensemble
        if self.ensemble_enabled:
            self._update_ensemble(X_batch, y_batch, metrics['batch_accuracy'])

        # Update state
        self._state.n_samples_seen += len(y_batch)
        self._state.last_update = datetime.now()
        self._state.performance_history.append(metrics)

        # Clear buffer
        self._state.batch_buffer.clear()

        logger.debug(f"Trained batch: {metrics}")
        return metrics

    def _update_learning_rate(self) -> None:
        """Update learning rate using scheduling."""
        # Decay learning rate based on samples seen
        decay_rate = 0.95
        decay_steps = 1000

        new_lr = self.initial_learning_rate * (
            decay_rate ** (self._state.n_samples_seen / decay_steps)
        )

        self._state.current_learning_rate = max(new_lr, self.min_learning_rate)

        # Update model learning rate if supported
        if hasattr(self.base_model, 'eta0'):
            self.base_model.eta0 = self._state.current_learning_rate

    def _update_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        accuracy: float
    ) -> None:
        """Update ensemble with new model snapshot."""
        # Create a snapshot of the current model
        try:
            model_snapshot = clone(self.base_model)
            model_snapshot.fit(X, y)
        except Exception as e:
            logger.warning(f"Could not create model snapshot: {e}")
            return

        # Add to ensemble with weight based on accuracy
        weight = accuracy
        self._ensemble.append((model_snapshot, weight))

        # Limit ensemble size
        if len(self._ensemble) > self.max_ensemble_size:
            # Remove lowest weighted model
            self._ensemble.sort(key=lambda x: x[1], reverse=True)
            self._ensemble = self._ensemble[:self.max_ensemble_size]

        # Update ensemble weights with forgetting
        total_weight = sum(w for _, w in self._ensemble)
        if total_weight > 0:
            self._state.ensemble_weights = {
                str(i): w / total_weight
                for i, (_, w) in enumerate(self._ensemble)
            }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble or base model.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        X = np.asarray(X)

        if not self._scaler_fitted:
            raise ValueError("Model not fitted yet")

        X_scaled = self._scaler.transform(X)

        if self.ensemble_enabled and self._ensemble:
            return self._ensemble_predict(X_scaled)

        return self.base_model.predict(X_scaled)

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Make weighted ensemble predictions."""
        predictions = []
        weights = []

        for model, weight in self._ensemble:
            try:
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(weight)
            except Exception as e:
                logger.warning(f"Ensemble member prediction failed: {e}")

        if not predictions:
            return self.base_model.predict(X)

        # Weighted voting
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()

        # For classification, use weighted majority voting
        final_predictions = []
        for i in range(len(X)):
            sample_preds = predictions[:, i]
            unique, counts = np.unique(sample_preds, return_counts=True)

            # Weight the counts
            weighted_counts = np.zeros(len(unique))
            for j, pred in enumerate(sample_preds):
                idx = np.where(unique == pred)[0][0]
                weighted_counts[idx] += weights[j]

            final_predictions.append(unique[np.argmax(weighted_counts)])

        return np.array(final_predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability array
        """
        X = np.asarray(X)

        if not self._scaler_fitted:
            raise ValueError("Model not fitted yet")

        X_scaled = self._scaler.transform(X)

        if self.ensemble_enabled and self._ensemble:
            return self._ensemble_predict_proba(X_scaled)

        if hasattr(self.base_model, 'predict_proba'):
            return self.base_model.predict_proba(X_scaled)

        # Fallback to hard predictions
        predictions = self.base_model.predict(X_scaled)
        n_classes = len(np.unique(predictions))
        proba = np.zeros((len(X), max(2, n_classes)))
        for i, pred in enumerate(predictions):
            proba[i, int(pred)] = 1.0
        return proba

    def _ensemble_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get weighted ensemble probabilities."""
        probas = []
        weights = []

        for model, weight in self._ensemble:
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)
                    probas.append(prob)
                    weights.append(weight)
            except Exception:
                pass

        if not probas:
            if hasattr(self.base_model, 'predict_proba'):
                return self.base_model.predict_proba(X)
            return np.zeros((len(X), 2))

        # Weighted average of probabilities
        weights = np.array(weights)
        weights = weights / weights.sum()

        avg_proba = np.zeros_like(probas[0])
        for prob, weight in zip(probas, weights):
            avg_proba += prob * weight

        return avg_proba

    def get_state(self) -> OnlineLearningState:
        """Get current learning state."""
        return self._state

    def get_performance_trend(
        self,
        window_size: int = 10
    ) -> Dict[str, float]:
        """
        Get performance trend over recent batches.

        Args:
            window_size: Number of recent batches to consider

        Returns:
            Dict with trend statistics
        """
        history = self._state.performance_history[-window_size:]

        if not history:
            return {}

        accuracies = [h.get('batch_accuracy', 0) for h in history]

        return {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'min_accuracy': float(np.min(accuracies)),
            'max_accuracy': float(np.max(accuracies)),
            'trend': float(np.polyfit(range(len(accuracies)), accuracies, 1)[0])
                    if len(accuracies) > 1 else 0.0,
            'n_batches': len(history)
        }

    def save_state(self, path: str) -> None:
        """Save pipeline state to disk."""
        try:
            state_data = {
                'state': {
                    'model_id': self._state.model_id,
                    'n_samples_seen': self._state.n_samples_seen,
                    'current_learning_rate': self._state.current_learning_rate,
                    'performance_history': self._state.performance_history,
                    'ensemble_weights': self._state.ensemble_weights
                },
                'scaler_fitted': self._scaler_fitted
            }

            with open(f"{path}_state.json", 'w') as f:
                json.dump(state_data, f)

            with open(f"{path}_model.pkl", 'wb') as f:
                pickle.dump(self.base_model, f)

            with open(f"{path}_scaler.pkl", 'wb') as f:
                pickle.dump(self._scaler, f)

            if self._ensemble:
                with open(f"{path}_ensemble.pkl", 'wb') as f:
                    pickle.dump(self._ensemble, f)

            logger.info(f"Saved online learning state to {path}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def load_state(self, path: str) -> None:
        """Load pipeline state from disk."""
        try:
            with open(f"{path}_state.json", 'r') as f:
                state_data = json.load(f)

            self._state.model_id = state_data['state']['model_id']
            self._state.n_samples_seen = state_data['state']['n_samples_seen']
            self._state.current_learning_rate = state_data['state']['current_learning_rate']
            self._state.performance_history = state_data['state']['performance_history']
            self._state.ensemble_weights = state_data['state']['ensemble_weights']
            self._scaler_fitted = state_data['scaler_fitted']

            with open(f"{path}_model.pkl", 'rb') as f:
                self.base_model = pickle.load(f)

            with open(f"{path}_scaler.pkl", 'rb') as f:
                self._scaler = pickle.load(f)

            if os.path.exists(f"{path}_ensemble.pkl"):
                with open(f"{path}_ensemble.pkl", 'rb') as f:
                    self._ensemble = pickle.load(f)

            logger.info(f"Loaded online learning state from {path}")
        except Exception as e:
            logger.error(f"Error loading state: {e}")


# =============================================================================
# PREDICTION CONFIDENCE & EXPLANATION
# =============================================================================

class PredictionExplainer:
    """
    Provides prediction confidence estimation, feature contributions,
    and human-readable explanations.
    """

    def __init__(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        confidence_threshold: float = 0.6,
        n_bootstrap_samples: int = 100
    ):
        """
        Initialize the prediction explainer.

        Args:
            model: Trained model
            feature_names: List of feature names
            confidence_threshold: Minimum confidence for predictions
            n_bootstrap_samples: Number of bootstrap samples for CI
        """
        self.model = model
        self.feature_names = feature_names
        self.confidence_threshold = confidence_threshold
        self.n_bootstrap_samples = n_bootstrap_samples

        self._baseline_values: Optional[np.ndarray] = None
        self._feature_stats: Dict[str, Dict[str, float]] = {}

        logger.info("PredictionExplainer initialized")

    def set_baseline(self, X_train: np.ndarray) -> None:
        """
        Set baseline values for feature contribution calculation.

        Args:
            X_train: Training data for baseline statistics
        """
        self._baseline_values = np.mean(X_train, axis=0)

        for i, name in enumerate(self.feature_names):
            self._feature_stats[name] = {
                'mean': float(np.mean(X_train[:, i])),
                'std': float(np.std(X_train[:, i])),
                'min': float(np.min(X_train[:, i])),
                'max': float(np.max(X_train[:, i]))
            }

    def explain_prediction(
        self,
        X: np.ndarray,
        model_id: str = ""
    ) -> PredictionResult:
        """
        Generate an explained prediction with confidence.

        Args:
            X: Feature vector (1D or 2D with single row)
            model_id: Optional model identifier

        Returns:
            PredictionResult with explanation
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get prediction
        prediction = self.model.predict(X)[0]

        # Get probabilities if available
        probability = None
        confidence = 0.5

        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(X)[0]
            confidence = float(np.max(probability))

        # Calculate confidence interval
        ci = self._bootstrap_confidence_interval(X)

        # Calculate feature contributions
        contributions = self._calculate_feature_contributions(X)

        # Generate explanation
        explanation = self._generate_explanation(
            prediction, confidence, contributions
        )

        return PredictionResult(
            prediction=prediction,
            probability=probability,
            confidence=confidence,
            confidence_interval=ci,
            feature_contributions=contributions,
            explanation=explanation,
            model_id=model_id,
            timestamp=datetime.now()
        )

    def _bootstrap_confidence_interval(
        self,
        X: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for prediction."""
        if not hasattr(self.model, 'predict_proba'):
            return (0.0, 1.0)

        # Simple bootstrap CI based on probability
        proba = self.model.predict_proba(X)[0]
        max_prob = np.max(proba)

        # Approximate CI based on probability
        margin = np.sqrt(max_prob * (1 - max_prob) / self.n_bootstrap_samples)

        lower = max(0.0, max_prob - 1.96 * margin)
        upper = min(1.0, max_prob + 1.96 * margin)

        return (lower, upper)

    def _calculate_feature_contributions(
        self,
        X: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate feature contributions using a SHAP-like approximation.

        Uses permutation-based approach to estimate feature importance
        for a single prediction.
        """
        contributions = {}

        if self._baseline_values is None:
            # Simple deviation-based contribution
            for i, name in enumerate(self.feature_names):
                contributions[name] = float(X[0, i])
            return contributions

        # Get baseline prediction
        baseline_X = self._baseline_values.reshape(1, -1)

        if hasattr(self.model, 'predict_proba'):
            baseline_prob = self.model.predict_proba(baseline_X)[0]
            current_prob = self.model.predict_proba(X)[0]
            baseline_score = np.max(baseline_prob)
            current_score = np.max(current_prob)
        else:
            baseline_score = self.model.predict(baseline_X)[0]
            current_score = self.model.predict(X)[0]

        # Calculate marginal contribution of each feature
        for i, name in enumerate(self.feature_names):
            # Create modified input with feature replaced by baseline
            X_modified = X.copy()
            X_modified[0, i] = self._baseline_values[i]

            if hasattr(self.model, 'predict_proba'):
                modified_prob = self.model.predict_proba(X_modified)[0]
                modified_score = np.max(modified_prob)
            else:
                modified_score = self.model.predict(X_modified)[0]

            # Contribution is the change when feature is present
            contributions[name] = float(current_score - modified_score)

        return contributions

    def _generate_explanation(
        self,
        prediction: Any,
        confidence: float,
        contributions: Dict[str, float]
    ) -> str:
        """Generate human-readable explanation."""
        # Sort features by contribution magnitude
        sorted_features = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Get top contributing features
        top_features = sorted_features[:5]

        explanation_parts = []

        # Prediction statement
        if confidence >= self.confidence_threshold:
            explanation_parts.append(
                f"Prediction: {prediction} (High confidence: {confidence:.1%})"
            )
        else:
            explanation_parts.append(
                f"Prediction: {prediction} (Low confidence: {confidence:.1%} - "
                f"below threshold of {self.confidence_threshold:.1%})"
            )

        # Feature contributions
        explanation_parts.append("\nKey factors:")

        for feature, contribution in top_features:
            direction = "increases" if contribution > 0 else "decreases"
            magnitude = abs(contribution)

            if magnitude > 0.1:
                strength = "strongly"
            elif magnitude > 0.05:
                strength = "moderately"
            else:
                strength = "slightly"

            explanation_parts.append(
                f"  - {feature}: {strength} {direction} prediction "
                f"(contribution: {contribution:+.4f})"
            )

        # Confidence interpretation
        if confidence < self.confidence_threshold:
            explanation_parts.append(
                f"\nWARNING: Confidence ({confidence:.1%}) is below threshold. "
                "Consider additional validation before acting on this prediction."
            )

        return "\n".join(explanation_parts)

    def batch_explain(
        self,
        X: np.ndarray,
        model_id: str = ""
    ) -> List[PredictionResult]:
        """
        Generate explanations for multiple predictions.

        Args:
            X: Feature matrix
            model_id: Optional model identifier

        Returns:
            List of PredictionResults
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        results = []
        for i in range(len(X)):
            result = self.explain_prediction(X[i:i+1], model_id)
            results.append(result)

        return results

    def filter_low_confidence(
        self,
        predictions: List[PredictionResult]
    ) -> Tuple[List[PredictionResult], List[PredictionResult]]:
        """
        Filter predictions by confidence threshold.

        Args:
            predictions: List of prediction results

        Returns:
            Tuple of (high_confidence, low_confidence) predictions
        """
        high_confidence = []
        low_confidence = []

        for pred in predictions:
            if pred.confidence >= self.confidence_threshold:
                high_confidence.append(pred)
            else:
                low_confidence.append(pred)

        return high_confidence, low_confidence

    def get_feature_importance_summary(
        self,
        predictions: List[PredictionResult]
    ) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics of feature contributions across predictions.

        Args:
            predictions: List of prediction results

        Returns:
            Dict of feature name to contribution statistics
        """
        if not predictions:
            return {}

        # Collect all contributions
        all_contributions: Dict[str, List[float]] = defaultdict(list)

        for pred in predictions:
            for feature, contribution in pred.feature_contributions.items():
                all_contributions[feature].append(contribution)

        # Calculate statistics
        summary = {}
        for feature, values in all_contributions.items():
            values = np.array(values)
            summary[feature] = {
                'mean_contribution': float(np.mean(values)),
                'std_contribution': float(np.std(values)),
                'abs_mean_contribution': float(np.mean(np.abs(values))),
                'positive_count': int(np.sum(values > 0)),
                'negative_count': int(np.sum(values < 0)),
                'total_predictions': len(values)
            }

        # Sort by absolute mean contribution
        summary = dict(sorted(
            summary.items(),
            key=lambda x: x[1]['abs_mean_contribution'],
            reverse=True
        ))

        return summary


# =============================================================================
# UNIFIED ML PIPELINE
# =============================================================================

class MLPipeline:
    """
    Unified ML Pipeline orchestrating all components.

    Provides a single interface for:
    - Feature management
    - Model training and deployment
    - A/B testing
    - Drift detection
    - Online learning
    - Prediction explanation
    """

    def __init__(
        self,
        storage_path: str = "ml_pipeline_data",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the unified ML pipeline.

        Args:
            storage_path: Base path for all storage
            config: Optional configuration dictionary
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        config = config or {}

        # Initialize components
        self.feature_store = FeatureStore(
            storage_path=str(self.storage_path / "features"),
            cache_size_mb=config.get('feature_cache_mb', 100),
            enable_versioning=config.get('feature_versioning', True)
        )

        self.model_registry = ModelRegistry(
            storage_path=str(self.storage_path / "models"),
            require_approval=config.get('require_model_approval', True)
        )

        self.ab_testing = ABTestingFramework(
            confidence_level=config.get('ab_confidence_level', 0.95),
            min_sample_size=config.get('ab_min_samples', 100),
            auto_select_winner=config.get('ab_auto_select', True)
        )

        self.drift_detector = DriftDetector(
            reference_window_size=config.get('drift_ref_window', 1000),
            detection_window_size=config.get('drift_detect_window', 100),
            auto_retrain_threshold=DriftSeverity[
                config.get('drift_retrain_severity', 'HIGH')
            ]
        )

        self.online_learning: Optional[OnlineLearningPipeline] = None
        self.prediction_explainer: Optional[PredictionExplainer] = None

        self._lock = threading.RLock()

        # Register drift callback
        self.drift_detector.register_retrain_callback(self._on_drift_detected)

        logger.info(f"MLPipeline initialized at {storage_path}")

    def setup_online_learning(
        self,
        base_model: Optional[BaseEstimator] = None,
        **kwargs
    ) -> OnlineLearningPipeline:
        """
        Set up online learning pipeline.

        Args:
            base_model: Base model for online learning
            **kwargs: Additional arguments for OnlineLearningPipeline

        Returns:
            Configured OnlineLearningPipeline
        """
        self.online_learning = OnlineLearningPipeline(
            base_model=base_model,
            **kwargs
        )
        return self.online_learning

    def setup_prediction_explainer(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        X_train: Optional[np.ndarray] = None,
        **kwargs
    ) -> PredictionExplainer:
        """
        Set up prediction explainer.

        Args:
            model: Model to explain
            feature_names: List of feature names
            X_train: Training data for baseline
            **kwargs: Additional arguments

        Returns:
            Configured PredictionExplainer
        """
        self.prediction_explainer = PredictionExplainer(
            model=model,
            feature_names=feature_names,
            **kwargs
        )

        if X_train is not None:
            self.prediction_explainer.set_baseline(X_train)

        return self.prediction_explainer

    async def predict_with_explanation(
        self,
        X: np.ndarray,
        as_of_timestamp: Optional[float] = None
    ) -> PredictionResult:
        """
        Make a prediction with full explanation (async).

        Args:
            X: Feature vector
            as_of_timestamp: Optional timestamp for point-in-time features

        Returns:
            PredictionResult with explanation
        """
        deployed = self.model_registry.get_deployed_model()

        if deployed is None:
            raise ValueError("No model deployed")

        model, metadata = deployed

        if self.prediction_explainer is None:
            self.prediction_explainer = PredictionExplainer(
                model=model,
                feature_names=metadata.feature_names
            )

        # Get prediction
        result = self.prediction_explainer.explain_prediction(X, metadata.model_id)

        # Add data for drift detection
        if len(metadata.feature_names) == X.shape[-1]:
            feature_values = dict(zip(metadata.feature_names, X.flatten()))
            drift_report = self.drift_detector.add_data_point(
                feature_values,
                float(result.prediction)
            )

            if drift_report and drift_report.severity != DriftSeverity.NONE:
                logger.warning(f"Drift detected: {drift_report.severity.name}")

        return result

    def train_and_register_model(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        name: str,
        version: str,
        feature_names: List[str],
        **kwargs
    ) -> ModelMetadata:
        """
        Train a model and register it in the registry.

        Args:
            model: Model to train
            X_train: Training features
            y_train: Training labels
            name: Model name
            version: Model version
            feature_names: Feature names
            **kwargs: Additional registration arguments

        Returns:
            ModelMetadata for registered model
        """
        # Train model
        model.fit(X_train, y_train)

        # Calculate metrics
        predictions = model.predict(X_train)
        metrics = {
            'train_accuracy': float(accuracy_score(y_train, predictions)),
            'train_samples': len(y_train)
        }

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_train)
            metrics['train_log_loss'] = float(log_loss(y_train, proba))

        # Get algorithm name
        algorithm = type(model).__name__

        # Get hyperparameters
        hyperparameters = {}
        if hasattr(model, 'get_params'):
            hyperparameters = model.get_params()

        # Create training data hash
        data_hash = hashlib.md5(
            X_train.tobytes() + y_train.tobytes()
        ).hexdigest()

        # Register model
        metadata = self.model_registry.register_model(
            model=model,
            name=name,
            version=version,
            algorithm=algorithm,
            feature_names=feature_names,
            metrics=metrics,
            hyperparameters=hyperparameters,
            training_data_hash=data_hash,
            **kwargs
        )

        # Set up drift detection baseline
        self._setup_drift_baseline(X_train, predictions, model, feature_names)

        return metadata

    def _setup_drift_baseline(
        self,
        X_train: np.ndarray,
        predictions: np.ndarray,
        model: BaseEstimator,
        feature_names: List[str]
    ) -> None:
        """Set up drift detection baselines from training data."""
        # Set feature baselines
        for i, name in enumerate(feature_names):
            self.drift_detector.set_reference_data(name, X_train[:, i])

        # Set prediction baseline
        self.drift_detector.set_reference_predictions(predictions)

        # Set feature importance baseline if available
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(feature_names, model.feature_importances_))
            self.drift_detector.set_reference_feature_importance(importance)

    def start_ab_test(
        self,
        champion_model_id: str,
        challenger_model_id: str,
        test_id: Optional[str] = None,
        traffic_split: float = 0.1
    ) -> ABTestResult:
        """
        Start an A/B test between two models.

        Args:
            champion_model_id: ID of champion (control) model
            challenger_model_id: ID of challenger (treatment) model
            test_id: Optional test identifier
            traffic_split: Fraction of traffic to challenger

        Returns:
            ABTestResult for the new test
        """
        champion = self.model_registry.get_model(champion_model_id)
        challenger = self.model_registry.get_model(challenger_model_id)

        if champion is None or challenger is None:
            raise ValueError("Model(s) not found")

        test_id = test_id or f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return self.ab_testing.create_test(
            test_id=test_id,
            model_a=champion,
            model_b=challenger,
            model_a_id=champion_model_id,
            model_b_id=challenger_model_id,
            traffic_split=traffic_split
        )

    def _on_drift_detected(self) -> None:
        """Callback when significant drift is detected."""
        logger.warning("Significant drift detected - consider model retraining")

        # Could trigger automatic retraining here
        # For now, just log the event

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of all pipeline components."""
        deployed = self.model_registry.get_deployed_model()

        return {
            'deployed_model': {
                'id': deployed[1].model_id if deployed else None,
                'name': deployed[1].name if deployed else None,
                'version': deployed[1].version if deployed else None,
                'metrics': deployed[1].metrics if deployed else {}
            },
            'active_ab_tests': self.ab_testing.list_active_tests(),
            'recent_drift': [
                r.to_dict()
                for r in self.drift_detector.get_drift_history(limit=5)
            ],
            'online_learning': {
                'enabled': self.online_learning is not None,
                'samples_seen': (
                    self.online_learning.get_state().n_samples_seen
                    if self.online_learning else 0
                ),
                'performance_trend': (
                    self.online_learning.get_performance_trend()
                    if self.online_learning else {}
                )
            },
            'feature_store': {
                'registered_features': len(self.feature_store.list_features())
            },
            'model_registry': {
                'total_models': len(self.model_registry.list_models()),
                'approved_models': len(
                    self.model_registry.list_models(status=ModelStatus.APPROVED)
                )
            }
        }

    def save_pipeline_state(self, path: Optional[str] = None) -> None:
        """Save complete pipeline state to disk."""
        path = path or str(self.storage_path / "pipeline_state")

        # Save feature store
        self.feature_store.persist_features()

        # Save online learning state
        if self.online_learning:
            self.online_learning.save_state(f"{path}_online")

        logger.info(f"Pipeline state saved to {path}")

    def load_pipeline_state(self, path: Optional[str] = None) -> None:
        """Load pipeline state from disk."""
        path = path or str(self.storage_path / "pipeline_state")

        # Load feature store
        self.feature_store.load_features()

        # Load online learning state
        if self.online_learning and os.path.exists(f"{path}_online_state.json"):
            self.online_learning.load_state(f"{path}_online")

        logger.info(f"Pipeline state loaded from {path}")


# =============================================================================
# ASYNC WRAPPERS
# =============================================================================

async def async_predict(
    pipeline: MLPipeline,
    X: np.ndarray,
    **kwargs
) -> PredictionResult:
    """
    Async wrapper for pipeline prediction.

    Args:
        pipeline: MLPipeline instance
        X: Feature vector
        **kwargs: Additional arguments

    Returns:
        PredictionResult
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: pipeline.predict_with_explanation(X, **kwargs)
    )


async def async_train(
    pipeline: MLPipeline,
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    **kwargs
) -> ModelMetadata:
    """
    Async wrapper for model training.

    Args:
        pipeline: MLPipeline instance
        model: Model to train
        X_train: Training features
        y_train: Training labels
        **kwargs: Additional arguments

    Returns:
        ModelMetadata
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: pipeline.train_and_register_model(
            model, X_train, y_train, **kwargs
        )
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Demonstrate ML Pipeline usage."""
    print("=== ML Pipeline Example ===\n")

    # Initialize pipeline
    pipeline = MLPipeline(
        storage_path="example_ml_pipeline",
        config={
            'feature_versioning': True,
            'require_model_approval': False,
            'ab_min_samples': 50
        }
    )

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)

    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Register features
    print("1. Registering features...")
    for i, name in enumerate(feature_names):
        pipeline.feature_store.register_feature(
            name=name,
            feature_type=FeatureType.NUMERICAL,
            description=f"Synthetic feature {i}"
        )
        # Store feature values
        pipeline.feature_store.store_feature_values(
            feature_name=name,
            values=X[:, i],
            timestamps=np.arange(n_samples).astype(float)
        )

    print(f"   Registered {len(feature_names)} features\n")

    # Train and register model
    print("2. Training model...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)

    metadata = pipeline.train_and_register_model(
        model=model,
        X_train=X[:800],
        y_train=y[:800],
        name="trading_classifier",
        version="1.0.0",
        feature_names=feature_names,
        description="Example trading signal classifier"
    )

    print(f"   Model registered: {metadata.model_id}")
    print(f"   Training accuracy: {metadata.metrics.get('train_accuracy', 0):.4f}\n")

    # Deploy model
    print("3. Deploying model...")
    pipeline.model_registry.update_status(metadata.model_id, ModelStatus.APPROVED)
    pipeline.model_registry.deploy_model(metadata.model_id)
    print("   Model deployed!\n")

    # Set up prediction explainer
    print("4. Setting up prediction explainer...")
    pipeline.setup_prediction_explainer(
        model=model,
        feature_names=feature_names,
        X_train=X[:800]
    )
    print("   Explainer ready!\n")

    # Make explained predictions
    print("5. Making predictions with explanations...")
    for i in range(3):
        sample = X[800 + i:801 + i]
        result = pipeline.prediction_explainer.explain_prediction(sample)
        print(f"\n   Sample {i + 1}:")
        print(f"   {result.explanation}")

    # Check for drift
    print("\n\n6. Checking for drift...")
    # Add some data points
    for i in range(100):
        feature_values = dict(zip(feature_names, X[800 + i]))
        pipeline.drift_detector.add_data_point(feature_values, float(y[800 + i]))

    drift_report = pipeline.drift_detector.detect_data_drift()
    print(f"   Drift severity: {drift_report.severity.name}")
    print(f"   Recommendation: {drift_report.recommendation}\n")

    # Get pipeline status
    print("7. Pipeline status:")
    status = pipeline.get_pipeline_status()
    print(f"   Deployed model: {status['deployed_model']['name']} v{status['deployed_model']['version']}")
    print(f"   Registered features: {status['feature_store']['registered_features']}")
    print(f"   Total models: {status['model_registry']['total_models']}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    example_usage()
