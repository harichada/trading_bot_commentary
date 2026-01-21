#!/usr/bin/env python3
"""
Comprehensive Observability System for Trading Bot

This module provides production-grade observability including:
- Structured JSON logging with correlation IDs and sensitive data masking
- Metrics collection (Counter, Gauge, Histogram) with time-series storage
- Multi-channel alerting (Telegram, Email, Webhook) with throttling
- Immutable audit trail with event sourcing and tamper-evident hashing
- Dashboard data provider for real-time and historical metrics

Author: Trading Bot Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import smtplib
import sys
import time
import threading
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum, auto
from functools import wraps
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from statistics import mean, median, stdev
from typing import (
    Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Type, Union,
    TypeVar, Generic, Protocol, AsyncIterator
)

import aiohttp

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


# =============================================================================
# TYPE DEFINITIONS AND PROTOCOLS
# =============================================================================

T = TypeVar('T')
MetricValue = Union[int, float]


class MetricCallback(Protocol):
    """Protocol for metric change callbacks."""
    def __call__(self, name: str, value: MetricValue, labels: Dict[str, str]) -> None: ...


class AlertCallback(Protocol):
    """Protocol for alert callbacks."""
    async def __call__(self, alert: 'Alert') -> bool: ...


# =============================================================================
# ENUMERATIONS
# =============================================================================

class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class AlertSeverity(Enum):
    """Alert severity levels with escalation priority."""
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3

    @property
    def requires_acknowledgment(self) -> bool:
        return self.value >= AlertSeverity.CRITICAL.value


class MetricType(Enum):
    """Types of metrics supported."""
    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()
    SUMMARY = auto()


class AuditEventType(Enum):
    """Types of auditable events."""
    # Trading events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_MODIFIED = "position_modified"

    # Risk events
    RISK_LIMIT_BREACH = "risk_limit_breach"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    MARGIN_CALL = "margin_call"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"

    # System events
    SYSTEM_START = "system_start"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_CHANGED = "config_changed"
    PARAMETER_UPDATED = "parameter_updated"
    MODEL_UPDATED = "model_updated"

    # Manual interventions
    MANUAL_OVERRIDE = "manual_override"
    MANUAL_TRADE = "manual_trade"
    EMERGENCY_STOP = "emergency_stop"

    # Strategy events
    STRATEGY_ENABLED = "strategy_enabled"
    STRATEGY_DISABLED = "strategy_disabled"
    SIGNAL_GENERATED = "signal_generated"

    # Data events
    DATA_ANOMALY = "data_anomaly"
    DATA_RECOVERY = "data_recovery"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CorrelationContext:
    """Context for request tracing with correlation IDs."""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    service_name: str = "trading_bot"
    trace_start: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def child_context(self, operation: str) -> 'CorrelationContext':
        """Create a child context for nested operations."""
        return CorrelationContext(
            correlation_id=self.correlation_id,
            parent_id=self.span_id,
            service_name=self.service_name,
            metadata={**self.metadata, 'operation': operation}
        )


@dataclass
class LogEntry:
    """Structured log entry with full context."""
    timestamp: datetime
    level: LogLevel
    message: str
    correlation_id: str
    logger_name: str
    module: str
    function: str
    line_number: int
    extra: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    duration_ms: Optional[float] = None

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.name,
            'message': self.message,
            'correlation_id': self.correlation_id,
            'logger': self.logger_name,
            'location': {
                'module': self.module,
                'function': self.function,
                'line': self.line_number
            }
        }
        if self.extra:
            data['extra'] = self.extra
        if self.exception:
            data['exception'] = self.exception
        if self.stack_trace:
            data['stack_trace'] = self.stack_trace
        if self.duration_ms is not None:
            data['duration_ms'] = self.duration_ms
        return json.dumps(data)


@dataclass
class MetricSample:
    """Single metric sample with timestamp and labels."""
    timestamp: datetime
    value: MetricValue
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class HistogramBucket:
    """Histogram bucket for distribution tracking."""
    le: float  # Less than or equal to
    count: int = 0


@dataclass
class Alert:
    """Alert with full context and tracking."""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    source: str
    timestamp: datetime
    correlation_id: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[MetricValue] = None
    threshold: Optional[MetricValue] = None
    labels: Dict[str, str] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    escalation_level: int = 0
    notification_count: int = 0
    last_notification: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'severity': self.severity.name,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'labels': self.labels,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'escalation_level': self.escalation_level,
            'notification_count': self.notification_count
        }


@dataclass
class AuditEvent:
    """Immutable audit event with cryptographic verification."""
    id: str
    event_type: AuditEventType
    timestamp: datetime
    actor: str
    action: str
    resource: str
    details: Dict[str, Any]
    correlation_id: Optional[str] = None
    previous_hash: Optional[str] = None
    hash: str = field(default="")

    def __post_init__(self):
        """Calculate hash after initialization."""
        if not self.hash:
            self.hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the event."""
        data = {
            'id': self.id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'actor': self.actor,
            'action': self.action,
            'resource': self.resource,
            'details': self.details,
            'correlation_id': self.correlation_id,
            'previous_hash': self.previous_hash
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify the event hash matches its content."""
        return self.hash == self._calculate_hash()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'actor': self.actor,
            'action': self.action,
            'resource': self.resource,
            'details': self.details,
            'correlation_id': self.correlation_id,
            'previous_hash': self.previous_hash,
            'hash': self.hash
        }


@dataclass
class EscalationPolicy:
    """Policy for alert escalation."""
    name: str
    severity_threshold: AlertSeverity
    initial_delay_seconds: int = 300  # 5 minutes
    escalation_interval_seconds: int = 600  # 10 minutes
    max_escalations: int = 3
    channels: List[str] = field(default_factory=list)
    notify_on_resolve: bool = True


# =============================================================================
# SENSITIVE DATA MASKING
# =============================================================================

class SensitiveDataMasker:
    """Masks sensitive data in log messages and dictionaries."""

    # Patterns to mask
    SENSITIVE_PATTERNS = [
        (re.compile(r'(password|passwd|pwd)["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.I), r'\1=***MASKED***'),
        (re.compile(r'(api[_-]?key|apikey)["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.I), r'\1=***MASKED***'),
        (re.compile(r'(secret|token|auth)["\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', re.I), r'\1=***MASKED***'),
        (re.compile(r'(bearer\s+)([a-zA-Z0-9._-]+)', re.I), r'\1***MASKED***'),
        (re.compile(r'\b([A-Z0-9]{20,})\b'), r'***API_KEY***'),  # Long alphanumeric strings
        (re.compile(r'(\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})'), r'***CARD***'),  # Credit card
        (re.compile(r'(\d{3}[-.]?\d{2}[-.]?\d{4})'), r'***SSN***'),  # SSN
        (re.compile(r'(account[_-]?(?:number|num|id))["\']?\s*[:=]\s*["\']?(\d+)', re.I), r'\1=***MASKED***'),
    ]

    # Keys to mask in dictionaries
    SENSITIVE_KEYS = {
        'password', 'passwd', 'pwd', 'secret', 'token', 'api_key', 'apikey',
        'api-key', 'auth', 'authorization', 'credentials', 'private_key',
        'privatekey', 'access_token', 'refresh_token', 'client_secret',
        'account_number', 'routing_number', 'ssn', 'credit_card', 'cvv'
    }

    @classmethod
    def mask_string(cls, text: str) -> str:
        """Mask sensitive data in a string."""
        if not isinstance(text, str):
            return text

        result = text
        for pattern, replacement in cls.SENSITIVE_PATTERNS:
            result = pattern.sub(replacement, result)
        return result

    @classmethod
    def mask_dict(cls, data: Dict[str, Any], depth: int = 0, max_depth: int = 10) -> Dict[str, Any]:
        """Recursively mask sensitive data in a dictionary."""
        if depth > max_depth:
            return {'__truncated__': 'max depth exceeded'}

        result = {}
        for key, value in data.items():
            key_lower = key.lower().replace('-', '_')

            if key_lower in cls.SENSITIVE_KEYS:
                result[key] = '***MASKED***'
            elif isinstance(value, dict):
                result[key] = cls.mask_dict(value, depth + 1, max_depth)
            elif isinstance(value, list):
                result[key] = [
                    cls.mask_dict(item, depth + 1, max_depth) if isinstance(item, dict)
                    else cls.mask_string(str(item)) if isinstance(item, str)
                    else item
                    for item in value
                ]
            elif isinstance(value, str):
                result[key] = cls.mask_string(value)
            else:
                result[key] = value

        return result


# =============================================================================
# STRUCTURED JSON LOGGING
# =============================================================================

class JsonLogFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, mask_sensitive: bool = True):
        super().__init__()
        self.mask_sensitive = mask_sensitive
        self.masker = SensitiveDataMasker()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Build base log entry
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'location': {
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'pathname': record.pathname
            }
        }

        # Add correlation ID if present
        correlation_id = getattr(record, 'correlation_id', None)
        if correlation_id:
            log_data['correlation_id'] = correlation_id

        # Add span information if present
        span_id = getattr(record, 'span_id', None)
        if span_id:
            log_data['span_id'] = span_id

        # Add duration if present
        duration_ms = getattr(record, 'duration_ms', None)
        if duration_ms is not None:
            log_data['duration_ms'] = duration_ms

        # Add extra fields
        extra = getattr(record, 'extra', {})
        if extra:
            log_data['extra'] = extra

        # Add exception info
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }

        # Mask sensitive data
        if self.mask_sensitive:
            log_data = self.masker.mask_dict(log_data)
            if isinstance(log_data.get('message'), str):
                log_data['message'] = self.masker.mask_string(log_data['message'])

        return json.dumps(log_data)


class CorrelatedLogger:
    """Logger wrapper that automatically includes correlation context."""

    def __init__(self, logger: logging.Logger, context: Optional[CorrelationContext] = None):
        self._logger = logger
        self._context = context

    @property
    def context(self) -> Optional[CorrelationContext]:
        return self._context

    @context.setter
    def context(self, value: CorrelationContext):
        self._context = value

    def _log(self, level: int, msg: str, *args, **kwargs):
        """Internal logging method with correlation context."""
        extra = kwargs.pop('extra', {})

        if self._context:
            extra['correlation_id'] = self._context.correlation_id
            extra['span_id'] = self._context.span_id
            if self._context.parent_id:
                extra['parent_span_id'] = self._context.parent_id

        kwargs['extra'] = extra
        self._logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        kwargs['exc_info'] = True
        self._log(logging.ERROR, msg, *args, **kwargs)


class ObservabilityLogger:
    """
    Central logging manager with structured JSON output, rotation, and correlation.

    Usage:
        logger = ObservabilityLogger.get_logger("trading_engine")
        with logger.correlation_context() as ctx:
            logger.info("Processing order", extra={'order_id': '123'})
    """

    _instance: Optional['ObservabilityLogger'] = None
    _lock = threading.Lock()
    _loggers: Dict[str, CorrelatedLogger] = {}
    _context_var: Optional[CorrelationContext] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        log_dir: str = "logs",
        log_file: str = "trading_bot.json.log",
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 10,
        level: int = logging.INFO,
        mask_sensitive: bool = True
    ):
        if hasattr(self, '_initialized'):
            return

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / log_file
        self.mask_sensitive = mask_sensitive

        # Create root handler with JSON formatting
        self.json_formatter = JsonLogFormatter(mask_sensitive=mask_sensitive)

        # Rotating file handler
        self.file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        self.file_handler.setFormatter(self.json_formatter)
        self.file_handler.setLevel(level)

        # Console handler (optional, human-readable)
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        )
        self.console_handler.setLevel(level)

        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.addHandler(self.file_handler)

        self._initialized = True

    @classmethod
    def get_logger(cls, name: str) -> CorrelatedLogger:
        """Get or create a correlated logger by name."""
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = CorrelatedLogger(logger)
        return cls._loggers[name]

    @classmethod
    @contextmanager
    def correlation_context(
        cls,
        operation: Optional[str] = None,
        parent_context: Optional[CorrelationContext] = None
    ):
        """Context manager for correlation tracking."""
        if parent_context:
            context = parent_context.child_context(operation or "child")
        else:
            context = CorrelationContext(
                metadata={'operation': operation} if operation else {}
            )

        old_context = cls._context_var
        cls._context_var = context

        # Update all loggers with new context
        for logger in cls._loggers.values():
            logger.context = context

        try:
            yield context
        finally:
            cls._context_var = old_context
            for logger in cls._loggers.values():
                logger.context = old_context

    @classmethod
    def current_context(cls) -> Optional[CorrelationContext]:
        """Get the current correlation context."""
        return cls._context_var


def timed_operation(operation_name: str):
    """Decorator to log operation duration."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = ObservabilityLogger.get_logger(func.__module__)
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    f"Operation '{operation_name}' completed",
                    extra={'duration_ms': duration_ms, 'operation': operation_name}
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"Operation '{operation_name}' failed: {e}",
                    extra={'duration_ms': duration_ms, 'operation': operation_name}
                )
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = ObservabilityLogger.get_logger(func.__module__)
            start_time = time.perf_counter()

            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    f"Operation '{operation_name}' completed",
                    extra={'duration_ms': duration_ms, 'operation': operation_name}
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"Operation '{operation_name}' failed: {e}",
                    extra={'duration_ms': duration_ms, 'operation': operation_name}
                )
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# =============================================================================
# METRICS COLLECTION
# =============================================================================

class BaseMetric(ABC):
    """Base class for all metrics."""

    def __init__(
        self,
        name: str,
        description: str,
        labels: List[str] = None,
        retention_hours: int = 24
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.retention_hours = retention_hours
        self._samples: Dict[str, deque] = defaultdict(lambda: deque(maxlen=86400))  # 1 day at 1s intervals
        self._callbacks: List[MetricCallback] = []
        self._lock = threading.RLock()

    def _label_key(self, labels: Dict[str, str]) -> str:
        """Generate a unique key for a label combination."""
        if not labels:
            return "__default__"
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def _validate_labels(self, labels: Dict[str, str]):
        """Validate that provided labels match expected label names."""
        provided = set(labels.keys())
        expected = set(self.label_names)
        if provided != expected:
            raise ValueError(f"Labels mismatch. Expected {expected}, got {provided}")

    def add_callback(self, callback: MetricCallback):
        """Add a callback to be called on metric changes."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, value: MetricValue, labels: Dict[str, str]):
        """Notify all callbacks of a metric change."""
        for callback in self._callbacks:
            try:
                callback(self.name, value, labels)
            except Exception:
                pass  # Don't let callback errors affect metrics

    def _add_sample(self, value: MetricValue, labels: Dict[str, str]):
        """Add a sample to the time series."""
        key = self._label_key(labels)
        sample = MetricSample(timestamp=datetime.now(), value=value, labels=labels)
        with self._lock:
            self._samples[key].append(sample)
        self._notify_callbacks(value, labels)

    def _prune_old_samples(self):
        """Remove samples older than retention period."""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        with self._lock:
            for key in list(self._samples.keys()):
                while self._samples[key] and self._samples[key][0].timestamp < cutoff:
                    self._samples[key].popleft()

    def get_samples(
        self,
        labels: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MetricSample]:
        """Get samples optionally filtered by labels and time range."""
        with self._lock:
            if labels:
                key = self._label_key(labels)
                samples = list(self._samples.get(key, []))
            else:
                samples = []
                for sample_list in self._samples.values():
                    samples.extend(sample_list)

        # Filter by time range
        if start_time:
            samples = [s for s in samples if s.timestamp >= start_time]
        if end_time:
            samples = [s for s in samples if s.timestamp <= end_time]

        return sorted(samples, key=lambda s: s.timestamp)

    @abstractmethod
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> MetricValue:
        """Get current metric value."""
        pass


class Counter(BaseMetric):
    """
    Monotonically increasing counter metric.

    Usage:
        orders = Counter("orders_total", "Total orders", ["status"])
        orders.inc({"status": "filled"})
        orders.inc({"status": "rejected"}, amount=5)
    """

    def __init__(self, name: str, description: str, labels: List[str] = None, **kwargs):
        super().__init__(name, description, labels, **kwargs)
        self._values: Dict[str, float] = defaultdict(float)

    def inc(self, labels: Optional[Dict[str, str]] = None, amount: float = 1.0):
        """Increment the counter."""
        labels = labels or {}
        if self.label_names:
            self._validate_labels(labels)

        key = self._label_key(labels)
        with self._lock:
            self._values[key] += amount
            value = self._values[key]

        self._add_sample(value, labels)

    def get_value(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        key = self._label_key(labels or {})
        with self._lock:
            return self._values.get(key, 0.0)

    def reset(self, labels: Optional[Dict[str, str]] = None):
        """Reset counter (use sparingly - counters should be monotonic)."""
        key = self._label_key(labels or {})
        with self._lock:
            self._values[key] = 0.0


class Gauge(BaseMetric):
    """
    Metric that can increase and decrease.

    Usage:
        exposure = Gauge("portfolio_exposure", "Current exposure", ["asset"])
        exposure.set({"asset": "SPY"}, 10000.0)
        exposure.inc({"asset": "SPY"}, 500.0)
    """

    def __init__(self, name: str, description: str, labels: List[str] = None, **kwargs):
        super().__init__(name, description, labels, **kwargs)
        self._values: Dict[str, float] = defaultdict(float)

    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Set the gauge value."""
        labels = labels or {}
        if self.label_names:
            self._validate_labels(labels)

        key = self._label_key(labels)
        with self._lock:
            self._values[key] = value

        self._add_sample(value, labels)

    def inc(self, labels: Optional[Dict[str, str]] = None, amount: float = 1.0):
        """Increment the gauge."""
        labels = labels or {}
        if self.label_names:
            self._validate_labels(labels)

        key = self._label_key(labels)
        with self._lock:
            self._values[key] += amount
            value = self._values[key]

        self._add_sample(value, labels)

    def dec(self, labels: Optional[Dict[str, str]] = None, amount: float = 1.0):
        """Decrement the gauge."""
        self.inc(labels, -amount)

    def get_value(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        key = self._label_key(labels or {})
        with self._lock:
            return self._values.get(key, 0.0)


class Histogram(BaseMetric):
    """
    Distribution metric with configurable buckets.

    Usage:
        latency = Histogram("order_latency_ms", "Order latency",
                           buckets=[10, 50, 100, 250, 500, 1000])
        latency.observe(45.5)
    """

    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(
        self,
        name: str,
        description: str,
        labels: List[str] = None,
        buckets: List[float] = None,
        **kwargs
    ):
        super().__init__(name, description, labels, **kwargs)
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._bucket_counts: Dict[str, Dict[float, int]] = defaultdict(
            lambda: {b: 0 for b in self.buckets + [float('inf')]}
        )
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        self._values: Dict[str, List[float]] = defaultdict(list)

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Record an observation."""
        labels = labels or {}
        if self.label_names:
            self._validate_labels(labels)

        key = self._label_key(labels)
        with self._lock:
            self._sums[key] += value
            self._counts[key] += 1
            self._values[key].append(value)

            # Keep only last 10000 values for percentile calculation
            if len(self._values[key]) > 10000:
                self._values[key] = self._values[key][-10000:]

            for bucket in self.buckets + [float('inf')]:
                if value <= bucket:
                    self._bucket_counts[key][bucket] += 1

        self._add_sample(value, labels)

    def get_value(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get the sum of all observations."""
        key = self._label_key(labels or {})
        with self._lock:
            return self._sums.get(key, 0.0)

    def get_count(self, labels: Optional[Dict[str, str]] = None) -> int:
        """Get the count of observations."""
        key = self._label_key(labels or {})
        with self._lock:
            return self._counts.get(key, 0)

    def get_mean(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get the mean of observations."""
        key = self._label_key(labels or {})
        with self._lock:
            count = self._counts.get(key, 0)
            if count == 0:
                return 0.0
            return self._sums.get(key, 0.0) / count

    def get_percentile(self, percentile: float, labels: Optional[Dict[str, str]] = None) -> float:
        """Get a percentile value (0-100)."""
        key = self._label_key(labels or {})
        with self._lock:
            values = self._values.get(key, [])
            if not values:
                return 0.0
            sorted_values = sorted(values)
            index = int(len(sorted_values) * percentile / 100)
            return sorted_values[min(index, len(sorted_values) - 1)]

    def get_buckets(self, labels: Optional[Dict[str, str]] = None) -> Dict[float, int]:
        """Get bucket counts."""
        key = self._label_key(labels or {})
        with self._lock:
            return dict(self._bucket_counts.get(key, {}))


class MetricsRegistry:
    """
    Central registry for all metrics with convenient access patterns.

    Usage:
        registry = MetricsRegistry()
        registry.counter("orders_total", "Total orders", ["status"])
        registry.inc("orders_total", {"status": "filled"})
    """

    _instance: Optional['MetricsRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._metrics: Dict[str, BaseMetric] = {}
        self._lock = threading.RLock()
        self._initialized = True

        # Initialize default trading metrics
        self._init_default_metrics()

    def _init_default_metrics(self):
        """Initialize default trading bot metrics."""
        # Order metrics
        self.counter("orders_submitted_total", "Total orders submitted", ["symbol", "side", "type"])
        self.counter("orders_filled_total", "Total orders filled", ["symbol", "side"])
        self.counter("orders_cancelled_total", "Total orders cancelled", ["symbol", "reason"])
        self.counter("orders_rejected_total", "Total orders rejected", ["symbol", "reason"])

        # P&L metrics
        self.gauge("pnl_realized_total", "Total realized P&L", ["strategy"])
        self.gauge("pnl_unrealized_total", "Total unrealized P&L", ["strategy"])
        self.gauge("pnl_daily", "Daily P&L")

        # Position metrics
        self.gauge("positions_count", "Number of open positions")
        self.gauge("position_value", "Position value", ["symbol"])
        self.gauge("portfolio_exposure", "Total portfolio exposure")

        # Risk metrics
        self.gauge("risk_var_95", "95% Value at Risk")
        self.gauge("risk_max_drawdown", "Maximum drawdown")
        self.gauge("risk_current_drawdown", "Current drawdown")
        self.gauge("risk_sharpe_ratio", "Sharpe ratio")

        # Latency metrics
        self.histogram("order_latency_ms", "Order execution latency", buckets=[10, 25, 50, 100, 250, 500, 1000, 2500])
        self.histogram("data_latency_ms", "Market data latency", buckets=[1, 5, 10, 25, 50, 100, 250])
        self.histogram("signal_latency_ms", "Signal generation latency", buckets=[1, 5, 10, 25, 50, 100])

        # System metrics
        self.gauge("system_cpu_percent", "CPU usage percentage")
        self.gauge("system_memory_percent", "Memory usage percentage")
        self.gauge("system_connections_active", "Active connections")

        # Tick data metrics
        self.counter("ticks_received_total", "Total ticks received", ["symbol"])
        self.gauge("tick_rate_per_second", "Ticks per second", ["symbol"])
        self.histogram("tick_spread_bps", "Tick spread in basis points", ["symbol"], buckets=[1, 2, 5, 10, 25, 50, 100])

    def register(self, metric: BaseMetric):
        """Register a metric."""
        with self._lock:
            if metric.name in self._metrics:
                raise ValueError(f"Metric '{metric.name}' already registered")
            self._metrics[metric.name] = metric

    def get(self, name: str) -> Optional[BaseMetric]:
        """Get a metric by name."""
        return self._metrics.get(name)

    def counter(self, name: str, description: str, labels: List[str] = None, **kwargs) -> Counter:
        """Create and register a counter."""
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            counter = Counter(name, description, labels, **kwargs)
            self._metrics[name] = counter
            return counter

    def gauge(self, name: str, description: str, labels: List[str] = None, **kwargs) -> Gauge:
        """Create and register a gauge."""
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            gauge = Gauge(name, description, labels, **kwargs)
            self._metrics[name] = gauge
            return gauge

    def histogram(self, name: str, description: str, labels: List[str] = None, **kwargs) -> Histogram:
        """Create and register a histogram."""
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            histogram = Histogram(name, description, labels, **kwargs)
            self._metrics[name] = histogram
            return histogram

    def inc(self, name: str, labels: Optional[Dict[str, str]] = None, amount: float = 1.0):
        """Increment a counter or gauge."""
        metric = self._metrics.get(name)
        if metric and hasattr(metric, 'inc'):
            metric.inc(labels, amount)

    def set(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge value."""
        metric = self._metrics.get(name)
        if metric and isinstance(metric, Gauge):
            metric.set(value, labels)

    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram observation."""
        metric = self._metrics.get(name)
        if metric and isinstance(metric, Histogram):
            metric.observe(value, labels)

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics with their current values."""
        result = {}
        with self._lock:
            for name, metric in self._metrics.items():
                if isinstance(metric, Histogram):
                    result[name] = {
                        'type': 'histogram',
                        'description': metric.description,
                        'count': metric.get_count(),
                        'sum': metric.get_value(),
                        'mean': metric.get_mean(),
                        'p50': metric.get_percentile(50),
                        'p90': metric.get_percentile(90),
                        'p99': metric.get_percentile(99)
                    }
                else:
                    result[name] = {
                        'type': 'counter' if isinstance(metric, Counter) else 'gauge',
                        'description': metric.description,
                        'value': metric.get_value()
                    }
        return result


# =============================================================================
# MULTI-CHANNEL ALERTING
# =============================================================================

class AlertChannel(ABC):
    """Base class for alert notification channels."""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.logger = ObservabilityLogger.get_logger(f"alert.{name}")

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send an alert through this channel."""
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the channel is properly configured."""
        pass


class TelegramChannel(AlertChannel):
    """Telegram bot alert channel."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
        parse_mode: str = "HTML"
    ):
        super().__init__("telegram", enabled)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.parse_mode = parse_mode
        self.api_url = f"https://api.telegram.org/bot{bot_token}"

    def _format_message(self, alert: Alert) -> str:
        """Format alert for Telegram."""
        severity_emoji = {
            AlertSeverity.INFO: "INFO",
            AlertSeverity.WARNING: "WARNING",
            AlertSeverity.CRITICAL: "CRITICAL",
            AlertSeverity.EMERGENCY: "EMERGENCY"
        }

        emoji = severity_emoji.get(alert.severity, "")

        message = f"<b>[{emoji}] {alert.title}</b>\n\n"
        message += f"{alert.message}\n\n"
        message += f"<b>Source:</b> {alert.source}\n"
        message += f"<b>Time:</b> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"

        if alert.metric_name:
            message += f"<b>Metric:</b> {alert.metric_name} = {alert.metric_value}\n"
        if alert.threshold:
            message += f"<b>Threshold:</b> {alert.threshold}\n"
        if alert.labels:
            message += f"<b>Labels:</b> {json.dumps(alert.labels)}\n"

        message += f"\n<code>Alert ID: {alert.id}</code>"

        return message

    async def send(self, alert: Alert) -> bool:
        """Send alert via Telegram."""
        if not self.enabled:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    'chat_id': self.chat_id,
                    'text': self._format_message(alert),
                    'parse_mode': self.parse_mode
                }

                async with session.post(
                    f"{self.api_url}/sendMessage",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Alert sent to Telegram: {alert.id}")
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Telegram API error: {error_text}")
                        return False
        except Exception as e:
            self.logger.error(f"Failed to send Telegram alert: {e}")
            return False

    async def test_connection(self) -> bool:
        """Test Telegram connection."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/getMe",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except Exception:
            return False


class EmailChannel(AlertChannel):
    """SMTP email alert channel."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str],
        use_tls: bool = True,
        enabled: bool = True
    ):
        super().__init__("email", enabled)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.use_tls = use_tls

    def _create_message(self, alert: Alert) -> MIMEMultipart:
        """Create email message from alert."""
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[{alert.severity.name}] {alert.title}"
        msg['From'] = self.from_addr
        msg['To'] = ", ".join(self.to_addrs)

        # Plain text version
        text = f"""
Alert: {alert.title}
Severity: {alert.severity.name}
Source: {alert.source}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

{alert.message}

Metric: {alert.metric_name or 'N/A'}
Value: {alert.metric_value or 'N/A'}
Threshold: {alert.threshold or 'N/A'}
Labels: {json.dumps(alert.labels) if alert.labels else 'N/A'}

Alert ID: {alert.id}
Correlation ID: {alert.correlation_id or 'N/A'}
"""

        # HTML version
        html = f"""
<html>
<body>
<h2 style="color: {'red' if alert.severity.value >= 2 else 'orange' if alert.severity.value == 1 else 'blue'};">
    [{alert.severity.name}] {alert.title}
</h2>
<p><strong>Source:</strong> {alert.source}</p>
<p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
<hr>
<p>{alert.message}</p>
<hr>
<table>
    <tr><td><strong>Metric:</strong></td><td>{alert.metric_name or 'N/A'}</td></tr>
    <tr><td><strong>Value:</strong></td><td>{alert.metric_value or 'N/A'}</td></tr>
    <tr><td><strong>Threshold:</strong></td><td>{alert.threshold or 'N/A'}</td></tr>
</table>
<hr>
<p><small>Alert ID: {alert.id}</small></p>
</body>
</html>
"""

        msg.attach(MIMEText(text, 'plain'))
        msg.attach(MIMEText(html, 'html'))

        return msg

    async def send(self, alert: Alert) -> bool:
        """Send alert via email."""
        if not self.enabled:
            return False

        try:
            # Run SMTP in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._send_sync, alert)
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return False

    def _send_sync(self, alert: Alert) -> bool:
        """Synchronous email send."""
        try:
            msg = self._create_message(alert)

            if self.use_tls:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
                server.starttls()
            else:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)

            server.login(self.username, self.password)
            server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            server.quit()

            self.logger.info(f"Alert sent via email: {alert.id}")
            return True
        except Exception as e:
            self.logger.error(f"SMTP error: {e}")
            return False

    async def test_connection(self) -> bool:
        """Test SMTP connection."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._test_sync)
        except Exception:
            return False

    def _test_sync(self) -> bool:
        """Synchronous connection test."""
        try:
            if self.use_tls:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=5)
                server.starttls()
            else:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=5)
            server.login(self.username, self.password)
            server.quit()
            return True
        except Exception:
            return False


class WebhookChannel(AlertChannel):
    """Generic webhook alert channel."""

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        method: str = "POST",
        enabled: bool = True
    ):
        super().__init__("webhook", enabled)
        self.url = url
        self.headers = headers or {'Content-Type': 'application/json'}
        self.method = method.upper()

    async def send(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        if not self.enabled:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                payload = alert.to_dict()

                async with session.request(
                    self.method,
                    self.url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status in (200, 201, 202, 204):
                        self.logger.info(f"Alert sent via webhook: {alert.id}")
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Webhook error ({response.status}): {error_text}")
                        return False
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
            return False

    async def test_connection(self) -> bool:
        """Test webhook endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.head(
                    self.url,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status < 500
        except Exception:
            return False


class AlertManager:
    """
    Central alert management with throttling, grouping, and escalation.

    Usage:
        manager = AlertManager()
        manager.add_channel(TelegramChannel(token, chat_id))
        await manager.alert(
            AlertSeverity.CRITICAL,
            "High Drawdown",
            "Portfolio drawdown exceeded 5%",
            source="risk_manager"
        )
    """

    _instance: Optional['AlertManager'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        throttle_seconds: int = 300,  # 5 minutes default throttle
        max_alerts_per_hour: int = 60,
        enable_grouping: bool = True
    ):
        if self._initialized:
            return

        self._channels: Dict[str, AlertChannel] = {}
        self._escalation_policies: Dict[str, EscalationPolicy] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=10000)
        self._throttle_cache: Dict[str, datetime] = {}
        self._hourly_alert_count: int = 0
        self._hour_start: datetime = datetime.now()

        self.throttle_seconds = throttle_seconds
        self.max_alerts_per_hour = max_alerts_per_hour
        self.enable_grouping = enable_grouping

        self._lock = threading.RLock()
        self.logger = ObservabilityLogger.get_logger("alert_manager")
        self._initialized = True

        # Start escalation checker
        self._escalation_task: Optional[asyncio.Task] = None

    def add_channel(self, channel: AlertChannel):
        """Add a notification channel."""
        self._channels[channel.name] = channel
        self.logger.info(f"Added alert channel: {channel.name}")

    def remove_channel(self, name: str):
        """Remove a notification channel."""
        if name in self._channels:
            del self._channels[name]

    def add_escalation_policy(self, policy: EscalationPolicy):
        """Add an escalation policy."""
        self._escalation_policies[policy.name] = policy

    def _should_throttle(self, alert_key: str) -> bool:
        """Check if alert should be throttled."""
        now = datetime.now()

        # Check hourly limit
        if now - self._hour_start > timedelta(hours=1):
            self._hourly_alert_count = 0
            self._hour_start = now

        if self._hourly_alert_count >= self.max_alerts_per_hour:
            return True

        # Check individual throttle
        if alert_key in self._throttle_cache:
            last_alert = self._throttle_cache[alert_key]
            if (now - last_alert).total_seconds() < self.throttle_seconds:
                return True

        return False

    def _get_alert_key(self, severity: AlertSeverity, title: str, source: str) -> str:
        """Generate a unique key for throttling."""
        return f"{severity.name}:{source}:{title}"

    async def alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        metric_name: Optional[str] = None,
        metric_value: Optional[MetricValue] = None,
        threshold: Optional[MetricValue] = None,
        labels: Optional[Dict[str, str]] = None,
        correlation_id: Optional[str] = None,
        bypass_throttle: bool = False
    ) -> Optional[str]:
        """
        Create and send an alert.

        Returns alert ID if sent, None if throttled.
        """
        alert_key = self._get_alert_key(severity, title, source)

        # Check throttling
        if not bypass_throttle and self._should_throttle(alert_key):
            self.logger.debug(f"Alert throttled: {alert_key}")
            return None

        # Create alert
        alert = Alert(
            id=str(uuid.uuid4()),
            severity=severity,
            title=title,
            message=message,
            source=source,
            timestamp=datetime.now(),
            correlation_id=correlation_id or ObservabilityLogger.current_context().correlation_id if ObservabilityLogger.current_context() else None,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            labels=labels or {}
        )

        # Update throttle cache
        with self._lock:
            self._throttle_cache[alert_key] = alert.timestamp
            self._hourly_alert_count += 1
            self._active_alerts[alert.id] = alert
            self._alert_history.append(alert)

        # Send to all enabled channels
        tasks = []
        for channel in self._channels.values():
            if channel.enabled:
                tasks.append(channel.send(alert))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            alert.notification_count = success_count
            alert.last_notification = datetime.now()

            self.logger.info(
                f"Alert {alert.id} sent to {success_count}/{len(tasks)} channels",
                extra={'alert_id': alert.id, 'severity': severity.name}
            )

        return alert.id

    async def acknowledge(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id not in self._active_alerts:
                return False

            alert = self._active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()

        self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True

    async def resolve(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id not in self._active_alerts:
                return False

            alert = self._active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()

        self.logger.info(f"Alert {alert_id} resolved")

        # Send resolution notification if policy requires it
        for policy in self._escalation_policies.values():
            if policy.notify_on_resolve:
                for channel_name in policy.channels:
                    if channel_name in self._channels:
                        resolution_alert = Alert(
                            id=str(uuid.uuid4()),
                            severity=AlertSeverity.INFO,
                            title=f"Resolved: {alert.title}",
                            message=f"Alert has been resolved.\n\nOriginal: {alert.message}",
                            source=alert.source,
                            timestamp=datetime.now()
                        )
                        await self._channels[channel_name].send(resolution_alert)

        return True

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            return [a for a in self._active_alerts.values() if not a.resolved]

    def get_alert_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get alert history with optional filters."""
        with self._lock:
            alerts = list(self._alert_history)

        if start_time:
            alerts = [a for a in alerts if a.timestamp >= start_time]
        if end_time:
            alerts = [a for a in alerts if a.timestamp <= end_time]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts


# =============================================================================
# IMMUTABLE AUDIT TRAIL
# =============================================================================

class AuditTrail:
    """
    Immutable audit trail with event sourcing and tamper-evident hashing.

    Every event is cryptographically linked to its predecessor, forming
    a chain that can be verified for integrity.

    Usage:
        audit = AuditTrail()
        audit.log_event(
            AuditEventType.ORDER_SUBMITTED,
            actor="trading_engine",
            action="submit_order",
            resource="order:12345",
            details={"symbol": "AAPL", "quantity": 100}
        )
    """

    _instance: Optional['AuditTrail'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        storage_path: str = "audit_trail.jsonl",
        max_memory_events: int = 100000
    ):
        if self._initialized:
            return

        self.storage_path = Path(storage_path)
        self._events: deque = deque(maxlen=max_memory_events)
        self._last_hash: Optional[str] = None
        self._lock = threading.RLock()
        self.logger = ObservabilityLogger.get_logger("audit_trail")
        self._initialized = True

        # Load existing events to get last hash
        self._load_last_hash()

    def _load_last_hash(self):
        """Load the last hash from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_event = json.loads(lines[-1])
                        self._last_hash = last_event.get('hash')
            except Exception as e:
                self.logger.error(f"Failed to load last hash: {e}")

    def log_event(
        self,
        event_type: AuditEventType,
        actor: str,
        action: str,
        resource: str,
        details: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> AuditEvent:
        """Log an audit event."""
        with self._lock:
            event = AuditEvent(
                id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=datetime.now(),
                actor=actor,
                action=action,
                resource=resource,
                details=SensitiveDataMasker.mask_dict(details),
                correlation_id=correlation_id or (
                    ObservabilityLogger.current_context().correlation_id
                    if ObservabilityLogger.current_context() else None
                ),
                previous_hash=self._last_hash
            )

            self._events.append(event)
            self._last_hash = event.hash

            # Persist to storage
            self._persist_event(event)

        self.logger.debug(f"Audit event logged: {event_type.value}")
        return event

    def _persist_event(self, event: AuditEvent):
        """Persist event to storage."""
        try:
            with open(self.storage_path, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to persist audit event: {e}")

    def verify_chain_integrity(self) -> Tuple[bool, List[str]]:
        """
        Verify the integrity of the entire audit chain.

        Returns (is_valid, list_of_errors)
        """
        errors = []

        try:
            with open(self.storage_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            return True, []  # No events yet

        previous_hash = None

        for i, line in enumerate(lines):
            try:
                data = json.loads(line)

                # Recreate event to verify hash
                event = AuditEvent(
                    id=data['id'],
                    event_type=AuditEventType(data['event_type']),
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    actor=data['actor'],
                    action=data['action'],
                    resource=data['resource'],
                    details=data['details'],
                    correlation_id=data.get('correlation_id'),
                    previous_hash=data.get('previous_hash'),
                    hash=""  # Will be calculated
                )

                # Verify hash
                if event.hash != data['hash']:
                    errors.append(f"Event {i}: Hash mismatch (possible tampering)")

                # Verify chain link
                if event.previous_hash != previous_hash:
                    errors.append(f"Event {i}: Chain link broken")

                previous_hash = data['hash']

            except Exception as e:
                errors.append(f"Event {i}: Parse error - {e}")

        return len(errors) == 0, errors

    def query(
        self,
        event_type: Optional[AuditEventType] = None,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        correlation_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Query audit events with filters."""
        results = []

        with self._lock:
            for event in reversed(self._events):
                if len(results) >= limit:
                    break

                if event_type and event.event_type != event_type:
                    continue
                if actor and event.actor != actor:
                    continue
                if resource and resource not in event.resource:
                    continue
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                if correlation_id and event.correlation_id != correlation_id:
                    continue

                results.append(event)

        return results

    def get_events_for_resource(self, resource: str) -> List[AuditEvent]:
        """Get all events related to a specific resource."""
        return self.query(resource=resource)

    def export_to_json(self, output_path: str, start_time: Optional[datetime] = None):
        """Export audit trail to JSON file."""
        events = self.query(start_time=start_time, limit=1000000)

        with open(output_path, 'w') as f:
            json.dump([e.to_dict() for e in events], f, indent=2)


# Convenience functions for common audit events
def audit_order_submitted(order_id: str, symbol: str, side: str, quantity: int, price: float, **kwargs):
    """Log order submission."""
    AuditTrail().log_event(
        AuditEventType.ORDER_SUBMITTED,
        actor="trading_engine",
        action="submit_order",
        resource=f"order:{order_id}",
        details={
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            **kwargs
        }
    )


def audit_order_filled(order_id: str, fill_price: float, fill_quantity: int, **kwargs):
    """Log order fill."""
    AuditTrail().log_event(
        AuditEventType.ORDER_FILLED,
        actor="trading_engine",
        action="order_filled",
        resource=f"order:{order_id}",
        details={
            "fill_price": fill_price,
            "fill_quantity": fill_quantity,
            **kwargs
        }
    )


def audit_risk_breach(breach_type: str, current_value: float, limit_value: float, **kwargs):
    """Log risk limit breach."""
    AuditTrail().log_event(
        AuditEventType.RISK_LIMIT_BREACH,
        actor="risk_manager",
        action="breach_detected",
        resource=f"risk:{breach_type}",
        details={
            "breach_type": breach_type,
            "current_value": current_value,
            "limit_value": limit_value,
            **kwargs
        }
    )


def audit_manual_intervention(user: str, action: str, reason: str, **kwargs):
    """Log manual intervention."""
    AuditTrail().log_event(
        AuditEventType.MANUAL_OVERRIDE,
        actor=user,
        action=action,
        resource="manual_intervention",
        details={
            "reason": reason,
            **kwargs
        }
    )


def audit_config_change(parameter: str, old_value: Any, new_value: Any, changed_by: str):
    """Log configuration change."""
    AuditTrail().log_event(
        AuditEventType.CONFIG_CHANGED,
        actor=changed_by,
        action="update_config",
        resource=f"config:{parameter}",
        details={
            "parameter": parameter,
            "old_value": old_value,
            "new_value": new_value
        }
    )


# =============================================================================
# DASHBOARD DATA PROVIDER
# =============================================================================

class DashboardDataProvider:
    """
    Provides data for dashboards with real-time and historical queries.

    Supports WebSocket streaming and REST-style queries.

    Usage:
        provider = DashboardDataProvider()

        # Get current metrics
        metrics = provider.get_current_metrics()

        # Get historical data
        history = provider.get_metric_history("pnl_daily", hours=24)

        # Subscribe to updates
        async for update in provider.subscribe():
            print(update)
    """

    _instance: Optional['DashboardDataProvider'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.metrics = MetricsRegistry()
        self.alerts = AlertManager()
        self.audit = AuditTrail()

        self._subscribers: List[asyncio.Queue] = []
        self._lock = threading.RLock()
        self.logger = ObservabilityLogger.get_logger("dashboard")
        self._initialized = True

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get all current metric values."""
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics.get_all_metrics()
        }

    def get_metric_history(
        self,
        metric_name: str,
        hours: int = 24,
        labels: Optional[Dict[str, str]] = None,
        aggregation: str = 'raw'
    ) -> Dict[str, Any]:
        """
        Get historical metric data.

        Args:
            metric_name: Name of the metric
            hours: Hours of history to retrieve
            labels: Label filter
            aggregation: 'raw', 'minute', 'hour', 'day'
        """
        metric = self.metrics.get(metric_name)
        if not metric:
            return {'error': f"Metric '{metric_name}' not found"}

        start_time = datetime.now() - timedelta(hours=hours)
        samples = metric.get_samples(labels=labels, start_time=start_time)

        if aggregation == 'raw':
            data = [
                {'timestamp': s.timestamp.isoformat(), 'value': s.value}
                for s in samples
            ]
        else:
            data = self._aggregate_samples(samples, aggregation)

        return {
            'metric': metric_name,
            'description': metric.description,
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'aggregation': aggregation,
            'data': data
        }

    def _aggregate_samples(
        self,
        samples: List[MetricSample],
        aggregation: str
    ) -> List[Dict[str, Any]]:
        """Aggregate samples by time bucket."""
        if not samples:
            return []

        # Determine bucket size
        if aggregation == 'minute':
            bucket_format = '%Y-%m-%d %H:%M'
        elif aggregation == 'hour':
            bucket_format = '%Y-%m-%d %H'
        elif aggregation == 'day':
            bucket_format = '%Y-%m-%d'
        else:
            return [{'timestamp': s.timestamp.isoformat(), 'value': s.value} for s in samples]

        # Group by bucket
        buckets: Dict[str, List[float]] = defaultdict(list)
        for sample in samples:
            bucket_key = sample.timestamp.strftime(bucket_format)
            buckets[bucket_key].append(sample.value)

        # Aggregate
        result = []
        for bucket_key, values in sorted(buckets.items()):
            result.append({
                'timestamp': bucket_key,
                'min': min(values),
                'max': max(values),
                'mean': mean(values),
                'count': len(values)
            })

        return result

    def get_trading_summary(self) -> Dict[str, Any]:
        """Get trading summary for dashboard."""
        return {
            'timestamp': datetime.now().isoformat(),
            'pnl': {
                'realized': self.metrics.get('pnl_realized_total').get_value() if self.metrics.get('pnl_realized_total') else 0,
                'unrealized': self.metrics.get('pnl_unrealized_total').get_value() if self.metrics.get('pnl_unrealized_total') else 0,
                'daily': self.metrics.get('pnl_daily').get_value() if self.metrics.get('pnl_daily') else 0
            },
            'orders': {
                'submitted': self.metrics.get('orders_submitted_total').get_value() if self.metrics.get('orders_submitted_total') else 0,
                'filled': self.metrics.get('orders_filled_total').get_value() if self.metrics.get('orders_filled_total') else 0,
                'cancelled': self.metrics.get('orders_cancelled_total').get_value() if self.metrics.get('orders_cancelled_total') else 0
            },
            'risk': {
                'var_95': self.metrics.get('risk_var_95').get_value() if self.metrics.get('risk_var_95') else 0,
                'max_drawdown': self.metrics.get('risk_max_drawdown').get_value() if self.metrics.get('risk_max_drawdown') else 0,
                'current_drawdown': self.metrics.get('risk_current_drawdown').get_value() if self.metrics.get('risk_current_drawdown') else 0
            },
            'latency': {
                'order_p50': self.metrics.get('order_latency_ms').get_percentile(50) if self.metrics.get('order_latency_ms') else 0,
                'order_p99': self.metrics.get('order_latency_ms').get_percentile(99) if self.metrics.get('order_latency_ms') else 0,
                'data_p50': self.metrics.get('data_latency_ms').get_percentile(50) if self.metrics.get('data_latency_ms') else 0
            },
            'active_alerts': len(self.alerts.get_active_alerts())
        }

    def get_alerts_summary(self) -> Dict[str, Any]:
        """Get alerts summary for dashboard."""
        active = self.alerts.get_active_alerts()

        by_severity = defaultdict(int)
        for alert in active:
            by_severity[alert.severity.name] += 1

        return {
            'timestamp': datetime.now().isoformat(),
            'active_count': len(active),
            'by_severity': dict(by_severity),
            'recent_alerts': [a.to_dict() for a in active[:10]]
        }

    def get_audit_summary(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get audit trail summary."""
        start_time = datetime.now() - timedelta(hours=hours)
        events = self.audit.query(start_time=start_time)

        by_type = defaultdict(int)
        by_actor = defaultdict(int)

        for event in events:
            by_type[event.event_type.value] += 1
            by_actor[event.actor] += 1

        return {
            'timestamp': datetime.now().isoformat(),
            'period_hours': hours,
            'total_events': len(events),
            'by_type': dict(by_type),
            'by_actor': dict(by_actor),
            'recent_events': [e.to_dict() for e in events[:20]]
        }

    async def subscribe(self) -> AsyncIterator[Dict[str, Any]]:
        """Subscribe to real-time updates."""
        queue: asyncio.Queue = asyncio.Queue()

        with self._lock:
            self._subscribers.append(queue)

        try:
            while True:
                update = await queue.get()
                yield update
        finally:
            with self._lock:
                self._subscribers.remove(queue)

    async def publish_update(self, update_type: str, data: Dict[str, Any]):
        """Publish update to all subscribers."""
        message = {
            'type': update_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }

        with self._lock:
            for queue in self._subscribers:
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    pass  # Skip slow consumers

    async def start_periodic_updates(self, interval_seconds: int = 1):
        """Start publishing periodic metric updates."""
        while True:
            try:
                summary = self.get_trading_summary()
                await self.publish_update('metrics', summary)
            except Exception as e:
                self.logger.error(f"Error publishing update: {e}")

            await asyncio.sleep(interval_seconds)


# =============================================================================
# OBSERVABILITY FACADE
# =============================================================================

class Observability:
    """
    Unified facade for all observability features.

    Usage:
        obs = Observability()

        # Logging with correlation
        with obs.trace("process_order"):
            obs.logger.info("Processing order")
            obs.metrics.inc("orders_total", {"status": "processing"})

        # Alert
        await obs.alert(AlertSeverity.WARNING, "High Latency", "...")

        # Audit
        obs.audit_event(AuditEventType.ORDER_SUBMITTED, ...)
    """

    _instance: Optional['Observability'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        log_dir: str = "logs",
        audit_path: str = "audit_trail.jsonl"
    ):
        if self._initialized:
            return

        # Initialize components
        self._logging = ObservabilityLogger(log_dir=log_dir)
        self._metrics = MetricsRegistry()
        self._alerts = AlertManager()
        self._audit = AuditTrail(storage_path=audit_path)
        self._dashboard = DashboardDataProvider()

        self._initialized = True

    @property
    def logger(self) -> CorrelatedLogger:
        """Get the main logger."""
        return ObservabilityLogger.get_logger("trading_bot")

    def get_logger(self, name: str) -> CorrelatedLogger:
        """Get a named logger."""
        return ObservabilityLogger.get_logger(name)

    @property
    def metrics(self) -> MetricsRegistry:
        """Get the metrics registry."""
        return self._metrics

    @property
    def alerts(self) -> AlertManager:
        """Get the alert manager."""
        return self._alerts

    @property
    def audit(self) -> AuditTrail:
        """Get the audit trail."""
        return self._audit

    @property
    def dashboard(self) -> DashboardDataProvider:
        """Get the dashboard data provider."""
        return self._dashboard

    @contextmanager
    def trace(self, operation: str):
        """Context manager for distributed tracing."""
        with ObservabilityLogger.correlation_context(operation) as ctx:
            start_time = time.perf_counter()
            try:
                yield ctx
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.logger.info(
                    f"Trace '{operation}' completed",
                    extra={'duration_ms': duration_ms, 'operation': operation}
                )

    async def alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str = "trading_bot",
        **kwargs
    ) -> Optional[str]:
        """Send an alert."""
        return await self._alerts.alert(severity, title, message, source, **kwargs)

    def audit_event(
        self,
        event_type: AuditEventType,
        actor: str,
        action: str,
        resource: str,
        details: Dict[str, Any]
    ) -> AuditEvent:
        """Log an audit event."""
        return self._audit.log_event(event_type, actor, action, resource, details)

    def record_latency(self, metric_name: str, latency_ms: float, labels: Optional[Dict[str, str]] = None):
        """Record a latency measurement."""
        self._metrics.observe(metric_name, latency_ms, labels)

    def record_order(self, status: str, symbol: str, side: str, order_type: str = "market"):
        """Record an order metric."""
        if status == "submitted":
            self._metrics.inc("orders_submitted_total", {"symbol": symbol, "side": side, "type": order_type})
        elif status == "filled":
            self._metrics.inc("orders_filled_total", {"symbol": symbol, "side": side})
        elif status == "cancelled":
            self._metrics.inc("orders_cancelled_total", {"symbol": symbol, "reason": "user"})
        elif status == "rejected":
            self._metrics.inc("orders_rejected_total", {"symbol": symbol, "reason": "unknown"})

    def update_pnl(self, realized: float, unrealized: float, strategy: str = "default"):
        """Update P&L metrics."""
        self._metrics.set("pnl_realized_total", realized, {"strategy": strategy})
        self._metrics.set("pnl_unrealized_total", unrealized, {"strategy": strategy})

    def update_risk_metrics(
        self,
        var_95: float,
        max_drawdown: float,
        current_drawdown: float,
        sharpe_ratio: float
    ):
        """Update risk metrics."""
        self._metrics.set("risk_var_95", var_95)
        self._metrics.set("risk_max_drawdown", max_drawdown)
        self._metrics.set("risk_current_drawdown", current_drawdown)
        self._metrics.set("risk_sharpe_ratio", sharpe_ratio)

    def update_system_metrics(self, cpu_percent: float, memory_percent: float, connections: int):
        """Update system metrics."""
        self._metrics.set("system_cpu_percent", cpu_percent)
        self._metrics.set("system_memory_percent", memory_percent)
        self._metrics.set("system_connections_active", connections)


# =============================================================================
# FACTORY AND CONFIGURATION
# =============================================================================

@dataclass
class ObservabilityConfig:
    """Configuration for observability system."""
    # Logging
    log_dir: str = "logs"
    log_level: str = "INFO"
    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_count: int = 10
    mask_sensitive_data: bool = True

    # Metrics
    metrics_retention_hours: int = 24

    # Alerts
    alert_throttle_seconds: int = 300
    alert_max_per_hour: int = 60

    # Telegram
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Email
    email_enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)

    # Webhook
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)

    # Audit
    audit_path: str = "audit_trail.jsonl"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ObservabilityConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> 'ObservabilityConfig':
        """Load config from YAML file."""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data.get('observability', {}))


def create_observability(config: ObservabilityConfig) -> Observability:
    """Factory function to create configured observability system."""
    # Initialize main observability
    obs = Observability(
        log_dir=config.log_dir,
        audit_path=config.audit_path
    )

    # Configure alert channels
    if config.telegram_enabled and config.telegram_bot_token:
        obs.alerts.add_channel(TelegramChannel(
            bot_token=config.telegram_bot_token,
            chat_id=config.telegram_chat_id
        ))

    if config.email_enabled and config.smtp_host:
        obs.alerts.add_channel(EmailChannel(
            smtp_host=config.smtp_host,
            smtp_port=config.smtp_port,
            username=config.smtp_username,
            password=config.smtp_password,
            from_addr=config.email_from,
            to_addrs=config.email_to
        ))

    if config.webhook_enabled and config.webhook_url:
        obs.alerts.add_channel(WebhookChannel(
            url=config.webhook_url,
            headers=config.webhook_headers
        ))

    # Add default escalation policy
    obs.alerts.add_escalation_policy(EscalationPolicy(
        name="default",
        severity_threshold=AlertSeverity.CRITICAL,
        initial_delay_seconds=300,
        escalation_interval_seconds=600,
        channels=["telegram", "email"],
        notify_on_resolve=True
    ))

    return obs


# =============================================================================
# GLOBAL INSTANCE ACCESS
# =============================================================================

def get_observability() -> Observability:
    """Get the global observability instance."""
    return Observability()


def get_logger(name: str = "trading_bot") -> CorrelatedLogger:
    """Get a correlated logger by name."""
    return ObservabilityLogger.get_logger(name)


def get_metrics() -> MetricsRegistry:
    """Get the metrics registry."""
    return MetricsRegistry()


def get_alerts() -> AlertManager:
    """Get the alert manager."""
    return AlertManager()


def get_audit() -> AuditTrail:
    """Get the audit trail."""
    return AuditTrail()


def get_dashboard() -> DashboardDataProvider:
    """Get the dashboard data provider."""
    return DashboardDataProvider()


# =============================================================================
# MAIN ENTRY POINT FOR TESTING
# =============================================================================

async def main():
    """Test the observability system."""
    print("Initializing Observability System...")

    # Create with default config
    obs = Observability()

    # Test logging with correlation
    print("\n--- Testing Logging ---")
    with obs.trace("test_operation"):
        obs.logger.info("Starting test operation")
        obs.logger.warning("This is a warning", extra={'test_key': 'test_value'})

        # Nested trace
        with obs.trace("nested_operation"):
            obs.logger.debug("Inside nested operation")

    # Test metrics
    print("\n--- Testing Metrics ---")
    obs.metrics.inc("orders_submitted_total", {"symbol": "AAPL", "side": "buy", "type": "market"})
    obs.metrics.inc("orders_filled_total", {"symbol": "AAPL", "side": "buy"})
    obs.metrics.set("pnl_daily", 1250.50)
    obs.metrics.observe("order_latency_ms", 45.5)
    obs.metrics.observe("order_latency_ms", 52.3)
    obs.metrics.observe("order_latency_ms", 38.1)

    print(f"Orders submitted: {obs.metrics.get('orders_submitted_total').get_value()}")
    print(f"Order latency p50: {obs.metrics.get('order_latency_ms').get_percentile(50):.2f}ms")
    print(f"Order latency p99: {obs.metrics.get('order_latency_ms').get_percentile(99):.2f}ms")

    # Test audit trail
    print("\n--- Testing Audit Trail ---")
    obs.audit_event(
        AuditEventType.ORDER_SUBMITTED,
        actor="trading_engine",
        action="submit_order",
        resource="order:12345",
        details={
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
            "price": 150.25
        }
    )

    obs.audit_event(
        AuditEventType.ORDER_FILLED,
        actor="exchange",
        action="fill_order",
        resource="order:12345",
        details={
            "fill_price": 150.20,
            "fill_quantity": 100
        }
    )

    # Verify chain integrity
    is_valid, errors = obs.audit.verify_chain_integrity()
    print(f"Audit chain integrity: {'VALID' if is_valid else 'INVALID'}")
    if errors:
        for error in errors:
            print(f"  Error: {error}")

    # Test dashboard data
    print("\n--- Testing Dashboard Data ---")
    summary = obs.dashboard.get_trading_summary()
    print(f"Trading Summary: {json.dumps(summary, indent=2)}")

    # Test sensitive data masking
    print("\n--- Testing Sensitive Data Masking ---")
    test_data = {
        "user": "trader1",
        "password": "secret123",
        "api_key": "sk_live_abc123xyz",
        "order": {"symbol": "AAPL", "quantity": 100}
    }
    masked = SensitiveDataMasker.mask_dict(test_data)
    print(f"Masked data: {json.dumps(masked, indent=2)}")

    print("\n--- Observability System Test Complete ---")


if __name__ == "__main__":
    asyncio.run(main())
