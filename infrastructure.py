#!/usr/bin/env python3
"""
Infrastructure Module for Institutional Trading
===============================================
Configuration management, deployment utilities, and operational tooling.

Features:
- Environment-specific configuration
- Secret management
- Feature flags
- Graceful shutdown
- Health endpoints
- Deployment utilities
"""

import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
import yaml

logger = logging.getLogger('Infrastructure')


# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

class Environment(Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    PAPER = "paper"  # Paper trading


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "trading"
    user: str = ""
    password: str = ""  # Should come from secrets
    pool_size: int = 10
    ssl_enabled: bool = True


@dataclass
class APIConfig:
    """API configuration"""
    schwab_app_key: str = ""
    schwab_app_secret: str = ""  # Should come from secrets
    schwab_callback_url: str = "https://127.0.0.1"
    schwab_token_path: str = "token_1.json"
    rate_limit_per_second: int = 10
    timeout_seconds: float = 30.0
    max_retries: int = 3


@dataclass
class TradingConfig:
    """Trading parameters configuration"""
    max_positions: int = 10
    max_position_value: float = 50000.0
    max_portfolio_percent: float = 0.20
    max_daily_loss_percent: float = 0.02
    max_daily_loss_absolute: float = 5000.0
    max_drawdown_percent: float = 0.10
    default_stop_loss_percent: float = 0.02
    default_take_profit_percent: float = 0.05
    min_position_value: float = 100.0
    use_paper_trading: bool = True


@dataclass
class RiskConfig:
    """Risk management configuration"""
    var_confidence_level: float = 0.95
    var_lookback_days: int = 252
    correlation_threshold: float = 0.80
    max_sector_exposure: float = 0.30
    max_single_stock_exposure: float = 0.10
    circuit_breaker_loss_threshold: float = 0.03
    consecutive_loss_limit: int = 5


@dataclass
class MLConfig:
    """Machine learning configuration"""
    model_path: str = "models/"
    feature_store_path: str = "feature_store/"
    retrain_threshold_days: int = 7
    min_training_samples: int = 1000
    prediction_confidence_threshold: float = 0.60
    enable_online_learning: bool = True
    drift_detection_enabled: bool = True
    drift_threshold: float = 0.10


@dataclass
class AlertingConfig:
    """Alerting configuration"""
    enabled: bool = True
    telegram_bot_token: str = ""  # From secrets
    telegram_chat_id: str = ""
    email_smtp_host: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)
    email_password: str = ""  # From secrets
    webhook_url: str = ""
    throttle_minutes: int = 5
    escalation_minutes: int = 30


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "json"  # json or text
    log_file: str = "trading_bot.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    include_timestamps: bool = True
    mask_sensitive_data: bool = True


@dataclass
class InfrastructureConfig:
    """Complete infrastructure configuration"""
    environment: Environment = Environment.DEVELOPMENT
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Feature flags
    feature_flags: Dict[str, bool] = field(default_factory=dict)


class ConfigurationManager:
    """
    Manages application configuration with environment support.
    Supports YAML files, environment variables, and secrets.
    """

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._config: Optional[InfrastructureConfig] = None
        self._secrets: Dict[str, str] = {}
        self._env = self._detect_environment()
        self._watchers: List[Callable] = []

    def _detect_environment(self) -> Environment:
        """Detect current environment from env var"""
        env_str = os.getenv('TRADING_ENV', 'development').lower()
        try:
            return Environment(env_str)
        except ValueError:
            logger.warning(f"Unknown environment '{env_str}', defaulting to development")
            return Environment.DEVELOPMENT

    def load_config(self, config_file: str = None) -> InfrastructureConfig:
        """Load configuration from file and environment"""
        if config_file is None:
            config_file = self.config_dir / f"config.{self._env.value}.yaml"
        else:
            config_file = Path(config_file)

        # Start with defaults
        config_dict = {}

        # Load from YAML if exists
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {config_file}")

        # Load secrets
        self._load_secrets()

        # Build configuration with overrides
        self._config = self._build_config(config_dict)

        # Apply environment variable overrides
        self._apply_env_overrides()

        # Validate configuration
        self._validate_config()

        return self._config

    def _load_secrets(self):
        """Load secrets from secure storage"""
        # Check for secrets file
        secrets_file = self.config_dir / "secrets.yaml"
        if secrets_file.exists():
            with open(secrets_file, 'r') as f:
                self._secrets = yaml.safe_load(f) or {}
            logger.info("Loaded secrets from file")

        # Override with environment variables (preferred for production)
        secret_keys = [
            'SCHWAB_APP_SECRET',
            'TELEGRAM_BOT_TOKEN',
            'EMAIL_PASSWORD',
            'DATABASE_PASSWORD',
        ]
        for key in secret_keys:
            value = os.getenv(key)
            if value:
                self._secrets[key.lower()] = value

    def _build_config(self, config_dict: Dict) -> InfrastructureConfig:
        """Build configuration object from dictionary"""
        config = InfrastructureConfig()
        config.environment = self._env

        # Database
        if 'database' in config_dict:
            db = config_dict['database']
            config.database = DatabaseConfig(
                host=db.get('host', config.database.host),
                port=db.get('port', config.database.port),
                database=db.get('database', config.database.database),
                user=db.get('user', config.database.user),
                password=self._secrets.get('database_password', ''),
                pool_size=db.get('pool_size', config.database.pool_size),
                ssl_enabled=db.get('ssl_enabled', config.database.ssl_enabled),
            )

        # API
        if 'api' in config_dict:
            api = config_dict['api']
            config.api = APIConfig(
                schwab_app_key=api.get('schwab_app_key', ''),
                schwab_app_secret=self._secrets.get('schwab_app_secret', ''),
                schwab_callback_url=api.get('schwab_callback_url', config.api.schwab_callback_url),
                schwab_token_path=api.get('schwab_token_path', config.api.schwab_token_path),
                rate_limit_per_second=api.get('rate_limit_per_second', config.api.rate_limit_per_second),
                timeout_seconds=api.get('timeout_seconds', config.api.timeout_seconds),
                max_retries=api.get('max_retries', config.api.max_retries),
            )

        # Trading
        if 'trading' in config_dict:
            t = config_dict['trading']
            config.trading = TradingConfig(
                max_positions=t.get('max_positions', config.trading.max_positions),
                max_position_value=t.get('max_position_value', config.trading.max_position_value),
                max_portfolio_percent=t.get('max_portfolio_percent', config.trading.max_portfolio_percent),
                max_daily_loss_percent=t.get('max_daily_loss_percent', config.trading.max_daily_loss_percent),
                max_daily_loss_absolute=t.get('max_daily_loss_absolute', config.trading.max_daily_loss_absolute),
                max_drawdown_percent=t.get('max_drawdown_percent', config.trading.max_drawdown_percent),
                default_stop_loss_percent=t.get('default_stop_loss_percent', config.trading.default_stop_loss_percent),
                default_take_profit_percent=t.get('default_take_profit_percent', config.trading.default_take_profit_percent),
                min_position_value=t.get('min_position_value', config.trading.min_position_value),
                use_paper_trading=t.get('use_paper_trading', config.trading.use_paper_trading),
            )

        # Risk
        if 'risk' in config_dict:
            r = config_dict['risk']
            config.risk = RiskConfig(
                var_confidence_level=r.get('var_confidence_level', config.risk.var_confidence_level),
                var_lookback_days=r.get('var_lookback_days', config.risk.var_lookback_days),
                correlation_threshold=r.get('correlation_threshold', config.risk.correlation_threshold),
                max_sector_exposure=r.get('max_sector_exposure', config.risk.max_sector_exposure),
                max_single_stock_exposure=r.get('max_single_stock_exposure', config.risk.max_single_stock_exposure),
                circuit_breaker_loss_threshold=r.get('circuit_breaker_loss_threshold', config.risk.circuit_breaker_loss_threshold),
                consecutive_loss_limit=r.get('consecutive_loss_limit', config.risk.consecutive_loss_limit),
            )

        # ML
        if 'ml' in config_dict:
            m = config_dict['ml']
            config.ml = MLConfig(
                model_path=m.get('model_path', config.ml.model_path),
                feature_store_path=m.get('feature_store_path', config.ml.feature_store_path),
                retrain_threshold_days=m.get('retrain_threshold_days', config.ml.retrain_threshold_days),
                min_training_samples=m.get('min_training_samples', config.ml.min_training_samples),
                prediction_confidence_threshold=m.get('prediction_confidence_threshold', config.ml.prediction_confidence_threshold),
                enable_online_learning=m.get('enable_online_learning', config.ml.enable_online_learning),
                drift_detection_enabled=m.get('drift_detection_enabled', config.ml.drift_detection_enabled),
                drift_threshold=m.get('drift_threshold', config.ml.drift_threshold),
            )

        # Alerting
        if 'alerting' in config_dict:
            a = config_dict['alerting']
            config.alerting = AlertingConfig(
                enabled=a.get('enabled', config.alerting.enabled),
                telegram_bot_token=self._secrets.get('telegram_bot_token', ''),
                telegram_chat_id=a.get('telegram_chat_id', ''),
                email_smtp_host=a.get('email_smtp_host', config.alerting.email_smtp_host),
                email_smtp_port=a.get('email_smtp_port', config.alerting.email_smtp_port),
                email_from=a.get('email_from', ''),
                email_to=a.get('email_to', []),
                email_password=self._secrets.get('email_password', ''),
                webhook_url=a.get('webhook_url', ''),
                throttle_minutes=a.get('throttle_minutes', config.alerting.throttle_minutes),
                escalation_minutes=a.get('escalation_minutes', config.alerting.escalation_minutes),
            )

        # Logging
        if 'logging' in config_dict:
            lg = config_dict['logging']
            config.logging = LoggingConfig(
                level=lg.get('level', config.logging.level),
                format=lg.get('format', config.logging.format),
                log_file=lg.get('log_file', config.logging.log_file),
                max_file_size_mb=lg.get('max_file_size_mb', config.logging.max_file_size_mb),
                backup_count=lg.get('backup_count', config.logging.backup_count),
                include_timestamps=lg.get('include_timestamps', config.logging.include_timestamps),
                mask_sensitive_data=lg.get('mask_sensitive_data', config.logging.mask_sensitive_data),
            )

        # Feature flags
        config.feature_flags = config_dict.get('feature_flags', {})

        return config

    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Trading overrides
        if os.getenv('MAX_POSITIONS'):
            self._config.trading.max_positions = int(os.getenv('MAX_POSITIONS'))
        if os.getenv('MAX_DAILY_LOSS_PERCENT'):
            self._config.trading.max_daily_loss_percent = float(os.getenv('MAX_DAILY_LOSS_PERCENT'))
        if os.getenv('USE_PAPER_TRADING'):
            self._config.trading.use_paper_trading = os.getenv('USE_PAPER_TRADING').lower() == 'true'

    def _validate_config(self):
        """Validate configuration values"""
        errors = []

        # Trading validations
        if self._config.trading.max_positions < 1:
            errors.append("max_positions must be >= 1")
        if not 0 < self._config.trading.max_portfolio_percent <= 1:
            errors.append("max_portfolio_percent must be between 0 and 1")
        if not 0 < self._config.trading.max_daily_loss_percent <= 1:
            errors.append("max_daily_loss_percent must be between 0 and 1")

        # Risk validations
        if not 0.9 <= self._config.risk.var_confidence_level < 1:
            errors.append("var_confidence_level must be between 0.9 and 1")

        # Production-specific validations
        if self._config.environment == Environment.PRODUCTION:
            if not self._config.api.schwab_app_secret:
                errors.append("schwab_app_secret required for production")
            if self._config.trading.use_paper_trading:
                logger.warning("Paper trading enabled in production environment")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

        logger.info(f"Configuration validated for {self._config.environment.value} environment")

    @property
    def config(self) -> InfrastructureConfig:
        """Get current configuration"""
        if self._config is None:
            self.load_config()
        return self._config

    def get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """Get feature flag value"""
        return self.config.feature_flags.get(flag_name, default)

    def register_change_watcher(self, callback: Callable):
        """Register callback for configuration changes"""
        self._watchers.append(callback)

    def reload_config(self):
        """Reload configuration and notify watchers"""
        old_config = self._config
        self.load_config()

        for watcher in self._watchers:
            try:
                watcher(old_config, self._config)
            except Exception as e:
                logger.error(f"Config watcher error: {e}")


# ============================================================================
# FEATURE FLAGS
# ============================================================================

class FeatureFlagManager:
    """
    Manages feature flags for gradual rollouts and A/B testing.
    """

    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self._overrides: Dict[str, bool] = {}
        self._flag_stats: Dict[str, Dict[str, int]] = {}

    def is_enabled(self, flag_name: str, default: bool = False) -> bool:
        """Check if feature flag is enabled"""
        # Check overrides first
        if flag_name in self._overrides:
            result = self._overrides[flag_name]
        else:
            result = self.config_manager.get_feature_flag(flag_name, default)

        # Track statistics
        if flag_name not in self._flag_stats:
            self._flag_stats[flag_name] = {'checks': 0, 'enabled': 0}
        self._flag_stats[flag_name]['checks'] += 1
        if result:
            self._flag_stats[flag_name]['enabled'] += 1

        return result

    def set_override(self, flag_name: str, value: bool):
        """Set a temporary override for a feature flag"""
        self._overrides[flag_name] = value
        logger.info(f"Feature flag override: {flag_name} = {value}")

    def clear_override(self, flag_name: str):
        """Clear a feature flag override"""
        if flag_name in self._overrides:
            del self._overrides[flag_name]
            logger.info(f"Feature flag override cleared: {flag_name}")

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get feature flag usage statistics"""
        return self._flag_stats.copy()


# ============================================================================
# GRACEFUL SHUTDOWN
# ============================================================================

class GracefulShutdownManager:
    """
    Manages graceful shutdown of the trading system.
    Ensures positions are handled and state is saved.
    """

    def __init__(self):
        self._shutdown_requested = False
        self._shutdown_handlers: List[Callable] = []
        self._shutdown_complete = asyncio.Event()
        self._lock = asyncio.Lock()
        self._shutdown_timeout = 60  # seconds

        # Register signal handlers
        self._register_signals()

    def _register_signals(self):
        """Register OS signal handlers"""
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._signal_handler)
            except Exception as e:
                logger.warning(f"Could not register signal {sig}: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self._shutdown_requested = True

        # Schedule async shutdown
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.shutdown())
        except RuntimeError:
            # No event loop, run synchronously
            pass

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown has been requested"""
        return self._shutdown_requested

    def register_handler(self, handler: Callable, priority: int = 50):
        """
        Register a shutdown handler.
        Lower priority numbers run first.
        """
        self._shutdown_handlers.append((priority, handler))
        self._shutdown_handlers.sort(key=lambda x: x[0])

    async def shutdown(self):
        """Execute graceful shutdown"""
        async with self._lock:
            if self._shutdown_complete.is_set():
                return

            logger.info("Starting graceful shutdown...")
            start_time = time.time()

            for priority, handler in self._shutdown_handlers:
                try:
                    logger.info(f"Running shutdown handler (priority {priority})")

                    if asyncio.iscoroutinefunction(handler):
                        await asyncio.wait_for(
                            handler(),
                            timeout=self._shutdown_timeout / len(self._shutdown_handlers)
                        )
                    else:
                        handler()

                except asyncio.TimeoutError:
                    logger.error(f"Shutdown handler timed out (priority {priority})")
                except Exception as e:
                    logger.error(f"Shutdown handler error: {e}")

            elapsed = time.time() - start_time
            logger.info(f"Graceful shutdown complete in {elapsed:.2f}s")
            self._shutdown_complete.set()

    async def wait_for_shutdown(self):
        """Wait for shutdown to complete"""
        await self._shutdown_complete.wait()


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    """

    def __init__(self, rate: float, burst: int = None):
        """
        Initialize rate limiter.

        Args:
            rate: Tokens per second
            burst: Maximum burst size (default: rate)
        """
        self.rate = rate
        self.burst = burst or int(rate)
        self._tokens = float(self.burst)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

        # Statistics
        self._total_requests = 0
        self._total_limited = 0

    async def acquire(self, tokens: int = 1, timeout: float = None) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait (None = no wait)

        Returns:
            True if tokens acquired, False if rate limited
        """
        start_time = time.monotonic()

        while True:
            async with self._lock:
                self._refill()

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._total_requests += 1
                    return True

                if timeout is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        self._total_limited += 1
                        return False
                else:
                    self._total_limited += 1
                    return False

            # Wait for tokens to refill
            wait_time = (tokens - self._tokens) / self.rate
            if timeout is not None:
                remaining = timeout - (time.monotonic() - start_time)
                wait_time = min(wait_time, remaining)

            await asyncio.sleep(min(wait_time, 0.1))

    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_update = now

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            'rate': self.rate,
            'burst': self.burst,
            'current_tokens': self._tokens,
            'total_requests': self._total_requests,
            'total_limited': self._total_limited,
            'limit_rate': self._total_limited / max(self._total_requests, 1),
        }


# ============================================================================
# DEPLOYMENT UTILITIES
# ============================================================================

@dataclass
class DeploymentInfo:
    """Information about current deployment"""
    version: str
    commit_hash: str
    deployed_at: datetime
    environment: Environment
    hostname: str
    config_checksum: str


class DeploymentManager:
    """
    Manages deployment information and health checks.
    """

    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self._deployment_info: Optional[DeploymentInfo] = None
        self._startup_time = datetime.now()
        self._ready = False
        self._live = True

    def initialize(self, version: str = "1.0.0", commit_hash: str = "unknown"):
        """Initialize deployment information"""
        import socket

        # Calculate config checksum
        config_str = json.dumps(asdict(self.config_manager.config), default=str, sort_keys=True)
        config_checksum = hashlib.sha256(config_str.encode()).hexdigest()[:12]

        self._deployment_info = DeploymentInfo(
            version=version,
            commit_hash=commit_hash,
            deployed_at=datetime.now(),
            environment=self.config_manager.config.environment,
            hostname=socket.gethostname(),
            config_checksum=config_checksum,
        )

        logger.info(f"Deployment initialized: v{version} ({commit_hash[:8]})")

    @property
    def info(self) -> DeploymentInfo:
        """Get deployment information"""
        return self._deployment_info

    def set_ready(self, ready: bool = True):
        """Set readiness status"""
        self._ready = ready
        logger.info(f"Readiness status: {ready}")

    def set_live(self, live: bool = True):
        """Set liveness status"""
        self._live = live
        logger.info(f"Liveness status: {live}")

    def is_ready(self) -> bool:
        """Check if system is ready to receive traffic"""
        return self._ready

    def is_live(self) -> bool:
        """Check if system is alive"""
        return self._live

    def get_uptime(self) -> timedelta:
        """Get system uptime"""
        return datetime.now() - self._startup_time

    def health_check(self) -> Dict[str, Any]:
        """Get health check response"""
        return {
            'status': 'healthy' if self._ready and self._live else 'unhealthy',
            'ready': self._ready,
            'live': self._live,
            'uptime_seconds': self.get_uptime().total_seconds(),
            'version': self._deployment_info.version if self._deployment_info else 'unknown',
            'environment': self.config_manager.config.environment.value,
        }

    def get_info(self) -> Dict[str, Any]:
        """Get full deployment information"""
        if not self._deployment_info:
            return {'error': 'Not initialized'}

        return {
            **asdict(self._deployment_info),
            'deployed_at': self._deployment_info.deployed_at.isoformat(),
            'environment': self._deployment_info.environment.value,
            'uptime': str(self.get_uptime()),
            'ready': self._ready,
            'live': self._live,
        }


# ============================================================================
# DEPENDENCY INJECTION CONTAINER
# ============================================================================

class DependencyContainer:
    """
    Simple dependency injection container for managing service instances.
    """

    def __init__(self):
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Set[str] = set()

    def register(self, name: str, factory: Callable, singleton: bool = True):
        """Register a service factory"""
        self._factories[name] = factory
        if singleton:
            self._singletons.add(name)

    def register_instance(self, name: str, instance: Any):
        """Register an existing instance"""
        self._instances[name] = instance
        self._singletons.add(name)

    def get(self, name: str) -> Any:
        """Get a service instance"""
        # Check for existing singleton instance
        if name in self._instances:
            return self._instances[name]

        # Create new instance
        if name not in self._factories:
            raise KeyError(f"Service '{name}' not registered")

        instance = self._factories[name]()

        # Store singleton
        if name in self._singletons:
            self._instances[name] = instance

        return instance

    def has(self, name: str) -> bool:
        """Check if service is registered"""
        return name in self._factories or name in self._instances


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Environment
    'Environment',

    # Config
    'DatabaseConfig',
    'APIConfig',
    'TradingConfig',
    'RiskConfig',
    'MLConfig',
    'AlertingConfig',
    'LoggingConfig',
    'InfrastructureConfig',
    'ConfigurationManager',

    # Feature Flags
    'FeatureFlagManager',

    # Shutdown
    'GracefulShutdownManager',

    # Rate Limiting
    'RateLimiter',

    # Deployment
    'DeploymentInfo',
    'DeploymentManager',

    # DI
    'DependencyContainer',
]
