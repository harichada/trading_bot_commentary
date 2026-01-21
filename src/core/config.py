"""
Unified Configuration System

All configuration in one place with validation and type safety.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode enumeration"""
    SIMULATION = "simulation"
    PAPER = "paper"
    LIVE = "live"


class PositionSizingMethod(Enum):
    """Position sizing methods"""
    FIXED_AMOUNT = "fixed_amount"
    FIXED_PERCENTAGE = "fixed_percentage"
    KELLY_CRITERION = "kelly"
    VOLATILITY_ADJUSTED = "volatility"


@dataclass
class BrokerConfig:
    """Broker connection configuration"""
    api_key: str = ""
    api_secret: str = ""
    account_id: str = ""
    token_path: str = "token_1.json"
    callback_url: str = "https://127.0.0.1:8182/callback"

    @classmethod
    def from_env(cls) -> 'BrokerConfig':
        """Load from environment variables"""
        return cls(
            api_key=os.getenv("SCHWAB_API_KEY", ""),
            api_secret=os.getenv("SCHWAB_API_SECRET", ""),
            account_id=os.getenv("SCHWAB_ACCOUNT_ID", ""),
            token_path=os.getenv("SCHWAB_TOKEN_PATH", "token_1.json"),
        )


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size: float = 0.10  # 10% of portfolio
    max_portfolio_risk: float = 0.02  # 2% total risk
    max_daily_loss: float = 0.05  # 5% daily loss limit
    max_drawdown: float = 0.15  # 15% max drawdown
    max_positions: int = 5
    max_position_value: float = 10000.0
    min_position_size: int = 1
    min_buying_power: float = 100.0

    # Stop loss / Take profit defaults
    default_stop_loss_pct: float = 0.02  # 2%
    default_take_profit_pct: float = 0.04  # 4%
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 0.015  # 1.5%

    # Circuit breakers
    max_consecutive_losses: int = 3
    circuit_breaker_enabled: bool = True
    emergency_stop_loss_pct: float = 0.10  # 10%

    # Position sizing
    sizing_method: PositionSizingMethod = PositionSizingMethod.FIXED_PERCENTAGE


@dataclass
class StrategyConfig:
    """Strategy configuration"""
    enabled_strategies: List[str] = field(default_factory=lambda: [
        "momentum", "mean_reversion", "breakout"
    ])
    min_consensus: int = 2  # Minimum strategies agreeing
    ml_enabled: bool = False
    ml_confidence_threshold: float = 0.60

    # Strategy-specific params
    momentum_lookback: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    bollinger_period: int = 20
    bollinger_std: float = 2.0


@dataclass
class TradingConfig:
    """Trading session configuration"""
    mode: TradingMode = TradingMode.SIMULATION
    watchlist: List[str] = field(default_factory=lambda: [
        "NVDA", "TSLA", "AAPL", "MSFT", "GOOGL"
    ])

    # Trading hours
    allow_premarket: bool = False
    allow_afterhours: bool = False
    use_limit_orders: bool = True

    # Analysis settings
    analysis_interval_seconds: int = 30
    price_update_interval_seconds: int = 5

    # Order settings
    require_confirmations: bool = True
    manual_close_only: bool = True
    confirm_only_losses: bool = True
    confirm_threshold_pct: float = 1.0


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    log_dir: str = "logs"
    max_file_size_mb: int = 10
    backup_count: int = 5
    log_trades: bool = True
    log_decisions: bool = True
    log_performance: bool = True


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 9000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class Config:
    """Master configuration class"""
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)

    _instance: Optional['Config'] = None

    @classmethod
    def get_instance(cls) -> 'Config':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls.load()
        return cls._instance

    @classmethod
    def load(cls, config_path: str = "config/config.yaml") -> 'Config':
        """Load configuration from YAML file"""
        config = cls()

        path = Path(config_path)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = yaml.safe_load(f) or {}

                config = cls._from_dict(data)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        else:
            logger.info(f"No config file found at {config_path}, using defaults")

        # Override with environment variables
        config = cls._apply_env_overrides(config)

        return config

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create config from dictionary"""
        config = cls()

        if 'broker' in data:
            config.broker = BrokerConfig(**data['broker'])
        if 'risk' in data:
            risk_data = data['risk'].copy()
            if 'sizing_method' in risk_data:
                risk_data['sizing_method'] = PositionSizingMethod(risk_data['sizing_method'])
            config.risk = RiskConfig(**risk_data)
        if 'strategy' in data:
            config.strategy = StrategyConfig(**data['strategy'])
        if 'trading' in data:
            trading_data = data['trading'].copy()
            if 'mode' in trading_data:
                trading_data['mode'] = TradingMode(trading_data['mode'])
            config.trading = TradingConfig(**trading_data)
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])
        if 'api' in data:
            config.api = APIConfig(**data['api'])

        return config

    @classmethod
    def _apply_env_overrides(cls, config: 'Config') -> 'Config':
        """Apply environment variable overrides"""
        # Broker overrides
        if os.getenv("SCHWAB_API_KEY"):
            config.broker.api_key = os.getenv("SCHWAB_API_KEY")
        if os.getenv("SCHWAB_API_SECRET"):
            config.broker.api_secret = os.getenv("SCHWAB_API_SECRET")
        if os.getenv("SCHWAB_ACCOUNT_ID"):
            config.broker.account_id = os.getenv("SCHWAB_ACCOUNT_ID")

        # Trading mode override
        if os.getenv("TRADING_MODE"):
            config.trading.mode = TradingMode(os.getenv("TRADING_MODE"))

        # API overrides
        if os.getenv("API_PORT"):
            config.api.port = int(os.getenv("API_PORT"))

        return config

    def save(self, config_path: str = "config/config.yaml"):
        """Save configuration to YAML file"""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved configuration to {config_path}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'broker': {
                'api_key': self.broker.api_key,
                'api_secret': self.broker.api_secret,
                'account_id': self.broker.account_id,
                'token_path': self.broker.token_path,
                'callback_url': self.broker.callback_url,
            },
            'risk': {
                'max_position_size': self.risk.max_position_size,
                'max_portfolio_risk': self.risk.max_portfolio_risk,
                'max_daily_loss': self.risk.max_daily_loss,
                'max_drawdown': self.risk.max_drawdown,
                'max_positions': self.risk.max_positions,
                'max_position_value': self.risk.max_position_value,
                'min_position_size': self.risk.min_position_size,
                'default_stop_loss_pct': self.risk.default_stop_loss_pct,
                'default_take_profit_pct': self.risk.default_take_profit_pct,
                'trailing_stop_enabled': self.risk.trailing_stop_enabled,
                'max_consecutive_losses': self.risk.max_consecutive_losses,
                'circuit_breaker_enabled': self.risk.circuit_breaker_enabled,
                'sizing_method': self.risk.sizing_method.value,
            },
            'strategy': {
                'enabled_strategies': self.strategy.enabled_strategies,
                'min_consensus': self.strategy.min_consensus,
                'ml_enabled': self.strategy.ml_enabled,
                'ml_confidence_threshold': self.strategy.ml_confidence_threshold,
            },
            'trading': {
                'mode': self.trading.mode.value,
                'watchlist': self.trading.watchlist,
                'allow_premarket': self.trading.allow_premarket,
                'allow_afterhours': self.trading.allow_afterhours,
                'use_limit_orders': self.trading.use_limit_orders,
                'analysis_interval_seconds': self.trading.analysis_interval_seconds,
                'require_confirmations': self.trading.require_confirmations,
                'manual_close_only': self.trading.manual_close_only,
            },
            'logging': {
                'level': self.logging.level,
                'log_dir': self.logging.log_dir,
                'log_trades': self.logging.log_trades,
            },
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'debug': self.api.debug,
            }
        }

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        # Risk validation
        if self.risk.max_position_size <= 0 or self.risk.max_position_size > 1:
            errors.append("max_position_size must be between 0 and 1")
        if self.risk.max_daily_loss <= 0 or self.risk.max_daily_loss > 1:
            errors.append("max_daily_loss must be between 0 and 1")
        if self.risk.max_positions < 1:
            errors.append("max_positions must be at least 1")

        # Trading validation
        if not self.trading.watchlist:
            errors.append("watchlist cannot be empty")

        # Broker validation for live mode
        if self.trading.mode == TradingMode.LIVE:
            if not self.broker.api_key:
                errors.append("api_key required for live trading")
            if not self.broker.api_secret:
                errors.append("api_secret required for live trading")

        return errors


# Convenience function
def get_config() -> Config:
    """Get the global configuration instance"""
    return Config.get_instance()
