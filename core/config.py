import os
import logging
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class ConfigManager:
    """Manages configuration loading and validation"""

    def __init__(self, config_path: str = "Config().yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> dict:
        """Load configuration from YAML file or create default"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            config = self._get_default_config()
            self._save_config(config)
            return config

    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'schwab': {
                'callback_url': "https://127.0.0.1",
                'token_path': "token_1.json"
            },
            'trading': {
                'ml_prediction_enabled': True,
                'max_risk_per_trade': 0.02,
                'min_risk_reward_ratio': 2.0,
                'max_daily_loss': 0.05,
                'max_consecutive_losses': 3,
                'extended_hours': {
                    'allow_premarket': False,  # DANGER: Wide spreads, low liquidity
                    'allow_afterhours': False,  # DANGER: Wide spreads, low liquidity
                    'use_limit_orders': True,  # Always use limit orders in extended hours
                    'premarket_start': "04:00",  # 4:00 AM ET
                    'market_open': "09:30",      # 9:30 AM ET
                    'market_close': "16:00",      # 4:00 PM ET
                    'afterhours_end': "20:00"    # 8:00 PM ET
                },
                'position_size_kelly_fraction': 0.25,
                'max_positions': 5,
                'reserve_cash_percent': 0.1,
                'min_position_size': 1,
                'max_position_value': 10000,
                'min_buying_power': 100,
                'limit_order_slippage': 0.001,
                'default_stop_loss_pct': 0.05,
                'default_take_profit_pct': 0.10,
                'max_position_value_pct': 0.25
            },
            'commentary': {
                'enabled': True,
                'detail_level': "verbose",
                'max_history': 100
            },
            'technical_analysis': {
                'timeframes': ['1min', '5min', '15min', '1hour', '1day'],
                'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786]
            },
            'order_management': {
                'auto_cancel_existing_orders': True,
                'require_close_confirmation': True,
                'confirm_only_losses': False,
                'confirm_threshold_percent': 5
            },
            'paths': {
                'trade_journal': "trade_journal.json",
                'model': "trading_model.pkl",
                'log': "trading_bot.log",
                'commentary_log': "trading_commentary.json",
                'backtest_results': "backtest_results.json"
            },
            'backtesting': {
                'enabled': True,
                'start_date': "2023-01-01",
                'end_date': "2024-01-01",
                'initial_capital': 100000,
                'commission': 0.005,
                'slippage': 0.001
            },
            'performance_metrics': {
                'calculate_sharpe': True,
                'calculate_sortino': True,
                'calculate_max_drawdown': True,
                'risk_free_rate': 0.02
            },
            'error_recovery': {
                'max_retries': 3,
                'retry_delay': 1.0,
                'circuit_breaker_enabled': True,
                'max_daily_loss_threshold': 0.10,
                'emergency_stop_loss': 0.15
            }
        }

    def _save_config(self, config: dict):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def _validate_config(self):
        """Validate configuration values"""
        required_sections = ['schwab', 'trading', 'commentary', 'technical_analysis']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate trading parameters are within safe ranges
        trading = self.config.get('trading', {})
        validations = [
            ('max_risk_per_trade', 0.001, 0.10, 'Max risk per trade must be 0.1%-10%'),
            ('min_risk_reward_ratio', 0.5, 10.0, 'Risk/reward ratio must be 0.5-10.0'),
            ('max_daily_loss', 0.01, 0.20, 'Max daily loss must be 1%-20%'),
            ('max_positions', 1, 50, 'Max positions must be 1-50'),
            ('reserve_cash_percent', 0.0, 0.50, 'Reserve cash must be 0%-50%'),
        ]
        for key, min_val, max_val, msg in validations:
            val = trading.get(key)
            if val is not None and not (min_val <= val <= max_val):
                raise ValueError(f"Config validation error: {msg} (got {val})")

    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def update(self, key: str, value):
        """Update configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self._save_config(self.config)

# Initialize configuration
config_manager = ConfigManager()

# ============================================================================
# TRADING LOSS CIRCUIT BREAKER
# ============================================================================
# Note: This is distinct from circuit_breaker.py which handles API failures.
# This class monitors daily P&L and halts trading on excessive losses.

class TradingLossBreaker:
    """Circuit breaker that halts trading when daily losses exceed thresholds"""

    def __init__(self, max_daily_loss: float = 0.10, emergency_stop: float = 0.15):
        self.max_daily_loss = max_daily_loss
        self.emergency_stop = emergency_stop
        self.daily_pnl = 0.0
        self.is_tripped = False
        self.trip_time = None
        self.reset_time = None

    def update_daily_pnl(self, pnl: float, account_balance: float = 100000):
        """Update daily P&L and check circuit breaker"""
        self.daily_pnl = pnl

        if self.daily_pnl >= 0:
            return

        daily_loss_ratio = abs(self.daily_pnl) / account_balance

        if daily_loss_ratio >= self.emergency_stop:
            self._trip("EMERGENCY_STOP", daily_loss_ratio)
        elif daily_loss_ratio >= self.max_daily_loss:
            self._trip("MAX_DAILY_LOSS", daily_loss_ratio)

    def _trip(self, reason: str, loss_ratio: float):
        """Trip the circuit breaker"""
        self.is_tripped = True
        self.trip_time = datetime.now()
        self.reset_time = self.trip_time + timedelta(hours=24)
        logger.critical(f"Trading loss breaker tripped: {reason} - Loss ratio: {loss_ratio:.2%}")

    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        if not self.is_tripped:
            return True
        if datetime.now() >= self.reset_time:
            self.reset()
            return True
        return False

    def reset(self):
        """Reset the circuit breaker"""
        self.is_tripped = False
        self.trip_time = None
        self.reset_time = None
        logger.info("Trading loss breaker reset")

# Backward compatibility alias
CircuitBreaker = TradingLossBreaker

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

class Config:
    """Central configuration for the trading bot"""
    # Use config manager instead of hardcoded values
    def __init__(self):
        self.manager = config_manager

    @property
    def SCHWAB_API_KEY(self):
        return os.getenv("SCHWAB_API_KEY")

    @property
    def SCHWAB_APP_SECRET(self):
        # Support both SCHWAB_SECRET and SCHWAB_APP_SECRET for compatibility
        return os.getenv("SCHWAB_SECRET") or os.getenv("SCHWAB_APP_SECRET")

    @property
    def SCHWAB_CALLBACK_URL(self):
        return self.manager.get('schwab.callback_url', 'https://127.0.0.1')

    @property
    def SCHWAB_TOKEN_PATH(self):
        return Path(self.manager.get('schwab.token_path', 'token_1.json'))

    @property
    def SCHWAB_ACCOUNT_NUMBER(self):
        return os.getenv("SCHWAB_ACCOUNT_NUMBER")

    @property
    def MAX_RISK_PER_TRADE(self):
        return self.manager.get('trading.max_risk_per_trade', 0.02)

    @property
    def MIN_RISK_REWARD_RATIO(self):
        return self.manager.get('trading.min_risk_reward_ratio', 2.0)

    @property
    def MAX_DAILY_LOSS(self):
        return self.manager.get('trading.max_daily_loss', 0.05)

    @property
    def MAX_CONSECUTIVE_LOSSES(self):
        return self.manager.get('trading.max_consecutive_losses', 3)

    @property
    def POSITION_SIZE_KELLY_FRACTION(self):
        return self.manager.get('trading.position_size_kelly_fraction', 0.25)

    @property
    def COMMENTARY_ENABLED(self):
        return self.manager.get('commentary.enabled', True)

    @property
    def COMMENTARY_DETAIL_LEVEL(self):
        return self.manager.get('commentary.detail_level', 'verbose')

    @property
    def MAX_COMMENTARY_HISTORY(self):
        return self.manager.get('commentary.max_history', 100)

    @property
    def TIMEFRAMES(self):
        return self.manager.get('technical_analysis.timeframes')

    @property
    def FIBONACCI_LEVELS(self):
        return self.manager.get('technical_analysis.fibonacci_levels')

    @property
    def TRADE_JOURNAL_PATH(self):
        return Path(self.manager.get('paths.trade_journal'))

    @property
    def MODEL_PATH(self):
        return Path(self.manager.get('paths.model'))

    @property
    def LOG_PATH(self):
        return Path(self.manager.get('paths.log'))

    @property
    def COMMENTARY_LOG_PATH(self):
        return Path(self.manager.get('paths.commentary_log'))

    @property
    def AUTO_CANCEL_EXISTING_ORDERS(self):
        return self.manager.get('order_management.auto_cancel_existing_orders')

    @property
    def MAX_POSITIONS(self):
        return self.manager.get('trading.max_positions', 5)

    @property
    def RESERVE_CASH_PERCENT(self):
        return self.manager.get('trading.reserve_cash_percent', 0.10)

    @property
    def MIN_POSITION_SIZE(self):
        return self.manager.get('trading.min_position_size', 1)

    @property
    def MAX_POSITION_VALUE(self):
        return self.manager.get('trading.max_position_value', 10000)

    @property
    def MIN_BUYING_POWER(self):
        return self.manager.get('trading.min_buying_power', 100)

    @property
    def REQUIRE_CLOSE_CONFIRMATION(self):
        return self.manager.get('order_management.require_close_confirmation', True)

    @property
    def CONFIRM_ONLY_LOSSES(self):
        return self.manager.get('order_management.confirm_only_losses', False)

    @property
    def CONFIRM_THRESHOLD_PERCENT(self):
        return self.manager.get('order_management.confirm_threshold_percent', 5)

    @property
    def ML_PREDICTION_ENABLED(self):
        return self.manager.get('trading.ml_prediction_enabled', True)

    @property
    def ML_VETO_CONFIDENCE(self) -> float:
        """ML confidence (0-1) at or above which a disagreeing ML signal vetoes
        the trade. Default 0.65 — set higher for fewer vetoes, lower for stricter."""
        return float(self.manager.get('trading.ml_veto_confidence', 0.65))

    @property
    def WARMUP_MINUTES_BEFORE_OPEN(self) -> int:
        """Minutes before regular market open (09:30 ET) to wake the bot so
        it can scan the watchlist and build indicator context before the
        first tradable bar. New entries are still blocked until regular
        hours by the signal-router gate.

        Default 30 — set to 0 to disable warmup."""
        return int(self.manager.get('trading.warmup_minutes_before_open', 30))

    @property
    def TRADING_API_KEY(self) -> Optional[str]:
        """Bearer token that protects the REST and WebSocket API endpoints.

        Returns the value of the TRADING_API_KEY environment variable, or None
        when the variable is not set (auth disabled / dev mode).
        """
        return os.getenv("TRADING_API_KEY")

    # Order execution constants
    @property
    def LIMIT_ORDER_SLIPPAGE(self):
        """Slippage factor for limit orders (e.g., 0.001 = 0.1%)"""
        return self.manager.get('trading.limit_order_slippage', 0.001)

    @property
    def DEFAULT_STOP_LOSS_PCT(self):
        """Default stop loss percentage for positions without explicit stops"""
        return self.manager.get('trading.default_stop_loss_pct', 0.05)

    @property
    def DEFAULT_TAKE_PROFIT_PCT(self):
        """Default take profit percentage for positions without explicit targets"""
        return self.manager.get('trading.default_take_profit_pct', 0.10)

    @property
    def MAX_POSITION_VALUE_PCT(self):
        """Maximum position value as fraction of portfolio"""
        return self.manager.get('trading.max_position_value_pct', 0.25)

    @property
    def ATR_STOP_MULTIPLIER(self):
        """ATR multiplier for stop-loss distance (1.5 = stop at 1.5×ATR from entry)."""
        return self.manager.get('trading.atr_stop_multiplier', 1.5)

    @property
    def ATR_REWARD_RISK_RATIO(self):
        """R:R ratio for take-profit relative to stop distance (2.0 = 2:1 R:R)."""
        return self.manager.get('trading.atr_reward_risk_ratio', 2.0)

    @property
    def RISK_PER_TRADE_PCT(self):
        """Fraction of equity risked per trade for ATR-based sizing (0.01 = 1%)."""
        return self.manager.get('trading.risk_per_trade_pct', 0.01)

# Initialize configuration
config = Config()
# ============================================================================
# LOGGING SETUP
# ============================================================================

class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for file output"""

    def format(self, record):
        import json as _json
        log_entry = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        if record.exc_info and record.exc_info[0]:
            log_entry['exception'] = self.formatException(record.exc_info)
        # Include extra fields for trade events
        for key in ('symbol', 'action', 'signal_type', 'order_id', 'price',
                     'quantity', 'pnl', 'reason'):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return _json.dumps(log_entry)


def setup_logging():
    """Configure logging with JSON file output and human-readable console output"""
    logger = logging.getLogger('TradingBot')
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on re-import
    if logger.handlers:
        return logger

    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        str(Config().LOG_PATH),
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(JSONFormatter())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()
