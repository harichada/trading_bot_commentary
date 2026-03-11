"""Tests for configuration management."""
import pytest
import yaml
import tempfile
import os
from pathlib import Path


class TestConfigManager:
    """Test ConfigManager loading and validation."""

    def test_default_config_created(self, tmp_path):
        """ConfigManager creates a default config file if none exists."""
        from core.config import ConfigManager
        config_path = tmp_path / "test_config.yaml"
        mgr = ConfigManager(config_path=str(config_path))
        assert config_path.exists()
        assert 'schwab' in mgr.config
        assert 'trading' in mgr.config

    def test_load_existing_config(self, tmp_path):
        """ConfigManager loads an existing YAML config."""
        from core.config import ConfigManager
        config_path = tmp_path / "test_config.yaml"
        config_data = {
            'schwab': {'callback_url': 'https://test', 'token_path': 'test.json'},
            'trading': {'max_risk_per_trade': 0.03, 'min_risk_reward_ratio': 1.5,
                        'max_daily_loss': 0.05, 'max_consecutive_losses': 3,
                        'max_positions': 10, 'reserve_cash_percent': 0.1},
            'commentary': {'enabled': True, 'detail_level': 'verbose', 'max_history': 50},
            'technical_analysis': {'timeframes': ['1min'], 'fibonacci_levels': [0.5]},
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        mgr = ConfigManager(config_path=str(config_path))
        assert mgr.get('trading.max_risk_per_trade') == 0.03

    def test_get_dot_notation(self, tmp_path):
        """ConfigManager.get() supports dot notation."""
        from core.config import ConfigManager
        config_path = tmp_path / "test_config.yaml"
        mgr = ConfigManager(config_path=str(config_path))
        assert mgr.get('trading.max_positions') == 5
        assert mgr.get('nonexistent.key', 'default') == 'default'

    def test_validation_rejects_bad_risk(self, tmp_path):
        """Config validation rejects out-of-range trading parameters."""
        config_path = tmp_path / "bad_config.yaml"
        config_data = {
            'schwab': {'callback_url': 'https://test', 'token_path': 'test.json'},
            'trading': {'max_risk_per_trade': 0.50},  # 50% - way too high
            'commentary': {'enabled': True},
            'technical_analysis': {'timeframes': ['1min']},
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        from core.config import ConfigManager
        with pytest.raises(ValueError, match="Max risk per trade"):
            ConfigManager(config_path=str(config_path))

    def test_validation_rejects_missing_section(self, tmp_path):
        """Config validation rejects configs missing required sections."""
        config_path = tmp_path / "incomplete.yaml"
        with open(config_path, 'w') as f:
            yaml.dump({'schwab': {}}, f)

        from core.config import ConfigManager
        with pytest.raises(ValueError, match="Missing required"):
            ConfigManager(config_path=str(config_path))


class TestConfig:
    """Test Config class properties and defaults."""

    def test_defaults_never_none(self):
        """All Config properties return a default, never None."""
        from core.config import Config
        c = Config()
        assert c.MAX_RISK_PER_TRADE is not None
        assert c.MIN_RISK_REWARD_RATIO is not None
        assert c.MAX_DAILY_LOSS is not None
        assert c.MAX_CONSECUTIVE_LOSSES is not None
        assert c.POSITION_SIZE_KELLY_FRACTION is not None
        assert c.COMMENTARY_ENABLED is not None
        assert c.MAX_POSITIONS is not None
        assert c.RESERVE_CASH_PERCENT is not None
        assert c.MIN_POSITION_SIZE is not None
        assert c.MAX_POSITION_VALUE is not None
        assert c.MIN_BUYING_POWER is not None
        assert c.DEFAULT_STOP_LOSS_PCT is not None
        assert c.DEFAULT_TAKE_PROFIT_PCT is not None
        assert c.LIMIT_ORDER_SLIPPAGE is not None
        assert c.MAX_POSITION_VALUE_PCT is not None

    def test_default_values_reasonable(self):
        """Default config values are within reasonable ranges."""
        from core.config import Config
        c = Config()
        assert 0 < c.MAX_RISK_PER_TRADE <= 0.10
        assert c.MIN_RISK_REWARD_RATIO >= 1.0
        assert 0 < c.MAX_DAILY_LOSS <= 0.20
        assert c.MAX_POSITIONS >= 1
        assert 0 <= c.RESERVE_CASH_PERCENT <= 1.0
