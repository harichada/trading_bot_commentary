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


class TestEnableBotOnlyPnlCircuitEnvOverride:
    """v-env-override-bot-only-pnl-circuit-2026-09-14: test that
    ENABLE_BOT_ONLY_PNL_CIRCUIT honors env override the same way
    DAY_TRADE_LIVE_ENTRIES_ENABLED / ENABLE_ORB_STRATEGY do.

    Priority order:
      1. env ENABLE_BOT_ONLY_PNL_CIRCUIT (truthy/falsy)
      2. yaml trading.enable_bot_only_pnl_circuit
      3. hardcoded default True

    Fixes false trip on 2026-09-14 where .env had flag=1 but yaml
    had flag=false — yaml won, causing account-wide P&L to fire
    EMERGENCY STOP while bot-only P&L was $0.
    """

    def test_env_1_overrides_yaml_false(self, tmp_path, monkeypatch):
        """env=1 forces True even when yaml says false."""
        monkeypatch.setenv("ENABLE_BOT_ONLY_PNL_CIRCUIT", "1")
        config_path = tmp_path / "test_config.yaml"
        config_data = {
            'schwab': {'callback_url': 'https://test', 'token_path': 'test.json'},
            'trading': {
                'max_risk_per_trade': 0.02,
                'min_risk_reward_ratio': 2.0,
                'max_daily_loss': 0.05,
                'max_consecutive_losses': 3,
                'max_positions': 5,
                'reserve_cash_percent': 0.1,
                'enable_bot_only_pnl_circuit': False,
            },
            'commentary': {'enabled': True, 'detail_level': 'verbose', 'max_history': 50},
            'technical_analysis': {'timeframes': ['1min'], 'fibonacci_levels': [0.5]},
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        from core.config import ConfigManager, Config
        mgr = ConfigManager(config_path=str(config_path))
        c = Config()
        c.manager = mgr
        assert c.ENABLE_BOT_ONLY_PNL_CIRCUIT is True, (
            "env=1 must override yaml=false → True"
        )

    def test_env_0_overrides_yaml_true(self, tmp_path, monkeypatch):
        """env=0 forces False even when yaml says true."""
        monkeypatch.setenv("ENABLE_BOT_ONLY_PNL_CIRCUIT", "0")
        config_path = tmp_path / "test_config.yaml"
        config_data = {
            'schwab': {'callback_url': 'https://test', 'token_path': 'test.json'},
            'trading': {
                'max_risk_per_trade': 0.02,
                'min_risk_reward_ratio': 2.0,
                'max_daily_loss': 0.05,
                'max_consecutive_losses': 3,
                'max_positions': 5,
                'reserve_cash_percent': 0.1,
                'enable_bot_only_pnl_circuit': True,
            },
            'commentary': {'enabled': True, 'detail_level': 'verbose', 'max_history': 50},
            'technical_analysis': {'timeframes': ['1min'], 'fibonacci_levels': [0.5]},
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        from core.config import ConfigManager, Config
        mgr = ConfigManager(config_path=str(config_path))
        c = Config()
        c.manager = mgr
        assert c.ENABLE_BOT_ONLY_PNL_CIRCUIT is False, (
            "env=0 must override yaml=true → False"
        )

    def test_env_unset_uses_yaml_false(self, tmp_path, monkeypatch):
        """When env is unset, yaml value is used (False case)."""
        monkeypatch.delenv("ENABLE_BOT_ONLY_PNL_CIRCUIT", raising=False)
        config_path = tmp_path / "test_config.yaml"
        config_data = {
            'schwab': {'callback_url': 'https://test', 'token_path': 'test.json'},
            'trading': {
                'max_risk_per_trade': 0.02,
                'min_risk_reward_ratio': 2.0,
                'max_daily_loss': 0.05,
                'max_consecutive_losses': 3,
                'max_positions': 5,
                'reserve_cash_percent': 0.1,
                'enable_bot_only_pnl_circuit': False,
            },
            'commentary': {'enabled': True, 'detail_level': 'verbose', 'max_history': 50},
            'technical_analysis': {'timeframes': ['1min'], 'fibonacci_levels': [0.5]},
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        from core.config import ConfigManager, Config
        mgr = ConfigManager(config_path=str(config_path))
        c = Config()
        c.manager = mgr
        assert c.ENABLE_BOT_ONLY_PNL_CIRCUIT is False, (
            "env unset must fall back to yaml=false"
        )

    def test_env_unset_uses_yaml_true(self, tmp_path, monkeypatch):
        """When env is unset, yaml value is used (True case)."""
        monkeypatch.delenv("ENABLE_BOT_ONLY_PNL_CIRCUIT", raising=False)
        config_path = tmp_path / "test_config.yaml"
        config_data = {
            'schwab': {'callback_url': 'https://test', 'token_path': 'test.json'},
            'trading': {
                'max_risk_per_trade': 0.02,
                'min_risk_reward_ratio': 2.0,
                'max_daily_loss': 0.05,
                'max_consecutive_losses': 3,
                'max_positions': 5,
                'reserve_cash_percent': 0.1,
                'enable_bot_only_pnl_circuit': True,
            },
            'commentary': {'enabled': True, 'detail_level': 'verbose', 'max_history': 50},
            'technical_analysis': {'timeframes': ['1min'], 'fibonacci_levels': [0.5]},
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        from core.config import ConfigManager, Config
        mgr = ConfigManager(config_path=str(config_path))
        c = Config()
        c.manager = mgr
        assert c.ENABLE_BOT_ONLY_PNL_CIRCUIT is True, (
            "env unset must fall back to yaml=true"
        )

    def test_env_unset_yaml_unset_uses_default_true(self, tmp_path, monkeypatch):
        """When both env and yaml are unset, default True is used."""
        monkeypatch.delenv("ENABLE_BOT_ONLY_PNL_CIRCUIT", raising=False)
        config_path = tmp_path / "test_config.yaml"
        config_data = {
            'schwab': {'callback_url': 'https://test', 'token_path': 'test.json'},
            'trading': {
                'max_risk_per_trade': 0.02,
                'min_risk_reward_ratio': 2.0,
                'max_daily_loss': 0.05,
                'max_consecutive_losses': 3,
                'max_positions': 5,
                'reserve_cash_percent': 0.1,
            },
            'commentary': {'enabled': True, 'detail_level': 'verbose', 'max_history': 50},
            'technical_analysis': {'timeframes': ['1min'], 'fibonacci_levels': [0.5]},
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        from core.config import ConfigManager, Config
        mgr = ConfigManager(config_path=str(config_path))
        c = Config()
        c.manager = mgr
        assert c.ENABLE_BOT_ONLY_PNL_CIRCUIT is True, (
            "env unset + yaml unset must default to True"
        )

    def test_env_truthy_variants(self, monkeypatch):
        """All truthy env values (1/true/yes/on) resolve to True."""
        from core.config import Config
        for truthy in ("1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"):
            monkeypatch.setenv("ENABLE_BOT_ONLY_PNL_CIRCUIT", truthy)
            c = Config()
            assert c.ENABLE_BOT_ONLY_PNL_CIRCUIT is True, (
                f"env={truthy!r} must resolve to True"
            )

    def test_env_falsy_variants(self, monkeypatch):
        """All falsy env values (0/false/no/off) resolve to False."""
        from core.config import Config
        for falsy in ("0", "false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF"):
            monkeypatch.setenv("ENABLE_BOT_ONLY_PNL_CIRCUIT", falsy)
            c = Config()
            assert c.ENABLE_BOT_ONLY_PNL_CIRCUIT is False, (
                f"env={falsy!r} must resolve to False"
            )


def test_block_new_entries_yaml_only(tmp_path, monkeypatch):
    from core.config import Config, ConfigManager
    import yaml
    p = tmp_path / "c.yaml"
    # ConfigManager._validate_config requires these top-level sections to be
    # present (not necessarily populated) — see tests/test_config.py's other
    # temp-yaml fixtures for the same minimal shape.
    p.write_text(yaml.safe_dump({
        "schwab": {},
        "trading": {"block_new_entries": True},
        "commentary": {},
        "technical_analysis": {},
    }))
    cfg = Config()
    monkeypatch.setattr(cfg, "manager", ConfigManager(config_path=str(p)))
    monkeypatch.setenv("BLOCK_NEW_ENTRIES", "0")   # env must NOT override
    assert cfg.BLOCK_NEW_ENTRIES is True
    cfg.manager.update("trading.block_new_entries", False)
    assert cfg.BLOCK_NEW_ENTRIES is False


def test_settings_promote_time_default_and_env(monkeypatch):
    from core.config import Config
    monkeypatch.delenv("SETTINGS_PROMOTE_TIME_ET", raising=False)
    assert Config().SETTINGS_PROMOTE_TIME_ET == "09:25"
    monkeypatch.setenv("SETTINGS_PROMOTE_TIME_ET", "09:31")
    assert Config().SETTINGS_PROMOTE_TIME_ET == "09:31"


def test_config_manager_update_is_thread_safe_and_atomic(tmp_path):
    """v-risk-console-2026-09-28: concurrent updates from the API thread and
    the desk loop must not lose keys or leave a torn yaml behind."""
    import threading
    import yaml
    from core.config import ConfigManager
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"schwab": {}, "trading": {}, "commentary": {},
                                 "technical_analysis": {}}))
    mgr = ConfigManager(config_path=str(p))

    def writer(prefix):
        for i in range(25):
            mgr.update(f"trading.{prefix}_{i}", i)

    threads = [threading.Thread(target=writer, args=(f"k{t}",)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    on_disk = yaml.safe_load(p.read_text())["trading"]
    assert len(on_disk) == 100
    assert not (tmp_path / "c.yaml.tmp").exists()
