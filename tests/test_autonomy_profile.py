"""Tests for autonomy profile configuration.

v-autonomy-profile-2026-09-08: Tests covering:
  - Profile selection via environment variable
  - Profile selection via config file
  - Confirmation timeout action behavior
  - Flatten-on-circuit behavior
  - Profile-aware defaults
"""
import os
import pytest
from unittest.mock import patch, MagicMock

from core.config import Config, ConfigManager


class TestProfileSelection:
    def test_default_profile_is_supervised(self):
        """Test that the default profile is 'supervised'."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove TRADING_PROFILE if set
            os.environ.pop("TRADING_PROFILE", None)
            cfg = Config()
            # Reset manager's config to remove any cached profile
            cfg.manager.config.pop('profile', None)
            assert cfg.TRADING_PROFILE == "supervised"

    def test_profile_from_environment(self):
        """Test profile selection from environment variable."""
        with patch.dict(os.environ, {"TRADING_PROFILE": "autonomous_live"}):
            cfg = Config()
            assert cfg.TRADING_PROFILE == "autonomous_live"

    def test_profile_is_case_insensitive(self):
        """Test that profile names are case-insensitive."""
        with patch.dict(os.environ, {"TRADING_PROFILE": "AUTONOMOUS_LIVE"}):
            cfg = Config()
            assert cfg.TRADING_PROFILE == "autonomous_live"


class TestConfirmationTimeoutAction:
    def test_supervised_defaults_to_deny(self):
        """Test that supervised profile defaults to deny on timeout."""
        with patch.dict(os.environ, {"TRADING_PROFILE": "supervised"}):
            cfg = Config()
            cfg.manager.config.pop('order_management', None)
            assert cfg.CONFIRMATION_TIMEOUT_ACTION == "deny"

    def test_autonomous_live_defaults_to_execute(self):
        """Test that autonomous_live profile defaults to execute on timeout."""
        with patch.dict(os.environ, {"TRADING_PROFILE": "autonomous_live"}):
            cfg = Config()
            cfg.manager.config.pop('order_management', None)
            assert cfg.CONFIRMATION_TIMEOUT_ACTION == "execute"


class TestFlattenOnCircuit:
    def test_supervised_defaults_to_false(self):
        """Test that supervised profile defaults to no flatten-on-circuit."""
        with patch.dict(os.environ, {"TRADING_PROFILE": "supervised"}):
            cfg = Config()
            cfg.manager.config.pop('trading', None)
            assert cfg.FLATTEN_ON_CIRCUIT is False

    def test_autonomous_live_defaults_to_true(self):
        """Test that autonomous_live profile defaults to flatten-on-circuit."""
        with patch.dict(os.environ, {"TRADING_PROFILE": "autonomous_live"}):
            cfg = Config()
            cfg.manager.config.pop('trading', None)
            assert cfg.FLATTEN_ON_CIRCUIT is True


class TestConfirmationTimeoutSec:
    def test_default_timeout(self):
        """Test default confirmation timeout is 30 seconds."""
        cfg = Config()
        cfg.manager.config.pop('order_management', None)
        assert cfg.CONFIRMATION_TIMEOUT_SEC == 30.0


class TestNewsLoopSec:
    def test_default_news_loop_cadence(self):
        """Test default news loop cadence is 20 seconds."""
        cfg = Config()
        cfg.manager.config.pop('trading', None)
        assert cfg.NEWS_LOOP_SEC == 20.0


class TestProfileConfigOverrides:
    def test_config_override_timeout_action(self):
        """Test that config can override timeout action."""
        cfg = Config()
        cfg.manager.config['order_management'] = {
            'confirmation_timeout_action': 'deny'
        }
        assert cfg.CONFIRMATION_TIMEOUT_ACTION == "deny"

    def test_config_override_flatten_on_circuit(self):
        """Test that config can override flatten-on-circuit."""
        cfg = Config()
        cfg.manager.config['trading'] = {
            'flatten_on_circuit': True
        }
        assert cfg.FLATTEN_ON_CIRCUIT is True


class TestProfileDocumentation:
    """Tests to ensure profile documentation matches implementation."""
    
    def test_supervised_profile_characteristics(self):
        """Verify supervised profile has documented characteristics."""
        with patch.dict(os.environ, {"TRADING_PROFILE": "supervised"}):
            cfg = Config()
            cfg.manager.config.pop('order_management', None)
            cfg.manager.config.pop('trading', None)
            
            # Supervised profile should:
            # - Default require_close_confirmation to True
            assert cfg.REQUIRE_CLOSE_CONFIRMATION is True
            # - Default confirmation_timeout_action to deny
            assert cfg.CONFIRMATION_TIMEOUT_ACTION == "deny"
            # - Default flatten_on_circuit to False
            assert cfg.FLATTEN_ON_CIRCUIT is False

    def test_autonomous_live_profile_characteristics(self):
        """Verify autonomous_live profile has documented characteristics."""
        with patch.dict(os.environ, {"TRADING_PROFILE": "autonomous_live"}):
            cfg = Config()
            cfg.manager.config.pop('order_management', None)
            cfg.manager.config.pop('trading', None)
            
            # autonomous_live profile should:
            # - Default confirmation_timeout_action to execute
            assert cfg.CONFIRMATION_TIMEOUT_ACTION == "execute"
            # - Default flatten_on_circuit to True
            assert cfg.FLATTEN_ON_CIRCUIT is True
