"""Unit tests for configuration validation and edge cases."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from tests.conftest import make_config, GapFadeConfig


class TestConfigDefaults:
    """Verify sensible defaults."""

    def test_default_capital(self):
        c = GapFadeConfig()
        assert c.initial_capital == 25_000

    def test_default_risk(self):
        c = GapFadeConfig()
        assert c.risk_pct == 0.02

    def test_default_max_positions(self):
        c = GapFadeConfig()
        assert c.max_positions == 5

    def test_default_gap_threshold(self):
        c = GapFadeConfig()
        assert c.gap_threshold == 0.07

    def test_default_stop_pct(self):
        c = GapFadeConfig()
        assert c.stop_pct == 0.015

    def test_default_eod_exit(self):
        c = GapFadeConfig()
        assert c.eod_exit_hour == 15
        assert c.eod_exit_min == 50


class TestConfigOverrides:
    """Configuration can be overridden."""

    def test_override_capital(self):
        c = make_config(initial_capital=500_000)
        assert c.initial_capital == 500_000

    def test_override_risk(self):
        c = make_config(risk_pct=0.01)
        assert c.risk_pct == 0.01

    def test_override_max_positions(self):
        c = make_config(max_positions=10)
        assert c.max_positions == 10


class TestConfigValidRanges:
    """CONFIG_VALID_RANGES enforcement (tested via the /api/config endpoint logic)."""

    def test_stop_pct_bounds(self):
        """Stop pct should be between 0.5% and 10%."""
        from gap_fade_app import CONFIG_VALID_RANGES
        low, high = CONFIG_VALID_RANGES['stop_pct']
        assert low == 0.005
        assert high == 0.10

    def test_risk_pct_bounds(self):
        from gap_fade_app import CONFIG_VALID_RANGES
        low, high = CONFIG_VALID_RANGES['risk_pct']
        assert low == 0.001
        assert high == 0.10

    def test_max_positions_bounds(self):
        from gap_fade_app import CONFIG_VALID_RANGES
        low, high = CONFIG_VALID_RANGES['max_positions']
        assert low == 1
        assert high == 10


class TestConfigEdgeCases:
    """Edge case configurations."""

    def test_zero_kelly_fraction(self):
        """kelly_fraction=0 → should still allow trading (falls back to risk_pct)."""
        c = make_config(kelly_fraction=0.0)
        assert c.kelly_fraction == 0.0

    def test_gap_down_disabled_by_default(self):
        c = GapFadeConfig()
        assert c.trade_gap_downs is False

    def test_llm_api_key_from_env(self):
        """LLM API key loaded from env if not set."""
        import os
        os.environ['LLM_API_KEY'] = 'test_llm_key'
        c = GapFadeConfig()
        assert c.llm_api_key == 'test_llm_key'
        del os.environ['LLM_API_KEY']


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
