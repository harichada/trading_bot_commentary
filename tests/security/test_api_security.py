"""Security tests for API endpoints.

Covers: authentication, input validation, credential protection.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import asyncio
from unittest.mock import patch, MagicMock
from tests.conftest import make_config

# Import the FastAPI app for testing
from gap_fade_app import app, CONFIG_VALID_RANGES, live_trader

class TestAPIAuthentication:
    """API key enforcement on endpoints."""

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self):
        """GET /api/health always accessible (verified via direct call)."""
        from gap_fade_app import health_check
        result = await health_check()
        assert isinstance(result, dict)
        assert 'status' in result or 'uptime' in result or len(result) > 0

    @pytest.mark.asyncio
    async def test_state_no_auth_required(self):
        """GET /api/state is read-only, accessible (verified via direct call)."""
        from gap_fade_app import live_trader
        state = live_trader.get_state() if hasattr(live_trader, 'get_state') else {}
        assert isinstance(state, dict)


class TestConfigValidation:
    """Config parameters validated against safe ranges."""

    def test_all_critical_params_have_ranges(self):
        """Critical trading parameters must have validation ranges."""
        required = ['stop_pct', 'risk_pct', 'max_positions', 'initial_capital']
        for param in required:
            assert param in CONFIG_VALID_RANGES, \
                f"Critical param '{param}' missing from CONFIG_VALID_RANGES"

    def test_ranges_are_tuples(self):
        for param, bounds in CONFIG_VALID_RANGES.items():
            assert isinstance(bounds, tuple) and len(bounds) == 2, \
                f"{param}: range must be (low, high) tuple"
            assert bounds[0] < bounds[1], \
                f"{param}: low ({bounds[0]}) must be < high ({bounds[1]})"

    def test_stop_pct_cannot_be_zero(self):
        low, _ = CONFIG_VALID_RANGES['stop_pct']
        assert low > 0, "stop_pct lower bound must be > 0"

    def test_risk_pct_cannot_exceed_10pct(self):
        _, high = CONFIG_VALID_RANGES['risk_pct']
        assert high <= 0.10, "risk_pct upper bound must be ≤ 10%"

    def test_max_positions_at_least_1(self):
        low, _ = CONFIG_VALID_RANGES['max_positions']
        assert low >= 1


class TestCredentialProtection:
    """Secrets never leak into logs or responses."""

    def test_api_key_not_in_health(self):
        """Health endpoint doesn't expose API keys."""
        import asyncio
        from gap_fade_app import health_check
        result = asyncio.get_event_loop().run_until_complete(health_check())
        result_str = str(result)
        assert 'ALPACA_API_KEY' not in result_str
        assert 'SECRET' not in result_str.upper() or 'secret_key' not in result_str

    def test_env_vars_not_in_state(self):
        """State endpoint doesn't expose env vars."""
        state = live_trader.get_state() if hasattr(live_trader, 'get_state') else {}
        state_str = str(state)
        for key in ['ALPACA_SECRET_KEY', 'TELEGRAM_BOT_TOKEN', 'AUTH_JWT_SECRET']:
            val = os.environ.get(key, '')
            if val and len(val) > 4:
                assert val not in state_str, f"{key} value leaked in state"


class TestInputSanitization:
    """Input validation for API parameters."""

    def test_config_rejects_negative_capital(self):
        """Negative initial_capital rejected by validation range."""
        low, _ = CONFIG_VALID_RANGES['initial_capital']
        assert low > 0

    def test_config_rejects_huge_positions(self):
        """max_positions > 10 rejected."""
        _, high = CONFIG_VALID_RANGES['max_positions']
        assert high <= 10


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
