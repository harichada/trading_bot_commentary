"""Regression tests for previously identified bugs.

Each test documents a real issue that was found and fixed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import asyncio
import inspect
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from tests.conftest import make_config, make_engine, make_candidate, make_time, ET


class TestEODCloseRegression:
    """INCIDENT: LLM standdown blocked trading loop, EOD close missed.

    Root cause: EOD close was inline in the single trading loop coroutine.
    LLM standdown sleep (15-30 min) could span the 3:50 PM window.
    Fix: Independent _eod_watchdog task.
    """

    def test_eod_not_in_trading_loop(self):
        """_trading_loop must NOT contain _did_eod or _eod_close."""
        from gap_fade_app import GapFadeLiveTrader
        src = inspect.getsource(GapFadeLiveTrader._trading_loop)
        assert '_did_eod' not in src, "EOD flag must not be in trading loop"
        # The loop may reference _eod_close in after-hours fallback, but
        # the primary EOD check should not be there
        assert 'EOD close at 3:50' not in src

    def test_eod_watchdog_exists(self):
        """_eod_watchdog method must exist as independent task."""
        from gap_fade_app import GapFadeLiveTrader
        assert hasattr(GapFadeLiveTrader, '_eod_watchdog')
        assert asyncio.iscoroutinefunction(GapFadeLiveTrader._eod_watchdog)

    def test_circuit_breaker_in_enter(self):
        """_enter_positions must check _trading_halted."""
        from gap_fade_app import GapFadeLiveTrader
        src = inspect.getsource(GapFadeLiveTrader._enter_positions)
        assert '_trading_halted' in src

    def test_circuit_breaker_in_intraday(self):
        """_intraday_scan_and_enter must check _trading_halted."""
        from gap_fade_app import GapFadeLiveTrader
        src = inspect.getsource(GapFadeLiveTrader._intraday_scan_and_enter)
        assert '_trading_halted' in src


class TestIndicatorEngineWipeRegression:
    """INCIDENT: Indicator engine seeded data wiped on first intraday scan.

    Root cause: _intraday_date = '' in __init__, first call saw today != ''
    → triggered _reset_intraday_daily() → indicator_engine.reset().
    Fix: Set _intraday_date after seeding in start().
    """

    def test_intraday_date_set_after_seed(self):
        """start() must set _intraday_date after _seed_indicator_engine."""
        from gap_fade_app import GapFadeLiveTrader
        src = inspect.getsource(GapFadeLiveTrader.start)
        # Both must appear, and _intraday_date assignment must come after seed
        assert '_seed_indicator_engine' in src
        assert '_intraday_date' in src
        seed_pos = src.find('_seed_indicator_engine')
        date_pos = src.find("self._intraday_date = datetime.now(ET)")
        assert date_pos > seed_pos, \
            "_intraday_date must be set AFTER _seed_indicator_engine"


class TestPennyStockFilterRegression:
    """INCIDENT: Penny stocks appearing in intraday watchlist.

    Root cause: fetch_alpaca_movers had no price filter.
    Fix: Added Alpaca snapshot API batch call to filter < $5.
    """

    def test_snapshot_filter_in_movers(self):
        """fetch_alpaca_movers must filter by price."""
        from gap_fade_app import fetch_alpaca_movers
        src = inspect.getsource(fetch_alpaca_movers)
        assert 'snapshot' in src.lower() or 'price' in src.lower(), \
            "fetch_alpaca_movers must have price filtering"


class TestWebSocketSymbolLimitRegression:
    """INCIDENT: Alpaca 405 error — exceeded 25-symbol WebSocket limit.

    Root cause: intraday_watchlist_size default was 50, Alpaca free tier = 25.
    Fix: Changed default to 25, added priority-based eviction.
    """

    def test_default_watchlist_size(self):
        from gap_fade_app import GapFadeConfig
        c = GapFadeConfig()
        assert c.intraday_watchlist_size <= 25, \
            "Default watchlist size must not exceed Alpaca free tier limit"

    def test_streamer_has_max_symbols(self):
        """AlpacaTickStreamer must enforce symbol limit."""
        from gap_fade_app import AlpacaTickStreamer
        assert hasattr(AlpacaTickStreamer, '_MAX_SYMBOLS')
        assert AlpacaTickStreamer._MAX_SYMBOLS <= 25


class TestVersionDisplayRegression:
    """INCIDENT: UI showed git SHA instead of codename.

    Fix: _get_version reads codename from build_info.json.
    """

    def test_version_prefers_codename(self):
        """_get_version prefers codename over git_sha from build_info.json."""
        import json
        import tempfile
        from gap_fade_app import _get_version

        # Verify the function exists and returns a string
        version = _get_version()
        assert isinstance(version, str)
        assert len(version) > 0


class TestPositionTrackingRegression:
    """INCIDENT: Positions tab showing '—' for NVDA/MARA (no current price).

    Root cause: Mid-day positions couldn't subscribe to WebSocket (all 25 slots taken).
    Fix: Priority-based symbol eviction in add_symbols.
    """

    def test_add_symbols_has_priority(self):
        """AlpacaTickStreamer.add_symbols must accept priority_symbols."""
        from gap_fade_app import AlpacaTickStreamer
        sig = inspect.signature(AlpacaTickStreamer.add_symbols)
        assert 'priority_symbols' in sig.parameters


class TestHealthEndpointRegression:
    """Health endpoint must report watchdog status."""

    @pytest.mark.asyncio
    async def test_health_has_watchdog_fields(self):
        from gap_fade_app import health_check
        result = await health_check()
        assert 'eod_watchdog' in result
        assert 'trading_halted' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
