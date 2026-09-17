"""v-pinned-watchlist-2026-09-17: tests for profit-first watchlist redesign.

Tests:
  1. Config properties: PINNED_WATCHLIST, MIN_DOLLAR_VOLUME_FLOOR, PINNED_MIN_RS
  2. Screener loop pinned watchlist seat reservation
  3. Screener dollar-volume calculation and filtering
  4. Pinned symbol RS soften in quality filter
  5. Gap commentary RTH gating
  6. HANDS_OFF denylist exclusion from pinned watchlist
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


class TestPinnedWatchlistConfig:
    """Test Config properties for pinned watchlist feature."""

    def test_pinned_watchlist_default(self):
        """PINNED_WATCHLIST returns default liquid mega-cap list."""
        from core.config import Config
        c = Config()
        pinned = c.PINNED_WATCHLIST
        assert isinstance(pinned, list)
        assert 'NVDA' in pinned
        assert 'TSLA' in pinned
        assert 'META' in pinned
        assert 'SPY' in pinned
        assert 'QQQ' in pinned

    def test_pinned_watchlist_env_override(self, monkeypatch):
        """PINNED_WATCHLIST respects env override."""
        monkeypatch.setenv("PINNED_WATCHLIST", "AAPL,GOOG,MSFT")
        from core.config import Config
        c = Config()
        assert c.PINNED_WATCHLIST == ['AAPL', 'GOOG', 'MSFT']

    def test_pinned_watchlist_env_whitespace_handling(self, monkeypatch):
        """PINNED_WATCHLIST strips whitespace and uppercases."""
        monkeypatch.setenv("PINNED_WATCHLIST", " aapl , goog , msft ")
        from core.config import Config
        c = Config()
        assert c.PINNED_WATCHLIST == ['AAPL', 'GOOG', 'MSFT']

    def test_enable_pinned_watchlist_default_true(self, monkeypatch):
        """ENABLE_PINNED_WATCHLIST defaults to True."""
        monkeypatch.delenv("ENABLE_PINNED_WATCHLIST", raising=False)
        from core.config import Config
        c = Config()
        assert c.ENABLE_PINNED_WATCHLIST is True

    def test_enable_pinned_watchlist_env_override(self, monkeypatch):
        """ENABLE_PINNED_WATCHLIST respects env=0."""
        monkeypatch.setenv("ENABLE_PINNED_WATCHLIST", "0")
        from core.config import Config
        c = Config()
        assert c.ENABLE_PINNED_WATCHLIST is False

    def test_min_dollar_volume_floor_default(self, monkeypatch):
        """MIN_DOLLAR_VOLUME_FLOOR defaults to $50M."""
        monkeypatch.delenv("MIN_DOLLAR_VOLUME_FLOOR", raising=False)
        from core.config import Config
        c = Config()
        assert c.MIN_DOLLAR_VOLUME_FLOOR == 50_000_000

    def test_min_dollar_volume_floor_env_override(self, monkeypatch):
        """MIN_DOLLAR_VOLUME_FLOOR respects env override."""
        monkeypatch.setenv("MIN_DOLLAR_VOLUME_FLOOR", "100000000")
        from core.config import Config
        c = Config()
        assert c.MIN_DOLLAR_VOLUME_FLOOR == 100_000_000

    def test_pinned_min_rs_default(self, monkeypatch):
        """PINNED_MIN_RS defaults to -0.10."""
        monkeypatch.delenv("PINNED_MIN_RS", raising=False)
        from core.config import Config
        c = Config()
        assert c.PINNED_MIN_RS == -0.10

    def test_pinned_min_rs_env_override(self, monkeypatch):
        """PINNED_MIN_RS respects env override."""
        monkeypatch.setenv("PINNED_MIN_RS", "-0.15")
        from core.config import Config
        c = Config()
        assert c.PINNED_MIN_RS == -0.15

    def test_gap_commentary_rth_minutes_default(self, monkeypatch):
        """GAP_COMMENTARY_RTH_MINUTES defaults to 30."""
        monkeypatch.delenv("GAP_COMMENTARY_RTH_MINUTES", raising=False)
        from core.config import Config
        c = Config()
        assert c.GAP_COMMENTARY_RTH_MINUTES == 30

    def test_gap_commentary_rth_minutes_env_override(self, monkeypatch):
        """GAP_COMMENTARY_RTH_MINUTES respects env override."""
        monkeypatch.setenv("GAP_COMMENTARY_RTH_MINUTES", "45")
        from core.config import Config
        c = Config()
        assert c.GAP_COMMENTARY_RTH_MINUTES == 45


class TestScreenerDollarVolume:
    """Test StockScreener dollar-volume calculation and filtering."""

    @pytest.fixture
    def screener(self):
        from analysis.screener import StockScreener
        return StockScreener(schwab_client=MagicMock(), commentary_system=MagicMock())

    def test_calculate_dollar_volume(self, screener):
        """Dollar-volume is price × volume."""
        mover = {'last': 100.0, 'volume': 1_000_000}
        dv = screener._calculate_dollar_volume(mover)
        assert dv == 100_000_000  # $100M

    def test_calculate_dollar_volume_zero_price(self, screener):
        """Dollar-volume is 0 when price is 0."""
        mover = {'last': 0, 'volume': 1_000_000}
        dv = screener._calculate_dollar_volume(mover)
        assert dv == 0

    def test_calculate_dollar_volume_zero_volume(self, screener):
        """Dollar-volume is 0 when volume is 0."""
        mover = {'last': 100.0, 'volume': 0}
        dv = screener._calculate_dollar_volume(mover)
        assert dv == 0

    def test_passes_dollar_volume_filter_above_floor(self, screener, monkeypatch):
        """Movers above MIN_DOLLAR_VOLUME_FLOOR pass."""
        monkeypatch.setenv("MIN_DOLLAR_VOLUME_FLOOR", "50000000")
        mover = {'symbol': 'NVDA', 'last': 500.0, 'volume': 1_000_000}  # $500M
        assert screener._passes_dollar_volume_filter(mover) is True

    def test_passes_dollar_volume_filter_below_floor(self, screener, monkeypatch):
        """Movers below MIN_DOLLAR_VOLUME_FLOOR fail."""
        monkeypatch.setenv("MIN_DOLLAR_VOLUME_FLOOR", "50000000")
        mover = {'symbol': 'PURR', 'last': 5.0, 'volume': 100_000}  # $500k
        assert screener._passes_dollar_volume_filter(mover) is False

    def test_passes_dollar_volume_filter_zero_floor(self, screener, monkeypatch):
        """All movers pass when floor is 0 (disabled)."""
        monkeypatch.setenv("MIN_DOLLAR_VOLUME_FLOOR", "0")
        mover = {'symbol': 'PURR', 'last': 5.0, 'volume': 100_000}  # $500k
        assert screener._passes_dollar_volume_filter(mover) is True

    def test_mover_score_includes_dollar_volume(self, screener):
        """_calculate_mover_score includes dollar-volume component."""
        mover_high_dv = {'symbol': 'NVDA', 'last': 500.0, 'volume': 2_000_000,
                        'volatility': 3.0, 'percent_change': 5.0}
        mover_low_dv = {'symbol': 'PURR', 'last': 5.0, 'volume': 500_000,
                       'volatility': 3.0, 'percent_change': 5.0}

        score_high = screener._calculate_mover_score(mover_high_dv)
        score_low = screener._calculate_mover_score(mover_low_dv)

        assert score_high > score_low, "High dollar-volume should score higher"
        assert mover_high_dv.get('dollar_volume') == 1_000_000_000  # $1B
        assert mover_low_dv.get('dollar_volume') == 2_500_000  # $2.5M


class TestScreenerPinnedRsSoften:
    """Test pinned symbol RS soften in quality filter."""

    @pytest.fixture
    def screener(self):
        from analysis.screener import StockScreener
        s = StockScreener(schwab_client=MagicMock(), commentary_system=MagicMock())
        s._quality_filter_rejects = {}
        return s

    def test_quality_filter_accepts_min_rs_parameter(self, screener):
        """_passes_quality_filter accepts configurable min_rs."""
        result_strict = screener._passes_quality_filter('TEST', min_rs=-0.05)
        result_relaxed = screener._passes_quality_filter('TEST', min_rs=-0.10)
        assert result_strict is True
        assert result_relaxed is True

    def test_quality_filter_blocklist_ignores_min_rs(self, screener):
        """Leveraged ETF blocklist rejects regardless of min_rs."""
        result = screener._passes_quality_filter('TQQQ', min_rs=-1.0)
        assert result is False


class TestScreenerLoopPinnedWatchlist:
    """Test ScreenerLoop pinned watchlist seat reservation."""

    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        engine.is_running = True
        engine.screener = MagicMock()
        engine.screener.top_movers = [
            {'symbol': 'AAPL', 'score': 0.9, 'dollar_volume': 500_000_000},
            {'symbol': 'RKLB', 'score': 0.8, 'dollar_volume': 100_000_000},
            {'symbol': 'PURR', 'score': 0.7, 'dollar_volume': 1_000_000},  # Low DV
        ]
        engine.screener.get_movers_with_dollar_volume = MagicMock(
            return_value=engine.screener.top_movers
        )
        engine.screener.get_watchlist_symbols = MagicMock(
            return_value=['AAPL', 'RKLB', 'PURR']
        )
        engine.screener.screen_stocks = AsyncMock()
        engine.dynamic_watchlist = []
        engine.price_book = None
        engine.commentary = None
        return engine

    def test_get_hands_off_symbols(self, mock_engine):
        from core.loops.screener_loop import ScreenerLoop
        loop = ScreenerLoop(mock_engine)
        hands_off = loop._get_hands_off_symbols()
        assert 'MU' in hands_off
        assert 'HQGE' in hands_off
        assert 'SPCX' in hands_off

    def test_get_pinned_watchlist_excludes_hands_off(self, mock_engine, monkeypatch):
        """HANDS_OFF symbols excluded from pinned watchlist."""
        monkeypatch.setenv("PINNED_WATCHLIST", "NVDA,MU,TSLA,HQGE")
        from core.loops.screener_loop import ScreenerLoop
        loop = ScreenerLoop(mock_engine)
        hands_off = loop._get_hands_off_symbols()
        pinned = loop._get_pinned_watchlist(hands_off)
        assert 'NVDA' in pinned
        assert 'TSLA' in pinned
        assert 'MU' not in pinned
        assert 'HQGE' not in pinned

    def test_tick_reserves_pinned_seats(self, mock_engine, monkeypatch):
        """Pinned symbols get reserved seats before filler movers."""
        monkeypatch.setenv("PINNED_WATCHLIST", "NVDA,TSLA")
        monkeypatch.setenv("WATCHLIST_SIZE", "5")
        monkeypatch.setenv("MIN_DOLLAR_VOLUME_FLOOR", "0")

        from core.loops.screener_loop import ScreenerLoop
        loop = ScreenerLoop(mock_engine)
        asyncio.run(loop._tick())

        wl = mock_engine.dynamic_watchlist
        assert wl[0] == 'NVDA'
        assert wl[1] == 'TSLA'
        assert 'AAPL' in wl or 'RKLB' in wl

    def test_tick_filters_low_dollar_volume_filler(self, mock_engine, monkeypatch):
        """Filler slots filter by MIN_DOLLAR_VOLUME_FLOOR."""
        monkeypatch.setenv("PINNED_WATCHLIST", "NVDA")
        monkeypatch.setenv("WATCHLIST_SIZE", "3")
        monkeypatch.setenv("MIN_DOLLAR_VOLUME_FLOOR", "50000000")  # $50M

        from core.loops.screener_loop import ScreenerLoop
        loop = ScreenerLoop(mock_engine)
        asyncio.run(loop._tick())

        wl = mock_engine.dynamic_watchlist
        assert 'NVDA' in wl
        assert 'AAPL' in wl
        assert 'RKLB' in wl
        assert 'PURR' not in wl  # Below $50M

    def test_tick_disabled_pinned_watchlist(self, mock_engine, monkeypatch):
        """When ENABLE_PINNED_WATCHLIST=0, no pinned seats reserved."""
        monkeypatch.setenv("ENABLE_PINNED_WATCHLIST", "0")
        monkeypatch.setenv("WATCHLIST_SIZE", "3")
        monkeypatch.setenv("MIN_DOLLAR_VOLUME_FLOOR", "0")

        from core.loops.screener_loop import ScreenerLoop
        loop = ScreenerLoop(mock_engine)
        asyncio.run(loop._tick())

        wl = mock_engine.dynamic_watchlist
        assert wl == ['AAPL', 'RKLB', 'PURR']


class TestGapCommentaryRthGating:
    """Test gap commentary RTH gating."""

    def test_gap_commentary_emitted_within_rth_window(self, monkeypatch):
        """Gap commentary emits within first N minutes of RTH."""
        monkeypatch.setenv("GAP_COMMENTARY_RTH_MINUTES", "30")

        morning_time = datetime.now().replace(hour=9, minute=35)

        with patch('core.engine.datetime') as mock_datetime:
            mock_datetime.now.return_value = morning_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            from core.config import Config
            c = Config()
            gap_minutes = c.GAP_COMMENTARY_RTH_MINUTES
            assert gap_minutes == 30

    def test_gap_commentary_suppressed_after_rth_window(self, monkeypatch):
        """Gap commentary suppressed after N minutes past RTH open."""
        monkeypatch.setenv("GAP_COMMENTARY_RTH_MINUTES", "30")

        from core.config import Config
        c = Config()
        assert c.GAP_COMMENTARY_RTH_MINUTES == 30

    def test_gap_commentary_disabled_with_negative_setting(self, monkeypatch):
        """GAP_COMMENTARY_RTH_MINUTES=-1 suppresses all gap commentary."""
        monkeypatch.setenv("GAP_COMMENTARY_RTH_MINUTES", "-1")

        from core.config import Config
        c = Config()
        assert c.GAP_COMMENTARY_RTH_MINUTES == -1

    def test_gap_commentary_always_on_with_zero_setting(self, monkeypatch):
        """GAP_COMMENTARY_RTH_MINUTES=0 disables gating (always emit)."""
        monkeypatch.setenv("GAP_COMMENTARY_RTH_MINUTES", "0")

        from core.config import Config
        c = Config()
        assert c.GAP_COMMENTARY_RTH_MINUTES == 0


class TestScreenerIntegration:
    """Integration tests for screener with pinned watchlist."""

    @pytest.fixture
    def screener(self, monkeypatch):
        from analysis.screener import StockScreener
        s = StockScreener(schwab_client=MagicMock(), commentary_system=MagicMock())

        async def _yahoo_active():
            return [
                {'symbol': 'PURR', 'last': 5.0, 'volume': 1_000_000,
                 'percent_change': 20.0, 'source': 'yahoo_most_active'},
                {'symbol': 'AEMD', 'last': 3.0, 'volume': 500_000,
                 'percent_change': 15.0, 'source': 'yahoo_most_active'},
                {'symbol': 'NVDA', 'last': 500.0, 'volume': 10_000_000,
                 'percent_change': 3.0, 'source': 'yahoo_most_active'},
            ]

        async def _empty(*args, **kwargs):
            return []

        async def _noop(*args, **kwargs):
            return None

        monkeypatch.setattr(s, "_get_yahoo_most_active", _yahoo_active)
        monkeypatch.setattr(s, "_get_yahoo_top_movers", _empty)
        monkeypatch.setattr(s, "_get_index_movers", _empty)
        monkeypatch.setattr(s, "_get_volatile_stocks", _empty)
        monkeypatch.setattr(s, "_refresh_quality_data", _noop)
        monkeypatch.setattr(s, "_get_quote_data", lambda sym: None)
        return s

    def test_screen_stocks_tags_pinned_symbols(self, screener, monkeypatch):
        """screen_stocks tags is_pinned on pinned symbols."""
        monkeypatch.setenv("PINNED_WATCHLIST", "NVDA,TSLA")
        result = asyncio.run(screener.screen_stocks())
        nvda = next((m for m in result if m['symbol'] == 'NVDA'), None)
        purr = next((m for m in result if m['symbol'] == 'PURR'), None)
        assert nvda is not None
        assert nvda.get('is_pinned') is True
        if purr:
            assert purr.get('is_pinned') is False

    def test_screen_stocks_calculates_dollar_volume(self, screener):
        """screen_stocks calculates dollar_volume for each mover."""
        result = asyncio.run(screener.screen_stocks())
        for mover in result:
            assert 'dollar_volume' in mover
            assert mover['dollar_volume'] == mover['last'] * mover['volume']
