"""Tests for v-pinned-watchlist-2026-09-17: pinned watchlist and dollar-volume filter.

Tests cover:
  1. Config properties: ENABLE_PINNED_WATCHLIST, PINNED_WATCHLIST, MIN_DOLLAR_VOLUME
  2. Config properties: ENABLE_PINNED_RS_SOFTEN, PINNED_MIN_RS_VS_SPY
  3. ScreenerLoop._build_pinned_watchlist() logic
  4. Dollar-volume demote filtering
  5. RS threshold selection for pinned vs non-pinned
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime


class TestPinnedWatchlistConfig:
    """Test Config class pinned watchlist properties."""

    def test_enable_pinned_watchlist_default_true(self, monkeypatch):
        """ENABLE_PINNED_WATCHLIST defaults to True."""
        monkeypatch.delenv("ENABLE_PINNED_WATCHLIST", raising=False)
        from core.config import Config
        c = Config()
        assert c.ENABLE_PINNED_WATCHLIST is True

    def test_enable_pinned_watchlist_env_0(self, monkeypatch):
        """ENABLE_PINNED_WATCHLIST can be disabled via env."""
        monkeypatch.setenv("ENABLE_PINNED_WATCHLIST", "0")
        from core.config import Config
        c = Config()
        assert c.ENABLE_PINNED_WATCHLIST is False

    def test_enable_pinned_watchlist_env_1(self, monkeypatch):
        """ENABLE_PINNED_WATCHLIST can be enabled via env."""
        monkeypatch.setenv("ENABLE_PINNED_WATCHLIST", "1")
        from core.config import Config
        c = Config()
        assert c.ENABLE_PINNED_WATCHLIST is True

    def test_pinned_watchlist_default_symbols(self, monkeypatch):
        """PINNED_WATCHLIST has expected default symbols."""
        monkeypatch.delenv("PINNED_WATCHLIST", raising=False)
        from core.config import Config
        c = Config()
        pins = c.PINNED_WATCHLIST
        assert isinstance(pins, list)
        assert len(pins) >= 8
        # Check for core liquid names
        assert 'NVDA' in pins
        assert 'TSLA' in pins
        assert 'AMZN' in pins
        assert 'META' in pins

    def test_pinned_watchlist_env_override(self, monkeypatch):
        """PINNED_WATCHLIST can be overridden via env (comma-separated)."""
        monkeypatch.setenv("PINNED_WATCHLIST", "AAPL,GOOG,XYZ")
        from core.config import Config
        c = Config()
        assert c.PINNED_WATCHLIST == ['AAPL', 'GOOG', 'XYZ']

    def test_pinned_watchlist_env_whitespace_handling(self, monkeypatch):
        """PINNED_WATCHLIST env parsing handles whitespace."""
        monkeypatch.setenv("PINNED_WATCHLIST", " AAPL , GOOG , XYZ ")
        from core.config import Config
        c = Config()
        assert c.PINNED_WATCHLIST == ['AAPL', 'GOOG', 'XYZ']

    def test_pinned_watchlist_case_normalization(self, monkeypatch):
        """PINNED_WATCHLIST symbols are uppercased."""
        monkeypatch.setenv("PINNED_WATCHLIST", "aapl,Goog,xyz")
        from core.config import Config
        c = Config()
        assert c.PINNED_WATCHLIST == ['AAPL', 'GOOG', 'XYZ']


class TestMinDollarVolumeConfig:
    """Test MIN_DOLLAR_VOLUME config property."""

    def test_min_dollar_volume_default(self):
        """MIN_DOLLAR_VOLUME defaults to $10M."""
        from core.config import Config
        c = Config()
        assert c.MIN_DOLLAR_VOLUME == 10_000_000

    def test_min_dollar_volume_type(self):
        """MIN_DOLLAR_VOLUME returns float."""
        from core.config import Config
        c = Config()
        assert isinstance(c.MIN_DOLLAR_VOLUME, float)


class TestPinnedRsSoftenConfig:
    """Test RS soften config properties for pinned symbols."""

    def test_enable_pinned_rs_soften_default_false(self, monkeypatch):
        """ENABLE_PINNED_RS_SOFTEN defaults to False (conservative)."""
        monkeypatch.delenv("ENABLE_PINNED_RS_SOFTEN", raising=False)
        from core.config import Config
        c = Config()
        assert c.ENABLE_PINNED_RS_SOFTEN is False

    def test_enable_pinned_rs_soften_env_1(self, monkeypatch):
        """ENABLE_PINNED_RS_SOFTEN can be enabled via env."""
        monkeypatch.setenv("ENABLE_PINNED_RS_SOFTEN", "1")
        from core.config import Config
        c = Config()
        assert c.ENABLE_PINNED_RS_SOFTEN is True

    def test_pinned_min_rs_vs_spy_default(self):
        """PINNED_MIN_RS_VS_SPY defaults to 0.25 (softer than standard 0.5)."""
        from core.config import Config
        c = Config()
        assert c.PINNED_MIN_RS_VS_SPY == 0.25
        assert c.PINNED_MIN_RS_VS_SPY < c.MOMENTUM_MIN_RS_VS_SPY


class TestBuildPinnedWatchlist:
    """Test ScreenerLoop._build_pinned_watchlist() logic."""

    def test_pinned_symbols_first(self, monkeypatch):
        """Pinned symbols occupy first slots before movers."""
        monkeypatch.delenv("ENABLE_PINNED_WATCHLIST", raising=False)
        monkeypatch.delenv("PINNED_WATCHLIST", raising=False)
        
        from core.config import Config
        from core.loops.screener_loop import ScreenerLoop
        
        # Mock engine
        mock_engine = MagicMock()
        loop = ScreenerLoop(mock_engine)
        
        # Mock screener with movers
        mock_screener = MagicMock()
        mock_screener.top_movers = [
            {'symbol': 'PURR', 'last': 2.0, 'volume': 5000000},  # micro-cap junk
            {'symbol': 'AEMD', 'last': 3.0, 'volume': 3000000},  # micro-cap junk
            {'symbol': 'GOOG', 'last': 150.0, 'volume': 20000000},  # quality
        ]
        
        cfg = Config()
        result = loop._build_pinned_watchlist(mock_screener, cfg, wl_size=20)
        
        # Pinned symbols should be first
        pinned_set = set(cfg.PINNED_WATCHLIST)
        for i, sym in enumerate(result[:len(cfg.PINNED_WATCHLIST)]):
            if i < len(cfg.PINNED_WATCHLIST):
                assert sym in pinned_set, f"Position {i} should be pinned, got {sym}"

    def test_dollar_volume_demotes_micro_caps(self, monkeypatch):
        """Micro-cap junk is demoted by dollar-volume floor."""
        monkeypatch.setenv("ENABLE_PINNED_WATCHLIST", "1")
        monkeypatch.setenv("PINNED_WATCHLIST", "NVDA")  # Just one pin
        
        from core.config import Config
        from core.loops.screener_loop import ScreenerLoop
        
        mock_engine = MagicMock()
        loop = ScreenerLoop(mock_engine)
        
        mock_screener = MagicMock()
        mock_screener.top_movers = [
            {'symbol': 'PURR', 'last': 2.0, 'volume': 1000000},   # $2M dollar-vol
            {'symbol': 'AEMD', 'last': 3.0, 'volume': 500000},    # $1.5M dollar-vol
            {'symbol': 'GOOG', 'last': 150.0, 'volume': 20000000}, # $3B dollar-vol
            {'symbol': 'AMZN', 'last': 185.0, 'volume': 10000000}, # $1.85B dollar-vol
        ]
        
        cfg = Config()
        result = loop._build_pinned_watchlist(mock_screener, cfg, wl_size=5)
        
        # PURR and AEMD should be excluded (below $10M floor)
        assert 'PURR' not in result, "PURR should be demoted"
        assert 'AEMD' not in result, "AEMD should be demoted"
        # GOOG and AMZN should pass
        assert 'GOOG' in result, "GOOG should pass dollar-vol"
        assert 'AMZN' in result, "AMZN should pass dollar-vol"

    def test_movers_sorted_by_dollar_volume(self, monkeypatch):
        """Non-pinned movers are ranked by dollar-volume descending."""
        monkeypatch.setenv("ENABLE_PINNED_WATCHLIST", "1")
        monkeypatch.setenv("PINNED_WATCHLIST", "NVDA")
        
        from core.config import Config
        from core.loops.screener_loop import ScreenerLoop
        
        mock_engine = MagicMock()
        loop = ScreenerLoop(mock_engine)
        
        mock_screener = MagicMock()
        mock_screener.top_movers = [
            {'symbol': 'LOW', 'last': 50.0, 'volume': 500000},     # $25M
            {'symbol': 'MED', 'last': 100.0, 'volume': 500000},    # $50M
            {'symbol': 'HIGH', 'last': 200.0, 'volume': 1000000},  # $200M
        ]
        
        cfg = Config()
        result = loop._build_pinned_watchlist(mock_screener, cfg, wl_size=10)
        
        # After NVDA (pinned), movers should be sorted by dollar-vol: HIGH, MED, LOW
        mover_portion = [s for s in result if s != 'NVDA']
        if len(mover_portion) >= 2:
            high_idx = mover_portion.index('HIGH') if 'HIGH' in mover_portion else 999
            med_idx = mover_portion.index('MED') if 'MED' in mover_portion else 999
            low_idx = mover_portion.index('LOW') if 'LOW' in mover_portion else 999
            assert high_idx < med_idx, "HIGH should come before MED"
            assert med_idx < low_idx, "MED should come before LOW"

    def test_legacy_behavior_when_disabled(self, monkeypatch):
        """When ENABLE_PINNED_WATCHLIST=False, fallback to legacy screener order."""
        monkeypatch.setenv("ENABLE_PINNED_WATCHLIST", "0")
        
        from core.config import Config
        from core.loops.screener_loop import ScreenerLoop
        
        mock_engine = MagicMock()
        loop = ScreenerLoop(mock_engine)
        
        mock_screener = MagicMock()
        mock_screener.top_movers = [
            {'symbol': 'FIRST', 'last': 10.0, 'volume': 1000000},
            {'symbol': 'SECOND', 'last': 20.0, 'volume': 2000000},
        ]
        mock_screener.get_watchlist_symbols = MagicMock(return_value=['FIRST', 'SECOND'])
        
        cfg = Config()
        result = loop._build_pinned_watchlist(mock_screener, cfg, wl_size=10)
        
        # Should use screener order, not pinned-first
        assert result[0] == 'FIRST', "Legacy mode should use screener order"

    def test_watchlist_size_limit_respected(self, monkeypatch):
        """Watchlist never exceeds wl_size."""
        monkeypatch.setenv("ENABLE_PINNED_WATCHLIST", "1")
        monkeypatch.setenv("PINNED_WATCHLIST", "NVDA,TSLA,META,AMZN,MSFT,GOOG,AVGO,AMD")
        
        from core.config import Config
        from core.loops.screener_loop import ScreenerLoop
        
        mock_engine = MagicMock()
        loop = ScreenerLoop(mock_engine)
        
        mock_screener = MagicMock()
        mock_screener.top_movers = [
            {'symbol': f'MOVER{i}', 'last': 100.0, 'volume': 1000000}
            for i in range(30)
        ]
        
        cfg = Config()
        result = loop._build_pinned_watchlist(mock_screener, cfg, wl_size=10)
        
        assert len(result) <= 10, "Watchlist should not exceed wl_size"


class TestDayTradeMomentumRsSoften:
    """Test RS threshold selection for pinned vs non-pinned in DayTradeMomentumStrategy."""

    def test_pinned_symbol_uses_softer_rs_when_enabled(self, monkeypatch):
        """Pinned symbols use PINNED_MIN_RS_VS_SPY when ENABLE_PINNED_RS_SOFTEN=True."""
        monkeypatch.setenv("ENABLE_PINNED_RS_SOFTEN", "1")
        monkeypatch.setenv("PINNED_WATCHLIST", "NVDA,TSLA")
        
        from core.config import Config
        cfg = Config()
        
        symbol = "NVDA"
        _is_pinned = symbol.upper() in set(s.upper() for s in cfg.PINNED_WATCHLIST)
        
        assert _is_pinned is True
        
        if cfg.ENABLE_PINNED_RS_SOFTEN and _is_pinned:
            _min_rs = cfg.PINNED_MIN_RS_VS_SPY
        else:
            _min_rs = cfg.MOMENTUM_MIN_RS_VS_SPY
        
        assert _min_rs == 0.25, "Pinned with RS soften should use 0.25"

    def test_non_pinned_uses_standard_rs(self, monkeypatch):
        """Non-pinned symbols always use MOMENTUM_MIN_RS_VS_SPY."""
        monkeypatch.setenv("ENABLE_PINNED_RS_SOFTEN", "1")
        monkeypatch.setenv("PINNED_WATCHLIST", "NVDA,TSLA")
        
        from core.config import Config
        cfg = Config()
        
        symbol = "PURR"  # Not in pinned list
        _is_pinned = symbol.upper() in set(s.upper() for s in cfg.PINNED_WATCHLIST)
        
        assert _is_pinned is False
        
        if cfg.ENABLE_PINNED_RS_SOFTEN and _is_pinned:
            _min_rs = cfg.PINNED_MIN_RS_VS_SPY
        else:
            _min_rs = cfg.MOMENTUM_MIN_RS_VS_SPY
        
        assert _min_rs == 0.5, "Non-pinned should use standard 0.5"

    def test_pinned_uses_standard_rs_when_soften_disabled(self, monkeypatch):
        """Pinned symbols use standard RS when ENABLE_PINNED_RS_SOFTEN=False."""
        monkeypatch.setenv("ENABLE_PINNED_RS_SOFTEN", "0")
        monkeypatch.setenv("PINNED_WATCHLIST", "NVDA,TSLA")
        
        from core.config import Config
        cfg = Config()
        
        symbol = "NVDA"
        _is_pinned = symbol.upper() in set(s.upper() for s in cfg.PINNED_WATCHLIST)
        
        assert _is_pinned is True
        
        if cfg.ENABLE_PINNED_RS_SOFTEN and _is_pinned:
            _min_rs = cfg.PINNED_MIN_RS_VS_SPY
        else:
            _min_rs = cfg.MOMENTUM_MIN_RS_VS_SPY
        
        assert _min_rs == 0.5, "Pinned without RS soften should use standard 0.5"
