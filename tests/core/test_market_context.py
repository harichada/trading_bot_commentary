"""Unit tests for core/market_context.py.

Built 2026-06-08 as Phase 1 of the MarketContext rollout. Tests
verify the pure-function classifiers (direction, regime, time-of-day,
conviction) and the symbol→sector mapping. read_market_context is
exercised against monkeypatched cache states so we don't need a live
Schwab connection.

The strategy wiring (Phase 2) lands tonight after market close.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

from core.market_context import (
    MarketContext,
    _classify_regime,
    _classify_spy_direction,
    _classify_time_of_day,
    _compute_conviction,
    get_sector_etf,
    read_market_context,
)


# ────────────────────────────────────────────────────────────────────
# Symbol → sector mapping
# ────────────────────────────────────────────────────────────────────

class TestSymbolToSector:
    def test_nvda_is_tech(self):
        assert get_sector_etf("NVDA") == "XLK"

    def test_amzn_is_consumer_discretionary(self):
        assert get_sector_etf("AMZN") == "XLY"

    def test_coin_is_financials(self):
        """COIN classified as financials per crypto-broker grouping."""
        assert get_sector_etf("COIN") == "XLF"

    def test_meta_is_communication(self):
        assert get_sector_etf("META") == "XLC"

    def test_lowercase_input(self):
        """Symbols are stored uppercase; lookup should normalize."""
        assert get_sector_etf("nvda") == "XLK"

    def test_unknown_returns_empty(self):
        assert get_sector_etf("XYZQWERTY") == ""

    def test_empty_string_returns_empty(self):
        assert get_sector_etf("") == ""


# ────────────────────────────────────────────────────────────────────
# SPY direction classifier
# ────────────────────────────────────────────────────────────────────

class TestSpyDirection:
    def test_bullish_above_threshold(self):
        assert _classify_spy_direction(0.5) == "bullish"
        assert _classify_spy_direction(2.0) == "bullish"

    def test_bearish_below_threshold(self):
        assert _classify_spy_direction(-0.5) == "bearish"
        assert _classify_spy_direction(-1.5) == "bearish"

    def test_neutral_within_band(self):
        """SPY moving 0.1% isn't directional tape."""
        assert _classify_spy_direction(0.1) == "neutral"
        assert _classify_spy_direction(-0.2) == "neutral"
        assert _classify_spy_direction(0.0) == "neutral"

    def test_boundary_exactly_at_threshold(self):
        """0.3% exactly — not greater than → neutral."""
        assert _classify_spy_direction(0.3) == "neutral"
        assert _classify_spy_direction(-0.3) == "neutral"


# ────────────────────────────────────────────────────────────────────
# Regime classifier
# ────────────────────────────────────────────────────────────────────

class TestRegimeClassifier:
    def test_risk_on_classic(self):
        """SPY up + VIX down: textbook risk-on."""
        r = _classify_regime(spy_change_pct=0.8, vix_level=14.0, vix_change_pct=-3.0)
        assert r == "risk_on"

    def test_risk_on_low_vol_even_with_flat_vix(self):
        """SPY up + low VIX level (even if VIX flat) is still risk-on."""
        r = _classify_regime(spy_change_pct=0.5, vix_level=13.0, vix_change_pct=0.5)
        assert r == "risk_on"

    def test_risk_off_panic(self):
        """SPY down + VIX spike: panic."""
        r = _classify_regime(spy_change_pct=-1.2, vix_level=22.0, vix_change_pct=8.0)
        assert r == "risk_off"

    def test_risk_off_vix_spike_alone(self):
        """VIX spike >10% is risk-off regardless of SPY."""
        r = _classify_regime(spy_change_pct=0.1, vix_level=18.0, vix_change_pct=12.0)
        assert r == "risk_off"

    def test_mixed_nervous_rally(self):
        """SPY up but VIX up = unhealthy rally."""
        r = _classify_regime(spy_change_pct=0.5, vix_level=16.0, vix_change_pct=4.0)
        assert r == "mixed"

    def test_mixed_orderly_pullback(self):
        """SPY down but VIX flat/down = orderly profit-taking."""
        r = _classify_regime(spy_change_pct=-0.4, vix_level=14.0, vix_change_pct=-1.0)
        assert r == "mixed"

    def test_default_to_mixed_when_unclear(self):
        """SPY flat, VIX flat → mixed (no strong signal)."""
        r = _classify_regime(spy_change_pct=0.0, vix_level=16.0, vix_change_pct=0.0)
        assert r == "mixed"


# ────────────────────────────────────────────────────────────────────
# Time-of-day classifier
# ────────────────────────────────────────────────────────────────────

class TestTimeOfDay:
    def test_before_market_open_is_off_hours(self):
        # 08:00 ET = 12:00 UTC
        t = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
        assert _classify_time_of_day(t) == "off_hours"

    def test_opening_30_window(self):
        # 09:35 ET = 13:35 UTC
        t = datetime(2026, 6, 8, 13, 35, tzinfo=timezone.utc)
        assert _classify_time_of_day(t) == "opening_30"

    def test_morning(self):
        # 11:00 ET = 15:00 UTC
        t = datetime(2026, 6, 8, 15, 0, tzinfo=timezone.utc)
        assert _classify_time_of_day(t) == "morning"

    def test_midday(self):
        # 12:30 ET = 16:30 UTC
        t = datetime(2026, 6, 8, 16, 30, tzinfo=timezone.utc)
        assert _classify_time_of_day(t) == "midday"

    def test_afternoon(self):
        # 14:30 ET = 18:30 UTC
        t = datetime(2026, 6, 8, 18, 30, tzinfo=timezone.utc)
        assert _classify_time_of_day(t) == "afternoon"

    def test_closing_30(self):
        # 15:45 ET = 19:45 UTC
        t = datetime(2026, 6, 8, 19, 45, tzinfo=timezone.utc)
        assert _classify_time_of_day(t) == "closing_30"

    def test_after_market_close(self):
        # 17:00 ET = 21:00 UTC
        t = datetime(2026, 6, 8, 21, 0, tzinfo=timezone.utc)
        assert _classify_time_of_day(t) == "off_hours"


# ────────────────────────────────────────────────────────────────────
# Conviction multiplier
# ────────────────────────────────────────────────────────────────────

class TestConvictionMultiplier:
    def test_perfect_risk_on_setup(self):
        """Risk-on regime + sector positive + morning = high conviction."""
        m = _compute_conviction(
            regime="risk_on",
            spy_change_pct=1.0,
            sector_strength_pct=0.8,
            has_sector=True,
            time_of_day="morning",
        )
        # base 1.0 + risk_on 0.2 + sector 0.15 = 1.35
        assert m == pytest.approx(1.35)

    def test_perfect_risk_off_setup(self):
        """Risk-off + sector down + closing30 = low conviction."""
        m = _compute_conviction(
            regime="risk_off",
            spy_change_pct=-1.0,
            sector_strength_pct=-0.8,
            has_sector=True,
            time_of_day="closing_30",
        )
        # base 1.0 - risk_off 0.3 - sector 0.15 - closing 0.1 = 0.45 → clamp to 0.5
        assert m == 0.5

    def test_neutral_regime_returns_near_1(self):
        m = _compute_conviction(
            regime="mixed",
            spy_change_pct=0.0,
            sector_strength_pct=0.0,
            has_sector=False,
            time_of_day="morning",
        )
        # base 1.0 - mixed 0.1 = 0.9
        assert m == 0.9

    def test_no_sector_data_no_sector_adjustment(self):
        """When has_sector=False, sector_strength_pct is ignored."""
        m = _compute_conviction(
            regime="risk_on",
            spy_change_pct=1.0,
            sector_strength_pct=2.0,  # would normally help
            has_sector=False,
            time_of_day="morning",
        )
        # Only regime contribution: 1.0 + 0.2 = 1.2 (no sector boost)
        assert m == 1.2

    def test_clamping_at_upper_bound(self):
        m = _compute_conviction(
            regime="risk_on",
            spy_change_pct=2.0,
            sector_strength_pct=3.0,
            has_sector=True,
            time_of_day="morning",
        )
        assert m <= 1.5

    def test_clamping_at_lower_bound(self):
        m = _compute_conviction(
            regime="risk_off",
            spy_change_pct=-2.0,
            sector_strength_pct=-3.0,
            has_sector=True,
            time_of_day="closing_30",
        )
        assert m >= 0.5


# ────────────────────────────────────────────────────────────────────
# Real scenarios using read_market_context with mocked cache
# ────────────────────────────────────────────────────────────────────

class _MockQuote:
    """Mimics the IndexQuote interface for test fixtures."""
    def __init__(self, change_pct, last=100.0):
        self.change_pct = change_pct
        self.last = last


class _MockCache:
    """Stand-in for MarketIndicesCache.instance() in tests."""
    def __init__(self, quotes: dict):
        self._quotes = quotes

    def get_quote(self, symbol):
        return self._quotes.get(symbol)


class TestReadMarketContextScenarios:
    """End-to-end: read_market_context returns sensible MarketContext
    given various synthetic cache states."""

    def _patch_cache(self, monkeypatch, quotes):
        """Wire a mock cache through MarketIndicesCache.instance()."""
        import core.market_context as mc
        import core.market_indices as mi
        mock = _MockCache(quotes)
        # Replace instance() method to return our mock
        monkeypatch.setattr(
            mi.MarketIndicesCache, "instance",
            classmethod(lambda cls: mock),
        )

    def test_strong_risk_on_nvda(self, monkeypatch):
        """SPY +0.8, VIX -5%, XLK +1.5 — perfect tech bull setup.
        NVDA should get high conviction multiplier."""
        self._patch_cache(monkeypatch, {
            "SPY":  _MockQuote(0.8, 760),
            "$VIX": _MockQuote(-5.0, 14.0),
            "XLK":  _MockQuote(1.5, 240),
        })
        ctx = read_market_context("NVDA")
        assert ctx.regime == "risk_on"
        assert ctx.sector_etf == "XLK"
        assert ctx.sector_strength_pct == 1.5
        assert ctx.conviction_multiplier >= 1.3
        assert ctx.allows_long
        assert not ctx.allows_short

    def test_risk_off_with_unknown_symbol(self, monkeypatch):
        """Unknown symbol still gets SPY + VIX regime, just no
        sector contribution."""
        self._patch_cache(monkeypatch, {
            "SPY":  _MockQuote(-1.0, 750),
            "$VIX": _MockQuote(12.0, 20.0),
        })
        ctx = read_market_context("XYZQWERTY")
        assert ctx.regime == "risk_off"
        assert ctx.sector_etf == ""
        assert ctx.conviction_multiplier <= 0.7
        assert not ctx.allows_long

    def test_sector_opposite_to_market_reduces_conviction(self, monkeypatch):
        """SPY up but financials down — XLF underperforming.
        A COIN signal should get LESS conviction than a tech name
        the same day."""
        self._patch_cache(monkeypatch, {
            "SPY":  _MockQuote(0.6, 758),
            "$VIX": _MockQuote(-1.0, 15.5),
            "XLK":  _MockQuote(1.2, 240),    # tech leading
            "XLF":  _MockQuote(-0.8, 45),    # financials lagging
        })
        nvda_ctx = read_market_context("NVDA")
        coin_ctx = read_market_context("COIN")
        # Same regime
        assert nvda_ctx.regime == coin_ctx.regime
        # But NVDA gets sector tailwind, COIN gets headwind
        assert nvda_ctx.conviction_multiplier > coin_ctx.conviction_multiplier

    def test_empty_cache_fails_open(self, monkeypatch):
        """No data in cache → return MarketContext with defaults +
        conviction 1.0 (no behavior change from baseline)."""
        self._patch_cache(monkeypatch, {})
        ctx = read_market_context("NVDA")
        assert ctx.spy_change_pct == 0.0
        # Default regime when SPY=0, VIX=16, VIX_change=0 is 'mixed'
        # per the classifier rules. conviction 0.9 (mixed → -0.1).
        assert ctx.conviction_multiplier == 0.9


# ────────────────────────────────────────────────────────────────────
# Defensive — bad inputs don't crash
# ────────────────────────────────────────────────────────────────────

class TestRobustness:
    def test_unknown_symbol_doesnt_crash(self, monkeypatch):
        import core.market_indices as mi
        monkeypatch.setattr(
            mi.MarketIndicesCache, "instance",
            classmethod(lambda cls: _MockCache({
                "SPY": _MockQuote(0.5, 760),
            })),
        )
        ctx = read_market_context("ZZTOPABC")
        assert isinstance(ctx, MarketContext)
        assert ctx.sector_etf == ""

    def test_cache_raises_returns_default(self, monkeypatch):
        """If MarketIndicesCache itself blows up, return a safe
        default context — fail-open, never raise upward."""
        class _BrokenCache:
            @classmethod
            def instance(cls):
                raise RuntimeError("simulated boom")
        import core.market_indices as mi
        monkeypatch.setattr(mi, "MarketIndicesCache", _BrokenCache)
        ctx = read_market_context("NVDA")
        assert ctx.regime == "unknown"
        assert ctx.conviction_multiplier == 1.0  # fail-open neutral
