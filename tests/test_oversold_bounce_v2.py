"""v-oversold-v2-tests-2026-05-19: tests for OversoldBounceV2Strategy.

Unit-tests for the strategy live here. The strategy depends on three
external collaborators:
  * a commentary system (we stub it out — strategy uses `_log_decision`
    which writes to the logger, not the commentary)
  * a news verifier (we inject mock verifiers via constructor)
  * the SPY regime cache (we patch with monkeypatch)

No Postgres, no real news API, no Schwab calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.models import MarketData, SignalType
from strategies.oversold_bounce_v2 import (
    OversoldBounceV2Strategy,
    bar_has_capitulation_structure,
    find_nearest_swing_low,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@dataclass
class _StubVerifyResult:
    fresh_count: int = 0
    avg_fresh_sentiment: float = 0.0
    is_verified: bool = True
    reason: str = "ok"


class _StubVerifier:
    """Async-compatible stub. Configure result or exception per-test."""

    def __init__(self, result: Optional[_StubVerifyResult] = None,
                 raises: Optional[Exception] = None) -> None:
        self._result = result if result is not None else _StubVerifyResult()
        self._raises = raises

    async def verify(self, symbol: str, expected_direction: int) -> _StubVerifyResult:
        if self._raises is not None:
            raise self._raises
        return self._result


def _make_market_data(
    *,
    symbol: str = "TEST",
    open_: float = 100.0,
    high: float = 100.5,
    low: float = 98.0,
    close: float = 99.5,
    indicators: Optional[Dict[str, Any]] = None,
) -> MarketData:
    """Build a MarketData object with sensible defaults that PASS all gates.

    Tests override individual fields to fail a chosen gate.
    """
    base_indicators: Dict[str, Any] = {
        "rsi": 25.0,
        "bb_lower": 100.0,            # close=99.5 < bb_lower → Gate A pass
        "volume_ratio": 2.5,           # > 2.0 → Gate B pass
        "atr": 1.0,
        # Gate C — bar with long lower wick: open=100, low=98, close=99.5 →
        #   lower_wick = min(100, 99.5) - 98 = 1.5, body = |99.5-100| = 0.5,
        #   1.5 > 1.5 * 0.5 = 0.75 → True
        "prev_low": 98.2,
        # Gate D — swing low at index -3: 99.0 is a local min,
        # |99.5 - 99.0| = 0.5 == 0.5 * atr(1.0) → pass.
        "lows_50": [101.0, 100.5, 100.0, 99.5, 100.0, 99.5, 99.0, 99.5, 99.8, 99.6],
    }
    if indicators:
        base_indicators.update(indicators)
    return MarketData(
        symbol=symbol,
        timestamp=datetime(2026, 5, 19, 13, 30),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1_000_000,
        timeframe="5m",
        indicators=base_indicators,
    )


@pytest.fixture
def commentary_system():
    return MagicMock()


@pytest.fixture
def strategy(commentary_system, monkeypatch):
    """Default strategy: clean news verifier + flat SPY regime."""
    monkeypatch.setattr(
        "core.spy_regime.SpyRegimeCache.instance",
        lambda: MagicMock(ema_slope_pct=MagicMock(return_value=0.0)),
    )
    verifier = _StubVerifier(result=_StubVerifyResult(fresh_count=0))
    return OversoldBounceV2Strategy(commentary_system, adverse_news_verifier=verifier)


# ----------------------------------------------------------------------
# TestFindSwingLow
# ----------------------------------------------------------------------
class TestFindSwingLow:
    def test_flat_array_returns_none(self):
        # All identical → no strict local minima exist.
        lows = [100.0] * 10
        assert find_nearest_swing_low(lows, current_price=100.0, atr=1.0) is None

    def test_v_shape_within_threshold_returns_swing(self):
        # Swing low of 99.5 at index 2; current 99.6, atr 1.0 → threshold 0.5.
        lows = [100.0, 99.8, 99.5, 99.7, 100.0]
        result = find_nearest_swing_low(lows, current_price=99.6, atr=1.0)
        assert result == pytest.approx(99.5)

    def test_v_shape_outside_threshold_returns_none(self):
        # Swing low 99.5; current 101.0 — distance 1.5 > 0.5 * atr(1.0).
        lows = [100.0, 99.8, 99.5, 99.7, 100.0]
        assert find_nearest_swing_low(lows, current_price=101.0, atr=1.0) is None

    def test_two_valid_returns_nearest_recent(self):
        # Two strict local minima: 99.0 at idx 2 (older) and 99.4 at idx 6
        # (more recent). Both within threshold. Most-recent-first scan must
        # return the newer one.
        lows = [100.0, 99.5, 99.0, 99.5, 100.0, 99.6, 99.4, 99.6, 99.8]
        result = find_nearest_swing_low(lows, current_price=99.5, atr=1.0)
        assert result == pytest.approx(99.4)


# ----------------------------------------------------------------------
# TestBarStructure
# ----------------------------------------------------------------------
class TestBarStructure:
    def test_doji_no_wick_fails(self):
        # open=close, no wick at all → fails both branches.
        assert bar_has_capitulation_structure(
            open_=100.0, high=100.0, low=100.0, close=100.0, prev_low=None,
        ) is False

    def test_hammer_passes(self):
        # Hammer: open=100, close=99.8, low=98.0 → lower_wick=1.8, body=0.2,
        # 1.8 > 1.5 * 0.2 = 0.3 → pass.
        assert bar_has_capitulation_structure(
            open_=100.0, high=100.1, low=98.0, close=99.8, prev_low=None,
        ) is True

    def test_close_above_prev_low_passes(self):
        # No wick (body=2.0 vs wick=0) — but close >= prev_low triggers pass.
        assert bar_has_capitulation_structure(
            open_=100.0, high=102.0, low=100.0, close=102.0, prev_low=101.5,
        ) is True


# ----------------------------------------------------------------------
# Gate-level tests (one gate at a time)
# ----------------------------------------------------------------------
@pytest.mark.asyncio
class TestGateA_StatisticalExtreme:
    async def test_rsi_25_close_below_bb_passes(self, strategy):
        md = _make_market_data(indicators={"rsi": 25.0, "bb_lower": 100.0}, close=99.5)
        signal = await strategy.generate_signal_with_commentary(md)
        assert signal is not None
        assert signal.signal_type == SignalType.BUY

    async def test_rsi_31_fails(self, strategy):
        md = _make_market_data(indicators={"rsi": 31.0})
        assert await strategy.generate_signal_with_commentary(md) is None

    async def test_close_above_bb_lower_fails(self, strategy):
        md = _make_market_data(close=101.0, indicators={"bb_lower": 100.0})
        assert await strategy.generate_signal_with_commentary(md) is None


@pytest.mark.asyncio
class TestGateB_CapitulationVolume:
    async def test_volume_ratio_3_passes(self, strategy):
        md = _make_market_data(indicators={"volume_ratio": 3.0})
        assert await strategy.generate_signal_with_commentary(md) is not None

    async def test_volume_ratio_2_exactly_fails(self, strategy):
        # Spec is strict ">"; 2.0 itself must fail.
        md = _make_market_data(indicators={"volume_ratio": 2.0})
        assert await strategy.generate_signal_with_commentary(md) is None

    async def test_volume_ratio_1_5_fails(self, strategy):
        md = _make_market_data(indicators={"volume_ratio": 1.5})
        assert await strategy.generate_signal_with_commentary(md) is None


@pytest.mark.asyncio
class TestGateE_Regime:
    async def test_slope_zero_passes(self, commentary_system, monkeypatch):
        monkeypatch.setattr(
            "core.spy_regime.SpyRegimeCache.instance",
            lambda: MagicMock(ema_slope_pct=MagicMock(return_value=0.0)),
        )
        strat = OversoldBounceV2Strategy(
            commentary_system, adverse_news_verifier=_StubVerifier(),
        )
        md = _make_market_data()
        signal = await strat.generate_signal_with_commentary(md)
        assert signal is not None
        assert signal.reasoning["regime_factor"] == 1.0

    async def test_slope_mildly_down_passes_with_half_factor(
        self, commentary_system, monkeypatch,
    ):
        monkeypatch.setattr(
            "core.spy_regime.SpyRegimeCache.instance",
            lambda: MagicMock(ema_slope_pct=MagicMock(return_value=-0.07)),
        )
        strat = OversoldBounceV2Strategy(
            commentary_system, adverse_news_verifier=_StubVerifier(),
        )
        md = _make_market_data()
        signal = await strat.generate_signal_with_commentary(md)
        assert signal is not None
        assert signal.reasoning["regime_factor"] == 0.5

    async def test_slope_strongly_down_fails(self, commentary_system, monkeypatch):
        monkeypatch.setattr(
            "core.spy_regime.SpyRegimeCache.instance",
            lambda: MagicMock(ema_slope_pct=MagicMock(return_value=-0.15)),
        )
        strat = OversoldBounceV2Strategy(
            commentary_system, adverse_news_verifier=_StubVerifier(),
        )
        md = _make_market_data()
        assert await strat.generate_signal_with_commentary(md) is None


@pytest.mark.asyncio
class TestGateF_AdverseNews:
    async def test_clean_news_passes(self, commentary_system, monkeypatch):
        monkeypatch.setattr(
            "core.spy_regime.SpyRegimeCache.instance",
            lambda: MagicMock(ema_slope_pct=MagicMock(return_value=0.0)),
        )
        verifier = _StubVerifier(
            result=_StubVerifyResult(fresh_count=0, avg_fresh_sentiment=0.0),
        )
        strat = OversoldBounceV2Strategy(commentary_system, adverse_news_verifier=verifier)
        md = _make_market_data()
        signal = await strat.generate_signal_with_commentary(md)
        assert signal is not None
        assert signal.reasoning["news_gate_status"] == "ok"

    async def test_adverse_news_fails(self, commentary_system, monkeypatch):
        monkeypatch.setattr(
            "core.spy_regime.SpyRegimeCache.instance",
            lambda: MagicMock(ema_slope_pct=MagicMock(return_value=0.0)),
        )
        verifier = _StubVerifier(
            result=_StubVerifyResult(fresh_count=3, avg_fresh_sentiment=-0.5),
        )
        strat = OversoldBounceV2Strategy(commentary_system, adverse_news_verifier=verifier)
        md = _make_market_data()
        assert await strat.generate_signal_with_commentary(md) is None

    async def test_news_api_exception_treated_as_unavailable_pass(
        self, commentary_system, monkeypatch,
    ):
        monkeypatch.setattr(
            "core.spy_regime.SpyRegimeCache.instance",
            lambda: MagicMock(ema_slope_pct=MagicMock(return_value=0.0)),
        )
        verifier = _StubVerifier(raises=RuntimeError("alpaca timed out"))
        strat = OversoldBounceV2Strategy(commentary_system, adverse_news_verifier=verifier)
        md = _make_market_data()
        signal = await strat.generate_signal_with_commentary(md)
        assert signal is not None
        assert signal.reasoning["news_gate_status"] == "api_unavailable"


# ----------------------------------------------------------------------
# Full signal path tests
# ----------------------------------------------------------------------
@pytest.mark.asyncio
class TestFullSignalPath_AllPass:
    async def test_full_signal_populated(self, strategy):
        md = _make_market_data()
        signal = await strategy.generate_signal_with_commentary(md)

        assert signal is not None
        assert signal.symbol == "TEST"
        assert signal.signal_type == SignalType.BUY
        assert signal.entry_price == pytest.approx(99.5)
        # stop_distance = max(2*atr=2.0, 0.5*bar_range=0.5*2.5=1.25) = 2.0
        assert signal.reasoning["stop_distance"] == pytest.approx(2.0)
        assert signal.stop_loss == pytest.approx(97.5)
        assert signal.take_profit == pytest.approx(105.5)  # 99.5 + 3*2.0
        assert signal.confidence == 0.65
        assert signal.strength == 0.65
        assert signal.reasoning["strategy"] == "oversold_bounce_v2"
        assert signal.reasoning["primary_reason"] == "oversold_bounce_v2"
        assert signal.reasoning["time_stop_minutes"] == 30
        assert signal.reasoning["scale_out_r_override"] == 0.5
        assert signal.reasoning["breakeven_atr_mult"] == 1.5
        # And the regime/sizing context is captured for audit.
        assert "regime_factor" in signal.reasoning
        assert "atr" in signal.reasoning
        assert "volume_ratio" in signal.reasoning


# Parametrized: each call fails one gate, signal must be None.
_FAIL_CASES = [
    pytest.param({"rsi": 50.0}, {}, id="gate_a_rsi_high"),
    pytest.param({"volume_ratio": 1.0}, {}, id="gate_b_low_volume"),
    # Gate C — no wick, body bigger than wick, prev_low above close.
    pytest.param(
        {"prev_low": 99.0},
        {"open_": 100.0, "high": 100.2, "low": 100.1, "close": 100.15},
        id="gate_c_no_capitulation_structure",
    ),
    # Gate D — empty lows_50.
    pytest.param({"lows_50": []}, {}, id="gate_d_no_support"),
    # Gate E — strongly down market (override via fixture-replacement test).
    pytest.param({"__regime_slope": -0.2}, {}, id="gate_e_strong_down"),
    # Gate F — fresh adverse news (override via fixture-replacement test).
    pytest.param({"__news_adverse": True}, {}, id="gate_f_adverse_news"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("ind_override,md_override", _FAIL_CASES)
async def test_full_signal_path_fail_at_each_gate(
    ind_override, md_override, commentary_system, monkeypatch,
):
    # Pull out test-control sentinels (not real indicator keys).
    regime_slope = ind_override.pop("__regime_slope", 0.0)
    news_adverse = ind_override.pop("__news_adverse", False)

    monkeypatch.setattr(
        "core.spy_regime.SpyRegimeCache.instance",
        lambda: MagicMock(ema_slope_pct=MagicMock(return_value=regime_slope)),
    )
    if news_adverse:
        verifier = _StubVerifier(
            result=_StubVerifyResult(fresh_count=3, avg_fresh_sentiment=-0.5),
        )
    else:
        verifier = _StubVerifier(result=_StubVerifyResult(fresh_count=0))

    strat = OversoldBounceV2Strategy(commentary_system, adverse_news_verifier=verifier)
    md = _make_market_data(indicators=ind_override or None, **md_override)
    signal = await strat.generate_signal_with_commentary(md)
    assert signal is None
