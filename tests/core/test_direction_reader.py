"""Unit tests for core/direction_reader.py.

Test approach: feed the algorithm REAL scenario inputs (price, RSI,
EMAs, ADX, etc.) from this week's bot log and assert the output
matches the trader-intuitive read of those setups.

This is the v-direction-reader-2026-05-28 module — the "deep dive"
fix for the operator's complaint that the bot couldn't determine
price movement and direction.

Each test corresponds to a real trade or signal from the log so
future-me can grep the timestamp and verify the algorithm's read
against what actually happened.
"""
from __future__ import annotations

import pytest

from core.direction_reader import (
    DirectionRead,
    read_direction,
)


# ────────────────────────────────────────────────────────────────────
# Output shape tests
# ────────────────────────────────────────────────────────────────────

class TestDirectionReadShape:
    def test_returns_DirectionRead(self):
        result = read_direction(close=100.0, indicators={})
        assert isinstance(result, DirectionRead)

    def test_direction_in_range(self):
        result = read_direction(close=100.0, indicators={})
        assert -10.0 <= result.direction <= 10.0

    def test_strength_in_range(self):
        result = read_direction(close=100.0, indicators={})
        assert 0.0 <= result.strength <= 1.0

    def test_phase_is_valid_label(self):
        result = read_direction(close=100.0, indicators={})
        assert result.phase in {"ranging", "early", "middle", "late", "exhausted"}

    def test_ema_stack_is_valid_label(self):
        result = read_direction(close=100.0, indicators={})
        assert result.ema_stack in {"aligned_up", "aligned_down", "mixed", "no_data"}

    def test_empty_indicators_gives_no_data_stack(self):
        """Missing ema_20 / sma_50 should produce 'no_data' stack
        and a 'ranging' phase, not crash."""
        result = read_direction(close=100.0, indicators={})
        assert result.ema_stack == "no_data"
        assert result.phase == "ranging"
        assert result.direction == 0.0


# ────────────────────────────────────────────────────────────────────
# Component-level correctness — the building blocks before scenarios
# ────────────────────────────────────────────────────────────────────

class TestEmaStackComponent:
    def test_perfect_bullish_stack_scores_plus_3(self):
        """close > ema_20 > sma_50 — full bullish alignment."""
        r = read_direction(
            close=110.0,
            indicators={"ema_20": 105.0, "sma_50": 100.0, "rsi": 60},
        )
        assert r.components["ema_stack"] == 3.0
        assert r.ema_stack == "aligned_up"

    def test_perfect_bearish_stack_scores_minus_3(self):
        r = read_direction(
            close=90.0,
            indicators={"ema_20": 95.0, "sma_50": 100.0, "rsi": 40},
        )
        assert r.components["ema_stack"] == -3.0
        assert r.ema_stack == "aligned_down"

    def test_mixed_stack_partial(self):
        """close > ema_20 but ema_20 < sma_50 — mixed, partial up."""
        r = read_direction(
            close=98.0,
            indicators={"ema_20": 97.0, "sma_50": 100.0, "rsi": 50},
        )
        assert r.ema_stack == "mixed"
        # Score should be in range — fast above mid (+1), mid below
        # slow (-1), fast below slow (-0.5) = -0.5
        assert -1.0 <= r.components["ema_stack"] <= 0.5


class TestSlopeComponent:
    def test_steep_uptrend_slope_scores_plus_2(self):
        r = read_direction(
            close=100.0,
            indicators={
                "ema_20": 100, "sma_50": 100, "rsi": 60,
                "ema_20_slope_pct": 1.5,
            },
        )
        assert r.components["slope"] == 2.0

    def test_flat_slope_scores_0(self):
        r = read_direction(
            close=100.0,
            indicators={
                "ema_20": 100, "sma_50": 100, "rsi": 50,
                "ema_20_slope_pct": 0.0,
            },
        )
        assert r.components["slope"] == 0.0

    def test_steep_downtrend_slope_scores_minus_2(self):
        r = read_direction(
            close=100.0,
            indicators={
                "ema_20": 100, "sma_50": 100, "rsi": 40,
                "ema_20_slope_pct": -1.5,
            },
        )
        assert r.components["slope"] == -2.0


class TestTrendStrengthDirComponent:
    def test_low_adx_no_signal_regardless_of_macd(self):
        r = read_direction(
            close=100.0,
            indicators={
                "ema_20": 100, "sma_50": 100, "rsi": 50,
                "adx": 15, "macd_histogram": 0.5,
            },
        )
        assert r.components["trend_strength_dir"] == 0.0

    def test_strong_adx_positive_macd_hist_scores_plus_2(self):
        r = read_direction(
            close=100.0,
            indicators={
                "ema_20": 100, "sma_50": 100, "rsi": 60,
                "adx": 45, "macd_histogram": 0.5,
            },
        )
        assert r.components["trend_strength_dir"] == 2.0

    def test_strong_adx_negative_macd_hist_scores_minus_2(self):
        r = read_direction(
            close=100.0,
            indicators={
                "ema_20": 100, "sma_50": 100, "rsi": 40,
                "adx": 45, "macd_histogram": -0.5,
            },
        )
        assert r.components["trend_strength_dir"] == -2.0


class TestVolumeComponent:
    def test_strong_obv_uptrend_scores_plus_1(self):
        r = read_direction(
            close=100.0,
            indicators={
                "ema_20": 100, "sma_50": 100, "rsi": 55,
                "obv_slope_pct": 10.0,
            },
        )
        assert r.components["volume"] == 1.0

    def test_strong_obv_downtrend_scores_minus_1(self):
        r = read_direction(
            close=100.0,
            indicators={
                "ema_20": 100, "sma_50": 100, "rsi": 45,
                "obv_slope_pct": -10.0,
            },
        )
        assert r.components["volume"] == -1.0


# ────────────────────────────────────────────────────────────────────
# Real-world scenarios from this week's log
# ────────────────────────────────────────────────────────────────────

class TestRealScenarios:
    """Each scenario maps to a real signal/trade so we can verify
    the algorithm reads what a trader would have read."""

    def test_tsla_uptrend_pullback_2026_05_28_13_39(self):
        """TSLA fired at 13:39 today via uptrend_pullback path. Inputs:
        close $441.94, RSI 54.81, near BB lower (+0.22%), above SMA50.
        TSLA was in a defined uptrend pulling back to support.

        Expected read: direction in +3..+5, phase 'early' or 'middle'
        (RSI 54 is mid-range, price near support so not extended)."""
        r = read_direction(
            close=441.94,
            indicators={
                "rsi": 54.81,
                "ema_20": 445.0,           # close slightly below ema_20 (pullback)
                "sma_50": 430.0,           # below close (uptrend confirmed)
                "adx": 28.0,               # clear trend
                "macd_histogram": 0.2,     # bullish histogram
                "ema_20_slope_pct": 0.8,   # ema_20 rising
                "close_vs_sma50_pct": 2.78,  # 2.78% above sma_50
                "obv_slope_pct": 8.0,      # accumulating
            },
        )
        # Should be bullish, not exhausted, suitable for long entry
        assert r.direction >= 2.0, f"expected mild bullish, got {r.direction}"
        assert r.phase in {"early", "middle"}, f"phase was {r.phase}"
        assert r.allows_long_entry, "should permit long entry"

    def test_fly_news_chase_2026_05_26_14_47_should_warn(self):
        """FLY fired BUY at 14:47 on bullish news at $60.93, stopped
        out 13 minutes later at $59.07 (-$72.50). The pattern: stock
        was extended above its MAs, RSI high, late in the move when
        the news triggered the entry.

        Expected: phase 'late' or 'exhausted' so a direction-gated
        strategy would have REFUSED to enter."""
        r = read_direction(
            close=60.93,
            indicators={
                "rsi": 72.0,                # near overbought
                "ema_20": 58.5,             # close well above ema_20
                "sma_50": 55.0,             # close well above sma_50 (extended)
                "adx": 32.0,                # strong trend
                "macd_histogram": 0.3,
                "ema_20_slope_pct": 1.2,    # steep uptrend
                "close_vs_sma50_pct": 10.78, # 10.78% above sma_50 — extended
                "obv_slope_pct": 6.0,
            },
        )
        # Direction is up, but the phase warns it's late / exhausted
        assert r.direction > 0
        assert r.phase in {"late", "exhausted"}, (
            f"expected late/exhausted at +10% from sma_50, got {r.phase}"
        )

    def test_asts_falling_knife_buy_2026_05_26_14_46_should_warn(self):
        """ASTS at 14:46 — news bypassed the falling-knife guard via
        conviction_overrides_knife. Close $121.42 BELOW sma_50 of
        $124.55, MACD negative. Stopped out 23 min later.

        Expected: bearish direction or at worst 'mixed' — should NOT
        read as a bullish entry setup."""
        r = read_direction(
            close=121.42,
            indicators={
                "rsi": 48.0,
                "ema_20": 123.0,            # close below ema_20
                "sma_50": 124.55,           # close below sma_50 (downtrend)
                "adx": 22.0,
                "macd_histogram": -0.65,    # negative MACD
                "ema_20_slope_pct": -0.4,   # falling
                "close_vs_sma50_pct": -2.5,
                "obv_slope_pct": -3.0,
            },
        )
        # Should NOT be bullish — direction should be negative or near zero
        assert r.direction <= 1.0, (
            f"ASTS chart read should not be strongly bullish; got {r.direction}"
        )
        assert not r.is_bullish

    def test_ionq_missed_pullback_2026_05_27(self):
        """IONQ pullback the operator was frustrated about missing.
        Approximate setup: RSI 34.66, close 60.01, near BB lower
        (+0.12%), in uptrend (above sma_50).

        Expected: direction positive, phase 'early' (RSI <50 in
        uptrend = early-cycle entry — good setup)."""
        r = read_direction(
            close=60.01,
            indicators={
                "rsi": 34.66,
                "ema_20": 60.5,
                "sma_50": 58.0,             # close above sma_50
                "adx": 26.0,
                "macd_histogram": 0.05,
                "ema_20_slope_pct": 0.5,
                "close_vs_sma50_pct": 3.46,
                "obv_slope_pct": 4.0,
            },
        )
        assert r.direction >= 1.0, f"IONQ should read as mild uptrend, got {r.direction}"
        # RSI 34 in uptrend → phase 'early' per the classification
        assert r.phase in {"early", "middle"}, f"phase was {r.phase}"
        assert r.allows_long_entry


# ────────────────────────────────────────────────────────────────────
# Phase classification edge cases
# ────────────────────────────────────────────────────────────────────

class TestPhaseClassification:
    def _build(self, **kw) -> DirectionRead:
        """Helper: bullish ema_20 + sma_50 stack + tunable rest."""
        base = {
            "ema_20": 105.0, "sma_50": 100.0,
            "adx": 30.0, "macd_histogram": 0.2,
            "ema_20_slope_pct": 0.7,
            "obv_slope_pct": 3.0,
        }
        base.update(kw)
        return read_direction(close=base.pop("close", 110.0), indicators=base)

    def test_rsi_85_in_uptrend_is_exhausted(self):
        r = self._build(rsi=85, close_vs_sma50_pct=10)
        assert r.phase == "exhausted"

    def test_far_above_sma50_is_late(self):
        r = self._build(rsi=65, close_vs_sma50_pct=8)
        assert r.phase == "late"

    def test_rsi_low_in_uptrend_is_early(self):
        r = self._build(rsi=42, close_vs_sma50_pct=2)
        assert r.phase == "early"

    def test_rsi_mid_in_uptrend_is_middle(self):
        r = self._build(rsi=60, close_vs_sma50_pct=2)
        assert r.phase == "middle"

    def test_no_trend_is_ranging(self):
        r = read_direction(
            close=100.0,
            indicators={
                "ema_20": 100.5, "sma_50": 100.2,   # tightly mixed MAs
                "rsi": 50,
                "adx": 12,                          # low ADX = ranging
                "macd_histogram": 0.0,
                "ema_20_slope_pct": 0.0,
                "obv_slope_pct": 0.0,
            },
        )
        assert r.phase == "ranging"


# ────────────────────────────────────────────────────────────────────
# Robustness — bad data shouldn't crash the reader
# ────────────────────────────────────────────────────────────────────

class TestRobustness:
    def test_nan_in_indicators_doesnt_crash(self):
        import math
        r = read_direction(
            close=100.0,
            indicators={
                "rsi": math.nan,
                "ema_20": math.nan,
                "sma_50": math.nan,
            },
        )
        assert isinstance(r, DirectionRead)

    def test_inf_in_indicators_doesnt_crash(self):
        r = read_direction(
            close=100.0,
            indicators={"ema_20": float("inf"), "sma_50": float("-inf")},
        )
        assert isinstance(r, DirectionRead)

    def test_string_values_dont_crash(self):
        """Some downstream consumers stringify numbers. Be defensive."""
        r = read_direction(
            close=100.0,
            indicators={"rsi": "55.0", "ema_20": "100", "sma_50": "98"},
        )
        assert isinstance(r, DirectionRead)

    def test_missing_optional_keys_uses_neutral_defaults(self):
        """Only sma_50 + ema_20 + rsi provided. All slope/volume
        features missing. Should still produce a valid read."""
        r = read_direction(
            close=110.0,
            indicators={"ema_20": 105, "sma_50": 100, "rsi": 60},
        )
        # EMA stack should score (no slope/strength/volume signals)
        assert r.ema_stack == "aligned_up"
        # Direction should reflect just the stack + price location
        assert r.direction > 0


# ────────────────────────────────────────────────────────────────────
# Strategy-gate convenience properties
# ────────────────────────────────────────────────────────────────────

class TestStrategyGates:
    def test_strong_uptrend_allows_long_blocks_short(self):
        r = read_direction(
            close=110.0,
            indicators={
                "ema_20": 105, "sma_50": 100, "rsi": 60,
                "adx": 30, "macd_histogram": 0.3,
                "ema_20_slope_pct": 0.8, "close_vs_sma50_pct": 3,
            },
        )
        assert r.allows_long_entry
        assert not r.allows_short_entry
        assert r.is_bullish

    def test_strong_downtrend_blocks_long_allows_short(self):
        r = read_direction(
            close=90.0,
            indicators={
                "ema_20": 95, "sma_50": 100, "rsi": 40,
                "adx": 30, "macd_histogram": -0.3,
                "ema_20_slope_pct": -0.8, "close_vs_sma50_pct": -3,
            },
        )
        assert r.allows_short_entry
        assert not r.allows_long_entry
        assert r.is_bearish

    def test_exhausted_uptrend_blocks_long(self):
        """The whole point: don't buy at the top."""
        r = read_direction(
            close=110.0,
            indicators={
                "ema_20": 105, "sma_50": 100, "rsi": 85,
                "adx": 35, "macd_histogram": 0.4,
                "ema_20_slope_pct": 1.5, "close_vs_sma50_pct": 10,
            },
        )
        # direction is positive but phase is exhausted
        assert r.direction > 0
        assert r.phase == "exhausted"
        assert not r.allows_long_entry, (
            "exhausted phase must block long entries even if direction up"
        )
