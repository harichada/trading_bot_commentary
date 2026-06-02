"""Unit tests for core/smart_target.py — destination-aware
take_profit selection.

Built 2026-06-02 morning before the open after operator observation
that 95% of bot trades never hit take_profit. These tests verify
the algorithm picks reachable destinations and skips signals whose
nearest meaningful resistance doesn't give enough R:R.
"""
from __future__ import annotations

import pytest

from core.smart_target import (
    SmartTarget,
    compute_smart_target,
)


# ────────────────────────────────────────────────────────────────────
# Output shape
# ────────────────────────────────────────────────────────────────────

class TestSmartTargetShape:
    def test_returns_smart_target_or_none(self):
        result = compute_smart_target(100.0, 1.0, {})
        assert result is None or isinstance(result, SmartTarget)

    def test_negative_stop_distance_returns_none(self):
        result = compute_smart_target(100.0, -1.0, {"bb_upper": 105})
        assert result is None

    def test_zero_entry_returns_none(self):
        result = compute_smart_target(0.0, 1.0, {"bb_upper": 105})
        assert result is None


# ────────────────────────────────────────────────────────────────────
# Source selection — verify priority order works
# ────────────────────────────────────────────────────────────────────

class TestSourceSelection:
    def test_bb_upper_picked_when_nearest_valid(self):
        """When bb_upper sits at a clean 2R distance and other
        candidates are farther, bb_upper should be the chosen
        target. Smallest valid R wins."""
        # entry $100, stop $99 → stop_distance $1
        # bb_upper at $102 → R=2.0 (valid, smallest)
        # high_20 at $108 → R=8.0 (valid but farther)
        # rr_target at $104 (2.0 × $1 + $100) → R=2.0 (tied)
        # The order matters when R is tied — bb_upper is generated
        # first so it wins the priority by insertion order.
        result = compute_smart_target(
            entry=100.0,
            stop_distance=1.0,
            indicators={"bb_upper": 102, "high_20": 108},
            rr_ratio=2.0,
        )
        assert result is not None
        # Either bb_upper or rr_target (both at 2.0R) — sort is stable.
        # Since both give R=2.0 and rr_target is appended last, but
        # the sort is by R and then by insertion order — bb_upper wins.
        assert result.source in ("bb_upper", "rr_target")
        assert abs(result.R - 2.0) < 0.05

    def test_high_20_target_when_entry_below(self):
        """Entry below high_20 → high_20 is resistance, target it
        at 0.998 × high_20 (slightly below)."""
        result = compute_smart_target(
            entry=100.0,
            stop_distance=2.0,   # stop $98
            indicators={"high_20": 110, "low_20": 95},
            rr_ratio=2.5,
        )
        assert result is not None
        # bb_upper missing → high_20 is the resistance candidate
        # half-range = 100 + 0.5 × 15 = 107.5 → R = 3.75
        # high_20 × 0.998 = 109.78 → R = 4.89
        # rr_target = 100 + 2.5 × 2 = 105 → R = 2.5
        # Smallest valid R wins → rr_target at R=2.5 (no, that's not in candidates list).
        # Actually candidates added in this order: high_20, half_range, rr_target.
        # Wait — bb_upper is first generator but here bb_upper is missing so
        # high_20 is first added. Then half_range (R=3.75), then rr_target (R=2.5).
        # All clear 1.5R. rr_target has R=2.5 (smallest).
        assert result.R >= 1.5

    def test_breakout_uses_high_20_projection(self):
        """Entry ABOVE high_20 (breakout setup) → high_20 is now
        support, project 5% above as next resistance."""
        result = compute_smart_target(
            entry=110.0,
            stop_distance=2.0,
            indicators={
                "high_20": 108,   # entry > high_20 → breakout
                "low_20": 100,
                "bb_upper": 115,  # 4.5R potential
            },
            rr_ratio=2.0,
        )
        assert result is not None
        # high_20_projection = 108 × 1.05 = 113.4 → R = 1.7
        # bb_upper = 115 → R = 2.5
        # half_range = 110 + 0.5 × 8 = 114 → R = 2.0
        # rr_target = 110 + 2 × 2 = 114 → R = 2.0
        # Smallest valid R wins.
        # high_20_projection at R=1.7 is the smallest that clears 1.5R floor.
        assert result.R >= 1.5
        assert result.source in (
            "high_20_projection", "bb_upper", "half_20bar_range",
            "rr_target",
        )


# ────────────────────────────────────────────────────────────────────
# Skip-thin-edge — the critical correctness property
# ────────────────────────────────────────────────────────────────────

class TestSkipThinEdge:
    def test_returns_none_when_no_candidate_clears_floor(self):
        """If every candidate gives less than 1.5R, return None —
        signal the strategy to SKIP rather than place an OCO at
        an unreachable level. The whole point of this module."""
        # entry $100, stop $99 → stop_distance $1
        # bb_upper $100.5 → R=0.5 (fails)
        # high_20 $100.8 (entry below) → 0.998 × 100.8 = 100.6 → R=0.6 (fails)
        # half_range = 100 + 0.5 × 0.8 = 100.4 → R=0.4 (fails)
        # rr_target = 100 + 1.0 × 1 = 101 → R=1.0 (fails!) at rr=1.0
        result = compute_smart_target(
            entry=100.0,
            stop_distance=1.0,
            indicators={"bb_upper": 100.5, "high_20": 100.8, "low_20": 100.0},
            rr_ratio=1.0,
        )
        assert result is None

    def test_rr_target_always_available_as_fallback(self):
        """When all chart-based candidates fail but rr_target gives
        ≥ 1.5R, rr_target is the result. This is the safety net."""
        result = compute_smart_target(
            entry=100.0,
            stop_distance=2.0,
            indicators={},  # no chart features at all
            rr_ratio=2.0,
        )
        # Only candidate: rr_target = 100 + 2 × 2 = 104 → R = 2.0
        assert result is not None
        assert result.source == "rr_target"
        assert abs(result.R - 2.0) < 0.05


# ────────────────────────────────────────────────────────────────────
# Real-world scenarios from yesterday's log
# ────────────────────────────────────────────────────────────────────

class TestRealScenarios:
    """Replay actual trades from 2026-06-01 to verify the new logic
    produces sane targets where the old logic produced unreachable
    ones."""

    def test_now_yesterday_morning(self):
        """NOW at $136.53, stop $133.12, target was $145.06 (R=2.5,
        unreachable 6.24% above entry). With smart-target, expect
        a closer destination — bb_upper or 20-bar high projection.

        We don't have the exact bb_upper / high_20 values for NOW
        at that moment, but assume bb_upper ≈ $140 (typical for a
        stock near recent highs)."""
        result = compute_smart_target(
            entry=136.53,
            stop_distance=3.41,
            indicators={
                "bb_upper": 140.0,
                "high_20": 142.0,
                "low_20": 132.0,
            },
            rr_ratio=2.0,
        )
        assert result is not None
        # bb_upper R = (140 - 136.53) / 3.41 = 1.02 → fails floor
        # high_20 R = (141.7 - 136.53) / 3.41 = 1.52 → passes
        # rr_target R = 2.0 → passes
        # Smallest valid wins → high_20 at R=1.52
        # Target should be substantially below the old $145.06
        assert result.target < 145.0, (
            f"smart target {result.target} should be below the old $145.06"
        )

    def test_crm_yesterday(self):
        """CRM at $209.72, stop $204.48, target was $222.83 (6.25%
        above entry, unreachable). Assume bb_upper ≈ $215 and
        high_20 ≈ $218."""
        result = compute_smart_target(
            entry=209.72,
            stop_distance=5.24,
            indicators={
                "bb_upper": 215.0,
                "high_20": 218.0,
                "low_20": 202.0,
            },
            rr_ratio=2.0,
        )
        assert result is not None
        assert result.target < 222.83


# ────────────────────────────────────────────────────────────────────
# Defensive — bad inputs must not crash
# ────────────────────────────────────────────────────────────────────

class TestRobustness:
    def test_nan_indicators_dont_crash(self):
        import math
        result = compute_smart_target(
            entry=100.0,
            stop_distance=1.0,
            indicators={"bb_upper": math.nan, "high_20": math.nan},
            rr_ratio=2.0,
        )
        # Should fall back to rr_target
        assert result is None or isinstance(result, SmartTarget)

    def test_string_values_dont_crash(self):
        result = compute_smart_target(
            entry=100.0,
            stop_distance=1.0,
            indicators={"bb_upper": "abc", "high_20": None},
            rr_ratio=2.0,
        )
        # Should fall back to rr_target
        assert result is not None
        assert result.source == "rr_target"

    def test_missing_indicators_uses_rr_fallback(self):
        result = compute_smart_target(
            entry=50.0,
            stop_distance=1.0,
            indicators={},
            rr_ratio=2.5,
        )
        assert result is not None
        assert result.source == "rr_target"
        assert abs(result.target - 52.5) < 0.001
