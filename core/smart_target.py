"""v-smart-target-2026-06-02: destination-aware take_profit selection.

Operator observation 2026-06-01 12:15 ET: "the take profit you are
placing for certain trades are meaningless, there is no way it will
reach there." Data confirmed: across 100 historical bot trades, only
5% closed at take_profit. The rest exited via stop (10%), trailing
stop (4%), or external close (84%, mix of manual closes and OCO
mechanics). The configured targets at 2.5R - 6.25 × ATR above entry
were aspirational at best.

The fix: instead of blindly computing target = entry + rr_ratio ×
stop_distance, pick from real destinations encoded in the chart:

  * Bollinger upper band   (natural resistance for mean-rev longs)
  * 20-bar high            (multi-bar resistance level)
  * 50% of recent 20-bar range from entry  (typical-move expectation)
  * rr_target              (aspirational fallback)

Try each candidate in priority order. Use the FIRST one that gives
at least 1.5R reward-to-risk. If none qualify, return None — signal
the strategy to SKIP THE TRADE rather than place an OCO target at
an unreachable level.

This module is a pure function. No engine deps. Strategies call it
inline; failure mode is "return None and skip" so a defect in this
module degrades gracefully to "no trade" rather than producing
bogus targets.

Strategy consumption (wired in this same commit):
  mean-reversion BUY:  if compute_smart_target returns None, skip
  breakout BUY:        same
  news BUY:            same

Behind Config.ENABLE_SMART_TAKE_PROFIT (default True). Toggle False
via /api/settings/trading to fall back to the old rr-target formula
without a code change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


# ────────────────────────────────────────────────────────────────────
# Public output
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SmartTarget:
    """Result of compute_smart_target.

    Strategies that want to emit a signal use this to set
    take_profit. The source + R fields are useful for log audit and
    post-trade analysis (so we can grade "did bb_upper-targeted
    trades close better than high_20-targeted ones?" later).
    """
    target: float       # absolute price
    source: str         # 'bb_upper' | 'high_20' | 'high_20_projection' | 'half_20bar_range' | 'rr_target'
    R: float            # reward-to-risk ratio (target_dist / stop_dist)


# ────────────────────────────────────────────────────────────────────
# Candidate generators — each returns (source_name, target_price)
# or None if the candidate isn't applicable for this setup.
# ────────────────────────────────────────────────────────────────────

def _candidate_bb_upper(entry: float, indicators: Dict[str, Any]) -> Optional[tuple]:
    """Bollinger upper band — strong single-bar resistance for
    mean-reverting longs. Only valid if bb_upper sits at least 0.5%
    above entry (otherwise the target would be too close)."""
    bb_upper = _safe_float(indicators.get('bb_upper'))
    if bb_upper > entry * 1.005:
        return ('bb_upper', bb_upper)
    return None


def _candidate_high_20(entry: float, indicators: Dict[str, Any]) -> Optional[tuple]:
    """20-bar high — multi-bar resistance level.

    Two regimes:
      * Entry below the 20-bar high → target the 20-bar high
        (mean-reversion / pullback setup). Use 0.2% below the level
        to avoid hitting exact resistance.
      * Entry above the 20-bar high → breakout setup. The 20-bar high
        is now support, not resistance. Project 5% above as next
        resistance estimate.
    """
    high_20 = _safe_float(indicators.get('high_20'))
    if high_20 <= 0:
        return None
    if high_20 > entry * 1.005:
        return ('high_20', high_20 * 0.998)
    if entry > high_20:
        # Breakout regime — project 5% above the broken level.
        return ('high_20_projection', high_20 * 1.05)
    return None


def _candidate_half_20bar_range(entry: float, indicators: Dict[str, Any]) -> Optional[tuple]:
    """Recent-range projection: entry + 50% of the 20-bar range.

    Captures "this stock typically moves $X in 20 bars — targeting
    half that move from entry is realistic." Useful when other
    resistance levels are unreachable.
    """
    high_20 = _safe_float(indicators.get('high_20'))
    low_20 = _safe_float(indicators.get('low_20'))
    if high_20 <= 0 or low_20 <= 0 or high_20 <= low_20:
        return None
    range_20 = high_20 - low_20
    target = entry + 0.5 * range_20
    if target > entry * 1.005:
        return ('half_20bar_range', target)
    return None


def _candidate_rr_target(entry: float, rr_ratio: float, stop_distance: float) -> tuple:
    """Aspirational fallback. ALWAYS a candidate; the floor check
    filters it out if rr_ratio × stop_distance is too tight to
    matter."""
    return ('rr_target', entry + rr_ratio * stop_distance)


# ────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────

def compute_smart_target(
    entry: float,
    stop_distance: float,
    indicators: Dict[str, Any],
    rr_ratio: float = 2.0,
    min_R: float = 1.5,
) -> Optional[SmartTarget]:
    """Pick the nearest meaningful destination above entry.

    Args:
        entry: signal's entry price
        stop_distance: |entry - stop_loss|, always positive
        indicators: market_data.indicators dict
        rr_ratio: configured R:R ratio for the rr_target fallback
        min_R: floor — if no candidate gives at least this R, skip

    Returns:
        SmartTarget with the chosen destination, source name, and R.
        None if no candidate clears the min_R floor (caller should
        skip the trade rather than place an unreachable target).
    """
    if stop_distance <= 0 or entry <= 0:
        return None

    # Build candidates in priority order. The order matters: when
    # multiple candidates clear the floor, we want to pick the most
    # meaningful one, not the largest reward.
    candidates: list = []

    # Priority 1: bb_upper — strongest single-bar resistance
    c = _candidate_bb_upper(entry, indicators)
    if c is not None:
        candidates.append(c)

    # Priority 2: 20-bar high (or projection if entry is already above)
    c = _candidate_high_20(entry, indicators)
    if c is not None:
        candidates.append(c)

    # Priority 3: half-range projection
    c = _candidate_half_20bar_range(entry, indicators)
    if c is not None:
        candidates.append(c)

    # Priority 4: rr_target — always available
    candidates.append(_candidate_rr_target(entry, rr_ratio, stop_distance))

    # Among candidates that clear the min_R floor, pick the SMALLEST
    # R that qualifies. "Smallest valid R" means "nearest reachable
    # destination that's still worth the risk." This is the central
    # design choice: we want REACHABLE, not ASPIRATIONAL.
    valid: list = []
    for source, target_price in candidates:
        dist = target_price - entry
        if dist <= 0:
            continue
        R = dist / stop_distance
        if R >= min_R:
            valid.append((R, target_price, source))

    if not valid:
        return None

    # Sort by R ascending → pick smallest valid R.
    valid.sort(key=lambda x: x[0])
    R, target_price, source = valid[0]

    return SmartTarget(
        target=round(target_price, 4),
        source=source,
        R=round(R, 2),
    )


def _safe_float(v: Any) -> float:
    """Coerce to float defensively. NaN, inf, None, and unparseable
    strings all return 0.0 — which the candidate generators treat
    as "missing data" and skip."""
    if v is None:
        return 0.0
    try:
        f = float(v)
        if f != f or f == float('inf') or f == float('-inf'):
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0
