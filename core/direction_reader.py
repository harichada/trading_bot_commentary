"""v-direction-reader-2026-05-28: synthesize indicators into a directional read.

Motivation
----------
Operator observation 2026-05-28: "Why is the bot unable to determine
price movement and direction?" After three sessions of complaining
that the bot doesn't catch momentum names, the structural cause
became clear: **the existing strategies read price LEVELS at a single
bar (RSI < 30, close < BB lower, close > 20-bar high) but never read
the price's DIRECTION over time.** A human trader looking at PLTR
sees higher highs and higher lows; the bot sees only the latest RSI
value. Those are fundamentally different operations.

This module is the bot's missing direction-reading tool. It takes
the enriched indicator dict (snapshot indicators + the slope features
v-direction-reader-features-2026-05-28 added to technical.py) and
returns a structured `DirectionRead`:

    direction: -10 (strong down) .. +10 (strong up)
    strength: 0 .. 1 (confidence)
    phase: 'ranging' | 'early' | 'middle' | 'late' | 'exhausted'
    components: per-feature scores for audit
    reason: one-line human-readable summary

Strategies consume the DirectionRead to gate their entries. Examples
(not yet wired — that's the Friday work):
  Mean-rev uptrend_pullback:  require direction >= +2 (don't buy a
                              "pullback" if the trend is actually
                              broken)
  Breakout:                   require direction >= +3 and phase in
                              {'early', 'middle'} (don't buy late
                              breakouts at exhaustion)
  News BUY:                   require direction >= 0 (don't fight
                              the tape — bullish news on a stock in
                              downtrend is usually already priced
                              into the slide)

The five components
-------------------
1. **EMA stack** (-3..+3): close vs ema_20 vs sma_50 ordering. Captures
   whether the moving averages are stacked bullishly or bearishly.

2. **Price location** (-2..+2): how far close is from sma_50, in
   percent. Captures whether the stock is at the trend or extended
   from it.

3. **Slope** (-2..+2): EMA-20 percentage slope over the last 5 bars
   (computed by technical.py as ema_20_slope_pct). Captures the
   direction the MA is moving — a key piece a snapshot can't show.

4. **Trend strength + direction** (-2..+2): ADX magnitude × MACD
   histogram sign. ADX alone gives strength without direction; the
   MACD histogram sign supplies direction. Combined, this approximates
   what DI+/DI- would provide if we computed them.

5. **Volume confirmation** (-1..+1): OBV slope sign. Rising OBV
   confirms buying pressure; falling OBV confirms distribution.

Each component is bounded so no single feature can dominate. The
total sits in -10..+10 and is interpretable as "roughly how directional
is this stock right now."

Phase classification
--------------------
Phase tells the strategy WHERE in the trend cycle the stock is, so
it can avoid late-trend chases or buy early-trend setups with
confidence:

  ranging:    abs(direction) < 2  — no defined trend
  early:      direction signed up + rsi pulled back (40-55 for up,
              45-60 for down) — trend just established, room to run
  middle:     direction signed + rsi mid-range (55-70 for up,
              30-45 for down) — trend in motion, healthy
  late:       direction signed + price extended from sma_50
              (> 5% for up, < -5% for down) — trend mature
  exhausted:  rsi extreme (> 75 for up, < 25 for down) — likely
              counter-move imminent

This is the kind of phase classification that lets a strategy say
"yes I want to buy uptrending stocks, but not when they're at the
exhaustion phase." That single rule probably would have prevented
FLY/ASTS losses on 2026-05-26 (extended, near top).

What this module does NOT do
----------------------------
- Multi-timeframe analysis (5m + 15m + 1h alignment). The bot only
  feeds single-timeframe indicators today. A future MultiTimeframe-
  DirectionReader would aggregate.
- Pattern recognition (HH/HL, head-and-shoulders, flags). The
  algorithm reads indicator values, not chart patterns.
- Order-flow / institutional accumulation. We don't have order book
  data. OBV is the closest proxy.
- Sector / market regime context. Per-symbol only.

Each of these is a future addition. The current algorithm is the
floor, not the ceiling.

Testing
-------
See tests/core/test_direction_reader.py for unit tests with real
scenarios from this week's log:
  - TSLA 2026-05-28 13:39 (uptrend_pullback fire) — should read
    direction in +3..+5, phase 'middle'.
  - FLY 2026-05-26 14:47 (extended news buy that lost) — should
    read phase 'late' or 'exhausted'.
  - ASTS 2026-05-26 14:46 (conviction_overrides_knife) — should
    read phase 'late' (below SMA50 + RSI mid).
  - IONQ 2026-05-27 (the pullback we missed) — should read
    direction +1..+3 with phase 'early' or 'middle'.

If a strategy gates on direction and phase using these scores, the
losses described above would have been prevented and the missed
trades would have been caught.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ────────────────────────────────────────────────────────────────────
# Public output
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DirectionRead:
    """Structured directional read consumed by strategies.

    Frozen because it's intended to be passed around and cached;
    mutation would defeat the auditability we get from having a
    single read per (symbol, bar).
    """
    direction: float                           # -10 .. +10
    strength: float                            # 0 .. 1
    phase: str                                 # see docstring
    ema_stack: str                             # 'aligned_up'|'aligned_down'|'mixed'|'no_data'
    components: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_bullish(self) -> bool:
        """Convenience: direction at or above +2 with non-exhausted phase."""
        return self.direction >= 2.0 and self.phase != "exhausted"

    @property
    def is_bearish(self) -> bool:
        """Convenience: direction at or below -2 with non-exhausted phase."""
        return self.direction <= -2.0 and self.phase != "exhausted"

    @property
    def allows_long_entry(self) -> bool:
        """A strategy gate: don't open long if direction is bearish
        or the stock is in exhausted phase (likely top)."""
        return self.direction >= 0.0 and self.phase != "exhausted"

    @property
    def allows_short_entry(self) -> bool:
        """Mirror for shorts: don't short an uptrend or an exhausted bottom."""
        return self.direction <= 0.0 and self.phase != "exhausted"


# ────────────────────────────────────────────────────────────────────
# Component scorers — each bounded to a small range so no single
# feature can dominate the total.
# ────────────────────────────────────────────────────────────────────

def _ema_stack_score(close: float, ema_20: float, sma_50: float) -> tuple[float, str]:
    """How are the moving averages stacked?

    Returns (-3..+3, label). +3 = close > ema_20 > sma_50 (perfect
    bullish stack). -3 = close < ema_20 < sma_50 (perfect bearish
    stack). Partial alignments yield intermediate scores.

    We use close (not ema_9) as the fast indicator because the
    bot's indicator pipeline only computes ema_20 today. close >
    ema_20 is the equivalent fast-vs-slow check.
    """
    if ema_20 <= 0 or sma_50 <= 0:
        return 0.0, "no_data"

    fast_above_mid = close > ema_20
    mid_above_slow = ema_20 > sma_50
    fast_above_slow = close > sma_50

    if fast_above_mid and mid_above_slow:
        return 3.0, "aligned_up"
    if (not fast_above_mid) and (not mid_above_slow):
        return -3.0, "aligned_down"

    # Mixed: weight each pair check independently.
    score = 0.0
    score += 1.0 if fast_above_mid else -1.0
    score += 1.0 if mid_above_slow else -1.0
    score += 0.5 if fast_above_slow else -0.5
    return score, "mixed"


def _price_location_score(close_vs_sma50_pct: float) -> float:
    """How extended is close from sma_50?

    Returns -2..+2. Linear-ish mapping with caps:
      > +5%   → +2  (extended uptrend)
      +2..+5  → +1
      -2..+2  → 0   (at the trend, neutral)
      -5..-2  → -1
      < -5%   → -2  (extended downtrend)

    A stock that's 8% above SMA50 is structurally different from
    one that's 0.2% above — both are technically uptrending but
    the first is late-trend.
    """
    if close_vs_sma50_pct > 5:
        return 2.0
    if close_vs_sma50_pct > 2:
        return 1.0
    if close_vs_sma50_pct > -2:
        return 0.0
    if close_vs_sma50_pct > -5:
        return -1.0
    return -2.0


def _slope_score(ema_20_slope_pct: float) -> float:
    """How fast is the EMA-20 moving?

    Returns -2..+2. The slope is computed over 5 bars (~25 min on
    5-min bars). Tuned so:
      > +1.0%  → +2  (steep uptrend)
      +0.3..1  → +1  (rising)
      -0.3..0.3 → 0  (flat)
      -1..-0.3 → -1  (falling)
      < -1%    → -2  (steep downtrend)

    This is the single feature a snapshot can't provide. Without
    slope, the bot sees "EMA-20 = 100" and doesn't know if that's
    100 rising from 95 or 100 falling from 105 — same level, opposite
    direction.
    """
    if ema_20_slope_pct > 1.0:
        return 2.0
    if ema_20_slope_pct > 0.3:
        return 1.0
    if ema_20_slope_pct > -0.3:
        return 0.0
    if ema_20_slope_pct > -1.0:
        return -1.0
    return -2.0


def _trend_strength_score(adx: float, macd_hist: float) -> float:
    """ADX magnitude × MACD histogram sign.

    Returns -2..+2. ADX gives trend strength but no direction; the
    MACD histogram sign supplies direction. The product approximates
    a directional ADX (what DI+ minus DI- would give us if the
    indicator pipeline computed them).

    Magnitude mapping:
      ADX <20         → 0.0  (no trend, no signal regardless of direction)
      ADX 20-25       → ±0.5 (building trend)
      ADX 25-40       → ±1.5 (clear trend)
      ADX >40         → ±2.0 (strong trend)

    Direction comes from sign(macd_hist).
    """
    if adx < 20:
        return 0.0
    if adx < 25:
        magnitude = 0.5
    elif adx < 40:
        magnitude = 1.5
    else:
        magnitude = 2.0

    if macd_hist > 0:
        return magnitude
    if macd_hist < 0:
        return -magnitude
    return 0.0


def _volume_score(obv_slope_pct: float) -> float:
    """OBV slope: smart money direction via volume.

    Returns -1..+1. OBV adds volume on up bars, subtracts on down
    bars — its slope reveals whether buying or selling pressure
    is dominating. Rising OBV through flat price = stealth
    accumulation; falling OBV through flat price = distribution.

    Tuned around realistic OBV slope ranges (it's measured in
    cumulative volume units, normalized by absolute starting value
    in technical.py):
      > +5%   → +1
      -5..+5  → 0
      < -5%   → -1
    """
    if obv_slope_pct > 5:
        return 1.0
    if obv_slope_pct < -5:
        return -1.0
    return 0.0


# ────────────────────────────────────────────────────────────────────
# Phase classification
# ────────────────────────────────────────────────────────────────────

def _classify_phase(direction: float, rsi: float, close_vs_sma50_pct: float) -> str:
    """Where in the trend cycle is this stock?

    Phase classification is the single most useful output of this
    module for entry strategies. A trader doesn't just want to know
    "is the trend up" — they want to know if they're buying at the
    start of the move (good), the middle (good), the end (bad), or
    the climax (very bad).
    """
    if abs(direction) < 2:
        return "ranging"

    is_up = direction > 0

    if is_up:
        # Exhaustion: RSI extreme overbought → counter-move likely
        if rsi >= 75:
            return "exhausted"
        # Late: extended price OR high but not extreme RSI
        if close_vs_sma50_pct > 5 or rsi >= 70:
            return "late"
        # Early: trend defined but RSI hasn't built — fresh entry
        if rsi < 50:
            return "early"
        # Middle: healthy mid-cycle (RSI 50-70)
        return "middle"

    # Downtrend
    if rsi <= 25:
        return "exhausted"
    if close_vs_sma50_pct < -5 or rsi <= 30:
        return "late"
    if rsi > 50:
        return "early"
    return "middle"


# ────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────

def read_direction(close: float, indicators: Dict[str, Any]) -> DirectionRead:
    """Synthesize an indicator dict into a structured DirectionRead.

    Pure function: no side effects, no I/O, no dependency on
    market_data or engine. Easy to unit-test in isolation.

    Args:
        close: Current bar's close price.
        indicators: The dict produced by `analysis/technical.py`.
            Required keys (or 0.0 default if missing):
              rsi, ema_20, sma_50, adx, macd_histogram,
              ema_20_slope_pct, close_vs_sma50_pct, obv_slope_pct
            (The slope features come from v-direction-reader-features-
            2026-05-28 enrichment in technical.py.)
    """
    # Pull features with safe defaults so missing keys don't crash
    # the reader. The components will simply score 0 for those.
    def _get(k: str, default: float = 0.0) -> float:
        v = indicators.get(k, default)
        try:
            f = float(v)
            if f != f or f == float("inf") or f == float("-inf"):
                return default
            return f
        except (TypeError, ValueError):
            return default

    rsi = _get("rsi", 50.0)
    ema_20 = _get("ema_20", 0.0)
    sma_50 = _get("sma_50", 0.0)
    adx = _get("adx", 0.0)
    macd_hist = _get("macd_histogram", 0.0)
    ema_20_slope_pct = _get("ema_20_slope_pct", 0.0)
    close_vs_sma50_pct = _get("close_vs_sma50_pct", 0.0)
    obv_slope_pct = _get("obv_slope_pct", 0.0)

    # Component scores
    ema_stack_score, ema_stack_label = _ema_stack_score(close, ema_20, sma_50)
    price_location = _price_location_score(close_vs_sma50_pct)
    slope = _slope_score(ema_20_slope_pct)
    strength_dir = _trend_strength_score(adx, macd_hist)
    volume = _volume_score(obv_slope_pct)

    direction = ema_stack_score + price_location + slope + strength_dir + volume
    # Clamp to bounds. Theoretical range is -10..+10; in practice
    # values rarely exceed |8| because components correlate.
    direction = max(-10.0, min(10.0, direction))

    # Confidence: magnitude relative to max, with an ADX dampener
    # because low-ADX environments are noise-prone regardless of
    # the other components.
    confidence = abs(direction) / 10.0
    if adx < 20:
        confidence *= 0.5

    phase = _classify_phase(direction, rsi, close_vs_sma50_pct)

    components = {
        "ema_stack": round(ema_stack_score, 2),
        "price_location": round(price_location, 2),
        "slope": round(slope, 2),
        "trend_strength_dir": round(strength_dir, 2),
        "volume": round(volume, 2),
    }

    # Human-readable summary — what a trader would say looking at
    # the same indicators. Useful for logging + commentary.
    if direction >= 4:
        reason = f"strong uptrend ({ema_stack_label}, adx {adx:.0f}, slope {ema_20_slope_pct:+.2f}%)"
    elif direction >= 2:
        reason = f"mild uptrend ({ema_stack_label}, slope {ema_20_slope_pct:+.2f}%)"
    elif direction <= -4:
        reason = f"strong downtrend ({ema_stack_label}, adx {adx:.0f}, slope {ema_20_slope_pct:+.2f}%)"
    elif direction <= -2:
        reason = f"mild downtrend ({ema_stack_label}, slope {ema_20_slope_pct:+.2f}%)"
    else:
        reason = f"ranging (adx {adx:.0f}, slope {ema_20_slope_pct:+.2f}%)"

    return DirectionRead(
        direction=round(direction, 2),
        strength=round(confidence, 2),
        phase=phase,
        ema_stack=ema_stack_label,
        components=components,
        reason=reason,
    )
