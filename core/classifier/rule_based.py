"""Rule-based side classifier.

v-rule-based-classifier-2026-05-13. Implements
``.claude/PLAN_symbol_side_classifier.md`` §5 — two-stage decision:

  1. Hard gates: any one fires ⇒ ``allowed_sides = ∅``
  2. Side-only hard gates: remove LONG and/or SHORT based on
     unambiguous signals (HTB removes SHORT, confirmed downtrend
     removes LONG, etc.)
  3. Soft scoring: linear combination of trend, RS, volume, sentiment
     features → ``long_score``, ``short_score``. A side is allowed if
     its score meets the threshold AND wasn't removed in step 2.

Pure function. No I/O. No engine state. Unit-testable.

This module is intentionally NOT imported by ``core/engine.py``. The
bot's live trading path remains untouched until the operator flips
``Config.USE_SIDE_CLASSIFIER``. The meta-test in
``tests/test_recent_fixes.py::TestClassifierStaysIsolated`` enforces
that invariant.
"""
from __future__ import annotations

import math
from typing import Tuple

from core.classifier.types import Side, SymbolFeatures, SymbolSideDecision

# ── Hard-gate thresholds ─────────────────────────────────────────────

# Daily ATR fraction above which the symbol is too volatile for
# systematic side selection (parabolic / news-shock regime).
_MAX_DAILY_ATR_PCT = 0.08

# Below this average daily share volume the symbol is illiquid and
# both sides are refused. (Today's price-filter sub-$5 rule lives in
# the engine; we mirror the spirit here for liquidity at the symbol
# level.)
_MIN_AVG_DAILY_VOLUME = 500_000

# ── Side-only gate thresholds ────────────────────────────────────────

_CONFIRMED_TREND_DAYS = 3      # 3 consecutive HH/LL = no contra-side
_STRONG_NEWS_THRESHOLD = 0.6   # |sentiment_7d| > this removes opposing side

# ── Soft-score weights (PLAN §5.3) ───────────────────────────────────
#
# Each weight is the contribution to its side's pre-sigmoid logit. Sign
# of the input feature is normalized so that "argues for this side"
# always yields a positive contribution.

_W_SMA_ALIGNMENT = 0.30
_W_VOLUME_CONFIRM = 0.20
_W_RS_VS_SPY = 0.15
_W_INTRADAY_TREND = 0.15
_W_NEWS_SENTIMENT = 0.10
_W_RANGE_POSITION = 0.10

# Default threshold for "this side has enough edge to be allowed."
_DEFAULT_SCORE_THRESHOLD = 0.55


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid; bounds output to (0, 1)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _hard_gates(f: SymbolFeatures) -> Tuple[bool, str]:
    """Return (forbidden, reason)."""
    if f.earnings_within_2d:
        return True, "earnings_within_2d"
    if f.halt_today:
        return True, "halt_today"
    if (f.daily_atr_pct is not None
            and f.daily_atr_pct > _MAX_DAILY_ATR_PCT):
        return True, f"daily_atr_pct={f.daily_atr_pct:.3f}_exceeds_{_MAX_DAILY_ATR_PCT}"
    if (f.avg_daily_volume_20d is not None
            and f.avg_daily_volume_20d < _MIN_AVG_DAILY_VOLUME):
        return True, (
            f"low_volume_{int(f.avg_daily_volume_20d)}_<_{_MIN_AVG_DAILY_VOLUME}"
        )
    return False, ""


def _side_only_gates(f: SymbolFeatures) -> Tuple[set, list]:
    """Return (sides_removed, reasons_list)."""
    removed = set()
    reasons = []

    if f.hard_to_borrow:
        removed.add(Side.SHORT)
        reasons.append("htb_removes_short")

    if f.lower_lows >= _CONFIRMED_TREND_DAYS:
        removed.add(Side.LONG)
        reasons.append(f"lower_lows={f.lower_lows}_removes_long")

    if f.higher_highs >= _CONFIRMED_TREND_DAYS:
        removed.add(Side.SHORT)
        reasons.append(f"higher_highs={f.higher_highs}_removes_short")

    if f.news_sentiment_7d > _STRONG_NEWS_THRESHOLD:
        removed.add(Side.SHORT)
        reasons.append(f"news_sent={f.news_sentiment_7d:.2f}_removes_short")
    elif f.news_sentiment_7d < -_STRONG_NEWS_THRESHOLD:
        removed.add(Side.LONG)
        reasons.append(f"news_sent={f.news_sentiment_7d:.2f}_removes_long")

    return removed, reasons


def _compute_scores(f: SymbolFeatures) -> Tuple[float, float, dict]:
    """Soft scoring per PLAN §5.3. Returns (long_score, short_score, dump)."""
    # Trend: SMA20 vs SMA50 alignment. +1 clean uptrend, -1 clean
    # downtrend, ±0.3 mixed (only one of {SMA, price} confirms), and
    # 0.0 if everything is at the same level (fully neutral —
    # equality must not bias either side).
    sma_alignment = 0.0
    if f.sma_20_daily is not None and f.sma_50_daily is not None:
        sma_above_50 = f.sma_20_daily > f.sma_50_daily
        sma_below_50 = f.sma_20_daily < f.sma_50_daily
        price_above_20 = f.close > f.sma_20_daily
        price_below_20 = f.close < f.sma_20_daily
        if sma_above_50 and price_above_20:
            sma_alignment = 1.0
        elif sma_below_50 and price_below_20:
            sma_alignment = -1.0
        elif sma_above_50 or price_above_20:
            sma_alignment = 0.3
        elif sma_below_50 or price_below_20:
            sma_alignment = -0.3
        # else: all equal, sma_alignment stays 0.0

    # Volume: up_vs_down_volume_5d in [0, ∞); normalize to [-1, +1].
    # Ratio 1.0 = neutral. log-scaled so 2.0 ≈ +0.7, 0.5 ≈ -0.7.
    vol_ratio = max(f.up_vs_down_volume_5d, 0.01)
    volume_norm = _clip(math.log(vol_ratio), -1.0, 1.0)

    # RS vs SPY: 5-day cum return delta in fraction. ±5% caps out.
    rs_norm = _clip(f.rs_vs_spy_5d / 0.05, -1.0, 1.0)

    # Intraday trend slope: caller normalizes; we clip.
    intraday_norm = _clip(f.intraday_trend_slope, -1.0, 1.0)

    # News sentiment in [-1, 1] already.
    news_norm = _clip(f.news_sentiment_7d, -1.0, 1.0)

    # Range position: small-negative-pct_from_sma20 favors LONG (buy
    # the dip in uptrend), small-positive favors SHORT (short the
    # bounce in downtrend). Built only when sma_20 is present.
    range_pos_long = 0.0
    range_pos_short = 0.0
    if f.sma_20_daily is not None:
        pct = (f.close - f.sma_20_daily) / f.sma_20_daily
        # Sweet spot ±2% from SMA20.
        if -0.02 < pct < 0:
            range_pos_long = -pct / 0.02
        elif 0 < pct < 0.02:
            range_pos_short = pct / 0.02

    long_logit = (
        _W_SMA_ALIGNMENT * sma_alignment
        + _W_VOLUME_CONFIRM * volume_norm
        + _W_RS_VS_SPY * rs_norm
        + _W_INTRADAY_TREND * intraday_norm
        + _W_NEWS_SENTIMENT * news_norm
        + _W_RANGE_POSITION * range_pos_long
    )
    short_logit = (
        _W_SMA_ALIGNMENT * (-sma_alignment)
        + _W_VOLUME_CONFIRM * (-volume_norm)
        + _W_RS_VS_SPY * (-rs_norm)
        + _W_INTRADAY_TREND * (-intraday_norm)
        + _W_NEWS_SENTIMENT * (-news_norm)
        + _W_RANGE_POSITION * range_pos_short
    )

    # Sigmoid temperature 4.0 — modest spread without saturating.
    # With logit range roughly [-1.0, +1.0] from the weights summing
    # to 1.0, this puts strong-trend scores around 0.88 and neutral at
    # 0.50.
    TEMPERATURE = 4.0
    long_score = _sigmoid(long_logit * TEMPERATURE)
    short_score = _sigmoid(short_logit * TEMPERATURE)

    dump = {
        "sma_alignment": sma_alignment,
        "volume_norm": volume_norm,
        "rs_norm": rs_norm,
        "intraday_norm": intraday_norm,
        "news_norm": news_norm,
        "range_pos_long": range_pos_long,
        "range_pos_short": range_pos_short,
        "long_logit": long_logit,
        "short_logit": short_logit,
        "long_score": long_score,
        "short_score": short_score,
    }
    return long_score, short_score, dump


def classify_rule_based(
    features: SymbolFeatures,
    *,
    long_threshold: float = _DEFAULT_SCORE_THRESHOLD,
    short_threshold: float = _DEFAULT_SCORE_THRESHOLD,
) -> SymbolSideDecision:
    """Apply the rule-based classifier to one symbol's features.

    Pure function — same inputs always produce same outputs. Safe to
    call from anywhere (engine signal_router, research/backtest, unit
    tests) without side effects.
    """
    # Stage 1: hard gates
    forbidden, hard_reason = _hard_gates(features)
    if forbidden:
        return SymbolSideDecision(
            symbol=features.symbol,
            allowed_sides=frozenset(),
            long_score=0.0,
            short_score=0.0,
            primary_reason=f"hard_gate:{hard_reason}",
            feature_dump={"hard_gate": hard_reason},
        )

    # Stage 2: soft scoring
    long_score, short_score, dump = _compute_scores(features)

    # Stage 3: side-only hard gates (removes sides from the candidate
    # set even if their score passes threshold)
    removed, side_reasons = _side_only_gates(features)

    # Stage 4: apply thresholds + removals
    candidates = set()
    if long_score >= long_threshold and Side.LONG not in removed:
        candidates.add(Side.LONG)
    if short_score >= short_threshold and Side.SHORT not in removed:
        candidates.add(Side.SHORT)

    # Primary reason: side-only-gate reasons first, else best score wins.
    if side_reasons:
        primary = ";".join(side_reasons)
    elif candidates == {Side.LONG, Side.SHORT}:
        primary = "range_bound:both_pass_threshold"
    elif Side.LONG in candidates:
        primary = f"long_only:score={long_score:.2f}"
    elif Side.SHORT in candidates:
        primary = f"short_only:score={short_score:.2f}"
    else:
        primary = (
            f"no_edge:long={long_score:.2f}_short={short_score:.2f}_"
            f"below_threshold={long_threshold:.2f}"
        )

    dump["side_only_reasons"] = side_reasons
    dump["removed_sides"] = sorted(s.value for s in removed)
    dump["long_threshold"] = long_threshold
    dump["short_threshold"] = short_threshold

    return SymbolSideDecision(
        symbol=features.symbol,
        allowed_sides=frozenset(candidates),
        long_score=long_score,
        short_score=short_score,
        primary_reason=primary,
        feature_dump=dump,
    )
