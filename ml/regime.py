"""7-state regime classifier with median-split trend/range cohorts.

States:
    trend_up_low_vol    trend_up_high_vol
    trend_dn_low_vol    trend_dn_high_vol
    range_tight         range_wide
    chop

Methodology:
    ADX (Wilder, *New Concepts in Technical Trading Systems*, 1978):
        > 25 → trending; ≤ 25 → ranging.
    Choppiness Index (Bill Dreiss):
        > 61.8 → chop (Fibonacci 0.618 above 50%). Takes priority over ADX.
    Trend direction:
        sign(EMA_fast − EMA_slow), default fast=50 / slow=200.
    Realized vol:
        EWMA(20) of squared 1-bar log-returns, square-rooted.
    Cohort split:
        rv > rolling-median rv within trending bars  → trend_*_high_vol
        rv ≤ rolling-median rv within trending bars  → trend_*_low_vol
        rv > rolling-median rv within ranging  bars  → range_wide
        rv ≤ rolling-median rv within ranging  bars  → range_tight
    No-look-ahead invariant:
        every feature at bar t uses only data ≤ t-1.

Outputs the precomputed feature DataFrame and a per-row ``classify_regime``
callable so the P3 runner can align labels to test-event timestamps via
``.loc[event_time]`` without re-deriving features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Default thresholds. Configurable via pro_trading_config.yaml::regime_detector.
DEFAULT_ADX_PERIOD = 14
DEFAULT_CI_PERIOD = 14
DEFAULT_EMA_FAST = 50
DEFAULT_EMA_SLOW = 200
DEFAULT_RV_PERIOD = 20
DEFAULT_RV_MEDIAN_LOOKBACK = 252
DEFAULT_ADX_TRENDING_THRESHOLD = 25.0
DEFAULT_CHOP_INDEX_THRESHOLD = 61.8

SEVEN_STATES = (
    "trend_up_low_vol", "trend_up_high_vol",
    "trend_dn_low_vol", "trend_dn_high_vol",
    "range_tight", "range_wide", "chop",
)


@dataclass(frozen=True)
class RegimeThresholds:
    adx_trending: float = DEFAULT_ADX_TRENDING_THRESHOLD
    chop_index: float = DEFAULT_CHOP_INDEX_THRESHOLD
    adx_period: int = DEFAULT_ADX_PERIOD
    ci_period: int = DEFAULT_CI_PERIOD
    ema_fast: int = DEFAULT_EMA_FAST
    ema_slow: int = DEFAULT_EMA_SLOW
    rv_period: int = DEFAULT_RV_PERIOD
    rv_median_lookback: int = DEFAULT_RV_MEDIAN_LOOKBACK


# ---------------------------------------------------------------------------
# Indicator primitives — all strictly past-only (use .shift(1) where needed).
# ---------------------------------------------------------------------------

def _wilder_smooth(s: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA): equivalent to EMA with alpha = 1/period."""
    return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int) -> pd.Series:
    """Wilder ADX (1978). Strict past-only via the .shift(1) on close
    used inside true-range and directional movement."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high.diff()
    dn_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > dn_move) & (up_move > 0),
                                 up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0),
                                  dn_move, 0.0), index=high.index)

    atr = _wilder_smooth(tr, period)
    plus_di = 100.0 * _wilder_smooth(plus_dm, period) / atr
    minus_di = 100.0 * _wilder_smooth(minus_dm, period) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = _wilder_smooth(dx, period)
    return adx


def _choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series,
                      period: int) -> pd.Series:
    """Dreiss Choppiness Index over `period` bars."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    sum_tr = tr.rolling(window=period, min_periods=period).sum()
    high_n = high.rolling(window=period, min_periods=period).max()
    low_n = low.rolling(window=period, min_periods=period).min()
    rng = (high_n - low_n).replace(0.0, np.nan)
    ci = 100.0 * np.log10(sum_tr / rng) / np.log10(period)
    return ci


def _realized_vol(close: pd.Series, period: int) -> pd.Series:
    """sqrt of EWMA(period) of squared 1-bar log-returns."""
    log_ret = np.log(close / close.shift(1))
    var_ewm = (log_ret ** 2).ewm(
        span=period, adjust=False, min_periods=period,
    ).mean()
    return np.sqrt(var_ewm)


def compute_regime_features(
    bars: pd.DataFrame,
    *,
    thresholds: RegimeThresholds | None = None,
) -> pd.DataFrame:
    """Compute the feature columns required by ``classify_regime``.

    Parameters
    ----------
    bars:
        DataFrame indexed by bar timestamp with at least columns
        ``high``, ``low``, ``close`` (case-insensitive). Must be
        chronologically sorted.
    thresholds:
        Optional ``RegimeThresholds``. Defaults capture P3 spec.

    Returns
    -------
    DataFrame with columns:
        ``adx``, ``ema_diff``, ``ci``, ``rv``,
        ``rv_trend_median``, ``rv_range_median``.

    No-look-ahead guarantee: every column at bar t depends only on bars
    indexed strictly before t (or at t for purely-instantaneous quantities
    that don't require future). The P3 runner aligns events to bars via
    ``.loc[event_time]``; the value at that timestamp is the value the
    classifier would have produced if asked at that bar with no future
    information.
    """
    th = thresholds or RegimeThresholds()
    cols = {c.lower(): c for c in bars.columns}
    high = bars[cols["high"]].astype(float)
    low = bars[cols["low"]].astype(float)
    close = bars[cols["close"]].astype(float)

    adx = _adx(high, low, close, th.adx_period)
    ema_fast = close.ewm(span=th.ema_fast, adjust=False,
                         min_periods=th.ema_fast).mean()
    ema_slow = close.ewm(span=th.ema_slow, adjust=False,
                         min_periods=th.ema_slow).mean()
    ema_diff = ema_fast - ema_slow
    ci = _choppiness_index(high, low, close, th.ci_period)
    rv = _realized_vol(close, th.rv_period)

    # Rolling median realized vol restricted to {trending, ranging} bars
    # over the lookback window. Computed by masking rv per bar with the
    # ADX trending classification, then taking the rolling median over
    # the masked series. Past-only by construction (rolling on a series
    # whose values themselves are past-only).
    is_trending = adx > th.adx_trending
    rv_trend_only = rv.where(is_trending)
    rv_range_only = rv.where(~is_trending)
    rv_trend_median = rv_trend_only.rolling(
        window=th.rv_median_lookback, min_periods=20,
    ).median()
    rv_range_median = rv_range_only.rolling(
        window=th.rv_median_lookback, min_periods=20,
    ).median()

    out = pd.DataFrame({
        "adx": adx,
        "ema_diff": ema_diff,
        "ci": ci,
        "rv": rv,
        "rv_trend_median": rv_trend_median,
        "rv_range_median": rv_range_median,
    }, index=bars.index)
    return out


def classify_regime(
    row: pd.Series,
    *,
    thresholds: RegimeThresholds | None = None,
) -> str:
    """Classify a single bar's regime.

    Required keys in ``row``: ``adx``, ``ema_diff``, ``ci``, ``rv``,
    ``rv_trend_median``, ``rv_range_median``. NaN in any of them yields
    ``chop`` (insufficient history → conservative default).
    """
    th = thresholds or RegimeThresholds()
    adx = row.get("adx", np.nan)
    ema_diff = row.get("ema_diff", np.nan)
    ci = row.get("ci", np.nan)
    rv = row.get("rv", np.nan)
    rv_trend_med = row.get("rv_trend_median", np.nan)
    rv_range_med = row.get("rv_range_median", np.nan)

    # Insufficient history → chop.
    if any(pd.isna(v) for v in (adx, ema_diff, ci, rv)):
        return "chop"

    # Chop priority — Dreiss CI > 61.8 trumps trend strength.
    if ci > th.chop_index:
        return "chop"

    is_trending = adx > th.adx_trending
    if is_trending:
        # Trend direction.
        is_up = ema_diff > 0
        # Cohort split: median rv within trending bars over lookback.
        # If median is NaN (cold start), fall back to low_vol bucket.
        if pd.isna(rv_trend_med) or rv <= rv_trend_med:
            return "trend_up_low_vol" if is_up else "trend_dn_low_vol"
        return "trend_up_high_vol" if is_up else "trend_dn_high_vol"

    # Ranging.
    if pd.isna(rv_range_med) or rv <= rv_range_med:
        return "range_tight"
    return "range_wide"


# ---------------------------------------------------------------------------
# RegimeStabilityGate
# ---------------------------------------------------------------------------

class RegimeStabilityGate:
    """Debouncer over a stream of regime classifications.

    Maintains a "committed regime" — the regime config the bot is
    currently operating under. Switches only when a candidate regime
    has been the raw classifier's output for ``n`` consecutive events.

    Returns the committed regime as the event's label. This matches the
    "what would we have actually traded under?" attribution: if the raw
    classifier flickers into chop for 3 events but never reaches the
    n-event threshold, the trades during that window were taken under
    the previous regime's config and must be attributed to it.

    Pure (no I/O). Thread-unsafe — call from a single thread per gate
    instance, or wrap in a lock.

    Practitioner discipline rationale: the gate exists to prevent the
    bot from acting on transient classifier flips (whipsaw on the
    regime signal itself). Without it, a noisy ADX or CI series can
    flap between two states each bar, producing a parade of regime
    transitions that have no economic meaning. n=5 is a discretionary
    buffer; tune via pro_trading_config.yaml::regime_detector::stability_n.
    """

    def __init__(self, n: int = 5) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n
        self.committed_regime: str | None = None
        self.pending_regime: str | None = None
        self.pending_streak: int = 0
        self.transition_count: int = 0

    def step(self, raw_regime: str) -> str:
        """Process one event's raw regime; return the committed label."""
        if self.committed_regime is None:
            # Cold start: commit the very first regime we see.
            self.committed_regime = raw_regime
            self.pending_regime = None
            self.pending_streak = 0
            self.transition_count += 1
            return self.committed_regime

        if raw_regime == self.committed_regime:
            # Stable — reset any pending streak.
            self.pending_regime = None
            self.pending_streak = 0
            return self.committed_regime

        # raw differs from committed; track / advance the pending streak.
        if raw_regime == self.pending_regime:
            self.pending_streak += 1
        else:
            self.pending_regime = raw_regime
            self.pending_streak = 1

        if self.pending_streak >= self.n:
            # Promote.
            self.committed_regime = raw_regime
            self.pending_regime = None
            self.pending_streak = 0
            self.transition_count += 1
            return self.committed_regime

        # Buffer not full — keep using last committed.
        return self.committed_regime
