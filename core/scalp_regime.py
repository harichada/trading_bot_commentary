"""
Intraday market regime detector for the scalp reversal strategy.

Uses SPY's cumulative VWAP to determine whether the session favours longs,
shorts, or is neutral enough to allow both.

Decision rule (threshold = ±0.15%, ≈$1 on a $590 SPY):
  current > VWAP + threshold  →  bullish   → "long_only"
  current < VWAP - threshold  →  bearish   → "short_only"
  |current - VWAP| ≤ threshold → neutral  → "both"

The dead-band prevents direction flipping when price oscillates at VWAP.
Input df must contain today's RTH 1-min bars only (OHLCV, UTC-indexed).
"""
from __future__ import annotations

import pandas as pd

_VWAP_THRESHOLD_PCT = 0.15   # ±0.15% → neutral zone
_MIN_BARS = 5                # need at least this many bars for a stable VWAP


def compute_vwap(df: pd.DataFrame) -> float:
    """Cumulative dollar-volume VWAP from a 1-min OHLCV DataFrame."""
    total_vol = float(df["Volume"].sum())
    if total_vol == 0.0:
        return float(df["Close"].iloc[-1])
    return float((df["Close"] * df["Volume"]).sum() / total_vol)


def spy_regime(spy_df: pd.DataFrame) -> str:
    """
    Return "long_only", "short_only", or "both" based on SPY VWAP position.

    Falls back to "both" (neutral) when fewer than _MIN_BARS are available so
    the strategy does not go dark on startup or after a data gap.
    """
    if spy_df is None or spy_df.empty or len(spy_df) < _MIN_BARS:
        return "both"

    vwap    = compute_vwap(spy_df)
    current = float(spy_df["Close"].iloc[-1])
    pct     = (current - vwap) / vwap * 100.0

    if pct > _VWAP_THRESHOLD_PCT:
        return "long_only"
    elif pct < -_VWAP_THRESHOLD_PCT:
        return "short_only"
    else:
        return "both"
