"""Vectorized historical replay of primary strategies.

Reproduces each rule-based strategy's entry conditions over an OHLCV
DataFrame so that meta-labeling can train on the timestamps the primary
would have fired. Rules mirror ``strategies/builtin.py`` exactly; logic
is inlined here for speed (strategy classes invoke per-bar).

If the original strategy rules change, update these replay functions
side-by-side to keep primary ↔ meta in sync.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI (matches ``ta.momentum.RSIIndicator`` at steady state)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def replay_mean_reversion(
    df: pd.DataFrame,
    rsi_threshold: float = 30.0,
) -> pd.DataFrame:
    """Replay the mean-reversion strategy's entry rules over ``df``.

    Long entry when all of:
      - RSI(14) < rsi_threshold
      - Close < lower Bollinger band (SMA20 ± 2×σ20)
      - Falling-knife filter passes: close ≥ SMA50 OR macd ≥ macd_signal

    Short entry when both of:
      - RSI(14) > 70
      - Close > upper Bollinger band

    Parameters
    ----------
    df:
        OHLCV DataFrame indexed by timestamp with ``Close`` column.
    rsi_threshold:
        Oversold cutoff for long entries (default 30, matches live config).

    Returns
    -------
    DataFrame indexed by signal timestamp with a single ``side`` column
    of +1 (long) or -1 (short). Empty if fewer than 50 bars of history
    are available (SMA50 warmup).
    """
    if len(df) < 50:
        return pd.DataFrame(columns=["side"], index=pd.DatetimeIndex([]))

    close = df["Close"]

    rsi = _rsi(close, window=14)
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20

    sma_50 = close.rolling(50).mean()
    macd_line, macd_signal = _macd(close)

    long_raw = (rsi < rsi_threshold) & (close < bb_lower)
    # Falling-knife filter: skip when price below SMA50 AND macd below signal.
    falling_knife = (close < sma_50) & (macd_line < macd_signal)
    long_mask = long_raw & ~falling_knife

    short_mask = (rsi > 70) & (close > bb_upper)

    long_idx = df.index[long_mask.fillna(False)]
    short_idx = df.index[short_mask.fillna(False)]

    # DatetimeIndex.append was removed in pandas 2.1. Concatenate via
    # numpy arrays, then wrap as a fresh DatetimeIndex.
    combined = pd.DatetimeIndex(
        np.concatenate([long_idx.to_numpy(), short_idx.to_numpy()])
    )
    events = pd.DataFrame(
        {"side": [1] * len(long_idx) + [-1] * len(short_idx)},
        index=combined,
    ).sort_index()

    return events
