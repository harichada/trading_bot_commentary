"""Bar-construction primitives.

López de Prado, AFML §2.3, argues that **time bars** (5-min OHLCV) are
statistically inferior for ML: returns are autocorrelated, intraday
seasonality is heteroskedastic, and information arrives in bursts rather
than at constant time intervals. Information-driven bars (dollar /
volume / imbalance) sample at constant *information*, restoring closer-
to-IID return statistics.

This module exposes the API surface so downstream callers can switch
``bar_type`` once tick (or 1-min) data is available. Today the data
provider emits 5-min OHLCV; we cannot construct true dollar bars from
that without lying about cumulative dollar volume. Rather than silently
faking, the dollar/volume/imbalance constructors raise
``NotImplementedError`` with an explicit "needs tick data" message.

This is a P4 design choice: ship the surface, gate the implementation
on real data — better to fail loud than to ship a placebo.
"""
from __future__ import annotations

import pandas as pd


_TICK_DATA_REQUIRED = (
    "{kind} bars require tick (or sub-minute) data with reliable per-bar "
    "dollar volume; current data feed emits 5-minute OHLCV. Wire the tick "
    "ingestion path before enabling this bar type."
)


def time_bars(df: pd.DataFrame, frequency_min: int) -> pd.DataFrame:
    """Pass-through for already-time-bucketed OHLCV.

    If ``df``'s native frequency matches ``frequency_min``, the frame is
    returned as-is. We do not down-resample to a finer grid (that would
    invent data); up-resampling to a coarser grid is supported by pandas
    ``resample`` directly and is left to the caller.
    """
    del frequency_min  # informational only on the pass-through path
    return df


def dollar_bars(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Dollar bars (AFML §2.3.2.1) — emit a bar each time cumulative
    dollar volume crosses ``threshold``."""
    del df, threshold
    raise NotImplementedError(_TICK_DATA_REQUIRED.format(kind="Dollar"))


def volume_bars(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Volume bars (AFML §2.3.1.2)."""
    del df, threshold
    raise NotImplementedError(_TICK_DATA_REQUIRED.format(kind="Volume"))


def imbalance_bars(df: pd.DataFrame, theta_threshold: float) -> pd.DataFrame:
    """Tick-imbalance bars (AFML §2.3.2.2)."""
    del df, theta_threshold
    raise NotImplementedError(_TICK_DATA_REQUIRED.format(kind="Imbalance"))
