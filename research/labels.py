"""Triple-barrier labeling for the side-classifier backtest.

v-triple-barrier-labels-2026-05-13. For each (symbol, day) we ask:
within the next H bars, did the path hit the upper barrier
(`+k × ATR` above entry) first, the lower barrier
(`-k × ATR` below entry) first, or neither (time barrier)?

Pure functions over pandas. No I/O. Tests pin the boundary behavior
since label leakage at the boundary is the most common silent bug.

Label encoding:
  LONG_WINS  → upper barrier hit first
  SHORT_WINS → lower barrier hit first
  NEITHER    → time barrier hit first (range / chop)
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd


class BarrierLabel(str, Enum):
    LONG_WINS = "long_wins"
    SHORT_WINS = "short_wins"
    NEITHER = "neither"


def label_triple_barrier(
    daily_bars: pd.DataFrame,
    entry_idx: int,
    *,
    atr_value: float,
    k_upper: float = 1.5,
    k_lower: float = 1.5,
    horizon_bars: int = 5,
) -> BarrierLabel:
    """Apply the triple-barrier rule to a single example.

    Parameters
    ----------
    daily_bars
        DataFrame indexed by date, columns at least {open, high, low,
        close}. The entry bar is ``daily_bars.iloc[entry_idx]``.
    entry_idx
        Integer index of the entry bar. The label is computed using
        bars at indices ``entry_idx + 1 .. entry_idx + horizon_bars``.
    atr_value
        ATR-14 (or other ATR) value AT entry_idx. The barriers are
        ``entry_close ± k * atr_value`` — same units as price.
    k_upper, k_lower
        Multipliers on ATR. Defaults to 1.5 per PLAN §3.2.
    horizon_bars
        Time barrier in bars. Default 5 trading days.

    Returns
    -------
    BarrierLabel
        Which barrier was hit first.

    Boundary semantics
    ------------------
    - If both barriers are hit on the same bar, LONG_WINS is preferred
      iff that bar closed at or above its open (intrabar direction
      heuristic); else SHORT_WINS. Deterministic, no NaN labels.
    - If the entry bar's close is the last bar in the frame, returns
      NEITHER (no future to evaluate). Caller should filter these out.
    """
    if entry_idx < 0 or entry_idx >= len(daily_bars):
        raise ValueError(
            f"entry_idx={entry_idx} out of range [0, {len(daily_bars)})"
        )
    if atr_value is None or atr_value <= 0:
        raise ValueError(f"atr_value must be positive, got {atr_value}")

    entry_close = float(daily_bars["close"].iloc[entry_idx])
    upper = entry_close + k_upper * atr_value
    lower = entry_close - k_lower * atr_value

    # Walk forward up to horizon_bars
    end_idx = min(entry_idx + horizon_bars, len(daily_bars) - 1)
    if end_idx <= entry_idx:
        return BarrierLabel.NEITHER

    forward = daily_bars.iloc[entry_idx + 1: end_idx + 1]

    for i in range(len(forward)):
        bar = forward.iloc[i]
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        hit_upper = bar_high >= upper
        hit_lower = bar_low <= lower

        if hit_upper and hit_lower:
            # Both touched intrabar — break the tie by close-vs-open.
            if bar["close"] >= bar["open"]:
                return BarrierLabel.LONG_WINS
            return BarrierLabel.SHORT_WINS
        if hit_upper:
            return BarrierLabel.LONG_WINS
        if hit_lower:
            return BarrierLabel.SHORT_WINS

    # Time barrier hit.
    return BarrierLabel.NEITHER


def label_dataset(
    daily_bars: pd.DataFrame,
    atr_series: pd.Series,
    *,
    k: float = 1.5,
    horizon_bars: int = 5,
) -> pd.Series:
    """Apply triple-barrier labeling to every bar in ``daily_bars``.

    Returns a Series of BarrierLabel values indexed identically to
    daily_bars. Bars with insufficient ATR or where the horizon
    would walk past the end of the frame are labeled NEITHER.
    """
    labels = []
    for i in range(len(daily_bars)):
        atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0.0
        if atr <= 0 or i + horizon_bars >= len(daily_bars):
            labels.append(BarrierLabel.NEITHER)
            continue
        labels.append(
            label_triple_barrier(
                daily_bars, i, atr_value=atr,
                k_upper=k, k_lower=k, horizon_bars=horizon_bars,
            )
        )
    return pd.Series(labels, index=daily_bars.index, name="label")
