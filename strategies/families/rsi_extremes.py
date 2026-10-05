"""RSI-extreme families (2026-10-05), from `research/rsi_extremes_study.py`.

The event study over 15 large names (2021 → 2026-10) found one candidate and one
anti-pattern:

* `rsi_dip` (long): a daily RSI(14) print below 25 was followed by a bounce
  61-66% of the time (+1.8% at 5 sessions, +4.1% at 10, n=61) — almost all of
  it in 2024-2026 and below the 200-day SMA, i.e. plausibly bull-market dip
  buying rather than a timing edge. Thresholds 30/35 showed nothing. This
  family puts the rule through the walk-forward pipeline with costs, the
  shuffled baseline and the selection baseline, which is the only verdict that
  counts here.
* `rsi_fade` (short): shorting a high RSI on these names LOST money at every
  threshold and horizon (they trend). Included so the pipeline records the
  failure formally rather than leaving it to a study table.

Both evaluate at each session open from lagged daily features (same contract
as `mr_20d`): the signal is yesterday's close, the fill is today's open.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.families.common import Family, make_entry, session_opens, spaced_by_sessions
from strategies.families.models import Entry

TREND_FILTERS = ("none", "sma200")


def _vals(ind: pd.DataFrame, j: int, cols: tuple[str, ...]) -> dict[str, float]:
    return {c: float(ind[c].iloc[j]) for c in cols}


def _trend_ok(filter_name: str, v: dict[str, float], side: str) -> bool:
    if filter_name == "none":
        return True
    if side == "long":
        return v["d_close"] > v["d_sma200"]
    return v["d_close"] < v["d_sma200"]


def rsi_dip_entries(bars5, ind, daily, news, params) -> list[Entry]:
    """Long when yesterday's RSI(14) closed at or below `rsi_low`; stop
    `stop_atr` x ATR14 below the open, target the 20-day mean, at least 1R away,
    `time_stop_days` sessions. `trend_filter` = sma200 requires the close above
    the 200-day SMA (the "dip in an uptrend" variant the study could not
    separate from the raw rule)."""
    rsi_low = float(params["rsi_low"])
    stop_atr = float(params["stop_atr"])
    hold = int(params["time_stop_days"])
    trend = str(params.get("trend_filter", "none"))
    out: list[Entry] = []
    for j in session_opens(ind):
        v = _vals(ind, j, ("d_close", "d_rsi14", "d_bb_mid", "d_atr14", "d_sma200"))
        if any(np.isnan(x) for x in v.values()):
            continue
        if v["d_rsi14"] > rsi_low or not _trend_ok(trend, v, "long"):
            continue
        px = float(bars5["open"].iloc[j])
        dist = stop_atr * v["d_atr14"]
        if dist <= 0 or v["d_bb_mid"] - px < dist:
            continue
        e = make_entry(bars5, j - 1, "long", px - dist, v["d_bb_mid"], hold, "multiday", "rsi_dip")
        if e:
            out.append(e)
    return spaced_by_sessions(out, hold)


def rsi_fade_entries(bars5, ind, daily, news, params) -> list[Entry]:
    """Short when yesterday's RSI(14) closed at or above `rsi_high`; mirror of
    `rsi_dip` (stop above, target the 20-day mean)."""
    rsi_high = float(params["rsi_high"])
    stop_atr = float(params["stop_atr"])
    hold = int(params["time_stop_days"])
    out: list[Entry] = []
    for j in session_opens(ind):
        v = _vals(ind, j, ("d_close", "d_rsi14", "d_bb_mid", "d_atr14"))
        if any(np.isnan(x) for x in v.values()):
            continue
        if v["d_rsi14"] < rsi_high:
            continue
        px = float(bars5["open"].iloc[j])
        dist = stop_atr * v["d_atr14"]
        if dist <= 0 or px - v["d_bb_mid"] < dist:
            continue
        e = make_entry(bars5, j - 1, "short", px + dist, v["d_bb_mid"], hold, "multiday", "rsi_fade")
        if e:
            out.append(e)
    return spaced_by_sessions(out, hold)


FAMILIES_RSI: tuple[Family, ...] = (
    # 3 x 3 x 3 = 27 combos. The grid is laid out so its MEDIAN (what the live
    # fit lane trades when a symbol has no fit row) is the set the pooled
    # fixed-rule evaluation ranked best: rsi_low 25, stop 1.5 ATR, 10 sessions,
    # no trend filter (the SMA200 variant produced 3 trades in five years).
    Family("rsi_dip", rsi_dip_entries,
           {"rsi_low": [20, 25, 30], "stop_atr": [1.0, 1.5, 2.5], "time_stop_days": [5, 10, 15],
            "trend_filter": ["none"]},
           "multiday", ("long",)),
    # 3 x 2 x 2 = 12 combos.
    Family("rsi_fade", rsi_fade_entries,
           {"rsi_high": [70, 75, 80], "stop_atr": [1.5, 2.5], "time_stop_days": [5, 10]},
           "multiday", ("short",)),
)
