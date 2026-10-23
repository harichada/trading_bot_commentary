"""Setup C — gap/PEAD (marginal; SHADOW ONLY). SPEC §3.

gap >= params.gap and rvol >= 2.5 and clv >= 0.7 and close > open, and (with a
calendar) an earnings report within the prior 1 trading day. Without a calendar
the lane already blocks entry (earnings_data_missing).
stop = clamp(low5 - 0.1*ATR, entry-3*ATR, entry-0.75*ATR); target = entry + 3R.
"""
from __future__ import annotations

import math

import pandas as pd

from core.swing_lane.config import MIN_DVOL, MIN_PRICE
from core.swing_lane.setups import Signal


def _prev_weekday(d):
    d = d - pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def scan(bars, ind, params, sym: str = "", earnings: set = None) -> list:
    if bars is None or len(bars) < 2:
        return []
    r = ind.iloc[-1]; c = float(bars.close.iat[-1]); o = float(bars.open.iat[-1])
    vals = [r.gap, r.rvol, r.clv, r.atr, r.low5, r.dvol50]
    if any(pd.isna(x) for x in vals):
        return []
    if not (r.gap >= float(params["gap"]) and r.rvol >= 2.5 and r.clv >= 0.7 and c > o
            and c >= MIN_PRICE and r.dvol50 >= MIN_DVOL):
        return []
    d = pd.Timestamp(bars.index[-1]).date()
    if not earnings or not ({d, _prev_weekday(pd.Timestamp(d)).date()} & set(earnings)):
        return []  # real PEAD version requires an earnings report in the prior 1 trading day
    a = float(r.atr)
    stop = min(max(float(r.low5) - 0.1 * a, c - 3 * a), c - 0.75 * a)
    return [Signal(setup="C", sym=sym, signal_date=bars.index[-1], close=c, atr14=a,
                   rvol=float(r.rvol), gap=float(r.gap),
                   rng20_atr=float(r.rng20) / float(ind.atr.iat[-2]) if not math.isnan(float(ind.atr.iat[-2])) else float("nan"),
                   planned_stop=stop, planned_target=c + 3 * (c - stop), score=float(r.gap))]
