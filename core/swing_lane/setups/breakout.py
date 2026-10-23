"""Setup B — base breakout (passed OOS: PF 1.74, +0.30R, n=335). SPEC §3.

close > hh20 and volume >= 1.5*vol50 and rng20 <= k*ATR14[t-1] and close > SMA50
stop = entry - 2*ATR14[t]; rank by rvol desc; exit = trail (no fixed target).
Params n, k come from best_params.json["B"].
"""
from __future__ import annotations

import math

from core.swing_lane.config import MIN_DVOL, MIN_PRICE
from core.swing_lane.setups import Signal

RVOL_MIN = 1.5
STOP_ATR = 2.0


def _f(x) -> float:
    try:
        x = float(x)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def is_signal(bars, ind, params, i: int = -1) -> bool:
    n, k = int(params["n"]), float(params["k"])
    if n != 20:
        raise ValueError("breakout: only n=20 is supported by the indicator set")
    c = _f(bars.close.iat[i]); v = _f(bars.volume.iat[i])
    hh = _f(ind["hh20"].iat[i]); vol50 = _f(ind["vol50"].iat[i])
    rng = _f(ind["rng20"].iat[i]); atr_prev = _f(ind["atr"].iat[i - 1]) if len(ind) > 1 else float("nan")
    sma50 = _f(ind["sma50"].iat[i]); dvol = _f(ind["dvol50"].iat[i])
    vals = (c, v, hh, vol50, rng, atr_prev, sma50, dvol)
    if any(math.isnan(x) for x in vals):
        return False
    return (c > hh and v >= RVOL_MIN * vol50 and rng <= k * atr_prev and c > sma50
            and c >= MIN_PRICE and dvol >= MIN_DVOL)


def scan(bars, ind, params, sym: str = "") -> list:
    if bars is None or len(bars) < 2 or not is_signal(bars, ind, params):
        return []
    c = float(bars.close.iat[-1]); atr = float(ind["atr"].iat[-1])
    atr_prev = float(ind["atr"].iat[-2])
    return [Signal(
        setup="B", sym=sym, signal_date=bars.index[-1], close=c, atr14=atr,
        rvol=float(ind["rvol"].iat[-1]), gap=_f(ind["gap"].iat[-1]),
        rng20_atr=float(ind["rng20"].iat[-1]) / atr_prev if atr_prev else float("nan"),
        planned_stop=c - STOP_ATR * atr, planned_target=None,
        score=float(ind["rvol"].iat[-1]),
    )]
