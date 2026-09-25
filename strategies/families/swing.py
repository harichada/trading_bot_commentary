"""Multi-day swing families evaluated at each session open from lagged daily features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.families.common import Family, make_entry, session_opens, spaced_by_sessions
from strategies.families.models import Entry

PULLBACK_HOLD = 10
VOLBO_HOLD = 15
MR_ATR_STOP = 1.5


def _vals(ind: pd.DataFrame, j: int, cols: tuple[str, ...]) -> dict[str, float]:
    return {c: float(ind[c].iloc[j]) for c in cols}


def pullback_trend_entries(bars5, ind, daily, news, params) -> list[Entry]:
    ema_col = "d_ema20" if params["pullback_to"] == "ema20" else "d_ema10"
    lo, hi = (float(x) for x in str(params["rsi_band"]).split("-"))
    atr_stop = params["atr_stop"]
    out: list[Entry] = []
    for j in session_opens(ind):
        v = _vals(ind, j, ("d_close", "d_low", "d_sma50", ema_col, "d_rsi14", "d_ret1", "d_atr14"))
        if any(np.isnan(x) for x in v.values()):
            continue
        if not (v["d_close"] > v["d_sma50"] and v["d_low"] <= v[ema_col] <= v["d_close"] and lo <= v["d_rsi14"] <= hi and v["d_ret1"] > 0):
            continue
        px = float(bars5["open"].iloc[j])
        dist = atr_stop * v["d_atr14"]
        e = make_entry(bars5, j - 1, "long", px - dist, px + 2 * dist, PULLBACK_HOLD, "multiday", "pullback")
        if e:
            out.append(e)
    return spaced_by_sessions(out, PULLBACK_HOLD)


def mr_20d_entries(bars5, ind, daily, news, params) -> list[Entry]:
    sig = "20" if float(params["band_sigma"]) == 2.0 else "25"
    ext, hold = float(params["rsi3_extreme"]), int(params["time_stop_days"])
    out: list[Entry] = []
    for j in session_opens(ind):
        v = _vals(ind, j, ("d_close", "d_rsi3", "d_bb_mid", f"d_bb{sig}_upper", f"d_bb{sig}_lower", "d_atr14"))
        if any(np.isnan(x) for x in v.values()):
            continue
        px = float(bars5["open"].iloc[j])
        dist = MR_ATR_STOP * v["d_atr14"]
        if v["d_close"] < v[f"d_bb{sig}_lower"] and v["d_rsi3"] <= ext and v["d_bb_mid"] - px >= dist:
            e = make_entry(bars5, j - 1, "long", px - dist, v["d_bb_mid"], hold, "multiday", "mr20_long")
        elif v["d_close"] > v[f"d_bb{sig}_upper"] and v["d_rsi3"] >= 100 - ext and px - v["d_bb_mid"] >= dist:
            e = make_entry(bars5, j - 1, "short", px + dist, v["d_bb_mid"], hold, "multiday", "mr20_short")
        else:
            e = None
        if e:
            out.append(e)
    return spaced_by_sessions(out, hold)


def vol_breakout_entries(bars5, ind, daily, news, params) -> list[Entry]:
    n, comp, atr_stop = int(params["lookback"]), float(params["compression"]), params["atr_stop"]
    out: list[Entry] = []
    for j in session_opens(ind):
        v = _vals(ind, j, ("d_close", "d_atr20", "d_atr100", "d_atr14", f"d_high{n}", f"d_low{n}"))
        if any(np.isnan(x) for x in v.values()) or v["d_atr100"] <= 0 or v["d_atr20"] / v["d_atr100"] > comp:
            continue
        px = float(bars5["open"].iloc[j])
        dist = atr_stop * v["d_atr14"]
        if v["d_close"] > v[f"d_high{n}"]:
            e = make_entry(bars5, j - 1, "long", px - dist, px + 2 * dist, VOLBO_HOLD, "multiday", "volbo_long")
        elif v["d_close"] < v[f"d_low{n}"]:
            e = make_entry(bars5, j - 1, "short", px + dist, px - 2 * dist, VOLBO_HOLD, "multiday", "volbo_short")
        else:
            e = None
        if e:
            out.append(e)
    return spaced_by_sessions(out, VOLBO_HOLD)


FAMILIES_SWING: tuple[Family, ...] = (
    Family("pullback_trend", pullback_trend_entries,
           {"pullback_to": ["ema20", "ema10"], "rsi_band": ["35-50", "40-55"], "atr_stop": [1.5, 2.0, 2.5]},
           "multiday", ("long",)),
    Family("mr_20d", mr_20d_entries,
           {"band_sigma": [2.0, 2.5], "rsi3_extreme": [10, 15, 20], "time_stop_days": [5, 7]},
           "multiday", ("long", "short")),
    Family("vol_breakout", vol_breakout_entries,
           {"lookback": [20, 55], "compression": [0.6, 0.75, 0.9], "atr_stop": [1.5, 2.0]},
           "multiday", ("long", "short")),
)
