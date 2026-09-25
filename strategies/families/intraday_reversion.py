"""Intraday reversion families (mask-based): VWAP reversion, gap go/fade, oversold bounce."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.families.common import Family, bars_to_1500, candidate_positions, col, make_entry, one_per_session
from strategies.families.models import Entry


def vwap_reversion_entries(bars5, ind, daily, news, params) -> list[Entry]:
    k, tsb, atr_stop = float(params["k"]), int(params["time_stop_bars"]), float(params["atr_stop"])
    o, c = bars5["open"].to_numpy(dtype=float), bars5["close"].to_numpy(dtype=float)
    bod, vwap, sd, rsi, atr = (col(ind, x) for x in ("bar_of_day", "vwap", "vwap_sd", "rsi", "atr"))
    with np.errstate(invalid="ignore"):
        base = (bod >= 6) & (bod <= 60) & (sd > 0) & ~np.isnan(vwap) & ~np.isnan(rsi) & ~np.isnan(atr)
        long_m = base & (c <= vwap - k * sd) & (rsi < 35) & (c > o)
        short_m = base & (c >= vwap + k * sd) & (rsi > 65) & (c < o)
    out: list[Entry] = []
    for i in candidate_positions(long_m):
        px = float(bars5["open"].iloc[i + 1]); stop = px - atr_stop * atr[i]
        if vwap[i] - px >= px - stop:
            e = make_entry(bars5, int(i), "long", stop, float(vwap[i]), tsb, "intraday", "vwap_long")
            if e: out.append(e)
    for i in candidate_positions(short_m):
        px = float(bars5["open"].iloc[i + 1]); stop = px + atr_stop * atr[i]
        if px - vwap[i] >= stop - px:
            e = make_entry(bars5, int(i), "short", stop, float(vwap[i]), tsb, "intraday", "vwap_short")
            if e: out.append(e)
    return one_per_session(out)


def gap_play_entries(bars5, ind, daily, news, params) -> list[Entry]:
    gap_min, atr_stop = float(params["gap_pct"]), float(params["atr_stop"])
    confirm_bar = int(params["confirm_minutes"]) // 5 - 1
    c = bars5["close"].to_numpy(dtype=float)
    bod, gap, s_open, atr = (col(ind, x) for x in ("bar_of_day", "gap_pct", "session_open", "atr"))
    with np.errstate(invalid="ignore"):
        m = (bod == confirm_bar) & (np.abs(gap) >= gap_min) & ~np.isnan(s_open) & ~np.isnan(atr)
    out: list[Entry] = []
    for i in candidate_positions(m):
        holding = c[i] > s_open[i] if gap[i] > 0 else c[i] < s_open[i]
        with_gap = "long" if gap[i] > 0 else "short"
        side = with_gap if holding else ("short" if with_gap == "long" else "long")
        px = float(bars5["open"].iloc[i + 1]); dist = atr_stop * atr[i]
        stop = px - dist if side == "long" else px + dist
        target = px + 2 * dist if side == "long" else px - 2 * dist
        e = make_entry(bars5, int(i), side, stop, target, bars_to_1500(confirm_bar + 1), "intraday", "gap_go" if holding else "gap_fade")
        if e: out.append(e)
    return one_per_session(out)


def mean_rev_intraday_entries(bars5, ind, daily, news, params) -> list[Entry]:
    rsi_max, atr_stop, target_r = float(params["rsi_max"]), float(params["atr_stop"]), float(params["target_r"])
    o, c = bars5["open"].to_numpy(dtype=float), bars5["close"].to_numpy(dtype=float)
    bod, bb_lo, rsi, atr, vr = (col(ind, x) for x in ("bar_of_day", "bb_lower", "rsi", "atr", "volume_ratio"))
    with np.errstate(invalid="ignore"):
        m = (bod >= 3) & (bod <= 66) & (c < bb_lo) & (rsi < rsi_max) & (c > o) & (vr >= 1.2) & ~np.isnan(atr)
    out: list[Entry] = []
    for i in candidate_positions(m):
        px = float(bars5["open"].iloc[i + 1]); dist = atr_stop * atr[i]
        e = make_entry(bars5, int(i), "long", px - dist, px + target_r * dist, 2, "multiday", "oversold_bounce")
        if e: out.append(e)
    return one_per_session(out)


FAMILIES_INTRADAY_REVERSION: tuple[Family, ...] = (
    Family("vwap_reversion", vwap_reversion_entries, {"k": [1.5, 2.0, 2.5], "time_stop_bars": [6, 12, 24], "atr_stop": [1.0, 1.5]}, "intraday", ("long", "short")),
    Family("gap_play", gap_play_entries, {"gap_pct": [2.0, 3.0, 5.0], "confirm_minutes": [15, 30], "atr_stop": [1.0, 1.5]}, "intraday", ("long", "short")),
    Family("mean_rev_intraday", mean_rev_intraday_entries, {"rsi_max": [25, 30, 35], "atr_stop": [2.0, 2.5, 3.0], "target_r": [1.5, 2.0]}, "multiday", ("long",)),
)
