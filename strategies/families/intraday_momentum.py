# strategies/families/intraday_momentum.py
"""Intraday momentum families (mask-based): day-trade momentum long/short and opening-range breakout."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.families.common import Family, bars_to_1500, candidate_positions, col, make_entry, one_per_session
from strategies.families.models import Entry

ENTRY_FIRST_BAR, ENTRY_LAST_BAR = 6, 60
ORB_LAST_BAR = 12


def dt_momentum_entries(bars5: pd.DataFrame, ind: pd.DataFrame, daily: pd.DataFrame, news: pd.DataFrame, params: dict) -> list[Entry]:
    rs_floor, atr_stop, target_r = float(params["rs_floor"]), float(params["atr_stop"]), float(params["target_r"])
    c = bars5["close"].to_numpy(dtype=float)
    bod, rs, vr, rsi = col(ind, "bar_of_day"), col(ind, "rs_vs_spy_pct"), col(ind, "volume_ratio"), col(ind, "rsi")
    hi20, lo20, adx, sma20 = col(ind, "high_20"), col(ind, "low_20"), col(ind, "adx"), col(ind, "sma_20")
    macd, sig, atr = col(ind, "macd"), col(ind, "macd_signal"), col(ind, "atr")
    with np.errstate(invalid="ignore", divide="ignore"):
        base = (bod >= ENTRY_FIRST_BAR) & (bod <= ENTRY_LAST_BAR) & (vr >= 1.5) & ~np.isnan(rs) & ~np.isnan(atr) & ~np.isnan(hi20) & ~np.isnan(sma20)
        long_ok = base & (rs >= rs_floor) & (rsi < 70)
        breakout = long_ok & (c > hi20) & (adx > 20) & (c <= hi20 * 1.03)
        pullback = long_ok & (rsi >= 40) & (rsi <= 60) & (np.abs(c - sma20) / sma20 <= 0.02) & (macd > sig)
        cont = long_ok & (rsi >= 55) & (rsi <= 75) & (adx > 25) & (macd > sig) & (c > sma20)
        short_ok = base & (rs <= -rs_floor) & (rsi >= 30) & (rsi <= 85)
        breakdown = short_ok & (c < lo20) & (adx > 20) & (c >= lo20 * 0.97)
        cont_down = short_ok & (rsi >= 30) & (rsi <= 50) & (c < sma20) & (macd < sig)
        rejection = short_ok & (c >= hi20 * 0.98) & (c < hi20) & (macd < sig) & (adx > 20)
    tags = [("breakout", breakout, "long"), ("pullback", pullback, "long"), ("continuation", cont, "long"),
            ("breakdown", breakdown, "short"), ("continuation_down", cont_down, "short"), ("rejection", rejection, "short")]
    out: list[Entry] = []
    seen: set[int] = set()
    for tag, mask, side in tags:
        for i in candidate_positions(mask):
            if i in seen:
                continue
            seen.add(int(i))
            px = float(bars5["open"].iloc[i + 1])
            dist = atr_stop * atr[i]
            stop = px - dist if side == "long" else px + dist
            target = px + target_r * dist if side == "long" else px - target_r * dist
            e = make_entry(bars5, int(i), side, stop, target, bars_to_1500(bod[i] + 1), "intraday", tag)
            if e:
                out.append(e)
    return one_per_session(out)


def orb_entries(bars5: pd.DataFrame, ind: pd.DataFrame, daily: pd.DataFrame, news: pd.DataFrame, params: dict) -> list[Entry]:
    mins = int(params["range_minutes"])
    k, rvol_min, atr_stop = mins // 5, float(params["rvol_min"]), float(params["atr_stop"])
    c = bars5["close"].to_numpy(dtype=float)
    bod, rvol, atr = col(ind, "bar_of_day"), col(ind, "rvol"), col(ind, "atr")
    or_hi, or_lo, avg = col(ind, f"or{mins}_high"), col(ind, f"or{mins}_low"), col(ind, f"or{mins}_range_avg20")
    with np.errstate(invalid="ignore", divide="ignore"):
        base = (bod >= k) & (bod <= ORB_LAST_BAR) & (rvol >= rvol_min) & (avg > 0) & ~np.isnan(atr) & ~np.isnan(or_hi) & ~np.isnan(or_lo)
        contracted = base & (1 - (or_hi - or_lo) / avg >= 0.2)
        long_m, short_m = contracted & (c > or_hi), contracted & (c < or_lo)
    out: list[Entry] = []
    for i in candidate_positions(long_m):
        px = float(bars5["open"].iloc[i + 1])
        stop = max(px - atr_stop * atr[i], or_lo[i] - 0.5 * atr[i])
        e = make_entry(bars5, int(i), "long", stop, px + 2 * (px - stop), bars_to_1500(bod[i] + 1), "intraday", "orb_long")
        if e:
            out.append(e)
    for i in candidate_positions(short_m):
        px = float(bars5["open"].iloc[i + 1])
        stop = min(px + atr_stop * atr[i], or_hi[i] + 0.5 * atr[i])
        e = make_entry(bars5, int(i), "short", stop, px - 2 * (stop - px), bars_to_1500(bod[i] + 1), "intraday", "orb_short")
        if e:
            out.append(e)
    return one_per_session(out)


FAMILIES_INTRADAY_MOMENTUM: tuple[Family, ...] = (
    Family("dt_momentum", dt_momentum_entries, {"rs_floor": [0.25, 0.5, 1.0], "atr_stop": [1.0, 1.5, 2.0], "target_r": [1.5, 2.0, 3.0]},
           "intraday", ("long", "short")),
    Family("orb", orb_entries, {"range_minutes": [15, 30], "rvol_min": [1.2, 1.5, 2.0], "atr_stop": [0.75, 1.0]},
           "intraday", ("long", "short")),
)
