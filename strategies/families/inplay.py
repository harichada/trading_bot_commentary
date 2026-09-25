# strategies/families/inplay.py
"""In-play families (mask/session-based): opening-range breakout on the day's in-play list,
post-earnings-announcement drift for small caps, and a relative-strength leader pullback-resume.

All three read the cross-sectional `ctx_*` columns from strategies/families/context.py, which are
NaN by default when the runner has not supplied context (so these families propose nothing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.families.common import Family, candidate_positions, col, make_entry, one_per_session, session_opens, spaced_by_sessions
from strategies.families.context import ctx, in_play
from strategies.families.models import Entry

ORB_INPLAY_ENTRY_BAR = 1        # bar_of_day == 1: the second 5-min bar, evaluated once the first has closed
ORB_INPLAY_LAST_BAR = 77        # 15:55 bar-of-day on a 5-min RTH grid (09:30 = bar 0, 78 bars/session)
ORB_INPLAY_MIN_PRICE = 5.0      # dollars; bar-0 close must clear this
ORB_INPLAY_MIN_ATR14_PCT = 0.5  # percent; ctx_atr14_pct floor
PEAD_ENTRY_BAR = 6              # bar_of_day == 6: the 10:00 bar has just closed
PEAD_STOP_ATR = 1.5
PEAD_TARGET_R = 3.0
RS_LEADER_HOLD = 20
RS_LEADER_TARGET_R = 3.0
RS_LEADER_MIN_PCT_FROM_52W_HIGH = -15.0   # percent; ctx_pct_from_52w_high floor


def _bar0_values(bars5: pd.DataFrame) -> dict[str, np.ndarray]:
    """Open/high/low/close of each session's first bar, broadcast to every bar of that session."""
    sessions = bars5.index.normalize()
    return {c: bars5[c].groupby(sessions).transform("first").to_numpy(dtype=float) for c in ("open", "high", "low", "close")}


def orb_inplay_entries(bars5: pd.DataFrame, ind: pd.DataFrame, daily: pd.DataFrame, news: pd.DataFrame, params: dict) -> list[Entry]:
    top_n, stop_atr, trail_r = int(params["top_n"]), float(params["stop_atr"]), float(params["trail_r"])
    bod = col(ind, "bar_of_day")
    rank_ok = in_play(ind, top_n)
    atr_pct = ctx(ind, "atr14_pct")
    b0 = _bar0_values(bars5)
    o0, h0, l0, c0 = b0["open"], b0["high"], b0["low"], b0["close"]
    high, low = bars5["high"].to_numpy(dtype=float), bars5["low"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        base = (bod == ORB_INPLAY_ENTRY_BAR) & rank_ok & (c0 >= ORB_INPLAY_MIN_PRICE) & (atr_pct >= ORB_INPLAY_MIN_ATR14_PCT)
        long_m = base & (c0 > o0) & (high > h0)
        short_m = base & (c0 < o0) & (low < l0)
    out: list[Entry] = []
    for i in candidate_positions(long_m):
        px = float(bars5["open"].iloc[i + 1])
        atr_dollars = atr_pct[i] / 100.0 * px
        stop = max(px - stop_atr * atr_dollars, l0[i])
        risk = px - stop
        if risk <= 0:
            continue
        time_stop = max(1, ORB_INPLAY_LAST_BAR - (int(bod[i]) + 1))
        target = px + trail_r * risk if trail_r > 0 else px + 20 * risk
        e = make_entry(bars5, int(i), "long", stop, target, time_stop, "intraday", "orb_inplay_long")
        if e:
            out.append(e)
    for i in candidate_positions(short_m):
        px = float(bars5["open"].iloc[i + 1])
        atr_dollars = atr_pct[i] / 100.0 * px
        stop = min(px + stop_atr * atr_dollars, h0[i])
        risk = stop - px
        if risk <= 0:
            continue
        time_stop = max(1, ORB_INPLAY_LAST_BAR - (int(bod[i]) + 1))
        target = px - trail_r * risk if trail_r > 0 else px - 20 * risk
        e = make_entry(bars5, int(i), "short", stop, target, time_stop, "intraday", "orb_inplay_short")
        if e:
            out.append(e)
    return one_per_session(out)


def pead_smallcap_entries(bars5: pd.DataFrame, ind: pd.DataFrame, daily: pd.DataFrame, news: pd.DataFrame, params: dict) -> list[Entry]:
    s, hold_days, min_gap = float(params["s"]), int(params["hold_days"]), float(params["min_gap"])
    bod = col(ind, "bar_of_day")
    earnings_prev = ctx(ind, "earnings_prev")
    adv_bucket = ctx(ind, "adv_bucket")            # mega=3, large=2, mid=1, small=0
    gap_pct = ctx(ind, "gap_pct")
    eps_surprise = ctx(ind, "eps_surprise_pct")
    atr_pct = ctx(ind, "atr14_pct")
    session_open = col(ind, "session_open")
    close = bars5["close"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        base = (bod == PEAD_ENTRY_BAR) & (earnings_prev == 1) & (adv_bucket <= 1) & (np.abs(gap_pct) >= min_gap)
        long_m = base & (eps_surprise >= s) & (close > session_open)
        short_m = base & (eps_surprise <= -s) & (close < session_open)
    out: list[Entry] = []
    for i in candidate_positions(long_m):
        px = float(bars5["open"].iloc[i + 1])
        atr_dollars = atr_pct[i] / 100.0 * px
        stop = px - PEAD_STOP_ATR * atr_dollars
        risk = px - stop
        if risk <= 0:
            continue
        e = make_entry(bars5, int(i), "long", stop, px + PEAD_TARGET_R * risk, hold_days, "multiday", "pead_long")
        if e:
            out.append(e)
    for i in candidate_positions(short_m):
        px = float(bars5["open"].iloc[i + 1])
        atr_dollars = atr_pct[i] / 100.0 * px
        stop = px + PEAD_STOP_ATR * atr_dollars
        risk = stop - px
        if risk <= 0:
            continue
        e = make_entry(bars5, int(i), "short", stop, px - PEAD_TARGET_R * risk, hold_days, "multiday", "pead_short")
        if e:
            out.append(e)
    return spaced_by_sessions(out, hold_days)


def rs_leader_entries(bars5: pd.DataFrame, ind: pd.DataFrame, daily: pd.DataFrame, news: pd.DataFrame, params: dict) -> list[Entry]:
    ema_col = f"d_ema{int(params['ema'])}"
    rs_floor, atr_stop = float(params["rs_floor"]), float(params["atr_stop"])
    rs_rank, pct_52w = ctx(ind, "rs_rank_60"), ctx(ind, "pct_from_52w_high")
    out: list[Entry] = []
    for j in session_opens(ind):
        # NaN comparisons are already False, so a missing/NaN rs_rank or pct_52w falls straight
        # through to `continue` without an explicit isnan check (consistent with the mask-based
        # filters in orb_inplay_entries/pead_smallcap_entries above).
        if not (rs_rank[j] >= rs_floor and pct_52w[j] >= RS_LEADER_MIN_PCT_FROM_52W_HIGH):
            continue
        cols = ("d_low", "d_close", ema_col, "d_ret1", "d_atr14")
        v = {c: (float(ind[c].iloc[j]) if c in ind.columns else np.nan) for c in cols}
        if any(np.isnan(x) for x in v.values()):
            continue
        if not (v["d_low"] <= v[ema_col] <= v["d_close"] and v["d_ret1"] > 0):
            continue
        px = float(bars5["open"].iloc[j])
        dist = atr_stop * v["d_atr14"]
        e = make_entry(bars5, j - 1, "long", px - dist, px + RS_LEADER_TARGET_R * dist, RS_LEADER_HOLD, "multiday", "rs_leader")
        if e:
            out.append(e)
    return spaced_by_sessions(out, RS_LEADER_HOLD)


FAMILIES_INPLAY: tuple[Family, ...] = (
    Family("orb_inplay", orb_inplay_entries, {"top_n": [10, 20, 30], "stop_atr": [0.05, 0.10, 0.20], "trail_r": [0, 5, 10]},
           "intraday", ("long", "short")),
    Family("pead_smallcap", pead_smallcap_entries, {"s": [5, 10, 20], "hold_days": [5, 10, 20], "min_gap": [0, 2, 4]},
           "multiday", ("long", "short")),
    Family("rs_leader", rs_leader_entries, {"ema": [10, 20], "rs_floor": [0.85, 0.90, 0.95], "atr_stop": [1.5, 2.0, 2.5]},
           "multiday", ("long",)),
)
