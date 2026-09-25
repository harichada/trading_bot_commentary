"""News-driven families using Alpaca event timestamps: intraday catalyst and multi-day post-news drift."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.families.common import Family, bars_to_1500, col, make_entry, one_per_session, spaced_by_sessions
from strategies.families.models import Entry

CATALYST_ATR_STOP = 1.5
DRIFT_ATR_STOP = 1.5
DRIFT_TARGET_R = 3.0
RVOL_MIN = 2.0


def _catalyst_entry(bars5, ind, i, side, target_r):
    atr = col(ind, "atr")[i]
    if np.isnan(atr):
        return None
    px = float(bars5["open"].iloc[i + 1])
    dist = CATALYST_ATR_STOP * atr
    stop = px - dist if side == "long" else px + dist
    target = px + target_r * dist if side == "long" else px - target_r * dist
    bar_of_day = col(ind, "bar_of_day")[i]
    return make_entry(bars5, i, side, stop, target, bars_to_1500(bar_of_day + 1), "intraday", f"news_{side}")


def news_catalyst_entries(bars5: pd.DataFrame, ind: pd.DataFrame, daily: pd.DataFrame, news: pd.DataFrame,
                          params: dict) -> list[Entry]:
    if news is None or news.empty or bars5.empty:
        return []
    win_bars = int(params["window_minutes"]) // 5
    s_min, target_r = params["min_abs_sentiment"], params["target_r"]
    idx = bars5.index
    o, c = bars5["open"].to_numpy(), bars5["close"].to_numpy()
    rvol_arr, vwap_arr, bod_arr = col(ind, "rvol"), col(ind, "vwap"), col(ind, "bar_of_day")
    out: list[Entry] = []
    for _, item in news.iterrows():
        ts, sent = pd.Timestamp(item["created_at"]), float(item["sentiment"])
        if abs(sent) < s_min or not (pd.Timestamp("09:30").time() <= ts.time() <= pd.Timestamp("15:00").time()):
            continue
        i0 = int(idx.searchsorted(ts))
        if i0 >= len(idx) or idx[i0].normalize() != ts.normalize():
            continue
        side = "long" if sent > 0 else "short"
        for i in range(i0, min(i0 + win_bars, len(idx) - 1)):
            if idx[i].normalize() != ts.normalize() or bod_arr[i] > 60:
                break
            rvol, vwap = rvol_arr[i], vwap_arr[i]
            if np.isnan(rvol) or np.isnan(vwap) or rvol < RVOL_MIN:
                continue
            confirmed = (c[i] > o[i] and c[i] > vwap) if side == "long" else (c[i] < o[i] and c[i] < vwap)
            if confirmed:
                e = _catalyst_entry(bars5, ind, i, side, target_r)
                if e:
                    out.append(e)
                break
    return one_per_session(out)


def post_news_drift_entries(bars5: pd.DataFrame, ind: pd.DataFrame, daily: pd.DataFrame, news: pd.DataFrame,
                            params: dict) -> list[Entry]:
    if daily is None or daily.empty or news is None or news.empty:
        return []
    vol_mult, min_ret, hold = params["vol_mult"], params["min_ret_pct"], int(params["hold_days"])
    ret1 = daily["close"].pct_change() * 100
    avg_vol = daily["volume"].shift(1).rolling(20, min_periods=10).mean()
    news_days = {pd.Timestamp(t).normalize() for t in news["created_at"]}
    session_first = {d: i for i, d in reversed(list(enumerate(bars5.index.normalize())))}
    d_atr14_arr = col(ind, "d_atr14")
    days = list(daily.index)
    out: list[Entry] = []
    for k, d in enumerate(days[:-1]):
        r, v, av = ret1.iloc[k], daily["volume"].iloc[k], avg_vol.iloc[k]
        if np.isnan(r) or np.isnan(av) or abs(r) < min_ret or v < vol_mult * av or d.normalize() not in news_days:
            continue
        nxt = days[k + 1].normalize()
        j = session_first.get(nxt)
        if j is None or j == 0:
            continue
        atr = d_atr14_arr[j]
        if np.isnan(atr):
            continue
        side = "long" if r > 0 else "short"
        px = float(bars5["open"].iloc[j])
        dist = DRIFT_ATR_STOP * atr
        stop = px - dist if side == "long" else px + dist
        target = px + DRIFT_TARGET_R * dist if side == "long" else px - DRIFT_TARGET_R * dist
        e = make_entry(bars5, j - 1, side, stop, target, hold, "multiday", f"drift_{side}")
        if e:
            out.append(e)
    return spaced_by_sessions(out, hold)


FAMILIES_NEWS: tuple[Family, ...] = (
    Family("news_catalyst", news_catalyst_entries,
           {"window_minutes": [15, 30, 60], "min_abs_sentiment": [0.3, 0.5], "target_r": [1.5, 2.0, 3.0]},
           "intraday", ("long", "short")),
    Family("post_news_drift", post_news_drift_entries,
           {"vol_mult": [2.0, 3.0], "min_ret_pct": [3.0, 4.0, 6.0], "hold_days": [3, 5, 10]},
           "multiday", ("long", "short")),
)
