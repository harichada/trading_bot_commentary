"""Closed-bar indicator frame for candidate families. Daily features are lagged one session (no lookahead)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import ta

RVOL_LOOKBACK = 20


def _safe_atr(h: pd.Series, l: pd.Series, c: pd.Series, window: int) -> pd.Series:
    """ta.volatility.average_true_range indexes out of bounds when len(c) < window; every other
    indicator here degrades to NaN in that case, so match that contract instead of crashing."""
    if len(c) < window:
        return pd.Series(np.nan, index=c.index)
    return ta.volatility.average_true_range(h, l, c, window=window)


def _session_groups(bars5: pd.DataFrame):
    return bars5.groupby(bars5.index.normalize(), sort=False)


def _intraday(bars5: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = bars5["close"], bars5["high"], bars5["low"], bars5["volume"].astype(float)
    ind = pd.DataFrame(index=bars5.index)
    ind["sma_20"] = c.rolling(20).mean()
    ind["sma_50"] = c.rolling(50).mean()
    ind["ema_20"] = ta.trend.ema_indicator(c, window=20)
    ind["rsi"] = ta.momentum.rsi(c, window=14)
    macd = ta.trend.MACD(c)
    ind["macd"], ind["macd_signal"], ind["macd_hist"] = macd.macd(), macd.macd_signal(), macd.macd_diff()
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    ind["bb_upper"], ind["bb_mid"], ind["bb_lower"] = bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()
    ind["atr"] = _safe_atr(h, l, c, window=14)
    ind["adx"] = ta.trend.ADXIndicator(h, l, c, window=14).adx()
    ind["high_20"] = h.rolling(20).max().shift(1)
    ind["low_20"] = l.rolling(20).min().shift(1)
    ind["volume_ratio"] = v / (v.rolling(20).mean() + 1e-10)
    return ind


def _session_columns(bars5: pd.DataFrame) -> pd.DataFrame:
    g = _session_groups(bars5)
    out = pd.DataFrame(index=bars5.index)
    out["bar_of_day"] = g.cumcount()
    out["session_open"] = g["open"].transform("first")
    session_close = g["close"].transform("last")
    last_close_by_day = session_close.groupby(bars5.index.normalize()).first()
    prev_close_by_day = last_close_by_day.shift(1)
    out["prev_close"] = prev_close_by_day.reindex(bars5.index.normalize()).to_numpy()
    out["gap_pct"] = (out["session_open"] / out["prev_close"] - 1) * 100
    tp = (bars5["high"] + bars5["low"] + bars5["close"]) / 3
    v = bars5["volume"].astype(float)
    cum_v = g["volume"].cumsum().astype(float)
    cum_tpv = (tp * v).groupby(bars5.index.normalize()).cumsum()
    out["vwap"] = cum_tpv / cum_v.replace(0, np.nan)
    cum_tp2v = (tp * tp * v).groupby(bars5.index.normalize()).cumsum()
    var = cum_tp2v / cum_v.replace(0, np.nan) - out["vwap"] ** 2
    out["vwap_sd"] = np.sqrt(var.clip(lower=0))
    # rvol: cumulative session volume vs mean cumulative volume at the same bar over the prior 20 sessions
    cum_by = pd.DataFrame({"day": bars5.index.normalize(), "bod": out["bar_of_day"].to_numpy(), "cum": cum_v.to_numpy()})
    pivot = cum_by.pivot(index="day", columns="bod", values="cum")
    avg = pivot.shift(1).rolling(RVOL_LOOKBACK, min_periods=RVOL_LOOKBACK).mean()
    avg_long = avg.stack().rename("avg")
    key = pd.MultiIndex.from_arrays([cum_by["day"], cum_by["bod"]])
    out["rvol"] = (cum_v.to_numpy() / avg_long.reindex(key).to_numpy())
    for mins in (15, 30):
        k = mins // 5
        first_k = out["bar_of_day"] < k
        hi = bars5["high"].where(first_k).groupby(bars5.index.normalize()).cummax()
        lo = bars5["low"].where(first_k).groupby(bars5.index.normalize()).cummin()
        formed = out["bar_of_day"] >= k
        or_hi = hi.groupby(bars5.index.normalize()).transform("max").where(formed)
        or_lo = lo.groupby(bars5.index.normalize()).transform("min").where(formed)
        out[f"or{mins}_high"], out[f"or{mins}_low"] = or_hi, or_lo
        rng_by_day = (or_hi - or_lo).groupby(bars5.index.normalize()).first()
        avg_rng = rng_by_day.shift(1).rolling(20, min_periods=10).mean()
        out[f"or{mins}_range_avg20"] = avg_rng.reindex(bars5.index.normalize()).to_numpy()
    return out


def _down_streak(ret1: pd.Series) -> pd.Series:
    """Number of consecutive sessions with a negative close-to-close return, ending at each row.
    A NaN return (the first session) counts as "not down" and resets the streak to 0."""
    down = (ret1 < 0).astype(int)
    return down.groupby((down == 0).cumsum()).cumsum().astype(float)


def _week_start(dates) -> pd.Series:
    """1.0 on the first session of each week -- the session whose weekday is <= the previous
    session's (Monday after Friday, or a Tuesday after a Monday holiday) -- else 0.0.

    Unlike every d_* feature this is NOT lagged: a session's own date is known at its open, so
    lagging it would move the flag to the wrong session. The first row in the frame has no
    predecessor and is 0.0 (a week start cannot be established from one session)."""
    idx = pd.DatetimeIndex(dates)
    wd = pd.Series(idx.weekday.astype(float), index=dates)
    return (wd <= wd.shift(1)).astype(float)


def _month_start(dates) -> pd.Series:
    """1.0 on the first session of each calendar month -- the session whose day-of-month is <=
    the previous session's (a business-day calendar makes day-of-month strictly increase within a
    month, so this can only trip at a month boundary; mirrors `_week_start`'s weekday comparison,
    generalized from weeks to months) -- else 0.0.

    Unlike every d_* feature this is NOT lagged: a session's own date is known at its open, so
    lagging it would move the flag to the wrong session (same rationale as `_week_start`). The
    first row in the frame has no predecessor and is 0.0 (a month start cannot be established
    from one session).

    Known limitation: a single missing session (an ordinary market holiday) is always safe -- the
    next available session's day-of-month is still compared against the last real one, wherever
    it fell. But a large data hole (roughly a 3+ week gap, e.g. a symbol with an extended trading
    halt) can miss a boundary: if the session after the hole lands on a day-of-month that is still
    numerically greater than the last session's before the hole, the comparison reads as "still the
    same month" even though one or more calendar months were skipped entirely."""
    idx = pd.DatetimeIndex(dates)
    dom = pd.Series(idx.day.astype(float), index=dates)
    return (dom <= dom.shift(1)).astype(float)


def daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Row D holds values known at D's open: everything computed through D-1 and then shifted."""
    c, h, l, o, v = daily["close"], daily["high"], daily["low"], daily["open"], daily["volume"].astype(float)
    f = pd.DataFrame(index=daily.index)
    f["d_close"], f["d_open"], f["d_high"], f["d_low"], f["d_vol"] = c, o, h, l, v
    f["d_ret1"] = c.pct_change() * 100
    f["d_vol_avg20"] = v.rolling(20).mean()
    f["d_sma5"] = c.rolling(5).mean()
    f["d_sma50"] = c.rolling(50).mean()
    f["d_sma200"] = c.rolling(200).mean()
    f["d_ema20"] = ta.trend.ema_indicator(c, window=20)
    f["d_ema10"] = ta.trend.ema_indicator(c, window=10)
    f["d_rsi14"] = ta.momentum.rsi(c, window=14)
    f["d_rsi3"] = ta.momentum.rsi(c, window=3)
    f["d_rsi2"] = ta.momentum.rsi(c, window=2)
    f["d_down_streak"] = _down_streak(f["d_ret1"])
    bb20 = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    bb25 = ta.volatility.BollingerBands(c, window=20, window_dev=2.5)
    f["d_bb_mid"], f["d_bb20_upper"], f["d_bb20_lower"] = bb20.bollinger_mavg(), bb20.bollinger_hband(), bb20.bollinger_lband()
    f["d_bb25_upper"], f["d_bb25_lower"] = bb25.bollinger_hband(), bb25.bollinger_lband()
    f["d_atr14"] = _safe_atr(h, l, c, window=14)
    f["d_atr20"] = _safe_atr(h, l, c, window=20)
    f["d_atr100"] = _safe_atr(h, l, c, window=100)
    for n in (20, 55):
        f[f"d_high{n}"] = h.rolling(n).max().shift(1)   # excludes the row's own day
        f[f"d_low{n}"] = l.rolling(n).min().shift(1)
    lagged = f.shift(1)
    lagged["d_week_start"] = _week_start(daily.index)     # current session's own date: not lagged
    lagged["d_month_start"] = _month_start(daily.index)   # current session's own date: not lagged
    return lagged


def indicator_frame(bars5: pd.DataFrame, daily: pd.DataFrame, spy5: Optional[pd.DataFrame] = None,
                     context: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    ind = pd.concat([_intraday(bars5), _session_columns(bars5)], axis=1)
    if not daily.empty:
        df = daily_features(daily)
        ind = ind.join(df.reindex(bars5.index.normalize()).set_index(bars5.index))
    if spy5 is not None and not spy5.empty:
        spy_open = spy5.groupby(spy5.index.normalize())["open"].transform("first")
        spy_ret = ((spy5["close"] / spy_open - 1) * 100).reindex(bars5.index)
        own_ret = (bars5["close"] / ind["session_open"] - 1) * 100
        ind["rs_vs_spy_pct"] = own_ret - spy_ret
    else:
        ind["rs_vs_spy_pct"] = np.nan
    if context is not None:
        ctx_df = context.add_prefix("ctx_")
        ind = ind.join(ctx_df.reindex(bars5.index.normalize()).set_index(bars5.index))
        ind = _mask_opening_ctx(ind)
    return ind


# v-playbook-desk-b1-6 (final review, Minor 6). The universe-context frame is per *session*, so
# joining it onto 5-minute bars broadcasts a session's value onto every bar of that session --
# including bars before the value could exist. `rvol_open5` is built from the first 5-minute bar
# and `rvol_open15` from the first three, so a bar-0 caller reading either would be reading bars
# that had not closed yet. Masking them here makes the known-at-open contract mechanical instead
# of a comment in research/strategy_hunt/context.py that a new family could quietly violate.
# `rvol_rank5` is a same-date cross-sectional rank *of* rvol_open5 and inherits its bar.
CTX_MIN_BAR_OF_DAY = {"ctx_rvol_open5": 1, "ctx_rvol_rank5": 1, "ctx_rvol_open15": 3}


def _mask_opening_ctx(ind: pd.DataFrame) -> pd.DataFrame:
    """NaN the opening-window `ctx_*` columns on the bars that precede their inputs' close."""
    if "bar_of_day" not in ind.columns:
        return ind
    bod = ind["bar_of_day"]
    for name, min_bar in CTX_MIN_BAR_OF_DAY.items():
        if name in ind.columns:
            ind[name] = ind[name].where(bod >= min_bar)
    return ind


def closed_bars(bars5: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Drop the in-progress bar: any bar whose start is within 5 minutes of `now`. Used by the live adapter."""
    cutoff = pd.Timestamp(now) - pd.Timedelta(minutes=5)
    return bars5[bars5.index <= cutoff]
