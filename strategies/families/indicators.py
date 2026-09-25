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


def daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Row D holds values known at D's open: everything computed through D-1 and then shifted."""
    c, h, l, o, v = daily["close"], daily["high"], daily["low"], daily["open"], daily["volume"].astype(float)
    f = pd.DataFrame(index=daily.index)
    f["d_close"], f["d_open"], f["d_high"], f["d_low"], f["d_vol"] = c, o, h, l, v
    f["d_ret1"] = c.pct_change() * 100
    f["d_vol_avg20"] = v.rolling(20).mean()
    f["d_sma50"] = c.rolling(50).mean()
    f["d_ema20"] = ta.trend.ema_indicator(c, window=20)
    f["d_ema10"] = ta.trend.ema_indicator(c, window=10)
    f["d_rsi14"] = ta.momentum.rsi(c, window=14)
    f["d_rsi3"] = ta.momentum.rsi(c, window=3)
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
    return f.shift(1)


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
    return ind


def closed_bars(bars5: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Drop the in-progress bar: any bar whose start is within 5 minutes of `now`. Used by the live adapter."""
    cutoff = pd.Timestamp(now) - pd.Timedelta(minutes=5)
    return bars5[bars5.index <= cutoff]
