"""Feature builder for the side classifier.

v-classifier-features-2026-05-13. Takes raw OHLCV bars (pandas
DataFrame) + optional news + optional benchmark frames and produces
the SymbolFeatures dataclass the classifier consumes.

Pure functions over pandas. No I/O, no Schwab calls, no GPU. The
research backtest pipes historical bars in; the live bot (once
integrated behind ``Config.USE_SIDE_CLASSIFIER``) will pipe in the
PriceBook + the screener's recent-bar snapshot.

This module deliberately mirrors a subset of the indicator math the
strategies use; per PLAN §11 open-question 7 the duplication is
intentional so the classifier's feature definitions stay frozen
against the trained model.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.classifier.types import SymbolFeatures


def _sma(series: pd.Series, window: int) -> Optional[float]:
    if len(series) < window:
        return None
    return float(series.iloc[-window:].mean())


def _atr_pct(df: pd.DataFrame, window: int = 14) -> Optional[float]:
    """ATR-14 / latest close, daily. Returns None if insufficient
    history. Standard True Range = max(H-L, |H-prev_close|, |L-prev_close|)."""
    if len(df) < window + 1:
        return None
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.iloc[-window:].mean()
    return float(atr / close.iloc[-1])


def _count_consecutive_higher_highs(df: pd.DataFrame, max_n: int = 5) -> int:
    """Count consecutive trailing bars where high > prior high.
    Stops at the first non-higher-high. Returns 0 if no streak."""
    highs = df["high"].iloc[-(max_n + 1):]
    if len(highs) < 2:
        return 0
    n = 0
    for i in range(len(highs) - 1, 0, -1):
        if highs.iloc[i] > highs.iloc[i - 1]:
            n += 1
        else:
            break
    return n


def _count_consecutive_lower_lows(df: pd.DataFrame, max_n: int = 5) -> int:
    lows = df["low"].iloc[-(max_n + 1):]
    if len(lows) < 2:
        return 0
    n = 0
    for i in range(len(lows) - 1, 0, -1):
        if lows.iloc[i] < lows.iloc[i - 1]:
            n += 1
        else:
            break
    return n


def _up_vs_down_volume(df: pd.DataFrame, window: int = 5) -> float:
    """Sum of volume on up-close days / sum on down-close days,
    last ``window`` sessions. Returns 1.0 if denominator is zero
    (treat as neutral rather than infinite)."""
    if len(df) < window + 1:
        return 1.0
    recent = df.iloc[-window:]
    direction = recent["close"].diff()
    up_vol = recent.loc[direction > 0, "volume"].sum()
    down_vol = recent.loc[direction < 0, "volume"].sum()
    if down_vol == 0:
        return 1.0 if up_vol == 0 else 5.0   # cap to avoid runaway
    return float(up_vol / down_vol)


def _vol_ratio_today(df: pd.DataFrame, window: int = 20) -> float:
    if len(df) < window + 1:
        return 1.0
    today = float(df["volume"].iloc[-1])
    avg = float(df["volume"].iloc[-(window + 1):-1].mean())
    if avg <= 0:
        return 1.0
    return today / avg


def _avg_daily_volume(df: pd.DataFrame, window: int = 20) -> Optional[float]:
    if len(df) < window:
        return None
    return float(df["volume"].iloc[-window:].mean())


def _rs_vs_benchmark(symbol_df: pd.DataFrame, bench_df: pd.DataFrame, window: int = 5) -> float:
    """Cumulative return of symbol minus cumulative return of bench,
    last ``window`` sessions. Both DataFrames indexed by date."""
    if len(symbol_df) < window + 1 or len(bench_df) < window + 1:
        return 0.0
    sym_ret = symbol_df["close"].iloc[-1] / symbol_df["close"].iloc[-(window + 1)] - 1.0
    bench_ret = bench_df["close"].iloc[-1] / bench_df["close"].iloc[-(window + 1)] - 1.0
    return float(sym_ret - bench_ret)


def _intraday_trend_slope(intraday_df: Optional[pd.DataFrame]) -> float:
    """Regression slope of intraday close prices (today's session,
    5-min bars). Returned normalized by the latest close so the unit
    is "fraction-per-bar". Returns 0.0 if no data."""
    if intraday_df is None or len(intraday_df) < 2:
        return 0.0
    closes = intraday_df["close"].values
    x = np.arange(len(closes))
    slope, _intercept = np.polyfit(x, closes, 1)
    return float(slope / closes[-1])


def _rsi_14(df: pd.DataFrame) -> Optional[float]:
    """Wilder RSI-14 on closes. Returns None if < 15 bars."""
    if len(df) < 15:
        return None
    delta = df["close"].diff().iloc[-14:]
    up = delta.clip(lower=0).mean()
    down = (-delta.clip(upper=0)).mean()
    if down == 0:
        return 100.0
    rs = up / down
    return float(100.0 - (100.0 / (1.0 + rs)))


def build_features(
    symbol: str,
    daily_bars: pd.DataFrame,
    *,
    intraday_bars: Optional[pd.DataFrame] = None,
    benchmark_daily: Optional[pd.DataFrame] = None,
    sector_daily: Optional[pd.DataFrame] = None,
    news_sentiment_7d: float = 0.0,
    news_fresh_count_today: int = 0,
    news_recency_min: Optional[float] = None,
    earnings_within_2d: bool = False,
    halt_today: bool = False,
    hard_to_borrow: bool = False,
    ex_dividend_within_1d: bool = False,
    extras: Optional[dict] = None,
) -> SymbolFeatures:
    """Build a SymbolFeatures from raw bars.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    daily_bars : pd.DataFrame
        Indexed by date, columns at minimum {open, high, low, close,
        volume}. Most recent bar is the "current day" being classified.
        Use yfinance / Schwab daily endpoint to populate. Needs ≥ 50
        bars for SMA-50; if fewer, the corresponding fields are None.
    intraday_bars : pd.DataFrame | None
        Optional 5-min bars for today's intraday trend computation.
    benchmark_daily : pd.DataFrame | None
        Daily bars for SPY (or similar) for relative strength.
    sector_daily : pd.DataFrame | None
        Daily bars for sector ETF for sector-relative strength.
    news_*, earnings_within_2d, halt_today, hard_to_borrow,
    ex_dividend_within_1d
        External signals; the live bot will populate these from its
        existing news_aggregator + halt detector; the research
        harness uses Alpaca historical news + an earnings calendar.

    Returns
    -------
    SymbolFeatures
        Suitable for ``core.classifier.classify`` / ``classify_rule_based``.
    """
    if "close" not in daily_bars.columns:
        raise ValueError("daily_bars must contain a 'close' column")
    close = float(daily_bars["close"].iloc[-1])

    rs_vs_spy = (
        _rs_vs_benchmark(daily_bars, benchmark_daily)
        if benchmark_daily is not None else 0.0
    )
    rs_vs_sector = (
        _rs_vs_benchmark(daily_bars, sector_daily)
        if sector_daily is not None else 0.0
    )

    return SymbolFeatures(
        symbol=symbol,
        close=close,
        sma_20_daily=_sma(daily_bars["close"], 20),
        sma_50_daily=_sma(daily_bars["close"], 50),
        intraday_trend_slope=_intraday_trend_slope(intraday_bars),
        lower_lows=_count_consecutive_lower_lows(daily_bars),
        higher_highs=_count_consecutive_higher_highs(daily_bars),
        daily_atr_pct=_atr_pct(daily_bars),
        rsi_14=_rsi_14(daily_bars),
        vol_ratio_today=_vol_ratio_today(daily_bars),
        up_vs_down_volume_5d=_up_vs_down_volume(daily_bars),
        avg_daily_volume_20d=_avg_daily_volume(daily_bars),
        news_sentiment_7d=news_sentiment_7d,
        news_fresh_count_today=news_fresh_count_today,
        news_recency_min=news_recency_min,
        rs_vs_spy_5d=rs_vs_spy,
        rs_vs_sector_5d=rs_vs_sector,
        earnings_within_2d=earnings_within_2d,
        ex_dividend_within_1d=ex_dividend_within_1d,
        halt_today=halt_today,
        hard_to_borrow=hard_to_borrow,
        extras=extras or {},
    )
