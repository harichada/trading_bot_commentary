"""Earnings dates: yfinance cache plus a gap-and-volume fallback inferred from daily bars."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger("strategy_hunt.earnings")


def earnings_path(root: Path, symbol: str) -> Path:
    return Path(root) / "earnings" / f"{symbol.upper()}.parquet"


def infer_event_days(daily: pd.DataFrame, gap_pct: float = 8.0, vol_mult: float = 3.0) -> set[pd.Timestamp]:
    """Sessions that opened >= gap_pct from prior close with volume >= vol_mult x trailing-20 mean."""
    if daily.empty or len(daily) < 2:
        return set()
    prev_close = daily["close"].shift(1)
    gap = (daily["open"] - prev_close).abs() / prev_close * 100
    avg_vol = daily["volume"].shift(1).rolling(20, min_periods=5).mean()
    hit = (gap >= gap_pct) & (daily["volume"] >= vol_mult * avg_vol)
    return {pd.Timestamp(d).normalize() for d in daily.index[hit.fillna(False)]}


def update_earnings(root: Path, symbol: str, fetcher: Callable[[str], pd.DatetimeIndex]) -> int:
    dates = pd.DatetimeIndex(fetcher(symbol))
    dates = pd.DatetimeIndex(sorted({pd.Timestamp(d).tz_localize(None).normalize() if getattr(d, "tzinfo", None)
                                     else pd.Timestamp(d).normalize() for d in dates}))
    p = earnings_path(root, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.parquet")
    pd.DataFrame({"date": dates}).to_parquet(tmp, index=False)
    tmp.replace(p)
    return len(dates)


def load_earnings(root: Path, symbol: str) -> set[pd.Timestamp]:
    p = earnings_path(root, symbol)
    if not p.exists():
        return set()
    df = pd.read_parquet(p)
    return {pd.Timestamp(d).normalize() for d in pd.to_datetime(df["date"])}


def earnings_days(root: Path, symbol: str, daily: pd.DataFrame) -> set[pd.Timestamp]:
    return load_earnings(root, symbol) | infer_event_days(daily)


def yfinance_fetcher(symbol: str) -> pd.DatetimeIndex:
    """Network call; returns an empty index on any failure so the fallback still applies."""
    try:
        import yfinance as yf

        df = yf.Ticker(symbol).earnings_dates
        if df is None or df.empty:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(df.index)
    except Exception as exc:  # network / parsing errors degrade to the gap fallback
        logger.warning("earnings fetch failed for %s: %s", symbol, exc)
        return pd.DatetimeIndex([])
