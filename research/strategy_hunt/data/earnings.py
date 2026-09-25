"""Earnings dates: yfinance cache plus a gap-and-volume fallback inferred from daily bars."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Union

import pandas as pd

logger = logging.getLogger("strategy_hunt.earnings")

_TABLE_COLUMNS = ["eps_estimate", "reported_eps", "surprise_pct"]
_SOURCE_COLUMNS = ("EPS Estimate", "Reported EPS", "Surprise(%)")

FetchResult = Union[pd.DatetimeIndex, pd.DataFrame]


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


def _normalize_date(d) -> pd.Timestamp:
    ts = pd.Timestamp(d)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _to_float(value) -> float:
    return float(value) if pd.notna(value) else float("nan")


def _rows_from_fetch(raw: FetchResult) -> dict[pd.Timestamp, tuple[float, float, float]]:
    """Normalize a fetcher's return value (legacy DatetimeIndex or a DataFrame with
    EPS Estimate / Reported EPS / Surprise(%) columns) into date -> (eps_estimate,
    reported_eps, surprise_pct). Later duplicate dates overwrite earlier ones."""
    rows: dict[pd.Timestamp, tuple[float, float, float]] = {}
    if isinstance(raw, pd.DataFrame):
        series = {col: raw[col] if col in raw.columns else None for col in _SOURCE_COLUMNS}
        for pos, d in enumerate(raw.index):
            rows[_normalize_date(d)] = tuple(
                _to_float(series[col].iloc[pos]) if series[col] is not None else float("nan")
                for col in _SOURCE_COLUMNS
            )
    else:
        for d in pd.DatetimeIndex(raw):
            rows[_normalize_date(d)] = (float("nan"), float("nan"), float("nan"))
    return rows


def update_earnings(root: Path, symbol: str, fetcher: Callable[[str], FetchResult]) -> int:
    rows = _rows_from_fetch(fetcher(symbol))
    dates = sorted(rows)
    table = pd.DataFrame(
        {
            "date": dates,
            "eps_estimate": [rows[d][0] for d in dates],
            "reported_eps": [rows[d][1] for d in dates],
            "surprise_pct": [rows[d][2] for d in dates],
        }
    )
    p = earnings_path(root, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.parquet")
    table.to_parquet(tmp, index=False)
    tmp.replace(p)
    return len(table)


def load_earnings(root: Path, symbol: str) -> set[pd.Timestamp]:
    p = earnings_path(root, symbol)
    if not p.exists():
        return set()
    df = pd.read_parquet(p)
    return {pd.Timestamp(d).normalize() for d in pd.to_datetime(df["date"])}


def load_earnings_table(root: Path, symbol: str) -> pd.DataFrame:
    """Full earnings cache table indexed by normalized, tz-naive date, with float
    columns eps_estimate, reported_eps, surprise_pct. Empty (but columned) frame
    when the cache is missing."""
    p = earnings_path(root, symbol)
    if not p.exists():
        empty = pd.DataFrame(columns=_TABLE_COLUMNS)
        empty.index = pd.DatetimeIndex([], name="date")
        return empty
    df = pd.read_parquet(p)
    df = df.copy()
    df["date"] = [pd.Timestamp(d).normalize() for d in pd.to_datetime(df["date"])]
    return df.set_index("date")[_TABLE_COLUMNS]


def earnings_days(root: Path, symbol: str, daily: pd.DataFrame) -> set[pd.Timestamp]:
    return load_earnings(root, symbol) | infer_event_days(daily)


def yfinance_fetcher(symbol: str) -> pd.DataFrame:
    """Network call; returns an empty DataFrame on any failure so the fallback still applies."""
    empty = pd.DataFrame(columns=list(_SOURCE_COLUMNS))
    try:
        import yfinance as yf

        df = yf.Ticker(symbol).earnings_dates
        if df is None or df.empty:
            return empty
        if "Event Type" in df.columns:
            df = df[df["Event Type"] == "Earnings"]
        idx = df.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        df = df.set_axis(idx)
        cols = [c for c in _SOURCE_COLUMNS if c in df.columns]
        return df[cols]
    except Exception as exc:  # network / parsing errors degrade to the gap fallback
        logger.warning("earnings fetch failed for %s: %s", symbol, exc)
        return empty
