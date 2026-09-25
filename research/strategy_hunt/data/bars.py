"""Postgres minute_bars -> per-symbol Parquet (5-min RTH + daily). Pure transforms plus thin I/O."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger("strategy_hunt.bars")

COLUMNS = ["open", "high", "low", "close", "volume"]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
RTH_START = "09:30"
RTH_LAST_BAR = "15:55"


def to_rth_5min(df1: pd.DataFrame) -> pd.DataFrame:
    """1-min bars (DatetimeIndex) -> 5-min regular-hours bars. Drops empty bins."""
    if df1.empty:
        return _empty("ts")
    df = df1[COLUMNS].sort_index()
    df = df.between_time(RTH_START, "15:59")
    out = df.resample("5min", label="left", closed="left").agg(AGG)
    out = out.dropna(subset=["open"])
    out = out.between_time(RTH_START, RTH_LAST_BAR)
    out.index.name = "ts"
    return out.astype({"volume": "int64"})


def to_daily(df5: pd.DataFrame) -> pd.DataFrame:
    """5-min bars -> one row per session, indexed by normalized date."""
    if df5.empty:
        return _empty("date")
    out = df5[COLUMNS].groupby(df5.index.normalize()).agg(AGG)
    out.index.name = "date"
    return out.astype({"volume": "int64"})


def _empty(index_name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name=index_name))


def bars5_path(root: Path, symbol: str) -> Path:
    return Path(root) / "bars5" / f"{symbol.upper()}.parquet"


def daily_path(root: Path, symbol: str) -> Path:
    return Path(root) / "daily" / f"{symbol.upper()}.parquet"


def load_bars5(root: Path, symbol: str) -> pd.DataFrame:
    return _load(bars5_path(root, symbol), "ts")


def load_daily(root: Path, symbol: str) -> pd.DataFrame:
    return _load(daily_path(root, symbol), "date")


def _load(path: Path, index_name: str) -> pd.DataFrame:
    if not path.exists():
        return _empty(index_name)
    df = pd.read_parquet(path)
    df[index_name] = pd.to_datetime(df[index_name])
    return df.set_index(index_name)[COLUMNS]


def last_ts(root: Path, symbol: str) -> Optional[pd.Timestamp]:
    df = load_bars5(root, symbol)
    return None if df.empty else df.index[-1]


def _read_minute_bars(engine: Any, symbol: str, since: Optional[pd.Timestamp]) -> pd.DataFrame:
    """Read raw 1-min rows for one symbol (optionally after `since`). Columns: ts, open, high, low, close, volume."""
    from sqlalchemy import text

    clause = "AND ts > :since" if since is not None else ""
    q = text(f"SELECT ts, open, high, low, close, volume FROM minute_bars WHERE symbol = :s {clause} ORDER BY ts")
    params = {"s": symbol.upper(), **({"since": since.to_pydatetime()} if since is not None else {})}
    with engine.connect() as conn:
        return pd.read_sql(q, conn, params=params)


def export_symbol(engine: Any, symbol: str, root: Path, since: Optional[pd.Timestamp] = None) -> int:
    """Append new 5-min RTH bars (and rebuild daily) for `symbol`. Returns rows appended."""
    if since is None:
        since = last_ts(root, symbol)
    raw = _read_minute_bars(engine, symbol, since)
    if raw.empty:
        return 0
    raw["ts"] = pd.to_datetime(raw["ts"])
    new5 = to_rth_5min(raw.set_index("ts"))
    if new5.empty:
        return 0
    existing = load_bars5(root, symbol)
    merged = pd.concat([existing, new5])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    _write(bars5_path(root, symbol), merged, "ts")
    _write(daily_path(root, symbol), to_daily(merged), "date")
    return len(new5)


def _write(path: Path, df: pd.DataFrame, index_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.reset_index().rename(columns={"index": index_name}).to_parquet(tmp, index=False)
    tmp.replace(path)


def select_universe(engine: Any, min_days: int = 250, min_adv_dollars: float = 2e7, adv_sessions: int = 60) -> list[str]:
    """Symbols with >= min_days sessions and trailing average dollar volume >= min_adv_dollars."""
    from sqlalchemy import text

    skip = text(
        """WITH RECURSIVE syms AS (
             (SELECT symbol FROM minute_bars ORDER BY symbol LIMIT 1)
             UNION ALL
             SELECT (SELECT symbol FROM minute_bars WHERE symbol > s.symbol ORDER BY symbol LIMIT 1)
             FROM syms s WHERE s.symbol IS NOT NULL)
           SELECT symbol FROM syms WHERE symbol IS NOT NULL"""
    )
    count_days = text("SELECT count(DISTINCT ts::date) FROM minute_bars WHERE symbol = :s")
    adv = text("SELECT avg(close*volume) FROM (SELECT close, volume FROM daily_bars WHERE symbol = :s ORDER BY date DESC LIMIT :k) t")
    keep: list[str] = []
    with engine.connect() as conn:
        for (s,) in conn.execute(skip):
            if conn.execute(count_days, {"s": s}).scalar() < min_days:
                continue
            if min_adv_dollars > 0:
                v = conn.execute(adv, {"s": s, "k": adv_sessions}).scalar()
                if v is None or float(v) < min_adv_dollars:
                    continue
            keep.append(s)
    return sorted(keep)
