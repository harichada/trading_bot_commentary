"""PostgreSQL data provider for ML training.

Adapts the rudra_dev minute_bars table to the same get_market_data interface
that IntegratedMLModel.collect_training_data expects from a Schwab provider.

Timezone contract
-----------------
The minute_bars table stores timestamps as naive ET (America/New_York).
alpaca_ingest converts UTC bar timestamps to ET naive before INSERT so that
timestamps are human-readable and RTH filtering is straightforward in SQL.

This provider owns that contract: get_market_data always returns a
UTC-aware DatetimeIndex, converting the ET-naive DB values on the way out.
Callers never need to handle the DB's internal storage format.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text


_ET = ZoneInfo("America/New_York")

# Shared engine pool — one engine per DSN, reused across all threads.
# SQLAlchemy's connection pool is thread-safe; creating one engine per
# call in a thread pool races during psycopg2 initialisation.
_engines: dict[str, object] = {}
_engines_lock = threading.Lock()


def _get_engine(dsn: str):
    with _engines_lock:
        if dsn not in _engines:
            _engines[dsn] = create_engine(dsn, pool_pre_ping=True, pool_size=10, max_overflow=5)
        return _engines[dsn]

_PERIOD_TO_DAYS = {
    "day": 1,
    "week": 7,
    "month": 30,
    "year": 365,
}

_FREQ_TO_MINUTES = {
    "minute": 1,
    "hour": 60,
    "day": 60 * 24,
}


def _to_et_naive(dt: datetime) -> datetime:
    """Convert any datetime to ET-naive for use in DB queries."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(_ET)
    return dt.replace(tzinfo=None)


class PostgresDataProvider:
    """Read-only OHLCV provider backed by Postgres minute_bars.

    The DB stores timestamps as ET-naive. This class is the single owner of
    that contract — callers always receive a UTC-aware DatetimeIndex.
    """

    def __init__(self, dsn: str, table: str = "minute_bars"):
        self.engine = _get_engine(dsn)
        self.table = table

    def get_market_data(
        self,
        symbol: str,
        period_type: str = "month",
        period: int = 1,
        frequency_type: str = "minute",
        frequency: int = 5,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame of OHLCV bars for `symbol`.

        Bars are resampled from 1-min source data to the requested frequency.
        Columns: Open, High, Low, Close, Volume. Index: UTC-aware DatetimeIndex.

        `end` defaults to now (UTC). Pass an explicit value to replay history.
        """
        if end is None:
            end = datetime.now(timezone.utc)

        days = _PERIOD_TO_DAYS.get(period_type, 30) * period
        # DB stores ET-naive: convert query bounds to ET-naive for correct filtering.
        end_et = _to_et_naive(end)
        start_et = end_et - timedelta(days=days)

        query = text(
            f"""
            SELECT ts, open, high, low, close, volume
            FROM {self.table}
            WHERE symbol = :symbol AND ts >= :start AND ts <= :end
            ORDER BY ts
            """
        )
        with self.engine.connect() as conn:
            df = pd.read_sql(
                query, conn,
                params={"symbol": symbol, "start": start_et, "end": end_et},
            )

        if df.empty:
            return df

        # Attach ET timezone to the naive DB timestamps, then normalise to UTC
        # so callers always work with a consistent UTC-aware index.
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(_ET).dt.tz_convert("UTC")
        df = df.set_index("ts")
        df = df[~df.index.duplicated(keep="last")]

        target_minutes = _FREQ_TO_MINUTES.get(frequency_type, 1) * frequency
        if target_minutes > 1:
            df = df.resample(f"{target_minutes}min").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()

        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        return df
