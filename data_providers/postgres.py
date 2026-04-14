"""PostgreSQL data provider for ML training.

Adapts the rudra_dev minute_bars table to the same get_market_data interface
that IntegratedMLModel.collect_training_data expects from a Schwab provider.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text


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


class PostgresDataProvider:
    """Read-only OHLCV provider backed by Postgres minute_bars."""

    def __init__(self, dsn: str, table: str = "minute_bars"):
        self.engine = create_engine(dsn, pool_pre_ping=True)
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
        Columns: Open, High, Low, Close, Volume. Index: timestamp.
        """
        days = _PERIOD_TO_DAYS.get(period_type, 30) * period
        if end is None:
            with self.engine.connect() as conn:
                row = conn.execute(
                    text(f"SELECT MAX(ts) FROM {self.table} WHERE symbol = :s"),
                    {"s": symbol},
                ).fetchone()
            end = row[0] if row and row[0] else datetime.utcnow()
        start = end - timedelta(days=days)

        query = text(
            f"""
            SELECT ts, open, high, low, close, volume
            FROM {self.table}
            WHERE symbol = :symbol AND ts >= :start AND ts <= :end
            ORDER BY ts
            """
        )
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"symbol": symbol, "start": start, "end": end})

        if df.empty:
            return df

        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts")

        target_minutes = _FREQ_TO_MINUTES.get(frequency_type, 1) * frequency
        if target_minutes > 1:
            df = df.resample(f"{target_minutes}min").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()

        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        return df
