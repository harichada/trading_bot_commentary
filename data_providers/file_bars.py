"""File-based bars loader for cloud/backtest environments.

v-file-bars-2026-09-17. Provides minute bar data from local parquet/CSV
files when Postgres is not available (e.g., Cursor cloud, CI, offline
research). Compatible with the backtest engine's BarsLoader signature.

Usage:
    # As environment variable (preferred for cloud)
    export BARS_DIR=/path/to/bars
    python -m backtest.cli --symbols NVDA AAPL --days 30

    # Programmatic
    from data_providers.file_bars import FileBarsProvider
    provider = FileBarsProvider("/path/to/bars")
    df = provider.get_market_data("NVDA", period_type="day", period=30)

File naming convention:
    {symbol}.parquet  (preferred - smaller, faster)
    {symbol}.csv      (fallback)

Expected columns:
    ts (or datetime index): timestamp
    open, high, low, close, volume: OHLCV data

Data can be exported from KiddoKingdom Postgres using:
    python -m data_providers.file_bars export \\
        --dsn postgresql://... \\
        --symbols NVDA AAPL TSLA \\
        --out /path/to/bars/ \\
        --days 365
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("file_bars")


class FileBarsProvider:
    """Read-only OHLCV provider backed by local parquet/CSV files."""

    def __init__(self, bars_dir: str | Path):
        self.bars_dir = Path(bars_dir)
        if not self.bars_dir.exists():
            logger.warning("Bars directory does not exist: %s", self.bars_dir)

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

        Compatible with PostgresDataProvider.get_market_data signature.
        Bars are resampled from 1-min source data to the requested frequency.
        Columns: Open, High, Low, Close, Volume. Index: timestamp.
        """
        symbol = symbol.upper()
        parquet_path = self.bars_dir / f"{symbol}.parquet"
        csv_path = self.bars_dir / f"{symbol}.csv"

        df = pd.DataFrame()
        try:
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                logger.debug("Loaded %s from parquet (%d rows)", symbol, len(df))
            elif csv_path.exists():
                df = pd.read_csv(csv_path)
                logger.debug("Loaded %s from csv (%d rows)", symbol, len(df))
            else:
                logger.debug("No bars file for %s in %s", symbol, self.bars_dir)
                return df
        except Exception as exc:
            logger.warning("Failed to load bars for %s: %s", symbol, exc)
            return pd.DataFrame()

        if df.empty:
            return df

        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"])
            df = df.set_index("ts")
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

        column_map = {
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                logger.warning("Missing column %s in %s", col, symbol)
                return pd.DataFrame()

        _PERIOD_TO_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
        days = _PERIOD_TO_DAYS.get(period_type, 30) * period

        if end is None:
            end = df.index.max() if not df.empty else datetime.utcnow()
        start = end - timedelta(days=days)

        df = df[(df.index >= start) & (df.index <= end)]

        _FREQ_TO_MINUTES = {"minute": 1, "hour": 60, "day": 60 * 24}
        target_minutes = _FREQ_TO_MINUTES.get(frequency_type, 1) * frequency
        if target_minutes > 1 and not df.empty:
            df = df.resample(f"{target_minutes}min").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }).dropna()

        return df

    def list_symbols(self) -> list[str]:
        """Return list of available symbols."""
        symbols = []
        if self.bars_dir.exists():
            for f in self.bars_dir.iterdir():
                if f.suffix == ".parquet":
                    symbols.append(f.stem.upper())
                elif f.suffix == ".csv" and f.stem.upper() not in symbols:
                    symbols.append(f.stem.upper())
        return sorted(symbols)


def make_file_bars_loader(bars_dir: Path) -> callable:
    """Create a BarsLoader function compatible with backtest engine.

    Returns:
        A function with signature (symbol, days, frequency) -> DataFrame
    """
    provider = FileBarsProvider(bars_dir)

    def _load(symbol: str, days: int, frequency: int) -> pd.DataFrame:
        period_type = "day" if days < 30 else "month"
        period = days if period_type == "day" else max(1, days // 30)
        return provider.get_market_data(
            symbol,
            period_type=period_type,
            period=period,
            frequency_type="minute",
            frequency=frequency,
        )

    return _load


def export_from_postgres(
    dsn: str,
    symbols: list[str],
    out_dir: Path,
    days: int = 365,
    format: str = "parquet",
) -> dict[str, int]:
    """Export minute bars from Postgres to local files.

    Args:
        dsn: PostgreSQL connection string
        symbols: List of symbols to export
        out_dir: Output directory
        days: Number of days to export
        format: Output format ("parquet" or "csv")

    Returns:
        Dict mapping symbol to row count exported
    """
    from data_providers.postgres import PostgresDataProvider

    provider = PostgresDataProvider(dsn)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for symbol in symbols:
        symbol = symbol.upper()
        try:
            df = provider.get_market_data(
                symbol,
                period_type="day" if days < 30 else "month",
                period=days if days < 30 else max(1, days // 30),
                frequency_type="minute",
                frequency=1,
            )
            if df.empty:
                logger.warning("No data for %s", symbol)
                results[symbol] = 0
                continue

            df = df.reset_index()
            df = df.rename(columns={"index": "ts"} if "index" in df.columns else {})
            df.columns = [c.lower() for c in df.columns]

            if format == "parquet":
                df.to_parquet(out_dir / f"{symbol}.parquet", index=False)
            else:
                df.to_csv(out_dir / f"{symbol}.csv", index=False)

            results[symbol] = len(df)
            logger.info("Exported %s: %d rows", symbol, len(df))
        except Exception as exc:
            logger.error("Failed to export %s: %s", symbol, exc)
            results[symbol] = 0

    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for exporting bars from Postgres."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Export minute bars from Postgres to local files",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    export_parser = subparsers.add_parser("export", help="Export bars from Postgres")
    export_parser.add_argument(
        "--dsn",
        default=os.environ.get(
            "POSTGRES_DSN",
            "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev"
        ),
        help="PostgreSQL DSN",
    )
    export_parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        required=True,
        help="Symbols to export",
    )
    export_parser.add_argument(
        "--out", "-o",
        type=Path,
        required=True,
        help="Output directory",
    )
    export_parser.add_argument(
        "--days", "-d",
        type=int,
        default=365,
        help="Days of history to export (default: 365)",
    )
    export_parser.add_argument(
        "--format", "-f",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format (default: parquet)",
    )

    list_parser = subparsers.add_parser("list", help="List available symbols in bars dir")
    list_parser.add_argument(
        "--bars-dir",
        type=Path,
        required=True,
        help="Directory with bars files",
    )

    args = parser.parse_args(argv)

    if args.command == "export":
        results = export_from_postgres(
            dsn=args.dsn,
            symbols=args.symbols,
            out_dir=args.out,
            days=args.days,
            format=args.format,
        )
        total = sum(results.values())
        print(f"\nExported {len(results)} symbols, {total:,} total rows")
        return 0

    elif args.command == "list":
        provider = FileBarsProvider(args.bars_dir)
        symbols = provider.list_symbols()
        print(f"Available symbols ({len(symbols)}):")
        for sym in symbols:
            print(f"  {sym}")
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
