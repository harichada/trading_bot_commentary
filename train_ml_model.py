"""Train the IntegratedMLModel from PostgreSQL minute_bars data.

Usage:
    python train_ml_model.py
    python train_ml_model.py --symbols NVDA TSLA AAPL --days 90 --frequency 5
    python train_ml_model.py --top-n 50 --days 60
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_ml")

DEFAULT_DSN = "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev"
DEFAULT_SYMBOLS = ["PLTR", "NVDA", "TSLA", "AAPL", "MSFT", "GOOG", "AMD", "META"]


def top_symbols_by_volume(dsn: str, n: int, days: int) -> list[str]:
    """Pick the N most-traded symbols over the last `days`."""
    engine = create_engine(dsn)
    sql = text(
        """
        SELECT symbol, SUM(volume) AS total_vol
        FROM minute_bars
        WHERE ts >= NOW() - (:days || ' days')::interval
        GROUP BY symbol
        ORDER BY total_vol DESC
        LIMIT :n
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"days": days, "n": n}).fetchall()
    return [r[0] for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", help="Symbols to train on")
    parser.add_argument("--top-n", type=int, help="Use top-N most-traded symbols instead of --symbols")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days (default 30)")
    parser.add_argument("--frequency", type=int, default=5, help="Bar size in minutes (default 5)")
    parser.add_argument("--dsn", default=os.environ.get("POSTGRES_DSN", DEFAULT_DSN))
    args = parser.parse_args()

    if args.symbols and args.top_n:
        parser.error("Use --symbols OR --top-n, not both")

    if args.top_n:
        logger.info("Selecting top %d symbols by volume over last %d days", args.top_n, args.days)
        symbols = top_symbols_by_volume(args.dsn, args.top_n, args.days)
        logger.info("Selected: %s", ", ".join(symbols[:10]) + (f", ... ({len(symbols)} total)" if len(symbols) > 10 else ""))
    else:
        symbols = args.symbols or DEFAULT_SYMBOLS

    from data_providers.postgres import PostgresDataProvider
    from ml.models import IntegratedMLModel

    provider = PostgresDataProvider(args.dsn)
    model = IntegratedMLModel(brain=None, commentary_system=None)

    period_type = "day" if args.days < 30 else "month"
    period = args.days if period_type == "day" else max(1, args.days // 30)

    logger.info(
        "Collecting training data: %d symbols, %d days, %dmin bars",
        len(symbols), args.days, args.frequency,
    )

    all_features: list = []
    all_labels: list = []
    per_symbol: dict[str, int] = {}

    for symbol in symbols:
        try:
            df = provider.get_market_data(
                symbol,
                period_type=period_type,
                period=period,
                frequency_type="minute",
                frequency=args.frequency,
            )
        except Exception as e:
            logger.error("Failed to load %s: %s", symbol, e)
            continue

        if df.empty or len(df) < 100:
            logger.warning("%s: insufficient data (%d bars), skipping", symbol, len(df))
            continue

        sym_features: list = []
        sym_labels: list = []
        for i in range(50, len(df) - 20):
            window = df.iloc[: i + 1]
            features = model.feature_extractor.extract_from_dataframe(window)

            current_price = df["Close"].iloc[i]
            future_price = df["Close"].iloc[i + 10]
            change = (future_price - current_price) / current_price

            if change > 0.003:
                label = 2
            elif change < -0.003:
                label = 0
            else:
                label = 1

            sym_features.append([features[name] for name in model.feature_names])
            sym_labels.append(label)

        all_features.extend(sym_features)
        all_labels.extend(sym_labels)
        per_symbol[symbol] = len(sym_labels)
        logger.info("%s: %d samples", symbol, len(sym_labels))

    if not all_features:
        logger.error("No training samples collected — aborting")
        return 1

    X = np.array(all_features)
    y = np.array(all_labels)
    dist = Counter(y.tolist())
    total = len(y)
    logger.info(
        "Total samples: %d  |  BUY=%d (%.1f%%)  HOLD=%d (%.1f%%)  SELL=%d (%.1f%%)",
        total,
        dist[2], 100 * dist[2] / total,
        dist[1], 100 * dist[1] / total,
        dist[0], 100 * dist[0] / total,
    )

    if dist[1] / total > 0.85:
        logger.warning(
            "HOLD class > 85%% — model will likely collapse to always-HOLD. "
            "Consider widening the label threshold or shortening the horizon."
        )

    logger.info("Training XGBoost classifier...")
    success = model.train(X, y)
    if not success:
        logger.error("Training failed")
        return 1

    logger.info("Done. Model saved to %s", model.model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
