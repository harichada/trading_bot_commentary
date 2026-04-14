"""Train IntegratedMLModel from PostgreSQL minute_bars with honest metrics.

Walk-forward cross-validation + per-class precision/recall report.
Model artifact format is inherited from IntegratedMLModel (unchanged).

Usage:
    # Default: 150 top symbols, 180 days, walk-forward CV, 5-min bars
    python train_ml_model.py

    # Explicit basket, quick no-CV training (for debugging)
    python train_ml_model.py --symbols NVDA TSLA --days 30 --no-walk-forward

    # Override CV fold count
    python train_ml_model.py --folds 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_ml")

DEFAULT_DSN = "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev"
DEFAULT_TOP_N = 150
DEFAULT_DAYS = 180
DEFAULT_FOLDS = 5


# Leveraged / inverse / short ETFs — intraday dynamics are artificial (3x
# amplified, daily decay, rebalancing). Their bars don't represent the
# underlying supply/demand we want the model to learn.
_LEVERAGED_ETFS = frozenset({
    "SOXL", "SOXS", "TQQQ", "SQQQ", "UPRO", "SPXU", "SPXS", "SPXL",
    "TNA", "TZA", "FAS", "FAZ", "UDOW", "SDOW", "URTY", "SRTY",
    "LABU", "LABD", "YINN", "YANG", "NUGT", "DUST", "JNUG", "JDST",
    "GUSH", "DRIP", "DRN", "DRV", "MIDU", "WEBL", "WEBS",
    # 2x pairs
    "QLD", "QID", "SSO", "SDS", "DDM", "DXD", "UYG", "SKF",
    # VIX ETNs / products (also pathological)
    "UVXY", "SVXY", "VIXY", "VXX", "TVIX",
})


def top_symbols_by_volume(
    dsn: str,
    n: int,
    days: int,
    exclude_leveraged: bool = True,
) -> list[str]:
    """Top-N by dollar-ish volume, optionally excluding leveraged ETFs."""
    engine = create_engine(dsn)
    # Fetch extra so we have room after filtering
    fetch = n * 2 if exclude_leveraged else n
    sql = text(
        "SELECT symbol, SUM(volume) AS total_vol "
        "FROM minute_bars "
        "WHERE ts >= NOW() - (:days || ' days')::interval "
        "GROUP BY symbol ORDER BY total_vol DESC LIMIT :fetch"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"days": days, "fetch": fetch}).fetchall()
    symbols = [r[0] for r in rows]
    if exclude_leveraged:
        symbols = [s for s in symbols if s not in _LEVERAGED_ETFS]
    return symbols[:n]


def collect_samples(
    symbols: list[str],
    days: int,
    frequency: int,
    dsn: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Return (X, y, timestamps, per_symbol_counts)."""
    from data_providers.postgres import PostgresDataProvider
    from ml.models import IntegratedMLModel

    provider = PostgresDataProvider(dsn)
    model = IntegratedMLModel(brain=None, commentary_system=None)

    period_type = "day" if days < 30 else "month"
    period = days if period_type == "day" else max(1, days // 30)

    all_feats: list[list[float]] = []
    all_labels: list[int] = []
    all_ts: list[Any] = []
    per_symbol: dict[str, int] = {}

    for i, symbol in enumerate(symbols):
        try:
            df = provider.get_market_data(
                symbol, period_type=period_type, period=period,
                frequency_type="minute", frequency=frequency,
            )
        except Exception as e:
            logger.error("load_failed symbol=%s err=%s", symbol, e)
            continue

        if df.empty or len(df) < 100:
            logger.warning("insufficient_data symbol=%s bars=%d", symbol, len(df))
            continue

        # Vectorised feature extraction over the whole series (~100x faster
        # than per-bar recomputation on growing windows).
        feat_df = model.feature_extractor.batch_extract(df)
        if feat_df.empty:
            logger.warning("no_features symbol=%s", symbol)
            continue

        close = df["Close"].to_numpy()
        # Slice [50, len-20) so we have 50 bars of history + 10-bar forward label
        start, stop = 50, len(df) - 20
        if stop <= start:
            continue

        sym_feats = feat_df.iloc[start:stop][model.feature_names].to_numpy()
        future = close[start + 10 : stop + 10]
        present = close[start:stop]
        change = (future - present) / np.where(present == 0, 1, present)

        labels = np.where(change > 0.003, 2,
                 np.where(change < -0.003, 0, 1))

        all_feats.extend(sym_feats.tolist())
        all_labels.extend(labels.tolist())
        all_ts.extend(df.index[start:stop].tolist())
        sym_count = len(labels)
        per_symbol[symbol] = sym_count
        if (i + 1) % 10 == 0 or i + 1 == len(symbols):
            logger.info(
                "collection_progress done=%d/%d cumulative_samples=%d latest=%s/%d",
                i + 1, len(symbols), len(all_feats), symbol, sym_count,
            )

    return (np.array(all_feats), np.array(all_labels),
            np.array(all_ts), per_symbol)


def walk_forward_evaluate(X: np.ndarray, y: np.ndarray, n_folds: int) -> dict[str, Any]:
    """TimeSeriesSplit k-fold CV. Returns aggregated per-class + overall metrics."""
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    tscv = TimeSeriesSplit(n_splits=n_folds)
    fold_reports: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="mlogloss", num_class=3,
            objective="multi:softprob",
        )
        clf.fit(X_tr_s, y_tr, verbose=False)
        y_pred = clf.predict(X_te_s)

        p, r, f1, support = precision_recall_fscore_support(
            y_te, y_pred, labels=[0, 1, 2], zero_division=0,
        )
        fold_reports.append({
            "fold": fold_idx,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "per_class": {
                "SELL": {"precision": float(p[0]), "recall": float(r[0]),
                         "f1": float(f1[0]), "support": int(support[0])},
                "HOLD": {"precision": float(p[1]), "recall": float(r[1]),
                         "f1": float(f1[1]), "support": int(support[1])},
                "BUY":  {"precision": float(p[2]), "recall": float(r[2]),
                         "f1": float(f1[2]), "support": int(support[2])},
            },
        })
        logger.info(
            "fold %d/%d train=%d test=%d acc=%.3f "
            "BUY p=%.2f r=%.2f | HOLD p=%.2f r=%.2f | SELL p=%.2f r=%.2f",
            fold_idx, n_folds, len(train_idx), len(test_idx),
            fold_reports[-1]["accuracy"],
            p[2], r[2], p[1], r[1], p[0], r[0],
        )

    def _mean_metric(cls: str, metric: str) -> float:
        return float(np.mean([f["per_class"][cls][metric] for f in fold_reports]))

    return {
        "folds": n_folds,
        "mean_accuracy": float(np.mean([f["accuracy"] for f in fold_reports])),
        "std_accuracy": float(np.std([f["accuracy"] for f in fold_reports])),
        "per_class_mean": {
            cls: {
                "precision": _mean_metric(cls, "precision"),
                "recall":    _mean_metric(cls, "recall"),
                "f1":        _mean_metric(cls, "f1"),
            }
            for cls in ("SELL", "HOLD", "BUY")
        },
        "fold_detail": fold_reports,
    }


def describe_labels(y: np.ndarray) -> dict[str, Any]:
    dist = Counter(y.tolist())
    total = len(y)
    return {
        "total": total,
        "BUY":  {"count": int(dist[2]), "pct": round(100 * dist[2] / total, 2)},
        "HOLD": {"count": int(dist[1]), "pct": round(100 * dist[1] / total, 2)},
        "SELL": {"count": int(dist[0]), "pct": round(100 * dist[0] / total, 2)},
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sym = p.add_mutually_exclusive_group()
    sym.add_argument("--symbols", nargs="+", help="Explicit symbols")
    sym.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                     help=f"Top-N by volume (default {DEFAULT_TOP_N})")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS,
                   help=f"Lookback days (default {DEFAULT_DAYS})")
    p.add_argument("--frequency", type=int, default=5,
                   help="Bar size in minutes (default 5)")
    p.add_argument("--folds", type=int, default=DEFAULT_FOLDS,
                   help=f"Walk-forward CV folds (default {DEFAULT_FOLDS})")
    p.add_argument("--no-walk-forward", action="store_true",
                   help="Skip CV, train once on all data")
    p.add_argument("--dsn", default=os.environ.get("POSTGRES_DSN", DEFAULT_DSN))
    p.add_argument("--report-path", default="ml_training_report.json")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        logger.info("selecting_symbols top_n=%d days=%d", args.top_n, args.days)
        symbols = top_symbols_by_volume(args.dsn, args.top_n, args.days)
    logger.info("symbols_selected n=%d first5=%s", len(symbols), symbols[:5])

    t0 = datetime.now()
    logger.info("collecting_samples symbols=%d days=%d frequency=%dmin",
                len(symbols), args.days, args.frequency)
    X, y, ts, per_sym = collect_samples(symbols, args.days, args.frequency, args.dsn)
    collect_seconds = (datetime.now() - t0).total_seconds()

    if len(X) == 0:
        logger.error("no_samples_collected — aborting")
        return 1

    # Sort by timestamp so time-series CV is meaningful (cross-symbol chronology)
    order = np.argsort(ts)
    X, y, ts = X[order], y[order], ts[order]

    label_dist = describe_labels(y)
    logger.info(
        "samples_collected total=%d BUY=%.1f%% HOLD=%.1f%% SELL=%.1f%% elapsed_s=%.1f",
        label_dist["total"],
        label_dist["BUY"]["pct"], label_dist["HOLD"]["pct"], label_dist["SELL"]["pct"],
        collect_seconds,
    )
    if label_dist["HOLD"]["pct"] > 85:
        logger.warning("HOLD > 85%% — model will likely collapse to always-HOLD")

    cv_summary: dict[str, Any] | None = None
    if not args.no_walk_forward:
        logger.info("walk_forward_eval folds=%d", args.folds)
        cv_summary = walk_forward_evaluate(X, y, args.folds)
        logger.info(
            "cv_summary mean_accuracy=%.3f +/- %.3f | "
            "BUY p=%.2f r=%.2f | SELL p=%.2f r=%.2f",
            cv_summary["mean_accuracy"], cv_summary["std_accuracy"],
            cv_summary["per_class_mean"]["BUY"]["precision"],
            cv_summary["per_class_mean"]["BUY"]["recall"],
            cv_summary["per_class_mean"]["SELL"]["precision"],
            cv_summary["per_class_mean"]["SELL"]["recall"],
        )

    # Final fit on all data — IntegratedMLModel handles atomic save internally
    from ml.models import IntegratedMLModel
    final_model = IntegratedMLModel(brain=None, commentary_system=None)
    logger.info("final_fit samples=%d features=%d", len(X), X.shape[1])
    ok = final_model.train(X, y)
    if not ok:
        logger.error("final_fit_failed")
        return 1

    report = {
        "timestamp": datetime.now().isoformat(),
        "symbols": symbols,
        "symbol_count": len(symbols),
        "per_symbol_samples": per_sym,
        "days": args.days,
        "frequency_minutes": args.frequency,
        "label_distribution": label_dist,
        "collection_seconds": round(collect_seconds, 1),
        "cv_summary": cv_summary,
        "model_path": str(final_model.model_path),
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2, default=str))
    logger.info("report_written path=%s", args.report_path)
    logger.info("training_complete model=%s samples=%d",
                final_model.model_path, len(X))
    return 0


if __name__ == "__main__":
    sys.exit(main())
