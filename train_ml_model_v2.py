"""Train v2 ML model with triple-barrier labels + purged CV + uniqueness weights.

López de Prado methodology (AFML ch. 3, 4, 7). Keeps v1 artifacts untouched.

Produces:
    ml_model_v2.pkl            — trained XGBoost classifier bundle (joblib)
    ml_training_report_v2.json — honest walk-forward metrics

Usage:
    python train_ml_model_v2.py                      # defaults
    python train_ml_model_v2.py --top-n 30 --days 60 # small/quick run
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

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ml.labels import apply_triple_barrier, compute_uniqueness
from ml.purged_cv import PurgedKFold
from train_ml_model import DEFAULT_DSN, top_symbols_by_volume  # reuse v1 universe logic


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_ml_v2")

# Barrier configuration — rationale in the CLI help text below.
DEFAULT_PT_MULT = 2.0
DEFAULT_SL_MULT = 1.0
DEFAULT_MAX_HOLDING = 60
DEFAULT_EVENT_STRIDE = 5
DEFAULT_EMBARGO_PCT = 0.01
DEFAULT_FOLDS = 5
DEFAULT_TOP_N = 150
DEFAULT_DAYS = 180
DEFAULT_FREQUENCY = 5

MODEL_PATH = Path("ml_model_v2.pkl")
REPORT_PATH = Path("ml_training_report_v2.json")


def _compute_raw_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """True-Range-based ATR in price units (not the ratio)."""
    import ta  # heavy optional import
    return ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=window, fillna=False,
    )


def label_symbol(
    df: pd.DataFrame,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
    event_stride: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (labels_df, atr_series) for one symbol.

    Events are placed every ``event_stride`` bars starting after a 50-bar
    feature warmup. Bars with non-positive ATR are excluded (flat series).
    """
    atr = _compute_raw_atr(df, window=14)
    prices = df["Close"]

    first = 50
    last = len(df) - max_holding - 1
    if last <= first:
        return pd.DataFrame(), atr
    candidate_times = prices.index[first:last:event_stride]

    # Drop events with degenerate ATR (zero-width barriers would fail).
    atr_at_events = atr.reindex(candidate_times)
    good_events = candidate_times[atr_at_events > 0]
    if len(good_events) == 0:
        return pd.DataFrame(), atr

    labels = apply_triple_barrier(
        prices=prices,
        events=good_events,
        atr=atr,
        pt_mult=pt_mult,
        sl_mult=sl_mult,
        max_holding=max_holding,
    )
    return labels, atr


def collect_v2_samples(
    symbols: list[str],
    days: int,
    frequency: int,
    dsn: str,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
    event_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index, pd.Index, np.ndarray, dict[str, int]]:
    """Return (X, y, ret, event_times, touch_times, symbol_groups, per_symbol_counts).

    The ``symbol_groups`` array is one entry per sample, naming the source
    symbol — used downstream so uniqueness weights are computed within each
    symbol's timeline rather than pooling all symbols into a single ticker-tape.
    """
    from data_providers.postgres import PostgresDataProvider
    from ml.models import IntegratedMLModel

    provider = PostgresDataProvider(dsn)
    feature_model = IntegratedMLModel(brain=None, commentary_system=None)

    period_type = "day" if days < 30 else "month"
    period = days if period_type == "day" else max(1, days // 30)

    X_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    ret_blocks: list[np.ndarray] = []
    event_blocks: list[pd.Index] = []
    touch_blocks: list[pd.Index] = []
    group_blocks: list[np.ndarray] = []
    per_symbol: dict[str, int] = {}

    for i, symbol in enumerate(symbols):
        try:
            df = provider.get_market_data(
                symbol, period_type=period_type, period=period,
                frequency_type="minute", frequency=frequency,
            )
        except Exception as exc:
            logger.error("load_failed symbol=%s err=%s", symbol, exc)
            continue

        if df.empty or len(df) < max_holding + 100:
            logger.warning("insufficient_data symbol=%s bars=%d", symbol, len(df))
            continue

        feat_df = feature_model.feature_extractor.batch_extract(df)
        if feat_df.empty:
            logger.warning("no_features symbol=%s", symbol)
            continue

        labels_df, _ = label_symbol(
            df, pt_mult=pt_mult, sl_mult=sl_mult,
            max_holding=max_holding, event_stride=event_stride,
        )
        if labels_df.empty:
            continue

        feat_at_events = feat_df.loc[labels_df.index, feature_model.feature_names]
        mask = feat_at_events.notna().all(axis=1)
        if not mask.all():
            labels_df = labels_df.loc[mask]
            feat_at_events = feat_at_events.loc[mask]

        X_blocks.append(feat_at_events.to_numpy())
        # Label map: {-1,0,+1} → {0,1,2} = {SELL, HOLD, BUY} for XGBoost.
        y_blocks.append((labels_df["label"].to_numpy() + 1).astype(np.int64))
        ret_blocks.append(labels_df["ret"].to_numpy())
        event_blocks.append(labels_df.index)
        touch_blocks.append(pd.DatetimeIndex(labels_df["touch_time"].to_numpy()))
        group_blocks.append(np.full(len(labels_df), symbol, dtype=object))
        per_symbol[symbol] = len(labels_df)

        if (i + 1) % 10 == 0 or i + 1 == len(symbols):
            logger.info(
                "collection_progress done=%d/%d cumulative_samples=%d latest=%s/%d",
                i + 1, len(symbols), sum(len(b) for b in X_blocks),
                symbol, len(labels_df),
            )

    if not X_blocks:
        empty_idx = pd.DatetimeIndex([])
        return np.array([]), np.array([]), np.array([]), empty_idx, empty_idx, np.array([]), {}

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    ret = np.concatenate(ret_blocks)
    event_times = pd.DatetimeIndex(np.concatenate([b.to_numpy() for b in event_blocks]))
    touch_times = pd.DatetimeIndex(np.concatenate([b.to_numpy() for b in touch_blocks]))
    sym_groups = np.concatenate(group_blocks)
    return X, y, ret, event_times, touch_times, sym_groups, per_symbol


def describe_labels(y: np.ndarray) -> dict[str, Any]:
    dist = Counter(y.tolist())
    total = int(len(y))
    return {
        "total": total,
        "BUY":  {"count": int(dist[2]), "pct": round(100 * dist[2] / total, 2)},
        "HOLD": {"count": int(dist[1]), "pct": round(100 * dist[1] / total, 2)},
        "SELL": {"count": int(dist[0]), "pct": round(100 * dist[0] / total, 2)},
    }


def purged_cv_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    touch_times: pd.Series,
    n_splits: int,
    embargo_pct: float,
) -> dict[str, Any]:
    from sklearn.metrics import (
        precision_recall_fscore_support,
        accuracy_score,
        matthews_corrcoef,
    )
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    cv = PurgedKFold(n_splits=n_splits, touch_times=touch_times, embargo_pct=embargo_pct)

    fold_reports: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            logger.warning("fold %d empty (train=%d test=%d) — skipping",
                           fold_idx, len(train_idx), len(test_idx))
            continue

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        w_tr = weights[train_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="mlogloss", num_class=3,
            objective="multi:softprob", n_jobs=-1,
        )
        clf.fit(X_tr_s, y_tr, sample_weight=w_tr, verbose=False)
        y_pred = clf.predict(X_te_s)

        p, r, f1, support = precision_recall_fscore_support(
            y_te, y_pred, labels=[0, 1, 2], zero_division=0,
        )
        fold_reports.append({
            "fold": fold_idx,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "mcc": float(matthews_corrcoef(y_te, y_pred)),
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
            "fold %d/%d train=%d test=%d acc=%.3f mcc=%.3f "
            "BUY p=%.2f r=%.2f | SELL p=%.2f r=%.2f",
            fold_idx, n_splits, len(train_idx), len(test_idx),
            fold_reports[-1]["accuracy"], fold_reports[-1]["mcc"],
            p[2], r[2], p[0], r[0],
        )

    def _mean(metric_path: list[str]) -> float:
        vals = []
        for f in fold_reports:
            ref = f
            for k in metric_path:
                ref = ref[k]
            vals.append(ref)
        return float(np.mean(vals)) if vals else 0.0

    return {
        "n_folds_executed": len(fold_reports),
        "mean_accuracy": _mean(["accuracy"]),
        "std_accuracy": float(np.std([f["accuracy"] for f in fold_reports])) if fold_reports else 0.0,
        "mean_mcc": _mean(["mcc"]),
        "per_class_mean": {
            cls: {
                "precision": _mean(["per_class", cls, "precision"]),
                "recall":    _mean(["per_class", cls, "recall"]),
                "f1":        _mean(["per_class", cls, "f1"]),
            }
            for cls in ("SELL", "HOLD", "BUY")
        },
        "fold_detail": fold_reports,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sym = p.add_mutually_exclusive_group()
    sym.add_argument("--symbols", nargs="+", help="Explicit symbols (overrides --top-n)")
    sym.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                     help=f"Top-N by volume (default {DEFAULT_TOP_N})")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS)
    p.add_argument("--frequency", type=int, default=DEFAULT_FREQUENCY,
                   help="Bar size in minutes (default 5)")
    p.add_argument("--pt-mult", type=float, default=DEFAULT_PT_MULT,
                   help=f"Profit-target ATR multiple (default {DEFAULT_PT_MULT})")
    p.add_argument("--sl-mult", type=float, default=DEFAULT_SL_MULT,
                   help=f"Stop-loss ATR multiple (default {DEFAULT_SL_MULT})")
    p.add_argument("--max-holding", type=int, default=DEFAULT_MAX_HOLDING,
                   help=f"Time barrier in bars (default {DEFAULT_MAX_HOLDING})")
    p.add_argument("--event-stride", type=int, default=DEFAULT_EVENT_STRIDE,
                   help=f"Bars between events (default {DEFAULT_EVENT_STRIDE})")
    p.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p.add_argument("--embargo-pct", type=float, default=DEFAULT_EMBARGO_PCT)
    p.add_argument("--dsn", default=os.environ.get("POSTGRES_DSN", DEFAULT_DSN))
    p.add_argument("--model-path", default=str(MODEL_PATH))
    p.add_argument("--report-path", default=str(REPORT_PATH))
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # --- Symbol universe ------------------------------------------------
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        logger.info("selecting_symbols top_n=%d days=%d", args.top_n, args.days)
        symbols = top_symbols_by_volume(args.dsn, args.top_n, args.days)
    logger.info("symbols_selected n=%d first5=%s", len(symbols), symbols[:5])

    # --- Collection + labeling -----------------------------------------
    t0 = datetime.now()
    logger.info(
        "collecting_v2_samples symbols=%d days=%d freq=%dmin pt=%.2f sl=%.2f mh=%d stride=%d",
        len(symbols), args.days, args.frequency,
        args.pt_mult, args.sl_mult, args.max_holding, args.event_stride,
    )
    X, y, ret, event_times, touch_times, sym_groups, per_sym = collect_v2_samples(
        symbols=symbols, days=args.days, frequency=args.frequency, dsn=args.dsn,
        pt_mult=args.pt_mult, sl_mult=args.sl_mult,
        max_holding=args.max_holding, event_stride=args.event_stride,
    )
    collect_seconds = (datetime.now() - t0).total_seconds()

    if len(X) == 0:
        logger.error("no_samples_collected — aborting")
        return 1

    # Sort by event time — required for purged CV to carve chronological folds.
    order = np.argsort(event_times.to_numpy())
    X = X[order]
    y = y[order]
    ret = ret[order]
    event_times = event_times[order]
    touch_times = touch_times[order]
    sym_groups = sym_groups[order]

    label_dist = describe_labels(y)
    logger.info(
        "samples_collected total=%d BUY=%.1f%% HOLD=%.1f%% SELL=%.1f%% elapsed_s=%.1f",
        label_dist["total"],
        label_dist["BUY"]["pct"], label_dist["HOLD"]["pct"], label_dist["SELL"]["pct"],
        collect_seconds,
    )

    # --- Uniqueness weights --------------------------------------------
    # Grouped by symbol: an AAPL event at 10:00 AM is independent of an
    # NVDA event at 10:00 AM, so they must not depress each other's weight.
    logger.info("computing_uniqueness n=%d groups=%d",
                len(event_times), len(np.unique(sym_groups)))
    t1 = datetime.now()
    touch_series = pd.Series(touch_times.to_numpy(), index=event_times)
    combined_index = pd.DatetimeIndex(
        np.unique(np.concatenate([event_times.to_numpy(), touch_times.to_numpy()]))
    ).sort_values()
    labels_for_weights = pd.DataFrame(
        {"touch_time": touch_series.values}, index=touch_series.index,
    )
    weights = compute_uniqueness(
        labels_for_weights, combined_index, groups=pd.Series(sym_groups),
    ).to_numpy()
    weights = np.clip(weights, 1e-6, 1.0)
    uniqueness_seconds = (datetime.now() - t1).total_seconds()
    effective_n = float(weights.sum())
    logger.info(
        "uniqueness_done mean=%.3f min=%.3f effective_n=%.0f elapsed_s=%.1f",
        float(weights.mean()), float(weights.min()), effective_n, uniqueness_seconds,
    )

    # --- Purged CV evaluation ------------------------------------------
    t2 = datetime.now()
    cv_summary = purged_cv_evaluate(
        X=X, y=y, weights=weights, touch_times=touch_series,
        n_splits=args.folds, embargo_pct=args.embargo_pct,
    )
    cv_seconds = (datetime.now() - t2).total_seconds()
    logger.info(
        "cv_done mean_acc=%.3f +/- %.3f mean_mcc=%.3f elapsed_s=%.1f",
        cv_summary["mean_accuracy"], cv_summary["std_accuracy"],
        cv_summary["mean_mcc"], cv_seconds,
    )

    # --- Final fit on all data -----------------------------------------
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    t3 = datetime.now()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    final_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="mlogloss", num_class=3,
        objective="multi:softprob", n_jobs=-1,
    )
    final_clf.fit(X_s, y, sample_weight=weights, verbose=False)
    final_seconds = (datetime.now() - t3).total_seconds()

    from ml.models import IntegratedMLModel
    feature_names = IntegratedMLModel(brain=None, commentary_system=None).feature_names

    bundle = {
        "version": 2,
        "classifier": final_clf,
        "scaler": scaler,
        "feature_names": feature_names,
        "label_map": {0: "SELL", 1: "HOLD", 2: "BUY"},
        "config": {
            "pt_mult": args.pt_mult,
            "sl_mult": args.sl_mult,
            "max_holding": args.max_holding,
            "event_stride": args.event_stride,
            "frequency": args.frequency,
            "embargo_pct": args.embargo_pct,
            "n_folds": args.folds,
            "n_symbols": len(symbols),
            "days": args.days,
        },
        "trained_at": datetime.now().isoformat(),
    }
    joblib.dump(bundle, args.model_path)
    logger.info("model_written path=%s elapsed_final_fit_s=%.1f",
                args.model_path, final_seconds)

    # --- Feature importance --------------------------------------------
    importances = dict(
        sorted(
            zip(feature_names, final_clf.feature_importances_.tolist()),
            key=lambda kv: kv[1], reverse=True,
        )
    )

    # --- Report ---------------------------------------------------------
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_path": args.model_path,
        "config": bundle["config"],
        "symbols": symbols,
        "symbol_count": len(symbols),
        "per_symbol_samples": per_sym,
        "label_distribution": label_dist,
        "weighting": {
            "mean": float(weights.mean()),
            "median": float(np.median(weights)),
            "min": float(weights.min()),
            "max": float(weights.max()),
            "effective_sample_size": effective_n,
            "compression_ratio": round(effective_n / len(weights), 4),
        },
        "timing_seconds": {
            "collection": round(collect_seconds, 1),
            "uniqueness": round(uniqueness_seconds, 1),
            "cv": round(cv_seconds, 1),
            "final_fit": round(final_seconds, 1),
        },
        "cv_summary": cv_summary,
        "feature_importance": importances,
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2, default=str))
    logger.info("report_written path=%s", args.report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
