"""Train binary meta-model on top of mean-reversion primary signals.

Follows López de Prado's meta-labeling workflow (AFML §3.6-3.7):

1. Replay the primary strategy (mean-reversion) over historical data to
   get (timestamp, side) pairs for every historical entry.
2. Side-aware triple-barrier label each event: 1 if the primary's bet
   paid (upper hit for long, lower hit for short), 0 otherwise.
3. Extract features at each event timestamp.
4. Train a binary XGBoost classifier with uniqueness sample weights
   and purged K-Fold cross-validation.

Produces:
    ml_meta_model.pkl             — trained classifier bundle
    ml_meta_training_report.json  — metrics + feature importances

Usage:
    python train_meta_model.py
    python train_meta_model.py --top-n 20 --days 60
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

from ml.labels import compute_uniqueness
from ml.meta_labels import apply_meta_triple_barrier
from ml.purged_cv import PurgedKFold
from ml.strategy_replay import replay_mean_reversion
from train_ml_model import DEFAULT_DSN, top_symbols_by_volume
from train_ml_model_v2 import _compute_raw_atr


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_meta")

DEFAULT_PT_MULT = 1.5
DEFAULT_SL_MULT = 1.5
DEFAULT_MAX_HOLDING = 60
DEFAULT_EMBARGO_PCT = 0.01
DEFAULT_FOLDS = 5
DEFAULT_TOP_N = 150
DEFAULT_DAYS = 180
DEFAULT_FREQUENCY = 5
DEFAULT_RSI_THRESHOLD = 30.0

MODEL_PATH = Path("ml_meta_model.pkl")
REPORT_PATH = Path("ml_meta_training_report.json")


def collect_meta_samples(
    symbols: list[str],
    days: int,
    frequency: int,
    dsn: str,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
    rsi_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex,
           pd.DatetimeIndex, np.ndarray, np.ndarray, dict[str, int]]:
    """Replay mean-reversion + meta-label + extract features.

    Returns
    -------
    X              — feature matrix (N, D)
    y              — binary labels (N,) in {0, 1}
    ret            — strategy-perspective return (N,)
    event_times    — entry timestamps
    touch_times    — barrier-resolution timestamps
    sym_groups     — symbol per sample (for per-symbol uniqueness)
    sides          — +1 long, -1 short
    per_symbol     — count of signals per symbol
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
    side_blocks: list[np.ndarray] = []
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
            continue

        events = replay_mean_reversion(df, rsi_threshold=rsi_threshold)
        if events.empty:
            per_symbol[symbol] = 0
            continue

        # Restrict events to bars with at least ``max_holding`` forward bars
        # so meta-labeling has a full outcome window.
        valid_mask = events.index <= df.index[-1 - max_holding]
        events = events.loc[valid_mask]
        if events.empty:
            per_symbol[symbol] = 0
            continue

        atr = _compute_raw_atr(df, window=14)
        # Drop events with non-positive ATR (degenerate barriers).
        atr_at_events = atr.reindex(events.index)
        events = events.loc[atr_at_events > 0]
        if events.empty:
            per_symbol[symbol] = 0
            continue

        labels = apply_meta_triple_barrier(
            prices=df["Close"], events=events, atr=atr,
            pt_mult=pt_mult, sl_mult=sl_mult, max_holding=max_holding,
        )

        feat_df = feature_model.feature_extractor.batch_extract(df)
        feat_at_events = feat_df.loc[labels.index, feature_model.feature_names]
        mask = feat_at_events.notna().all(axis=1)
        labels = labels.loc[mask]
        feat_at_events = feat_at_events.loc[mask]
        events_aligned = events.loc[labels.index]
        if labels.empty:
            per_symbol[symbol] = 0
            continue

        X_blocks.append(feat_at_events.to_numpy())
        y_blocks.append(labels["bin"].to_numpy().astype(np.int64))
        ret_blocks.append(labels["ret"].to_numpy())
        event_blocks.append(labels.index)
        touch_blocks.append(pd.DatetimeIndex(labels["touch_time"].to_numpy()))
        group_blocks.append(np.full(len(labels), symbol, dtype=object))
        side_blocks.append(events_aligned["side"].to_numpy().astype(np.int8))
        per_symbol[symbol] = len(labels)

        if (i + 1) % 10 == 0 or i + 1 == len(symbols):
            logger.info(
                "collection_progress done=%d/%d cumulative=%d latest=%s/%d",
                i + 1, len(symbols), sum(len(b) for b in X_blocks),
                symbol, len(labels),
            )

    if not X_blocks:
        empty_idx = pd.DatetimeIndex([])
        return (np.array([]), np.array([]), np.array([]), empty_idx,
                empty_idx, np.array([]), np.array([]), per_symbol)

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    ret = np.concatenate(ret_blocks)
    event_times = pd.DatetimeIndex(np.concatenate([b.to_numpy() for b in event_blocks]))
    touch_times = pd.DatetimeIndex(np.concatenate([b.to_numpy() for b in touch_blocks]))
    sym_groups = np.concatenate(group_blocks)
    sides = np.concatenate(side_blocks)
    return X, y, ret, event_times, touch_times, sym_groups, sides, per_symbol


def purged_cv_binary(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    touch_times: pd.Series,
    n_splits: int,
    embargo_pct: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run purged K-Fold CV for binary classification. Returns (summary, oos_proba)."""
    from sklearn.metrics import (
        precision_recall_fscore_support, accuracy_score, matthews_corrcoef,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    cv = PurgedKFold(n_splits=n_splits, touch_times=touch_times, embargo_pct=embargo_pct)

    fold_reports: list[dict[str, Any]] = []
    oos_proba = np.full(len(X), np.nan, dtype="float64")

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss",
            objective="binary:logistic", n_jobs=-1,
        )
        clf.fit(X_tr, y[train_idx], sample_weight=weights[train_idx],
                verbose=False)

        y_pred = clf.predict(X_te)
        proba = clf.predict_proba(X_te)[:, 1]
        oos_proba[test_idx] = proba

        p, r, f1, support = precision_recall_fscore_support(
            y[test_idx], y_pred, labels=[0, 1], zero_division=0,
        )
        auc = float(roc_auc_score(y[test_idx], proba)) if len(np.unique(y[test_idx])) == 2 else 0.0

        fold_reports.append({
            "fold": fold_idx,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "accuracy": float(accuracy_score(y[test_idx], y_pred)),
            "mcc": float(matthews_corrcoef(y[test_idx], y_pred)),
            "auc": auc,
            "lose": {"precision": float(p[0]), "recall": float(r[0]),
                     "f1": float(f1[0]), "support": int(support[0])},
            "win":  {"precision": float(p[1]), "recall": float(r[1]),
                     "f1": float(f1[1]), "support": int(support[1])},
        })
        logger.info(
            "fold %d/%d train=%d test=%d acc=%.3f mcc=%.3f auc=%.3f "
            "WIN p=%.2f r=%.2f",
            fold_idx, n_splits, len(train_idx), len(test_idx),
            fold_reports[-1]["accuracy"], fold_reports[-1]["mcc"], auc,
            p[1], r[1],
        )

    def _mean(key: str) -> float:
        vals = [f[key] for f in fold_reports]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "n_folds_executed": len(fold_reports),
        "mean_accuracy": _mean("accuracy"),
        "mean_mcc": _mean("mcc"),
        "mean_auc": _mean("auc"),
        "mean_win_precision": float(np.mean([f["win"]["precision"] for f in fold_reports])) if fold_reports else 0.0,
        "mean_win_recall": float(np.mean([f["win"]["recall"] for f in fold_reports])) if fold_reports else 0.0,
        "fold_detail": fold_reports,
    }
    return summary, oos_proba


def threshold_sweep(
    proba: np.ndarray,
    y: np.ndarray,
    ret: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    rows = []
    covered = ~np.isnan(proba)
    p = proba[covered]
    y_ = y[covered]
    r_ = ret[covered]
    for thr in thresholds:
        gate = p >= thr
        if not gate.any():
            rows.append({"threshold": thr, "n": 0, "precision": 0.0,
                         "avg_return": 0.0, "coverage": 0.0})
            continue
        rows.append({
            "threshold": thr,
            "n": int(gate.sum()),
            "precision": round(float((y_[gate] == 1).mean()), 4),
            "avg_return": round(float(r_[gate].mean()), 5),
            "coverage": round(float(gate.mean()), 4),
        })
    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sym = p.add_mutually_exclusive_group()
    sym.add_argument("--symbols", nargs="+")
    sym.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--days", type=int, default=DEFAULT_DAYS)
    p.add_argument("--frequency", type=int, default=DEFAULT_FREQUENCY)
    p.add_argument("--pt-mult", type=float, default=DEFAULT_PT_MULT)
    p.add_argument("--sl-mult", type=float, default=DEFAULT_SL_MULT)
    p.add_argument("--max-holding", type=int, default=DEFAULT_MAX_HOLDING)
    p.add_argument("--rsi-threshold", type=float, default=DEFAULT_RSI_THRESHOLD)
    p.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p.add_argument("--embargo-pct", type=float, default=DEFAULT_EMBARGO_PCT)
    p.add_argument("--dsn", default=os.environ.get("POSTGRES_DSN", DEFAULT_DSN))
    p.add_argument("--model-path", default=str(MODEL_PATH))
    p.add_argument("--report-path", default=str(REPORT_PATH))
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
    logger.info(
        "collecting_meta_samples symbols=%d days=%d freq=%dmin rsi_thr=%.1f pt=%.2f sl=%.2f mh=%d",
        len(symbols), args.days, args.frequency, args.rsi_threshold,
        args.pt_mult, args.sl_mult, args.max_holding,
    )
    (X, y, ret, event_times, touch_times, sym_groups, sides,
     per_sym) = collect_meta_samples(
        symbols=symbols, days=args.days, frequency=args.frequency, dsn=args.dsn,
        pt_mult=args.pt_mult, sl_mult=args.sl_mult,
        max_holding=args.max_holding, rsi_threshold=args.rsi_threshold,
    )
    collect_seconds = (datetime.now() - t0).total_seconds()

    if len(X) == 0:
        logger.error("no_meta_samples — strategy may never have fired on this universe")
        return 1

    order = np.argsort(event_times.to_numpy())
    X = X[order]; y = y[order]; ret = ret[order]
    event_times = event_times[order]; touch_times = touch_times[order]
    sym_groups = sym_groups[order]; sides = sides[order]

    dist = Counter(y.tolist())
    win_rate = dist[1] / len(y)
    logger.info(
        "samples_collected total=%d wins=%d (%.1f%%) losses=%d elapsed_s=%.1f",
        len(y), dist[1], 100 * win_rate, dist[0], collect_seconds,
    )

    # Uniqueness (per-symbol).
    t1 = datetime.now()
    touch_series = pd.Series(touch_times.to_numpy(), index=event_times)
    labels_df = pd.DataFrame({"touch_time": touch_series.values},
                             index=touch_series.index)
    master_idx = pd.DatetimeIndex(
        np.unique(np.concatenate([event_times.to_numpy(), touch_times.to_numpy()]))
    ).sort_values()
    weights = compute_uniqueness(
        labels_df, master_idx, groups=pd.Series(sym_groups),
    ).to_numpy()
    weights = np.clip(weights, 1e-6, 1.0)
    logger.info(
        "uniqueness_done mean=%.3f effective_n=%.0f elapsed_s=%.1f",
        float(weights.mean()), float(weights.sum()),
        (datetime.now() - t1).total_seconds(),
    )

    # Purged CV.
    t2 = datetime.now()
    cv_summary, oos_proba = purged_cv_binary(
        X=X, y=y, weights=weights, touch_times=touch_series,
        n_splits=args.folds, embargo_pct=args.embargo_pct,
    )
    logger.info(
        "cv_done acc=%.3f mcc=%.3f auc=%.3f win_prec=%.3f elapsed_s=%.1f",
        cv_summary["mean_accuracy"], cv_summary["mean_mcc"], cv_summary["mean_auc"],
        cv_summary["mean_win_precision"],
        (datetime.now() - t2).total_seconds(),
    )

    # Threshold sweep on OOS probabilities — this is the deployable gate.
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    sweep = threshold_sweep(oos_proba, y, ret, thresholds)
    for row in sweep:
        logger.info(
            "thr=%.2f n=%d win_prec=%.3f avg_ret=%+.4f cov=%.3f",
            row["threshold"], row["n"], row["precision"],
            row["avg_return"], row["coverage"],
        )

    # Final fit on all samples.
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    final_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="logloss", objective="binary:logistic", n_jobs=-1,
    )
    final_clf.fit(X_s, y, sample_weight=weights, verbose=False)

    from ml.models import IntegratedMLModel
    feature_names = IntegratedMLModel(brain=None, commentary_system=None).feature_names

    bundle = {
        "version": "meta-v1",
        "primary_strategy": "mean_reversion",
        "classifier": final_clf,
        "scaler": scaler,
        "feature_names": feature_names,
        "config": {
            "pt_mult": args.pt_mult, "sl_mult": args.sl_mult,
            "max_holding": args.max_holding, "rsi_threshold": args.rsi_threshold,
            "frequency": args.frequency, "embargo_pct": args.embargo_pct,
            "n_folds": args.folds, "days": args.days,
        },
        "trained_at": datetime.now().isoformat(),
    }
    joblib.dump(bundle, args.model_path)

    importances = dict(sorted(
        zip(feature_names, final_clf.feature_importances_.tolist()),
        key=lambda kv: kv[1], reverse=True,
    ))

    report = {
        "timestamp": datetime.now().isoformat(),
        "model_path": args.model_path,
        "primary_strategy": "mean_reversion",
        "config": bundle["config"],
        "symbols": symbols,
        "symbol_count": len(symbols),
        "per_symbol_samples": per_sym,
        "signal_counts": {
            "total": int(len(y)),
            "long": int((sides == 1).sum()),
            "short": int((sides == -1).sum()),
            "wins": int(dist[1]),
            "losses": int(dist[0]),
            "base_win_rate": round(win_rate, 4),
        },
        "weighting": {
            "mean": float(weights.mean()),
            "effective_sample_size": float(weights.sum()),
            "compression_ratio": round(float(weights.sum()) / len(weights), 4),
        },
        "cv_summary": cv_summary,
        "threshold_sweep": sweep,
        "feature_importance": importances,
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2, default=str))
    logger.info("report_written path=%s", args.report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
