"""Analyze v2 model precision/coverage/expected-return at confidence thresholds.

Re-runs Purged K-Fold with ``predict_proba`` and evaluates what happens if
we only trade when the model's BUY (or SELL) probability exceeds a threshold.
This is the cheapest test of whether the v2 model has usable signal at
its high-confidence tail — if precision climbs with threshold, we have an
edge worth deploying behind a confidence gate.

Usage:
    python analyze_v2_thresholds.py
    python analyze_v2_thresholds.py --thresholds 0.50 0.55 0.60 0.65 0.70
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml.labels import compute_uniqueness
from ml.purged_cv import PurgedKFold
from train_ml_model_v2 import collect_v2_samples, DEFAULT_DSN


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("threshold_analysis")

DEFAULT_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-path", default="ml_training_report_v2.json",
                   help="v2 training report to pull config from")
    p.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    p.add_argument("--output", default="ml_threshold_analysis_v2.json")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # --- Load v2 config -------------------------------------------------
    report = json.loads(Path(args.report_path).read_text())
    cfg = report["config"]
    symbols = report["symbols"]
    logger.info("loaded_config symbols=%d pt=%.2f sl=%.2f mh=%d stride=%d",
                len(symbols), cfg["pt_mult"], cfg["sl_mult"],
                cfg["max_holding"], cfg["event_stride"])

    # --- Re-collect samples with identical config -----------------------
    import os
    from dotenv import load_dotenv
    load_dotenv()
    dsn = os.environ.get("POSTGRES_DSN", DEFAULT_DSN)

    X, y, ret, event_times, touch_times, sym_groups, _ = collect_v2_samples(
        symbols=symbols, days=cfg["days"], frequency=cfg["frequency"], dsn=dsn,
        pt_mult=cfg["pt_mult"], sl_mult=cfg["sl_mult"],
        max_holding=cfg["max_holding"], event_stride=cfg["event_stride"],
    )
    order = np.argsort(event_times.to_numpy())
    X, y, ret = X[order], y[order], ret[order]
    event_times = event_times[order]
    touch_times = touch_times[order]
    sym_groups = sym_groups[order]

    # --- Uniqueness weights (same as training) --------------------------
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

    # --- Purged CV, accumulate out-of-sample probabilities --------------
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    cv = PurgedKFold(
        n_splits=cfg["n_folds"], touch_times=touch_series,
        embargo_pct=cfg["embargo_pct"],
    )

    n = len(X)
    oos_proba = np.full((n, 3), np.nan, dtype="float64")
    oos_covered = np.zeros(n, dtype=bool)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="mlogloss", num_class=3,
            objective="multi:softprob", n_jobs=-1,
        )
        clf.fit(X_tr, y[train_idx], sample_weight=weights[train_idx],
                verbose=False)
        oos_proba[test_idx] = clf.predict_proba(X_te)
        oos_covered[test_idx] = True
        logger.info("fold %d fit+predict done", fold_idx)

    # Some samples may have been purged from all folds — drop them.
    mask = oos_covered
    proba = oos_proba[mask]
    y_eval = y[mask]
    ret_eval = ret[mask]
    logger.info("oos_coverage covered=%d total=%d", mask.sum(), n)

    # --- Threshold sweep ------------------------------------------------
    #
    # Interpretation:
    # - p_buy  = proba for class 2 (upper barrier). "Gate a long trade"
    #   when p_buy >= threshold.
    # - p_sell = proba for class 0 (lower barrier).
    # - avg_return is the realised barrier-to-entry return among the
    #   subset of events the gate admits (positive = edge, zero = noise).
    p_sell = proba[:, 0]
    p_buy = proba[:, 2]

    rows = []
    total = len(y_eval)
    for thr in args.thresholds:
        # Long trades gated by p_buy >= thr
        buy_mask = p_buy >= thr
        if buy_mask.any():
            buy_hit = (y_eval[buy_mask] == 2).mean()
            buy_ret = ret_eval[buy_mask].mean()
            buy_cov = buy_mask.mean()
        else:
            buy_hit = buy_ret = buy_cov = 0.0

        # Short trades gated by p_sell >= thr (short earns when price drops)
        sell_mask = p_sell >= thr
        if sell_mask.any():
            sell_hit = (y_eval[sell_mask] == 0).mean()
            sell_ret = -ret_eval[sell_mask].mean()  # negate for short P&L
            sell_cov = sell_mask.mean()
        else:
            sell_hit = sell_ret = sell_cov = 0.0

        rows.append({
            "threshold": thr,
            "buy": {
                "precision": round(float(buy_hit), 4),
                "avg_return": round(float(buy_ret), 5),
                "coverage": round(float(buy_cov), 4),
                "n_signals": int(buy_mask.sum()),
            },
            "sell": {
                "precision": round(float(sell_hit), 4),
                "avg_return": round(float(sell_ret), 5),
                "coverage": round(float(sell_cov), 4),
                "n_signals": int(sell_mask.sum()),
            },
        })
        logger.info(
            "thr=%.2f | BUY prec=%.3f ret=%+.4f cov=%.3f n=%d"
            "  |  SELL prec=%.3f ret=%+.4f cov=%.3f n=%d",
            thr, buy_hit, buy_ret, buy_cov, buy_mask.sum(),
            sell_hit, sell_ret, sell_cov, sell_mask.sum(),
        )

    out = {
        "model_config": cfg,
        "oos_coverage_pct": round(100 * mask.mean(), 2),
        "n_samples_evaluated": int(mask.sum()),
        "thresholds": rows,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    logger.info("written %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
