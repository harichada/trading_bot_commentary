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

from ml.costs import CostModel, cost_in_r as _cost_in_r, ret_to_r_multiple
from ml.evaluation import FoldMetrics, grade_fold, sortino
from ml.labels import apply_triple_barrier, compute_uniqueness
from ml.pnl_objective import compute_sample_weights
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

# Cost-aware defaults match PROMPT_PACK P2 spec.
DEFAULT_FEE_BPS = 5.0
DEFAULT_SLIP_BPS = 10.0
DEFAULT_IMPACT_COEF = 0.1
DEFAULT_PARTICIPATION = 0.01  # 1% of bar dollar volume — typical retail fill
DEFAULT_REJECT_COST_DRAG_PCT = 50.0
DEFAULT_DECISION_THRESHOLD = 0.55

MODEL_PATH = Path("ml_model_v2.pkl")
REPORT_PATH = Path("ml_training_report_v2.json")
CONFIG_YAML = Path("pro_trading_config.yaml")


class CostDragRejection(RuntimeError):
    """Raised when median OOS cost-drag exceeds the reject threshold."""


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


def _fold_net_r_multiples(
    *,
    y_pred: np.ndarray,
    sample_r_multiples: np.ndarray,
    sample_cost_r: np.ndarray,
    decision_threshold: float,
    probas: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (gross_r, net_r) for trades the model chose to take.

    HOLD (class 1) is no-trade → excluded. BUY (2) and SELL (0) translate into
    long or short R-multiples: a short trade profits on negative returns, so
    its realized R-multiple sign is flipped.

    The decision threshold filters low-conviction predictions; if no class
    crosses it, the trade is skipped (HOLD by default).
    """
    if probas is not None:
        max_proba = probas.max(axis=1)
        take = max_proba >= decision_threshold
    else:
        take = np.ones_like(y_pred, dtype=bool)

    trade_mask = take & (y_pred != 1)  # 1 = HOLD
    if not trade_mask.any():
        return np.array([], dtype="float64"), np.array([], dtype="float64")

    side = np.where(y_pred[trade_mask] == 2, 1.0, -1.0)  # 2=BUY long, 0=SELL short
    gross_r = side * sample_r_multiples[trade_mask]
    net_r = gross_r - sample_cost_r[trade_mask]
    return gross_r, net_r


def purged_cv_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    touch_times: pd.Series,
    n_splits: int,
    embargo_pct: float,
    *,
    sample_r_multiples: np.ndarray,
    sample_cost_r: np.ndarray,
    bars_per_year: int,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> dict[str, Any]:
    """Cost-aware purged-CV grading.

    Replaces accuracy/MCC/classification-report with the six PnL-aware metrics
    required by PROMPT_PACK P2 §2. Per-fold ``FoldMetrics`` are collected;
    aggregation uses **median** across folds (López de Prado AFML §12.3 — the
    mean is dominated by single-fold luck when fold variance is high).
    """
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    cv = PurgedKFold(n_splits=n_splits, touch_times=touch_times, embargo_pct=embargo_pct)

    fold_reports: list[dict[str, Any]] = []
    fold_metrics_objs: list[FoldMetrics] = []

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

        try:
            probas = clf.predict_proba(X_te_s)
        except Exception:  # pragma: no cover — belt-and-braces for exotic sklearn versions
            probas = None
        y_pred = clf.predict(X_te_s)

        gross_r, net_r = _fold_net_r_multiples(
            y_pred=y_pred,
            sample_r_multiples=sample_r_multiples[test_idx],
            sample_cost_r=sample_cost_r[test_idx],
            decision_threshold=decision_threshold,
            probas=probas,
        )

        metrics = grade_fold(gross_r=gross_r, net_r=net_r, bars_per_year=bars_per_year)
        fold_metrics_objs.append(metrics)

        sharpe_proxy = sortino(net_r, bars_per_year=bars_per_year)
        fold_reports.append({
            "fold": fold_idx,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "n_trades_taken": metrics.n_trades,
            "metrics": {
                "expectancy_r": metrics.expectancy_r,
                "profit_factor": _finite_or_none(metrics.profit_factor),
                "sortino": _finite_or_none(metrics.sortino),
                "calmar": _finite_or_none(metrics.calmar),
                "max_adverse_excursion": metrics.max_adverse_excursion,
                "cost_drag_pct": _finite_or_none(metrics.cost_drag_pct),
            },
            "sharpe_proxy": _finite_or_none(sharpe_proxy),
        })
        logger.info(
            "fold %d/%d trades=%d expectancy=%.3fR PF=%.2f sortino=%.2f cost_drag=%.1f%%",
            fold_idx, n_splits, metrics.n_trades, metrics.expectancy_r,
            metrics.profit_factor if np.isfinite(metrics.profit_factor) else float("nan"),
            metrics.sortino if np.isfinite(metrics.sortino) else float("nan"),
            metrics.cost_drag_pct if np.isfinite(metrics.cost_drag_pct) else float("nan"),
        )

    aggregate = aggregate_fold_metrics(fold_metrics_objs)

    return {
        "n_folds_executed": len(fold_reports),
        "aggregate": aggregate,
        "fold_detail": fold_reports,
    }


def _finite_or_none(x: float) -> float | None:
    """JSON-serializable: turn ±inf / NaN into None so jq doesn't choke."""
    return float(x) if np.isfinite(x) else None


def aggregate_fold_metrics(folds: list[FoldMetrics]) -> dict[str, Any]:
    """Median-based aggregation (AFML §12.3).

    Also computes the *variance* of a Sortino-based Sharpe-proxy across folds,
    which the PROMPT_PACK acceptance criterion caps at 0.4.
    """
    if not folds:
        return {
            "median_expectancy_r": 0.0,
            "median_profit_factor": 0.0,
            "median_sortino": 0.0,
            "median_calmar": 0.0,
            "median_max_adverse_excursion": 0.0,
            "median_cost_drag_pct": float("inf"),
            "var_sharpe_across_folds": 0.0,
            "n_folds_executed": 0,
        }

    # Drop non-finite values before taking medians — keeps inf/nan from poisoning stats.
    def _finite(values: list[float]) -> np.ndarray:
        arr = np.asarray(values, dtype="float64")
        return arr[np.isfinite(arr)]

    def _median_or_inf(values: list[float]) -> float:
        arr = _finite(values)
        return float(np.median(arr)) if arr.size else float("inf")

    def _median_or_zero(values: list[float]) -> float:
        arr = _finite(values)
        return float(np.median(arr)) if arr.size else 0.0

    sortino_vals = _finite([f.sortino for f in folds])
    var_sharpe = float(np.var(sortino_vals, ddof=0)) if sortino_vals.size else 0.0

    return {
        "median_expectancy_r": _median_or_zero([f.expectancy_r for f in folds]),
        "median_profit_factor": _median_or_zero([f.profit_factor for f in folds]),
        "median_sortino": _median_or_zero([f.sortino for f in folds]),
        "median_calmar": _median_or_zero([f.calmar for f in folds]),
        "median_max_adverse_excursion": _median_or_zero(
            [f.max_adverse_excursion for f in folds]
        ),
        "median_cost_drag_pct": _median_or_inf([f.cost_drag_pct for f in folds]),
        "var_sharpe_across_folds": var_sharpe,
        "n_folds_executed": len(folds),
    }


def _r_multiples_from_labels(
    *,
    y: np.ndarray,
    ret: np.ndarray,
    pt_mult: float,
    sl_mult: float,
) -> np.ndarray:
    """R-multiples inferred from triple-barrier labels.

    Triple-barrier labels carry the information needed to reconstruct R:
      label +1 → upper barrier hit → gross R = pt_mult / sl_mult (long),
      label −1 → lower barrier hit → gross R = -1 (long loses 1R at the stop),
      label  0 → time-expired    → R = ret / barrier_pct. Because the
        labeller doesn't persist per-event ATR/entry, we fall back to the
        configured barrier ratio: ret_pct / (sl_mult / (pt_mult + sl_mult))
        is a conservative proxy when the event exits mid-channel.

    The label uses the "long" framing (+1 = upside hit). The CV grader
    corrects the sign for SELL (short) predictions.
    """
    y = np.asarray(y, dtype=np.int64)
    ret = np.asarray(ret, dtype="float64")
    out = np.zeros_like(ret)
    # XGB label map: 0 = SELL (lower hit), 1 = HOLD (timeout), 2 = BUY (upper hit).
    out[y == 2] = pt_mult / sl_mult
    out[y == 0] = -1.0
    # For timeouts, estimate r_multiple from the realized return vs. the mean
    # barrier width. This is a proxy (we don't have per-event ATR here) but
    # stays conservative for the cost weight: a ret straddling zero produces
    # a small |r|, which correctly deprioritizes the sample.
    barrier_pct_proxy = (sl_mult) / max(sl_mult + pt_mult, 1.0)  # dimensional proxy
    timeout_mask = y == 1
    if timeout_mask.any() and barrier_pct_proxy > 0:
        out[timeout_mask] = ret[timeout_mask] / barrier_pct_proxy
    return out


def _cost_r_from_constants(
    *,
    n: int,
    cost_model: CostModel,
    participation: float,
    sl_mult: float,
    pt_mult: float,
) -> np.ndarray:
    """Uniform per-sample cost_in_r given a single assumed participation.

    The labeller doesn't persist per-event entry_price/ATR; rather than
    plumb them through we derive a uniform cost_in_r from the cost model
    and the sl/pt multiples. Per-symbol refinement is a P6+ concern.
    """
    entry_price = np.full(n, 100.0)  # normalized
    atr = np.full(n, sl_mult)        # → barrier_pct = sl_mult/100 per unit; wash
    # Use _cost_in_r with sl_mult=1.0 so barrier_pct = atr/entry = sl_mult/100.
    return _cost_in_r(
        entry_price=entry_price, atr=atr, sl_mult=1.0,
        participation=np.full(n, participation), cost_model=cost_model,
    )


def _build_reject_report(
    *,
    args: argparse.Namespace,
    cv_summary: dict[str, Any],
    label_dist: dict[str, Any],
    per_sym: dict[str, int],
    symbols: list[str],
    cost_model: CostModel,
    collect_seconds: float,
    uniqueness_seconds: float,
    cv_seconds: float,
    weights: np.ndarray,
    effective_n: float,
) -> dict[str, Any]:
    """Minimal report for the hard-reject exit path."""
    agg = cv_summary["aggregate"]
    return {
        "timestamp": datetime.now().isoformat(),
        "rejected": True,
        "reject_reason": (
            f"cost_drag_pct={agg['median_cost_drag_pct']:.1f}% exceeds "
            f"threshold {args.reject_cost_drag_pct:.1f}%"
        ),
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
            "cost_model": {
                "fee_bps": cost_model.fee_bps,
                "slip_bps": cost_model.slip_bps,
                "impact_coef": cost_model.impact_coef,
                "participation": args.participation,
            },
            "decision_threshold": args.decision_threshold,
            "reject_cost_drag_pct": args.reject_cost_drag_pct,
            "cost_aware": args.cost_aware,
        },
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
        },
        "timing_seconds": {
            "collection": round(collect_seconds, 1),
            "uniqueness": round(uniqueness_seconds, 1),
            "cv": round(cv_seconds, 1),
        },
        "cv_summary": cv_summary,
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
    # --- Cost-aware objective (PROMPT_PACK P2) -------------------------
    p.add_argument("--cost-aware", dest="cost_aware", action="store_true",
                   default=True,
                   help="Use PnL-weighted sample weights + cost-aware grading (default).")
    p.add_argument("--no-cost-aware", dest="cost_aware", action="store_false",
                   help="Legacy path: uniqueness-only weights. For regression comparison only.")
    p.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS,
                   help=f"Per-side fee in bps (default {DEFAULT_FEE_BPS}).")
    p.add_argument("--slip-bps", type=float, default=DEFAULT_SLIP_BPS,
                   help=f"Per-side slippage in bps (default {DEFAULT_SLIP_BPS}).")
    p.add_argument("--impact-coef", type=float, default=DEFAULT_IMPACT_COEF,
                   help=f"Almgren-Chriss impact coef (default {DEFAULT_IMPACT_COEF}).")
    p.add_argument("--participation", type=float, default=DEFAULT_PARTICIPATION,
                   help="Assumed participation (fraction of bar $ volume).")
    p.add_argument("--reject-cost-drag-pct", type=float,
                   default=DEFAULT_REJECT_COST_DRAG_PCT,
                   help="Hard-reject threshold: median cost-drag above this → exit 2.")
    p.add_argument("--decision-threshold", type=float,
                   default=DEFAULT_DECISION_THRESHOLD,
                   help="Min predicted class probability to take a trade (default 0.55).")
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
    uniqueness = compute_uniqueness(
        labels_for_weights, combined_index, groups=pd.Series(sym_groups),
    ).to_numpy()
    uniqueness = np.clip(uniqueness, 1e-6, 1.0)
    uniqueness_seconds = (datetime.now() - t1).total_seconds()
    effective_n = float(uniqueness.sum())
    logger.info(
        "uniqueness_done mean=%.3f min=%.3f effective_n=%.0f elapsed_s=%.1f",
        float(uniqueness.mean()), float(uniqueness.min()), effective_n, uniqueness_seconds,
    )

    # --- Cost-aware weighting (PROMPT_PACK P2 §1) ----------------------
    # Sample weight = max(|r_R| - cost_in_r, 0) * uniqueness.
    # Samples where costs dominate gross edge get near-zero weight, so the
    # classifier spends its capacity on setups that actually pay after costs.
    cost_model = CostModel(
        fee_bps=args.fee_bps, slip_bps=args.slip_bps, impact_coef=args.impact_coef,
    )
    # Backfill a synthetic entry_price + ATR per sample from the realized return.
    # We don't persist them in the sample collector today, so we reconstruct:
    #   1R = sl_mult * ATR / entry_price ≡ barrier_pct. Given the labeller
    #   uses the same sl_mult for all symbols and ATR is already absorbed into
    #   `ret` via the triple-barrier, a proxy barrier_pct comes from |ret| at
    #   the stop-barrier hit (label == -1) — but for general grading we can
    #   assume entry_price=100, atr = args.sl_mult_pct_hint * entry; easier to
    #   use unit entry and let users pass normalized ATR. In practice, because
    #   ret is already a decimal return and 1R := sl_mult * ATR / entry_price,
    #   we compute r_multiples directly from the configured (sl_mult, pt_mult)
    #   using the realized-return magnitude and the barrier magnitude that
    #   fired for each label (pt for +1, sl for -1, ret for 0).
    sample_r_multiples = _r_multiples_from_labels(
        y=y, ret=ret, pt_mult=args.pt_mult, sl_mult=args.sl_mult,
    )
    sample_cost_r = _cost_r_from_constants(
        n=len(y), cost_model=cost_model,
        participation=args.participation,
        sl_mult=args.sl_mult, pt_mult=args.pt_mult,
    )

    if args.cost_aware:
        weights = compute_sample_weights(
            r_multiples=sample_r_multiples,
            cost_in_r=sample_cost_r,
            uniqueness=uniqueness,
        )
    else:
        weights = uniqueness.copy()

    # --- Purged CV evaluation ------------------------------------------
    t2 = datetime.now()
    bars_per_year = max(1, int(252 * (390 / max(1, args.frequency))))
    cv_summary = purged_cv_evaluate(
        X=X, y=y, weights=weights, touch_times=touch_series,
        n_splits=args.folds, embargo_pct=args.embargo_pct,
        sample_r_multiples=sample_r_multiples,
        sample_cost_r=sample_cost_r,
        bars_per_year=bars_per_year,
        decision_threshold=args.decision_threshold,
    )
    cv_seconds = (datetime.now() - t2).total_seconds()
    agg = cv_summary["aggregate"]
    logger.info(
        "cv_done median_expectancy=%.3fR median_PF=%.2f median_cost_drag=%.1f%% "
        "var_sharpe=%.3f folds=%d elapsed_s=%.1f",
        agg["median_expectancy_r"], agg["median_profit_factor"],
        agg["median_cost_drag_pct"], agg["var_sharpe_across_folds"],
        agg["n_folds_executed"], cv_seconds,
    )

    # --- Hard-reject on cost-drag (PROMPT_PACK P2 §4) ------------------
    reject = agg["median_cost_drag_pct"] > args.reject_cost_drag_pct
    if reject:
        logger.error(
            "model_rejected reason=cost_drag_pct median=%.1f%% threshold=%.1f%%",
            agg["median_cost_drag_pct"], args.reject_cost_drag_pct,
        )
        report = _build_reject_report(
            args=args, cv_summary=cv_summary,
            label_dist=label_dist, per_sym=per_sym, symbols=symbols,
            cost_model=cost_model,
            collect_seconds=collect_seconds,
            uniqueness_seconds=uniqueness_seconds,
            cv_seconds=cv_seconds,
            weights=weights, effective_n=effective_n,
        )
        Path(args.report_path).write_text(json.dumps(report, indent=2, default=str))
        return 2

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
            "cost_model": {
                "fee_bps": cost_model.fee_bps,
                "slip_bps": cost_model.slip_bps,
                "impact_coef": cost_model.impact_coef,
                "participation": args.participation,
            },
            "decision_threshold": args.decision_threshold,
            "cost_aware": args.cost_aware,
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
