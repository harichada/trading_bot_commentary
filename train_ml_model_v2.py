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
from ml.events import cusum_filter
from ml.iid_diagnostics import IIDReport, compute_iid_diagnostics, weighted_residuals
from ml.labels import apply_triple_barrier, compute_uniqueness, triple_barrier_labels
from ml.pnl_objective import compute_sample_weights
from ml.purged_cv import PurgedKFold
from ml.sample_weights import get_indicator_matrix, sequential_bootstrap_indices
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

# P4 defaults (PROMPT_PACK §P4).
DEFAULT_BAR_TYPE = "time"
DEFAULT_DEFERRED_BAR_TYPES = ["dollar", "volume", "imbalance"]
DEFAULT_VERTICAL_MULT_DURATION = 2.0  # vertical = pt-duration × this
DEFAULT_MIN_RET_ATR_MULT = 0.5         # AFML p.47 low-magnitude filter
DEFAULT_BOOTSTRAP_ITERATIONS = 5
DEFAULT_EVENT_SOURCE = "cusum"
DEFAULT_CUSUM_H = 0.005                # 50 bps cumulative-return threshold
# Per-fold AR(1) gate (replaces single stitched AR(1) gate). Stitched AR(1)
# stays in the report as an informational metric — it measures cross-fold
# calibration drift, not label IID-ness, so it belongs to P2 calibration
# / P14 drift work, not P4.
P4_AR1_MEDIAN_GATE = 0.10              # median per-fold |AR(1)| weighted residuals
P4_AR1_MAX_GATE = 0.20                 # max per-fold |AR(1)| (single-fold ceiling)
P4_BOOTSTRAP_CV_GATE = 0.5             # cv_of_expectancy must be < this

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


def _candidate_events(
    prices: pd.Series,
    *,
    event_source: str,
    event_stride: int,
    cusum_h: float,
    first: int,
    last: int,
) -> pd.DatetimeIndex:
    """Generate event timestamps from either stride or CUSUM filter.

    AFML §2.5.2 — CUSUM samples on cumulative-return excursions, putting
    label work where information lives. ``stride`` is the legacy P2 path
    retained for apples-to-apples regression.
    """
    if event_source == "stride":
        return prices.index[first:last:event_stride]
    if event_source == "cusum":
        # CUSUM on the post-warmup, pre-tail window so labels never reach
        # past the available bars.
        windowed = prices.iloc[first:last]
        ev = cusum_filter(windowed, h=cusum_h)
        return ev
    raise ValueError(f"unknown event_source: {event_source!r}")


def label_symbol(
    df: pd.DataFrame,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
    event_stride: int,
    *,
    event_source: str = DEFAULT_EVENT_SOURCE,
    cusum_h: float = DEFAULT_CUSUM_H,
    vertical_mult_duration: float = DEFAULT_VERTICAL_MULT_DURATION,
    min_ret_atr_mult: float = DEFAULT_MIN_RET_ATR_MULT,
) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """Return (labels_df, atr_series, drop_counts) for one symbol.

    P4: events come from the configured source (default CUSUM, AFML §2.5.2);
    labels apply the triple-barrier first-touch with the AFML p.47 low-
    magnitude filter (``min_ret * atr / entry``); the vertical barrier is
    set to ``vertical_mult_duration * max_holding`` (default 2× expected
    trade duration, satisfying P4 §1).

    The returned ``drop_counts`` dict carries pre/post counts so the report
    can show how many candidates were filtered for low-magnitude vs.
    no-barrier-hit (``bin == 0``) outcomes.
    """
    atr = _compute_raw_atr(df, window=14)
    prices = df["Close"]

    vertical = max(1, int(round(vertical_mult_duration * max_holding)))
    first = 50
    last = len(df) - vertical - 1
    drop_counts = {"candidates": 0, "dropped_low_magnitude": 0, "dropped_no_barrier_hit": 0}

    if last <= first:
        return pd.DataFrame(), atr, drop_counts

    candidate_times = _candidate_events(
        prices, event_source=event_source, event_stride=event_stride,
        cusum_h=cusum_h, first=first, last=last,
    )
    drop_counts["candidates"] = int(len(candidate_times))
    if len(candidate_times) == 0:
        return pd.DataFrame(), atr, drop_counts

    events_df = pd.DataFrame(
        {"vertical": [vertical] * len(candidate_times)},
        index=candidate_times,
    )
    labels = triple_barrier_labels(
        events=events_df, close=prices, atr=atr,
        pt_sl=(pt_mult, sl_mult),
        vertical_barrier=vertical,
        min_ret=min_ret_atr_mult,
    )

    # Translate to legacy schema (label/touch_time/ret) so the rest of the
    # collector code stays unchanged. P4 emits {-1, 0, +1}; we also expose
    # the side-adjusted bin via the same column.
    if labels.empty:
        # Distinguish "all candidates filtered by min_ret" from "no candidates".
        drop_counts["dropped_low_magnitude"] = drop_counts["candidates"]
        return pd.DataFrame(), atr, drop_counts

    drop_counts["dropped_low_magnitude"] = (
        drop_counts["candidates"] - len(labels)
    )
    drop_counts["dropped_no_barrier_hit"] = int((labels["bin"] == 0).sum())

    legacy = pd.DataFrame({
        "label": labels["bin"].astype(int).to_numpy(),
        "touch_time": labels["t1"].to_numpy(),
        "ret": labels["ret"].to_numpy(),
    }, index=labels.index)
    return legacy, atr, drop_counts


def collect_v2_samples(
    symbols: list[str],
    days: int,
    frequency: int,
    dsn: str,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
    event_stride: int,
    *,
    event_source: str = DEFAULT_EVENT_SOURCE,
    cusum_h: float = DEFAULT_CUSUM_H,
    vertical_mult_duration: float = DEFAULT_VERTICAL_MULT_DURATION,
    min_ret_atr_mult: float = DEFAULT_MIN_RET_ATR_MULT,
    drop_accumulator: dict[str, int] | None = None,
    prices_accumulator: dict[str, pd.DataFrame] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index, pd.Index, np.ndarray, dict[str, int]]:
    """Return (X, y, ret, event_times, touch_times, symbol_groups, per_symbol_counts).

    The ``symbol_groups`` array is one entry per sample, naming the source
    symbol — used downstream so uniqueness weights are computed within each
    symbol's timeline rather than pooling all symbols into a single ticker-tape.

    ``drop_accumulator`` (optional, mutated in place) collects aggregate
    drop counts across symbols: ``candidates``, ``dropped_low_magnitude``,
    ``dropped_no_barrier_hit``. Used by the P4 report.

    ``prices_accumulator`` (optional, mutated in place) retains the raw
    per-symbol bar DataFrames keyed by symbol. Required by the P3
    regime-routing pipeline; P4 leaves it ``None``.
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

        if prices_accumulator is not None:
            # P3 needs per-symbol bar history for regime classification.
            # Retain a reference; downstream consumers must not mutate.
            prices_accumulator[symbol] = df

        feat_df = feature_model.feature_extractor.batch_extract(df)
        if feat_df.empty:
            logger.warning("no_features symbol=%s", symbol)
            continue

        labels_df, _, drop_counts = label_symbol(
            df, pt_mult=pt_mult, sl_mult=sl_mult,
            max_holding=max_holding, event_stride=event_stride,
            event_source=event_source, cusum_h=cusum_h,
            vertical_mult_duration=vertical_mult_duration,
            min_ret_atr_mult=min_ret_atr_mult,
        )
        if drop_accumulator is not None:
            for k, v in drop_counts.items():
                drop_accumulator[k] = drop_accumulator.get(k, 0) + int(v)
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
    event_times: pd.DatetimeIndex,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    use_sequential_bootstrap: bool = True,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Cost-aware purged-CV grading.

    Replaces accuracy/MCC/classification-report with the six PnL-aware metrics
    required by PROMPT_PACK P2 §2. Per-fold ``FoldMetrics`` are collected;
    aggregation uses **median** across folds (López de Prado AFML §12.3 — the
    mean is dominated by single-fold luck when fold variance is high).

    Sortino is annualized by ``sqrt(trades_per_year)`` computed empirically
    per fold from the actual trade frequency over the fold's calendar window
    (trades_per_year = trades_taken / fold_calendar_days * 252). This matches
    the per-trade nature of R-multiples — scaling by bars-per-year would
    over-state the risk-adjusted return whenever the strategy is flat
    between trades.
    """
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    cv = PurgedKFold(n_splits=n_splits, touch_times=touch_times, embargo_pct=embargo_pct)

    fold_reports: list[dict[str, Any]] = []
    fold_metrics_objs: list[FoldMetrics] = []
    event_times_arr = np.asarray(event_times)
    touch_times_arr = touch_times.sort_index().to_numpy()
    event_index_arr = touch_times.sort_index().index.to_numpy()

    # Per-iteration, cross-fold accumulators for the bootstrap_stability block.
    iter_gross: list[list[float]] = [[] for _ in range(bootstrap_iterations)]
    iter_net: list[list[float]] = [[] for _ in range(bootstrap_iterations)]

    # Out-of-sample residual collection for IID diagnostics. We use the
    # iteration-mean predicted probability of the realized class as the
    # "predicted probability" entering the residual.
    oos_residual_inputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # Per-fold AR(1) — the gate input. Computed inside the fold loop so it
    # measures label IID-ness within a single CV split, not cross-fold
    # calibration drift.
    per_fold_ar1: list[float] = []
    per_fold_lb_p: list[float | None] = []

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

        # AFML §4.5.3: build the fold-local indicator matrix from the
        # training events' label windows so sequential bootstrap can score
        # candidate-row uniqueness conditional on already-drawn rows. The
        # matrix is shape (n_train_bars, n_train_events) — small enough
        # to keep in memory for retail-scale universes.
        if use_sequential_bootstrap:
            train_event_times = pd.DatetimeIndex(event_index_arr[train_idx])
            train_touch_times = pd.DatetimeIndex(touch_times_arr[train_idx])
            train_bar_index = pd.DatetimeIndex(
                np.unique(np.concatenate([
                    train_event_times.to_numpy(),
                    train_touch_times.to_numpy(),
                ]))
            ).sort_values()
            ind_matrix = get_indicator_matrix(
                train_bar_index, train_event_times, train_touch_times,
            )
        else:
            ind_matrix = None

        # Average per-iteration probas to drive the headline fold metric.
        proba_acc = np.zeros((len(test_idx), 3), dtype="float64")
        n_succ_iter = 0

        for it in range(bootstrap_iterations):
            if use_sequential_bootstrap and ind_matrix is not None:
                # Per-iteration deterministic seed → reproducible.
                resampled = sequential_bootstrap_indices(
                    ind_matrix, n_samples=len(train_idx),
                    seed=bootstrap_seed + 100 * fold_idx + it,
                )
                X_fit = X_tr_s[resampled]
                y_fit = y_tr[resampled]
                w_fit = w_tr[resampled]
                # AFML §4.5: when the bagging draw is sequential, the row's
                # original sample_weight is preserved (not recomputed) — the
                # bootstrap replaces the bagging-level sampling, not the
                # cost-aware loss weighting.
                subsample_param = 1.0
            else:
                X_fit, y_fit, w_fit = X_tr_s, y_tr, w_tr
                subsample_param = 0.8

            clf = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=subsample_param, colsample_bytree=0.8,
                random_state=42 + it,
                eval_metric="mlogloss", num_class=3,
                objective="multi:softprob", n_jobs=-1,
            )
            X_fit_p, y_fit_p, w_fit_p = _ensure_all_classes(X_fit, y_fit, w_fit)
            clf.fit(X_fit_p, y_fit_p, sample_weight=w_fit_p, verbose=False)

            try:
                probas_it = clf.predict_proba(X_te_s)
            except Exception:  # pragma: no cover
                probas_it = None
            y_pred_it = clf.predict(X_te_s)

            gross_it, net_it = _fold_net_r_multiples(
                y_pred=y_pred_it,
                sample_r_multiples=sample_r_multiples[test_idx],
                sample_cost_r=sample_cost_r[test_idx],
                decision_threshold=decision_threshold,
                probas=probas_it,
            )
            iter_gross[it].extend(gross_it.tolist())
            iter_net[it].extend(net_it.tolist())

            if probas_it is not None:
                proba_acc += probas_it
                n_succ_iter += 1

        # Headline per-fold prediction = average across bootstrap iterations.
        if n_succ_iter > 0:
            avg_proba = proba_acc / n_succ_iter
            y_pred = avg_proba.argmax(axis=1)
        else:
            avg_proba = None
            y_pred = clf.predict(X_te_s)  # last-iter fallback

        gross_r, net_r = _fold_net_r_multiples(
            y_pred=y_pred,
            sample_r_multiples=sample_r_multiples[test_idx],
            sample_cost_r=sample_cost_r[test_idx],
            decision_threshold=decision_threshold,
            probas=avg_proba,
        )

        if avg_proba is not None:
            # Capture iteration-mean predicted probability of the *realized*
            # class — used to form weighted residuals for IID diagnostics.
            p_realized = avg_proba[np.arange(len(test_idx)), y_te]
            oos_residual_inputs.append((y_te.copy(), p_realized, weights[test_idx].copy()))

            # Per-fold AR(1) (gate input). The label IID-ness check belongs
            # at fold scope; the stitched cross-fold version below is kept
            # only as an informational measure of calibration drift.
            fold_residuals = weighted_residuals(
                np.ones_like(y_te), p_realized, weights[test_idx],
            )
            fold_iid = compute_iid_diagnostics(fold_residuals)
            fold_reports[-1] if False else None  # placeholder; appended below
            per_fold_ar1.append(float(fold_iid.ar1_residual_autocorr))
            per_fold_lb_p.append(
                float(fold_iid.ljung_box_p_value)
                if fold_iid.ljung_box_p_value is not None else None
            )
        else:
            per_fold_ar1.append(float("nan"))
            per_fold_lb_p.append(None)

        # Empirical trades_per_year for this fold's calendar window.
        fold_event_times = event_times_arr[test_idx]
        fold_span_days = max(
            1.0,
            (pd.Timestamp(fold_event_times.max()) - pd.Timestamp(fold_event_times.min())).total_seconds() / 86400.0,
        )
        n_trades_taken = max(1, int(net_r.size))
        trades_per_year = max(1, int(round(n_trades_taken / fold_span_days * 252.0)))

        metrics = grade_fold(gross_r=gross_r, net_r=net_r, bars_per_year=trades_per_year)
        fold_metrics_objs.append(metrics)

        sharpe_proxy = sortino(net_r, bars_per_year=trades_per_year)
        fold_reports.append({
            "fold": fold_idx,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "n_trades_taken": metrics.n_trades,
            "fold_calendar_days": round(fold_span_days, 2),
            "trades_per_year": trades_per_year,
            "metrics": {
                "expectancy_r": metrics.expectancy_r,
                "profit_factor": _finite_or_none(metrics.profit_factor),
                "sortino": _finite_or_none(metrics.sortino),
                "calmar": _finite_or_none(metrics.calmar),
                "max_adverse_excursion": metrics.max_adverse_excursion,
                "cost_drag_pct": metrics.cost_drag_pct,
                "absolute_cost_per_trade_r": metrics.absolute_cost_per_trade_r,
                "ar1_residual_autocorr": per_fold_ar1[-1] if per_fold_ar1 else None,
                "ljung_box_p_value": per_fold_lb_p[-1] if per_fold_lb_p else None,
            },
            "sharpe_proxy": _finite_or_none(sharpe_proxy),
        })
        if metrics.cost_drag_pct is not None:
            cost_str = f"cost_drag={metrics.cost_drag_pct:.1f}%"
        else:
            cost_str = (
                f"cost_drag=N/A (gross ≤ 0)  "
                f"abs_cost={metrics.absolute_cost_per_trade_r:+.4f}R/trade"
            )
        logger.info(
            "fold %d/%d trades=%d expectancy=%.3fR PF=%.2f sortino=%.2f %s",
            fold_idx, n_splits, metrics.n_trades, metrics.expectancy_r,
            metrics.profit_factor if np.isfinite(metrics.profit_factor) else float("nan"),
            metrics.sortino if np.isfinite(metrics.sortino) else float("nan"),
            cost_str,
        )

    aggregate = aggregate_fold_metrics(fold_metrics_objs)

    # ---- Bootstrap stability (cross-fold per-iteration aggregation) ----
    bs_block = _bootstrap_stability_block(iter_gross, iter_net)

    # ---- IID diagnostics — per-fold (gate) + stitched (informational) --
    # Per-fold AR(1) is the *gate*: it measures label IID-ness within a
    # single CV split, which is what P4's labeling pipeline owns.
    finite_ar1 = [a for a in per_fold_ar1 if np.isfinite(a)]
    if finite_ar1:
        median_ar1 = float(np.median([abs(a) for a in finite_ar1]))
        max_ar1 = float(np.max([abs(a) for a in finite_ar1]))
    else:
        median_ar1 = 0.0
        max_ar1 = 0.0
    passes_per_fold = (
        median_ar1 < P4_AR1_MEDIAN_GATE and max_ar1 < P4_AR1_MAX_GATE
    )

    # Stitched cross-fold AR(1) — INFORMATIONAL. It measures cross-fold
    # calibration drift, not label IID-ness. Owned by P2 calibration / P14
    # drift work, not P4. Kept here for visibility before P14 wires it
    # formally.
    if oos_residual_inputs:
        y_all = np.concatenate([t[0] for t in oos_residual_inputs])
        p_real = np.concatenate([t[1] for t in oos_residual_inputs])
        w_all = np.concatenate([t[2] for t in oos_residual_inputs])
        residuals_stitched = weighted_residuals(np.ones_like(y_all), p_real, w_all)
        stitched_report = compute_iid_diagnostics(residuals_stitched)
    else:
        stitched_report = IIDReport(0.0, None, True, 0)

    return {
        "n_folds_executed": len(fold_reports),
        "aggregate": aggregate,
        "fold_detail": fold_reports,
        "bootstrap_stability": bs_block,
        "iid_diagnostics": {
            # Gate inputs (per-fold).
            "per_fold_ar1": [
                float(a) if np.isfinite(a) else None for a in per_fold_ar1
            ],
            "median_per_fold_ar1": median_ar1,
            "max_per_fold_ar1": max_ar1,
            "passes_per_fold_gate": bool(passes_per_fold),
            "gate_thresholds": {
                "median_lt": P4_AR1_MEDIAN_GATE,
                "max_lt": P4_AR1_MAX_GATE,
            },
            # Informational only — see note above.
            "stitched_residual_ar1": float(stitched_report.ar1_residual_autocorr),
            "stitched_residual_ar1_note": (
                "informational — measures cross-fold calibration drift, "
                "not label IID-ness; owned by P2 calibration / P14 drift "
                "work, not P4"
            ),
            "stitched_ljung_box_p_value": (
                float(stitched_report.ljung_box_p_value)
                if stitched_report.ljung_box_p_value is not None else None
            ),
            "stitched_n_residuals": int(stitched_report.n_residuals),
        },
    }


def _ensure_all_classes(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, n_classes: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append one zero-weight row per missing class.

    XGBoost's sklearn wrapper validates ``np.unique(y)`` against contiguous
    integers ``[0, n_classes)``. Sequential bootstrap resamples or the
    P4 ``min_ret`` filter can drop a class entirely (e.g. zero HOLDs when
    every event resolves at a price barrier). Injecting a single
    sample_weight=0 row per missing class keeps the multi-class objective
    valid without affecting the loss (the cost-aware weighting is preserved
    for every real row).
    """
    if X.size == 0:
        return X, y, w
    present = set(np.unique(y).tolist())
    missing = [c for c in range(n_classes) if c not in present]
    if not missing:
        return X, y, w
    pad_X = np.tile(X[0:1, :], (len(missing), 1))
    pad_y = np.asarray(missing, dtype=y.dtype)
    pad_w = np.zeros(len(missing), dtype=w.dtype)
    return (
        np.concatenate([X, pad_X], axis=0),
        np.concatenate([y, pad_y], axis=0),
        np.concatenate([w, pad_w], axis=0),
    )


def _bootstrap_stability_block(
    iter_gross: list[list[float]],
    iter_net: list[list[float]],
) -> dict[str, Any]:
    """Across-iteration variance of expectancy / profit-factor.

    AFML §4.5 caveat: a single sequential-bootstrap fit is one realization
    of a stochastic ensemble. Reporting the spread across iterations
    surfaces whether the model's edge is stable or seed-dependent.
    """
    expectancies: list[float] = []
    profit_factors: list[float] = []
    for gs, ns in zip(iter_gross, iter_net):
        if not ns:
            continue
        net = np.asarray(ns, dtype="float64")
        expectancies.append(float(net.mean()))
        wins = float(net[net > 0].sum())
        losses = float(-net[net < 0].sum())
        profit_factors.append(wins / losses if losses > 0 else float("inf"))

    if not expectancies:
        return {
            "iterations": len(iter_gross),
            "expectancy_r_per_iter": [],
            "profit_factor_per_iter": [],
            "cv_of_expectancy": float("nan"),
            "passes_cv_lt_0_5": False,
        }

    mean_exp = float(np.mean(expectancies))
    std_exp = float(np.std(expectancies, ddof=0))
    cv = std_exp / abs(mean_exp) if mean_exp != 0 else float("inf")
    return {
        "iterations": len(iter_gross),
        "expectancy_r_per_iter": [round(e, 6) for e in expectancies],
        "profit_factor_per_iter": [
            round(p, 4) if np.isfinite(p) else None for p in profit_factors
        ],
        "cv_of_expectancy": (
            round(cv, 4) if np.isfinite(cv) else None
        ),
        "passes_cv_lt_0_5": bool(np.isfinite(cv) and cv < P4_BOOTSTRAP_CV_GATE),
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
            "median_cost_drag_pct": None,
            "median_absolute_cost_per_trade_r": 0.0,
            "n_folds_with_undefined_cost_drag": 0,
            "var_sharpe_across_folds": 0.0,
            "n_folds_executed": 0,
        }

    # Drop non-finite values before taking medians — keeps inf/nan from poisoning stats.
    def _finite(values: list[float]) -> np.ndarray:
        arr = np.asarray(values, dtype="float64")
        return arr[np.isfinite(arr)]

    def _median_or_zero(values: list[float]) -> float:
        arr = _finite(values)
        return float(np.median(arr)) if arr.size else 0.0

    sortino_vals = _finite([f.sortino for f in folds])
    var_sharpe = float(np.var(sortino_vals, ddof=0)) if sortino_vals.size else 0.0

    drag_vals = [f.cost_drag_pct for f in folds if f.cost_drag_pct is not None]
    abs_cost_vals = [
        f.absolute_cost_per_trade_r for f in folds
        if f.absolute_cost_per_trade_r is not None
    ]
    median_drag = round(float(np.median(drag_vals)), 1) if drag_vals else None
    median_abs_cost = (
        round(float(np.median(abs_cost_vals)), 4) if abs_cost_vals else 0.0
    )

    return {
        "median_expectancy_r": _median_or_zero([f.expectancy_r for f in folds]),
        "median_profit_factor": _median_or_zero([f.profit_factor for f in folds]),
        "median_sortino": _median_or_zero([f.sortino for f in folds]),
        "median_calmar": _median_or_zero([f.calmar for f in folds]),
        "median_max_adverse_excursion": _median_or_zero(
            [f.max_adverse_excursion for f in folds]
        ),
        "median_cost_drag_pct": median_drag,
        "median_absolute_cost_per_trade_r": median_abs_cost,
        "n_folds_with_undefined_cost_drag": len(folds) - len(drag_vals),
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
    reject_reason: str,
) -> dict[str, Any]:
    """Minimal report for the hard-reject exit path."""
    return {
        "timestamp": datetime.now().isoformat(),
        "rejected": True,
        "reject_reason": reject_reason,
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
    # --- P4: triple-barrier + sequential bootstrap + IID --------------
    p.add_argument("--bar-type", choices=["time", "dollar", "volume", "imbalance"],
                   default=DEFAULT_BAR_TYPE,
                   help="Bar construction (AFML §2.3). dollar/volume/imbalance "
                        "require tick data and currently raise NotImplementedError.")
    p.add_argument("--event-source", choices=["cusum", "stride"],
                   default=DEFAULT_EVENT_SOURCE,
                   help="Event generator: cusum (AFML §2.5.2, default) or stride "
                        "(every Nth bar — for P2 apples-to-apples comparison only).")
    p.add_argument("--cusum-h", type=float, default=DEFAULT_CUSUM_H,
                   help="CUSUM cumulative-return threshold (default 0.005 = 50bps).")
    p.add_argument("--vertical-mult-duration", type=float,
                   default=DEFAULT_VERTICAL_MULT_DURATION,
                   help="Vertical barrier = this × max_holding (default 2×).")
    p.add_argument("--min-ret-atr-mult", type=float, default=DEFAULT_MIN_RET_ATR_MULT,
                   help="Drop events with |ret| < this × atr/entry (AFML p.47).")
    p.add_argument("--bootstrap-iterations", type=int,
                   default=DEFAULT_BOOTSTRAP_ITERATIONS,
                   help="Sequential-bootstrap fits per fold (default 5).")
    p.add_argument("--no-sequential-bootstrap", dest="sequential_bootstrap",
                   action="store_false",
                   help="Use random subsample=0.8 instead of sequential bootstrap "
                        "(regression comparison only — statistically wrong for "
                        "overlapping labels).")
    p.set_defaults(sequential_bootstrap=True)
    p.add_argument("--regression-check", default=None,
                   help="Path to a baseline JSON (e.g. P2_baseline_honest.json). "
                        "If set, the report includes a warn-only diff against the "
                        "four P2 anchor metrics. Per the user's P4 override, P2 "
                        "regression does NOT cause a hard reject — IID/uniqueness/"
                        "label-distribution gates do.")
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

    # --- Collection + labeling + uniqueness + cost-aware weights ------
    # Delegated to ml.event_view.build_event_view (Step 1 of P3
    # refactor; parity-tested in tests/test_event_view_parity.py).
    from ml.event_view import build_event_view as _build_event_view
    logger.info(
        "collecting_v2_samples symbols=%d days=%d freq=%dmin pt=%.2f sl=%.2f mh=%d stride=%d",
        len(symbols), args.days, args.frequency,
        args.pt_mult, args.sl_mult, args.max_holding, args.event_stride,
    )
    ev = _build_event_view(
        symbols=symbols, days=args.days, frequency=args.frequency, dsn=args.dsn,
        pt_mult=args.pt_mult, sl_mult=args.sl_mult,
        max_holding=args.max_holding, event_stride=args.event_stride,
        cost_aware=args.cost_aware,
        fee_bps=args.fee_bps, slip_bps=args.slip_bps,
        impact_coef=args.impact_coef, participation=args.participation,
        event_source=args.event_source, cusum_h=args.cusum_h,
        vertical_mult_duration=args.vertical_mult_duration,
        min_ret_atr_mult=args.min_ret_atr_mult,
        collect_prices=False,  # P4 doesn't need bar history; saves memory.
    )
    if len(ev.X) == 0:
        logger.error("no_samples_collected — aborting")
        return 1

    # Unpack into the existing local names so the rest of main() is
    # unchanged. This preserves byte-identical behavior of the post-
    # event-view pipeline (purged_cv_evaluate, report assembly).
    X = ev.X
    y = ev.y
    ret = ev.ret
    event_times = ev.event_times
    touch_times = ev.touch_times
    sym_groups = ev.sym_groups
    per_sym = ev.per_symbol_samples
    drop_acc = ev.drop_counts
    uniqueness = ev.uniqueness
    sample_r_multiples = ev.sample_r_multiples
    sample_cost_r = ev.sample_cost_r
    weights = ev.weights
    collect_seconds = ev.timing["collect"]
    uniqueness_seconds = ev.timing["uniqueness"]
    effective_n = float(uniqueness.sum())
    touch_series = pd.Series(touch_times.to_numpy(), index=event_times)
    cost_model = CostModel(
        fee_bps=args.fee_bps, slip_bps=args.slip_bps, impact_coef=args.impact_coef,
    )

    label_dist = describe_labels(y)
    logger.info(
        "samples_collected total=%d BUY=%.1f%% HOLD=%.1f%% SELL=%.1f%% elapsed_s=%.1f",
        label_dist["total"],
        label_dist["BUY"]["pct"], label_dist["HOLD"]["pct"], label_dist["SELL"]["pct"],
        collect_seconds,
    )
    logger.info("computing_uniqueness n=%d groups=%d",
                len(event_times), len(np.unique(sym_groups)))
    logger.info(
        "uniqueness_done mean=%.3f min=%.3f effective_n=%.0f elapsed_s=%.1f",
        float(uniqueness.mean()), float(uniqueness.min()), effective_n, uniqueness_seconds,
    )

    # --- Purged CV evaluation ------------------------------------------
    t2 = datetime.now()
    cv_summary = purged_cv_evaluate(
        X=X, y=y, weights=weights, touch_times=touch_series,
        n_splits=args.folds, embargo_pct=args.embargo_pct,
        sample_r_multiples=sample_r_multiples,
        sample_cost_r=sample_cost_r,
        event_times=event_times,
        decision_threshold=args.decision_threshold,
        bootstrap_iterations=args.bootstrap_iterations,
        use_sequential_bootstrap=args.sequential_bootstrap,
    )
    cv_seconds = (datetime.now() - t2).total_seconds()

    # Lift IID + bootstrap-stability blocks to top level — they are not
    # per-fold metrics, they characterize the cross-fold residual stream.
    iid_block = cv_summary.pop("iid_diagnostics")
    bootstrap_block = cv_summary.pop("bootstrap_stability")

    agg = cv_summary["aggregate"]
    if agg["median_cost_drag_pct"] is not None:
        cost_log = f"median_cost_drag={agg['median_cost_drag_pct']:.1f}%"
    else:
        cost_log = (
            f"median_cost_drag=N/A (gross ≤ 0 in "
            f"{agg['n_folds_with_undefined_cost_drag']}/{agg['n_folds_executed']} folds)  "
            f"median_abs_cost={agg['median_absolute_cost_per_trade_r']:+.4f}R/trade"
        )
    logger.info(
        "cv_done median_expectancy=%.3fR median_PF=%.2f %s "
        "var_sharpe=%.3f folds=%d elapsed_s=%.1f",
        agg["median_expectancy_r"], agg["median_profit_factor"],
        cost_log, agg["var_sharpe_across_folds"],
        agg["n_folds_executed"], cv_seconds,
    )

    # --- P4 hard-reject gates (PROMPT_PACK §P4 + user override) --------
    # These are P4-INTRINSIC: they test whether the labeling math itself
    # is sound, regardless of model edge. Failure here means the training
    # distribution is broken. Per the user's P4 override, P2 economic
    # anchors are NOT in this gate set — they live in the regression-check
    # warning block at the end.
    p4_reject_reasons: list[str] = []
    if not iid_block["passes_per_fold_gate"]:
        p4_reject_reasons.append(
            f"per-fold AR(1) gate failed: median={iid_block['median_per_fold_ar1']:.3f} "
            f"(< {P4_AR1_MEDIAN_GATE}), max={iid_block['max_per_fold_ar1']:.3f} "
            f"(< {P4_AR1_MAX_GATE}); per-fold values={iid_block['per_fold_ar1']}"
        )
    # Uniqueness sanity: weights must be present, in (0, 1].
    u_mean = float(weights.mean()) if len(weights) else 0.0
    u_min = float(weights.min()) if len(weights) else 0.0
    if u_mean <= 0.0 or u_mean > 1.0 or u_min <= 0.0:
        p4_reject_reasons.append(
            f"uniqueness weights degenerate (mean={u_mean:.4f}, min={u_min:.4f})"
        )
    # Label distribution sanity: P4's tighter labeling legitimately can produce
    # zero HOLDs when every event resolves at a price barrier (vertical=120 bars).
    # Gate: at least BUY+SELL present AND no class > 95%.
    classes_present = sum(1 for k in ("BUY", "HOLD", "SELL")
                          if label_dist[k]["count"] > 0)
    max_class_pct = max(label_dist[k]["pct"] for k in ("BUY", "HOLD", "SELL"))
    has_directional = label_dist["BUY"]["count"] > 0 and label_dist["SELL"]["count"] > 0
    if not has_directional or classes_present < 2 or max_class_pct > 95.0:
        p4_reject_reasons.append(
            f"label distribution degenerate (classes_present={classes_present}, "
            f"BUY/SELL_both_present={has_directional}, "
            f"max_class_pct={max_class_pct:.1f})"
        )

    if p4_reject_reasons:
        reason = " | ".join(p4_reject_reasons)
        logger.error("p4_hard_reject reason=%s", reason)
        report = _build_reject_report(
            args=args, cv_summary=cv_summary,
            label_dist=label_dist, per_sym=per_sym, symbols=symbols,
            cost_model=cost_model,
            collect_seconds=collect_seconds,
            uniqueness_seconds=uniqueness_seconds,
            cv_seconds=cv_seconds,
            weights=weights, effective_n=effective_n,
            reject_reason=reason,
        )
        report["iid_diagnostics"] = iid_block
        report["bootstrap_stability"] = bootstrap_block
        Path(args.report_path).write_text(json.dumps(report, indent=2, default=str))
        return 3

    # --- Hard-reject on cost-drag (PROMPT_PACK P2 §4) ------------------
    # Reject if median cost-drag exceeds the threshold OR if it's undefined
    # (gross edge non-positive — there is nothing to drag against, which is
    # itself a reject condition).
    if agg["median_cost_drag_pct"] is None:
        reject = True
        reject_reason = (
            f"median cost-drag is N/A (gross ≤ 0 in "
            f"{agg['n_folds_with_undefined_cost_drag']}/{agg['n_folds_executed']} folds); "
            f"median absolute cost = "
            f"{agg['median_absolute_cost_per_trade_r']:+.4f}R/trade"
        )
    else:
        reject = agg["median_cost_drag_pct"] > args.reject_cost_drag_pct
        reject_reason = (
            f"cost_drag_pct={agg['median_cost_drag_pct']:.1f}% exceeds "
            f"threshold {args.reject_cost_drag_pct:.1f}%"
        )
    if reject:
        logger.error("model_rejected reason=%s", reject_reason)
        report = _build_reject_report(
            args=args, cv_summary=cv_summary,
            label_dist=label_dist, per_sym=per_sym, symbols=symbols,
            cost_model=cost_model,
            collect_seconds=collect_seconds,
            uniqueness_seconds=uniqueness_seconds,
            cv_seconds=cv_seconds,
            weights=weights, effective_n=effective_n,
            reject_reason=reject_reason,
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
    X_s_p, y_p, w_p = _ensure_all_classes(X_s, y, weights)
    final_clf.fit(X_s_p, y_p, sample_weight=w_p, verbose=False)
    final_seconds = (datetime.now() - t3).total_seconds()

    from ml.models import IntegratedMLModel
    feature_names = IntegratedMLModel(brain=None, commentary_system=None).feature_names

    config_block = {
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
        "labeling": {
            "method": "triple_barrier",
            "pt_mult": args.pt_mult,
            "sl_mult": args.sl_mult,
            "vertical_max_bars": int(round(args.vertical_mult_duration * args.max_holding)),
            "vertical_mult_expected_duration": args.vertical_mult_duration,
            "min_ret_atr_mult": args.min_ret_atr_mult,
            "bar_type": args.bar_type,
            "bar_type_deferred": list(DEFAULT_DEFERRED_BAR_TYPES),
            "event_source": args.event_source,
            "cusum_h": args.cusum_h,
        },
        "bootstrap": {
            "method": "sequential" if args.sequential_bootstrap else "subsample_0.8",
            "iterations": args.bootstrap_iterations,
            "seed": 42,
        },
    }
    bundle = {
        "version": 2,
        "classifier": final_clf,
        "scaler": scaler,
        "feature_names": feature_names,
        "label_map": {0: "SELL", 1: "HOLD", 2: "BUY"},
        "config": config_block,
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
    label_dist_extended = dict(label_dist)
    label_dist_extended["dropped_low_magnitude"] = int(drop_acc.get("dropped_low_magnitude", 0))
    label_dist_extended["dropped_no_barrier_hit"] = int(drop_acc.get("dropped_no_barrier_hit", 0))
    label_dist_extended["candidate_events"] = int(drop_acc.get("candidates", 0))

    sample_uniqueness = {
        "mean": float(uniqueness.mean()),
        "median": float(np.median(uniqueness)),
        "min": float(uniqueness.min()),
        "effective_sample_size": effective_n,
        "n_samples": int(len(uniqueness)),
    }

    regression_block = _regression_check(args.regression_check, agg) if args.regression_check else None

    report = {
        "timestamp": datetime.now().isoformat(),
        "model_path": args.model_path,
        "config": bundle["config"],
        "symbols": symbols,
        "symbol_count": len(symbols),
        "per_symbol_samples": per_sym,
        "label_distribution": label_dist_extended,
        "weighting": {
            "mean": float(weights.mean()),
            "median": float(np.median(weights)),
            "min": float(weights.min()),
            "max": float(weights.max()),
            "effective_sample_size": effective_n,
            "compression_ratio": round(effective_n / len(weights), 4),
        },
        "sample_uniqueness": sample_uniqueness,
        "iid_diagnostics": iid_block,
        "bootstrap_stability": bootstrap_block,
        "timing_seconds": {
            "collection": round(collect_seconds, 1),
            "uniqueness": round(uniqueness_seconds, 1),
            "cv": round(cv_seconds, 1),
            "final_fit": round(final_seconds, 1),
        },
        "cv_summary": cv_summary,
        "feature_importance": importances,
    }
    if regression_block is not None:
        report["p2_regression_check"] = regression_block
    Path(args.report_path).write_text(json.dumps(report, indent=2, default=str))
    logger.info("report_written path=%s", args.report_path)
    return 0


def _regression_check(baseline_path: str, agg: dict[str, Any]) -> dict[str, Any]:
    """Warn-only diff against the four P2 anchor metrics.

    Per the user's P4 override, P2 economic metrics are NOT a hard-reject
    gate; they are reported here so the diff table can flag drift without
    blocking a merge.
    """
    try:
        baseline = json.loads(Path(baseline_path).read_text())
    except Exception as exc:  # pragma: no cover — bad path is the caller's bug
        logger.warning("regression_check_skipped baseline=%s err=%s", baseline_path, exc)
        return {"baseline_path": baseline_path, "error": str(exc)}
    base_agg = baseline.get("cv_summary", {}).get("aggregate", {})

    def _diff(key: str, *, lower_is_better: bool = False) -> dict[str, Any]:
        before = base_agg.get(key)
        after = agg.get(key)
        if before is None or after is None:
            return {"before": before, "after": after, "delta": None, "warn": False}
        delta = after - before
        warn = (delta < 0) if not lower_is_better else (delta > 0)
        return {"before": before, "after": after, "delta": delta, "warn": bool(warn)}

    diffs = {
        "median_profit_factor": _diff("median_profit_factor"),
        "median_expectancy_r": _diff("median_expectancy_r"),
        "median_absolute_cost_per_trade_r": _diff(
            "median_absolute_cost_per_trade_r", lower_is_better=True,
        ),
        "n_folds_with_undefined_cost_drag": _diff(
            "n_folds_with_undefined_cost_drag", lower_is_better=True,
        ),
    }
    any_warn = any(d.get("warn") for d in diffs.values())
    if any_warn:
        logger.warning("p2_regression_warn baseline=%s diffs=%s",
                       baseline_path, {k: v["delta"] for k, v in diffs.items()})
    return {
        "baseline_path": baseline_path,
        "warn_only": True,
        "any_warning": any_warn,
        "diffs": diffs,
    }


if __name__ == "__main__":
    sys.exit(main())
