"""Train the P1 meta-labeling filter on top of the P4 primary classifier.

López de Prado AFML §3.6-3.7. The primary (ml_model_v2.pkl) decides side;
the meta-classifier decides go/pass on those primary fires using the same
triple-barrier outcome the P4 model was trained against.

User-bound clarifications baked in:
  1. Operating threshold is picked on the **same chronological tail**
     used for isotonic calibration. Cost-adjusted Sortino objective with
     P2/P4 cost anchor (0.32 R/trade). Per-fold threshold reported;
     disagreement >0.20 across folds flagged but not a hard reject.
  2. Sample uniqueness is recomputed on the meta-event subset (post-fire
     events only). P4 weights are NOT reused.
  3. Cost model imported from ml.costs (single source of truth with
     P2/P4); per-row cost_in_r computed by the same formula.

Outputs:
    ml_meta_model_p1.pkl              — trained meta bundle
    backtest_results/P1_after.json    — full P1 report

Usage:
    python train_meta_model_p1.py \\
        --primary-bundle ml_model_v2.pkl \\
        --report-path backtest_results/P1_after.json
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

import hashlib

from ml.costs import CostModel, cost_in_r as _cost_in_r
from ml.events import cusum_filter
from ml.iid_diagnostics import compute_iid_diagnostics, weighted_residuals
from ml.labels import compute_uniqueness, triple_barrier_labels
from ml.meta_labels import apply_meta_triple_barrier
from ml.meta_model import (
    MetaThresholdRejection,
    P2_P4_COST_ANCHOR_R,
    calibration_deciles,
    compute_brier,
    evaluate_p1_gates,
    pick_threshold_on_calibration_tail,
    select_primary_fires,
    threshold_sweep_meta,
    train_meta_fold,
)
from ml.primary_oos_harness import refit_primary_on_fold
from ml.purged_cv import PurgedKFold
from train_ml_model import DEFAULT_DSN, top_symbols_by_volume
from train_ml_model_v2 import _compute_raw_atr


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_meta_p1")


DEFAULT_THRESHOLD_SWEEP = [
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
]
DEFAULT_FOLDS = 5
DEFAULT_DAYS = 60
DEFAULT_TOP_N = 10
DEFAULT_FREQUENCY = 5
DEFAULT_EMBARGO_PCT = 0.02  # 2× the P4 0.01 — honors "embargo = 2 × vertical"
DEFAULT_THRESHOLD_DISAGREEMENT_FLAG_GATE = 0.20  # report-only flag


def _make_provider(dsn: str) -> Any:
    """Factory — overridable in tests."""
    from data_providers.postgres import PostgresDataProvider
    return PostgresDataProvider(dsn)


# IntegratedMLModel imported lazily so tests can monkeypatch it.
try:  # pragma: no cover — import guarded for offline tests
    from ml.models import IntegratedMLModel
except Exception:  # pragma: no cover
    IntegratedMLModel = None  # type: ignore[assignment]


# ---- Sample collection ------------------------------------------------------


def collect_p1_samples(
    *,
    symbols: list[str],
    days: int,
    frequency: int,
    dsn: str,
    primary_bundle: dict[str, Any],
    cusum_h: float,
    decision_threshold_override: float | None,
) -> dict[str, Any]:
    """Run primary on each CUSUM event; meta-label the fires.

    Returns aligned arrays for the meta-training set: features, binary y
    (primary's bet hit pt before sl?), realized strategy return, event
    times, touch times, sides, symbols, and primary's max-proba per fire.
    """
    cfg = primary_bundle["config"]
    pt_mult = float(cfg["pt_mult"])
    sl_mult = float(cfg["sl_mult"])
    vertical = int(cfg["labeling"]["vertical_max_bars"])
    label_map = primary_bundle.get("label_map", {0: "SELL", 1: "HOLD", 2: "BUY"})
    decision_threshold = (
        float(decision_threshold_override)
        if decision_threshold_override is not None
        else float(cfg.get("decision_threshold", 0.55))
    )

    primary_clf = primary_bundle["classifier"]
    primary_scaler = primary_bundle["scaler"]
    feature_names = list(primary_bundle["feature_names"])

    provider = _make_provider(dsn)
    feature_model = IntegratedMLModel(brain=None, commentary_system=None)

    period_type = "day" if days < 30 else "month"
    period = days if period_type == "day" else max(1, days // 30)

    rows_X: list[np.ndarray] = []
    rows_y: list[int] = []
    rows_ret: list[float] = []
    rows_event_time: list[pd.Timestamp] = []
    rows_touch_time: list[pd.Timestamp] = []
    rows_side: list[int] = []
    rows_symbol: list[str] = []
    rows_max_proba: list[float] = []
    per_symbol_fires: dict[str, int] = {}
    per_symbol_events: dict[str, int] = {}

    for symbol in symbols:
        try:
            df = provider.get_market_data(
                symbol, period_type=period_type, period=period,
                frequency_type="minute", frequency=frequency,
            )
        except Exception as exc:
            logger.error("load_failed symbol=%s err=%s", symbol, exc)
            continue
        if df.empty or len(df) < vertical + 100:
            continue

        feat_df = feature_model.feature_extractor.batch_extract(df)
        if feat_df.empty:
            continue
        atr = _compute_raw_atr(df, window=14)

        # Restrict to bars that have a full forward vertical-barrier window.
        last_safe = df.index[-1 - vertical]
        usable_idx = feat_df.index[feat_df.index <= last_safe]

        # CUSUM events on close, intersected with usable bars.
        ev = cusum_filter(df["Close"], h=cusum_h)
        ev = ev.intersection(usable_idx)
        if len(ev) == 0:
            per_symbol_events[symbol] = 0
            per_symbol_fires[symbol] = 0
            continue
        per_symbol_events[symbol] = int(len(ev))

        # Drop events with missing features or non-positive ATR.
        feat_at_ev = feat_df.reindex(ev)[feature_names]
        atr_at_ev = atr.reindex(ev)
        valid = feat_at_ev.notna().all(axis=1) & (atr_at_ev > 0)
        feat_at_ev = feat_at_ev.loc[valid]
        ev = pd.DatetimeIndex(feat_at_ev.index)
        if len(ev) == 0:
            per_symbol_fires[symbol] = 0
            continue

        # Run the primary on every event.
        X_at_ev = feat_at_ev.to_numpy()
        proba = primary_clf.predict_proba(primary_scaler.transform(X_at_ev))

        fires = select_primary_fires(
            primary_proba=proba,
            primary_decision_threshold=decision_threshold,
            label_map=label_map,
        )
        n_fires = int(fires["mask"].sum())
        if n_fires == 0:
            per_symbol_fires[symbol] = 0
            continue

        fire_times = ev[fires["mask"]]
        sides = fires["sides"]
        events_df = pd.DataFrame({"side": sides}, index=fire_times)

        labels = apply_meta_triple_barrier(
            prices=df["Close"], events=events_df, atr=atr,
            pt_mult=pt_mult, sl_mult=sl_mult, max_holding=vertical,
        )
        # apply_meta_triple_barrier preserves the input event order.
        X_fire = X_at_ev[fires["mask"]]
        sides_arr = sides
        max_p_arr = fires["max_proba"]
        for i, ts in enumerate(labels.index):
            rows_X.append(X_fire[i])
            rows_y.append(int(labels["bin"].iloc[i]))
            rows_ret.append(float(labels["ret"].iloc[i]))
            rows_event_time.append(ts)
            rows_touch_time.append(pd.Timestamp(labels["touch_time"].iloc[i]))
            rows_side.append(int(sides_arr[i]))
            rows_symbol.append(symbol)
            rows_max_proba.append(float(max_p_arr[i]))
        per_symbol_fires[symbol] = int(len(labels))

    if not rows_X:
        return {
            "X": np.array([]).reshape(0, 0),
            "y": np.array([], dtype=np.int64),
            "ret": np.array([], dtype="float64"),
            "event_times": pd.DatetimeIndex([]),
            "touch_times": pd.DatetimeIndex([]),
            "sides": np.array([], dtype=np.int8),
            "symbols": np.array([], dtype=object),
            "max_proba": np.array([], dtype="float64"),
            "per_symbol_fires": per_symbol_fires,
            "per_symbol_events": per_symbol_events,
        }

    return {
        "X": np.vstack(rows_X),
        "y": np.asarray(rows_y, dtype=np.int64),
        "ret": np.asarray(rows_ret, dtype="float64"),
        "event_times": pd.DatetimeIndex(rows_event_time),
        "touch_times": pd.DatetimeIndex(rows_touch_time),
        "sides": np.asarray(rows_side, dtype=np.int8),
        "symbols": np.asarray(rows_symbol, dtype=object),
        "max_proba": np.asarray(rows_max_proba, dtype="float64"),
        "per_symbol_fires": per_symbol_fires,
        "per_symbol_events": per_symbol_events,
    }


# ---- Nested-CV candidate collection ----------------------------------------


def collect_p1_candidates_nested(
    *,
    symbols: list[str],
    days: int,
    frequency: int,
    dsn: str,
    primary_bundle: dict[str, Any],
    cusum_h: float,
) -> dict[str, Any]:
    """Gather **all** CUSUM events with P4-style triple-barrier labels.

    Unlike :func:`collect_p1_samples`, this does NOT run the released
    primary on the events. The candidate set covers every CUSUM event
    with valid features + ATR + a non-degenerate triple-barrier outcome,
    matching the data the P4 trainer itself sees. Per-fold the harness
    will refit a primary on train rows and predict on test rows; the
    OOS predictions filter the candidates down to that fold's fires.
    """
    cfg = primary_bundle["config"]
    pt_mult = float(cfg["pt_mult"])
    sl_mult = float(cfg["sl_mult"])
    vertical = int(cfg["labeling"]["vertical_max_bars"])
    min_ret_atr = float(cfg["labeling"]["min_ret_atr_mult"])
    feature_names = list(primary_bundle["feature_names"])

    provider = _make_provider(dsn)
    feature_model = IntegratedMLModel(brain=None, commentary_system=None)
    period_type = "day" if days < 30 else "month"
    period = days if period_type == "day" else max(1, days // 30)

    rows_X: list[np.ndarray] = []
    rows_y_primary: list[int] = []   # 0=SELL, 1=HOLD, 2=BUY (P4 schema)
    rows_ret: list[float] = []
    rows_event_time: list[pd.Timestamp] = []
    rows_touch_time: list[pd.Timestamp] = []
    rows_symbol: list[str] = []
    per_symbol_events: dict[str, int] = {}

    for symbol in symbols:
        try:
            df = provider.get_market_data(
                symbol, period_type=period_type, period=period,
                frequency_type="minute", frequency=frequency,
            )
        except Exception as exc:
            logger.error("load_failed symbol=%s err=%s", symbol, exc)
            continue
        if df.empty or len(df) < vertical + 100:
            continue

        feat_df = feature_model.feature_extractor.batch_extract(df)
        if feat_df.empty:
            continue
        atr = _compute_raw_atr(df, window=14)
        last_safe = df.index[-1 - vertical]
        usable_idx = feat_df.index[feat_df.index <= last_safe]

        ev = cusum_filter(df["Close"], h=cusum_h)
        ev = ev.intersection(usable_idx)
        if len(ev) == 0:
            per_symbol_events[symbol] = 0
            continue

        feat_at_ev = feat_df.reindex(ev)[feature_names]
        atr_at_ev = atr.reindex(ev)
        valid = feat_at_ev.notna().all(axis=1) & (atr_at_ev > 0)
        feat_at_ev = feat_at_ev.loc[valid]
        ev = pd.DatetimeIndex(feat_at_ev.index)
        if len(ev) == 0:
            per_symbol_events[symbol] = 0
            continue

        # P4-style triple-barrier labels — same schema as ml_model_v2.pkl.
        events_df = pd.DataFrame(
            {"vertical": [vertical] * len(ev)}, index=ev,
        )
        labels = triple_barrier_labels(
            events=events_df, close=df["Close"], atr=atr,
            pt_sl=(pt_mult, sl_mult),
            vertical_barrier=vertical,
            min_ret=min_ret_atr,
        )
        if labels.empty:
            per_symbol_events[symbol] = 0
            continue

        feat_kept = feat_at_ev.reindex(labels.index)
        # Map {-1, 0, +1} → {0, 1, 2} (P4 label_map).
        y_primary = (labels["bin"].astype(int).to_numpy() + 1)
        for i, ts in enumerate(labels.index):
            rows_X.append(feat_kept.iloc[i].to_numpy())
            rows_y_primary.append(int(y_primary[i]))
            rows_ret.append(float(labels["ret"].iloc[i]))
            rows_event_time.append(ts)
            rows_touch_time.append(pd.Timestamp(labels["t1"].iloc[i]))
            rows_symbol.append(symbol)
        per_symbol_events[symbol] = int(len(labels))

    if not rows_X:
        return {
            "X": np.array([]).reshape(0, 0),
            "y_primary": np.array([], dtype=np.int64),
            "ret": np.array([], dtype="float64"),
            "event_times": pd.DatetimeIndex([]),
            "touch_times": pd.DatetimeIndex([]),
            "symbols": np.array([], dtype=object),
            "per_symbol_events": per_symbol_events,
        }
    return {
        "X": np.vstack(rows_X),
        "y_primary": np.asarray(rows_y_primary, dtype=np.int64),
        "ret": np.asarray(rows_ret, dtype="float64"),
        "event_times": pd.DatetimeIndex(rows_event_time),
        "touch_times": pd.DatetimeIndex(rows_touch_time),
        "symbols": np.asarray(rows_symbol, dtype=object),
        "per_symbol_events": per_symbol_events,
    }


def _r_multiples_from_primary_labels(
    *, y: np.ndarray, ret: np.ndarray, pt_mult: float, sl_mult: float,
) -> np.ndarray:
    """Same conversion as train_ml_model_v2._r_multiples_from_labels."""
    y = np.asarray(y, dtype=np.int64)
    ret = np.asarray(ret, dtype="float64")
    out = np.zeros_like(ret)
    out[y == 2] = pt_mult / sl_mult
    out[y == 0] = -1.0
    barrier_pct_proxy = sl_mult / max(sl_mult + pt_mult, 1.0)
    timeout_mask = y == 1
    if timeout_mask.any() and barrier_pct_proxy > 0:
        out[timeout_mask] = ret[timeout_mask] / barrier_pct_proxy
    return out


# ---- Cost (single-source from ml.costs) ------------------------------------


def _cost_r_per_row(
    *,
    n: int, sl_mult: float, cost_model: CostModel, participation: float,
) -> np.ndarray:
    """Same proxy approach as train_ml_model_v2._cost_r_from_constants — uses
    the canonical cost_in_r formula from ml/costs.py (single source of truth
    with P2/P4)."""
    return _cost_in_r(
        entry_price=np.full(n, 100.0),
        atr=np.full(n, sl_mult),
        sl_mult=1.0,
        participation=np.full(n, participation),
        cost_model=cost_model,
    )


# ---- Purged-CV training loop -----------------------------------------------


def run_purged_cv_meta(
    *,
    X: np.ndarray, y: np.ndarray, weights: np.ndarray,
    gross_r: np.ndarray, cost_r: np.ndarray,
    touch_times: pd.Series,
    n_splits: int, embargo_pct: float, threshold_sweep: list[float],
    seed: int = 42,
) -> dict[str, Any]:
    """Per-fold meta-trainer with isotonic calibration and tail-based
    threshold pick.

    Returns OOS calibrated proba (full length, NaN where not covered),
    per-fold AR(1) on weighted binary residuals, per-fold picked
    threshold, per-fold tail proba/y/picked-threshold for inspection.
    """
    cv = PurgedKFold(n_splits=n_splits, touch_times=touch_times, embargo_pct=embargo_pct)
    n = len(y)
    oos_proba = np.full(n, np.nan, dtype="float64")
    per_fold_ar1: list[float] = []
    per_fold_thresholds: list[float | None] = []
    fold_detail: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            logger.warning("fold %d empty (train=%d test=%d)",
                           fold_idx, len(train_idx), len(test_idx))
            continue

        out = train_meta_fold(
            X_train=X[train_idx], y_train=y[train_idx],
            w_train=weights[train_idx],
            X_test=X[test_idx],
            calibration_tail_frac=0.25,
            seed=seed + fold_idx,
        )
        oos_proba[test_idx] = out["oos_calibrated_proba"]

        # Per-fold threshold pick on the calibration tail (same rows
        # used for isotonic calibration, never on OOS).
        tail_y = out["tail_y"]
        tail_proba = out["tail_proba_calibrated"]
        tail_pos = out["tail_indices"]  # positions within train_idx
        if len(tail_proba) > 0 and len(tail_pos) == len(tail_proba):
            tail_global = train_idx[tail_pos]
            tail_gross = gross_r[tail_global]
            tail_cost = cost_r[tail_global]
            try:
                chosen_row, why, sweep_tail = pick_threshold_on_calibration_tail(
                    tail_proba=tail_proba, tail_y=tail_y,
                    tail_gross_r=tail_gross, tail_cost_r=tail_cost,
                    thresholds=threshold_sweep,
                    cost_per_trade_r=P2_P4_COST_ANCHOR_R,
                    tiebreak_within_pct=0.05,
                )
                fold_threshold = float(chosen_row["threshold"])
                why_msg = why
            except MetaThresholdRejection as exc:
                fold_threshold = None
                why_msg = f"reject: {exc}"
                sweep_tail = []
        else:
            fold_threshold = None
            why_msg = "no calibration tail available"
            sweep_tail = []

        per_fold_thresholds.append(fold_threshold)

        # Per-fold AR(1) on weighted binary residuals — the gate input.
        residuals = weighted_residuals(
            y[test_idx], out["oos_calibrated_proba"], weights[test_idx],
        )
        iid = compute_iid_diagnostics(residuals)
        per_fold_ar1.append(float(iid.ar1_residual_autocorr))

        fold_detail.append({
            "fold": fold_idx,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "tail_size": int(len(tail_proba)),
            "ar1_residual_autocorr": float(iid.ar1_residual_autocorr),
            "ljung_box_p_value": (
                float(iid.ljung_box_p_value)
                if iid.ljung_box_p_value is not None else None
            ),
            "tail_threshold": fold_threshold,
            "tail_threshold_rationale": why_msg,
            "tail_sweep": sweep_tail,
        })
        logger.info(
            "fold %d/%d train=%d test=%d tail=%d ar1=%.3f thr=%s",
            fold_idx, n_splits, len(train_idx), len(test_idx),
            len(tail_proba),
            float(iid.ar1_residual_autocorr),
            f"{fold_threshold:.2f}" if fold_threshold is not None else "—",
        )

    finite = [a for a in per_fold_ar1 if np.isfinite(a)]
    median_ar1 = float(np.median([abs(a) for a in finite])) if finite else 0.0
    max_ar1 = float(np.max([abs(a) for a in finite])) if finite else 0.0
    return {
        "oos_proba": oos_proba,
        "per_fold_ar1": per_fold_ar1,
        "median_per_fold_ar1": median_ar1,
        "max_per_fold_ar1": max_ar1,
        "fold_detail": fold_detail,
        "per_fold_thresholds": per_fold_thresholds,
    }


def run_nested_cv_meta(
    *,
    X: np.ndarray,
    y_primary: np.ndarray,
    ret: np.ndarray,
    event_times: pd.DatetimeIndex,
    touch_times: pd.DatetimeIndex,
    symbols: np.ndarray,
    primary_bundle: dict[str, Any],
    cost_model: CostModel,
    participation: float,
    n_splits: int,
    embargo_pct: float,
    threshold_sweep: list[float],
    seed: int = 42,
) -> dict[str, Any]:
    """AFML §7.4 nested-CV meta-trainer.

    Per fold of the candidate-event set:
      1. Refit a fresh primary on the fold's train rows using
         :func:`refit_primary_on_fold` with config from ml_model_v2.pkl.
      2. Predict OOS softprob on the fold's test rows.
      3. Apply :func:`select_primary_fires` to the OOS predictions →
         the OOS fire set for this fold.
      4. Build meta features = X_test[fire_mask], meta labels via
         :func:`apply_meta_triple_barrier` on those fires (recomputing
         strict-subset uniqueness on the fire set).
      5. Train a binary meta-classifier with isotonic calibration on
         the chronological tail (same recipe as the leaky run).
      6. Pick the operating threshold on that fold's calibration tail
         (cost-adjusted Sortino, P2/P4 anchor, tie-break high).

    Concatenates OOS-meta probabilities across folds for the gate
    evaluation.
    """
    cfg = primary_bundle["config"]
    pt_mult = float(cfg["pt_mult"])
    sl_mult = float(cfg["sl_mult"])
    vertical = int(cfg["labeling"]["vertical_max_bars"])
    label_map = primary_bundle.get("label_map", {0: "SELL", 1: "HOLD", 2: "BUY"})
    decision_threshold = float(cfg.get("decision_threshold", 0.55))

    # Cost-aware sample weights for primary refit:
    # need r-multiples + cost_in_r aligned with X.
    sample_r_primary = _r_multiples_from_primary_labels(
        y=y_primary, ret=ret, pt_mult=pt_mult, sl_mult=sl_mult,
    )
    sample_cost_primary = _cost_r_per_row(
        n=len(y_primary), sl_mult=sl_mult, cost_model=cost_model,
        participation=participation,
    )

    # Uniqueness for the FULL candidate set — used by the primary's
    # sequential bootstrap (P4 reuses uniqueness across all events).
    candidate_touch_series = pd.Series(
        touch_times.to_numpy(), index=event_times,
    )
    candidate_labels_for_w = pd.DataFrame(
        {"touch_time": candidate_touch_series.values},
        index=candidate_touch_series.index,
    )
    candidate_master_idx = pd.DatetimeIndex(
        np.unique(np.concatenate([
            event_times.to_numpy(), touch_times.to_numpy(),
        ]))
    ).sort_values()
    candidate_uniqueness = compute_uniqueness(
        candidate_labels_for_w, candidate_master_idx,
        groups=pd.Series(symbols),
    ).to_numpy()
    candidate_uniqueness = np.clip(candidate_uniqueness, 1e-6, 1.0)

    cv = PurgedKFold(
        n_splits=n_splits, touch_times=candidate_touch_series,
        embargo_pct=embargo_pct,
    )

    # Accumulators across folds.
    oos_meta_proba: list[np.ndarray] = []
    oos_meta_y: list[np.ndarray] = []
    oos_meta_gross_r: list[np.ndarray] = []
    oos_meta_cost_r: list[np.ndarray] = []
    oos_meta_weights: list[np.ndarray] = []
    per_fold_ar1: list[float] = []
    per_fold_thresholds: list[float | None] = []
    per_fold_fire_counts: list[int] = []
    fold_detail: list[dict[str, Any]] = []
    all_max_proba_at_fire: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            logger.warning("fold %d empty (train=%d test=%d)",
                           fold_idx, len(train_idx), len(test_idx))
            continue

        # ---- Stage 1: refit primary, predict OOS on test fold ----
        primary_proba_oos = refit_primary_on_fold(
            X_train=X[train_idx],
            y_train=y_primary[train_idx],
            weights_train=candidate_uniqueness[train_idx],
            X_test=X[test_idx],
            event_times_train=event_times[train_idx],
            touch_times_train=touch_times[train_idx],
            sample_r_multiples_train=sample_r_primary[train_idx],
            sample_cost_r_train=sample_cost_primary[train_idx],
            primary_config=cfg,
            bootstrap_iterations=5,
            use_sequential_bootstrap=True,
            seed=seed + fold_idx,
        )

        # ---- Stage 2: identify OOS fires within this test fold ----
        fires = select_primary_fires(
            primary_proba=primary_proba_oos,
            primary_decision_threshold=decision_threshold,
            label_map=label_map,
        )
        n_fires = int(fires["mask"].sum())
        per_fold_fire_counts.append(n_fires)
        if n_fires < 30:
            logger.warning(
                "fold %d only %d OOS fires — skipping meta-CV fold",
                fold_idx, n_fires,
            )
            fold_detail.append({
                "fold": fold_idx, "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "oos_fires": n_fires,
                "skipped": True,
                "skip_reason": "fewer than 30 OOS fires",
            })
            continue

        fire_test_pos = np.where(fires["mask"])[0]   # positions within test fold
        fire_global_pos = test_idx[fire_test_pos]    # positions in full X
        fire_event_times = event_times[fire_global_pos]
        fire_touch_times = touch_times[fire_global_pos]
        fire_symbols = symbols[fire_global_pos]
        fire_max_proba = fires["max_proba"]
        all_max_proba_at_fire.extend(fire_max_proba.tolist())

        # ---- Stage 3: meta-target via apply_meta_triple_barrier ----
        # We re-derive prices/atr per symbol for the fires. Cheaper
        # path: reuse the candidate-level outcomes already in y_primary
        # / ret. y_primary maps {SELL=0, HOLD=1, BUY=2}; meta-bin = 1
        # iff side matches the realized barrier.
        sides = fires["sides"]
        realized_barrier = y_primary[fire_global_pos]  # 0=SELL hit, 2=BUY hit
        # Long (side=+1) wins iff barrier is BUY (2). Short (side=-1)
        # wins iff barrier is SELL (0). Time-expiry (1) is a meta loss.
        meta_y = np.where(
            (sides == 1) & (realized_barrier == 2), 1,
            np.where((sides == -1) & (realized_barrier == 0), 1, 0),
        ).astype(np.int64)

        # Strategy-perspective return and gross-R (signed).
        meta_ret = ret[fire_global_pos] * sides
        barrier_pct_proxy = sl_mult / 100.0
        meta_gross_r = meta_ret / barrier_pct_proxy
        meta_cost_r = sample_cost_primary[fire_global_pos]

        # ---- Recompute uniqueness on the OOS-fire subset ----
        fire_touch_series = pd.Series(
            fire_touch_times.to_numpy(), index=fire_event_times,
        )
        fire_labels_for_w = pd.DataFrame(
            {"touch_time": fire_touch_series.values},
            index=fire_touch_series.index,
        )
        fire_master_idx = pd.DatetimeIndex(
            np.unique(np.concatenate([
                fire_event_times.to_numpy(),
                fire_touch_times.to_numpy(),
            ]))
        ).sort_values()
        meta_weights = compute_uniqueness(
            fire_labels_for_w, fire_master_idx,
            groups=pd.Series(fire_symbols),
        ).to_numpy()
        meta_weights = np.clip(meta_weights, 1e-6, 1.0)

        # ---- Stage 4: train meta on this fold's OOS fires ----
        # Sub-split the fires chronologically: 75% inner-train + 25%
        # inner-tail for isotonic calibration AND threshold pick. The
        # outer test set was already used to GENERATE these fires;
        # there is no further "outer test" within this fold.
        fold_meta_X = X[fire_global_pos]
        order = np.argsort(fire_event_times.to_numpy())
        fold_meta_X = fold_meta_X[order]
        meta_y = meta_y[order]
        meta_gross_r = meta_gross_r[order]
        meta_cost_r = meta_cost_r[order]
        meta_weights = meta_weights[order]
        ordered_fire_symbols = fire_symbols[order]
        ordered_fire_event_times = fire_event_times[order]

        # Inner split: 75% train + 25% tail (the tail does double duty
        # for isotonic AND threshold pick AND meta-OOS evaluation).
        cut = int(round(0.75 * len(meta_y)))
        if cut < 30 or len(meta_y) - cut < 30:
            fold_detail.append({
                "fold": fold_idx, "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "oos_fires": n_fires,
                "skipped": True,
                "skip_reason": "inner split too small",
            })
            continue

        fold = train_meta_fold(
            X_train=fold_meta_X[:cut], y_train=meta_y[:cut],
            w_train=meta_weights[:cut],
            X_test=fold_meta_X[cut:],
            calibration_tail_frac=0.25,
            seed=seed + 1000 + fold_idx,
        )
        oos_test_proba = fold["oos_calibrated_proba"]

        # Threshold pick on inner-train's calibration tail (which is
        # the last 25% of the inner-train slice — strictly earlier
        # than the outer-test rows used here).
        try:
            chosen_row, why, _ = pick_threshold_on_calibration_tail(
                tail_proba=fold["tail_proba_calibrated"],
                tail_y=fold["tail_y"],
                tail_gross_r=meta_gross_r[fold["tail_indices"]],
                tail_cost_r=meta_cost_r[fold["tail_indices"]],
                thresholds=threshold_sweep,
                cost_per_trade_r=P2_P4_COST_ANCHOR_R,
                tiebreak_within_pct=0.05,
            )
            fold_threshold = float(chosen_row["threshold"])
            why_msg = why
        except MetaThresholdRejection as exc:
            fold_threshold = None
            why_msg = f"reject: {exc}"

        # Per-fold AR(1) on the meta-OOS test slice.
        meta_y_oos = meta_y[cut:]
        meta_w_oos = meta_weights[cut:]
        residuals = weighted_residuals(meta_y_oos, oos_test_proba, meta_w_oos)
        iid = compute_iid_diagnostics(residuals)
        per_fold_ar1.append(float(iid.ar1_residual_autocorr))
        per_fold_thresholds.append(fold_threshold)

        oos_meta_proba.append(oos_test_proba)
        oos_meta_y.append(meta_y_oos)
        oos_meta_gross_r.append(meta_gross_r[cut:])
        oos_meta_cost_r.append(meta_cost_r[cut:])
        oos_meta_weights.append(meta_w_oos)

        fold_detail.append({
            "fold": fold_idx,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "oos_fires": n_fires,
            "meta_inner_train": int(cut),
            "meta_inner_test": int(len(meta_y) - cut),
            "ar1_residual_autocorr": float(iid.ar1_residual_autocorr),
            "ljung_box_p_value": (
                float(iid.ljung_box_p_value)
                if iid.ljung_box_p_value is not None else None
            ),
            "tail_threshold": fold_threshold,
            "tail_threshold_rationale": why_msg,
            "meta_inner_win_rate": float(meta_y_oos.mean())
                if len(meta_y_oos) > 0 else None,
        })
        logger.info(
            "fold %d/%d primary_train=%d test=%d fires=%d "
            "meta_inner=%d/%d ar1=%.3f thr=%s",
            fold_idx, n_splits, len(train_idx), len(test_idx), n_fires,
            cut, len(meta_y) - cut,
            float(iid.ar1_residual_autocorr),
            f"{fold_threshold:.2f}" if fold_threshold is not None else "—",
        )

    if not oos_meta_proba:
        return {
            "oos_proba": np.array([]),
            "oos_y": np.array([], dtype=np.int64),
            "oos_gross_r": np.array([]),
            "oos_cost_r": np.array([]),
            "oos_weights": np.array([]),
            "per_fold_ar1": [],
            "median_per_fold_ar1": 0.0,
            "max_per_fold_ar1": 0.0,
            "per_fold_thresholds": [],
            "per_fold_fire_counts": per_fold_fire_counts,
            "fold_detail": fold_detail,
            "all_max_proba_at_fire": all_max_proba_at_fire,
        }

    oos_proba_full = np.concatenate(oos_meta_proba)
    oos_y_full = np.concatenate(oos_meta_y)
    oos_gross_full = np.concatenate(oos_meta_gross_r)
    oos_cost_full = np.concatenate(oos_meta_cost_r)
    oos_w_full = np.concatenate(oos_meta_weights)

    finite = [a for a in per_fold_ar1 if np.isfinite(a)]
    median_ar1 = float(np.median([abs(a) for a in finite])) if finite else 0.0
    max_ar1 = float(np.max([abs(a) for a in finite])) if finite else 0.0
    return {
        "oos_proba": oos_proba_full,
        "oos_y": oos_y_full,
        "oos_gross_r": oos_gross_full,
        "oos_cost_r": oos_cost_full,
        "oos_weights": oos_w_full,
        "per_fold_ar1": per_fold_ar1,
        "median_per_fold_ar1": median_ar1,
        "max_per_fold_ar1": max_ar1,
        "per_fold_thresholds": per_fold_thresholds,
        "per_fold_fire_counts": per_fold_fire_counts,
        "fold_detail": fold_detail,
        "all_max_proba_at_fire": all_max_proba_at_fire,
    }


def aggregate_per_fold_thresholds(
    per_fold: list[float | None],
    *,
    disagreement_flag_gate: float = DEFAULT_THRESHOLD_DISAGREEMENT_FLAG_GATE,
) -> dict[str, Any]:
    """Aggregate per-fold thresholds into a single deployment threshold.

    Median across folds. ``disagreement = max - min``; flagged when it
    exceeds the gate (visible in the report; not a hard reject).
    """
    finite = [t for t in per_fold if t is not None]
    if not finite:
        return {
            "per_fold": per_fold,
            "median": None,
            "min": None,
            "max": None,
            "disagreement": None,
            "disagreement_flag_gate": disagreement_flag_gate,
            "disagreement": True,
            "n_folds_with_threshold": 0,
        }
    arr = np.asarray(finite, dtype="float64")
    spread = float(arr.max() - arr.min())
    return {
        "per_fold": per_fold,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "disagreement_spread": spread,
        "disagreement_flag_gate": disagreement_flag_gate,
        "disagreement": bool(spread > disagreement_flag_gate),
        "n_folds_with_threshold": int(len(finite)),
    }


# ---- Cost-aware OOS evaluation at the deployment threshold -----------------


def evaluate_at_threshold(
    *,
    proba: np.ndarray, y: np.ndarray,
    gross_r: np.ndarray, cost_r: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute take-rate, expectancy, PF, win-rate, sortino at a single
    threshold using the OOS proba stitched across folds."""
    sweep = threshold_sweep_meta(
        proba=proba, y_true=y, gross_r=gross_r, cost_r=cost_r,
        thresholds=[threshold],
    )
    return sweep[0]


# ---- Argparse + main --------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sym = p.add_mutually_exclusive_group()
    sym.add_argument("--symbols", nargs="+")
    sym.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--days", type=int, default=DEFAULT_DAYS)
    p.add_argument("--frequency", type=int, default=DEFAULT_FREQUENCY)
    p.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p.add_argument("--embargo-pct", type=float, default=DEFAULT_EMBARGO_PCT)
    p.add_argument("--dsn", default=os.environ.get("POSTGRES_DSN", DEFAULT_DSN))
    p.add_argument("--primary-bundle", default="ml_model_v2.pkl")
    p.add_argument("--meta-bundle-path", default="ml_meta_model_p1.pkl")
    p.add_argument("--report-path", default="backtest_results/P1_after.json")
    p.add_argument("--baseline", default="backtest_results/P4_after.json")
    p.add_argument("--allow-rejection", action="store_true",
                   help="Still write the report on gate failure (always exits 3).")
    p.add_argument(
        "--threshold-sweep",
        default=",".join(f"{t:.2f}" for t in DEFAULT_THRESHOLD_SWEEP),
    )
    p.add_argument("--decision-threshold-override", type=float, default=None,
                   help="Override the primary's decision_threshold for this run.")
    p.add_argument("--cusum-h-override", type=float, default=None,
                   help="Override the primary bundle's CUSUM h for event generation.")
    p.add_argument("--nested-primary", action="store_true",
                   help="AFML §7.4 nested CV: refit a fresh primary on each "
                        "meta-fold's train rows for OOS evaluation. The "
                        "released ml_model_v2.pkl is not modified.")
    return p.parse_args()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _baseline_diff(baseline_path: str, report: dict[str, Any]) -> dict[str, Any]:
    try:
        base = json.loads(Path(baseline_path).read_text())
    except Exception as exc:
        return {"baseline_path": baseline_path, "error": str(exc)}
    base_agg = base.get("cv_summary", {}).get("aggregate", {})
    base_iid = base.get("iid_diagnostics", {})
    chosen = report.get("chosen_threshold") or {}
    rows = {
        "median_profit_factor": {
            "P4": base_agg.get("median_profit_factor"),
            "P1": chosen.get("profit_factor"),
        },
        "median_expectancy_r": {
            "P4": base_agg.get("median_expectancy_r"),
            "P1": chosen.get("expectancy_net_r"),
        },
        "take_rate": {
            "P4": "≈0.70 (decision_threshold 0.55, no meta-filter)",
            "P1": chosen.get("take_rate"),
        },
        "median_per_fold_ar1": {
            "P4": base_iid.get("median_per_fold_ar1"),
            "P1": report["iid_diagnostics"]["median_per_fold_ar1"],
        },
        "max_per_fold_ar1": {
            "P4": base_iid.get("max_per_fold_ar1"),
            "P1": report["iid_diagnostics"]["max_per_fold_ar1"],
        },
        "median_absolute_cost_per_trade_r": {
            "P4": base_agg.get("median_absolute_cost_per_trade_r"),
            "P1_anchor": P2_P4_COST_ANCHOR_R,
        },
    }
    return {"baseline_path": baseline_path, "rows": rows}


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _run_nested_path(
    *,
    args: argparse.Namespace,
    primary_bundle: dict[str, Any],
    primary_bundle_path: Path,
    sha_before: str,
    cusum_h: float,
    cost_model: CostModel,
    participation: float,
    sl_mult: float,
    symbols: list[str],
    coll: dict[str, Any],
    collect_seconds: float,
) -> int:
    """Tail of main() for the AFML §7.4 nested-CV path.

    Refits primary per meta-fold (in :func:`run_nested_cv_meta`),
    collects OOS-meta proba, evaluates gates, writes the report.
    Persists nothing from the refitted primaries. Verifies the
    released primary bundle's SHA256 is unchanged.
    """
    cfg = primary_bundle["config"]
    n_candidates = int(coll["X"].shape[0]) if coll["X"].size else 0
    if n_candidates == 0:
        logger.error("no_candidate_events — nothing to nest-train on")
        sha_after = _sha256_file(primary_bundle_path)
        _write_report(Path(args.report_path), {
            "timestamp": datetime.now().isoformat(),
            "primary_bundle": str(primary_bundle_path),
            "primary_bundle_sha256": {
                "before": sha_before, "after": sha_after,
                "unchanged": sha_before == sha_after,
            },
            "config": {"nested_cv": True},
            "rejected": True,
            "reject_reason": "no_candidate_events",
        })
        return 3

    # Sort chronologically.
    order = np.argsort(coll["event_times"].to_numpy())
    X = coll["X"][order]
    y_primary = coll["y_primary"][order]
    ret = coll["ret"][order]
    event_times = coll["event_times"][order]
    touch_times = coll["touch_times"][order]
    sym_groups = coll["symbols"][order]

    threshold_sweep = [float(t) for t in args.threshold_sweep.split(",")]

    nested = run_nested_cv_meta(
        X=X, y_primary=y_primary, ret=ret,
        event_times=event_times, touch_times=touch_times,
        symbols=sym_groups,
        primary_bundle=primary_bundle,
        cost_model=cost_model, participation=participation,
        n_splits=args.folds, embargo_pct=args.embargo_pct,
        threshold_sweep=threshold_sweep,
    )

    sha_after = _sha256_file(primary_bundle_path)
    primary_unchanged = sha_before == sha_after

    if nested["oos_proba"].size == 0:
        logger.error("nested CV produced no OOS-meta predictions")
        _write_report(Path(args.report_path), {
            "timestamp": datetime.now().isoformat(),
            "primary_bundle": str(primary_bundle_path),
            "primary_bundle_sha256": {
                "before": sha_before, "after": sha_after,
                "unchanged": primary_unchanged,
            },
            "config": {"nested_cv": True},
            "rejected": True,
            "reject_reason": "no_oos_meta_predictions",
            "n_candidate_events": n_candidates,
            "per_fold_fire_counts": nested["per_fold_fire_counts"],
        })
        return 3

    oos_proba = nested["oos_proba"]
    oos_y = nested["oos_y"]
    oos_gross_r = nested["oos_gross_r"]
    oos_cost_r = nested["oos_cost_r"]

    brier = compute_brier(oos_y, oos_proba)
    cal = calibration_deciles(oos_y, oos_proba)

    threshold_agreement = aggregate_per_fold_thresholds(
        nested["per_fold_thresholds"],
    )
    deployment_threshold = threshold_agreement.get("median")

    rejected = False
    reject_reason = ""
    chosen_row: dict[str, Any] = {}
    if deployment_threshold is None:
        rejected = True
        reject_reason = "no fold yielded a usable calibration-tail threshold"
    else:
        chosen_row = evaluate_at_threshold(
            proba=oos_proba, y=oos_y,
            gross_r=oos_gross_r, cost_r=oos_cost_r,
            threshold=deployment_threshold,
        )

    full_sweep = threshold_sweep_meta(
        proba=oos_proba, y_true=oos_y,
        gross_r=oos_gross_r, cost_r=oos_cost_r,
        thresholds=threshold_sweep,
    )

    if not rejected:
        gates = evaluate_p1_gates(
            brier=brier,
            calibration_max_dev=cal["max_abs_deviation"],
            chosen_threshold_row=chosen_row,
            per_fold_ar1_median=nested["median_per_fold_ar1"],
            per_fold_ar1_max=nested["max_per_fold_ar1"],
        )
        if not gates["all_pass"]:
            rejected = True
            failed = [
                k for k, v in gates.items()
                if isinstance(v, dict) and v.get("pass") is False
            ]
            reject_reason = "gates_failed: " + ",".join(failed)
    else:
        gates = {
            "brier": {"value": brier, "threshold": 0.22, "pass": brier < 0.22},
            "calibration": {"value": cal["max_abs_deviation"], "pass": False},
            "take_rate": {"pass": False},
            "profit_factor": {"pass": False},
            "expectancy": {"pass": False},
            "ar1": {
                "median": nested["median_per_fold_ar1"],
                "max": nested["max_per_fold_ar1"],
                "pass": (
                    nested["median_per_fold_ar1"] < 0.10
                    and nested["max_per_fold_ar1"] < 0.20
                ),
            },
            "all_pass": False,
        }

    # ---- Build report ---------------------------------------------------
    base_win_rate = float(oos_y.mean()) if len(oos_y) else 0.0
    label_dist = {
        "wins": int(oos_y.sum()),
        "losses": int((1 - oos_y).sum()),
        "base_win_rate": round(base_win_rate, 4),
    }
    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "primary_bundle": str(primary_bundle_path),
        "primary_bundle_sha256": {
            "before": sha_before,
            "after": sha_after,
            "unchanged": bool(primary_unchanged),
        },
        "config": {
            "nested_cv": True,
            "folds": args.folds, "embargo_pct": args.embargo_pct,
            "frequency": args.frequency, "days": args.days,
            "primary_decision_threshold": float(
                cfg.get("decision_threshold", 0.55)
            ),
            "primary_pt_mult": float(cfg["pt_mult"]),
            "primary_sl_mult": float(cfg["sl_mult"]),
            "primary_max_holding": int(cfg["max_holding"]),
            "primary_vertical_max_bars": int(cfg["labeling"]["vertical_max_bars"]),
            "cusum_h": cusum_h,
            "cost_model": {
                "fee_bps": cost_model.fee_bps,
                "slip_bps": cost_model.slip_bps,
                "impact_coef": cost_model.impact_coef,
                "participation": participation,
                "anchor_r_per_trade": P2_P4_COST_ANCHOR_R,
                "source": "ml.costs (single source of truth shared with P2/P4)",
            },
            "threshold_sweep": threshold_sweep,
            "tiebreak_within_pct": 0.05,
            "uniqueness": "recomputed on OOS-fire subset per fold",
            "nested_recipe": "AFML §7.4: refit primary per meta-fold; "
                             "OOS predictions used to identify fires; "
                             "meta trained on those OOS fires.",
        },
        "symbols": symbols,
        "n_candidate_events": n_candidates,
        "per_symbol_events": coll["per_symbol_events"],
        "per_fold_fire_counts": nested["per_fold_fire_counts"],
        "n_oos_meta_samples": int(len(oos_y)),
        "label_distribution": label_dist,
        "cv_summary": {
            "n_folds_executed": len([f for f in nested["fold_detail"]
                                     if not f.get("skipped")]),
            "fold_detail": nested["fold_detail"],
        },
        "calibration": {
            "brier": brier,
            "deciles": cal,
        },
        "threshold_sweep": full_sweep,
        "per_fold_thresholds": nested["per_fold_thresholds"],
        "threshold_agreement": threshold_agreement,
        "chosen_threshold": chosen_row,
        "deployment_threshold": deployment_threshold,
        "iid_diagnostics": {
            "per_fold_ar1": nested["per_fold_ar1"],
            "median_per_fold_ar1": nested["median_per_fold_ar1"],
            "max_per_fold_ar1": nested["max_per_fold_ar1"],
            "gate_thresholds": {"median_lt": 0.10, "max_lt": 0.20},
        },
        "gates": gates,
        "rejected": rejected,
        "reject_reason": reject_reason if rejected else None,
        "primary_proba_summary": {
            "mean_max_proba_at_fire": (
                float(np.mean(nested["all_max_proba_at_fire"]))
                if nested["all_max_proba_at_fire"] else 0.0
            ),
        },
        "timing_seconds": {
            "collection": round(collect_seconds, 1),
        },
    }
    if args.baseline:
        report["p4_baseline_diff"] = _baseline_diff(args.baseline, report)

    _write_report(Path(args.report_path), report)
    logger.info("report_written path=%s rejected=%s sha_unchanged=%s",
                args.report_path, rejected, primary_unchanged)
    if not primary_unchanged:
        # Critical safety check — should never happen since the harness
        # is read-only, but if it does, hard-fail rather than silently
        # ship a corrupted release.
        logger.error("primary_bundle_modified during run! "
                     "before=%s after=%s", sha_before, sha_after)
        return 5
    return 3 if rejected else 0


def main() -> int:
    args = _parse_args()

    primary_bundle_path = Path(args.primary_bundle)
    if not primary_bundle_path.exists():
        logger.error("primary_bundle_missing path=%s", primary_bundle_path)
        return 4

    sha_before = _sha256_file(primary_bundle_path)
    primary_bundle = joblib.load(primary_bundle_path)
    cfg = primary_bundle["config"]

    cusum_h = (
        float(args.cusum_h_override)
        if args.cusum_h_override is not None
        else float(cfg["labeling"]["cusum_h"])
    )
    cost_model = CostModel(
        fee_bps=float(cfg["cost_model"]["fee_bps"]),
        slip_bps=float(cfg["cost_model"]["slip_bps"]),
        impact_coef=float(cfg["cost_model"]["impact_coef"]),
    )
    participation = float(cfg["cost_model"]["participation"])
    sl_mult = float(cfg["sl_mult"])

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = top_symbols_by_volume(args.dsn, args.top_n, args.days)
    logger.info("symbols=%d primary=%s cusum_h=%.4f",
                len(symbols), primary_bundle_path, cusum_h)

    # ---- Sample collection ------------------------------------------------
    t0 = datetime.now()
    if args.nested_primary:
        # Nested mode: gather full candidate set with P4 labels; primary
        # is refit per fold inside run_nested_cv_meta.
        coll_nested = collect_p1_candidates_nested(
            symbols=symbols, days=args.days, frequency=args.frequency,
            dsn=args.dsn, primary_bundle=primary_bundle, cusum_h=cusum_h,
        )
        collect_seconds = (datetime.now() - t0).total_seconds()
        return _run_nested_path(
            args=args, primary_bundle=primary_bundle,
            primary_bundle_path=primary_bundle_path,
            sha_before=sha_before, cusum_h=cusum_h,
            cost_model=cost_model, participation=participation,
            sl_mult=sl_mult, symbols=symbols,
            coll=coll_nested, collect_seconds=collect_seconds,
        )

    coll = collect_p1_samples(
        symbols=symbols, days=args.days, frequency=args.frequency,
        dsn=args.dsn, primary_bundle=primary_bundle, cusum_h=cusum_h,
        decision_threshold_override=args.decision_threshold_override,
    )
    collect_seconds = (datetime.now() - t0).total_seconds()

    n = int(coll["X"].shape[0]) if coll["X"].size else 0
    if n == 0:
        logger.error("no_primary_fires — meta-model has nothing to filter")
        _write_report(Path(args.report_path), {
            "timestamp": datetime.now().isoformat(),
            "primary_bundle": str(primary_bundle_path),
            "rejected": True,
            "reject_reason": "no_primary_fires",
            "n_primary_fires": 0,
            "per_symbol_fires": coll["per_symbol_fires"],
            "per_symbol_events": coll["per_symbol_events"],
        })
        return 3

    # Sort chronologically.
    order = np.argsort(coll["event_times"].to_numpy())
    X = coll["X"][order]
    y = coll["y"][order]
    ret = coll["ret"][order]
    event_times = coll["event_times"][order]
    touch_times = coll["touch_times"][order]
    sides = coll["sides"][order]
    sym_groups = coll["symbols"][order]
    max_proba_at_fire = coll["max_proba"][order]

    label_dist = Counter(y.tolist())
    base_win_rate = label_dist.get(1, 0) / max(1, len(y))
    logger.info(
        "samples=%d wins=%d (%.1f%%) elapsed_s=%.1f",
        len(y), label_dist.get(1, 0), 100 * base_win_rate, collect_seconds,
    )

    # ---- Sample uniqueness on the META-event subset (clarification 2) ----
    touch_series = pd.Series(touch_times.to_numpy(), index=event_times)
    labels_for_w = pd.DataFrame(
        {"touch_time": touch_series.values}, index=touch_series.index,
    )
    weights = compute_uniqueness(
        labels_for_w,
        pd.DatetimeIndex(np.unique(np.concatenate([
            event_times.to_numpy(), touch_times.to_numpy(),
        ]))).sort_values(),
        groups=pd.Series(sym_groups),
    ).to_numpy()
    weights = np.clip(weights, 1e-6, 1.0)

    # ---- Cost in R per row (single source via ml.costs) ------------------
    barrier_pct_proxy = sl_mult / 100.0
    gross_r = ret / barrier_pct_proxy
    cost_r = _cost_r_per_row(
        n=len(y), sl_mult=sl_mult, cost_model=cost_model,
        participation=participation,
    )

    # ---- Purged CV -------------------------------------------------------
    threshold_sweep = [float(t) for t in args.threshold_sweep.split(",")]
    cv_block = run_purged_cv_meta(
        X=X, y=y, weights=weights,
        gross_r=gross_r, cost_r=cost_r,
        touch_times=touch_series,
        n_splits=args.folds, embargo_pct=args.embargo_pct,
        threshold_sweep=threshold_sweep,
    )
    oos_proba = cv_block["oos_proba"]
    covered = ~np.isnan(oos_proba)
    if covered.sum() == 0:
        logger.error("no oos coverage — purged CV produced no test rows")
        _write_report(Path(args.report_path), {
            "timestamp": datetime.now().isoformat(),
            "primary_bundle": str(primary_bundle_path),
            "rejected": True,
            "reject_reason": "no_oos_coverage",
            "n_primary_fires": int(len(y)),
        })
        return 3

    brier = compute_brier(y[covered], oos_proba[covered])
    cal = calibration_deciles(y[covered], oos_proba[covered])

    # Per-fold threshold aggregation — median across folds is the
    # deployment threshold.
    threshold_agreement = aggregate_per_fold_thresholds(
        cv_block["per_fold_thresholds"],
    )
    deployment_threshold = threshold_agreement.get("median")

    rejected = False
    reject_reason = ""
    chosen_row: dict[str, Any] = {}
    if deployment_threshold is None:
        rejected = True
        reject_reason = "no fold yielded a usable calibration-tail threshold"
        chosen_row = {}
    else:
        chosen_row = evaluate_at_threshold(
            proba=oos_proba[covered], y=y[covered],
            gross_r=gross_r[covered], cost_r=cost_r[covered],
            threshold=deployment_threshold,
        )

    # OOS sweep (full grid) — informational, not the picker. The picker
    # already ran on the calibration tail.
    full_sweep = threshold_sweep_meta(
        proba=oos_proba[covered], y_true=y[covered],
        gross_r=gross_r[covered], cost_r=cost_r[covered],
        thresholds=threshold_sweep,
    )

    # ---- Acceptance gates ------------------------------------------------
    if not rejected:
        gates = evaluate_p1_gates(
            brier=brier,
            calibration_max_dev=cal["max_abs_deviation"],
            chosen_threshold_row=chosen_row,
            per_fold_ar1_median=cv_block["median_per_fold_ar1"],
            per_fold_ar1_max=cv_block["max_per_fold_ar1"],
        )
        if not gates["all_pass"]:
            rejected = True
            failed = [
                k for k, v in gates.items()
                if isinstance(v, dict) and v.get("pass") is False
            ]
            reject_reason = "gates_failed: " + ",".join(failed)
    else:
        gates = {
            "brier": {"value": brier, "threshold": 0.22, "pass": False},
            "calibration": {"value": cal["max_abs_deviation"], "pass": False},
            "take_rate": {"pass": False},
            "profit_factor": {"pass": False},
            "expectancy": {"pass": False},
            "ar1": {
                "median": cv_block["median_per_fold_ar1"],
                "max": cv_block["max_per_fold_ar1"],
                "pass": (
                    cv_block["median_per_fold_ar1"] < 0.10
                    and cv_block["max_per_fold_ar1"] < 0.20
                ),
            },
            "all_pass": False,
        }

    # ---- Persist meta bundle on success ----------------------------------
    if not rejected and deployment_threshold is not None:
        full = train_meta_fold(
            X_train=X, y_train=y, w_train=weights,
            X_test=X[:1],  # discarded
            calibration_tail_frac=0.25, seed=42,
        )
        bundle = {
            "version": "p1-v1",
            "primary_bundle_path": str(primary_bundle_path),
            "primary_decision_threshold": float(
                args.decision_threshold_override
                if args.decision_threshold_override is not None
                else cfg.get("decision_threshold", 0.55)
            ),
            "primary_label_map": primary_bundle.get(
                "label_map", {0: "SELL", 1: "HOLD", 2: "BUY"},
            ),
            "base_clf": full["base_clf"],
            "isotonic": full["isotonic"],
            "scaler": full["scaler"],
            "feature_names": list(primary_bundle["feature_names"]),
            "meta_threshold": float(deployment_threshold),
            "config": {
                "folds": args.folds, "embargo_pct": args.embargo_pct,
                "threshold_sweep": threshold_sweep,
                "cusum_h": cusum_h,
                "primary_pt_mult": float(cfg["pt_mult"]),
                "primary_sl_mult": float(cfg["sl_mult"]),
                "primary_max_holding": int(cfg["max_holding"]),
                "primary_vertical_max_bars": int(cfg["labeling"]["vertical_max_bars"]),
                "frequency": args.frequency,
                "cost_model_anchor_r": P2_P4_COST_ANCHOR_R,
            },
            "trained_at": datetime.now().isoformat(),
        }
        Path(args.meta_bundle_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, args.meta_bundle_path)
        logger.info("meta_bundle_written path=%s threshold=%.2f",
                    args.meta_bundle_path, deployment_threshold)

    # ---- Final report ----------------------------------------------------
    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "primary_bundle": str(primary_bundle_path),
        "config": {
            "folds": args.folds, "embargo_pct": args.embargo_pct,
            "frequency": args.frequency, "days": args.days,
            "primary_decision_threshold": float(
                args.decision_threshold_override
                if args.decision_threshold_override is not None
                else cfg.get("decision_threshold", 0.55)
            ),
            "primary_pt_mult": float(cfg["pt_mult"]),
            "primary_sl_mult": float(cfg["sl_mult"]),
            "primary_max_holding": int(cfg["max_holding"]),
            "primary_vertical_max_bars": int(cfg["labeling"]["vertical_max_bars"]),
            "cusum_h": cusum_h,
            "cost_model": {
                "fee_bps": cost_model.fee_bps,
                "slip_bps": cost_model.slip_bps,
                "impact_coef": cost_model.impact_coef,
                "participation": participation,
                "anchor_r_per_trade": P2_P4_COST_ANCHOR_R,
                "source": "ml.costs (single source of truth shared with P2/P4)",
            },
            "threshold_sweep": threshold_sweep,
            "tiebreak_within_pct": 0.05,
            "uniqueness": "recomputed on meta-event subset (post-primary-fire)",
        },
        "symbols": symbols,
        "n_primary_events": int(sum(coll["per_symbol_events"].values())),
        "n_primary_fires": int(len(y)),
        "per_symbol_events": coll["per_symbol_events"],
        "per_symbol_fires": coll["per_symbol_fires"],
        "label_distribution": {
            "wins": int(label_dist.get(1, 0)),
            "losses": int(label_dist.get(0, 0)),
            "base_win_rate": round(base_win_rate, 4),
        },
        "weighting": {
            "mean": float(weights.mean()),
            "median": float(np.median(weights)),
            "min": float(weights.min()),
            "max": float(weights.max()),
            "effective_sample_size": float(weights.sum()),
        },
        "cv_summary": {
            "n_folds_executed": len(cv_block["fold_detail"]),
            "fold_detail": cv_block["fold_detail"],
        },
        "calibration": {
            "brier": brier,
            "deciles": cal,
        },
        "threshold_sweep": full_sweep,
        "per_fold_thresholds": cv_block["per_fold_thresholds"],
        "threshold_agreement": threshold_agreement,
        "chosen_threshold": chosen_row,
        "deployment_threshold": deployment_threshold,
        "iid_diagnostics": {
            "per_fold_ar1": cv_block["per_fold_ar1"],
            "median_per_fold_ar1": cv_block["median_per_fold_ar1"],
            "max_per_fold_ar1": cv_block["max_per_fold_ar1"],
            "gate_thresholds": {"median_lt": 0.10, "max_lt": 0.20},
        },
        "gates": gates,
        "rejected": rejected,
        "reject_reason": reject_reason if rejected else None,
        "primary_proba_summary": {
            "mean_max_proba_at_fire": (
                float(np.mean(max_proba_at_fire))
                if len(max_proba_at_fire) else 0.0
            ),
            "median_max_proba_at_fire": (
                float(np.median(max_proba_at_fire))
                if len(max_proba_at_fire) else 0.0
            ),
        },
    }
    if args.baseline:
        report["p4_baseline_diff"] = _baseline_diff(args.baseline, report)

    _write_report(Path(args.report_path), report)
    logger.info("report_written path=%s rejected=%s", args.report_path, rejected)

    return 3 if rejected else 0


if __name__ == "__main__":
    sys.exit(main())
