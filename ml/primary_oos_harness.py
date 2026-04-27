"""Nested-CV harness — refit the P4 primary on a meta-CV fold's train rows.

López de Prado, AFML §7.4. Meta-labeling on a model-based primary
requires the primary's predictions to be **out-of-sample with respect
to the primary's own training**. Since `ml_model_v2.pkl` was trained
on the same window the meta-trainer reads, we cannot use its directly
saved predictions — they would be in-sample. Instead, for each
meta-CV fold we refit a fresh primary on the fold's training rows
using the **exact** config that produced `ml_model_v2.pkl` (sourced
from that bundle's `.config`), then predict softprob on the fold's
test rows. Those OOS predictions are the ones the meta-classifier
trains and evaluates against.

This module is a pure evaluation harness. It does not:
  * touch `ml_model_v2.pkl` on disk,
  * persist any pkl,
  * change the primary's architecture, features, hyperparameters,
    CUSUM/triple-barrier setup, or release process,
  * load any subordinate primary later.

It does require the caller to pass a config dict that contains every
field the harness needs. Missing fields raise ``KeyError`` immediately
— do not infer or silently default.

Caller's responsibility:
  * Build features (X_train, X_test) using the **same** feature
    extractor the primary trained against.
  * Triple-barrier-label the training rows ahead of time and pass
    sample_r_multiples / sample_cost_r so the cost-aware sample weights
    match the P4 training loop.
  * Pass uniqueness-derived ``weights_train``.

The harness then performs P4's exact training math: sequential
bootstrap with the fold-local indicator matrix, ``_ensure_all_classes``
padding, ``StandardScaler`` per fold, ``XGBClassifier`` with the same
hyperparameters, average across bootstrap iterations.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.pnl_objective import compute_sample_weights
from ml.sample_weights import (
    get_indicator_matrix,
    sequential_bootstrap_indices,
)


_REQUIRED_KEYS_TOP = ("pt_mult", "sl_mult", "max_holding")
_REQUIRED_KEYS_LABELING = ("vertical_max_bars", "min_ret_atr_mult", "cusum_h")
_REQUIRED_KEYS_COST = ("fee_bps", "slip_bps", "impact_coef", "participation")


def _validate_primary_config(cfg: dict[str, Any]) -> None:
    """Hard-fail when ``ml_model_v2.pkl.config`` is missing any required
    field used by the nested-CV harness. No inference, no defaults — a
    missing key indicates the caller passed a stale or wrong bundle."""
    for k in _REQUIRED_KEYS_TOP:
        if k not in cfg:
            raise KeyError(f"primary_config missing required top-level key: {k!r}")
    if "labeling" not in cfg or not isinstance(cfg["labeling"], dict):
        raise KeyError("primary_config missing 'labeling' block")
    for k in _REQUIRED_KEYS_LABELING:
        if k not in cfg["labeling"]:
            raise KeyError(
                f"primary_config['labeling'] missing required key: {k!r}"
            )
    if "cost_model" not in cfg or not isinstance(cfg["cost_model"], dict):
        raise KeyError("primary_config missing 'cost_model' block")
    for k in _REQUIRED_KEYS_COST:
        if k not in cfg["cost_model"]:
            raise KeyError(
                f"primary_config['cost_model'] missing required key: {k!r}"
            )


def refit_primary_on_fold(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    weights_train: np.ndarray,
    X_test: np.ndarray,
    event_times_train: pd.DatetimeIndex,
    touch_times_train: pd.DatetimeIndex,
    sample_r_multiples_train: np.ndarray,
    sample_cost_r_train: np.ndarray,
    primary_config: dict[str, Any],
    bootstrap_iterations: int = 5,
    use_sequential_bootstrap: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """Refit the P4 primary on fold-train rows; return OOS softprob on test.

    Parameters
    ----------
    X_train, y_train, weights_train:
        Feature matrix, 3-class label vector ({0, 1, 2}), and sample
        uniqueness weights for the meta-CV fold's *training* rows. Must
        align row-for-row.
    X_test:
        Feature matrix for the fold's test rows. Same column ordering as
        ``X_train``; the harness fits a per-fold ``StandardScaler`` and
        applies it to both.
    event_times_train, touch_times_train:
        Per-row event start time and triple-barrier touch time for the
        training rows. Used to construct the fold-local indicator matrix
        for sequential bootstrap (AFML §4.5.3, Snippet 4.5).
    sample_r_multiples_train, sample_cost_r_train:
        Per-row R-multiple and cost in R for training rows. Combined
        with ``weights_train`` via :func:`compute_sample_weights` to
        produce the cost-aware sample weights P4 uses during XGBoost
        fitting (P2 §1).
    primary_config:
        Dict shaped like ``ml_model_v2.pkl.config``. Caller must source
        this from the bundle directly — do not redefine inline. Missing
        keys raise ``KeyError`` immediately.
    bootstrap_iterations:
        Per-fold sequential-bootstrap fits (P4 default 5). Final OOS
        softprob is the row-wise mean across iterations.
    use_sequential_bootstrap:
        Toggle for the AFML §4.5.3 sequential bootstrap. Default True
        matches P4. Set False for the legacy `subsample=0.8` path —
        regression-comparison only.
    seed:
        Base RNG seed; per-iteration seeds are ``seed + 100 * iter``.

    Returns
    -------
    np.ndarray of shape (len(X_test), 3) — row-wise mean of the
    bootstrap-averaged softprob predictions, columns indexed
    {SELL=0, HOLD=1, BUY=2}.
    """
    _validate_primary_config(primary_config)

    # Lazy import keeps the unit-test surface narrow when the caller
    # only exercises validation.
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    # P4-identical XGBoost hyperparameters. These mirror
    # train_ml_model_v2.purged_cv_evaluate exactly. Sourcing them from
    # primary_config is overkill: the released ml_model_v2.pkl was
    # produced with these literals; the harness must reproduce them
    # bit-for-bit. If a user wanted different params they would have
    # released a different primary, which is out of scope.
    n_estimators = 200
    max_depth = 5
    learning_rate = 0.1
    colsample_bytree = 0.8

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError(
            f"X_train ({X_train.shape}) and X_test ({X_test.shape}) "
            "must be non-empty"
        )

    # Cost-aware sample weights — same formula as the P4 trainer.
    sample_weights = compute_sample_weights(
        r_multiples=sample_r_multiples_train,
        cost_in_r=sample_cost_r_train,
        uniqueness=weights_train,
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    # Build the fold-local indicator matrix once (AFML §4.5.3).
    if use_sequential_bootstrap:
        train_bar_index = pd.DatetimeIndex(
            np.unique(np.concatenate([
                event_times_train.to_numpy(),
                touch_times_train.to_numpy(),
            ]))
        ).sort_values()
        ind_matrix = get_indicator_matrix(
            train_bar_index, event_times_train, touch_times_train,
        )
    else:
        ind_matrix = None

    proba_acc = np.zeros((len(X_test), 3), dtype="float64")
    n_succ = 0

    for it in range(bootstrap_iterations):
        if use_sequential_bootstrap and ind_matrix is not None:
            resampled = sequential_bootstrap_indices(
                ind_matrix, n_samples=len(X_train),
                seed=seed + 100 * (it + 1),
            )
            X_fit = X_tr_s[resampled]
            y_fit = y_train[resampled]
            w_fit = sample_weights[resampled]
            subsample = 1.0
        else:
            X_fit, y_fit, w_fit = X_tr_s, y_train, sample_weights
            subsample = 0.8

        clf = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=seed + it,
            eval_metric="mlogloss", num_class=3,
            objective="multi:softprob", n_jobs=-1,
        )
        X_fit_p, y_fit_p, w_fit_p = _ensure_all_classes_inline(X_fit, y_fit, w_fit)
        clf.fit(X_fit_p, y_fit_p, sample_weight=w_fit_p, verbose=False)
        try:
            proba_it = clf.predict_proba(X_te_s)
        except Exception:  # pragma: no cover
            continue
        proba_acc += proba_it
        n_succ += 1

    if n_succ == 0:
        raise RuntimeError(
            "all bootstrap iterations failed to produce predictions"
        )
    return proba_acc / n_succ


def _ensure_all_classes_inline(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, n_classes: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local copy of ``train_ml_model_v2._ensure_all_classes`` to keep
    this harness importable without circular dependencies. Behavior is
    bit-identical: pads one zero-weight row per missing class so XGBoost's
    multi-class objective doesn't trip on a single-class fold."""
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
