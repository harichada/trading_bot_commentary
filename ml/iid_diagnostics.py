"""IID diagnostics for weighted residuals.

The triple-barrier method produces overlapping label windows; without
sample-uniqueness weighting (López de Prado AFML §4.3) the model fits a
non-IID training distribution. We can't observe IID-ness directly, but
we can audit the **residual** series of the trained model: if the
residuals exhibit AR(1) autocorrelation, then the labels carried serial
information the weighting failed to dampen.

The P4 acceptance gate is ``|AR(1)| < 0.10`` on the *weighted* residuals.
This is a structural check on the training pipeline, not a performance
metric — it tells you whether the math behind the model is right, before
you ask whether the model has edge.

References
----------
* López de Prado, *Advances in Financial Machine Learning* (2018), Ch. 4.
* Ljung & Box (1978), "On a Measure of Lack of Fit in Time Series Models".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IIDReport:
    ar1_residual_autocorr: float
    ljung_box_p_value: float | None
    passes_ar1_lt_0_1: bool
    n_residuals: int


def weighted_residuals(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    sample_weights: np.ndarray,
) -> np.ndarray:
    """Weighted Pearson residuals for a binary outcome.

    ``r_i = w_i^{1/2} * (y_i - p_i) / sqrt(p_i * (1 - p_i))``

    The square-root weighting is the standard variance-stabilizing form
    so that down-weighted samples contribute proportionally less to the
    AR(1) coefficient. Probabilities are clipped to ``[1e-6, 1-1e-6]`` to
    keep the denominator finite.
    """
    y_true = np.asarray(y_true, dtype="float64")
    p = np.clip(np.asarray(y_pred_proba, dtype="float64"), 1e-6, 1 - 1e-6)
    w = np.asarray(sample_weights, dtype="float64")
    if not (y_true.shape == p.shape == w.shape):
        raise ValueError("y_true, y_pred_proba, sample_weights must align")
    denom = np.sqrt(p * (1.0 - p))
    return np.sqrt(np.clip(w, 0.0, None)) * (y_true - p) / denom


def _ar1_coefficient(residuals: np.ndarray) -> float:
    """Lag-1 sample autocorrelation. Falls back to 0 on degenerate input."""
    r = np.asarray(residuals, dtype="float64")
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    r = r - r.mean()
    denom = float(np.dot(r, r))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(r[:-1], r[1:]) / denom)


def _ljung_box_p(residuals: np.ndarray, lags: int = 10) -> float | None:
    """Ljung–Box p-value at ``lags``. Returns None if statsmodels missing."""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:  # pragma: no cover — statsmodels is normally installed
        return None
    r = np.asarray(residuals, dtype="float64")
    r = r[np.isfinite(r)]
    if r.size < lags + 2:
        return None
    try:
        result = acorr_ljungbox(r, lags=[lags], return_df=True)
        return float(result["lb_pvalue"].iloc[0])
    except Exception:  # statsmodels has occasionally raised on tiny inputs
        return None


def compute_iid_diagnostics(
    residuals: np.ndarray,
    event_times: pd.DatetimeIndex | None = None,
) -> IIDReport:
    """Return AR(1) and Ljung–Box diagnostics for a residual series.

    ``event_times`` is accepted for future use (per-symbol AR(1)) but is
    not currently consumed — residuals are treated as a single ordered
    series.
    """
    del event_times  # reserved
    r = np.asarray(residuals, dtype="float64")
    finite = r[np.isfinite(r)]
    ar1 = _ar1_coefficient(finite)
    lb_p = _ljung_box_p(finite, lags=10)
    return IIDReport(
        ar1_residual_autocorr=ar1,
        ljung_box_p_value=lb_p,
        passes_ar1_lt_0_1=bool(abs(ar1) < 0.10),
        n_residuals=int(finite.size),
    )
