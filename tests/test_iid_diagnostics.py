"""Tests for ``ml/iid_diagnostics.py``.

These diagnostics gate P4: the report's ``passes_ar1_lt_0_1`` flag must
reflect whether the **weighted** residual series looks IID. If sample-uniqueness
weighting is doing its job, the weighted-residual AR(1) coefficient should be
close to zero even when the *unweighted* residuals are heavily autocorrelated
because of overlapping label windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.iid_diagnostics import (
    IIDReport,
    compute_iid_diagnostics,
    weighted_residuals,
)


def _ar1(series: np.ndarray, coef: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros_like(series, dtype="float64")
    for i in range(1, len(out)):
        out[i] = coef * out[i - 1] + rng.normal()
    return out


class TestComputeIIDDiagnostics:
    def test_white_noise_passes(self):
        rng = np.random.default_rng(0)
        residuals = rng.normal(size=2000)
        idx = pd.date_range("2024-01-02", periods=2000, freq="1min")

        report = compute_iid_diagnostics(residuals, idx)

        assert isinstance(report, IIDReport)
        assert abs(report.ar1_residual_autocorr) < 0.05
        assert report.passes_ar1_lt_0_1 is True
        assert report.n_residuals == 2000

    def test_high_autocorr_fails(self):
        residuals = _ar1(np.zeros(2000), coef=0.4, seed=1)
        idx = pd.date_range("2024-01-02", periods=2000, freq="1min")

        report = compute_iid_diagnostics(residuals, idx)

        assert report.ar1_residual_autocorr > 0.3
        assert report.passes_ar1_lt_0_1 is False

    def test_ljung_box_p_high_for_white_noise(self):
        rng = np.random.default_rng(2)
        residuals = rng.normal(size=2000)
        idx = pd.date_range("2024-01-02", periods=2000, freq="1min")

        report = compute_iid_diagnostics(residuals, idx)

        if report.ljung_box_p_value is not None:
            assert report.ljung_box_p_value > 0.05

    def test_ljung_box_p_low_for_correlated(self):
        residuals = _ar1(np.zeros(2000), coef=0.4, seed=3)
        idx = pd.date_range("2024-01-02", periods=2000, freq="1min")

        report = compute_iid_diagnostics(residuals, idx)

        if report.ljung_box_p_value is not None:
            assert report.ljung_box_p_value < 0.05

    def test_short_series_returns_finite_ar1(self):
        residuals = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
        idx = pd.date_range("2024-01-02", periods=5, freq="1min")

        report = compute_iid_diagnostics(residuals, idx)

        assert np.isfinite(report.ar1_residual_autocorr)


class TestWeightedResiduals:
    def test_weighting_reduces_ar1_under_overlap(self):
        """The whole point: weighted residuals on an overlapping label set
        have lower AR(1) than unweighted residuals on the same set."""
        rng = np.random.default_rng(7)
        n = 1000
        # Construct y_true with autocorrelation (overlapping labels share state).
        latent = _ar1(np.zeros(n), coef=0.5, seed=7)
        y_true = (latent > 0).astype(int)
        # Predictions: 0.5 + small noise (model has no edge).
        y_proba = 0.5 + 0.05 * rng.normal(size=n)
        y_proba = np.clip(y_proba, 1e-3, 1 - 1e-3)
        # Weights: highly overlapping samples down-weighted.
        weights = 1.0 / (1.0 + np.abs(latent))

        unweighted = weighted_residuals(y_true, y_proba, np.ones(n))
        weighted = weighted_residuals(y_true, y_proba, weights)

        idx = pd.date_range("2024-01-02", periods=n, freq="1min")
        ar1_unw = compute_iid_diagnostics(unweighted, idx).ar1_residual_autocorr
        ar1_w = compute_iid_diagnostics(weighted, idx).ar1_residual_autocorr

        assert abs(ar1_w) < abs(ar1_unw), (
            f"weighted AR(1)={ar1_w:.3f} should be < unweighted AR(1)={ar1_unw:.3f}"
        )

    def test_returns_finite_array(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([0.4, 0.6, 0.7, 0.3])
        w = np.array([1.0, 0.5, 0.8, 1.0])

        r = weighted_residuals(y, p, w)

        assert r.shape == y.shape
        assert np.isfinite(r).all()
