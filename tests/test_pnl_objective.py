"""Tests for PnL-weighted training objective (PROMPT_PACK P2 §1).

Primary path: sample weights ∝ max(|r_R| - cost_in_r, 0) × uniqueness.
Optional path: custom grad/hess for binary xentropy — numerical gradient
check against scipy.optimize.approx_fprime.
"""
from __future__ import annotations

import numpy as np
import pytest

from ml.pnl_objective import (
    binary_pnl_grad_hess,
    compute_sample_weights,
)


class TestSampleWeights:
    def test_zero_when_cost_exceeds_edge(self) -> None:
        """|r_R| < cost_in_r → weight floored at eps (≈ 0)."""
        r_multiples = np.array([0.2, -0.2, 0.1])
        costs = np.array([0.5, 0.5, 0.5])
        uniqueness = np.ones(3)

        w = compute_sample_weights(
            r_multiples=r_multiples, cost_in_r=costs, uniqueness=uniqueness,
        )

        assert (w <= 1e-3).all(), f"expected ≈0, got {w}"
        assert (w > 0).all(), "weights must stay strictly positive for LightGBM"

    def test_monotone_in_net_edge(self) -> None:
        """Larger (|r| - cost) → strictly larger weight (given equal uniqueness)."""
        r_multiples = np.array([0.5, 1.0, 2.0])
        costs = np.array([0.3, 0.3, 0.3])
        uniqueness = np.ones(3)

        w = compute_sample_weights(
            r_multiples=r_multiples, cost_in_r=costs, uniqueness=uniqueness,
        )

        assert w[0] < w[1] < w[2]

    def test_linear_in_uniqueness(self) -> None:
        """Doubling uniqueness doubles the effective weight (before eps floor)."""
        r = np.array([1.0, 1.0])
        c = np.array([0.3, 0.3])
        u_lo = np.array([0.5, 0.5])
        u_hi = np.array([1.0, 1.0])

        w_lo = compute_sample_weights(r_multiples=r, cost_in_r=c, uniqueness=u_lo)
        w_hi = compute_sample_weights(r_multiples=r, cost_in_r=c, uniqueness=u_hi)

        assert w_hi[0] == pytest.approx(2.0 * w_lo[0], rel=1e-6)

    def test_sign_of_return_does_not_change_weight(self) -> None:
        """Weight depends on |r|, not direction — a -2R hit is as informative as +2R."""
        w_pos = compute_sample_weights(
            r_multiples=np.array([2.0]),
            cost_in_r=np.array([0.3]),
            uniqueness=np.array([1.0]),
        )
        w_neg = compute_sample_weights(
            r_multiples=np.array([-2.0]),
            cost_in_r=np.array([0.3]),
            uniqueness=np.array([1.0]),
        )
        assert w_pos[0] == pytest.approx(w_neg[0], abs=1e-12)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_sample_weights(
                r_multiples=np.array([1.0, 1.0]),
                cost_in_r=np.array([0.3]),
                uniqueness=np.array([1.0, 1.0]),
            )


class TestBinaryPnLGradHess:
    def _loss(self, z: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """Weighted binary cross-entropy, logits-in."""
        # -w * (y*log(sigmoid(z)) + (1-y)*log(1 - sigmoid(z)))
        # Numerically-stable form:
        return float(np.sum(w * (np.logaddexp(0, z) - y * z)))

    def test_grad_numerical_match(self) -> None:
        """Analytical grad must match scipy.optimize.approx_fprime within 1e-5."""
        from scipy.optimize import approx_fprime

        rng = np.random.default_rng(7)
        n = 12
        y = rng.integers(0, 2, size=n).astype("float64")
        z = rng.normal(size=n)
        w = rng.uniform(0.2, 1.5, size=n)

        grad, _ = binary_pnl_grad_hess(preds=z, labels=y, sample_weights=w)

        num_grad = approx_fprime(
            z, lambda zz: self._loss(zz, y, w), epsilon=1e-6,
        )

        np.testing.assert_allclose(grad, num_grad, atol=1e-4)

    def test_hess_is_positive(self) -> None:
        """Hessian of binary xent is w * σ(z)(1-σ(z)) ≥ 0 pointwise."""
        z = np.linspace(-3, 3, 7)
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        w = np.ones(7)

        _, hess = binary_pnl_grad_hess(preds=z, labels=y, sample_weights=w)

        assert (hess >= 0).all()

    def test_output_shapes_match_lightgbm_expectation(self) -> None:
        z = np.zeros(20)
        y = np.ones(20)
        w = np.ones(20)
        grad, hess = binary_pnl_grad_hess(preds=z, labels=y, sample_weights=w)
        assert grad.shape == (20,)
        assert hess.shape == (20,)
        assert np.all(np.isfinite(grad))
        assert np.all(np.isfinite(hess))
