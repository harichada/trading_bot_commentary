"""PnL-weighted training objective (PROMPT_PACK P2 §1).

Primary path: sample weights ∝ max(|r_R| - cost_in_r, 0) * uniqueness.
Both XGBoost and LightGBM accept ``sample_weight=`` natively alongside their
built-in `multi:softprob` / `binary:logistic` losses, which is the canonical
way to get economically-weighted classification (AFML §3.6).

Optional path: custom grad/hess for binary cross-entropy, exposed for
callers who want LightGBM's ``objective=callable`` interface directly. The
multi-class variant is intentionally *not* implemented — keeping the grad/hess
path binary-only keeps the numerics trustworthy (the weight vector path is
framework-agnostic and handles multi-class cleanly).

Note on PROMPT_PACK wording: P2 names "LightGBM" as the target framework, but
``train_ml_model_v2.py`` uses XGBoost. The sample-weight vector produced here
is framework-agnostic and works with both; migration from XGB → LGBM is an
independent decision not scoped to P2.
"""
from __future__ import annotations

import numpy as np

# Floor keeps LightGBM happy (zero weights can trip some code paths) and
# matches López de Prado's recommendation to down-weight, not discard, cost-
# dominated samples — keeping a whisper of signal lets the booster correctly
# calibrate its probability head on marginal setups.
_WEIGHT_EPS = 1e-4


def compute_sample_weights(
    *,
    r_multiples: np.ndarray,
    cost_in_r: np.ndarray,
    uniqueness: np.ndarray,
) -> np.ndarray:
    """PnL-aware sample weights for gradient-boosted classifiers.

    weight_i = max(|r_multiples_i| - cost_in_r_i, 0) * uniqueness_i

    Floored at ``_WEIGHT_EPS`` to keep every sample live.

    Rationale (AFML §3.6):
      - |r| − cost is the *net expected magnitude* of this sample's outcome.
        Samples whose cost dominates their realized move carry ~no economic
        information and should not pull the decision boundary around.
      - Uniqueness corrects for the overlapping-label bias (AFML §4.3) —
        already computed upstream in ``ml/labels.py::compute_uniqueness``.
      - Sign symmetry: a ±2R outcome is equally informative. The *direction*
        enters the loss through the label, not the weight.
    """
    r_multiples = np.asarray(r_multiples, dtype="float64")
    cost_in_r = np.asarray(cost_in_r, dtype="float64")
    uniqueness = np.asarray(uniqueness, dtype="float64")

    if not (r_multiples.shape == cost_in_r.shape == uniqueness.shape):
        raise ValueError(
            "r_multiples, cost_in_r, uniqueness must share shape; "
            f"got {r_multiples.shape}, {cost_in_r.shape}, {uniqueness.shape}"
        )

    net_edge = np.maximum(np.abs(r_multiples) - cost_in_r, 0.0)
    weights = net_edge * uniqueness
    return np.maximum(weights, _WEIGHT_EPS)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable for large |z|.
    pos = z >= 0
    out = np.empty_like(z)
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    out[~pos] = np.exp(z[~pos]) / (1.0 + np.exp(z[~pos]))
    return out


def binary_pnl_grad_hess(
    *,
    preds: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient / hessian of weighted binary cross-entropy on logits.

    Loss per sample: ``w_i * [log(1 + exp(z_i)) - y_i * z_i]``
    Gradient:       ``w_i * (σ(z_i) - y_i)``
    Hessian:        ``w_i * σ(z_i) * (1 - σ(z_i))``

    Shape ``(n,), (n,)`` — matches LightGBM's ``objective=callable`` contract.
    """
    preds = np.asarray(preds, dtype="float64")
    labels = np.asarray(labels, dtype="float64")
    w = np.asarray(sample_weights, dtype="float64")

    p = _sigmoid(preds)
    grad = w * (p - labels)
    hess = w * p * (1.0 - p)
    return grad, hess
