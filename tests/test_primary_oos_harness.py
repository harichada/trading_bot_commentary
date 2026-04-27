"""Unit tests for ml/primary_oos_harness.py.

The harness implements the AFML §7.4 nested-CV primary refit:
refit a primary classifier on fold-train rows using a config dict
identical to ``ml_model_v2.pkl.config``, predict OOS softprob on
fold-test rows, persist nothing.

These tests were lifted verbatim from
``tests/test_meta_model_p1.py::TestRefitPrimaryOnFold`` on
``feat/P1-meta-labeling`` (commit d29bd48). They have no dependency
on the P1 meta-classifier module — only on the harness public API
and standard scientific stack.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest


class TestRefitPrimaryOnFold:
    """Tests for the AFML §7.4 nested-primary harness.

    The harness refits a primary classifier on fold-train rows using a
    config dict identical to ``ml_model_v2.pkl.config``, then predicts
    OOS softprob on fold-test rows. No disk I/O; no persistence.
    """

    def _synthetic_primary_data(self, n: int = 600, seed: int = 7
                                ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        n_features = 6
        signal = rng.normal(size=n)
        feat2 = 0.6 * signal + 0.4 * rng.normal(size=n)
        noise = rng.normal(size=(n, n_features - 2))
        X = np.column_stack([signal, feat2, noise])
        # 3-class labels biased by signal (BUY=2, SELL=0, HOLD=1).
        p_buy = 0.30 + 0.30 * np.tanh(signal)
        p_sell = 0.30 - 0.30 * np.tanh(signal)
        u = rng.uniform(size=n)
        y = np.where(u < p_buy, 2, np.where(u < p_buy + p_sell, 0, 1)).astype(np.int64)
        # Synthetic touches: each event closes 5 bars later.
        ev = pd.DatetimeIndex(
            [pd.Timestamp("2024-09-02") + pd.Timedelta(minutes=5 * i)
             for i in range(n)]
        )
        tc = ev + pd.Timedelta(minutes=25)
        ret = np.where(y == 2, 0.02, np.where(y == 0, -0.01, 0.0))
        return {"X": X, "y": y, "ret": ret, "ev": ev, "tc": tc}

    def test_refit_returns_oos_softprob_shape(self) -> None:
        from ml.primary_oos_harness import refit_primary_on_fold

        data = self._synthetic_primary_data(n=500, seed=11)
        cut = 350
        primary_config = {
            "pt_mult": 2.0, "sl_mult": 1.0, "max_holding": 60,
            "decision_threshold": 0.55,
            "labeling": {"vertical_max_bars": 120,
                         "min_ret_atr_mult": 1.0, "cusum_h": 0.005},
            "cost_model": {"fee_bps": 5.0, "slip_bps": 10.0,
                           "impact_coef": 0.1, "participation": 0.01},
            "frequency": 5,
        }
        weights = np.ones(len(data["y"]))
        proba = refit_primary_on_fold(
            X_train=data["X"][:cut], y_train=data["y"][:cut],
            weights_train=weights[:cut],
            X_test=data["X"][cut:],
            event_times_train=data["ev"][:cut],
            touch_times_train=data["tc"][:cut],
            primary_config=primary_config,
            sample_r_multiples_train=np.where(data["y"][:cut] == 2, 2.0,
                                              np.where(data["y"][:cut] == 0, -1.0, 0.0)),
            sample_cost_r_train=np.full(cut, 0.32),
            seed=42,
        )
        assert proba.shape == (len(data["y"]) - cut, 3)
        # Probas sum to 1 row-wise.
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_refit_does_not_mutate_disk_or_save(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Harness must NOT call joblib.dump or write any file."""
        from ml import primary_oos_harness as mod

        called = []
        import joblib
        monkeypatch.setattr(joblib, "dump",
                            lambda *a, **k: called.append(a))

        data = self._synthetic_primary_data(n=400, seed=12)
        cut = 280
        primary_config = {
            "pt_mult": 2.0, "sl_mult": 1.0, "max_holding": 60,
            "decision_threshold": 0.55,
            "labeling": {"vertical_max_bars": 120,
                         "min_ret_atr_mult": 1.0, "cusum_h": 0.005},
            "cost_model": {"fee_bps": 5.0, "slip_bps": 10.0,
                           "impact_coef": 0.1, "participation": 0.01},
            "frequency": 5,
        }
        weights = np.ones(len(data["y"]))
        mod.refit_primary_on_fold(
            X_train=data["X"][:cut], y_train=data["y"][:cut],
            weights_train=weights[:cut],
            X_test=data["X"][cut:],
            event_times_train=data["ev"][:cut],
            touch_times_train=data["tc"][:cut],
            primary_config=primary_config,
            sample_r_multiples_train=np.where(data["y"][:cut] == 2, 2.0,
                                              np.where(data["y"][:cut] == 0, -1.0, 0.0)),
            sample_cost_r_train=np.full(cut, 0.32),
            seed=42,
        )
        assert called == [], "harness must not persist any pkl"

    def test_refit_fails_loudly_on_missing_config_key(self) -> None:
        from ml.primary_oos_harness import refit_primary_on_fold

        data = self._synthetic_primary_data(n=300, seed=13)
        cut = 200
        # Missing labeling.vertical_max_bars deliberately.
        bad_config = {
            "pt_mult": 2.0, "sl_mult": 1.0, "max_holding": 60,
            "decision_threshold": 0.55,
            "labeling": {"min_ret_atr_mult": 1.0, "cusum_h": 0.005},
            "cost_model": {"fee_bps": 5.0, "slip_bps": 10.0,
                           "impact_coef": 0.1, "participation": 0.01},
            "frequency": 5,
        }
        weights = np.ones(len(data["y"]))
        with pytest.raises((KeyError, ValueError),
                           match=r"vertical_max_bars|labeling"):
            refit_primary_on_fold(
                X_train=data["X"][:cut], y_train=data["y"][:cut],
                weights_train=weights[:cut],
                X_test=data["X"][cut:],
                event_times_train=data["ev"][:cut],
                touch_times_train=data["tc"][:cut],
                primary_config=bad_config,
                sample_r_multiples_train=np.zeros(cut),
                sample_cost_r_train=np.full(cut, 0.32),
                seed=42,
            )

    def test_refit_oos_predictions_not_in_sample_perfect(self) -> None:
        """Sanity check: OOS proba on fresh rows should NOT be perfectly
        accurate (which would indicate leakage). Argmax accuracy on truly
        held-out data with synthetic noisy labels should be well below 1.0."""
        from ml.primary_oos_harness import refit_primary_on_fold

        data = self._synthetic_primary_data(n=800, seed=42)
        cut = 600
        primary_config = {
            "pt_mult": 2.0, "sl_mult": 1.0, "max_holding": 60,
            "decision_threshold": 0.55,
            "labeling": {"vertical_max_bars": 120,
                         "min_ret_atr_mult": 1.0, "cusum_h": 0.005},
            "cost_model": {"fee_bps": 5.0, "slip_bps": 10.0,
                           "impact_coef": 0.1, "participation": 0.01},
            "frequency": 5,
        }
        weights = np.ones(len(data["y"]))
        proba = refit_primary_on_fold(
            X_train=data["X"][:cut], y_train=data["y"][:cut],
            weights_train=weights[:cut],
            X_test=data["X"][cut:],
            event_times_train=data["ev"][:cut],
            touch_times_train=data["tc"][:cut],
            primary_config=primary_config,
            sample_r_multiples_train=np.where(data["y"][:cut] == 2, 2.0,
                                              np.where(data["y"][:cut] == 0, -1.0, 0.0)),
            sample_cost_r_train=np.full(cut, 0.32),
            seed=42,
        )
        argmax = proba.argmax(axis=1)
        oos_acc = float((argmax == data["y"][cut:]).mean())
        # Synthetic data has Bayes-rate accuracy well below 1; assert OOS
        # is not 100% (which would be the leakage signature).
        assert oos_acc < 0.95, (
            f"OOS accuracy {oos_acc:.3f} is too high — possible leakage"
        )
