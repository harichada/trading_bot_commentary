"""Unit tests for ml/meta_model.py — pure-function level.

P1 (López de Prado AFML §3.6-3.7) meta-labeling layer. These tests
validate the pure helpers: Brier, decile calibration, cost-adjusted
Sortino, calibration-tail threshold picker, per-fold meta trainer,
acceptance-gate evaluator. The training-loop integration is tested in
tests/test_train_meta_p1_integration.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.meta_model import (
    MetaThresholdRejection,
    calibration_deciles,
    compute_brier,
    cost_adjusted_sortino,
    evaluate_p1_gates,
    pick_threshold_on_calibration_tail,
    select_primary_fires,
    threshold_sweep_meta,
    train_meta_fold,
)


class TestComputeBrier:
    def test_perfect_predictions_zero_brier(self) -> None:
        y = np.array([0, 1, 0, 1, 1])
        p = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
        assert compute_brier(y, p) == pytest.approx(0.0, abs=1e-9)

    def test_uninformative_half_predictions(self) -> None:
        y = np.array([0, 1, 0, 1])
        p = np.full(4, 0.5)
        assert compute_brier(y, p) == pytest.approx(0.25, abs=1e-9)

    def test_array_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="align"):
            compute_brier(np.array([0, 1]), np.array([0.5, 0.5, 0.5]))

    def test_empty_returns_zero(self) -> None:
        assert compute_brier(np.array([]), np.array([])) == 0.0


class TestCalibrationDeciles:
    def test_perfect_calibration_zero_max_error(self) -> None:
        rng = np.random.default_rng(0)
        bin_centers = np.linspace(0.05, 0.95, 10)
        y_list, p_list = [], []
        for c in bin_centers:
            n = 200
            p = np.full(n, c)
            y = (rng.uniform(size=n) < c).astype(int)
            y_list.append(y)
            p_list.append(p)
        y = np.concatenate(y_list)
        p = np.concatenate(p_list)

        deciles = calibration_deciles(y, p, min_count=30)
        assert deciles["max_abs_deviation"] is not None
        assert deciles["max_abs_deviation"] < 0.05
        assert len(deciles["bins"]) == 10

    def test_undercount_bins_excluded(self) -> None:
        y = np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 1])
        p = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        deciles = calibration_deciles(y, p, min_count=5)
        assert deciles["max_abs_deviation"] is None
        assert deciles["n_qualifying_bins"] == 0

    def test_returns_per_bin_records(self) -> None:
        y = np.tile([0, 1], 50)
        p = np.full(100, 0.5)
        deciles = calibration_deciles(y, p, min_count=10)
        assert sum(b["n"] for b in deciles["bins"]) == 100


class TestCostAdjustedSortino:
    def test_zero_trades_returns_zero(self) -> None:
        assert cost_adjusted_sortino(
            net_r=np.array([]), cost_per_trade_r=0.32,
        ) == 0.0

    def test_costs_subtracted_from_each_trade(self) -> None:
        # +1R wins x10, -1R losses x10. With 0.32R cost subtracted:
        #   wins net = 0.68R, losses net = -1.32R, mean = -0.32R.
        net_r = np.concatenate([np.full(10, 1.0), np.full(10, -1.0)])
        s = cost_adjusted_sortino(net_r=net_r, cost_per_trade_r=0.32)
        # Sortino ratio is negative (mean is -0.32) and finite.
        assert s < 0
        assert np.isfinite(s)

    def test_higher_for_more_winners(self) -> None:
        net_winners = np.full(20, 1.0)
        net_losers = np.full(20, -1.0)
        s_win = cost_adjusted_sortino(net_r=net_winners, cost_per_trade_r=0.10)
        s_lose = cost_adjusted_sortino(net_r=net_losers, cost_per_trade_r=0.10)
        assert s_win > s_lose


class TestThresholdSweepMeta:
    def test_returns_one_row_per_threshold(self) -> None:
        rng = np.random.default_rng(0)
        n = 100
        proba = rng.uniform(0.3, 0.9, size=n)
        is_winner = (proba > np.median(proba)).astype(int)
        gross_r = np.where(is_winner == 1, 1.0, -1.0)
        cost_r = np.full(n, 0.32)

        sweep = threshold_sweep_meta(
            proba=proba,
            y_true=is_winner,
            gross_r=gross_r,
            cost_r=cost_r,
            thresholds=[0.40, 0.55, 0.70, 0.85],
        )
        assert len(sweep) == 4
        for row in sweep:
            assert {
                "threshold", "n_taken", "take_rate",
                "expectancy_net_r", "profit_factor",
                "win_rate", "sortino_net",
            }.issubset(row.keys())

    def test_higher_threshold_lowers_take_rate(self) -> None:
        rng = np.random.default_rng(1)
        n = 200
        proba = rng.uniform(0.0, 1.0, size=n)
        y = (proba > 0.5).astype(int)
        gross_r = np.where(y == 1, 2.0, -1.0)
        cost_r = np.full(n, 0.3)

        sweep = threshold_sweep_meta(
            proba=proba, y_true=y, gross_r=gross_r, cost_r=cost_r,
            thresholds=[0.30, 0.50, 0.70, 0.90],
        )
        rates = [r["take_rate"] for r in sweep]
        assert all(a >= b for a, b in zip(rates, rates[1:]))

    def test_zero_takes_returns_none_metrics(self) -> None:
        sweep = threshold_sweep_meta(
            proba=np.array([0.1, 0.2, 0.3]),
            y_true=np.array([1, 1, 1]),
            gross_r=np.array([1.0, 1.0, 1.0]),
            cost_r=np.array([0.3, 0.3, 0.3]),
            thresholds=[0.99],
        )
        assert sweep[0]["n_taken"] == 0
        assert sweep[0]["take_rate"] == 0.0
        assert sweep[0]["profit_factor"] is None
        assert sweep[0]["expectancy_net_r"] is None
        assert sweep[0]["sortino_net"] is None


class TestPickThresholdOnCalibrationTail:
    def test_picks_threshold_maximizing_sortino(self) -> None:
        # Tail proba sorted; only the top half wins. The optimal cutoff
        # sits near the median.
        rng = np.random.default_rng(0)
        n = 200
        proba = np.linspace(0.20, 0.95, n)
        y = (proba > 0.55).astype(int)
        gross_r = np.where(y == 1, 1.5, -1.0)
        cost_r = np.full(n, 0.32)
        thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80]

        chosen, why, sweep = pick_threshold_on_calibration_tail(
            tail_proba=proba, tail_y=y,
            tail_gross_r=gross_r, tail_cost_r=cost_r,
            thresholds=thresholds, cost_per_trade_r=0.32,
        )
        # Optimal threshold is at or just above 0.55 — picker can't pick
        # something below it because below 0.55 includes losers.
        assert chosen["threshold"] >= 0.55

    def test_tiebreak_prefers_higher_threshold(self) -> None:
        # Two thresholds with effectively identical Sortino — picker
        # picks the higher one.
        n = 100
        # All winners across all rows → Sortino is +inf at every threshold.
        # Use a deterministic dataset where Sortino is finite-equal at two
        # candidate thresholds.
        proba = np.linspace(0.1, 0.9, n)
        # Make winners distributed so 0.50 and 0.60 give same mean/std on
        # post-cost net.
        y = np.zeros(n, dtype=int)
        # 30 winners above 0.65, none between 0.50 and 0.65 — same
        # Sortino at thresholds 0.50, 0.55, 0.60.
        y[proba >= 0.65] = 1
        gross_r = np.where(y == 1, 1.0, -1.0)
        cost_r = np.full(n, 0.0)  # no cost so set is symmetric
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

        chosen, why, sweep = pick_threshold_on_calibration_tail(
            tail_proba=proba, tail_y=y,
            tail_gross_r=gross_r, tail_cost_r=cost_r,
            thresholds=thresholds, cost_per_trade_r=0.32,
            tiebreak_within_pct=0.05,
        )
        # 0.50, 0.55, 0.60 all yield the same trades → same Sortino.
        # 0.65 yields fewer trades but only winners → higher Sortino.
        # Picker should pick 0.65 (higher threshold) on tiebreak.
        assert chosen["threshold"] >= 0.60

    def test_raises_when_no_threshold_yields_trades(self) -> None:
        proba = np.full(50, 0.10)
        y = np.zeros(50, dtype=int)
        gross_r = np.zeros(50)
        cost_r = np.full(50, 0.32)
        with pytest.raises(MetaThresholdRejection, match="no threshold"):
            pick_threshold_on_calibration_tail(
                tail_proba=proba, tail_y=y,
                tail_gross_r=gross_r, tail_cost_r=cost_r,
                thresholds=[0.50, 0.60, 0.70],
                cost_per_trade_r=0.32,
            )


class TestEvaluateP1Gates:
    def test_all_pass_block(self) -> None:
        chosen = {"threshold": 0.65, "n_taken": 30, "take_rate": 0.30,
                  "expectancy_net_r": 0.18, "profit_factor": 1.55,
                  "win_rate": 0.60, "sortino_net": 0.5}
        verdicts = evaluate_p1_gates(
            brier=0.18,
            calibration_max_dev=0.03,
            chosen_threshold_row=chosen,
            per_fold_ar1_median=0.07,
            per_fold_ar1_max=0.14,
        )
        assert verdicts["all_pass"] is True
        for k in ("brier", "calibration", "take_rate", "profit_factor",
                  "expectancy", "ar1"):
            assert verdicts[k]["pass"] is True

    def test_brier_fail_blocks(self) -> None:
        chosen = {"threshold": 0.65, "n_taken": 30, "take_rate": 0.30,
                  "expectancy_net_r": 0.18, "profit_factor": 1.55,
                  "win_rate": 0.60, "sortino_net": 0.5}
        verdicts = evaluate_p1_gates(
            brier=0.30, calibration_max_dev=0.03,
            chosen_threshold_row=chosen,
            per_fold_ar1_median=0.07, per_fold_ar1_max=0.14,
        )
        assert verdicts["all_pass"] is False
        assert verdicts["brier"]["pass"] is False

    def test_calibration_none_treated_as_fail(self) -> None:
        chosen = {"threshold": 0.65, "n_taken": 30, "take_rate": 0.30,
                  "expectancy_net_r": 0.18, "profit_factor": 1.55,
                  "win_rate": 0.60, "sortino_net": 0.5}
        verdicts = evaluate_p1_gates(
            brier=0.18, calibration_max_dev=None,
            chosen_threshold_row=chosen,
            per_fold_ar1_median=0.07, per_fold_ar1_max=0.14,
        )
        assert verdicts["calibration"]["pass"] is False
        assert verdicts["all_pass"] is False

    def test_take_rate_outside_band_fails(self) -> None:
        chosen = {"threshold": 0.50, "n_taken": 80, "take_rate": 0.60,
                  "expectancy_net_r": 0.10, "profit_factor": 1.40,
                  "win_rate": 0.55, "sortino_net": 0.4}
        verdicts = evaluate_p1_gates(
            brier=0.18, calibration_max_dev=0.03,
            chosen_threshold_row=chosen,
            per_fold_ar1_median=0.07, per_fold_ar1_max=0.14,
        )
        assert verdicts["take_rate"]["pass"] is False
        assert verdicts["all_pass"] is False

    def test_ar1_max_above_gate_fails(self) -> None:
        chosen = {"threshold": 0.65, "n_taken": 30, "take_rate": 0.30,
                  "expectancy_net_r": 0.18, "profit_factor": 1.55,
                  "win_rate": 0.60, "sortino_net": 0.5}
        verdicts = evaluate_p1_gates(
            brier=0.18, calibration_max_dev=0.03,
            chosen_threshold_row=chosen,
            per_fold_ar1_median=0.07, per_fold_ar1_max=0.25,
        )
        assert verdicts["ar1"]["pass"] is False


class TestSelectPrimaryFires:
    def test_filters_to_buy_or_sell_above_threshold(self) -> None:
        proba = np.array([
            [0.60, 0.20, 0.20],   # SELL @ 0.60 ≥ 0.55 → fire short
            [0.20, 0.70, 0.10],   # HOLD → no fire
            [0.30, 0.20, 0.50],   # BUY @ 0.50 < 0.55 → no fire
            [0.05, 0.10, 0.85],   # BUY @ 0.85 → fire long
            [0.25, 0.50, 0.25],   # HOLD → no fire
        ])
        fires = select_primary_fires(
            primary_proba=proba,
            primary_decision_threshold=0.55,
            label_map={0: "SELL", 1: "HOLD", 2: "BUY"},
        )
        assert fires["mask"].tolist() == [True, False, False, True, False]
        assert fires["sides"].tolist() == [-1, 1]
        assert np.allclose(fires["max_proba"], [0.60, 0.85])

    def test_no_fires_returns_empty(self) -> None:
        proba = np.array([
            [0.30, 0.50, 0.20],
            [0.20, 0.50, 0.30],
        ])
        fires = select_primary_fires(
            primary_proba=proba,
            primary_decision_threshold=0.55,
            label_map={0: "SELL", 1: "HOLD", 2: "BUY"},
        )
        assert fires["mask"].sum() == 0
        assert fires["sides"].size == 0


class TestTrainMetaFold:
    def _synthetic_fold(self, n: int = 600, seed: int = 7) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        signal = rng.normal(size=n)
        feat2 = 0.6 * signal + 0.4 * rng.normal(size=n)
        noise = rng.normal(size=(n, 4))
        X = np.column_stack([signal, feat2, noise])
        p_true = 1.0 / (1.0 + np.exp(-1.5 * signal))
        y = (rng.uniform(size=n) < p_true).astype(int)
        w = np.ones(n)
        return {"X": X, "y": y, "w": w}

    def test_returns_calibrated_oos_proba(self) -> None:
        data = self._synthetic_fold(n=600, seed=11)
        cut = 420
        out = train_meta_fold(
            X_train=data["X"][:cut], y_train=data["y"][:cut],
            w_train=data["w"][:cut],
            X_test=data["X"][cut:],
            calibration_tail_frac=0.25,
            seed=42,
        )
        assert "base_clf" in out
        assert "isotonic" in out
        assert "scaler" in out
        proba = out["oos_calibrated_proba"]
        assert proba.shape == (data["X"].shape[0] - cut,)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_returns_tail_artefacts_for_threshold_picking(self) -> None:
        """The fold trainer must expose calibration-tail proba/y so the
        picker can choose the operating threshold without peeking at OOS."""
        data = self._synthetic_fold(n=600, seed=14)
        cut = 420
        out = train_meta_fold(
            X_train=data["X"][:cut], y_train=data["y"][:cut],
            w_train=data["w"][:cut],
            X_test=data["X"][cut:],
            calibration_tail_frac=0.25,
            seed=42,
        )
        for k in ("tail_proba_calibrated", "tail_y", "tail_indices"):
            assert k in out, f"train_meta_fold must expose {k}"
        # Tail size matches calibration_tail_frac * len(train).
        expected = int(round(0.25 * cut))
        assert len(out["tail_proba_calibrated"]) == expected
        assert len(out["tail_y"]) == expected

    def test_calibration_disabled_returns_no_isotonic(self) -> None:
        data = self._synthetic_fold(n=300, seed=18)
        out = train_meta_fold(
            X_train=data["X"][:200], y_train=data["y"][:200],
            w_train=data["w"][:200],
            X_test=data["X"][200:],
            calibration_tail_frac=0.0,
            seed=42,
        )
        assert out["isotonic"] is None


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
