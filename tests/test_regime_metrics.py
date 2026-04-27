"""Unit tests for ml/regime_metrics.py — per-regime aggregator.

Wraps ml/evaluation.py and ml/iid_diagnostics.py primitives to bucket
trades by their committed-regime label and report PF/expectancy/win-rate/
n_trades/AR(1)/cost_drag PER REGIME.

Per-fold AR(1) is computed only when the regime has n>=30 trades in
that fold (small-sample autocorrelation is meaningless). Below that
floor the AR(1) field is None for that fold-regime cell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


REGIMES = (
    "trend_up_low_vol", "trend_up_high_vol",
    "trend_dn_low_vol", "trend_dn_high_vol",
    "range_tight", "range_wide", "chop",
)


def _new_agg():
    from ml.regime_metrics import PerRegimeAggregator
    return PerRegimeAggregator(regimes=REGIMES)


class TestPerRegimeAggregator:

    def test_separates_buckets(self) -> None:
        agg = _new_agg()
        # 50 winning trades labeled trend_up_low_vol (PF=Inf since no losses)
        agg.add_fold(
            fold_idx=0,
            regime_labels=np.array(["trend_up_low_vol"] * 50
                                   + ["range_wide"] * 50, dtype=object),
            gross_r=np.concatenate([np.ones(50) * 2.0,    # +2R each
                                    np.ones(50) * -1.0]),  # -1R each
            net_r=np.concatenate([np.ones(50) * 1.7,
                                  np.ones(50) * -1.3]),
        )
        out = agg.summarize()
        assert out["trend_up_low_vol"]["total_n_trades"] == 50
        assert out["range_wide"]["total_n_trades"] == 50
        # trend_up: all wins → PF infinite or "all wins"
        assert out["trend_up_low_vol"]["median_pf"] > 1.3
        assert out["trend_up_low_vol"]["median_expectancy_r"] > 0
        # range_wide: all losses → PF=0
        assert out["range_wide"]["median_pf"] == 0.0
        assert out["range_wide"]["median_expectancy_r"] < 0

    def test_expectancy_correctness(self) -> None:
        agg = _new_agg()
        # Trades = [+1, +1, -1] in trend_up_low_vol → expectancy = 0.333R
        agg.add_fold(
            fold_idx=0,
            regime_labels=np.array(["trend_up_low_vol"] * 3, dtype=object),
            gross_r=np.array([1.0, 1.0, -1.0]),
            net_r=np.array([0.7, 0.7, -1.3]),
        )
        out = agg.summarize()
        # Net expectancy: (0.7 + 0.7 - 1.3) / 3 = 0.0333...
        assert out["trend_up_low_vol"]["median_expectancy_r"] == pytest.approx(
            0.0333, abs=1e-3,
        )

    def test_win_rate_correctness(self) -> None:
        agg = _new_agg()
        # 7 wins / 10 trades in trend_up_high_vol
        agg.add_fold(
            fold_idx=0,
            regime_labels=np.array(["trend_up_high_vol"] * 10, dtype=object),
            gross_r=np.array([1, 1, 1, 1, 1, 1, 1, -1, -1, -1], dtype=float),
            net_r=np.array([0.7] * 7 + [-1.3] * 3),
        )
        out = agg.summarize()
        assert out["trend_up_high_vol"]["median_win_rate"] == 0.7

    def test_handles_empty_regime(self) -> None:
        agg = _new_agg()
        # No trades anywhere → all regimes report n=0.
        agg.add_fold(
            fold_idx=0,
            regime_labels=np.array([], dtype=object),
            gross_r=np.array([], dtype=float),
            net_r=np.array([], dtype=float),
        )
        out = agg.summarize()
        for r in REGIMES:
            assert out[r]["total_n_trades"] == 0
            assert out[r]["median_pf"] is None
            assert out[r]["median_expectancy_r"] is None
            assert out[r]["median_win_rate"] is None

    def test_ar1_only_computed_when_n_geq_30(self) -> None:
        agg = _new_agg()
        # 25 trades — under the 30-trade floor → AR(1) is None per fold.
        rng = np.random.default_rng(42)
        n = 25
        agg.add_fold(
            fold_idx=0,
            regime_labels=np.array(["trend_up_low_vol"] * n, dtype=object),
            gross_r=rng.normal(size=n),
            net_r=rng.normal(size=n),
            residuals=rng.normal(size=n),
        )
        out = agg.summarize()
        assert out["trend_up_low_vol"]["per_fold_ar1"] == [None]
        assert out["trend_up_low_vol"]["median_per_fold_ar1"] is None

    def test_ar1_computed_above_30(self) -> None:
        agg = _new_agg()
        rng = np.random.default_rng(43)
        n = 50
        agg.add_fold(
            fold_idx=0,
            regime_labels=np.array(["trend_up_low_vol"] * n, dtype=object),
            gross_r=rng.normal(size=n),
            net_r=rng.normal(size=n),
            residuals=rng.normal(size=n),
        )
        out = agg.summarize()
        assert isinstance(out["trend_up_low_vol"]["per_fold_ar1"][0], float)
        assert out["trend_up_low_vol"]["median_per_fold_ar1"] is not None

    def test_aggregate_across_regimes_excludes_chop_when_no_chop_trades(self) -> None:
        """Chop is forbidden by gate C; if no chop trades exist, aggregate
        across regimes should equal sum of non-chop regimes."""
        agg = _new_agg()
        agg.add_fold(
            fold_idx=0,
            regime_labels=np.array(["trend_up_low_vol"] * 10
                                   + ["range_wide"] * 10, dtype=object),
            gross_r=np.concatenate([np.ones(10) * 2.0, np.ones(10) * -1.0]),
            net_r=np.concatenate([np.ones(10) * 1.7, np.ones(10) * -1.3]),
        )
        out = agg.summarize()
        agg_block = out["__aggregate__"]
        assert agg_block["total_n_trades"] == 20
        # Aggregate expectancy = mean of all non-chop net R.
        expected = (np.ones(10) * 1.7).sum() + (np.ones(10) * -1.3).sum()
        assert agg_block["total_n_trades"] * agg_block["median_expectancy_r"] == \
               pytest.approx(expected, abs=1e-2) or \
               agg_block["median_expectancy_r"] == \
               pytest.approx(expected / 20, abs=1e-3)

    def test_chop_zero_trades_invariant(self) -> None:
        """Gate-C contract: if no chop labels were ever submitted,
        chop reports total_n_trades == 0 across all folds."""
        agg = _new_agg()
        agg.add_fold(
            fold_idx=0,
            regime_labels=np.array(["trend_up_low_vol"] * 5, dtype=object),
            gross_r=np.ones(5),
            net_r=np.ones(5),
        )
        out = agg.summarize()
        assert out["chop"]["total_n_trades"] == 0
        assert out["chop"]["per_fold_n_trades"] == [0]

    def test_multi_fold_median_aggregation(self) -> None:
        """Per-regime metrics across folds are reported as median, not mean
        — guards against single-fold luck."""
        agg = _new_agg()
        # Fold 0: PF=Inf (all wins), expectancy=2.0
        agg.add_fold(
            fold_idx=0,
            regime_labels=np.array(["trend_up_low_vol"] * 10, dtype=object),
            gross_r=np.ones(10) * 2.0,
            net_r=np.ones(10) * 2.0,
        )
        # Fold 1: PF=0.5, expectancy=-0.333
        agg.add_fold(
            fold_idx=1,
            regime_labels=np.array(["trend_up_low_vol"] * 6, dtype=object),
            gross_r=np.array([1, 1, -1, -1, -1, -1], dtype=float),
            net_r=np.array([1, 1, -1, -1, -1, -1], dtype=float),
        )
        out = agg.summarize()
        # Median expectancy: median(2.0, -0.333) = (2.0 + -0.333)/2 = 0.833
        assert out["trend_up_low_vol"]["median_expectancy_r"] == pytest.approx(
            0.8333, abs=1e-3,
        )
        assert out["trend_up_low_vol"]["total_n_trades"] == 16

    def test_per_fold_n_trades_tracked(self) -> None:
        agg = _new_agg()
        agg.add_fold(
            fold_idx=0,
            regime_labels=np.array(["trend_up_low_vol"] * 7
                                   + ["chop"] * 0, dtype=object),
            gross_r=np.ones(7), net_r=np.ones(7),
        )
        agg.add_fold(
            fold_idx=1,
            regime_labels=np.array(["trend_up_low_vol"] * 12, dtype=object),
            gross_r=np.ones(12), net_r=np.ones(12),
        )
        out = agg.summarize()
        assert out["trend_up_low_vol"]["per_fold_n_trades"] == [7, 12]
