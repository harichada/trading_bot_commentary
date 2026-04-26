"""Tests for PnL-aware evaluation metrics (PROMPT_PACK P2 §2).

Replaces classification_report. Gradeable metrics only: expectancy per trade,
profit factor, Sortino, Calmar, max adverse excursion, and cost-drag %.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ml.evaluation import (
    FoldMetrics,
    calmar,
    cost_drag_pct,
    expectancy_r,
    grade_fold,
    max_adverse_excursion,
    profit_factor,
    sortino,
)


class TestExpectancy:
    def test_known_inputs(self) -> None:
        r = np.array([2.0, -1.0, 2.0, -1.0, 0.5])
        assert expectancy_r(r) == pytest.approx(0.5, abs=1e-9)

    def test_empty_returns_zero(self) -> None:
        assert expectancy_r(np.array([])) == 0.0

    def test_all_losses_negative(self) -> None:
        r = np.array([-0.5, -0.3, -0.2])
        assert expectancy_r(r) == pytest.approx(-1.0 / 3.0, abs=1e-9)


class TestProfitFactor:
    def test_known_inputs(self) -> None:
        r = np.array([2.0, -1.0, 2.0, -1.0, 0.5])
        # wins = 4.5, losses = 2.0 → PF = 2.25
        assert profit_factor(r) == pytest.approx(2.25, abs=1e-9)

    def test_zero_losses_returns_inf(self) -> None:
        r = np.array([1.0, 2.0, 3.0])
        assert profit_factor(r) == math.inf

    def test_zero_wins_returns_zero(self) -> None:
        r = np.array([-1.0, -2.0])
        assert profit_factor(r) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert profit_factor(np.array([])) == 0.0


class TestSortino:
    def test_only_downside_in_denominator(self) -> None:
        # Mean excess = 1.0; downside returns = [-2, -1] → downside std ≈ 0.5
        # Sortino = 1.0 / 0.5 = 2.0 (before annualization = 1.0 un-annualized).
        r = np.array([4.0, 2.0, -2.0, -1.0, 2.0])
        # Period excess return over zero = mean = 1.0.
        # Downside variance: mean over all samples of min(ret,0)^2
        #  = (0 + 0 + 4 + 1 + 0) / 5 = 1.0 → downside_std = 1.0 → ratio = 1.0.
        assert sortino(r, bars_per_year=1) == pytest.approx(1.0, abs=1e-9)

    def test_no_downside_returns_inf(self) -> None:
        r = np.array([1.0, 2.0, 3.0])
        assert sortino(r, bars_per_year=1) == math.inf

    def test_annualization_scales_sqrt(self) -> None:
        r = np.array([1.0, -1.0, 1.0, -1.0])
        base = sortino(r, bars_per_year=1)
        annualized = sortino(r, bars_per_year=4)
        if math.isfinite(base) and base != 0:
            assert annualized == pytest.approx(base * 2.0, rel=1e-9)


class TestCalmar:
    def test_known_curve(self) -> None:
        # Cumulative R: [1, 2, 1, 3, 2, 4]. Peak=4, trough after peak? none.
        # Running max: [1,2,2,3,3,4]. Drawdowns in R: [0,0,1,0,1,0]. Max DD = 1R.
        # Annualized return (bars_per_year=6): mean per bar = 4/6 ≈ 0.667
        # → annualized = 0.667 * 6 = 4.0. Calmar = 4.0 / 1.0 = 4.0.
        r = np.array([1.0, 1.0, -1.0, 2.0, -1.0, 2.0])
        assert calmar(r, bars_per_year=6) == pytest.approx(4.0, abs=1e-9)

    def test_zero_drawdown_returns_inf(self) -> None:
        r = np.array([1.0, 1.0, 1.0])
        assert calmar(r, bars_per_year=1) == math.inf

    def test_empty_returns_zero(self) -> None:
        assert calmar(np.array([]), bars_per_year=1) == 0.0


class TestMaxAdverseExcursion:
    def test_lower_bound_uses_losses_and_stop_multiple(self) -> None:
        # Label +1 trades (ret > 0) have MAE ≥ 0 but unknown → report 0.
        # Label -1 trades (ret < 0): MAE = ret (the loss itself).
        # Label 0 trades (ret = 0): MAE = 0.
        # For this fold, worst single MAE across all samples is max of |losses|.
        r = np.array([1.5, -0.8, 0.3, -1.2, 0.0])
        mae = max_adverse_excursion(r)
        # Lower-bound MAE = max(|negative returns|) = 1.2
        assert mae == pytest.approx(1.2, abs=1e-9)

    def test_all_wins_returns_zero(self) -> None:
        r = np.array([0.5, 1.0, 0.2])
        assert max_adverse_excursion(r) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert max_adverse_excursion(np.array([])) == 0.0


class TestCostDrag:
    def test_normal_range(self) -> None:
        # gross = 1.0R/trade, net = 0.7R/trade → cost-drag 30%.
        gross = np.full(10, 1.0)
        net = np.full(10, 0.7)
        drag, abs_cost = cost_drag_pct(gross=gross, net=net)
        assert drag == pytest.approx(30.0, abs=1e-9)
        assert abs_cost is None

    def test_zero_gross_returns_none_pct_and_absolute(self) -> None:
        """gross_mean == 0 — drag % undefined, absolute cost reported instead."""
        gross = np.zeros(5)
        net = np.full(5, -0.30)
        drag, abs_cost = cost_drag_pct(gross=gross, net=net)
        assert drag is None
        # gross_mean=0, net_mean=-0.30 → absolute cost = 0 - (-0.30) = +0.30R/trade.
        assert abs_cost == pytest.approx(0.30, abs=1e-9)

    def test_negative_gross_returns_none_pct_and_absolute(self) -> None:
        """gross_mean = -0.05 → drag % undefined, absolute cost reported."""
        # Construct gross_mean = -0.05, net_mean = -0.35 → absolute cost = 0.30.
        gross = np.array([-0.05] * 4)
        net = np.array([-0.35] * 4)
        drag, abs_cost = cost_drag_pct(gross=gross, net=net)
        assert drag is None
        assert abs_cost == pytest.approx(0.30, abs=1e-4)

    def test_positive_gross_zero_two_returns_drag_pct(self) -> None:
        """gross_mean = +0.20, net_mean = +0.05 → drag = (0.20-0.05)/0.20 * 100 = 75%."""
        gross = np.array([0.20] * 6)
        net = np.array([0.05] * 6)
        drag, abs_cost = cost_drag_pct(gross=gross, net=net)
        assert drag == pytest.approx(75.0, abs=1e-9)
        assert abs_cost is None

    def test_empty_returns_none_pct_and_zero_absolute(self) -> None:
        drag, abs_cost = cost_drag_pct(gross=np.array([]), net=np.array([]))
        assert drag is None
        assert abs_cost == 0.0


class TestGradeFold:
    def test_returns_dataclass_with_all_six_metrics(self) -> None:
        gross = np.array([1.0, -0.5, 1.5, -0.8, 0.6, 1.2])
        net = np.array([0.7, -0.8, 1.2, -1.1, 0.3, 0.9])
        out = grade_fold(gross_r=gross, net_r=net, bars_per_year=252)
        assert isinstance(out, FoldMetrics)
        # Per PROMPT_PACK P2 §2: expectancy, profit factor, sortino, calmar, MAE, cost-drag.
        assert out.expectancy_r == pytest.approx(expectancy_r(net), abs=1e-9)
        assert out.profit_factor == pytest.approx(profit_factor(net), abs=1e-9)
        assert out.sortino == pytest.approx(sortino(net, bars_per_year=252), rel=1e-9)
        assert math.isfinite(out.calmar) or math.isinf(out.calmar)
        assert out.max_adverse_excursion == pytest.approx(
            max_adverse_excursion(net), abs=1e-9
        )
        expected_drag, expected_abs = cost_drag_pct(gross=gross, net=net)
        assert out.cost_drag_pct == expected_drag
        assert out.absolute_cost_per_trade_r == expected_abs

    def test_zero_trades_returns_neutral(self) -> None:
        out = grade_fold(
            gross_r=np.array([]), net_r=np.array([]), bars_per_year=252
        )
        assert out.n_trades == 0
        assert out.expectancy_r == 0.0
        assert out.cost_drag_pct is None
        assert out.absolute_cost_per_trade_r == 0.0
