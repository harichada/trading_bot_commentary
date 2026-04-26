"""Integration tests for cost-aware training pipeline (PROMPT_PACK P2).

These tests bypass the Postgres data layer by monkeypatching ``collect_v2_samples``
with deterministic synthetic samples, then assert the end-to-end pipeline:
    - fits a model per fold with PnL-weighted sample weights,
    - grades each fold with the six cost-aware metrics,
    - selects by **median** OOS expectancy (not mean),
    - hard-rejects when median cost-drag > 50%,
    - writes a JSON report containing per-fold + aggregate metrics.

No database, no network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


# --- Helpers ------------------------------------------------------------------


def _synthesize_samples(
    n: int,
    *,
    n_features: int,
    profitable: bool,
    seed: int,
    cost_heavy: bool = False,
) -> dict[str, Any]:
    """Build a synthetic sample block the trainer can consume.

    When ``profitable=True``, the label distribution is tilted so that an
    XGBoost classifier trained on the features has a positive expectancy
    after costs. When ``cost_heavy=True``, the per-sample cost overwhelms
    the edge so the pipeline should reject on cost-drag > 50%.
    """
    rng = np.random.default_rng(seed)
    # Two informative features + n_features-2 noise.
    signal = rng.normal(size=n)
    noise = rng.normal(size=(n, max(n_features - 2, 1)))
    feat2 = 0.5 * signal + 0.5 * rng.normal(size=n)
    X = np.column_stack([signal, feat2, noise])[:, :n_features]

    if profitable:
        # Label is mostly aligned with signal sign → model has edge.
        base_prob_up = 0.5 + 0.25 * np.tanh(signal)
        flips = rng.uniform(size=n) < 0.08
    else:
        base_prob_up = np.full(n, 0.5)
        flips = rng.uniform(size=n) < 0.5

    y = np.where(rng.uniform(size=n) < base_prob_up, 2, 0)  # 2=BUY, 0=SELL
    y = np.where(flips, 1, y)  # some HOLD samples
    y = y.astype(np.int64)

    # Returns in decimal; hit label=+1 → ~1.5R, −1 → ~-1R, 0 → tiny noise.
    # 1R = 1% here (ATR=1, entry=100).
    ret = np.where(
        y == 2, rng.uniform(0.010, 0.025, size=n),
        np.where(y == 0, -rng.uniform(0.005, 0.012, size=n),
                 rng.normal(0, 0.002, size=n)),
    )

    if cost_heavy:
        # Shrink raw returns so costs dominate edge.
        ret = ret * 0.25

    # Event/touch times: regularly spaced so purged CV carves cleanly.
    base = pd.Timestamp("2024-01-02 09:30")
    event_times = pd.DatetimeIndex(
        [base + pd.Timedelta(minutes=5 * i) for i in range(n)]
    )
    touch_times = event_times + pd.Timedelta(minutes=30)

    symbol_groups = np.array(["SYN"] * n, dtype=object)
    per_symbol = {"SYN": n}
    return dict(
        X=X, y=y, ret=ret,
        event_times=event_times, touch_times=touch_times,
        symbol_groups=symbol_groups, per_symbol=per_symbol,
    )


def _install_stub_collector(monkeypatch: pytest.MonkeyPatch, sample_block: dict) -> None:
    """Replace collect_v2_samples so the pipeline skips DB access."""
    import train_ml_model_v2 as mod

    def _stub(**_kwargs):
        return (
            sample_block["X"],
            sample_block["y"],
            sample_block["ret"],
            sample_block["event_times"],
            sample_block["touch_times"],
            sample_block["symbol_groups"],
            sample_block["per_symbol"],
        )

    monkeypatch.setattr(mod, "collect_v2_samples", _stub)


def _install_stub_feature_names(monkeypatch: pytest.MonkeyPatch, n_features: int) -> None:
    """Replace the feature-name lookup so tests don't instantiate the full model."""
    import train_ml_model_v2 as mod

    class _Stub:
        def __init__(self, *_a, **_kw) -> None:
            self.feature_names = [f"f{i}" for i in range(n_features)]
            self.feature_extractor = None  # unused in stub path

    # Patch the module-level symbol used inside the final-fit block.
    monkeypatch.setattr(mod, "IntegratedMLModel", _Stub, raising=False)


@pytest.fixture
def isolated_paths(tmp_path: Path) -> tuple[Path, Path]:
    model_path = tmp_path / "model.pkl"
    report_path = tmp_path / "report.json"
    return model_path, report_path


# --- Tests --------------------------------------------------------------------


class TestDryRunEmitsSixMetrics:
    def test_report_contains_cost_aware_metrics_per_fold_and_aggregate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolated_paths: tuple[Path, Path],
    ) -> None:
        n_features = 6
        block = _synthesize_samples(
            n=1500, n_features=n_features, profitable=True, seed=42
        )
        _install_stub_collector(monkeypatch, block)
        _install_stub_feature_names(monkeypatch, n_features)

        import train_ml_model_v2 as mod

        model_path, report_path = isolated_paths
        argv = [
            "train_ml_model_v2.py", "--symbols", "SYN",
            "--folds", "5", "--embargo-pct", "0.01",
            "--fee-bps", "5", "--slip-bps", "10", "--impact-coef", "0.1",
            "--model-path", str(model_path),
            "--report-path", str(report_path),
            "--cost-aware",
            # This test's purpose is to verify the report emits all six
            # cost-aware metrics. Disable the hard-reject so we reach the
            # final-fit + report block regardless of model edge on 1500
            # synthetic samples (the acceptance-gate test below exercises
            # the reject threshold with sized data).
            "--reject-cost-drag-pct", "1000",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        exit_code = mod.main()

        assert exit_code == 0, "expected pipeline to accept profitable dataset"
        report = json.loads(report_path.read_text())

        required = {
            "expectancy_r", "profit_factor", "sortino", "calmar",
            "max_adverse_excursion", "cost_drag_pct",
        }
        aggregate = report["cv_summary"]["aggregate"]
        assert set(aggregate.keys()) >= {
            "median_expectancy_r",
            "median_profit_factor",
            "median_cost_drag_pct",
            "var_sharpe_across_folds",
            "median_sortino",
            "n_folds_executed",
        }, f"aggregate missing keys: {aggregate.keys()}"

        per_fold = report["cv_summary"]["fold_detail"]
        assert len(per_fold) >= 3, "need ≥3 usable folds to trust median selection"
        for fold in per_fold:
            metrics = fold["metrics"]
            missing = required - set(metrics.keys())
            assert not missing, f"fold missing metric(s): {missing}"


class TestHardRejectOnCostDrag:
    def test_cost_drag_over_fifty_percent_triggers_reject(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolated_paths: tuple[Path, Path],
    ) -> None:
        n_features = 6
        block = _synthesize_samples(
            n=1500, n_features=n_features, profitable=True, seed=11,
            cost_heavy=True,
        )
        _install_stub_collector(monkeypatch, block)
        _install_stub_feature_names(monkeypatch, n_features)

        import train_ml_model_v2 as mod

        model_path, report_path = isolated_paths
        argv = [
            "train_ml_model_v2.py", "--symbols", "SYN",
            "--folds", "5", "--embargo-pct", "0.01",
            # Very high costs → cost-drag well above 50%.
            "--fee-bps", "40", "--slip-bps", "80", "--impact-coef", "0.5",
            "--model-path", str(model_path),
            "--report-path", str(report_path),
            "--cost-aware",
            "--reject-cost-drag-pct", "50",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        exit_code = mod.main()

        assert exit_code == 2, "expected hard-reject exit code on high cost-drag"
        report = json.loads(report_path.read_text())
        assert report.get("rejected") is True
        assert "cost_drag" in (report.get("reject_reason") or "").lower()


class TestMedianSelection:
    def test_median_accepts_where_mean_would_reject(self) -> None:
        """López de Prado (AFML §12.3) — one catastrophic fold shouldn't
        override four positive folds; median OOS expectancy is the right
        aggregator when fold variance is high. Mean of [0.5, 0.5, 0.5,
        0.5, -5.0] is -0.6; median is +0.5."""
        from ml.evaluation import FoldMetrics
        from train_ml_model_v2 import aggregate_fold_metrics

        folds = [
            FoldMetrics(n_trades=100, expectancy_r=0.5, profit_factor=1.8,
                        sortino=1.2, calmar=2.0,
                        max_adverse_excursion=2.0, cost_drag_pct=20.0),
            FoldMetrics(n_trades=100, expectancy_r=0.5, profit_factor=1.8,
                        sortino=1.2, calmar=2.0,
                        max_adverse_excursion=2.0, cost_drag_pct=20.0),
            FoldMetrics(n_trades=100, expectancy_r=0.5, profit_factor=1.8,
                        sortino=1.2, calmar=2.0,
                        max_adverse_excursion=2.0, cost_drag_pct=20.0),
            FoldMetrics(n_trades=100, expectancy_r=0.5, profit_factor=1.8,
                        sortino=1.2, calmar=2.0,
                        max_adverse_excursion=2.0, cost_drag_pct=20.0),
            FoldMetrics(n_trades=100, expectancy_r=-5.0, profit_factor=0.3,
                        sortino=-1.5, calmar=-0.5,
                        max_adverse_excursion=5.0, cost_drag_pct=25.0),
        ]

        agg = aggregate_fold_metrics(folds)

        assert agg["median_expectancy_r"] == pytest.approx(0.5, abs=1e-9)
        # Mean would be -0.0 → rejected; median is +0.5 → accepted.
        mean_exp = float(np.mean([f.expectancy_r for f in folds]))
        assert mean_exp < 0.0 < agg["median_expectancy_r"]
