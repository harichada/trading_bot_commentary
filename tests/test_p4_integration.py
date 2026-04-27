"""Integration tests for the P4 trainer (triple-barrier + sequential bootstrap + IID).

These tests stub the Postgres collector with synthetic samples and assert the
end-to-end pipeline produces a report with the new P4 schema, and that the
P4-intrinsic hard-reject gates fire on degenerate inputs.

Hard-reject gates per the user override (P4-intrinsic only):
  * IID weighted-residual AR(1) < 0.10
  * uniqueness weights present (mean ∈ (0, 1], min > 0)
  * label distribution sane (≥3 classes used or single class < 95% share)
  * report schema compatible (all required keys present)
P2 economic anchors are *warn-only* in the diff table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


def _synthesize_samples(
    n: int,
    *,
    n_features: int,
    profitable: bool,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    noise = rng.normal(size=(n, max(n_features - 2, 1)))
    feat2 = 0.5 * signal + 0.5 * rng.normal(size=n)
    X = np.column_stack([signal, feat2, noise])[:, :n_features]

    if profitable:
        base_prob_up = 0.5 + 0.25 * np.tanh(signal)
        flips = rng.uniform(size=n) < 0.08
    else:
        base_prob_up = np.full(n, 0.5)
        flips = rng.uniform(size=n) < 0.5

    y = np.where(rng.uniform(size=n) < base_prob_up, 2, 0)
    y = np.where(flips, 1, y)
    y = y.astype(np.int64)

    ret = np.where(
        y == 2, rng.uniform(0.010, 0.025, size=n),
        np.where(y == 0, -rng.uniform(0.005, 0.012, size=n),
                 rng.normal(0, 0.002, size=n)),
    )

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
    import train_ml_model_v2 as mod

    class _Stub:
        def __init__(self, *_a, **_kw) -> None:
            self.feature_names = [f"f{i}" for i in range(n_features)]
            self.feature_extractor = None

    monkeypatch.setattr(mod, "IntegratedMLModel", _Stub, raising=False)


@pytest.fixture
def isolated_paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "model.pkl", tmp_path / "report.json"


# --- Schema --------------------------------------------------------------------


class TestP4ReportSchema:
    def test_report_contains_all_p4_blocks(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolated_paths: tuple[Path, Path],
    ) -> None:
        n_features = 6
        block = _synthesize_samples(n=1500, n_features=n_features, profitable=True, seed=42)
        _install_stub_collector(monkeypatch, block)
        _install_stub_feature_names(monkeypatch, n_features)

        import train_ml_model_v2 as mod

        model_path, report_path = isolated_paths
        argv = [
            "train_ml_model_v2.py", "--symbols", "SYN",
            "--folds", "3", "--embargo-pct", "0.01",
            "--fee-bps", "5", "--slip-bps", "10", "--impact-coef", "0.1",
            "--model-path", str(model_path),
            "--report-path", str(report_path),
            "--cost-aware",
            "--reject-cost-drag-pct", "1000",
            "--bootstrap-iterations", "2",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        exit_code = mod.main()

        assert exit_code == 0
        report = json.loads(report_path.read_text())

        # Top-level P4 blocks.
        assert "labeling" in report["config"]
        assert report["config"]["labeling"]["method"] == "triple_barrier"
        assert "bar_type" in report["config"]["labeling"]
        assert report["config"]["labeling"]["bar_type"] == "time"
        assert "bar_type_deferred" in report["config"]["labeling"]

        assert "bootstrap" in report["config"]
        assert report["config"]["bootstrap"]["method"] == "sequential"
        assert report["config"]["bootstrap"]["iterations"] == 2

        # New diagnostic blocks.
        assert "sample_uniqueness" in report
        u = report["sample_uniqueness"]
        for k in ("mean", "median", "min", "effective_sample_size"):
            assert k in u

        assert "iid_diagnostics" in report
        iid = report["iid_diagnostics"]
        for k in ("per_fold_ar1", "median_per_fold_ar1", "max_per_fold_ar1",
                  "passes_per_fold_gate", "gate_thresholds",
                  "stitched_residual_ar1", "stitched_residual_ar1_note"):
            assert k in iid, f"missing iid key: {k}"
        assert isinstance(iid["passes_per_fold_gate"], bool)
        assert isinstance(iid["per_fold_ar1"], list)
        assert iid["gate_thresholds"]["median_lt"] == 0.10
        assert iid["gate_thresholds"]["max_lt"] == 0.20

        assert "bootstrap_stability" in report
        bs = report["bootstrap_stability"]
        for k in ("iterations", "expectancy_r_per_iter", "profit_factor_per_iter",
                  "cv_of_expectancy"):
            assert k in bs
        assert bs["iterations"] == 2
        assert len(bs["expectancy_r_per_iter"]) == 2

        assert "dropped_low_magnitude" in report["label_distribution"]
        assert "dropped_no_barrier_hit" in report["label_distribution"]


# --- Hard-reject gates (P4-intrinsic only) -----------------------------------


class TestP4HardRejectGates:
    def test_iid_failure_triggers_reject(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolated_paths: tuple[Path, Path],
    ) -> None:
        """If weighted-residual AR(1) >= 0.10, the trainer must reject.

        We force the failure by monkeypatching ``compute_iid_diagnostics`` to
        return a high AR(1) — bypasses the need to construct a degenerate
        sample set that would itself violate other assertions.
        """
        n_features = 6
        block = _synthesize_samples(n=1500, n_features=n_features, profitable=True, seed=42)
        _install_stub_collector(monkeypatch, block)
        _install_stub_feature_names(monkeypatch, n_features)

        import train_ml_model_v2 as mod
        from ml.iid_diagnostics import IIDReport

        bad = IIDReport(
            ar1_residual_autocorr=0.42,
            ljung_box_p_value=0.001,
            passes_ar1_lt_0_1=False,
            n_residuals=1500,
        )
        # Stub returns the same high AR(1) for every per-fold call → median
        # 0.42 ≥ 0.10 and max 0.42 ≥ 0.20 → gate fails on both bounds.
        monkeypatch.setattr(mod, "compute_iid_diagnostics", lambda *_a, **_kw: bad)

        model_path, report_path = isolated_paths
        argv = [
            "train_ml_model_v2.py", "--symbols", "SYN",
            "--folds", "3", "--embargo-pct", "0.01",
            "--model-path", str(model_path),
            "--report-path", str(report_path),
            "--reject-cost-drag-pct", "1000",
            "--bootstrap-iterations", "2",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        exit_code = mod.main()

        assert exit_code == 3, "expected P4 hard-reject exit code on per-fold AR(1) gate fail"
        report = json.loads(report_path.read_text())
        assert report.get("rejected") is True
        reason = (report.get("reject_reason") or "").lower()
        assert "ar(1)" in reason or "ar1" in reason

    def test_p2_economic_metrics_are_warn_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolated_paths: tuple[Path, Path],
    ) -> None:
        """P2 anchor regression must NOT cause exit 3 (warn-only per user override)."""
        n_features = 6
        # Unprofitable block — expectancy will be negative, worse than baseline.
        block = _synthesize_samples(n=1500, n_features=n_features, profitable=False, seed=99)
        _install_stub_collector(monkeypatch, block)
        _install_stub_feature_names(monkeypatch, n_features)

        # Write a baseline JSON with strict targets the new run cannot meet.
        tmp_baseline = isolated_paths[1].parent / "baseline.json"
        tmp_baseline.write_text(json.dumps({
            "cv_summary": {"aggregate": {
                "median_profit_factor": 5.0,           # impossible to beat
                "median_expectancy_r": 5.0,             # impossible to beat
                "n_folds_with_undefined_cost_drag": 0,  # impossible
                "median_absolute_cost_per_trade_r": 0.0,  # impossible
            }}
        }))

        import train_ml_model_v2 as mod

        model_path, report_path = isolated_paths
        argv = [
            "train_ml_model_v2.py", "--symbols", "SYN",
            "--folds", "3", "--embargo-pct", "0.01",
            "--model-path", str(model_path),
            "--report-path", str(report_path),
            "--reject-cost-drag-pct", "1e6",
            "--bootstrap-iterations", "2",
            "--regression-check", str(tmp_baseline),
        ]
        monkeypatch.setattr(sys, "argv", argv)

        exit_code = mod.main()

        # Must NOT be the P4 hard-reject code (3).
        assert exit_code != 3, "P2 economic regression must be warn-only, not hard-reject"


class TestEventSourceFlag:
    def test_stride_event_source_accepted(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolated_paths: tuple[Path, Path],
    ) -> None:
        n_features = 6
        block = _synthesize_samples(n=1500, n_features=n_features, profitable=True, seed=7)
        _install_stub_collector(monkeypatch, block)
        _install_stub_feature_names(monkeypatch, n_features)

        import train_ml_model_v2 as mod

        model_path, report_path = isolated_paths
        argv = [
            "train_ml_model_v2.py", "--symbols", "SYN",
            "--folds", "3", "--embargo-pct", "0.01",
            "--model-path", str(model_path),
            "--report-path", str(report_path),
            "--reject-cost-drag-pct", "1000",
            "--bootstrap-iterations", "1",
            "--event-source", "stride",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        exit_code = mod.main()

        assert exit_code == 0
        report = json.loads(report_path.read_text())
        assert report["config"]["labeling"].get("event_source") == "stride"

    def test_cusum_event_source_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolated_paths: tuple[Path, Path],
    ) -> None:
        n_features = 6
        block = _synthesize_samples(n=1500, n_features=n_features, profitable=True, seed=8)
        _install_stub_collector(monkeypatch, block)
        _install_stub_feature_names(monkeypatch, n_features)

        import train_ml_model_v2 as mod

        model_path, report_path = isolated_paths
        argv = [
            "train_ml_model_v2.py", "--symbols", "SYN",
            "--folds", "3", "--embargo-pct", "0.01",
            "--model-path", str(model_path),
            "--report-path", str(report_path),
            "--reject-cost-drag-pct", "1000",
            "--bootstrap-iterations", "1",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        exit_code = mod.main()

        assert exit_code == 0
        report = json.loads(report_path.read_text())
        assert report["config"]["labeling"].get("event_source") == "cusum"
