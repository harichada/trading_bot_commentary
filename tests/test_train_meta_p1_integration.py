"""End-to-end tests for the P1 meta-labeling trainer.

Stubs out the Postgres market-data layer and the heavy IntegratedMLModel
feature extractor so the test runs offline. Exercises the full
pipeline: primary-fire selection, triple-barrier labelling, purged CV
with isotonic calibration, calibration-tail threshold picker, gate
evaluation, report writing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest


# ---- Synthetic primary bundle -----------------------------------------------


def _make_primary_bundle(tmp_path: Path, *, n_features: int) -> Path:
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    rng = np.random.default_rng(0)
    n = 800
    signal = rng.normal(size=n)
    noise = rng.normal(size=(n, max(n_features - 1, 1)))
    X = np.column_stack([signal, noise])[:, :n_features]
    p_buy = 0.33 + 0.25 * np.tanh(signal)
    p_sell = 0.33 - 0.25 * np.tanh(signal)
    u = rng.uniform(size=n)
    y = np.where(u < p_buy, 2, np.where(u < p_buy + p_sell, 0, 1)).astype(np.int64)

    scaler = StandardScaler().fit(X)
    clf = xgb.XGBClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.1,
        eval_metric="mlogloss", num_class=3,
        objective="multi:softprob", n_jobs=-1, random_state=0,
    )
    clf.fit(scaler.transform(X), y)

    bundle = {
        "version": 2,
        "classifier": clf,
        "scaler": scaler,
        "feature_names": [f"f{i}" for i in range(n_features)],
        "label_map": {0: "SELL", 1: "HOLD", 2: "BUY"},
        "config": {
            "pt_mult": 2.0, "sl_mult": 1.0, "max_holding": 60,
            "decision_threshold": 0.55,
            "labeling": {
                "method": "triple_barrier",
                "vertical_max_bars": 120,
                "min_ret_atr_mult": 1.0,
                "event_source": "cusum",
                "cusum_h": 0.005,
            },
            "frequency": 5,
            "cost_model": {
                "fee_bps": 5.0, "slip_bps": 10.0,
                "impact_coef": 0.1, "participation": 0.01,
            },
            "n_folds": 3, "embargo_pct": 0.01,
        },
    }
    out = tmp_path / "primary.pkl"
    joblib.dump(bundle, out)
    return out


def _install_stubs(monkeypatch: pytest.MonkeyPatch, *, n_features: int) -> None:
    """Stub Postgres provider + IntegratedMLModel feature extractor."""
    import train_meta_model_p1 as mod

    class _StubFeat:
        def __init__(self) -> None:
            self.feature_names = [f"f{i}" for i in range(n_features)]

        def batch_extract(self, df: pd.DataFrame) -> pd.DataFrame:
            # Generate features from the same N(0,1) distribution the
            # primary trained on so the primary will fire on a
            # meaningful subset of bars. Seed deterministically off the
            # bar index so a given timestamp produces the same feature.
            n = len(df)
            rng = np.random.default_rng(int(df.index[0].value) & 0xFFFFFFFF)
            base = rng.normal(size=(n, n_features))
            df_out = pd.DataFrame(
                base, index=df.index, columns=self.feature_names,
            )
            df_out.iloc[:50] = np.nan  # warmup
            return df_out

    class _StubModel:
        def __init__(self, *_a, **_kw) -> None:
            self.feature_extractor = _StubFeat()
            self.feature_names = self.feature_extractor.feature_names

    monkeypatch.setattr(mod, "IntegratedMLModel", _StubModel, raising=False)
    monkeypatch.setattr(mod, "top_symbols_by_volume",
                        lambda *_a, **_kw: ["SYN1", "SYN2", "SYN3"])

    def _stub_factory(_dsn: str):
        class P:
            def get_market_data(self, symbol, *, period_type, period,
                                frequency_type, frequency):
                rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
                n_bars = 1500
                idx = pd.date_range(
                    start="2024-09-02 09:30",
                    periods=n_bars, freq=f"{frequency}min",
                )
                ret = rng.normal(0, 0.003, size=n_bars)
                close = 100 * np.exp(np.cumsum(ret))
                df = pd.DataFrame({
                    "Open": close * (1 + rng.normal(0, 0.0005, size=n_bars)),
                    "High": close * (1 + np.abs(rng.normal(0, 0.001, size=n_bars))),
                    "Low": close * (1 - np.abs(rng.normal(0, 0.001, size=n_bars))),
                    "Close": close,
                    "Volume": rng.integers(10_000, 100_000, size=n_bars),
                }, index=idx)
                return df
        return P()

    monkeypatch.setattr(mod, "_make_provider", _stub_factory)


@pytest.fixture
def isolated_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "primary": _make_primary_bundle(tmp_path, n_features=6),
        "meta_bundle": tmp_path / "meta.pkl",
        "report": tmp_path / "p1.json",
    }


# ---- Schema -----------------------------------------------------------------


class TestP1ReportSchema:
    def test_report_contains_all_p1_blocks(
        self, monkeypatch: pytest.MonkeyPatch, isolated_paths: dict[str, Path],
    ) -> None:
        _install_stubs(monkeypatch, n_features=6)
        import train_meta_model_p1 as mod

        argv = [
            "train_meta_model_p1.py",
            "--symbols", "SYN1", "SYN2", "SYN3",
            "--days", "5",
            "--folds", "3",
            "--primary-bundle", str(isolated_paths["primary"]),
            "--meta-bundle-path", str(isolated_paths["meta_bundle"]),
            "--report-path", str(isolated_paths["report"]),
            "--allow-rejection",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        # Allow either pass (0) or rejection (3) — synthetic data may
        # not satisfy gates, but the schema must always be written.
        exit_code = mod.main()
        assert exit_code in (0, 3)

        report = json.loads(isolated_paths["report"].read_text())

        for k in ("timestamp", "primary_bundle", "config", "symbols",
                  "n_primary_fires", "label_distribution",
                  "weighting", "cv_summary",
                  "calibration", "threshold_sweep", "chosen_threshold",
                  "per_fold_thresholds", "threshold_agreement",
                  "iid_diagnostics", "gates"):
            assert k in report, f"missing top-level key: {k}"

        for g in ("brier", "calibration", "take_rate", "profit_factor",
                  "expectancy", "ar1"):
            assert g in report["gates"], f"missing gate: {g}"

        # IID diagnostics gate thresholds present.
        assert report["iid_diagnostics"]["gate_thresholds"]["median_lt"] == 0.10
        assert report["iid_diagnostics"]["gate_thresholds"]["max_lt"] == 0.20

        # Per-fold threshold visibility.
        per_fold = report["per_fold_thresholds"]
        assert isinstance(per_fold, list)

    def test_baseline_diff_block_when_provided(
        self, monkeypatch: pytest.MonkeyPatch,
        isolated_paths: dict[str, Path], tmp_path: Path,
    ) -> None:
        _install_stubs(monkeypatch, n_features=6)
        import train_meta_model_p1 as mod

        baseline = tmp_path / "p4_baseline.json"
        baseline.write_text(json.dumps({
            "cv_summary": {"aggregate": {
                "median_profit_factor": 0.738,
                "median_expectancy_r": -0.196,
                "median_absolute_cost_per_trade_r": 0.32,
            }},
            "iid_diagnostics": {
                "median_per_fold_ar1": 0.092,
                "max_per_fold_ar1": 0.169,
            },
        }))

        argv = [
            "train_meta_model_p1.py",
            "--symbols", "SYN1", "SYN2",
            "--days", "5",
            "--folds", "3",
            "--primary-bundle", str(isolated_paths["primary"]),
            "--meta-bundle-path", str(isolated_paths["meta_bundle"]),
            "--report-path", str(isolated_paths["report"]),
            "--baseline", str(baseline),
            "--allow-rejection",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        assert mod.main() in (0, 3)
        report = json.loads(isolated_paths["report"].read_text())
        assert "p4_baseline_diff" in report
        rows = report["p4_baseline_diff"]["rows"]
        assert "median_profit_factor" in rows
        assert "median_expectancy_r" in rows


# ---- Hard-reject paths ------------------------------------------------------


class TestNestedPrimaryRun:
    """End-to-end with --nested-primary: verifies the harness is wired,
    SHA256 of ml_model_v2.pkl is recorded and unchanged, and the report
    flags ``nested_cv: true``."""

    def test_nested_run_records_sha_and_flag(
        self, monkeypatch: pytest.MonkeyPatch, isolated_paths: dict[str, Path],
    ) -> None:
        _install_stubs(monkeypatch, n_features=6)
        import train_meta_model_p1 as mod

        # Capture SHA before run.
        import hashlib
        primary_path = isolated_paths["primary"]
        before_sha = hashlib.sha256(primary_path.read_bytes()).hexdigest()

        argv = [
            "train_meta_model_p1.py",
            "--symbols", "SYN1", "SYN2", "SYN3",
            "--days", "5",
            "--folds", "3",
            "--primary-bundle", str(primary_path),
            "--meta-bundle-path", str(isolated_paths["meta_bundle"]),
            "--report-path", str(isolated_paths["report"]),
            "--nested-primary",
            "--allow-rejection",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        exit_code = mod.main()
        assert exit_code in (0, 3)

        # Primary on disk must be byte-identical.
        after_sha = hashlib.sha256(primary_path.read_bytes()).hexdigest()
        assert before_sha == after_sha, "primary bundle must not be modified"

        report = json.loads(isolated_paths["report"].read_text())
        assert report["config"].get("nested_cv") is True
        assert "primary_bundle_sha256" in report
        assert report["primary_bundle_sha256"]["before"] == before_sha
        assert report["primary_bundle_sha256"]["after"] == before_sha
        assert report["primary_bundle_sha256"]["unchanged"] is True


class TestP1HardReject:
    def test_no_primary_fires_rejects(
        self, monkeypatch: pytest.MonkeyPatch, isolated_paths: dict[str, Path],
    ) -> None:
        _install_stubs(monkeypatch, n_features=6)
        import train_meta_model_p1 as mod

        argv = [
            "train_meta_model_p1.py",
            "--symbols", "SYN1",
            "--days", "5",
            "--folds", "3",
            "--primary-bundle", str(isolated_paths["primary"]),
            "--meta-bundle-path", str(isolated_paths["meta_bundle"]),
            "--report-path", str(isolated_paths["report"]),
            # Push the primary decision threshold so high nothing fires.
            "--decision-threshold-override", "0.999",
            "--allow-rejection",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        exit_code = mod.main()
        assert exit_code == 3
        report = json.loads(isolated_paths["report"].read_text())
        assert report["rejected"] is True
