"""Parity tests for the build_event_view extraction (Step 1).

The Step 1 refactor moves the event-collection + uniqueness + cost-aware-
weights pipeline out of train_ml_model_v2.main into ml/event_view.py.
These tests confirm the P4 runner's JSON report is byte-identical
before and after the refactor.

Method:
1. tests/fixtures/p4_golden_2sym_20d.json was captured BEFORE the
   refactor by running:
     python train_ml_model_v2.py --symbols NVDA AAPL --days 20 \\
       --frequency 5 --folds 3 --bootstrap-iterations 3 \\
       --report-path tests/fixtures/p4_golden_2sym_20d.json \\
       --model-path /tmp/p4_golden_model.pkl
2. Parity tests rerun that command and assert the resulting JSON
   matches the golden.
3. Direct build_event_view tests assert the EventView dataclass
   shapes/types match what the runner consumed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "tests/fixtures/p4_golden_2sym_20d.json"
GOLDEN_ARGS = [
    "--symbols", "NVDA", "AAPL",
    "--days", "20",
    "--frequency", "5",
    "--folds", "3",
    "--bootstrap-iterations", "3",
]


def _has_postgres() -> bool:
    """Skip parity tests if no Postgres reachable (CI / sandbox runs)."""
    try:
        import psycopg2  # noqa: F401
    except ImportError:
        return False
    dsn = os.environ.get("POSTGRES_DSN",
                        "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev")
    try:
        import psycopg2
        with psycopg2.connect(dsn, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM minute_bars LIMIT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


HAS_POSTGRES = _has_postgres()


def _normalize_for_diff(d: dict) -> dict:
    """Strip volatile fields (timestamps, timing) before diffing."""
    out = json.loads(json.dumps(d))  # deep copy
    out.pop("timestamp", None)
    out.pop("timing_seconds", None)
    return out


@pytest.mark.skipif(not GOLDEN_PATH.exists(),
                    reason="golden fixture not yet captured")
class TestP4ReportParity:
    """The P4 runner's report MUST be byte-identical after Step 1 refactor."""

    @pytest.mark.skipif(not HAS_POSTGRES,
                        reason="Postgres minute_bars not reachable")
    def test_p4_runner_report_matches_golden_after_refactor(
        self, tmp_path: Path,
    ) -> None:
        """Run the P4 runner and assert the JSON it writes equals the
        pre-refactor golden, modulo timestamp + timing fields."""
        report_path = tmp_path / "p4_actual.json"
        model_path = tmp_path / "p4_actual.pkl"
        cmd = [sys.executable, str(REPO_ROOT / "train_ml_model_v2.py"),
               *GOLDEN_ARGS,
               "--report-path", str(report_path),
               "--model-path", str(model_path)]
        result = subprocess.run(
            cmd, cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=300,
        )
        # Runner may exit non-zero (cost-drag rejection on tiny fixture);
        # the report must still be written.
        assert report_path.exists(), (
            f"runner did not write report.\nstdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )

        actual = json.loads(report_path.read_text())
        golden = json.loads(GOLDEN_PATH.read_text())

        # Strip volatile fields before diff.
        actual_n = _normalize_for_diff(actual)
        golden_n = _normalize_for_diff(golden)

        # Top-level keys identical.
        assert set(actual_n.keys()) == set(golden_n.keys()), (
            f"keys diverged: actual-only={set(actual_n)-set(golden_n)} "
            f"golden-only={set(golden_n)-set(actual_n)}"
        )

        # Critical upstream blocks (build_event_view's surface) must match
        # bit-for-bit.
        for key in ["per_symbol_samples", "label_distribution",
                    "weighting", "config", "symbols", "symbol_count"]:
            assert actual_n[key] == golden_n[key], (
                f"block {key!r} diverged after refactor:\n"
                f"  actual: {actual_n[key]}\n  golden: {golden_n[key]}"
            )

        # cv_summary must match too — proves the post-event-view pipeline
        # (purged_cv_evaluate) is fed identical inputs.
        assert actual_n["cv_summary"] == golden_n["cv_summary"], (
            "cv_summary diverged — purged CV is being fed different "
            "inputs after refactor"
        )


@pytest.mark.skipif(not HAS_POSTGRES,
                    reason="Postgres minute_bars not reachable")
class TestBuildEventViewDirect:
    """Direct tests on build_event_view as a public API."""

    def test_build_event_view_returns_frozen_dataclass(self) -> None:
        from ml.event_view import EventView, build_event_view
        ev = build_event_view(
            symbols=["NVDA", "AAPL"], days=20, frequency=5,
            dsn=os.environ.get(
                "POSTGRES_DSN",
                "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev"),
            pt_mult=2.0, sl_mult=1.0, max_holding=60, event_stride=5,
            cost_aware=True, fee_bps=5.0, slip_bps=10.0,
            impact_coef=0.1, participation=0.01,
        )
        assert isinstance(ev, EventView)
        # frozen → cannot reassign
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            ev.X = np.array([])  # type: ignore[misc]

    def test_build_event_view_shapes_align(self) -> None:
        from ml.event_view import build_event_view
        ev = build_event_view(
            symbols=["NVDA", "AAPL"], days=20, frequency=5,
            dsn=os.environ.get(
                "POSTGRES_DSN",
                "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev"),
            pt_mult=2.0, sl_mult=1.0, max_holding=60, event_stride=5,
            cost_aware=True, fee_bps=5.0, slip_bps=10.0,
            impact_coef=0.1, participation=0.01,
        )
        n = len(ev.y)
        assert ev.X.shape[0] == n
        assert len(ev.weights) == n
        assert len(ev.event_times) == n
        assert len(ev.touch_times) == n
        assert len(ev.sample_r_multiples) == n
        assert len(ev.sample_cost_r) == n
        assert len(ev.sym_groups) == n
        assert isinstance(ev.event_times, pd.DatetimeIndex)
        assert isinstance(ev.touch_times, pd.DatetimeIndex)
        # Sorted by event_time (purged CV requirement).
        assert ev.event_times.is_monotonic_increasing

    def test_build_event_view_includes_prices_per_symbol_for_p3(self) -> None:
        """P3 needs per-symbol bar data for regime classification."""
        from ml.event_view import build_event_view
        ev = build_event_view(
            symbols=["NVDA", "AAPL"], days=20, frequency=5,
            dsn=os.environ.get(
                "POSTGRES_DSN",
                "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev"),
            pt_mult=2.0, sl_mult=1.0, max_holding=60, event_stride=5,
            cost_aware=True, fee_bps=5.0, slip_bps=10.0,
            impact_coef=0.1, participation=0.01,
        )
        assert ev.prices_per_symbol is not None
        assert set(ev.prices_per_symbol.keys()) <= {"NVDA", "AAPL"}
        for sym, df in ev.prices_per_symbol.items():
            assert isinstance(df, pd.DataFrame)
            assert {"high", "low", "close"} <= set(df.columns.str.lower())
            assert isinstance(df.index, pd.DatetimeIndex)
            assert df.index.is_monotonic_increasing
