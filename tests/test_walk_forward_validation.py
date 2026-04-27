"""Walk-forward validation gates W1-W5 for baseline-P4-retune.

Tests whether the retune's apparent edge (agg PF=0.838, exp=-0.083R on the
original 60-day window) holds on TWO disjoint hold-out windows. Read the
report at ``backtest_results/walk_forward_validation.json``; if the
aggregate metrics on held-out windows are far below the original, the
retune was a time-window artifact.

References
----------
* López de Prado, AFML §11 — combinatorial walk-forward CV; the discipline
  of validating against time-disjoint windows before declaring edge.
* AFML §12 — backtest overfitting; single-window 5-fold CV cannot
  control for time-window selection bias.

Gates
-----
W1 (replication): on the original 60-day training window, aggregate PF
   must be within ±0.05 of 0.838 AND aggregate expectancy within
   ±0.020R of -0.083R. Tighter than the held-out gates — this is a
   sanity / determinism check, not generalization.

W2 (oldest 60-day window, 2021-Q1): aggregate PF >= 0.70 AND aggregate
   expectancy >= -0.20R. Looser — different regime allowed.

W3 (mid-history 60-day window, 2022 bear): same threshold as W2.

W4 (median across all 3 windows): aggregate PF >= 0.78 AND expectancy
   >= -0.15R. The single-window result must hold on average.

W5 (release-bundle invariant): ``ml_model_v2_retune.pkl`` SHA256 byte-
   identical pre/post the entire validation run (every per-window
   bundle is a separate pkl; the released one is read-only).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WF_REPORT = REPO_ROOT / "backtest_results" / "walk_forward_validation.json"

# Hard thresholds (do NOT change without explicit user instruction).
W1_PF_TARGET = 0.838
W1_EXP_TARGET = -0.083
W1_PF_TOLERANCE = 0.05
W1_EXP_TOLERANCE = 0.020

W2_W3_PF_FLOOR = 0.70
W2_W3_EXP_FLOOR = -0.20

W4_MEDIAN_PF_FLOOR = 0.78
W4_MEDIAN_EXP_FLOOR = -0.15


@pytest.fixture(scope="module")
def wf_report() -> dict:
    if not WF_REPORT.exists():
        pytest.skip(
            f"Walk-forward report not found at {WF_REPORT} — run "
            "validate_walk_forward.py to produce it."
        )
    return json.loads(WF_REPORT.read_text())


def _agg(window: dict) -> tuple[float, float]:
    """Pull (median_pf, median_expectancy_r) from a per-window block."""
    a = window["aggregate_across_regimes"]
    pf = a.get("median_pf")
    exp_r = a.get("median_expectancy_r")
    assert pf is not None and exp_r is not None, (
        f"window aggregate metrics undefined: {window.get('label')}"
    )
    return float(pf), float(exp_r)


class TestWalkForwardGates:

    def test_w1_replication_within_tolerance(self, wf_report: dict) -> None:
        """W1 — original window replication. Tight tolerance (±0.05 PF,
        ±0.020R exp) — failure here means the harness or data layer
        has drifted from when baseline-P4-retune was committed."""
        w1 = wf_report["windows"]["W1"]
        pf, exp_r = _agg(w1)
        delta_pf = abs(pf - W1_PF_TARGET)
        delta_exp = abs(exp_r - W1_EXP_TARGET)
        assert delta_pf <= W1_PF_TOLERANCE, (
            f"W1 PF replication out of tolerance: {pf:.3f} vs target "
            f"{W1_PF_TARGET} (|delta|={delta_pf:.3f} > {W1_PF_TOLERANCE})"
        )
        assert delta_exp <= W1_EXP_TOLERANCE, (
            f"W1 exp replication out of tolerance: {exp_r:.3f}R vs target "
            f"{W1_EXP_TARGET}R (|delta|={delta_exp:.3f} > {W1_EXP_TOLERANCE})"
        )

    def test_w2_oldest_window_meets_floor(self, wf_report: dict) -> None:
        """W2 — oldest 60-day window (2021-Q1, post-COVID bull). Looser
        than W1 because different regime; PF>=0.70 / exp>=-0.20R."""
        w2 = wf_report["windows"]["W2"]
        pf, exp_r = _agg(w2)
        assert pf >= W2_W3_PF_FLOOR, (
            f"W2 PF {pf:.3f} < floor {W2_W3_PF_FLOOR}. The retune is "
            "time-window-specific."
        )
        assert exp_r >= W2_W3_EXP_FLOOR, (
            f"W2 exp {exp_r:.3f}R < floor {W2_W3_EXP_FLOOR}R."
        )

    def test_w3_alternate_window_meets_floor(self, wf_report: dict) -> None:
        """W3 — mid-history 60-day window (2022 bear)."""
        w3 = wf_report["windows"]["W3"]
        pf, exp_r = _agg(w3)
        assert pf >= W2_W3_PF_FLOOR, (
            f"W3 PF {pf:.3f} < floor {W2_W3_PF_FLOOR}."
        )
        assert exp_r >= W2_W3_EXP_FLOOR, (
            f"W3 exp {exp_r:.3f}R < floor {W2_W3_EXP_FLOOR}R."
        )

    def test_w4_median_across_windows(self, wf_report: dict) -> None:
        """W4 — median across 3 windows must clear an intermediate
        threshold. A single-window result must hold on average."""
        windows = wf_report["windows"]
        pfs = []
        exps = []
        for label in ("W1", "W2", "W3"):
            pf, exp_r = _agg(windows[label])
            pfs.append(pf)
            exps.append(exp_r)
        # Use sorted-middle (3 windows ⇒ index 1 of sorted ascending).
        median_pf = sorted(pfs)[1]
        median_exp = sorted(exps)[1]
        assert median_pf >= W4_MEDIAN_PF_FLOOR, (
            f"W4 median PF {median_pf:.3f} < floor {W4_MEDIAN_PF_FLOOR}.\n"
            f"Per-window PFs: W1={pfs[0]:.3f} W2={pfs[1]:.3f} W3={pfs[2]:.3f}"
        )
        assert median_exp >= W4_MEDIAN_EXP_FLOOR, (
            f"W4 median exp {median_exp:.3f}R < floor "
            f"{W4_MEDIAN_EXP_FLOOR}R."
        )

    def test_w5_released_pkl_sha256_unchanged(self, wf_report: dict) -> None:
        """W5 — ml_model_v2_retune.pkl byte-identical pre/post the
        entire walk-forward run. Per-window bundles are SEPARATE pkls
        (ml_model_v2_walk_window{1,2,3}.pkl); the released one is
        read-only via _load_primary_config."""
        sha = wf_report.get("released_bundle_sha256", {})
        assert sha.get("unchanged") is True, (
            f"ml_model_v2_retune.pkl SHA256 changed during walk-forward "
            f"run: {sha}"
        )

    def test_three_windows_present(self, wf_report: dict) -> None:
        """Structural: walk_forward_validation.json must contain
        per-window blocks for W1/W2/W3."""
        windows = wf_report.get("windows", {})
        assert set(windows.keys()) == {"W1", "W2", "W3"}, (
            f"expected windows {{W1,W2,W3}}, got {set(windows.keys())}"
        )
        for label, w in windows.items():
            assert "aggregate_across_regimes" in w, (
                f"{label} missing aggregate_across_regimes"
            )
            assert "per_regime_metrics" in w, (
                f"{label} missing per_regime_metrics"
            )
            assert "window_end_ts" in w, f"{label} missing window_end_ts"
            assert "window_start_ts" in w, f"{label} missing window_start_ts"
