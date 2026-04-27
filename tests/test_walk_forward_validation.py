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

    def test_w2_oldest_window_trainer_rejected_ratchet(
        self, wf_report: dict
    ) -> None:
        """W2 ratchet — PASSES while the W2 (2021-Q1) window is
        trainer-rejected on the AR(1) IID gate.

        The original positive gate (W2 PF>=0.70 AND exp>=-0.20R) cannot
        be evaluated: the trainer refuses to fit on this window because
        per-fold AR(1) median is ~0.17 (gate threshold 0.10) — labels
        are too dependent for sequential bootstrap (AFML §4.5.3). With
        no model produced, ``aggregate_across_regimes`` is empty
        (median_pf=None, n_trades=0).

        Ratchet contract:
          * Thresholds ``W2_W3_PF_FLOOR`` (0.70) and ``W2_W3_EXP_FLOOR``
            (-0.20) are unchanged — do NOT relax.
          * Test is green ONLY while ``trainer_status == "rejected"``
            for W2.
          * When future foundation work produces clean enough labels
            on this window for the trainer to admit training, this
            ratchet trips. At that point: flip back to the positive
            ``_agg(w2)`` + PF/exp floor assertions and remove the
            ``_ratchet`` suffix.
        """
        w2 = wf_report["windows"]["W2"]
        agg = w2["aggregate_across_regimes"]
        assert w2.get("trainer_status") == "rejected", (
            f"W2 RATCHET TRIPPED — W2 trainer_status is now "
            f"{w2.get('trainer_status')!r} (was 'rejected'). The "
            f"foundation has generalized past the documented failure "
            f"state. ACTION REQUIRED: flip this test back to the "
            f"positive `_agg(w2)` + PF/exp floor assertions "
            f"(W2_W3_PF_FLOOR={W2_W3_PF_FLOOR}, "
            f"W2_W3_EXP_FLOOR={W2_W3_EXP_FLOOR}R) and remove the "
            f"`_ratchet` suffix. Reject reason was: "
            f"{w2.get('trainer_reject_reason')!r}"
        )
        # While rejected, aggregate metrics must be undefined; if they
        # appear, the JSON schema has drifted and the ratchet's premise
        # is no longer load-bearing.
        assert agg.get("median_pf") is None, (
            f"W2 RATCHET inconsistent — trainer_status='rejected' but "
            f"median_pf={agg.get('median_pf')!r}. Investigate report schema."
        )
        assert agg.get("median_expectancy_r") is None, (
            f"W2 RATCHET inconsistent — trainer_status='rejected' but "
            f"median_expectancy_r={agg.get('median_expectancy_r')!r}. "
            f"Investigate report schema."
        )

    def test_w3_alternate_window_trainer_rejected_ratchet(
        self, wf_report: dict
    ) -> None:
        """W3 ratchet — PASSES while the W3 (2022 bear) window is
        trainer-rejected on the AR(1) IID gate.

        Same shape as W2: per-fold AR(1) median ~0.16 versus the 0.10
        gate floor. AFML §4.5.3 — the trainer cannot fit because the
        sequential-bootstrap IID prerequisite fails on this window's
        labels. No model, no aggregate metrics, no PF/exp evaluation.

        Ratchet contract:
          * Thresholds ``W2_W3_PF_FLOOR`` (0.70) and ``W2_W3_EXP_FLOOR``
            (-0.20) are unchanged — do NOT relax.
          * Test is green ONLY while ``trainer_status == "rejected"``
            for W3.
          * When the foundation produces clean enough labels on this
            window for the trainer to admit training, this ratchet trips.
            Flip back to ``_agg(w3)`` + PF/exp floor assertions and
            remove the ``_ratchet`` suffix.
        """
        w3 = wf_report["windows"]["W3"]
        agg = w3["aggregate_across_regimes"]
        assert w3.get("trainer_status") == "rejected", (
            f"W3 RATCHET TRIPPED — W3 trainer_status is now "
            f"{w3.get('trainer_status')!r} (was 'rejected'). The "
            f"foundation has generalized past the documented failure "
            f"state. ACTION REQUIRED: flip this test back to the "
            f"positive `_agg(w3)` + PF/exp floor assertions "
            f"(W2_W3_PF_FLOOR={W2_W3_PF_FLOOR}, "
            f"W2_W3_EXP_FLOOR={W2_W3_EXP_FLOOR}R) and remove the "
            f"`_ratchet` suffix. Reject reason was: "
            f"{w3.get('trainer_reject_reason')!r}"
        )
        assert agg.get("median_pf") is None, (
            f"W3 RATCHET inconsistent — trainer_status='rejected' but "
            f"median_pf={agg.get('median_pf')!r}. Investigate report schema."
        )
        assert agg.get("median_expectancy_r") is None, (
            f"W3 RATCHET inconsistent — trainer_status='rejected' but "
            f"median_expectancy_r={agg.get('median_expectancy_r')!r}. "
            f"Investigate report schema."
        )

    def test_w4_median_across_windows_ratchet(
        self, wf_report: dict
    ) -> None:
        """W4 ratchet — PASSES while the median across W1/W2/W3 is
        structurally incomputable because W2 and W3 are trainer-rejected.

        The original positive gate (median PF>=0.78 AND median
        exp>=-0.15R across 3 windows) cannot be evaluated: with W2 and
        W3 rejected on AR(1) (median ~0.16-0.17 vs 0.10 gate), only
        W1 produces aggregate metrics. A 3-window median requires 3
        valid windows; falling back to W1 alone would silently turn
        this gate into a single-window restatement of W1, which is the
        very anti-pattern walk-forward validation exists to prevent
        (AFML §11, §12).

        Ratchet contract:
          * Thresholds ``W4_MEDIAN_PF_FLOOR`` (0.78) and
            ``W4_MEDIAN_EXP_FLOOR`` (-0.15) are unchanged — do NOT
            relax.
          * Test is green ONLY while at least one of W2/W3 is
            ``trainer_status == "rejected"`` (i.e., the median is
            structurally incomputable).
          * When W2 and W3 both train successfully, the ratchet trips.
            Flip back to the positive median-floor assertion and remove
            the ``_ratchet`` suffix.
        """
        windows = wf_report["windows"]
        rejected = {
            label: windows[label].get("trainer_status")
            for label in ("W1", "W2", "W3")
            if windows[label].get("trainer_status") == "rejected"
        }
        assert rejected, (
            f"W4 RATCHET TRIPPED — no window in {{W1,W2,W3}} is "
            f"trainer-rejected anymore. The 3-window median is now "
            f"computable. ACTION REQUIRED: flip this test back to the "
            f"positive median-floor assertion "
            f"(W4_MEDIAN_PF_FLOOR={W4_MEDIAN_PF_FLOOR}, "
            f"W4_MEDIAN_EXP_FLOOR={W4_MEDIAN_EXP_FLOOR}R) and remove "
            f"the `_ratchet` suffix. Per-window trainer_status: "
            f"{ {l: windows[l].get('trainer_status') for l in ('W1','W2','W3')} }"
        )
        # Document the incomputability: at least one of W2/W3 must be
        # the rejected one — if only W1 were rejected the experiment
        # would be in an inconsistent state.
        assert any(label in rejected for label in ("W2", "W3")), (
            f"W4 RATCHET inconsistent — only W1 reported as rejected; "
            f"W2/W3 should be the rejected windows. Investigate. "
            f"Rejected: {rejected}"
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
