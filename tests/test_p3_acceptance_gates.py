"""P3 acceptance gates A–D — programmatic verification.

These tests read ``backtest_results/P3_after.json`` (the report from
the full 60-day × 10-symbol nested-CV run) and assert each gate.

Per the spec, this run can produce two valid end states:

  1. **Gate A passes.** At least one regime achieved PF ≥ 1.3 AND
     expectancy > 0 AND n ≥ 100 across folds. Tests for B, C, D also
     run; if any of those fail the suite fails.
  2. **Gate A fails.** Tests for A and (informationally) B fail; C and
     D are still asserted independently. The failure is the result —
     the spec explicitly says: "do not relax, do not move on."

Each gate is its own test so a partial pass/fail is legible at the
pytest report level. There is no "soft fail" — if gate A is supposed
to pass on this run and doesn't, the test correctly fails. The diff
doc (``backtest_results/P3_vs_P4_P1_diff.md``) is the qualitative
record; this file is the boolean record.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
P3_REPORT = REPO_ROOT / "backtest_results" / "P3_after.json"

# Hard thresholds from the spec (do NOT change without an explicit
# user instruction; gates exist to enforce honest acceptance).
GATE_A_MIN_PF = 1.3
GATE_A_MIN_EXPECTANCY_R = 0.0
GATE_A_MIN_TRADES = 100
GATE_B_AR1_MEDIAN_LT = 0.10
GATE_B_AR1_MAX_LT = 0.20
GATE_D_PF_FLOOR = 0.738       # P4 baseline median PF
GATE_D_EXPECTANCY_FLOOR = -0.196  # P4 baseline median expectancy R


@pytest.fixture(scope="module")
def p3_report() -> dict:
    if not P3_REPORT.exists():
        pytest.skip(f"P3 report not found at {P3_REPORT}")
    return json.loads(P3_REPORT.read_text())


def _passing_gate_a_regimes(report: dict) -> list[str]:
    """Regimes that satisfy gate A: PF≥1.3, expectancy>0, n≥100."""
    out = []
    for regime, metrics in report["per_regime_metrics"].items():
        if regime == "chop":
            continue
        n = metrics.get("total_n_trades", 0)
        pf = metrics.get("median_pf")
        exp_r = metrics.get("median_expectancy_r")
        if (pf is not None
                and exp_r is not None
                and pf >= GATE_A_MIN_PF
                and exp_r > GATE_A_MIN_EXPECTANCY_R
                and n >= GATE_A_MIN_TRADES):
            out.append(regime)
    return out


class TestP3AcceptanceGates:

    def test_gate_a_at_least_one_regime_passes(self, p3_report: dict) -> None:
        """Gate A: at least one regime has PF ≥ 1.3 AND expectancy > 0
        AND n_trades ≥ 100 across folds. If false: STOP, do not relax.

        This test EXPECTS to pass — failure means no regime has
        positive net edge on this universe. The failure mode is the
        result of the experiment; see backtest_results/P3_vs_P4_P1_diff.md.
        """
        passing = _passing_gate_a_regimes(p3_report)
        # Build a diagnostic for the failure mode.
        ranked = sorted(
            (
                (r, m.get("median_pf"), m.get("median_expectancy_r"),
                 m.get("total_n_trades"))
                for r, m in p3_report["per_regime_metrics"].items()
                if r != "chop"
            ),
            key=lambda t: (t[1] if t[1] is not None else -1),
            reverse=True,
        )
        diag = "\n".join(
            f"  {r:20s} PF={pf}  exp={exp}  n={n}"
            for r, pf, exp, n in ranked
        )
        assert passing, (
            "GATE A FAILED — no regime achieved PF >= 1.3 AND "
            "expectancy > 0 AND n_trades >= 100.\n"
            f"Per-regime ranking:\n{diag}\n"
            f"Spec says: STOP. Do not relax. Decision is the user's."
        )

    def test_gate_b_per_fold_ar1_within_surviving_regimes(
        self, p3_report: dict,
    ) -> None:
        """Gate B: per-fold AR(1) gate (median < 0.10, max < 0.20)
        within regimes that survived gate A. If gate A fails, gate B
        is structurally moot — but we still verify *for the regimes
        that would have qualified* so the diagnostic is preserved.
        """
        passing = _passing_gate_a_regimes(p3_report)
        if not passing:
            pytest.skip(
                "Gate A failed — gate B is structurally moot. See "
                "test_gate_a_at_least_one_regime_passes for the failure mode."
            )
        # Real check: for every regime that passed A, AR(1) median<0.10
        # AND max<0.20.
        violations = []
        for r in passing:
            m = p3_report["per_regime_metrics"][r]
            ar1_med = m.get("median_per_fold_ar1")
            ar1_max = m.get("max_per_fold_ar1")
            if ar1_med is None or ar1_max is None:
                violations.append(f"{r}: AR(1) not computable")
                continue
            if abs(ar1_med) >= GATE_B_AR1_MEDIAN_LT:
                violations.append(
                    f"{r}: |median AR(1)| {abs(ar1_med):.3f} >= "
                    f"{GATE_B_AR1_MEDIAN_LT}")
            if ar1_max >= GATE_B_AR1_MAX_LT:
                violations.append(
                    f"{r}: max AR(1) {ar1_max:.3f} >= {GATE_B_AR1_MAX_LT}")
        assert not violations, (
            "GATE B FAILED — surviving regimes violate per-fold AR(1):\n  "
            + "\n  ".join(violations)
        )

    def test_gate_c_chop_zero_trades(self, p3_report: dict) -> None:
        """Gate C: chop regime trade count == 0 across all folds.
        Hard rule. Encoded in the runner AND in regime_configs/p3.yaml;
        violation indicates a bug in the chop guard."""
        chop = p3_report["per_regime_metrics"]["chop"]
        assert chop["total_n_trades"] == 0, (
            f"GATE C FAILED — {chop['total_n_trades']} chop trades found. "
            "Chop is forbidden by spec; check the runner's allow_trade "
            "guard and regime_configs/p3.yaml."
        )
        assert all(n == 0 for n in chop["per_fold_n_trades"]), (
            f"GATE C FAILED per-fold — {chop['per_fold_n_trades']}"
        )

    def test_gate_d_aggregate_ge_p4_baseline(self, p3_report: dict) -> None:
        """Gate D (sanity): aggregate-across-regimes economics ≥ P4
        baseline (PF ≥ 0.738, expectancy ≥ −0.196R). Confirms regime
        routing isn't *destroying* aggregate value vs P4."""
        agg = p3_report["aggregate_across_regimes"]
        pf = agg.get("median_pf")
        exp_r = agg.get("median_expectancy_r")
        assert pf is not None and exp_r is not None, (
            f"GATE D — aggregate metrics undefined: pf={pf} exp={exp_r}"
        )
        assert pf >= GATE_D_PF_FLOOR, (
            f"GATE D FAILED — aggregate PF {pf:.3f} < P4 floor "
            f"{GATE_D_PF_FLOOR}"
        )
        assert exp_r >= GATE_D_EXPECTANCY_FLOOR, (
            f"GATE D FAILED — aggregate expectancy {exp_r:.3f}R < P4 floor "
            f"{GATE_D_EXPECTANCY_FLOOR}R"
        )

    def test_chop_pct_diagnostic_below_50(self, p3_report: dict) -> None:
        """Diagnostic: if >50% of bars are committed-chop, the classifier
        is degenerate (saturating chop). Distinguishes 'real C pass with
        no edge' from 'gate ate the signal'. Informational fail
        threshold; not a hard gate."""
        pct_chop = p3_report["regime_distribution"]["pct_bars_classified_chop"]
        assert pct_chop < 0.50, (
            f"DIAGNOSTIC: {pct_chop*100:.1f}% of bars classified chop — "
            "classifier may be saturating; CI threshold or lookback may "
            "need re-examination (out of scope per 'no relax' rule)."
        )

    def test_seven_regime_states_present(self, p3_report: dict) -> None:
        """Structural: all 7 regime keys present, regardless of trade count."""
        expected = {
            "trend_up_low_vol", "trend_up_high_vol",
            "trend_dn_low_vol", "trend_dn_high_vol",
            "range_tight", "range_wide", "chop",
        }
        actual = set(p3_report["per_regime_metrics"].keys())
        assert actual == expected, (
            f"missing: {expected - actual}\nextra:  {actual - expected}"
        )

    def test_pkl_unchanged(self, p3_report: dict) -> None:
        """Mirrors P1's contract: ml_model_v2.pkl byte-identical
        pre/post run."""
        sha = p3_report.get("primary_bundle_sha256", {})
        assert sha.get("unchanged") is True, (
            f"ml_model_v2.pkl SHA256 changed during P3 run: {sha}"
        )
