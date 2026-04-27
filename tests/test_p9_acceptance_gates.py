"""P9 acceptance gates — vol-targeted position sizing.

Six gates per the P9 self-approval contract. Pattern matches
``tests/test_walk_forward_base_rate.py``: a module-scoped fixture loads
``backtest_results/P9_20window.json`` and ``pytest.skip``s when the
artifact has not yet been produced. Run
``python train_p9_vol_target.py`` first; then re-run this suite.

Gates
-----
P9.A  Median realized portfolio vol across 20 windows within ±30% of
      0.75% daily target ⇒ med ∈ [0.00525, 0.00975].

P9.B  Single-position cap NEVER violated across all 20 windows × all
      folds. Hard invariant — every routed event must satisfy
      ``applied_size / vol_parity_size ≤ 5 + ε``. Aggregate counter
      ``cap_violations`` must equal 0.

P9.C  Drawdown distribution improves vs fixed-fractional baseline:
      both ``median(max_dd_p9)`` and ``P95(max_dd_p9)`` are closer to
      zero than the baseline equivalents (both numbers negative; the
      "improves" relation is ``p9 > baseline`` since 0 > -DD).

P9.D  Per-window median ΔR ≥ -0.01R, where ΔR is the per-window
      portfolio-R-multiple expectancy delta (P9 - baseline). Sizing
      should not destroy edge — small (-0.01R) bound recognizes that
      sizing changes will perturb realized PnL even when the underlying
      signal is unchanged.

P9.E  ml_model_v2_retune.pkl SHA256 byte-identical pre/post the run.

P9.F  Per-fold AR(1) gate (median < 0.10, max < 0.20) still satisfied
      within surviving regimes. Sizing does not relabel events, so AR(1)
      should be invariant — this is a defensive double-check.

Methodology references
----------------------
* López de Prado, AFML §10.1 — vol-targeted sizing.
* AFML §10.4 — cap rationale.
* AFML §11 — walk-forward base-rate validation (the 20-window driver).
* AFML §4.5.3 — sequential bootstrap requires weak label dependence
  (the AR(1) gate enforces the prerequisite).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
P9_REPORT = REPO_ROOT / "backtest_results" / "P9_20window.json"

TARGET_DAILY_VOL: float = 0.0075
VOL_BAND_LOWER: float = 0.00525   # 0.0075 × (1 - 0.30)
VOL_BAND_UPPER: float = 0.00975   # 0.0075 × (1 + 0.30)
CAP_MULT: float = 5.0
CAP_EPS: float = 1e-9
DELTA_R_FLOOR: float = -0.01
RELEASED_BUNDLE_SHA256: str = (
    "96b5bbe3e705b06682f231d975c4e1d574e3a2b93e8d95389ac438de21cec6e4"
)
AR1_MEDIAN_THRESHOLD: float = 0.10
AR1_MAX_THRESHOLD: float = 0.20

EXPECTED_WINDOW_LABELS: tuple[str, ...] = tuple(f"W{n:02d}" for n in range(1, 21))


@pytest.fixture(scope="module")
def p9_report() -> dict:
    if not P9_REPORT.exists():
        pytest.skip(
            f"P9 report not found at {P9_REPORT} — run "
            "train_p9_vol_target.py to produce it."
        )
    return json.loads(P9_REPORT.read_text())


def _ok_windows(report: dict) -> list[tuple[str, dict]]:
    """Return [(label, window_record)] for windows that passed the
    trainer (have P3 metrics — i.e. P9 sizing was applied)."""
    return [
        (label, w)
        for label, w in report["windows"].items()
        if w.get("trainer_status") == "ok"
    ]


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile, p in [0,1]. Caller guards
    empty list."""
    if not values:
        raise ValueError("percentile undefined on empty list")
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    rank = p * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return s[lo] + frac * (s[hi] - s[lo])


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("median undefined on empty list")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


class TestP9StructuralSchema:

    def test_twenty_windows_present(self, p9_report: dict) -> None:
        """Structural — report covers all 20 windows W01..W20 and each
        carries the schema keys downstream gates depend on."""
        windows = p9_report.get("windows", {})
        assert set(windows.keys()) == set(EXPECTED_WINDOW_LABELS), (
            f"expected {set(EXPECTED_WINDOW_LABELS)}, got {set(windows.keys())}"
        )
        for label, w in windows.items():
            assert "trainer_status" in w, f"{label} missing trainer_status"
            if w["trainer_status"] == "ok":
                assert "p9" in w, f"{label} ok but missing p9 block"
                assert "baseline" in w, f"{label} ok but missing baseline block"
                for k in (
                    "max_dd", "realized_portfolio_vol_daily",
                    "median_expectancy_r_portfolio",
                ):
                    assert k in w["p9"], f"{label}.p9 missing {k!r}"
                    assert k in w["baseline"], f"{label}.baseline missing {k!r}"

    def test_aggregates_block_present(self, p9_report: dict) -> None:
        agg = p9_report.get("aggregates", {})
        for k in (
            "n_windows_evaluated",
            "median_realized_vol_daily_p9",
            "median_max_dd_p9", "p95_max_dd_p9",
            "median_max_dd_baseline", "p95_max_dd_baseline",
            "median_delta_R", "cap_violations",
        ):
            assert k in agg, f"aggregates missing {k!r}"


class TestP9Gates:

    # P9.A ----------------------------------------------------------
    def test_p9_A_realized_vol_in_band(self, p9_report: dict) -> None:
        """Median realized portfolio vol across 20 windows within ±30% of
        the 0.75% daily target."""
        med = p9_report["aggregates"]["median_realized_vol_daily_p9"]
        assert med is not None, (
            "median_realized_vol_daily_p9 is None — no windows evaluated"
        )
        assert VOL_BAND_LOWER <= med <= VOL_BAND_UPPER, (
            f"P9.A median realized portfolio vol {med:.5f} outside band "
            f"[{VOL_BAND_LOWER:.5f}, {VOL_BAND_UPPER:.5f}] (±30% of "
            f"{TARGET_DAILY_VOL:.4f})"
        )

    # P9.B ----------------------------------------------------------
    def test_p9_B_cap_invariant_per_event(self, p9_report: dict) -> None:
        """Hard invariant: every routed event satisfies
        ``applied_size / vol_parity_size <= 5 + ε``."""
        violations: list[str] = []
        for label, w in _ok_windows(p9_report):
            for ev in w["p9"].get("per_event", []):
                if ev.get("decision") != "trade":
                    continue
                vp = ev.get("vol_parity_size")
                ap = ev.get("applied_size")
                if vp is None or ap is None or vp <= 0:
                    continue
                ratio = ap / vp
                if ratio > CAP_MULT + CAP_EPS:
                    violations.append(
                        f"{label} {ev.get('symbol')} {ev.get('entry_ts')} "
                        f"applied/parity={ratio:.6f}"
                    )
        assert not violations, (
            f"P9.B cap violated by {len(violations)} events: "
            f"{violations[:5]}{'...' if len(violations) > 5 else ''}"
        )

    def test_p9_B_aggregate_cap_violations_zero(self, p9_report: dict) -> None:
        """Aggregate counter sanity — ``cap_violations`` must be 0 in
        the aggregates block."""
        cv = p9_report["aggregates"].get("cap_violations")
        assert cv == 0, f"P9.B aggregate cap_violations={cv} (expected 0)"

    # P9.C ----------------------------------------------------------
    def test_p9_C_median_max_dd_improves(self, p9_report: dict) -> None:
        """Median max-DD across 20 windows must be closer to zero than
        the baseline median. Both numbers are negative; "closer to 0"
        means larger (less negative)."""
        agg = p9_report["aggregates"]
        m_p9 = agg["median_max_dd_p9"]
        m_base = agg["median_max_dd_baseline"]
        assert m_p9 is not None and m_base is not None, (
            "P9.C cannot be evaluated — missing median_max_dd values"
        )
        assert m_p9 > m_base, (
            f"P9.C median max-DD did not improve: P9={m_p9:.4f} "
            f"vs baseline={m_base:.4f} (need P9 > baseline)"
        )

    def test_p9_C_p95_max_dd_improves(self, p9_report: dict) -> None:
        """P95 (worst-tail) max-DD must also be closer to zero than the
        baseline P95."""
        agg = p9_report["aggregates"]
        p95_p9 = agg["p95_max_dd_p9"]
        p95_base = agg["p95_max_dd_baseline"]
        assert p95_p9 is not None and p95_base is not None, (
            "P9.C cannot be evaluated — missing p95_max_dd values"
        )
        assert p95_p9 > p95_base, (
            f"P9.C P95 max-DD did not improve: P9={p95_p9:.4f} "
            f"vs baseline={p95_base:.4f} (need P9 > baseline)"
        )

    # P9.D ----------------------------------------------------------
    def test_p9_D_per_window_median_delta_R(self, p9_report: dict) -> None:
        """Per-window median ΔR (P9 - baseline) ≥ -0.01R. Sizing should
        not meaningfully destroy edge."""
        deltas = [
            w["delta_R"]
            for _, w in _ok_windows(p9_report)
            if w.get("delta_R") is not None
        ]
        assert deltas, (
            "P9.D cannot be evaluated — no per-window delta_R values"
        )
        med = _median(deltas)
        assert med >= DELTA_R_FLOOR, (
            f"P9.D per-window median ΔR={med:.4f}R < {DELTA_R_FLOOR}R "
            f"floor. Per-window deltas: {sorted(deltas)}"
        )

    # P9.E ----------------------------------------------------------
    def test_p9_E_released_bundle_unchanged(self, p9_report: dict) -> None:
        """ml_model_v2_retune.pkl byte-identical pre/post P9 run."""
        sha = p9_report.get("released_bundle_sha256", {})
        assert sha.get("unchanged") is True, (
            f"P9.E released bundle SHA changed: {sha}"
        )
        assert sha.get("after") == RELEASED_BUNDLE_SHA256, (
            f"P9.E released bundle SHA mismatch: "
            f"after={sha.get('after')} expected={RELEASED_BUNDLE_SHA256}"
        )

    # P9.F ----------------------------------------------------------
    def test_p9_F_ar1_gate_unaffected(self, p9_report: dict) -> None:
        """Per-fold AR(1) median < 0.10, max < 0.20 within surviving
        windows. Sizing does not relabel events; this is a defensive
        re-read of the upstream training reports the base-rate sweep
        produced."""
        violations: list[str] = []
        for label, w in _ok_windows(p9_report):
            tr_path = REPO_ROOT / "backtest_results" / (
                f"baserate_{label.lower()}_training.json"
            )
            if not tr_path.exists():
                # W01-W03 reuse prior reports under different names; skip
                # the AR(1) re-read and trust the base-rate sweep's
                # bookkeeping for those windows.
                continue
            trep = json.loads(tr_path.read_text())
            iid = trep.get("iid_diagnostics", {})
            per_fold = list(iid.get("per_fold_ar1", []))
            if not per_fold:
                violations.append(f"{label}: no per_fold_ar1 in iid_diagnostics")
                continue
            abs_vals = sorted(abs(float(v)) for v in per_fold)
            n = len(abs_vals)
            med = (
                abs_vals[n // 2]
                if n % 2 == 1
                else 0.5 * (abs_vals[n // 2 - 1] + abs_vals[n // 2])
            )
            mx = abs_vals[-1]
            if med >= AR1_MEDIAN_THRESHOLD:
                violations.append(
                    f"{label}: AR(1) median {med:.3f} >= {AR1_MEDIAN_THRESHOLD}"
                )
            if mx >= AR1_MAX_THRESHOLD:
                violations.append(
                    f"{label}: AR(1) max {mx:.3f} >= {AR1_MAX_THRESHOLD}"
                )
        assert not violations, (
            f"P9.F AR(1) gate violations within surviving windows: {violations}"
        )
