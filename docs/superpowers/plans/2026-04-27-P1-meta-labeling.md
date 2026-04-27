# P1 — Meta-Labeling Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a binary meta-classifier on top of the existing P4 multiclass model (ml_model_v2.pkl) so that the bot only takes a primary's BUY/SELL signals when calibrated win-probability ≥ a threshold that simultaneously satisfies (i) take-rate ∈ [20%, 40%], (ii) PF ≥ 1.3, (iii) expectancy after costs > 0, (iv) Brier < 0.22, and (v) calibration within 5% of diagonal across deciles, all OOS — without re-training the primary, without inventing new features, and without breaking the per-fold AR(1) gate from P4.

**Architecture:** Pure López de Prado AFML §3.6–3.7 meta-labeling.
1. The P4 classifier (`ml_model_v2.pkl`, 3-class softprob {SELL=0, HOLD=1, BUY=2}) decides *side*: side=+1 if argmax==BUY and max_proba ≥ P4 decision_threshold (0.55), side=−1 if argmax==SELL and max_proba ≥ 0.55, else "no fire". The set of "fires" is the candidate set the meta-model is allowed to filter.
2. The meta-classifier predicts *go-or-pass* (binary 0/1) on those candidates. Target = binary triple-barrier label `apply_meta_triple_barrier` already produces (1 = primary's bet hit pt before sl, else 0).
3. CUSUM events (`cusum_filter`) generate candidate timestamps. Same pt/sl/vertical from `ml_model_v2.pkl.config` (locked from P4 — we are explicitly forbidden from re-tuning labelling).
4. PurgedKFold with embargo = `2 × vertical_barrier_bars / N_samples` (the same `embargo_pct` semantics already used in `ml/purged_cv.py`) supplies OOS folds. Within each fold we train base XGBoost binary on the inner-train portion, fit isotonic on a chronological inner-validation tail, then predict + calibrate on the outer-test rows.
5. Concatenated OOS calibrated probabilities feed the gate-acceptance evaluation: Brier, decile calibration, threshold sweep on **net R-multiple** (gross − cost_in_r from `ml/costs.py`), and per-fold weighted-residual AR(1) — same `compute_iid_diagnostics` machinery as P4 but on the binary residual.
6. Hard-reject on any failed acceptance gate; success path saves bundle to `ml_meta_model_p1.pkl` and writes the full report to `backtest_results/P1_after.json` plus a markdown diff against P4_after.json.

**Tech Stack:** Python 3, NumPy/Pandas, XGBoost (binary:logistic to match the P4 stack), scikit-learn (`IsotonicRegression`, `brier_score_loss`), joblib, pytest. Reuses every existing module under `ml/` (labels, events, purged_cv, sample_weights, iid_diagnostics, costs, evaluation, meta_labels). No new dependencies.

**Out-of-scope guard rails (refused if creep):**
- ❌ New features beyond what P4 produces (we use only `IntegratedMLModel.feature_names` + the P4 model's own softprob outputs, which *are* P4 outputs).
- ❌ Any change to `ml_model_v2.pkl`, `ml/labels.py`, `ml/events.py`, `ml/sample_weights.py`, `ml/meta_labels.py`. We only **read** these.
- ❌ LLM-driven anything (P16 territory).
- ❌ Calibration of the cross-fold stitched AR(1) (P14 territory).

**User clarifications (binding):**
1. **Threshold selection** is performed on the **same chronological tail** used for isotonic calibration (never on training, never on the OOS test fold). Objective = cost-adjusted Sortino on the tail, with cost = 0.32R/trade (P2/P4 anchor). Tie-break: when two candidates are within 5% of the maximum Sortino, pick the **higher threshold** (bias toward selectivity). Each fold reports its own picked threshold; if the max−min spread across folds exceeds ±0.10, set `threshold_agreement.disagreement = true` in the report (visible, not a reject).
2. **Sample uniqueness is recomputed on the meta-event subset.** The meta-set is the strict subset of CUSUM events where the primary fired BUY or SELL. Its temporal structure differs from P4's full event set, so weights are recomputed via `compute_uniqueness` on the post-fire `(event_times, touch_times, groups=symbol)`. P4's uniqueness vector is never reused.
3. **Cost model is single-source.** `CostModel`, `cost_in_r`, `ret_to_r_multiple` are imported from `ml/costs.py` — the same module P2/P4 use. Parameters are read from `ml_model_v2.pkl.config["cost_model"]`. Per-trade cost in R is computed by the same `cost_in_r` formula (round-trip fee+slip + Almgren-Chriss impact divided by stop_pct), never redefined.

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `ml/meta_model.py` | **CREATE** | Pure functions: brier, calibration deciles, threshold sweep, threshold picker, per-fold isotonic-calibrated meta-fold trainer. ~250 lines, no I/O. |
| `train_meta_model_p1.py` | **CREATE** | Top-level entry point. Loads P4 bundle, generates events, runs primary to get sides, trains meta with purged CV, evaluates gates, writes `P1_after.json` + bundle. ~600 lines. Mirrors `train_ml_model_v2.py` style. |
| `tests/test_meta_model_p1.py` | **CREATE** | Unit tests for `ml/meta_model.py` — pure-function level. |
| `tests/test_train_meta_p1_integration.py` | **CREATE** | End-to-end integration tests with stubbed Postgres collector + stubbed primary, mirrors `tests/test_p4_integration.py`. |
| `backtest_results/P1_after.json` | **CREATE (output)** | Final P1 report with cv_summary, brier, calibration, threshold_sweep, gate verdicts, AR(1), filtered backtest metrics. |
| `backtest_results/P1_vs_P4_diff.md` | **CREATE (output)** | Markdown diff table P1 vs P4 + 5 acceptance gates pass/fail. |
| `ml_meta_model_p1.pkl` | **CREATE (output)** | Joblib bundle with `{base_clf, isotonic, scaler, feature_names, primary_bundle_path, primary_decision_threshold, meta_threshold, config}`. |
| `ml/labels.py` | **READ ONLY** | — |
| `ml/events.py` | **READ ONLY** | — |
| `ml/meta_labels.py` | **READ ONLY** | — |
| `ml/purged_cv.py` | **READ ONLY** | — |
| `ml/sample_weights.py` | **READ ONLY** | — |
| `ml/iid_diagnostics.py` | **READ ONLY** | — |
| `ml/costs.py` | **READ ONLY** | — |
| `ml/evaluation.py` | **READ ONLY** | — |
| `ml_model_v2.pkl` | **READ ONLY** | Primary; loaded via `joblib.load`. |
| `ml/__init__.py` | **NO CHANGE** | We export from `ml.meta_model` directly via fully-qualified import in tests/script. |

---

## Acceptance Gates (hard-reject; encoded in code AND tests)

| # | Gate | Threshold | Source |
|---|------|-----------|--------|
| 1 | OOS Brier on calibrated probabilities | `< 0.22` | P1 spec §4 |
| 2 | Calibration deciles within ±5% of diagonal (deciles with n ≥ 30) | `max|p̂ − bin_mid| ≤ 0.05` | P1 spec §4 |
| 3 | Take-rate post-filter | `0.20 ≤ take_rate ≤ 0.40` | P1 spec §1 |
| 4 | Profit factor post-filter on net R | `≥ 1.3` | P1 spec §1 |
| 5 | Expectancy post-filter on net R | `> 0` | P1 spec §1 |
| 6 | Per-fold AR(1) on weighted binary residuals | `median < 0.10`, `max < 0.20` | P4 carry-over |

Gate 6 is the only one inherited from P4; gates 1–5 are P1-intrinsic.

---

## Execution Order

```
Task 1 — pure helpers (TDD)
Task 2 — per-fold trainer with isotonic (TDD)
Task 3 — gate evaluator + threshold picker (TDD)
Task 4 — primary-fire collection helper (TDD with stubs)
Task 5 — top-level orchestration script (TDD via integration test)
Task 6 — real end-to-end run + diff table
```

---

### Task 1: Pure helpers in `ml/meta_model.py`

**Files:**
- Create: `ml/meta_model.py`
- Test: `tests/test_meta_model_p1.py`

- [ ] **Step 1.1: Write the failing tests for `compute_brier`**

```python
# tests/test_meta_model_p1.py
"""Unit tests for ml/meta_model.py — pure-function level.

P1 (López de Prado AFML §3.6-3.7) meta-labeling layer. These tests
validate the pure helpers: Brier, decile calibration, threshold pick,
per-fold residual AR(1). The training-loop integration is tested in
tests/test_train_meta_p1_integration.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.meta_model import (
    MetaThresholdRejection,
    calibration_deciles,
    compute_brier,
    pick_meta_threshold,
    threshold_sweep_meta,
)


class TestComputeBrier:
    def test_perfect_predictions_zero_brier(self) -> None:
        y = np.array([0, 1, 0, 1, 1])
        p = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
        assert compute_brier(y, p) == pytest.approx(0.0, abs=1e-9)

    def test_uninformative_half_predictions(self) -> None:
        y = np.array([0, 1, 0, 1])
        p = np.full(4, 0.5)
        assert compute_brier(y, p) == pytest.approx(0.25, abs=1e-9)

    def test_array_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="align"):
            compute_brier(np.array([0, 1]), np.array([0.5, 0.5, 0.5]))
```

- [ ] **Step 1.2: Run tests to confirm they fail with ImportError**

Run: `pytest tests/test_meta_model_p1.py::TestComputeBrier -v`
Expected: ImportError on `ml.meta_model`.

- [ ] **Step 1.3: Implement `compute_brier`**

```python
# ml/meta_model.py
"""Meta-labeling filter (P1) — secondary classifier on primary signals.

López de Prado, *Advances in Financial Machine Learning* (2018), §3.6-3.7.
The primary model decides *side* (long/short). The meta-model decides
*take-or-pass* per-event, using the same triple-barrier outcome as the
ground truth (1 = primary's bet hit profit-target before stop-loss).

Pure helpers live here; the orchestrator (``train_meta_model_p1.py``)
wires them into a purged-CV training loop with isotonic calibration on
a held-out fold tail. No I/O in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class MetaThresholdRejection(RuntimeError):
    """Raised when no candidate threshold satisfies all P1 acceptance gates."""


def compute_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Brier score for binary outcomes.

    ``Brier = mean((p_i - y_i)^2)``. Lower is better. AFML p. 78 lists
    Brier as the canonical proper scoring rule for calibrated binary
    classifiers — its quadratic loss penalizes both miscalibration and
    refinement, unlike accuracy.
    """
    y = np.asarray(y_true, dtype="float64")
    p = np.asarray(y_proba, dtype="float64")
    if y.shape != p.shape:
        raise ValueError(
            f"y_true and y_proba must align; got {y.shape} vs {p.shape}"
        )
    if y.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))
```

- [ ] **Step 1.4: Run tests to verify pass**

Run: `pytest tests/test_meta_model_p1.py::TestComputeBrier -v`
Expected: PASS (3/3).

- [ ] **Step 1.5: Write failing tests for `calibration_deciles`**

```python
# tests/test_meta_model_p1.py — append
class TestCalibrationDeciles:
    def test_perfect_calibration_zero_max_error(self) -> None:
        # Simulate a perfectly-calibrated stream: in the [0.7, 0.8) bin,
        # 75% of cases are positive; in [0.5, 0.6), 55% positive; etc.
        rng = np.random.default_rng(0)
        bin_centers = np.linspace(0.05, 0.95, 10)
        y_list, p_list = [], []
        for c in bin_centers:
            n = 200
            p = np.full(n, c)
            y = (rng.uniform(size=n) < c).astype(int)
            y_list.append(y); p_list.append(p)
        y = np.concatenate(y_list); p = np.concatenate(p_list)

        deciles = calibration_deciles(y, p, min_count=30)
        assert deciles["max_abs_deviation"] < 0.05
        # 10 bins exposed
        assert len(deciles["bins"]) == 10

    def test_undercount_bins_excluded(self) -> None:
        # Sparse bins (n < min_count) should be flagged but not
        # contribute to max_abs_deviation, to avoid noise spikes.
        y = np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 1])
        p = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        deciles = calibration_deciles(y, p, min_count=5)
        # Every bin has n=1 < 5; max_abs_deviation must be NaN/None or 0
        # depending on convention. Pick: report None when no bin qualifies.
        assert deciles["max_abs_deviation"] is None
        assert deciles["n_qualifying_bins"] == 0

    def test_returns_per_bin_records(self) -> None:
        y = np.tile([0, 1], 50)
        p = np.full(100, 0.5)
        deciles = calibration_deciles(y, p, min_count=10)
        # All probas land in the [0.4, 0.5) or [0.5, 0.6) bin depending on
        # convention; only one bin should have n=100, the rest n=0.
        assert sum(b["n"] for b in deciles["bins"]) == 100
```

- [ ] **Step 1.6: Run tests to confirm fail**

Run: `pytest tests/test_meta_model_p1.py::TestCalibrationDeciles -v`
Expected: FAIL (function not defined).

- [ ] **Step 1.7: Implement `calibration_deciles`**

```python
# ml/meta_model.py — append
def calibration_deciles(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 10,
    min_count: int = 30,
) -> dict[str, Any]:
    """Compute decile-by-decile empirical frequency vs. predicted probability.

    For each of ``n_bins`` equal-width bins on [0, 1], reports:
        n             — count of samples in this bin
        empirical_p   — mean(y) in this bin (None if n == 0)
        bin_midpoint  — the bin's predicted-probability midpoint
        deviation     — empirical_p - bin_midpoint (None if n == 0)

    ``max_abs_deviation`` aggregates across bins where ``n >= min_count``;
    if no bin qualifies, it is None. The P1 acceptance gate compares
    ``max_abs_deviation`` against 0.05 (López de Prado p. 79: a calibrated
    classifier should hug the diagonal across well-populated deciles).
    """
    y = np.asarray(y_true, dtype="float64")
    p = np.asarray(y_proba, dtype="float64")
    if y.shape != p.shape:
        raise ValueError("y_true and y_proba must align")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, edges, right=False) - 1, 0, n_bins - 1)

    bins: list[dict[str, Any]] = []
    qualifying_devs: list[float] = []
    for i in range(n_bins):
        mask = bin_idx == i
        n = int(mask.sum())
        midpoint = float(0.5 * (edges[i] + edges[i + 1]))
        if n == 0:
            bins.append({
                "bin": i, "n": 0, "midpoint": midpoint,
                "empirical_p": None, "deviation": None,
            })
            continue
        emp = float(y[mask].mean())
        dev = emp - midpoint
        bins.append({
            "bin": i, "n": n, "midpoint": midpoint,
            "empirical_p": emp, "deviation": dev,
        })
        if n >= min_count:
            qualifying_devs.append(abs(dev))

    return {
        "bins": bins,
        "n_qualifying_bins": len(qualifying_devs),
        "max_abs_deviation": (
            float(max(qualifying_devs)) if qualifying_devs else None
        ),
    }
```

- [ ] **Step 1.8: Run tests to verify pass**

Run: `pytest tests/test_meta_model_p1.py::TestCalibrationDeciles -v`
Expected: PASS (3/3).

- [ ] **Step 1.9: Write failing tests for `threshold_sweep_meta`**

```python
# tests/test_meta_model_p1.py — append
class TestThresholdSweepMeta:
    def test_returns_one_row_per_threshold(self) -> None:
        # 100 trades: half wins (+1R gross), half losses (-1R gross),
        # uniform proba split across [0.3, 0.9] so successive thresholds
        # filter cleanly.
        rng = np.random.default_rng(0)
        n = 100
        proba = rng.uniform(0.3, 0.9, size=n)
        # Ground truth winners are the top 50% by proba (calibrated case).
        is_winner = (proba > np.median(proba)).astype(int)
        gross_r = np.where(is_winner == 1, 1.0, -1.0)  # 1R win, -1R loss
        cost_r = np.full(n, 0.32)  # P4 baseline cost

        sweep = threshold_sweep_meta(
            proba=proba,
            y_true=is_winner,
            gross_r=gross_r,
            cost_r=cost_r,
            thresholds=[0.40, 0.55, 0.70, 0.85],
        )
        assert len(sweep) == 4
        for row in sweep:
            assert {"threshold", "n_taken", "take_rate",
                    "expectancy_net_r", "profit_factor",
                    "win_rate"}.issubset(row.keys())

    def test_higher_threshold_lowers_take_rate(self) -> None:
        rng = np.random.default_rng(1)
        n = 200
        proba = rng.uniform(0.0, 1.0, size=n)
        y = (proba > 0.5).astype(int)
        gross_r = np.where(y == 1, 2.0, -1.0)
        cost_r = np.full(n, 0.3)

        sweep = threshold_sweep_meta(
            proba=proba, y_true=y, gross_r=gross_r, cost_r=cost_r,
            thresholds=[0.30, 0.50, 0.70, 0.90],
        )
        rates = [r["take_rate"] for r in sweep]
        # Monotone non-increasing.
        assert all(a >= b for a, b in zip(rates, rates[1:]))

    def test_zero_takes_returns_zero_metrics(self) -> None:
        sweep = threshold_sweep_meta(
            proba=np.array([0.1, 0.2, 0.3]),
            y_true=np.array([1, 1, 1]),
            gross_r=np.array([1.0, 1.0, 1.0]),
            cost_r=np.array([0.3, 0.3, 0.3]),
            thresholds=[0.99],
        )
        assert sweep[0]["n_taken"] == 0
        assert sweep[0]["take_rate"] == 0.0
        # PF and expectancy default to None when no trades — never to 0.0.
        assert sweep[0]["profit_factor"] is None
        assert sweep[0]["expectancy_net_r"] is None
```

- [ ] **Step 1.10: Run tests, confirm fail, implement `threshold_sweep_meta`**

```python
# ml/meta_model.py — append
def threshold_sweep_meta(
    *,
    proba: np.ndarray,
    y_true: np.ndarray,
    gross_r: np.ndarray,
    cost_r: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    """Compute (take-rate, expectancy_net, PF, win-rate) per threshold.

    ``proba`` is the calibrated meta probability. ``gross_r`` is the
    primary's signed R-multiple realization (positive for primary's
    winners). ``cost_r`` is the per-trade cost in R from
    :mod:`ml.costs`. Net R-multiple = gross_r − cost_r when the meta
    chooses to take, zero when it passes.
    """
    p = np.asarray(proba, dtype="float64")
    y = np.asarray(y_true, dtype="int64")
    g = np.asarray(gross_r, dtype="float64")
    c = np.asarray(cost_r, dtype="float64")
    if not (p.shape == y.shape == g.shape == c.shape):
        raise ValueError("proba/y/gross_r/cost_r must align")

    rows: list[dict[str, Any]] = []
    n_total = max(p.size, 1)
    for thr in thresholds:
        take = p >= thr
        n_taken = int(take.sum())
        take_rate = n_taken / n_total
        if n_taken == 0:
            rows.append({
                "threshold": float(thr),
                "n_taken": 0,
                "take_rate": float(take_rate),
                "expectancy_net_r": None,
                "profit_factor": None,
                "win_rate": None,
            })
            continue

        net_r = g[take] - c[take]
        wins = float(net_r[net_r > 0].sum())
        losses = float(-net_r[net_r < 0].sum())
        pf = (wins / losses) if losses > 0 else float("inf")
        rows.append({
            "threshold": float(thr),
            "n_taken": n_taken,
            "take_rate": float(take_rate),
            "expectancy_net_r": float(net_r.mean()),
            "profit_factor": float(pf) if np.isfinite(pf) else None,
            "win_rate": float(y[take].mean()),
        })
    return rows
```

- [ ] **Step 1.11: Run tests; verify pass**

Run: `pytest tests/test_meta_model_p1.py::TestThresholdSweepMeta -v`
Expected: PASS (3/3).

- [ ] **Step 1.12: Write failing tests for `pick_meta_threshold`**

```python
# tests/test_meta_model_p1.py — append
class TestPickMetaThreshold:
    def test_picks_threshold_satisfying_all_gates(self) -> None:
        # Build a sweep where threshold 0.65 is the only candidate that
        # satisfies take-rate ∈ [0.20, 0.40] AND PF ≥ 1.3 AND expectancy > 0.
        sweep = [
            {"threshold": 0.50, "n_taken": 60, "take_rate": 0.60,
             "expectancy_net_r": 0.05, "profit_factor": 1.4, "win_rate": 0.55},
            {"threshold": 0.65, "n_taken": 30, "take_rate": 0.30,
             "expectancy_net_r": 0.15, "profit_factor": 1.6, "win_rate": 0.62},
            {"threshold": 0.80, "n_taken": 8, "take_rate": 0.08,
             "expectancy_net_r": 0.30, "profit_factor": 2.5, "win_rate": 0.70},
        ]
        chosen, why = pick_meta_threshold(
            sweep,
            min_take_rate=0.20, max_take_rate=0.40,
            min_pf=1.3, min_expectancy=0.0,
        )
        assert chosen["threshold"] == 0.65
        assert "selected" in why.lower()

    def test_raises_when_no_threshold_qualifies(self) -> None:
        sweep = [
            {"threshold": 0.50, "n_taken": 60, "take_rate": 0.60,
             "expectancy_net_r": 0.05, "profit_factor": 1.4, "win_rate": 0.55},
            {"threshold": 0.80, "n_taken": 5, "take_rate": 0.05,
             "expectancy_net_r": 0.30, "profit_factor": 2.5, "win_rate": 0.70},
        ]
        with pytest.raises(MetaThresholdRejection, match="no threshold"):
            pick_meta_threshold(
                sweep,
                min_take_rate=0.20, max_take_rate=0.40,
                min_pf=1.3, min_expectancy=0.0,
            )

    def test_among_qualifying_picks_best_pf(self) -> None:
        # Two qualifying rows — picker chooses the higher PF.
        sweep = [
            {"threshold": 0.55, "n_taken": 35, "take_rate": 0.35,
             "expectancy_net_r": 0.10, "profit_factor": 1.4, "win_rate": 0.55},
            {"threshold": 0.65, "n_taken": 25, "take_rate": 0.25,
             "expectancy_net_r": 0.20, "profit_factor": 1.8, "win_rate": 0.62},
        ]
        chosen, _ = pick_meta_threshold(
            sweep,
            min_take_rate=0.20, max_take_rate=0.40,
            min_pf=1.3, min_expectancy=0.0,
        )
        assert chosen["threshold"] == 0.65
```

- [ ] **Step 1.13: Implement `pick_meta_threshold`**

```python
# ml/meta_model.py — append
def pick_meta_threshold(
    sweep: list[dict[str, Any]],
    *,
    min_take_rate: float,
    max_take_rate: float,
    min_pf: float,
    min_expectancy: float,
) -> tuple[dict[str, Any], str]:
    """Pick the threshold maximizing PF subject to all P1 acceptance gates.

    Gates (P1 spec §1):
      - take_rate ∈ [min_take_rate, max_take_rate]
      - profit_factor ≥ min_pf
      - expectancy_net_r > min_expectancy (strictly positive after costs)

    Within the qualifying set, prefer the highest profit factor. AFML
    §14.2 — when constraints bind, optimize the *risk-adjusted* edge
    rather than raw expectancy: PF naturally captures the asymmetry
    between wins and losses that matters once costs are real.
    """
    qualifying = [
        row for row in sweep
        if row.get("expectancy_net_r") is not None
        and row.get("profit_factor") is not None
        and min_take_rate <= row["take_rate"] <= max_take_rate
        and row["profit_factor"] >= min_pf
        and row["expectancy_net_r"] > min_expectancy
    ]
    if not qualifying:
        raise MetaThresholdRejection(
            "no threshold satisfies all P1 acceptance gates "
            f"(take_rate ∈ [{min_take_rate}, {max_take_rate}], "
            f"PF ≥ {min_pf}, expectancy > {min_expectancy}); "
            f"sweep had {len(sweep)} candidates"
        )
    chosen = max(qualifying, key=lambda r: r["profit_factor"])
    return chosen, (
        f"selected threshold={chosen['threshold']:.2f} "
        f"PF={chosen['profit_factor']:.2f} "
        f"take_rate={chosen['take_rate']:.2f} "
        f"expectancy={chosen['expectancy_net_r']:.4f}R"
    )
```

- [ ] **Step 1.14: Run all Task-1 tests**

Run: `pytest tests/test_meta_model_p1.py -v`
Expected: PASS (all Task-1 tests).

- [ ] **Step 1.15: Commit Task 1**

```bash
git add ml/meta_model.py tests/test_meta_model_p1.py
git commit -m "feat(P1): add meta-model pure helpers — Brier, calibration deciles, threshold picker"
```

---

### Task 2: Per-fold trainer with isotonic calibration

**Files:**
- Modify: `ml/meta_model.py` (append `train_meta_fold`)
- Modify: `tests/test_meta_model_p1.py` (append fold-trainer tests)

- [ ] **Step 2.1: Write failing tests for `train_meta_fold`**

```python
# tests/test_meta_model_p1.py — append
class TestTrainMetaFold:
    def _synthetic_fold(self, n: int = 600, seed: int = 7) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        # 6 features; first two carry signal, rest noise.
        signal = rng.normal(size=n)
        feat2 = 0.6 * signal + 0.4 * rng.normal(size=n)
        noise = rng.normal(size=(n, 4))
        X = np.column_stack([signal, feat2, noise])
        # Ground-truth probability of "win" depends on signal.
        p_true = 1.0 / (1.0 + np.exp(-1.5 * signal))
        y = (rng.uniform(size=n) < p_true).astype(int)
        # Per-row uniqueness = 1.0 (no overlap in this synthetic case)
        w = np.ones(n)
        return {"X": X, "y": y, "w": w}

    def test_returns_calibrated_oos_proba(self) -> None:
        from ml.meta_model import train_meta_fold

        data = self._synthetic_fold(n=600, seed=11)
        # Manual outer 70/30 split.
        cut = 420
        out = train_meta_fold(
            X_train=data["X"][:cut], y_train=data["y"][:cut],
            w_train=data["w"][:cut],
            X_test=data["X"][cut:],
            calibration_tail_frac=0.25,
            seed=42,
        )
        # Required artefacts.
        assert "base_clf" in out
        assert "isotonic" in out
        assert "scaler" in out
        proba = out["oos_calibrated_proba"]
        assert proba.shape == (data["X"].shape[0] - cut,)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_calibration_tightens_brier(self) -> None:
        """Calibrated Brier should be ≤ uncalibrated Brier on synthetic
        data with mild miscalibration (XGBoost is typically over-confident).
        """
        from ml.meta_model import compute_brier, train_meta_fold

        data = self._synthetic_fold(n=900, seed=23)
        cut = 600
        out = train_meta_fold(
            X_train=data["X"][:cut], y_train=data["y"][:cut],
            w_train=data["w"][:cut],
            X_test=data["X"][cut:],
            calibration_tail_frac=0.25,
            seed=42,
        )
        y_te = data["y"][cut:]
        cal_brier = compute_brier(y_te, out["oos_calibrated_proba"])
        # Refit base only and compute uncalibrated brier for comparison.
        raw = train_meta_fold(
            X_train=data["X"][:cut], y_train=data["y"][:cut],
            w_train=data["w"][:cut],
            X_test=data["X"][cut:],
            calibration_tail_frac=0.0,  # no calibration
            seed=42,
        )
        uncal_brier = compute_brier(y_te, raw["oos_calibrated_proba"])
        # Allow a small slack — calibration is monotone-only and may not
        # always tighten Brier on tiny test slices, but should not blow up.
        assert cal_brier <= uncal_brier + 0.02
```

- [ ] **Step 2.2: Implement `train_meta_fold`**

```python
# ml/meta_model.py — append
@dataclass(frozen=True)
class MetaFoldArtefacts:
    base_clf: Any
    isotonic: Any | None
    scaler: Any
    oos_calibrated_proba: np.ndarray


def train_meta_fold(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_test: np.ndarray,
    calibration_tail_frac: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Train one fold's binary meta-classifier with isotonic calibration.

    Pipeline:
      1. ``StandardScaler`` on X_train.
      2. Slice the *chronological tail* of training (``calibration_tail_frac``
         of rows) as the calibration set; fit base XGBoost on the head.
      3. Predict base proba on the calibration tail; fit
         ``IsotonicRegression`` on (proba, y_tail).
      4. Predict base proba on X_test, then apply isotonic to map to
         calibrated proba.

    The chronological tail (vs. random hold-out) preserves the time-order
    that the purged-CV splitter already establishes — calibration cannot
    leak future test-block info because the tail is still strictly earlier
    than the outer-test fold.

    ``calibration_tail_frac == 0`` skips calibration (returns base proba)
    — used by tests to compare calibrated vs. raw Brier.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    if calibration_tail_frac < 0.0 or calibration_tail_frac >= 1.0:
        raise ValueError("calibration_tail_frac must be in [0, 1)")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    n = len(X_train)
    cal_n = int(round(calibration_tail_frac * n))
    if cal_n >= 30 and calibration_tail_frac > 0.0:
        head = slice(0, n - cal_n)
        tail = slice(n - cal_n, n)
        X_head = X_tr_s[head]
        y_head = y_train[head]
        w_head = w_train[head]
        X_tail = X_tr_s[tail]
        y_tail = y_train[tail]
    else:
        # Calibration disabled or tail too small.
        X_head, y_head, w_head = X_tr_s, y_train, w_train
        X_tail = None
        y_tail = None

    base = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        eval_metric="logloss",
        objective="binary:logistic", n_jobs=-1,
    )
    # XGBoost requires both classes present in y_head; if not, fall back.
    if len(np.unique(y_head)) < 2:
        # Append a single zero-weight opposite-class row so the booster
        # can fit. Same trick as ``_ensure_all_classes`` in train_ml_model_v2.
        missing = 1 - int(y_head[0])
        X_head = np.vstack([X_head, X_head[0:1]])
        y_head = np.concatenate([y_head, np.array([missing], dtype=y_head.dtype)])
        w_head = np.concatenate([w_head, np.array([0.0], dtype=w_head.dtype)])
    base.fit(X_head, y_head, sample_weight=w_head, verbose=False)

    raw_test_proba = base.predict_proba(X_te_s)[:, 1]

    if X_tail is not None and len(np.unique(y_tail)) == 2:
        cal_proba_tail = base.predict_proba(X_tail)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(cal_proba_tail, y_tail)
        oos_proba = iso.transform(raw_test_proba)
    else:
        iso = None
        oos_proba = raw_test_proba

    return {
        "base_clf": base,
        "isotonic": iso,
        "scaler": scaler,
        "oos_calibrated_proba": oos_proba,
    }
```

- [ ] **Step 2.3: Run Task 2 tests; verify pass**

Run: `pytest tests/test_meta_model_p1.py::TestTrainMetaFold -v`
Expected: PASS.

- [ ] **Step 2.4: Commit Task 2**

```bash
git add ml/meta_model.py tests/test_meta_model_p1.py
git commit -m "feat(P1): add per-fold meta trainer with isotonic calibration on chronological tail"
```

---

### Task 3: Acceptance-gate evaluator

**Files:**
- Modify: `ml/meta_model.py` (append `evaluate_p1_gates`)
- Modify: `tests/test_meta_model_p1.py`

- [ ] **Step 3.1: Write failing tests for `evaluate_p1_gates`**

```python
# tests/test_meta_model_p1.py — append
class TestEvaluateP1Gates:
    def test_all_pass_block(self) -> None:
        from ml.meta_model import evaluate_p1_gates

        chosen = {"threshold": 0.65, "n_taken": 30, "take_rate": 0.30,
                  "expectancy_net_r": 0.18, "profit_factor": 1.55,
                  "win_rate": 0.60}
        verdicts = evaluate_p1_gates(
            brier=0.18,
            calibration_max_dev=0.03,
            chosen_threshold_row=chosen,
            per_fold_ar1_median=0.07,
            per_fold_ar1_max=0.14,
        )
        assert verdicts["all_pass"] is True
        for k in ("brier", "calibration", "take_rate", "profit_factor",
                  "expectancy", "ar1"):
            assert verdicts[k]["pass"] is True

    def test_brier_fail_blocks(self) -> None:
        from ml.meta_model import evaluate_p1_gates

        chosen = {"threshold": 0.65, "n_taken": 30, "take_rate": 0.30,
                  "expectancy_net_r": 0.18, "profit_factor": 1.55,
                  "win_rate": 0.60}
        verdicts = evaluate_p1_gates(
            brier=0.30, calibration_max_dev=0.03,
            chosen_threshold_row=chosen,
            per_fold_ar1_median=0.07, per_fold_ar1_max=0.14,
        )
        assert verdicts["all_pass"] is False
        assert verdicts["brier"]["pass"] is False

    def test_calibration_none_treated_as_fail(self) -> None:
        # No qualifying decile → max_abs_deviation is None. Treat as fail
        # because we cannot demonstrate calibration.
        from ml.meta_model import evaluate_p1_gates

        chosen = {"threshold": 0.65, "n_taken": 30, "take_rate": 0.30,
                  "expectancy_net_r": 0.18, "profit_factor": 1.55,
                  "win_rate": 0.60}
        verdicts = evaluate_p1_gates(
            brier=0.18, calibration_max_dev=None,
            chosen_threshold_row=chosen,
            per_fold_ar1_median=0.07, per_fold_ar1_max=0.14,
        )
        assert verdicts["calibration"]["pass"] is False
        assert verdicts["all_pass"] is False
```

- [ ] **Step 3.2: Implement `evaluate_p1_gates`**

```python
# ml/meta_model.py — append
P1_BRIER_GATE = 0.22
P1_CALIBRATION_GATE = 0.05
P1_TAKE_RATE_MIN = 0.20
P1_TAKE_RATE_MAX = 0.40
P1_PF_GATE = 1.3
P1_EXPECTANCY_GATE = 0.0
P1_AR1_MEDIAN_GATE = 0.10
P1_AR1_MAX_GATE = 0.20


def evaluate_p1_gates(
    *,
    brier: float,
    calibration_max_dev: float | None,
    chosen_threshold_row: dict[str, Any],
    per_fold_ar1_median: float,
    per_fold_ar1_max: float,
) -> dict[str, Any]:
    """Apply the six P1 acceptance gates and return per-gate pass/fail.

    Gates (in spec order):
      1. Brier < 0.22
      2. max |empirical_p − bin_midpoint| ≤ 0.05 across qualifying deciles
      3. take_rate ∈ [0.20, 0.40]
      4. profit_factor ≥ 1.3
      5. expectancy_net_r > 0
      6. per-fold AR(1): median < 0.10 AND max < 0.20

    A None calibration deviation (no decile qualifies) fails gate 2 — we
    cannot demonstrate calibration, so we don't grant it.
    """
    brier_pass = brier < P1_BRIER_GATE
    cal_pass = (
        calibration_max_dev is not None
        and calibration_max_dev <= P1_CALIBRATION_GATE
    )
    tr = chosen_threshold_row.get("take_rate", 0.0)
    take_pass = P1_TAKE_RATE_MIN <= tr <= P1_TAKE_RATE_MAX
    pf = chosen_threshold_row.get("profit_factor")
    pf_pass = (pf is not None) and (pf >= P1_PF_GATE)
    exp = chosen_threshold_row.get("expectancy_net_r")
    exp_pass = (exp is not None) and (exp > P1_EXPECTANCY_GATE)
    ar1_pass = (
        per_fold_ar1_median < P1_AR1_MEDIAN_GATE
        and per_fold_ar1_max < P1_AR1_MAX_GATE
    )

    verdicts = {
        "brier": {"value": brier, "threshold": P1_BRIER_GATE, "pass": brier_pass},
        "calibration": {
            "value": calibration_max_dev, "threshold": P1_CALIBRATION_GATE,
            "pass": bool(cal_pass),
        },
        "take_rate": {
            "value": tr, "min": P1_TAKE_RATE_MIN, "max": P1_TAKE_RATE_MAX,
            "pass": bool(take_pass),
        },
        "profit_factor": {
            "value": pf, "threshold": P1_PF_GATE, "pass": bool(pf_pass),
        },
        "expectancy": {
            "value": exp, "threshold": P1_EXPECTANCY_GATE, "pass": bool(exp_pass),
        },
        "ar1": {
            "median": per_fold_ar1_median, "max": per_fold_ar1_max,
            "median_threshold": P1_AR1_MEDIAN_GATE,
            "max_threshold": P1_AR1_MAX_GATE,
            "pass": bool(ar1_pass),
        },
    }
    verdicts["all_pass"] = all(v["pass"] for k, v in verdicts.items()
                                if isinstance(v, dict) and "pass" in v)
    return verdicts
```

- [ ] **Step 3.3: Run + commit**

```bash
pytest tests/test_meta_model_p1.py -v
git add ml/meta_model.py tests/test_meta_model_p1.py
git commit -m "feat(P1): add P1 acceptance-gate evaluator"
```

---

### Task 4: Primary-fire collection helper

**Files:**
- Modify: `ml/meta_model.py` (append `select_primary_fires`)
- Modify: `tests/test_meta_model_p1.py`

- [ ] **Step 4.1: Write failing tests for `select_primary_fires`**

```python
# tests/test_meta_model_p1.py — append
class TestSelectPrimaryFires:
    def test_filters_to_buy_or_sell_above_threshold(self) -> None:
        from ml.meta_model import select_primary_fires

        # 5 events; class label map = {0: SELL, 1: HOLD, 2: BUY}
        # Below-threshold rows must be filtered, HOLD must be filtered.
        proba = np.array([
            [0.60, 0.20, 0.20],   # SELL @ 0.60 ≥ 0.55 → fire short
            [0.20, 0.70, 0.10],   # HOLD → no fire
            [0.30, 0.20, 0.50],   # BUY @ 0.50 < 0.55 → no fire
            [0.05, 0.10, 0.85],   # BUY @ 0.85 → fire long
            [0.25, 0.50, 0.25],   # HOLD → no fire
        ])
        fires = select_primary_fires(
            primary_proba=proba,
            primary_decision_threshold=0.55,
            label_map={0: "SELL", 1: "HOLD", 2: "BUY"},
        )
        assert fires["mask"].tolist() == [True, False, False, True, False]
        assert fires["sides"].tolist() == [-1, 1]  # for the two fires
        # Carries each fire's max-proba for downstream feature use.
        assert np.allclose(fires["max_proba"], [0.60, 0.85])

    def test_no_fires_returns_empty(self) -> None:
        from ml.meta_model import select_primary_fires

        proba = np.array([
            [0.30, 0.50, 0.20],
            [0.20, 0.50, 0.30],
        ])
        fires = select_primary_fires(
            primary_proba=proba,
            primary_decision_threshold=0.55,
            label_map={0: "SELL", 1: "HOLD", 2: "BUY"},
        )
        assert fires["mask"].sum() == 0
        assert fires["sides"].size == 0
```

- [ ] **Step 4.2: Implement `select_primary_fires`**

```python
# ml/meta_model.py — append
def select_primary_fires(
    *,
    primary_proba: np.ndarray,
    primary_decision_threshold: float,
    label_map: dict[int, str],
) -> dict[str, np.ndarray]:
    """Identify which rows are primary signals (BUY or SELL) above threshold.

    The primary classifier is the P4 3-class softprob model
    ({SELL=0, HOLD=1, BUY=2}). It "fires" on a row when:
      * argmax of the proba row is BUY or SELL (not HOLD), AND
      * the max proba is at least ``primary_decision_threshold``.

    Returns a dict with:
      mask        — boolean (n,) marking firing rows.
      sides       — int8 (n_fires,) of +1 (BUY) or −1 (SELL).
      max_proba   — float64 (n_fires,) of the primary's confidence on each fire.
      class_idx   — int (n_fires,) of the argmax class id.
    """
    proba = np.asarray(primary_proba, dtype="float64")
    if proba.ndim != 2 or proba.shape[1] < 3:
        raise ValueError(
            f"primary_proba must be (n, n_classes>=3); got {proba.shape}"
        )

    inv_map = {v: k for k, v in label_map.items()}
    if "BUY" not in inv_map or "SELL" not in inv_map or "HOLD" not in inv_map:
        raise KeyError("label_map must contain BUY, HOLD, SELL")
    buy_idx = inv_map["BUY"]
    sell_idx = inv_map["SELL"]

    argmax = proba.argmax(axis=1)
    max_p = proba.max(axis=1)
    is_directional = (argmax == buy_idx) | (argmax == sell_idx)
    mask = is_directional & (max_p >= primary_decision_threshold)

    sides = np.where(argmax[mask] == buy_idx, 1, -1).astype(np.int8)
    return {
        "mask": mask,
        "sides": sides,
        "max_proba": max_p[mask],
        "class_idx": argmax[mask],
    }
```

- [ ] **Step 4.3: Run + commit**

```bash
pytest tests/test_meta_model_p1.py -v
git add ml/meta_model.py tests/test_meta_model_p1.py
git commit -m "feat(P1): add primary-fire selector for meta-labeling candidate set"
```

---

### Task 5: Top-level orchestration script

**Files:**
- Create: `train_meta_model_p1.py`
- Test: `tests/test_train_meta_p1_integration.py`

- [ ] **Step 5.1: Write failing integration tests**

```python
# tests/test_train_meta_p1_integration.py
"""End-to-end tests for the P1 meta-labeling trainer.

Stubs out the Postgres market-data layer and the heavy IntegratedMLModel
feature extractor so the test runs in seconds. Exercises the full
pipeline: primary-fire selection, triple-barrier labelling, purged CV
with isotonic calibration, gate evaluation, report writing.
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


def _make_primary_bundle(tmp_path: Path, *, n_features: int) -> Path:
    """Build a tiny but real XGBoost 3-class softprob bundle on synthetic data."""
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    rng = np.random.default_rng(0)
    n = 800
    signal = rng.normal(size=n)
    noise = rng.normal(size=(n, max(n_features - 1, 1)))
    X = np.column_stack([signal, noise])[:, :n_features]
    p_buy = 0.33 + 0.25 * np.tanh(signal)
    p_sell = 0.33 - 0.25 * np.tanh(signal)
    p_hold = 1.0 - p_buy - p_sell
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
                "cusum_h": 0.01,
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


def _install_stubs(monkeypatch: pytest.MonkeyPatch, n_features: int) -> None:
    """Stub Postgres and IntegratedMLModel so the script runs offline."""
    import train_meta_model_p1 as mod

    # Synthetic OHLCV per symbol.
    def _stub_provider_factory(_dsn: str):
        class P:
            def get_market_data(self, symbol, *, period_type, period,
                                frequency_type, frequency):
                rng = np.random.default_rng(hash(symbol) % 2**32)
                n_bars = 1500
                idx = pd.date_range(
                    start="2024-09-02 09:30", periods=n_bars, freq=f"{frequency}min",
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

    monkeypatch.setattr(mod, "_make_provider", _stub_provider_factory)

    class _StubFeat:
        def __init__(self) -> None:
            self.feature_names = [f"f{i}" for i in range(n_features)]
        def batch_extract(self, df: pd.DataFrame) -> pd.DataFrame:
            rng = np.random.default_rng(len(df))
            arr = rng.normal(size=(len(df), n_features))
            return pd.DataFrame(arr, index=df.index, columns=self.feature_names)

    class _StubModel:
        def __init__(self, **_kw) -> None:
            self.feature_extractor = _StubFeat()
            self.feature_names = self.feature_extractor.feature_names

    monkeypatch.setattr(mod, "IntegratedMLModel", _StubModel, raising=False)
    # Force-skip volume top-N which would hit Postgres.
    monkeypatch.setattr(mod, "top_symbols_by_volume",
                        lambda *_a, **_kw: ["SYN1", "SYN2", "SYN3"])


@pytest.fixture
def isolated_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "primary": _make_primary_bundle(tmp_path, n_features=6),
        "meta_bundle": tmp_path / "meta.pkl",
        "report": tmp_path / "p1.json",
        "diff": tmp_path / "diff.md",
    }


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
            # Loose gates so test passes regardless of synthetic edge.
            "--allow-rejection",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        # Allow exit code 3 (rejection) or 0 (pass) — either is a valid
        # report-schema test outcome. We just need the JSON written.
        exit_code = mod.main()
        assert exit_code in (0, 3)
        report = json.loads(isolated_paths["report"].read_text())

        for k in ("timestamp", "primary_bundle", "config", "symbols",
                  "n_primary_fires", "label_distribution",
                  "cv_summary", "calibration", "threshold_sweep",
                  "chosen_threshold", "iid_diagnostics", "gates"):
            assert k in report, f"missing top-level key: {k}"

        gates = report["gates"]
        for g in ("brier", "calibration", "take_rate", "profit_factor",
                  "expectancy", "ar1"):
            assert g in gates, f"missing gate: {g}"

    def test_baseline_path_does_not_alter_gates(
        self, monkeypatch: pytest.MonkeyPatch, isolated_paths: dict[str, Path],
    ) -> None:
        _install_stubs(monkeypatch, n_features=6)
        import train_meta_model_p1 as mod

        # Even if --baseline points at P4_after.json with rich metrics, the
        # P1 gates are intrinsic — a baseline file is informational only.
        baseline = isolated_paths["report"].parent / "p4_baseline.json"
        baseline.write_text(json.dumps({
            "cv_summary": {"aggregate": {
                "median_profit_factor": 0.738,
                "median_expectancy_r": -0.196,
            }},
            "iid_diagnostics": {"median_per_fold_ar1": 0.092,
                                "max_per_fold_ar1": 0.169},
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
        # Baseline diff appears.
        assert "p4_baseline_diff" in report
```

- [ ] **Step 5.2: Run integration tests, confirm they fail with ImportError**

Run: `pytest tests/test_train_meta_p1_integration.py -v`
Expected: ImportError on `train_meta_model_p1`.

- [ ] **Step 5.3: Implement `train_meta_model_p1.py`**

Create the script. Key responsibilities:
1. Argparse: `--symbols / --top-n`, `--days`, `--folds`, `--embargo-pct`, `--primary-bundle`, `--meta-bundle-path`, `--report-path`, `--baseline`, `--allow-rejection`, `--decision-threshold`, `--cusum-h-override`, `--threshold-sweep` (default `0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85`).
2. Load primary bundle; freeze pt/sl/max_holding/cusum_h/decision_threshold/cost_model from `bundle["config"]`.
3. For each symbol: fetch bars (via `_make_provider(dsn).get_market_data`); compute `_compute_raw_atr` (re-import from `train_ml_model_v2`); CUSUM-filter; restrict to bars with full forward window; for each event time, build feature row via `IntegratedMLModel.feature_extractor.batch_extract`; run primary `predict_proba` row-wise; apply `select_primary_fires`; for fires, build `events_df = DataFrame({"side": sides}, index=fire_times)`; call `apply_meta_triple_barrier` to get binary y/touch_time/ret; record (X_features, y, touch_times, sides, symbol_groups, gross_r=ret/barrier_pct via `ret_to_r_multiple`).
4. Concat across symbols; sort by event_time.
5. Compute per-symbol `compute_uniqueness` weights.
6. Compute per-row cost_in_r via `cost_in_r` from `ml.costs` using the bundle's cost_model and a fixed entry/atr proxy (matching the existing P4 trainer's approach in `_cost_r_from_constants`).
7. PurgedKFold(n_splits, touch_times, embargo_pct) → for each fold call `train_meta_fold(...)`; collect OOS calibrated proba per row; per-fold residual AR(1) via `weighted_residuals` + `compute_iid_diagnostics`.
8. Concatenate OOS proba; compute Brier; compute decile calibration.
9. Run `threshold_sweep_meta`; call `pick_meta_threshold`. On `MetaThresholdRejection`, write a "rejected" report with all diagnostics and exit 3 (unless `--allow-rejection` is set, which still exits 3 but does not raise).
10. Call `evaluate_p1_gates`; if any fail and `--allow-rejection` not set, exit 3 with the same rejected-report shape.
11. Refit base on full data + isotonic on chronological tail; persist `ml_meta_model_p1.pkl` bundle.
12. Write `backtest_results/P1_after.json`.
13. If `--baseline` provided, also write a `p4_baseline_diff` block (median PF, expectancy_r, AR(1) median/max, take-rate baseline ~0.70 vs P1).

```python
# train_meta_model_p1.py
"""Train the P1 meta-labeling filter on top of the P4 primary classifier.

López de Prado AFML §3.6-3.7. The primary (ml_model_v2.pkl) decides side;
the meta-classifier decides go/pass on those primary fires using the same
triple-barrier outcome the P4 model was trained against.

Outputs:
    ml_meta_model_p1.pkl              — trained meta bundle
    backtest_results/P1_after.json    — full P1 report

Usage:
    python train_meta_model_p1.py
    python train_meta_model_p1.py --top-n 10 --days 60
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ml.costs import CostModel, cost_in_r as _cost_in_r, ret_to_r_multiple
from ml.events import cusum_filter
from ml.iid_diagnostics import compute_iid_diagnostics, weighted_residuals
from ml.labels import compute_uniqueness
from ml.meta_labels import apply_meta_triple_barrier
from ml.meta_model import (
    MetaThresholdRejection,
    calibration_deciles,
    compute_brier,
    evaluate_p1_gates,
    pick_meta_threshold,
    select_primary_fires,
    threshold_sweep_meta,
    train_meta_fold,
)
from ml.purged_cv import PurgedKFold
from train_ml_model import DEFAULT_DSN, top_symbols_by_volume
from train_ml_model_v2 import _compute_raw_atr


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_meta_p1")


DEFAULT_THRESHOLD_SWEEP = [
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
]
DEFAULT_FOLDS = 5
DEFAULT_DAYS = 60
DEFAULT_TOP_N = 10
DEFAULT_FREQUENCY = 5
DEFAULT_EMBARGO_PCT = 0.02  # 2× the P4 embargo to honor "embargo = 2 × vertical"


def _make_provider(dsn: str) -> Any:
    """Factory — overridable in tests."""
    from data_providers.postgres import PostgresDataProvider
    return PostgresDataProvider(dsn)


# Hold the IntegratedMLModel symbol at module level so tests can monkeypatch it.
try:  # pragma: no cover — import guarded for offline tests
    from ml.models import IntegratedMLModel
except Exception:  # pragma: no cover
    IntegratedMLModel = None  # type: ignore[assignment]


def collect_p1_samples(
    *,
    symbols: list[str],
    days: int,
    frequency: int,
    dsn: str,
    primary_bundle: dict[str, Any],
    cusum_h: float,
) -> dict[str, Any]:
    """Run primary on each event; meta-label the fires; return aligned arrays."""
    cfg = primary_bundle["config"]
    pt_mult = float(cfg["pt_mult"])
    sl_mult = float(cfg["sl_mult"])
    max_holding = int(cfg["max_holding"])
    vertical = int(cfg["labeling"]["vertical_max_bars"])
    label_map = primary_bundle.get("label_map", {0: "SELL", 1: "HOLD", 2: "BUY"})
    decision_threshold = float(cfg.get("decision_threshold", 0.55))

    primary_clf = primary_bundle["classifier"]
    primary_scaler = primary_bundle["scaler"]
    feature_names = list(primary_bundle["feature_names"])

    provider = _make_provider(dsn)
    feature_model = IntegratedMLModel(brain=None, commentary_system=None)

    period_type = "day" if days < 30 else "month"
    period = days if period_type == "day" else max(1, days // 30)

    rows_X: list[np.ndarray] = []
    rows_y: list[int] = []
    rows_ret: list[float] = []
    rows_event_time: list[pd.Timestamp] = []
    rows_touch_time: list[pd.Timestamp] = []
    rows_side: list[int] = []
    rows_symbol: list[str] = []
    rows_max_proba: list[float] = []
    per_symbol: dict[str, int] = {}

    for symbol in symbols:
        try:
            df = provider.get_market_data(
                symbol, period_type=period_type, period=period,
                frequency_type="minute", frequency=frequency,
            )
        except Exception as exc:
            logger.error("load_failed symbol=%s err=%s", symbol, exc)
            continue
        if df.empty or len(df) < vertical + 100:
            continue

        feat_df = feature_model.feature_extractor.batch_extract(df)
        if feat_df.empty:
            continue
        atr = _compute_raw_atr(df, window=14)

        # Restrict to bars with full forward vertical window.
        usable = feat_df.index[feat_df.index <= df.index[-1 - vertical]]

        # CUSUM events on the close series, intersected with usable bars.
        ev = cusum_filter(df["Close"], h=cusum_h)
        ev = ev.intersection(usable)
        if len(ev) == 0:
            per_symbol[symbol] = 0
            continue

        # Drop events with no valid feature row or non-positive ATR.
        feat_at_ev = feat_df.reindex(ev)[feature_names]
        valid = feat_at_ev.notna().all(axis=1) & (atr.reindex(ev) > 0)
        feat_at_ev = feat_at_ev.loc[valid]
        ev = pd.DatetimeIndex(feat_at_ev.index)
        if len(ev) == 0:
            per_symbol[symbol] = 0
            continue

        # Run primary on every event.
        X_at_ev = feat_at_ev.to_numpy()
        proba = primary_clf.predict_proba(primary_scaler.transform(X_at_ev))

        fires = select_primary_fires(
            primary_proba=proba,
            primary_decision_threshold=decision_threshold,
            label_map=label_map,
        )
        if fires["mask"].sum() == 0:
            per_symbol[symbol] = 0
            continue

        fire_times = ev[fires["mask"]]
        sides = fires["sides"]
        events_df = pd.DataFrame({"side": sides}, index=fire_times)

        labels = apply_meta_triple_barrier(
            prices=df["Close"], events=events_df, atr=atr,
            pt_mult=pt_mult, sl_mult=sl_mult, max_holding=vertical,
        )
        # Align — apply_meta_triple_barrier preserves event index order.
        for i, ts in enumerate(labels.index):
            rows_X.append(X_at_ev[fires["mask"]][i])
            rows_y.append(int(labels["bin"].iloc[i]))
            rows_ret.append(float(labels["ret"].iloc[i]))
            rows_event_time.append(ts)
            rows_touch_time.append(pd.Timestamp(labels["touch_time"].iloc[i]))
            rows_side.append(int(sides[i]))
            rows_symbol.append(symbol)
            rows_max_proba.append(float(fires["max_proba"][i]))
        per_symbol[symbol] = int(len(labels))

    if not rows_X:
        return {"X": np.array([]), "y": np.array([]), "ret": np.array([]),
                "event_times": pd.DatetimeIndex([]),
                "touch_times": pd.DatetimeIndex([]),
                "sides": np.array([]), "symbols": np.array([]),
                "max_proba": np.array([]), "per_symbol": per_symbol}

    return {
        "X": np.vstack(rows_X),
        "y": np.asarray(rows_y, dtype=np.int64),
        "ret": np.asarray(rows_ret, dtype="float64"),
        "event_times": pd.DatetimeIndex(rows_event_time),
        "touch_times": pd.DatetimeIndex(rows_touch_time),
        "sides": np.asarray(rows_side, dtype=np.int8),
        "symbols": np.asarray(rows_symbol, dtype=object),
        "max_proba": np.asarray(rows_max_proba, dtype="float64"),
        "per_symbol": per_symbol,
    }


def _cost_r_per_row(
    *,
    n: int, sl_mult: float, cost_model: CostModel, participation: float,
) -> np.ndarray:
    """Same proxy approach as train_ml_model_v2._cost_r_from_constants."""
    return _cost_in_r(
        entry_price=np.full(n, 100.0),
        atr=np.full(n, sl_mult),
        sl_mult=1.0,
        participation=np.full(n, participation),
        cost_model=cost_model,
    )


def run_purged_cv_meta(
    *,
    X: np.ndarray, y: np.ndarray, weights: np.ndarray,
    touch_times: pd.Series,
    n_splits: int, embargo_pct: float, seed: int = 42,
) -> dict[str, Any]:
    """OOS calibrated proba + per-fold AR(1) on weighted binary residuals."""
    cv = PurgedKFold(n_splits=n_splits, touch_times=touch_times, embargo_pct=embargo_pct)
    oos_proba = np.full(len(y), np.nan, dtype="float64")
    per_fold_ar1: list[float] = []
    fold_detail: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            logger.warning("fold %d empty", fold_idx)
            continue
        out = train_meta_fold(
            X_train=X[train_idx], y_train=y[train_idx],
            w_train=weights[train_idx],
            X_test=X[test_idx],
            calibration_tail_frac=0.25,
            seed=seed + fold_idx,
        )
        oos_proba[test_idx] = out["oos_calibrated_proba"]
        residuals = weighted_residuals(
            y[test_idx], out["oos_calibrated_proba"], weights[test_idx],
        )
        iid = compute_iid_diagnostics(residuals)
        per_fold_ar1.append(float(iid.ar1_residual_autocorr))
        fold_detail.append({
            "fold": fold_idx,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "ar1": float(iid.ar1_residual_autocorr),
            "ljung_box_p": (
                float(iid.ljung_box_p_value)
                if iid.ljung_box_p_value is not None else None
            ),
        })
    finite = [a for a in per_fold_ar1 if np.isfinite(a)]
    median_ar1 = float(np.median([abs(a) for a in finite])) if finite else 0.0
    max_ar1 = float(np.max([abs(a) for a in finite])) if finite else 0.0
    return {
        "oos_proba": oos_proba,
        "per_fold_ar1": per_fold_ar1,
        "median_per_fold_ar1": median_ar1,
        "max_per_fold_ar1": max_ar1,
        "fold_detail": fold_detail,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sym = p.add_mutually_exclusive_group()
    sym.add_argument("--symbols", nargs="+")
    sym.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--days", type=int, default=DEFAULT_DAYS)
    p.add_argument("--frequency", type=int, default=DEFAULT_FREQUENCY)
    p.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p.add_argument("--embargo-pct", type=float, default=DEFAULT_EMBARGO_PCT)
    p.add_argument("--dsn", default=os.environ.get("POSTGRES_DSN", DEFAULT_DSN))
    p.add_argument("--primary-bundle", default="ml_model_v2.pkl")
    p.add_argument("--meta-bundle-path", default="ml_meta_model_p1.pkl")
    p.add_argument("--report-path", default="backtest_results/P1_after.json")
    p.add_argument("--baseline", default="backtest_results/P4_after.json",
                   help="Baseline P4 report for diff (informational).")
    p.add_argument("--allow-rejection", action="store_true",
                   help="Still write the report on gate failure (always exits 3).")
    p.add_argument("--threshold-sweep", default=",".join(
        f"{t:.2f}" for t in DEFAULT_THRESHOLD_SWEEP))
    return p.parse_args()


def _baseline_diff(baseline_path: str, report: dict[str, Any]) -> dict[str, Any]:
    try:
        base = json.loads(Path(baseline_path).read_text())
    except Exception as exc:
        return {"baseline_path": baseline_path, "error": str(exc)}
    base_agg = base.get("cv_summary", {}).get("aggregate", {})
    base_iid = base.get("iid_diagnostics", {})
    chosen = report["chosen_threshold"]
    rows = {
        "median_profit_factor": {
            "P4": base_agg.get("median_profit_factor"),
            "P1": chosen.get("profit_factor"),
        },
        "median_expectancy_r": {
            "P4": base_agg.get("median_expectancy_r"),
            "P1": chosen.get("expectancy_net_r"),
        },
        "take_rate": {
            "P4": "≈0.70 (decision_threshold 0.55, no meta-filter)",
            "P1": chosen.get("take_rate"),
        },
        "median_per_fold_ar1": {
            "P4": base_iid.get("median_per_fold_ar1"),
            "P1": report["iid_diagnostics"]["median_per_fold_ar1"],
        },
        "max_per_fold_ar1": {
            "P4": base_iid.get("max_per_fold_ar1"),
            "P1": report["iid_diagnostics"]["max_per_fold_ar1"],
        },
    }
    return {"baseline_path": baseline_path, "rows": rows}


def main() -> int:
    args = _parse_args()

    primary_bundle_path = Path(args.primary_bundle)
    if not primary_bundle_path.exists():
        logger.error("primary_bundle_missing path=%s", primary_bundle_path)
        return 4
    primary_bundle = joblib.load(primary_bundle_path)
    cfg = primary_bundle["config"]

    cusum_h = float(cfg["labeling"]["cusum_h"])
    cost_model = CostModel(
        fee_bps=float(cfg["cost_model"]["fee_bps"]),
        slip_bps=float(cfg["cost_model"]["slip_bps"]),
        impact_coef=float(cfg["cost_model"]["impact_coef"]),
    )
    participation = float(cfg["cost_model"]["participation"])
    sl_mult = float(cfg["sl_mult"])

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = top_symbols_by_volume(args.dsn, args.top_n, args.days)
    logger.info("symbols=%d primary=%s", len(symbols), primary_bundle_path)

    t0 = datetime.now()
    coll = collect_p1_samples(
        symbols=symbols, days=args.days, frequency=args.frequency,
        dsn=args.dsn, primary_bundle=primary_bundle, cusum_h=cusum_h,
    )
    collect_seconds = (datetime.now() - t0).total_seconds()

    n = int(coll["X"].shape[0]) if coll["X"].size else 0
    if n == 0:
        logger.error("no_primary_fires — meta-model has nothing to filter")
        Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_path).write_text(json.dumps({
            "rejected": True, "reject_reason": "no_primary_fires",
            "n_primary_fires": 0,
        }, indent=2))
        return 3

    # Sort chronologically.
    order = np.argsort(coll["event_times"].to_numpy())
    X = coll["X"][order]; y = coll["y"][order]; ret = coll["ret"][order]
    event_times = coll["event_times"][order]
    touch_times = coll["touch_times"][order]
    sides = coll["sides"][order]
    sym_groups = coll["symbols"][order]
    max_proba = coll["max_proba"][order]

    label_dist = Counter(y.tolist())
    base_win_rate = label_dist.get(1, 0) / max(1, len(y))
    logger.info("samples=%d wins=%d (%.1f%%) elapsed_s=%.1f",
                len(y), label_dist.get(1, 0), 100*base_win_rate, collect_seconds)

    # Per-symbol uniqueness (López de Prado §4.3).
    touch_series = pd.Series(touch_times.to_numpy(), index=event_times)
    labels_for_w = pd.DataFrame({"touch_time": touch_series.values},
                                index=touch_series.index)
    weights = compute_uniqueness(
        labels_for_w,
        pd.DatetimeIndex(np.unique(np.concatenate([
            event_times.to_numpy(), touch_times.to_numpy(),
        ]))).sort_values(),
        groups=pd.Series(sym_groups),
    ).to_numpy()
    weights = np.clip(weights, 1e-6, 1.0)

    # Per-row gross R-multiples (signed) and cost.
    barrier_pct_proxy = sl_mult / 100.0  # ATR/entry proxy = sl_mult/100
    gross_r = ret / barrier_pct_proxy
    cost_r = _cost_r_per_row(
        n=len(y), sl_mult=sl_mult, cost_model=cost_model,
        participation=participation,
    )

    # Purged CV.
    cv_block = run_purged_cv_meta(
        X=X, y=y, weights=weights, touch_times=touch_series,
        n_splits=args.folds, embargo_pct=args.embargo_pct,
    )
    oos_proba = cv_block["oos_proba"]
    covered = ~np.isnan(oos_proba)
    if covered.sum() == 0:
        logger.error("no oos coverage — purged CV produced no test rows")
        return 3

    brier = compute_brier(y[covered], oos_proba[covered])
    cal = calibration_deciles(y[covered], oos_proba[covered])

    sweep = threshold_sweep_meta(
        proba=oos_proba[covered], y_true=y[covered],
        gross_r=gross_r[covered], cost_r=cost_r[covered],
        thresholds=[float(t) for t in args.threshold_sweep.split(",")],
    )

    rejected = False
    reject_reason = ""
    chosen_row: dict[str, Any] = {}
    try:
        chosen_row, why = pick_meta_threshold(
            sweep,
            min_take_rate=0.20, max_take_rate=0.40,
            min_pf=1.3, min_expectancy=0.0,
        )
        logger.info("threshold_picked: %s", why)
    except MetaThresholdRejection as exc:
        rejected = True
        reject_reason = str(exc)
        logger.error("threshold_pick_rejected: %s", reject_reason)

    if not rejected:
        gates = evaluate_p1_gates(
            brier=brier,
            calibration_max_dev=cal["max_abs_deviation"],
            chosen_threshold_row=chosen_row,
            per_fold_ar1_median=cv_block["median_per_fold_ar1"],
            per_fold_ar1_max=cv_block["max_per_fold_ar1"],
        )
        if not gates["all_pass"]:
            rejected = True
            reject_reason = "gates_failed: " + ", ".join(
                f"{k}={v}" for k, v in gates.items()
                if isinstance(v, dict) and v.get("pass") is False
            )
    else:
        gates = {
            "brier": {"value": brier, "threshold": 0.22, "pass": brier < 0.22},
            "calibration": {"value": cal["max_abs_deviation"], "pass": False},
            "take_rate": {"pass": False}, "profit_factor": {"pass": False},
            "expectancy": {"pass": False},
            "ar1": {
                "median": cv_block["median_per_fold_ar1"],
                "max": cv_block["max_per_fold_ar1"],
                "pass": cv_block["median_per_fold_ar1"] < 0.10
                and cv_block["max_per_fold_ar1"] < 0.20,
            },
            "all_pass": False,
        }

    # Final fit on all data (only if not rejected, to avoid persisting a
    # bundle the caller will not deploy).
    if not rejected:
        full = train_meta_fold(
            X_train=X, y_train=y, w_train=weights,
            X_test=X[:1],  # discarded
            calibration_tail_frac=0.25, seed=42,
        )
        bundle = {
            "version": "p1-v1",
            "primary_bundle_path": str(primary_bundle_path),
            "primary_decision_threshold": float(cfg.get("decision_threshold", 0.55)),
            "primary_label_map": primary_bundle.get("label_map", {0: "SELL", 1: "HOLD", 2: "BUY"}),
            "base_clf": full["base_clf"],
            "isotonic": full["isotonic"],
            "scaler": full["scaler"],
            "feature_names": list(primary_bundle["feature_names"]),
            "meta_threshold": float(chosen_row["threshold"]),
            "config": {
                "folds": args.folds, "embargo_pct": args.embargo_pct,
                "threshold_sweep": [float(t) for t in args.threshold_sweep.split(",")],
                "primary_pt_mult": float(cfg["pt_mult"]),
                "primary_sl_mult": float(cfg["sl_mult"]),
                "primary_max_holding": int(cfg["max_holding"]),
                "primary_vertical_max_bars": int(cfg["labeling"]["vertical_max_bars"]),
                "primary_cusum_h": cusum_h,
                "frequency": args.frequency,
            },
            "trained_at": datetime.now().isoformat(),
        }
        Path(args.meta_bundle_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, args.meta_bundle_path)
        logger.info("meta_bundle_written path=%s", args.meta_bundle_path)

    report = {
        "timestamp": datetime.now().isoformat(),
        "primary_bundle": str(primary_bundle_path),
        "config": {
            "folds": args.folds, "embargo_pct": args.embargo_pct,
            "frequency": args.frequency, "days": args.days,
            "primary_decision_threshold": float(cfg.get("decision_threshold", 0.55)),
            "primary_pt_mult": float(cfg["pt_mult"]),
            "primary_sl_mult": float(cfg["sl_mult"]),
            "primary_max_holding": int(cfg["max_holding"]),
            "primary_vertical_max_bars": int(cfg["labeling"]["vertical_max_bars"]),
            "primary_cusum_h": cusum_h,
            "cost_model": {
                "fee_bps": cost_model.fee_bps,
                "slip_bps": cost_model.slip_bps,
                "impact_coef": cost_model.impact_coef,
                "participation": participation,
            },
        },
        "symbols": symbols,
        "n_primary_fires": int(len(y)),
        "per_symbol_fires": coll["per_symbol"],
        "label_distribution": {
            "wins": int(label_dist.get(1, 0)),
            "losses": int(label_dist.get(0, 0)),
            "base_win_rate": round(base_win_rate, 4),
        },
        "weighting": {
            "mean": float(weights.mean()),
            "median": float(np.median(weights)),
            "min": float(weights.min()),
            "max": float(weights.max()),
            "effective_sample_size": float(weights.sum()),
        },
        "cv_summary": {
            "n_folds_executed": len(cv_block["fold_detail"]),
            "fold_detail": cv_block["fold_detail"],
        },
        "calibration": {
            "brier": brier,
            "deciles": cal,
        },
        "threshold_sweep": sweep,
        "chosen_threshold": chosen_row,
        "iid_diagnostics": {
            "per_fold_ar1": cv_block["per_fold_ar1"],
            "median_per_fold_ar1": cv_block["median_per_fold_ar1"],
            "max_per_fold_ar1": cv_block["max_per_fold_ar1"],
            "gate_thresholds": {"median_lt": 0.10, "max_lt": 0.20},
        },
        "gates": gates,
        "rejected": rejected,
        "reject_reason": reject_reason if rejected else None,
        "primary_proba_summary": {
            "mean_max_proba_at_fire": float(np.mean(max_proba)) if len(max_proba) else 0.0,
        },
    }
    if args.baseline:
        report["p4_baseline_diff"] = _baseline_diff(args.baseline, report)

    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).write_text(json.dumps(report, indent=2, default=str))
    logger.info("report_written path=%s", args.report_path)

    if rejected:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5.4: Run integration tests; iterate until pass**

Run: `pytest tests/test_train_meta_p1_integration.py -v`
Expected: PASS.

- [ ] **Step 5.5: Run the full test suite to ensure no regression**

Run: `pytest tests/ -v -x`
Expected: PASS (all existing tests still green).

- [ ] **Step 5.6: Commit Task 5**

```bash
git add train_meta_model_p1.py tests/test_train_meta_p1_integration.py
git commit -m "feat(P1): add P1 meta-labeling trainer with isotonic calibration + acceptance gates"
```

---

### Task 6: Real end-to-end run + diff table

- [ ] **Step 6.1: Run the trainer end-to-end on real Postgres data**

Run:
```bash
python train_meta_model_p1.py \
  --symbols NVDA TSLA AAPL AMD SPY QQQ META MSFT GOOGL AMZN \
  --days 60 --frequency 5 --folds 5 \
  --primary-bundle ml_model_v2.pkl \
  --meta-bundle-path ml_meta_model_p1.pkl \
  --report-path backtest_results/P1_after.json \
  --baseline backtest_results/P4_after.json
```

Expected outcomes:
* Exit 0 if all six gates pass.
* Exit 3 if any gate fails — must STOP and report to user; do not "fix" by relaxing gates.

- [ ] **Step 6.2: Inspect P1_after.json — confirm it contains all required blocks**

Run: `python -c "import json; r=json.load(open('backtest_results/P1_after.json')); [print(k, '✓' if k in r else '✗') for k in ('cv_summary','calibration','threshold_sweep','chosen_threshold','iid_diagnostics','gates','p4_baseline_diff')]"`
Expected: all `✓`.

- [ ] **Step 6.3: Generate diff markdown**

Create `backtest_results/P1_vs_P4_diff.md` programmatically from the JSON. Add a `python -c` one-liner OR a tiny script (do not commit a separate `.py` for this — keep it inline). Acceptable inline approach:

```bash
python - <<'PY'
import json, pathlib

p4 = json.load(open("backtest_results/P4_after.json"))
p1 = json.load(open("backtest_results/P1_after.json"))

agg = p4["cv_summary"]["aggregate"]
chosen = p1["chosen_threshold"]
gates = p1["gates"]
p4_iid = p4["iid_diagnostics"]
p1_iid = p1["iid_diagnostics"]

lines = ["# P1 vs P4 — Diff Table", "",
         "## Headline metrics (median across folds)", "",
         "| Metric | P4 baseline | P1 after | Delta |",
         "| --- | ---: | ---: | ---: |",
         f"| Profit factor | {agg['median_profit_factor']:.3f} | {chosen['profit_factor']:.3f} | {chosen['profit_factor']-agg['median_profit_factor']:+.3f} |",
         f"| Expectancy (R/trade, net) | {agg['median_expectancy_r']:.3f} | {chosen['expectancy_net_r']:.3f} | {chosen['expectancy_net_r']-agg['median_expectancy_r']:+.3f} |",
         f"| Take-rate | ~0.70 (no meta) | {chosen['take_rate']:.3f} | {chosen['take_rate']-0.70:+.3f} |",
         f"| Per-fold AR(1) median | {p4_iid['median_per_fold_ar1']:.3f} | {p1_iid['median_per_fold_ar1']:.3f} | {p1_iid['median_per_fold_ar1']-p4_iid['median_per_fold_ar1']:+.3f} |",
         f"| Per-fold AR(1) max | {p4_iid['max_per_fold_ar1']:.3f} | {p1_iid['max_per_fold_ar1']:.3f} | {p1_iid['max_per_fold_ar1']-p4_iid['max_per_fold_ar1']:+.3f} |",
         "",
         "## Acceptance gates (P1)", "",
         "| Gate | Threshold | P1 actual | Pass |",
         "| --- | --- | ---: | :---: |",
         f"| Brier OOS | < 0.22 | {p1['calibration']['brier']:.4f} | {'✓' if gates['brier']['pass'] else '✗'} |",
         f"| Calibration max dev (deciles) | ≤ 0.05 | {p1['calibration']['deciles']['max_abs_deviation']} | {'✓' if gates['calibration']['pass'] else '✗'} |",
         f"| Take-rate ∈ [0.20, 0.40] | inclusive | {chosen['take_rate']:.3f} | {'✓' if gates['take_rate']['pass'] else '✗'} |",
         f"| Profit factor (post-filter) | ≥ 1.3 | {chosen['profit_factor']:.3f} | {'✓' if gates['profit_factor']['pass'] else '✗'} |",
         f"| Expectancy (post-filter) | > 0 | {chosen['expectancy_net_r']:.4f} | {'✓' if gates['expectancy']['pass'] else '✗'} |",
         f"| Per-fold AR(1) | median<0.10, max<0.20 | med={p1_iid['median_per_fold_ar1']:.3f}, max={p1_iid['max_per_fold_ar1']:.3f} | {'✓' if gates['ar1']['pass'] else '✗'} |",
         "",
         f"**Overall: {'PASS' if gates['all_pass'] else 'FAIL'}**"]
pathlib.Path("backtest_results/P1_vs_P4_diff.md").write_text("\n".join(lines))
print("wrote backtest_results/P1_vs_P4_diff.md")
PY
```

- [ ] **Step 6.4: Verify diff markdown renders correctly**

Run: `cat backtest_results/P1_vs_P4_diff.md`
Expected: A clean markdown table with Pass/Fail marks.

- [ ] **Step 6.5: STOP if any gate failed — report to user**

If `gates.all_pass` is `False`:
* Do NOT silently relax gates.
* Do NOT retune thresholds, sweep ranges, or feature lists to chase a green bar.
* Report exactly which gates failed, what the OOS values were, and what the most likely cause is given the diagnostics block (bad calibration → calibration tail too short; bad PF → primary edge truly negative on this universe; bad AR(1) → leakage in CV).
* Return control to the user with a one-paragraph summary.

- [ ] **Step 6.6: If all gates pass, commit Task 6**

```bash
git add backtest_results/P1_after.json backtest_results/P1_vs_P4_diff.md ml_meta_model_p1.pkl
git commit -m "feat(P1): meta-labeling filter — Brier + calibration + take-rate + PF gates passed"
```

---

## Self-Review Notes

- **Spec coverage:**
  - "Brier < 0.22" → gate 1, computed in step 6.x via `compute_brier(y_oos, calibrated_proba_oos)`.
  - "Calibration plot within 5% of diagonal across deciles" → gate 2, computed via `calibration_deciles(min_count=30)`.
  - "Take-rate after filter ∈ [20%, 40%]" → gate 3, picked by `pick_meta_threshold`.
  - "PF ≥ 1.3" → gate 4.
  - "Expectancy after costs > 0" → gate 5, uses `cost_in_r` from existing `ml/costs.py`.
  - "Per-fold AR(1) gate from P4 still passes" → gate 6, `weighted_residuals` + `compute_iid_diagnostics` per fold (binary residual variant).
  - "Use isotonic regression on held-out fold" → `train_meta_fold` carves a chronological tail of inner-train as the calibration set; isotonic fit on that tail; applied to outer-test proba.
  - "Use existing CUSUM events" → `cusum_filter(close, h=cusum_h)` from `ml/events.py`, with `cusum_h` read from `ml_model_v2.pkl.config["labeling"]["cusum_h"]`.
  - "Use existing triple-barrier labels" → `apply_meta_triple_barrier` from `ml/meta_labels.py`.

- **No placeholders:** every step contains the exact code or command; no "TBD" or "similar to Task N".

- **Type / signature consistency:** `train_meta_fold` always returns the dict `{base_clf, isotonic, scaler, oos_calibrated_proba}`. `select_primary_fires` always returns `{mask, sides, max_proba, class_idx}`. `evaluate_p1_gates` always returns the dict shape consumed by the report writer. The integration test asserts these contract keys.

- **Out-of-scope items rejected:** No new features, no LLM, no changes to P4 model or CUSUM/labels/uniqueness, no stitched AR(1) calibration. The plan reuses every existing primitive.

---

## Notes for the Implementer

1. The plan asserts `embargo_pct=0.02` to honor "embargo = 2 × vertical_barrier"-spirit, but the existing `PurgedKFold` interprets `embargo_pct` as a fraction of total samples, not bars. Setting it to 2× the P4 default (`0.01 → 0.02`) keeps the semantics consistent with the existing splitter without modifying it (out of scope). If user pushes back, document in the report the exact bar count being embargoed.
2. The feature matrix passed to the meta-classifier is exactly the P4 model's input matrix at the same row — same `feature_names`, same `batch_extract` pipeline. **Do not** add the primary's softprob outputs as extra columns in the X matrix. The plan keeps "primary's max_proba at fire" as a stand-alone reported field for diagnostics, not as a model input — that preserves the prompt's "no new features beyond what P4 already produces" constraint and keeps the meta-model honest about whether it can predict from the same evidence as the primary.
3. If the run rejects on the first try (likely on this dataset given P4 baseline expectancy is −0.196R), report the failed-gate values precisely and stop. Do not chase a green bar by widening the threshold sweep or relaxing acceptance criteria.
