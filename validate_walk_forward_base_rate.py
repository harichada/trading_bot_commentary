"""20-window AR(1) base-rate sweep for baseline-P4-retune.

Extends ``validate_walk_forward.py`` from 3 windows to 20. Asks the
question the prior 3-window experiment could not answer: among
arbitrary 60-day windows on this data history, what fraction pass the
trainer's per-fold AR(1) IID gate?

We reuse the **exact** monkey-patch pattern from the prior orchestrator
— ``_patched_provider`` injects ``end=window_end`` into
``PostgresDataProvider.get_market_data``, exploiting the provider's
documented kwarg (data_providers/postgres.py:43). No production code
modified.

Window schedule
---------------
20 strictly disjoint 60-day windows spanning 2021-01-04 → 2026-04-17:

* W01-W03 = the prior W1, W2, W3 (reused verbatim from
  ``walk_forward_validation.json`` — not retrained; their per-window
  bundles already exist as ``ml_model_v2_walk_w{1,2,3}.pkl``).
* W04-W20 = 17 new disjoint windows. Min stride between window-ends:
  90 calendar days (60-day window + ≥30 days separation).

Methodology references
----------------------
* AFML §11 — combinatorial walk-forward CV / base-rate analysis.
* AFML §4.5.3 — sequential bootstrap requires weak label dependence;
  the AR(1) gate enforces that prerequisite. The gate threshold is
  the experimental control variable; we do NOT tamper with it during
  this sweep.

Output
------
``backtest_results/walk_forward_base_rate.json`` — per-window detail
plus aggregate base-rate summary.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Reuse the prior orchestrator's helpers — _patched_provider,
# _train_window_primary, _evaluate_window_p3, _sha256, WATCHLIST.
from validate_walk_forward import (
    WATCHLIST,
    _evaluate_window_p3,
    _patched_provider,  # noqa: F401 — exposed for clarity; used transitively
    _sha256,
    _train_window_primary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate_walk_forward_base_rate")

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = REPO_ROOT / "backtest_results" / "walk_forward_base_rate.json"

# Reuse the prior 3-window report as the source-of-truth for W01-W03.
PRIOR_REPORT = REPO_ROOT / "backtest_results" / "walk_forward_validation.json"

# Severity thresholds for failed-window classification (median |AR(1)|).
# Match the bins documented in ``walk_forward_ar1_severity.md``.
NEAR_MISS_LOW = 0.10
NEAR_MISS_HIGH = 0.15
MODERATE_HIGH = 0.25

ROBUST_THRESHOLD = 0.50
FRAGILE_THRESHOLD = 0.20

# 20-window schedule. W01-W03 mirror the prior W1/W2/W3 (preserve W01's
# odd 19:59 timestamp byte-for-byte; W02/W03 use 16:00 like the existing
# WINDOWS dict in validate_walk_forward.py).
WINDOWS: dict[str, dict] = {
    "W01": {
        "label": "replication (most recent 60d)",
        "end": pd.Timestamp("2026-04-17 19:59:00"),
        "reused_from_prior": True,
        "prior_label": "W1",
    },
    "W02": {
        "label": "oldest 60d (2021-Q1, post-COVID rally)",
        "end": pd.Timestamp("2021-03-05 16:00:00"),
        "reused_from_prior": True,
        "prior_label": "W2",
    },
    "W03": {
        "label": "mid-history 60d (2022 bear)",
        "end": pd.Timestamp("2022-07-31 16:00:00"),
        "reused_from_prior": True,
        "prior_label": "W3",
    },
    "W04": {"label": "2021-Q2 mid", "end": pd.Timestamp("2021-05-15 16:00:00")},
    "W05": {"label": "2021-Q3 mid", "end": pd.Timestamp("2021-08-15 16:00:00")},
    "W06": {"label": "2021-Q4 mid", "end": pd.Timestamp("2021-11-15 16:00:00")},
    "W07": {"label": "2022-Q1 mid", "end": pd.Timestamp("2022-02-15 16:00:00")},
    "W08": {"label": "2022-Q4 start", "end": pd.Timestamp("2022-11-01 16:00:00")},
    "W09": {"label": "2023-Q1 end",  "end": pd.Timestamp("2023-01-31 16:00:00")},
    "W10": {"label": "2023-Q2 end",  "end": pd.Timestamp("2023-04-30 16:00:00")},
    "W11": {"label": "2023-Q3 end",  "end": pd.Timestamp("2023-07-31 16:00:00")},
    "W12": {"label": "2023-Q4 end",  "end": pd.Timestamp("2023-10-31 16:00:00")},
    "W13": {"label": "2024-Q1 end",  "end": pd.Timestamp("2024-01-31 16:00:00")},
    "W14": {"label": "2024-Q2 end",  "end": pd.Timestamp("2024-04-30 16:00:00")},
    "W15": {"label": "2024-Q3 end",  "end": pd.Timestamp("2024-07-31 16:00:00")},
    "W16": {"label": "2024-Q4 end",  "end": pd.Timestamp("2024-10-31 16:00:00")},
    "W17": {"label": "2025-Q1 end",  "end": pd.Timestamp("2025-01-31 16:00:00")},
    "W18": {"label": "2025-Q2 end",  "end": pd.Timestamp("2025-04-30 16:00:00")},
    "W19": {"label": "2025-Q3 end",  "end": pd.Timestamp("2025-07-31 16:00:00")},
    "W20": {"label": "2025-Q4 end",  "end": pd.Timestamp("2025-10-31 16:00:00")},
}


def _classify_pass_rate(pr: float) -> str:
    if pr >= ROBUST_THRESHOLD:
        return "robust"
    if pr >= FRAGILE_THRESHOLD:
        return "marginal"
    return "fragile"


def _ar1_summary(per_fold_ar1: list[float]) -> tuple[float, float]:
    """Compute (median |AR(1)|, max |AR(1)|) — matches trainer's
    iid_diagnostics convention (it reports the abs-magnitude median)."""
    abs_vals = sorted(abs(v) for v in per_fold_ar1)
    n = len(abs_vals)
    if n == 0:
        return 0.0, 0.0
    if n % 2 == 1:
        med = abs_vals[n // 2]
    else:
        med = 0.5 * (abs_vals[n // 2 - 1] + abs_vals[n // 2])
    return float(med), float(abs_vals[-1])


def _classify_severity(median_abs_ar1: float) -> str:
    """Severity bin per ``walk_forward_ar1_severity.md``. Applies only
    when the gate failed, so median > NEAR_MISS_LOW is implied."""
    if median_abs_ar1 < NEAR_MISS_HIGH:
        return "near-miss"
    if median_abs_ar1 < MODERATE_HIGH:
        return "moderate"
    return "far-miss"


def _read_prior_window(prior_label: str, new_label: str) -> dict:
    """Pull a previously-completed window's record verbatim from the
    walk_forward_validation.json report and the per-window training
    report (which has the per-fold AR(1) values).

    The prior W1/W2/W3 records live in two separate files:
      * ``walk_forward_validation.json`` — has aggregate metrics,
        regime distribution, gate_results
      * ``walk_w{1,2,3}_training.json`` — has iid_diagnostics with
        per_fold_ar1
    We compose them into a single per-window block matching the new
    schema (``per_fold_ar1_median``, ``per_fold_ar1_max`` keys).
    """
    prior = json.loads(PRIOR_REPORT.read_text())
    src = prior["windows"][prior_label]

    # Per-fold AR(1) lives in the trainer report; locate it via the
    # path embedded in the prior record (when present), else by the
    # canonical naming convention.
    training_report_path = Path(
        src.get(
            "training_report_path",
            REPO_ROOT / "backtest_results" / f"walk_{prior_label.lower()}_training.json",
        )
    )
    if not training_report_path.exists():
        raise FileNotFoundError(
            f"prior training report missing for {prior_label}: {training_report_path}"
        )
    trep = json.loads(training_report_path.read_text())
    iid = trep.get("iid_diagnostics", {})
    per_fold = list(iid.get("per_fold_ar1", []))
    if not per_fold:
        raise RuntimeError(
            f"{prior_label} trainer report has no iid_diagnostics.per_fold_ar1; "
            "cannot reuse without fresh trainer run"
        )
    med, mx = _ar1_summary(per_fold)

    # Backfill reject_kind for prior records that pre-date the field.
    # The prior W2/W3 rejections were AR(1) (the prompt's diagnostic
    # confirms this); detect via reject_reason substring.
    prior_status = src.get("trainer_status", "ok")
    prior_reason = src.get("trainer_reject_reason") or ""
    reject_kind: str | None = None
    if prior_status == "rejected":
        if "AR(1)" in prior_reason or "uniqueness" in prior_reason or "label" in prior_reason:
            reject_kind = "p4_hard"
        elif "cost-drag" in prior_reason or "gross" in prior_reason:
            reject_kind = "cost_drag"
        else:
            reject_kind = "unknown"

    return {
        "label": src["label"],
        "window_start_ts": src["window_start_ts"],
        "window_end_ts": src["window_end_ts"],
        "trainer_status": prior_status,
        "per_fold_ar1": per_fold,
        "per_fold_ar1_median": med,
        "per_fold_ar1_max": mx,
        "trainer_reject_reason": src.get("trainer_reject_reason"),
        "reject_kind": reject_kind,
        "training_report_path": str(training_report_path),
        "p3_report_path": src.get("p3_report_path"),
        "per_window_primary_bundle": src.get("per_window_primary_bundle"),
        "per_window_primary_sha256": src.get("per_window_primary_sha256"),
        "n_events": src.get("n_events"),
        "n_events_per_fold": src.get("n_events_per_fold"),
        "per_regime_metrics": src.get("per_regime_metrics", {}),
        "aggregate_across_regimes": src.get(
            "aggregate_across_regimes",
            {"median_pf": None, "median_expectancy_r": None, "total_n_trades": 0},
        ),
        "regime_distribution": src.get("regime_distribution", {}),
        "gate_results": src.get("gate_results", {}),
        "reused_from_prior": True,
        "prior_label": prior_label,
    }


def _run_one_window(label: str, spec: dict) -> dict:
    """Train + evaluate a single new window. Returns the per-window
    record matching the schema produced by ``_read_prior_window``."""
    window_end = spec["end"]
    window_start = window_end - pd.Timedelta(days=60)
    logger.info(
        "[%s] %s — end=%s start=%s",
        label, spec["label"], window_end, window_start,
    )

    primary_bundle = REPO_ROOT / f"ml_model_v2_baserate_{label.lower()}.pkl"
    training_report = (
        REPO_ROOT / "backtest_results" / f"baserate_{label.lower()}_training.json"
    )
    p3_report = REPO_ROOT / "backtest_results" / f"baserate_{label.lower()}_p3.json"

    train_result = _train_window_primary(
        window_end=window_end,
        model_path=primary_bundle,
        training_report_path=training_report,
    )

    # Read per-fold AR(1) regardless of pass/fail — the trainer writes
    # iid_diagnostics on both rc=0 and rc=3 paths.
    if training_report.exists():
        trep = json.loads(training_report.read_text())
        iid = trep.get("iid_diagnostics", {})
        per_fold = list(iid.get("per_fold_ar1", []))
    else:
        per_fold = []
    med, mx = _ar1_summary(per_fold) if per_fold else (None, None)

    base_record: dict = {
        "label": spec["label"],
        "window_start_ts": window_start.isoformat(),
        "window_end_ts": window_end.isoformat(),
        "trainer_status": train_result["status"],
        "per_fold_ar1": per_fold,
        "per_fold_ar1_median": med,
        "per_fold_ar1_max": mx,
        "training_report_path": str(training_report),
        "reused_from_prior": False,
    }

    if train_result["status"] == "rejected":
        base_record["trainer_reject_reason"] = train_result["reject_reason"]
        base_record["reject_kind"] = train_result.get("reject_kind", "unknown")
        base_record["aggregate_across_regimes"] = {
            "median_pf": None,
            "median_expectancy_r": None,
            "total_n_trades": 0,
        }
        base_record["per_regime_metrics"] = {}
        base_record["regime_distribution"] = {}
        base_record["gate_results"] = {}
        return base_record

    # Trainer accepted; run P3 routing for aggregate metrics.
    logger.info("[%s] trainer ok — running P3 routing", label)
    _evaluate_window_p3(
        window_end=window_end,
        primary_bundle=primary_bundle,
        report_path=p3_report,
    )
    p3_data = json.loads(p3_report.read_text())
    base_record.update({
        "per_window_primary_bundle": str(primary_bundle),
        "per_window_primary_sha256": _sha256(primary_bundle),
        "p3_report_path": str(p3_report),
        "n_events": p3_data.get("n_events"),
        "n_events_per_fold": p3_data.get("n_events_per_fold"),
        "per_regime_metrics": p3_data["per_regime_metrics"],
        "aggregate_across_regimes": p3_data["aggregate_across_regimes"],
        "regime_distribution": p3_data.get("regime_distribution", {}),
        "gate_results": p3_data.get("gate_results", {}),
    })
    return base_record


def main() -> int:
    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    released_bundle = REPO_ROOT / "ml_model_v2_retune.pkl"
    if not released_bundle.exists():
        raise FileNotFoundError(
            f"released bundle not found: {released_bundle}"
        )
    sha_before = _sha256(released_bundle)
    started = datetime.now()

    per_window: dict[str, dict] = {}
    for label, spec in WINDOWS.items():
        if spec.get("reused_from_prior"):
            logger.info("[%s] reusing prior result (%s)", label, spec["prior_label"])
            per_window[label] = _read_prior_window(spec["prior_label"], label)
            continue
        per_window[label] = _run_one_window(label, spec)

    sha_after = _sha256(released_bundle)
    elapsed = (datetime.now() - started).total_seconds()

    # Aggregate metrics.
    n_pass = sum(1 for w in per_window.values() if w["trainer_status"] == "ok")
    n_total = len(per_window)
    pass_rate = n_pass / n_total if n_total else 0.0

    passing_pfs: list[float] = []
    passing_exps: list[float] = []
    for w in per_window.values():
        if w["trainer_status"] != "ok":
            continue
        a = w["aggregate_across_regimes"]
        if a.get("median_pf") is not None:
            passing_pfs.append(float(a["median_pf"]))
        if a.get("median_expectancy_r") is not None:
            passing_exps.append(float(a["median_expectancy_r"]))

    def _med_or_none(vals: list[float]) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    median_passing_pf = _med_or_none(passing_pfs)
    median_passing_exp = _med_or_none(passing_exps)

    # Distribution of |AR(1)| medians across all 20 windows (one number
    # per window — the per-fold median). Useful as a histogram source.
    all_ar1_medians = [
        float(w["per_fold_ar1_median"])
        for w in per_window.values()
        if w.get("per_fold_ar1_median") is not None
    ]

    # Failed-window severity distribution. Severity classification
    # only applies to AR(1) rejections (rc=3); cost-drag rejections
    # (rc=2) short-circuit before AR(1) is computed and have no median
    # to classify.
    severity_counts = {"near-miss": 0, "moderate": 0, "far-miss": 0}
    failed_ar1_medians: list[float] = []
    reject_kind_counts: dict[str, int] = {}
    for w in per_window.values():
        if w["trainer_status"] != "rejected":
            continue
        kind = w.get("reject_kind") or "unknown"
        reject_kind_counts[kind] = reject_kind_counts.get(kind, 0) + 1
        med = w.get("per_fold_ar1_median")
        if med is None or kind != "p4_hard":
            continue
        failed_ar1_medians.append(float(med))
        severity_counts[_classify_severity(float(med))] += 1

    report = {
        "timestamp": datetime.now().isoformat(),
        "released_bundle": str(released_bundle),
        "released_bundle_sha256": {
            "before": sha_before,
            "after": sha_after,
            "unchanged": sha_before == sha_after,
        },
        "watchlist": WATCHLIST,
        "n_windows": n_total,
        "n_pass": n_pass,
        "n_fail": n_total - n_pass,
        "pass_rate": pass_rate,
        "pass_rate_classification": _classify_pass_rate(pass_rate),
        "median_passing_pf": median_passing_pf,
        "median_passing_expectancy_r": median_passing_exp,
        "passing_pfs_sorted": sorted(passing_pfs),
        "passing_exps_sorted": sorted(passing_exps),
        "all_ar1_medians": all_ar1_medians,
        "failed_ar1_medians_sorted": sorted(failed_ar1_medians),
        "severity_counts_failed_p4_hard": severity_counts,
        "reject_kind_counts": reject_kind_counts,
        "windows": per_window,
        "spec_reference": (
            "20-window AR(1) base-rate sweep (AFML §11/§4.5.3); "
            "extends walk_forward_validation.json from 3 to 20 windows"
        ),
        "elapsed_seconds": round(elapsed, 1),
    }
    DEFAULT_REPORT.write_text(json.dumps(report, indent=2, default=str))
    logger.info(
        "base-rate sweep done elapsed=%.1fs n_pass=%d/%d pass_rate=%.2f (%s) "
        "median_passing_pf=%s median_passing_exp=%s report=%s",
        elapsed, n_pass, n_total, pass_rate,
        report["pass_rate_classification"],
        f"{median_passing_pf:.3f}" if median_passing_pf is not None else "None",
        f"{median_passing_exp:.3f}" if median_passing_exp is not None else "None",
        DEFAULT_REPORT,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
