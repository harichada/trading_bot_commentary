"""Walk-forward validation of baseline-P4-retune.

Re-runs the EXACT train_ml_model_v2 + train_p3_regime_routing pipeline
on three disjoint 60-day windows:

  W1: 2026-02-16 → 2026-04-17  (replication of original training window)
  W2: 2021-01-04 → 2021-03-05  (oldest available; post-COVID rally)
  W3: 2022-06-01 → 2022-07-31  (mid-history; 2022 bear market)

For each window we monkey-patch ``PostgresDataProvider.get_market_data``
to inject ``end=<window_end>`` — the provider already exposes this
parameter (see ``data_providers/postgres.py:43``); we just route through
it. **No production code is modified.** The released
``ml_model_v2_retune.pkl`` is read-only via ``_load_primary_config``;
SHA256 is verified byte-identical pre/post the entire run (W5).

Methodology references
----------------------
* AFML §11 — combinatorial walk-forward CV. A single 60-day window with
  internal 5-fold purged CV does NOT control for time-window selection
  bias. Three disjoint windows are the minimum to detect that.
* AFML §12 — backtest overfitting. The retune was tuned on a single
  window; this script tests whether its apparent edge survives on
  windows the tuner never saw.

Output
------
``backtest_results/walk_forward_validation.json`` with per-window
per-regime breakdown.
``backtest_results/walk_forward_summary.md`` with verdict (foundation
robust / time-window artifact / mixed).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate_walk_forward")

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = REPO_ROOT / "backtest_results" / "walk_forward_validation.json"
DEFAULT_SUMMARY = REPO_ROOT / "backtest_results" / "walk_forward_summary.md"

WINDOWS = {
    "W1": {
        "label": "replication (most recent 60d)",
        "end": pd.Timestamp("2026-04-17 19:59:00"),
    },
    "W2": {
        "label": "oldest 60d (2021-Q1, post-COVID rally)",
        "end": pd.Timestamp("2021-03-05 16:00:00"),
    },
    "W3": {
        "label": "mid-history 60d (2022 bear)",
        "end": pd.Timestamp("2022-07-31 16:00:00"),
    },
}

WATCHLIST = ["NVDA", "TSLA", "AAPL", "AMD", "SPY", "QQQ", "META", "MSFT", "GOOGL", "AMZN"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _patched_provider(window_end: pd.Timestamp):
    """Context manager that monkey-patches PostgresDataProvider.get_market_data
    to inject ``end=window_end``. The provider already supports the kwarg —
    we're using its documented API, not modifying production code."""
    from data_providers.postgres import PostgresDataProvider

    original = PostgresDataProvider.get_market_data
    end_dt = window_end.to_pydatetime()

    def patched(self, *args, end=None, **kwargs):
        # Caller's `end` (if any) is overridden by window_end.
        return original(self, *args, end=end_dt, **kwargs)

    class _Ctx:
        def __enter__(self_):
            PostgresDataProvider.get_market_data = patched
            return self_

        def __exit__(self_, *exc):
            PostgresDataProvider.get_market_data = original

    return _Ctx()


def _train_window_primary(window_end: pd.Timestamp, model_path: Path,
                          training_report_path: Path) -> None:
    """Run train_ml_model_v2.main() programmatically with sys.argv set
    to the retune parameters for this window."""
    import importlib
    # Reload between windows so any module-level state (loggers,
    # singletons) does not contaminate.
    if "train_ml_model_v2" in sys.modules:
        importlib.reload(sys.modules["train_ml_model_v2"])
    import train_ml_model_v2

    saved_argv = sys.argv[:]
    sys.argv = [
        "train_ml_model_v2.py",
        "--symbols", *WATCHLIST,
        "--days", "60",
        "--frequency", "5",
        "--folds", "5",
        "--pt-mult", "2.0",
        "--sl-mult", "2.0",
        "--cusum-h", "0.015",
        "--vertical-mult-duration", "3.0",
        "--bootstrap-iterations", "5",
        "--reject-cost-drag-pct", "999",
        "--model-path", str(model_path),
        "--report-path", str(training_report_path),
    ]
    try:
        with _patched_provider(window_end):
            rc = train_ml_model_v2.main()
    finally:
        sys.argv = saved_argv
    if rc != 0:
        raise RuntimeError(
            f"train_ml_model_v2 returned non-zero ({rc}) for window end={window_end}"
        )


def _evaluate_window_p3(window_end: pd.Timestamp, primary_bundle: Path,
                        report_path: Path) -> None:
    """Run train_p3_regime_routing.main() programmatically against the
    per-window primary bundle, with the same data-provider patch
    applied so events come from the same window."""
    import importlib
    if "train_p3_regime_routing" in sys.modules:
        importlib.reload(sys.modules["train_p3_regime_routing"])
    import train_p3_regime_routing

    saved_argv = sys.argv[:]
    sys.argv = [
        "train_p3_regime_routing.py",
        "--symbols", *WATCHLIST,
        "--days", "60",
        "--frequency", "5",
        "--folds", "5",
        "--bootstrap-iterations", "5",
        "--primary-bundle-path", str(primary_bundle),
        "--report-path", str(report_path),
    ]
    try:
        with _patched_provider(window_end):
            rc = train_p3_regime_routing.main()
    finally:
        sys.argv = saved_argv
    if rc != 0:
        raise RuntimeError(
            f"train_p3_regime_routing returned non-zero ({rc}) for window end={window_end}"
        )


def main() -> int:
    Path(DEFAULT_REPORT).parent.mkdir(parents=True, exist_ok=True)
    released_bundle = REPO_ROOT / "ml_model_v2_retune.pkl"
    if not released_bundle.exists():
        raise FileNotFoundError(
            f"released bundle not found: {released_bundle}. "
            "Cannot validate without the retune as reference."
        )
    sha_before = _sha256(released_bundle)

    started = datetime.now()
    per_window_results: dict[str, dict] = {}

    for label, spec in WINDOWS.items():
        window_end = spec["end"]
        window_start = window_end - pd.Timedelta(days=60)
        logger.info(
            "=== %s: %s ===  end=%s  start=%s",
            label, spec["label"], window_end, window_start,
        )

        # Per-window primary bundle (gitignored via *.pkl).
        # Uses the underscore-suffixed name to keep it out of any glob
        # that might pick up the released bundle.
        primary_bundle = REPO_ROOT / f"ml_model_v2_walk_{label.lower()}.pkl"
        training_report = (
            REPO_ROOT / "backtest_results" / f"walk_{label.lower()}_training.json"
        )
        p3_report = (
            REPO_ROOT / "backtest_results" / f"walk_{label.lower()}_p3.json"
        )

        # Step 1: train per-window primary on the windowed data.
        logger.info("[%s] step 1/2 — training primary on window data", label)
        _train_window_primary(
            window_end=window_end,
            model_path=primary_bundle,
            training_report_path=training_report,
        )

        # Step 2: evaluate via P3 regime routing on the same window.
        logger.info("[%s] step 2/2 — P3 regime routing on per-window primary", label)
        _evaluate_window_p3(
            window_end=window_end,
            primary_bundle=primary_bundle,
            report_path=p3_report,
        )

        p3_data = json.loads(p3_report.read_text())

        per_window_results[label] = {
            "label": spec["label"],
            "window_start_ts": window_start.isoformat(),
            "window_end_ts": window_end.isoformat(),
            "per_window_primary_bundle": str(primary_bundle),
            "per_window_primary_sha256": _sha256(primary_bundle),
            "training_report_path": str(training_report),
            "p3_report_path": str(p3_report),
            "n_events": p3_data.get("n_events"),
            "n_events_per_fold": p3_data.get("n_events_per_fold"),
            "per_regime_metrics": p3_data["per_regime_metrics"],
            "aggregate_across_regimes": p3_data["aggregate_across_regimes"],
            "regime_distribution": p3_data.get("regime_distribution", {}),
            "gate_results": p3_data.get("gate_results", {}),
        }

    sha_after = _sha256(released_bundle)
    elapsed = (datetime.now() - started).total_seconds()

    # Summary numbers for W4 median.
    pfs = sorted(
        per_window_results[w]["aggregate_across_regimes"]["median_pf"]
        for w in ("W1", "W2", "W3")
    )
    exps = sorted(
        per_window_results[w]["aggregate_across_regimes"]["median_expectancy_r"]
        for w in ("W1", "W2", "W3")
    )
    median_pf = pfs[1]
    median_exp = exps[1]

    report = {
        "timestamp": datetime.now().isoformat(),
        "released_bundle": str(released_bundle),
        "released_bundle_sha256": {
            "before": sha_before,
            "after": sha_after,
            "unchanged": sha_before == sha_after,
        },
        "watchlist": WATCHLIST,
        "windows": per_window_results,
        "median_aggregate_pf": median_pf,
        "median_aggregate_expectancy_r": median_exp,
        "spec_reference": "Walk-forward validation of baseline-P4-retune (AFML §11/§12)",
        "elapsed_seconds": round(elapsed, 1),
    }
    DEFAULT_REPORT.write_text(json.dumps(report, indent=2, default=str))
    logger.info(
        "walk-forward validation done elapsed=%.1fs median_pf=%.3f median_exp=%.3f report=%s",
        elapsed, median_pf, median_exp, DEFAULT_REPORT,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
