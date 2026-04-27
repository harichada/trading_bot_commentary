"""P3 — regime-conditional strategy routing on the P4 nested-CV primary.

Runs the AFML §7.4 nested-CV harness to refit the P4 primary per fold,
classifies the regime at each test event using ml/regime.py (7-state,
median-split, no-look-ahead), debounces with RegimeStabilityGate, and
buckets trades by their committed regime label. Reports per-regime
PF / expectancy / win-rate / n_trades / per-fold AR(1) / cost_drag and
an aggregate-across-regimes block.

Hard invariants (encoded in tests/test_p3_runner_smoke.py and
tests/test_p3_acceptance_gates.py):

  * ml_model_v2.pkl is NEVER loaded — we only use the nested-CV harness's
    OOS softprob. Direct predictions from the released bundle are
    in-sample on its training window. Set env P3_FORBID_PKL_LOAD=1 to
    enforce at runtime.
  * Chop regime trade count is exactly 0 across all folds (gate C).
  * 7 regime states reported, not 4. Always all 7, even if some are
    empty.

CLI mirrors the P4 runner where possible (--symbols, --days, --folds,
etc.) plus P3-specific flags (--regime-config, --stability-n).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_p3_regime")

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REGIME_CONFIG = REPO_ROOT / "regime_configs" / "p3.yaml"
DEFAULT_REPORT = REPO_ROOT / "backtest_results" / "P3_after.json"
DEFAULT_PRIMARY_BUNDLE = REPO_ROOT / "ml_model_v2.pkl"


def _enforce_no_pkl_load() -> None:
    """Optional safety: when env P3_FORBID_PKL_LOAD=1, monkey-patch
    joblib.load to raise on any attempt to read the released bundle.
    Used by tests to confirm the runner relies on the nested-CV harness
    rather than the released-bundle in-sample predictions. Alternative
    bundles (retune experiments) bypass this guard so config-only loads
    still work."""
    if os.environ.get("P3_FORBID_PKL_LOAD") != "1":
        return
    import joblib
    _orig_load = joblib.load

    def guarded(filename, *args, **kwargs):
        path_str = str(filename)
        if path_str.endswith("/ml_model_v2.pkl") or path_str == "ml_model_v2.pkl":
            sys.stderr.write(
                f"P3_FORBID_PKL_LOAD violation: refused to load {path_str}\n")
            sys.stderr.flush()
            raise RuntimeError(
                f"P3_FORBID_PKL_LOAD violation: refused to load {path_str}")
        return _orig_load(filename, *args, **kwargs)

    joblib.load = guarded  # type: ignore[assignment]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_primary_config(bundle_path: Path) -> dict:
    """Source primary_config from the bundle's .config dict — the same
    recipe the bundle was trained with. We DO NOT use the model's
    predictions (those would be in-sample on its training window); the
    nested-CV harness refits per fold. Reading the .config dict is
    the documented harness contract from P1.

    Set ``bundle_path`` to the released ``ml_model_v2.pkl`` for the
    standard run; point it at an alternative bundle (e.g. a barrier-
    retune experiment) to evaluate that recipe without overwriting the
    released artefact."""
    import joblib
    bundle = joblib.load(bundle_path)
    if isinstance(bundle, dict):
        cfg = bundle.get("config")
    else:
        cfg = getattr(bundle, "config", None)
    if cfg is None:
        raise RuntimeError(
            f"{bundle_path} has no .config attribute / 'config' key — "
            "harness cannot proceed without a verified primary recipe")
    return cfg


def _build_per_symbol_regime_track(
    *,
    prices_per_symbol: dict[str, pd.DataFrame],
    stability_n: int,
    thresholds: Any,
) -> dict[str, pd.Series]:
    """For each symbol, run compute_regime_features + classify_regime +
    RegimeStabilityGate over the entire bar timeline (chronological).
    Returns {symbol: Series indexed by bar timestamp with committed regime label}.

    Per-symbol gate state is appropriate because: a symbol's regime is
    independent of other symbols (an AAPL chop bar doesn't influence
    NVDA). Bot in production runs per-symbol regime tracking too.
    """
    from ml.regime import (
        RegimeStabilityGate, classify_regime, compute_regime_features,
    )
    out: dict[str, pd.Series] = {}
    for sym, bars in prices_per_symbol.items():
        if bars.empty:
            out[sym] = pd.Series(dtype=object)
            continue
        feats = compute_regime_features(bars, thresholds=thresholds)
        gate = RegimeStabilityGate(n=stability_n)
        committed = []
        for ts, row in feats.iterrows():
            raw = classify_regime(row, thresholds=thresholds)
            committed.append(gate.step(raw))
        out[sym] = pd.Series(committed, index=feats.index, name="committed_regime")
    return out


def _decide_trade(
    *,
    softprob_row: np.ndarray,
    decision_threshold: float,
) -> tuple[bool, int]:
    """Mirror of train_ml_model_v2._fold_net_r_multiples decision logic.
    Returns (do_trade, predicted_class). Trade only when max-class proba
    >= threshold AND class is BUY (2) or SELL (0)."""
    y_pred = int(np.argmax(softprob_row))
    p_max = float(softprob_row[y_pred])
    if p_max < decision_threshold:
        return False, y_pred
    if y_pred == 1:  # HOLD
        return False, y_pred
    return True, y_pred


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True,
                        help="Universe (e.g. NVDA TSLA AAPL ...)")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--frequency", type=int, default=5,
                        help="Bar frequency in minutes")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--embargo-pct", type=float, default=0.01)
    parser.add_argument("--bootstrap-iterations", type=int, default=5)
    parser.add_argument("--regime-config", default=str(DEFAULT_REGIME_CONFIG))
    parser.add_argument("--stability-n", type=int, default=None,
                        help="Override stability_n from regime config")
    parser.add_argument("--decision-threshold", type=float, default=None,
                        help="Override decision_threshold from regime config")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    parser.add_argument(
        "--primary-bundle-path", default=str(DEFAULT_PRIMARY_BUNDLE),
        help="Path to a primary bundle (.pkl with .config). Defaults to the "
             "released ml_model_v2.pkl. Point at an alternative bundle "
             "(e.g., barrier-retune experiment) to evaluate that recipe "
             "without touching the released one.",
    )
    parser.add_argument("--dsn", default=os.environ.get(
        "POSTGRES_DSN",
        "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev"))
    args = parser.parse_args()

    _enforce_no_pkl_load()
    started = time.time()
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)

    # Load primary config (recipe only — pkl bytes are untouched).
    primary_bundle_path = Path(args.primary_bundle_path)
    if not primary_bundle_path.exists():
        raise FileNotFoundError(
            f"primary bundle not found: {primary_bundle_path}")
    pkl_sha_before = _sha256(primary_bundle_path)
    primary_config = _load_primary_config(primary_bundle_path)

    # Load regime config.
    regime_cfg = _load_yaml(Path(args.regime_config))
    stability_n = args.stability_n or regime_cfg.get("stability_n", 5)
    decision_threshold = (args.decision_threshold
                          or regime_cfg.get("decision_threshold", 0.55))
    per_regime_cfg = regime_cfg.get("regimes", {})

    logger.info(
        "p3_start symbols=%d days=%d freq=%d folds=%d stability_n=%d "
        "decision_threshold=%.2f",
        len(args.symbols), args.days, args.frequency, args.folds,
        stability_n, decision_threshold,
    )

    # Reuse the P4 build_event_view (Step 1 refactor) — same labels,
    # same uniqueness, same cost-aware weights, same chronological order.
    from ml.event_view import build_event_view
    from ml.purged_cv import PurgedKFold
    from ml.iid_diagnostics import weighted_residuals
    from ml.primary_oos_harness import refit_primary_on_fold
    from ml.regime import (
        RegimeThresholds, SEVEN_STATES,
    )
    from ml.regime_metrics import PerRegimeAggregator

    pt_mult = primary_config["pt_mult"]
    sl_mult = primary_config["sl_mult"]
    max_holding = primary_config["max_holding"]
    cost_model_cfg = primary_config["cost_model"]
    labeling_cfg = primary_config["labeling"]
    cusum_h = labeling_cfg["cusum_h"]

    ev = build_event_view(
        symbols=[s.upper() for s in args.symbols],
        days=args.days, frequency=args.frequency, dsn=args.dsn,
        pt_mult=pt_mult, sl_mult=sl_mult,
        max_holding=max_holding, event_stride=5,
        cost_aware=True,
        fee_bps=cost_model_cfg["fee_bps"],
        slip_bps=cost_model_cfg["slip_bps"],
        impact_coef=cost_model_cfg["impact_coef"],
        participation=cost_model_cfg["participation"],
        event_source="cusum", cusum_h=cusum_h,
        vertical_mult_duration=2.0,
        min_ret_atr_mult=labeling_cfg.get("min_ret_atr_mult", 0.5),
        collect_prices=True,
    )
    if len(ev.X) == 0:
        logger.error("no events collected — aborting")
        return 1

    # Per-symbol committed-regime track over the full bar timeline.
    # Thresholds come from pro_trading_config.yaml::regime_detector if
    # available, else defaults.
    pro_cfg_path = REPO_ROOT / "pro_trading_config.yaml"
    th_kwargs = {}
    if pro_cfg_path.exists():
        pro_cfg = _load_yaml(pro_cfg_path)
        rd = pro_cfg.get("regime_detector", {}) or {}
        if "ema_fast" in rd:
            th_kwargs["ema_fast"] = int(rd["ema_fast"])
        if "ema_slow" in rd:
            th_kwargs["ema_slow"] = int(rd["ema_slow"])
        if "rv_period" in rd:
            th_kwargs["rv_period"] = int(rd["rv_period"])
        if "rv_median_lookback" in rd:
            th_kwargs["rv_median_lookback"] = int(rd["rv_median_lookback"])
        if "adx_trending_threshold" in rd:
            th_kwargs["adx_trending"] = float(rd["adx_trending_threshold"])
        if "chop_index_threshold" in rd:
            th_kwargs["chop_index"] = float(rd["chop_index_threshold"])
    thresholds = RegimeThresholds(**th_kwargs)

    logger.info("computing_regime_track per_symbol_n=%d stability_n=%d",
                len(ev.prices_per_symbol or {}), stability_n)
    regime_tracks = _build_per_symbol_regime_track(
        prices_per_symbol=ev.prices_per_symbol or {},
        stability_n=stability_n,
        thresholds=thresholds,
    )

    # Per-symbol regime distributions (for the Step 6 diagnostic).
    bars_per_regime: dict[str, int] = {r: 0 for r in SEVEN_STATES}
    transition_count_total = 0
    for sym, track in regime_tracks.items():
        if track.empty:
            continue
        vc = track.value_counts()
        for r, n in vc.items():
            if r in bars_per_regime:
                bars_per_regime[r] += int(n)
        # Approximate transitions: count where label changes between
        # consecutive bars.
        transitions = (track != track.shift(1)).sum() - 1  # -1 for the first bar
        transition_count_total += max(0, int(transitions))

    total_bars = sum(bars_per_regime.values())
    pct_bars_classified_chop = (
        bars_per_regime["chop"] / total_bars if total_bars else 0.0
    )
    pct_committed_per_regime = {
        r: (n / total_bars if total_bars else 0.0)
        for r, n in bars_per_regime.items()
    }

    # Per-fold loop with the nested-CV harness.
    touch_series = pd.Series(ev.touch_times.to_numpy(), index=ev.event_times)
    cv = PurgedKFold(
        n_splits=args.folds, touch_times=touch_series,
        embargo_pct=args.embargo_pct,
    )
    agg = PerRegimeAggregator(regimes=SEVEN_STATES)

    n_events_seen_per_fold: list[int] = []
    n_events_skipped_chop_per_fold: list[int] = []
    # Diagnostics for the retune experiment: per-regime gross R, per-trade
    # horizon in bars, per-event cost in R.
    gross_r_per_regime: dict[str, list[float]] = {r: [] for r in (
        "trend_up_low_vol", "trend_up_high_vol",
        "trend_dn_low_vol", "trend_dn_high_vol",
        "range_tight", "range_wide", "chop",
    )}
    trade_horizon_bars: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(ev.X), 1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            logger.warning(
                "fold %d empty (train=%d test=%d) — skipping",
                fold_idx, len(train_idx), len(test_idx),
            )
            continue

        logger.info("fold %d/%d train=%d test=%d — refitting primary",
                    fold_idx, args.folds, len(train_idx), len(test_idx))

        proba_test = refit_primary_on_fold(
            X_train=ev.X[train_idx], y_train=ev.y[train_idx],
            weights_train=ev.weights[train_idx],
            X_test=ev.X[test_idx],
            event_times_train=ev.event_times[train_idx],
            touch_times_train=ev.touch_times[train_idx],
            primary_config=primary_config,
            sample_r_multiples_train=ev.sample_r_multiples[train_idx],
            sample_cost_r_train=ev.sample_cost_r[train_idx],
            bootstrap_iterations=args.bootstrap_iterations,
            use_sequential_bootstrap=True,
            seed=42 + fold_idx,
        )

        # Walk test events in chronological order; route via regime gate;
        # bucket trades.
        regime_labels: list[str] = []
        gross_r_list: list[float] = []
        net_r_list: list[float] = []
        residuals_list: list[float] = []

        n_skipped_chop = 0
        for i, ev_idx in enumerate(test_idx):
            event_time = ev.event_times[ev_idx]
            sym = ev.sym_groups[ev_idx]
            track = regime_tracks.get(sym)
            if track is None or track.empty:
                committed = "chop"
            else:
                # Look up committed regime at this event's bar timestamp.
                # If event_time is between bars or missing, fall back to
                # the last committed regime at-or-before that timestamp.
                try:
                    committed = track.loc[event_time]
                except KeyError:
                    pos = track.index.searchsorted(event_time, side="right") - 1
                    committed = track.iloc[max(0, pos)] if len(track) else "chop"

            allow = per_regime_cfg.get(committed, {}).get("allow_trade", True)
            per_regime_thresh = per_regime_cfg.get(committed, {}).get(
                "decision_threshold", decision_threshold,
            )

            do_trade, y_pred = _decide_trade(
                softprob_row=proba_test[i],
                decision_threshold=per_regime_thresh,
            )

            # Hard rule: forbid trading in chop. This is enforced by
            # both the per-regime YAML allow_trade=false AND this
            # explicit guard, so the gate-C invariant cannot be relaxed
            # by editing YAML alone.
            if committed == "chop":
                allow = False

            if not (do_trade and allow):
                if committed == "chop":
                    n_skipped_chop += 1
                continue

            side = 1.0 if y_pred == 2 else -1.0
            gross_r = side * float(ev.sample_r_multiples[ev_idx])
            net_r = gross_r - float(ev.sample_cost_r[ev_idx])

            # Per-trade weighted residual: the realized class probability
            # under the OOS softprob, weighted by the event's sample weight.
            p_realized = float(proba_test[i, ev.y[ev_idx]])
            w_event = float(ev.weights[ev_idx])
            resid = weighted_residuals(
                np.array([1.0]),
                np.array([p_realized]),
                np.array([w_event]),
            )[0]

            regime_labels.append(committed)
            gross_r_list.append(gross_r)
            net_r_list.append(net_r)
            residuals_list.append(float(resid))
            gross_r_per_regime[committed].append(gross_r)
            # Per-trade horizon in bars: (touch - event) / freq_minutes.
            try:
                hz_min = (
                    pd.Timestamp(ev.touch_times[ev_idx])
                    - pd.Timestamp(event_time)
                ).total_seconds() / 60.0
                trade_horizon_bars.append(hz_min / max(1, args.frequency))
            except Exception:
                pass

        agg.add_fold(
            fold_idx=fold_idx,
            regime_labels=np.array(regime_labels, dtype=object),
            gross_r=np.array(gross_r_list, dtype="float64"),
            net_r=np.array(net_r_list, dtype="float64"),
            residuals=np.array(residuals_list, dtype="float64"),
        )
        n_events_seen_per_fold.append(int(len(test_idx)))
        n_events_skipped_chop_per_fold.append(int(n_skipped_chop))

        logger.info(
            "fold %d done: %d trades across %d test events; %d skipped (chop)",
            fold_idx, len(net_r_list), len(test_idx), n_skipped_chop,
        )

    summary = agg.summarize()
    per_regime_metrics = {r: summary[r] for r in SEVEN_STATES}
    aggregate_block = summary["__aggregate__"]

    # Acceptance gate evaluations (informational — full assertion lives in
    # tests/test_p3_acceptance_gates.py against the full-universe report).
    gate_a_passing_regimes = []
    for r in SEVEN_STATES:
        if r == "chop":
            continue
        m = per_regime_metrics[r]
        pf = m.get("median_pf")
        exp_r = m.get("median_expectancy_r")
        n = m.get("total_n_trades", 0)
        if pf is not None and exp_r is not None and pf >= 1.3 and exp_r > 0 and n >= 100:
            gate_a_passing_regimes.append(r)

    chop_n = per_regime_metrics["chop"]["total_n_trades"]
    gate_c_pass = chop_n == 0

    p4_pf_baseline = 0.738
    p4_exp_baseline = -0.196
    agg_pf = aggregate_block.get("median_pf")
    agg_exp = aggregate_block.get("median_expectancy_r")
    gate_d_pass = (
        agg_pf is not None and agg_exp is not None
        and agg_pf >= p4_pf_baseline and agg_exp >= p4_exp_baseline
    )

    pkl_sha_after = _sha256(primary_bundle_path)
    elapsed = time.time() - started

    report = {
        "timestamp": datetime.now().isoformat(),
        "primary_bundle": str(primary_bundle_path),
        "primary_bundle_sha256": {
            "before": pkl_sha_before, "after": pkl_sha_after,
            "unchanged": pkl_sha_before == pkl_sha_after,
        },
        "config": {
            "symbols": [s.upper() for s in args.symbols],
            "days": args.days, "frequency": args.frequency,
            "folds": args.folds, "embargo_pct": args.embargo_pct,
            "bootstrap_iterations": args.bootstrap_iterations,
            "stability_n": stability_n,
            "decision_threshold": decision_threshold,
            "regime_config_path": str(args.regime_config),
            "primary_config": primary_config,
            "regime_thresholds": {
                "adx_trending": thresholds.adx_trending,
                "chop_index": thresholds.chop_index,
                "ema_fast": thresholds.ema_fast,
                "ema_slow": thresholds.ema_slow,
                "rv_period": thresholds.rv_period,
                "rv_median_lookback": thresholds.rv_median_lookback,
            },
            "spec_reference": "PROMPT_PACK §P3 + user-pinned median splits",
        },
        "symbols": [s.upper() for s in args.symbols],
        "n_events": int(len(ev.X)),
        "per_symbol_events": ev.per_symbol_samples,
        "n_events_per_fold": n_events_seen_per_fold,
        "n_events_skipped_chop_per_fold": n_events_skipped_chop_per_fold,
        "regime_distribution": {
            "bars_per_regime": bars_per_regime,
            "total_bars": total_bars,
            "pct_bars_classified_chop": pct_bars_classified_chop,
            "pct_committed_per_regime": pct_committed_per_regime,
        },
        "regime_transition_count": transition_count_total,
        "per_regime_metrics": per_regime_metrics,
        "aggregate_across_regimes": aggregate_block,
        "gate_results": {
            "gate_a_at_least_one_regime_passes": len(gate_a_passing_regimes) > 0,
            "gate_a_passing_regimes": gate_a_passing_regimes,
            "gate_c_chop_zero_trades": gate_c_pass,
            "gate_d_aggregate_geq_p4_baseline": gate_d_pass,
            "p4_baseline": {
                "median_pf": p4_pf_baseline,
                "median_expectancy_r": p4_exp_baseline,
            },
            "thresholds": {
                "gate_a_min_pf": 1.3,
                "gate_a_min_expectancy_r": 0.0,
                "gate_a_min_trades": 100,
                "gate_b_ar1_median_lt": 0.10,
                "gate_b_ar1_max_lt": 0.20,
            },
        },
        "timing_seconds": {
            "total": round(elapsed, 1),
            "collect": round(ev.timing.get("collect", 0.0), 1),
            "uniqueness": round(ev.timing.get("uniqueness", 0.0), 1),
        },
        "diagnostics": {
            "median_sample_cost_r": float(np.median(ev.sample_cost_r))
            if len(ev.sample_cost_r) else None,
            "mean_sample_cost_r": float(np.mean(ev.sample_cost_r))
            if len(ev.sample_cost_r) else None,
            "median_trade_horizon_bars": (
                float(np.median(trade_horizon_bars))
                if trade_horizon_bars else None
            ),
            "median_gross_r_per_regime": {
                r: (float(np.median(g)) if g else None)
                for r, g in gross_r_per_regime.items()
            },
            "n_trades_per_regime_total": {
                r: len(g) for r, g in gross_r_per_regime.items()
            },
        },
    }

    Path(args.report_path).write_text(json.dumps(report, indent=2, default=str))
    logger.info(
        "p3_done passing_regimes=%s chop_trades=%d agg_pf=%s agg_exp=%s "
        "elapsed=%.1fs report=%s",
        gate_a_passing_regimes, chop_n,
        f"{agg_pf:.3f}" if agg_pf is not None else "None",
        f"{agg_exp:.3f}" if agg_exp is not None else "None",
        elapsed, args.report_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
