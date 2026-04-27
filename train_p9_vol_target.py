"""P9 — volatility-targeted position sizing on the 20-window sweep.

Wraps the base-rate sweep's per-window event evaluation with vol-target
sizing applied per event (López de Prado, AFML §10.1). Produces:

  * ``backtest_results/P9_20window.json``     — per-window detail
  * ``backtest_results/P9_vs_retune_20window_diff.md``  — diff vs baseline

Core invariants (encoded in tests/test_p9_acceptance_gates.py):
  P9.A  median realized portfolio vol within ±30% of 0.75% target
  P9.B  per-event applied_size / vol_parity_size ≤ 5 + ε (hard)
  P9.C  drawdown distribution improves vs fixed-fractional baseline
  P9.D  per-window median ΔR ≥ -0.01R (sizing should not destroy edge)
  P9.E  ml_model_v2_retune.pkl SHA256 byte-identical pre/post
  P9.F  per-fold AR(1) gate intact within surviving regimes

Module layout
-------------
``_apply_sizing_to_window(label, events, bars)``
    Pure event-walk with per-event vol-target sizing. Pluggable input
    (events + bars) — smoke-testable with synthetic data, no DB needed.

``_baseline_window_record(label, events, bars)``
    Same shape as P9 but with ``applied_size = 1.0`` for every traded
    event (fixed-fractional reference). Used to compute the diff.

``_run_window_events(label, window_end, primary_bundle, ...)``
    Reproduces the per-event walk from train_p3_regime_routing.py
    (lines 329-435) using the verbatim ``ml/primary_oos_harness.py`` +
    ``ml/regime.py`` modules. Returns the chronological events list
    consumed by ``_apply_sizing_to_window``. Called only from
    ``main()`` — the harness uses real data via the patched provider.

``main()``
    Orchestrates: read walk_forward_base_rate.json → iterate W01-W20 →
    run events per ok window → apply sizing + baseline → write reports.

Granularity bridge
------------------
Bars are 5-minute; vol-target spec is daily. We resample to daily-last-
close for vol math. The sizing primitive lives in ``risk/vol_target.py``;
this runner orchestrates it.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from risk.vol_target import (
    DEFAULT_CAP_MULT,
    DEFAULT_FLOOR_PERCENTILE,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_TARGET_DAILY_VOL,
    SizingDecision,
    daily_log_returns,
    realized_daily_vol_ewma,
    size_event,
    vol_floor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_p9_vol_target")

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = REPO_ROOT / "backtest_results" / "P9_20window.json"
DEFAULT_DIFF = REPO_ROOT / "backtest_results" / "P9_vs_retune_20window_diff.md"
BASE_RATE_REPORT = REPO_ROOT / "backtest_results" / "walk_forward_base_rate.json"


# ---------------------------------------------------------------------------
# SHA256
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Median / percentile helpers (avoid pulling sklearn just for these)
# ---------------------------------------------------------------------------

def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    rank = p * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return s[lo] + frac * (s[hi] - s[lo])


# ---------------------------------------------------------------------------
# Per-event sizing walks (pure — no DB, smoke-testable with synthetic input)
# ---------------------------------------------------------------------------

def _walk_events(
    *,
    label: str,
    events: list[dict],
    bars: dict[str, pd.DataFrame],
    sizer: Callable[[str, pd.Timestamp, pd.DataFrame, int], dict],
) -> list[dict]:
    """Generic chronological event-walk with a pluggable per-event sizer.

    ``sizer(symbol, entry_ts, sym_bars, n_active_at_entry)`` returns a dict:
        applied_size, vol_parity_size, pre_cap_size, realized_vol_daily,
        realized_vol_floor, cap_hit, floor_hit, error (None or string)
    """
    sorted_events = sorted(events, key=lambda e: pd.Timestamp(e["entry_ts"]))
    pending_exits: list[pd.Timestamp] = []
    n_active = 0
    out: list[dict] = []

    for ev in sorted_events:
        entry_ts = pd.Timestamp(ev["entry_ts"])
        # Drain finished positions whose exit ts is <= this entry.
        while pending_exits and pending_exits[0] <= entry_ts:
            heapq.heappop(pending_exits)
            n_active -= 1

        rec = dict(ev)
        rec["n_active_at_entry"] = n_active

        if ev.get("decision") != "trade":
            rec.update({
                "applied_size": 0.0,
                "vol_parity_size": 0.0,
                "pre_cap_size": 0.0,
                "realized_vol_daily": None,
                "realized_vol_floor": None,
                "cap_hit": False,
                "floor_hit": False,
                "portfolio_r": 0.0,
            })
            out.append(rec)
            continue

        sym = ev["symbol"]
        sym_bars = bars.get(sym)
        if sym_bars is None or sym_bars.empty:
            rec.update({
                "applied_size": 0.0,
                "vol_parity_size": 0.0,
                "pre_cap_size": 0.0,
                "realized_vol_daily": None,
                "realized_vol_floor": None,
                "cap_hit": False,
                "floor_hit": False,
                "portfolio_r": 0.0,
                "skip_reason": "no_bars",
            })
            out.append(rec)
            continue

        s = sizer(sym, entry_ts, sym_bars, n_active)
        if s.get("error"):
            rec.update({
                "applied_size": 0.0,
                "vol_parity_size": 0.0,
                "pre_cap_size": 0.0,
                "realized_vol_daily": None,
                "realized_vol_floor": None,
                "cap_hit": False,
                "floor_hit": False,
                "portfolio_r": 0.0,
                "skip_reason": s["error"],
            })
            out.append(rec)
            continue

        applied = float(s["applied_size"])
        rv_floor = float(s["realized_vol_floor"])
        portfolio_r = applied * float(ev["R"]) * rv_floor
        rec.update({
            "applied_size": applied,
            "vol_parity_size": float(s["vol_parity_size"]),
            "pre_cap_size": float(s["pre_cap_size"]),
            "realized_vol_daily": float(s["realized_vol_daily"]),
            "realized_vol_floor": rv_floor,
            "cap_hit": bool(s["cap_hit"]),
            "floor_hit": bool(s["floor_hit"]),
            "portfolio_r": portfolio_r,
        })
        out.append(rec)
        heapq.heappush(pending_exits, pd.Timestamp(ev["exit_ts"]))
        n_active += 1

    return out


def _summarize_per_event(per_event: list[dict]) -> dict:
    """Build equity curve, max-DD, realized portfolio vol, and per-regime
    diagnostics from a list of per-event records."""
    traded = [e for e in per_event if e.get("decision") == "trade"
              and e.get("portfolio_r") is not None]

    # Equity curve indexed by exit_ts (cumsum of portfolio_r)
    exit_pts = sorted(
        [(pd.Timestamp(e["exit_ts"]), float(e["portfolio_r"])) for e in traded],
        key=lambda x: x[0],
    )
    cum = 0.0
    equity_curve: list[tuple[str, float]] = []
    for ts, r in exit_pts:
        cum += r
        equity_curve.append((ts.isoformat(), cum))

    realized_vol_daily: float | None = None
    max_dd = 0.0
    if exit_pts:
        s = pd.Series(
            [v for _, v in exit_pts],
            index=pd.DatetimeIndex([t for t, _ in exit_pts]),
        )
        # Per-day total portfolio_r (sum of trades that closed that day)
        daily_pr = s.groupby(s.index.date).sum()
        if len(daily_pr) >= 2:
            realized_vol_daily = float(np.std(daily_pr.values, ddof=1))
        # Cumulative equity (per-day end-of-day)
        daily_eq = daily_pr.cumsum()
        if len(daily_eq) >= 1:
            peak = daily_eq.cummax()
            dd = daily_eq - peak
            max_dd = float(dd.min())

    # Per-regime sizing impact
    per_regime: dict[str, dict] = {}
    for ev in traded:
        r = ev.get("regime", "unknown")
        slot = per_regime.setdefault(r, {
            "n_trades": 0, "sum_applied_size": 0.0, "sum_R": 0.0,
            "sum_portfolio_r": 0.0, "n_cap_hits": 0, "n_floor_hits": 0,
        })
        slot["n_trades"] += 1
        slot["sum_applied_size"] += float(ev["applied_size"])
        slot["sum_R"] += float(ev["R"])
        slot["sum_portfolio_r"] += float(ev["portfolio_r"])
        slot["n_cap_hits"] += int(ev.get("cap_hit", False))
        slot["n_floor_hits"] += int(ev.get("floor_hit", False))
    for slot in per_regime.values():
        n = slot["n_trades"]
        slot["mean_applied_size"] = slot["sum_applied_size"] / n if n else None
        slot["mean_R"] = slot["sum_R"] / n if n else None
        slot["mean_portfolio_r"] = slot["sum_portfolio_r"] / n if n else None

    median_exp = _median([float(e["portfolio_r"]) for e in traded])
    cap_hits = sum(1 for e in traded if e.get("cap_hit"))
    floor_hits = sum(1 for e in traded if e.get("floor_hit"))

    return {
        "equity_curve": equity_curve,
        "max_dd": max_dd,
        "realized_portfolio_vol_daily": realized_vol_daily,
        "median_expectancy_r_portfolio": median_exp,
        "cap_hit_count": cap_hits,
        "floor_hit_count": floor_hits,
        "per_regime_sizing_impact": per_regime,
        "n_events_traded": len(traded),
        "n_events_skipped": sum(1 for e in per_event if e.get("decision") != "trade"),
    }


def _p9_sizer(
    target_daily_vol: float, lookback_days: int,
    floor_percentile: float, cap_mult: float,
    regime_size_multiplier: float, meta_p_win_multiplier: float,
):
    """Return a sizer callable that closes over P9 parameters."""
    def sizer(symbol: str, entry_ts: pd.Timestamp,
              sym_bars: pd.DataFrame, n_active: int) -> dict:
        try:
            d: SizingDecision = size_event(
                symbol=symbol, event_ts=entry_ts, bars_5min=sym_bars,
                n_active_positions=n_active,
                target_daily_vol=target_daily_vol,
                lookback_days=lookback_days,
                floor_percentile=floor_percentile,
                cap_mult=cap_mult,
                regime_size_multiplier=regime_size_multiplier,
                meta_p_win_multiplier=meta_p_win_multiplier,
            )
        except ValueError as e:
            return {"error": f"size_event_error: {e}"}
        return {
            "applied_size": d.applied_size,
            "vol_parity_size": d.vol_parity_size,
            "pre_cap_size": d.pre_cap_size,
            "realized_vol_daily": d.realized_vol_daily,
            "realized_vol_floor": d.realized_vol_floor,
            "cap_hit": d.cap_hit,
            "floor_hit": d.floor_hit,
            "error": None,
        }
    return sizer


def _baseline_sizer(lookback_days: int, floor_percentile: float):
    """Return a sizer callable that emits applied_size=1.0 with the SAME
    realized_vol_floor calc P9 uses (so portfolio_r is in the same units).
    Models a fixed-fractional reference where each trade contributes one
    unit of vol (≈ AFML §10.1 baseline against which P9 is compared)."""
    def sizer(symbol: str, entry_ts: pd.Timestamp,
              sym_bars: pd.DataFrame, n_active: int) -> dict:
        past_bars = sym_bars[sym_bars.index < entry_ts]
        try:
            daily_rets = daily_log_returns(past_bars)
            if len(daily_rets) < lookback_days:
                return {"error": "lookback too short"}
            rv_series = realized_daily_vol_ewma(
                daily_rets, lookback_days=lookback_days,
            )
            rv_clean = rv_series.dropna()
            if len(rv_clean) == 0:
                return {"error": "rv warmup empty"}
            floor = vol_floor(rv_clean, floor_percentile=floor_percentile)
            rv_raw = float(rv_clean.iloc[-1])
            rv_floored = max(rv_raw, floor)
            if rv_floored <= 0:
                return {"error": "degenerate vol"}
            floor_hit = rv_raw < floor
        except (KeyError, ValueError) as e:
            return {"error": f"baseline_vol_calc_error: {e}"}
        return {
            "applied_size": 1.0,
            "vol_parity_size": 1.0,
            "pre_cap_size": 1.0,
            "realized_vol_daily": rv_raw,
            "realized_vol_floor": rv_floored,
            "cap_hit": False,
            "floor_hit": floor_hit,
            "error": None,
        }
    return sizer


def _apply_sizing_to_window(
    *,
    label: str,
    events: list[dict],
    bars: dict[str, pd.DataFrame],
    target_daily_vol: float = DEFAULT_TARGET_DAILY_VOL,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    floor_percentile: float = DEFAULT_FLOOR_PERCENTILE,
    cap_mult: float = DEFAULT_CAP_MULT,
    regime_size_multiplier: float = 1.0,
    meta_p_win_multiplier: float = 1.0,
) -> dict:
    """Apply P9 vol-target sizing to a window's events. Pure — caller
    supplies events + bars, no DB access."""
    sizer = _p9_sizer(
        target_daily_vol=target_daily_vol,
        lookback_days=lookback_days,
        floor_percentile=floor_percentile,
        cap_mult=cap_mult,
        regime_size_multiplier=regime_size_multiplier,
        meta_p_win_multiplier=meta_p_win_multiplier,
    )
    per_event = _walk_events(
        label=label, events=events, bars=bars, sizer=sizer,
    )
    summary = _summarize_per_event(per_event)
    return {"label": label, "per_event": per_event, **summary}


def _baseline_window_record(
    *,
    label: str,
    events: list[dict],
    bars: dict[str, pd.DataFrame],
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    floor_percentile: float = DEFAULT_FLOOR_PERCENTILE,
) -> dict:
    """Fixed-fractional baseline reference: applied_size=1.0 for every
    traded event, same realized_vol_floor as P9 (so portfolio_r is in
    matching units). Used to compute per-window ΔR and the DD comparison."""
    sizer = _baseline_sizer(
        lookback_days=lookback_days, floor_percentile=floor_percentile,
    )
    per_event = _walk_events(
        label=label, events=events, bars=bars, sizer=sizer,
    )
    summary = _summarize_per_event(per_event)
    return {"label": label, "per_event": per_event, **summary}


# ---------------------------------------------------------------------------
# Per-window event derivation (called only from main; replicates
# train_p3_regime_routing.py:329-435 to avoid mutating that runner)
# ---------------------------------------------------------------------------

def _patched_provider_ctx(window_end: pd.Timestamp):
    """Re-export of validate_walk_forward._patched_provider for clarity."""
    from validate_walk_forward import _patched_provider
    return _patched_provider(window_end)


def _load_primary_config(bundle_path: Path) -> dict:
    import joblib
    bundle = joblib.load(bundle_path)
    if isinstance(bundle, dict):
        cfg = bundle.get("config")
    else:
        cfg = getattr(bundle, "config", None)
    if cfg is None:
        raise RuntimeError(
            f"{bundle_path} has no .config attribute / 'config' key"
        )
    return cfg


def _build_per_symbol_regime_track(
    *,
    prices_per_symbol: dict[str, pd.DataFrame],
    stability_n: int,
    thresholds,
) -> dict[str, pd.Series]:
    from ml.regime import (
        RegimeStabilityGate, classify_regime, compute_regime_features,
    )
    out: dict[str, pd.Series] = {}
    for sym, b in prices_per_symbol.items():
        if b.empty:
            out[sym] = pd.Series(dtype=object)
            continue
        feats = compute_regime_features(b, thresholds=thresholds)
        gate = RegimeStabilityGate(n=stability_n)
        committed = []
        for ts, row in feats.iterrows():
            raw = classify_regime(row, thresholds=thresholds)
            committed.append(gate.step(raw))
        out[sym] = pd.Series(committed, index=feats.index, name="committed_regime")
    return out


def _decide_trade(
    softprob_row: np.ndarray, decision_threshold: float,
) -> tuple[bool, int]:
    y_pred = int(np.argmax(softprob_row))
    p_max = float(softprob_row[y_pred])
    if p_max < decision_threshold:
        return False, y_pred
    if y_pred == 1:
        return False, y_pred
    return True, y_pred


def _run_window_events(
    *,
    label: str,
    window_end: pd.Timestamp,
    primary_bundle: Path,
    watchlist: list[str],
    days: int = 60,
    frequency: int = 5,
    folds: int = 5,
    embargo_pct: float = 0.01,
    bootstrap_iterations: int = 5,
    dsn: str | None = None,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """Reproduce the per-event walk from train_p3_regime_routing.py
    (lines 329-435). Returns (events, bars).

    events: list of dicts with keys {symbol, entry_ts, exit_ts, regime,
            decision, R, fold, p_max, y_pred}
    bars:   dict[symbol, DataFrame of 5-min bars covering the window],
            populated from the patched provider for vol-target sizing.

    Reuses ``ml/event_view.build_event_view``,
    ``ml/purged_cv.PurgedKFold``, ``ml/primary_oos_harness.refit_primary_on_fold``,
    ``ml/regime`` verbatim — no edits to those modules.
    """
    import os
    import yaml
    from ml.event_view import build_event_view
    from ml.purged_cv import PurgedKFold
    from ml.primary_oos_harness import refit_primary_on_fold
    from ml.regime import RegimeThresholds, SEVEN_STATES  # noqa: F401

    primary_config = _load_primary_config(primary_bundle)

    pt_mult = primary_config["pt_mult"]
    sl_mult = primary_config["sl_mult"]
    max_holding = primary_config["max_holding"]
    cost_model_cfg = primary_config["cost_model"]
    labeling_cfg = primary_config["labeling"]
    cusum_h = labeling_cfg["cusum_h"]

    if dsn is None:
        dsn = os.environ.get(
            "POSTGRES_DSN",
            "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev",
        )

    with _patched_provider_ctx(window_end):
        ev = build_event_view(
            symbols=[s.upper() for s in watchlist],
            days=days, frequency=frequency, dsn=dsn,
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
        logger.warning("[%s] no events collected", label)
        return [], {}

    # Regime config — match train_p3_regime_routing.py's defaults.
    regime_cfg_path = REPO_ROOT / "regime_configs" / "p3.yaml"
    with open(regime_cfg_path) as f:
        regime_cfg = yaml.safe_load(f)
    stability_n = regime_cfg.get("stability_n", 5)
    decision_threshold = regime_cfg.get("decision_threshold", 0.55)
    per_regime_cfg = regime_cfg.get("regimes", {})

    pro_cfg_path = REPO_ROOT / "pro_trading_config.yaml"
    th_kwargs: dict[str, Any] = {}
    if pro_cfg_path.exists():
        with open(pro_cfg_path) as f:
            pro_cfg = yaml.safe_load(f)
        rd = pro_cfg.get("regime_detector", {}) or {}
        for k_yaml, k_th in (
            ("ema_fast", "ema_fast"),
            ("ema_slow", "ema_slow"),
            ("rv_period", "rv_period"),
            ("rv_median_lookback", "rv_median_lookback"),
            ("adx_trending_threshold", "adx_trending"),
            ("chop_index_threshold", "chop_index"),
        ):
            if k_yaml in rd:
                th_kwargs[k_th] = rd[k_yaml]
                if k_th in ("ema_fast", "ema_slow", "rv_period",
                            "rv_median_lookback"):
                    th_kwargs[k_th] = int(th_kwargs[k_th])
                else:
                    th_kwargs[k_th] = float(th_kwargs[k_th])
    thresholds = RegimeThresholds(**th_kwargs)

    regime_tracks = _build_per_symbol_regime_track(
        prices_per_symbol=ev.prices_per_symbol or {},
        stability_n=stability_n,
        thresholds=thresholds,
    )

    # Per-fold OOS softprob → per-event decisions.
    touch_series = pd.Series(ev.touch_times.to_numpy(), index=ev.event_times)
    cv = PurgedKFold(
        n_splits=folds, touch_times=touch_series,
        embargo_pct=embargo_pct,
    )

    events_out: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(ev.X), 1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        proba_test = refit_primary_on_fold(
            X_train=ev.X[train_idx], y_train=ev.y[train_idx],
            weights_train=ev.weights[train_idx],
            X_test=ev.X[test_idx],
            event_times_train=ev.event_times[train_idx],
            touch_times_train=ev.touch_times[train_idx],
            primary_config=primary_config,
            sample_r_multiples_train=ev.sample_r_multiples[train_idx],
            sample_cost_r_train=ev.sample_cost_r[train_idx],
            bootstrap_iterations=bootstrap_iterations,
            use_sequential_bootstrap=True,
            seed=42 + fold_idx,
        )

        for i, ev_idx in enumerate(test_idx):
            event_time = ev.event_times[ev_idx]
            sym = ev.sym_groups[ev_idx]
            track = regime_tracks.get(sym)
            if track is None or track.empty:
                committed = "chop"
            else:
                try:
                    committed = track.loc[event_time]
                except KeyError:
                    pos = track.index.searchsorted(
                        event_time, side="right",
                    ) - 1
                    committed = (
                        track.iloc[max(0, pos)] if len(track) else "chop"
                    )

            allow = per_regime_cfg.get(committed, {}).get("allow_trade", True)
            per_regime_thresh = per_regime_cfg.get(committed, {}).get(
                "decision_threshold", decision_threshold,
            )
            do_trade, y_pred = _decide_trade(
                softprob_row=proba_test[i],
                decision_threshold=per_regime_thresh,
            )
            if committed == "chop":
                allow = False

            decision_str = "trade" if (do_trade and allow) else "skip"
            side = 1.0 if y_pred == 2 else -1.0
            gross_r = side * float(ev.sample_r_multiples[ev_idx])
            net_r = gross_r - float(ev.sample_cost_r[ev_idx])

            events_out.append({
                "symbol": str(sym),
                "entry_ts": pd.Timestamp(event_time).isoformat(),
                "exit_ts": pd.Timestamp(ev.touch_times[ev_idx]).isoformat(),
                "regime": str(committed),
                "decision": decision_str,
                "R": net_r,
                "fold": fold_idx,
                "y_pred": int(y_pred),
                "p_max": float(proba_test[i, y_pred]),
            })

    bars_out = {
        sym: df.copy()
        for sym, df in (ev.prices_per_symbol or {}).items()
    }
    return events_out, bars_out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _delta_R(p9_rec: dict, base_rec: dict) -> float | None:
    """Per-window ΔR = mean(portfolio_r_p9) - mean(portfolio_r_baseline)
    over events traded by both. Returns None when neither side has trades."""
    p9_traded = {
        (e["symbol"], e["entry_ts"]): float(e["portfolio_r"])
        for e in p9_rec["per_event"]
        if e.get("decision") == "trade"
        and e.get("portfolio_r") is not None
        and e.get("skip_reason") is None
    }
    base_traded = {
        (e["symbol"], e["entry_ts"]): float(e["portfolio_r"])
        for e in base_rec["per_event"]
        if e.get("decision") == "trade"
        and e.get("portfolio_r") is not None
        and e.get("skip_reason") is None
    }
    common = sorted(set(p9_traded) & set(base_traded))
    if not common:
        return None
    p9_mean = sum(p9_traded[k] for k in common) / len(common)
    base_mean = sum(base_traded[k] for k in common) / len(common)
    return p9_mean - base_mean


def _write_diff_md(report: dict, out_path: Path) -> None:
    """Render the per-window vol-target vs baseline diff to markdown."""
    lines: list[str] = []
    lines.append("# P9 vol-target vs baseline-P4-retune — 20-window diff\n")
    lines.append(f"_Report timestamp_: `{report['timestamp']}`")
    lines.append(f"_Released bundle SHA256_: "
                 f"`{report['released_bundle_sha256'].get('after')}` "
                 f"(unchanged: {report['released_bundle_sha256'].get('unchanged')})")
    cfg = report["config"]
    lines.append(
        f"_Config_: target_daily_vol={cfg['target_daily_vol']}, "
        f"lookback_days={cfg['lookback_days']}, "
        f"floor_pct={cfg['floor_percentile']}, "
        f"cap_mult={cfg['cap_mult']}, "
        f"regime_mult={cfg['regime_size_multiplier']}, "
        f"meta_mult={cfg['meta_p_win_multiplier']}\n"
    )
    agg = report["aggregates"]
    lines.append("## Aggregates across windows\n")
    lines.append("| Metric | P9 | Baseline |")
    lines.append("|---|---|---|")
    lines.append(f"| n_windows_evaluated | {agg['n_windows_evaluated']} | "
                 f"{agg['n_windows_evaluated']} |")
    lines.append(
        f"| median realized_portfolio_vol_daily | "
        f"{agg.get('median_realized_vol_daily_p9')} | "
        f"{agg.get('median_realized_vol_daily_baseline')} |"
    )
    lines.append(
        f"| median max_dd | {agg['median_max_dd_p9']} | "
        f"{agg['median_max_dd_baseline']} |"
    )
    lines.append(
        f"| P95 max_dd | {agg['p95_max_dd_p9']} | "
        f"{agg['p95_max_dd_baseline']} |"
    )
    lines.append(f"| median ΔR per window | {agg['median_delta_R']} | — |")
    lines.append(f"| cap_violations | {agg['cap_violations']} | — |\n")

    lines.append("## Per-window detail (ok windows only)\n")
    lines.append(
        "| W | Status | n_traded | realized_vol_p9 | realized_vol_base | "
        "max_dd_p9 | max_dd_base | ΔR | cap_hits | floor_hits |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for label in sorted(report["windows"]):
        w = report["windows"][label]
        if w["trainer_status"] != "ok" or "p9" not in w:
            status_cell = (
                w["trainer_status"]
                if "p9" in w or w["trainer_status"] != "ok"
                else f"err: {w.get('error', 'unknown')[:40]}"
            )
            if w["trainer_status"] == "ok" and "p9" not in w:
                status_cell = f"ok-but-error: {w.get('error', '?')[:40]}"
            lines.append(
                f"| {label} | {status_cell} | — | — | — | — | — | — | — | — |"
            )
            continue
        p9 = w["p9"]
        base = w["baseline"]
        lines.append(
            f"| {label} | ok | {p9.get('n_events_traded', 0)} | "
            f"{p9.get('realized_portfolio_vol_daily')} | "
            f"{base.get('realized_portfolio_vol_daily')} | "
            f"{p9.get('max_dd')} | {base.get('max_dd')} | "
            f"{w.get('delta_R')} | {p9.get('cap_hit_count', 0)} | "
            f"{p9.get('floor_hit_count', 0)} |"
        )
    lines.append("")
    lines.append("## Reading\n")
    lines.append(
        "Reading is left to the operator: this artifact reports the "
        "experimental measurements; pytest tests/test_p9_acceptance_gates.py "
        "evaluates the six P9 gates against these numbers.\n"
    )
    out_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    parser.add_argument("--diff-path", default=str(DEFAULT_DIFF))
    parser.add_argument(
        "--target-daily-vol", type=float, default=DEFAULT_TARGET_DAILY_VOL,
    )
    parser.add_argument(
        "--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
    )
    parser.add_argument(
        "--floor-percentile", type=float, default=DEFAULT_FLOOR_PERCENTILE,
    )
    parser.add_argument("--cap-mult", type=float, default=DEFAULT_CAP_MULT)
    parser.add_argument("--regime-size-multiplier", type=float, default=1.0)
    parser.add_argument("--meta-p-win-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--only-windows", default=None,
        help="Comma-separated subset of W01-W20 (e.g. W15,W19) — debug only",
    )
    parser.add_argument(
        "--save-progress-every", type=int, default=1,
        help="Save the JSON report after every N windows (default 1: every "
             "window), so a crash leaves a partial inspectable artifact.",
    )
    args = parser.parse_args()

    started = time.time()
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)

    released_bundle = REPO_ROOT / "ml_model_v2_retune.pkl"
    if not released_bundle.exists():
        raise FileNotFoundError(
            f"released bundle not found: {released_bundle}"
        )
    sha_before = _sha256(released_bundle)

    if not BASE_RATE_REPORT.exists():
        raise FileNotFoundError(
            f"base-rate report not found: {BASE_RATE_REPORT} — run "
            "validate_walk_forward_base_rate.py first"
        )
    base_rate = json.loads(BASE_RATE_REPORT.read_text())

    # Watchlist mirrors the base-rate sweep.
    from validate_walk_forward import WATCHLIST
    from validate_walk_forward_base_rate import WINDOWS

    only = (
        set(s.strip().upper() for s in args.only_windows.split(","))
        if args.only_windows else None
    )

    per_window: dict[str, dict] = {}
    n_done = 0
    for label, spec in WINDOWS.items():
        if only is not None and label.upper() not in only:
            continue
        br_w = base_rate["windows"].get(label, {})
        trainer_status = br_w.get("trainer_status", "missing")
        rec: dict[str, Any] = {
            "label": br_w.get("label", spec.get("label")),
            "window_end_ts": br_w.get(
                "window_end_ts", spec["end"].isoformat(),
            ),
            "trainer_status": trainer_status,
        }

        if trainer_status != "ok":
            rec["reject_reason"] = br_w.get("trainer_reject_reason")
            rec["reject_kind"] = br_w.get("reject_kind")
            per_window[label] = rec
            logger.info(
                "[%s] skipped — trainer_status=%s", label, trainer_status,
            )
            continue

        # Locate the per-window primary bundle from the base-rate sweep.
        per_window_bundle = REPO_ROOT / f"ml_model_v2_baserate_{label.lower()}.pkl"
        if not per_window_bundle.exists():
            # W01-W03 use the older per-window bundles from
            # walk_forward_validation. Naming convention there is
            # `ml_model_v2_walk_w{1,2,3}.pkl` — single digits with NO
            # leading zero. Try both with and without zero-padding.
            int_part = int(label[1:])  # 1, 2, 3, ..., 20
            alt_candidates = [
                REPO_ROOT / f"ml_model_v2_walk_w{int_part}.pkl",
                REPO_ROOT / f"ml_model_v2_walk_{label.lower()}.pkl",
            ]
            for alt in alt_candidates:
                if alt.exists():
                    per_window_bundle = alt
                    break
            else:
                logger.error(
                    "[%s] per-window primary bundle missing: tried %s — skipping",
                    label, [str(p) for p in [
                        REPO_ROOT / f"ml_model_v2_baserate_{label.lower()}.pkl",
                        *alt_candidates,
                    ]],
                )
                rec["error"] = (
                    f"missing per-window bundle: tried baserate_{label.lower()} "
                    f"and walk_w{int_part}"
                )
                per_window[label] = rec
                continue

        window_end = pd.Timestamp(spec["end"])
        logger.info("[%s] running event walk — bundle=%s window_end=%s",
                    label, per_window_bundle.name, window_end)
        try:
            events, bars = _run_window_events(
                label=label, window_end=window_end,
                primary_bundle=per_window_bundle, watchlist=WATCHLIST,
            )
        except Exception as e:
            logger.exception("[%s] event walk failed", label)
            rec["error"] = f"event_walk_failed: {e}"
            per_window[label] = rec
            continue

        logger.info("[%s] sizing — n_events=%d", label, len(events))
        p9 = _apply_sizing_to_window(
            label=label, events=events, bars=bars,
            target_daily_vol=args.target_daily_vol,
            lookback_days=args.lookback_days,
            floor_percentile=args.floor_percentile,
            cap_mult=args.cap_mult,
            regime_size_multiplier=args.regime_size_multiplier,
            meta_p_win_multiplier=args.meta_p_win_multiplier,
        )
        base = _baseline_window_record(
            label=label, events=events, bars=bars,
            lookback_days=args.lookback_days,
            floor_percentile=args.floor_percentile,
        )
        rec["p9"] = p9
        rec["baseline"] = base
        rec["delta_R"] = _delta_R(p9, base)
        rec["per_window_primary_bundle"] = str(per_window_bundle)
        rec["per_window_primary_sha256"] = _sha256(per_window_bundle)
        per_window[label] = rec
        logger.info(
            "[%s] done — p9_max_dd=%s base_max_dd=%s delta_R=%s "
            "n_traded=%d cap_hits=%d",
            label, p9.get("max_dd"), base.get("max_dd"), rec.get("delta_R"),
            p9.get("n_events_traded", 0), p9.get("cap_hit_count", 0),
        )

        n_done += 1
        if n_done % max(1, args.save_progress_every) == 0:
            _write_partial(
                per_window=per_window, sha_before=sha_before,
                released_bundle=released_bundle, started=started,
                args=args, report_path=Path(args.report_path),
            )

    sha_after = _sha256(released_bundle)
    elapsed = time.time() - started

    # Aggregates
    ok_windows = [
        w for w in per_window.values() if w["trainer_status"] == "ok"
        and w.get("p9") is not None and w.get("baseline") is not None
    ]
    realized_vols_p9 = [
        w["p9"]["realized_portfolio_vol_daily"]
        for w in ok_windows
        if w["p9"]["realized_portfolio_vol_daily"] is not None
    ]
    realized_vols_base = [
        w["baseline"]["realized_portfolio_vol_daily"]
        for w in ok_windows
        if w["baseline"]["realized_portfolio_vol_daily"] is not None
    ]
    max_dds_p9 = [w["p9"]["max_dd"] for w in ok_windows]
    max_dds_base = [w["baseline"]["max_dd"] for w in ok_windows]
    deltas = [w["delta_R"] for w in ok_windows if w.get("delta_R") is not None]
    cap_violations = sum(
        1
        for w in ok_windows
        for ev in w["p9"]["per_event"]
        if ev.get("decision") == "trade"
        and ev.get("vol_parity_size", 0) > 0
        and (ev.get("applied_size", 0) / ev.get("vol_parity_size", 1)) > 5.0 + 1e-9
    )

    report = {
        "timestamp": datetime.now().isoformat(),
        "released_bundle": str(released_bundle),
        "released_bundle_sha256": {
            "before": sha_before, "after": sha_after,
            "unchanged": sha_before == sha_after,
        },
        "config": {
            "target_daily_vol": args.target_daily_vol,
            "lookback_days": args.lookback_days,
            "floor_percentile": args.floor_percentile,
            "cap_mult": args.cap_mult,
            "regime_size_multiplier": args.regime_size_multiplier,
            "meta_p_win_multiplier": args.meta_p_win_multiplier,
        },
        "windows": per_window,
        "aggregates": {
            "n_windows_evaluated": len(ok_windows),
            "median_realized_vol_daily_p9": _median(realized_vols_p9),
            "median_realized_vol_daily_baseline": _median(realized_vols_base),
            "median_max_dd_p9": _median(max_dds_p9),
            "p95_max_dd_p9": _percentile(max_dds_p9, 0.05),
            "median_max_dd_baseline": _median(max_dds_base),
            "p95_max_dd_baseline": _percentile(max_dds_base, 0.05),
            "median_delta_R": _median(deltas),
            "cap_violations": cap_violations,
        },
        "spec_reference": (
            "P9 — vol-targeted position sizing (AFML §10.1) on the "
            "20-window AR(1) base-rate sweep"
        ),
        "elapsed_seconds": round(elapsed, 1),
    }
    Path(args.report_path).write_text(json.dumps(report, indent=2, default=str))
    _write_diff_md(report, Path(args.diff_path))

    logger.info(
        "p9_done elapsed=%.1fs n_ok=%d median_realized_vol=%s "
        "median_max_dd_p9=%s vs baseline=%s median_delta_R=%s "
        "cap_violations=%d report=%s",
        elapsed, len(ok_windows),
        f"{report['aggregates']['median_realized_vol_daily_p9']:.5f}"
        if report['aggregates']['median_realized_vol_daily_p9'] is not None
        else "None",
        report["aggregates"]["median_max_dd_p9"],
        report["aggregates"]["median_max_dd_baseline"],
        report["aggregates"]["median_delta_R"],
        cap_violations, args.report_path,
    )
    return 0


def _write_partial(
    *, per_window: dict, sha_before: str, released_bundle: Path,
    started: float, args, report_path: Path,
) -> None:
    """Persist a partial in-progress report — useful when the 20-window
    run is long. Aggregates may be ``None`` until run completes."""
    sha_now = _sha256(released_bundle)
    partial = {
        "timestamp": datetime.now().isoformat(),
        "in_progress": True,
        "released_bundle_sha256": {
            "before": sha_before, "after": sha_now,
            "unchanged": sha_before == sha_now,
        },
        "config": {
            "target_daily_vol": args.target_daily_vol,
            "lookback_days": args.lookback_days,
            "floor_percentile": args.floor_percentile,
            "cap_mult": args.cap_mult,
            "regime_size_multiplier": args.regime_size_multiplier,
            "meta_p_win_multiplier": args.meta_p_win_multiplier,
        },
        "windows": per_window,
        "aggregates": {},
        "elapsed_seconds_so_far": round(time.time() - started, 1),
    }
    report_path.write_text(json.dumps(partial, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
