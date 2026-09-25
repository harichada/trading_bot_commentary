"""Regime-conditioned fit rows and universe-pooled regime lanes (spec §6).

Two extra row families on top of the unconditional per-(symbol, family, side)
rows `fit/writer.build_row` produces:

* `regime_rows(...)` — one row per market regime the symbol actually traded in,
  scored against the *same* floors as the unconditional row. A symbol whose edge
  only exists in `risk_on` shows PASS there and FAIL elsewhere, which is what the
  live gate needs in order to stand a lane down when the tape turns.
* `pooled_rows(...)` — one row per (family, side, regime) across the whole
  universe, symbol `"*"`, source `"pooled"`, scored against `POOLED_FLOORS`.
  A pooled PASS is the family-level statement "this lane works in this regime",
  which lets a lane trade a symbol that has no history of its own yet.

Only the three real regime labels are scored (`REGIMES`). `regime_at` returns
"unknown" whenever the SPY/VIX context is missing for a session; a row built from
those trades would describe the data gap, not a regime, and the gate never asks
for it — so those trades are dropped from both row families (they remain in the
unconditional row, which is computed over every trade).
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from research.strategy_hunt.fit.writer import ROW_KEYS, build_row
from research.strategy_hunt.models import TRADE_COLUMNS, FitMetrics, SimTrade

REGIMES = ("risk_on", "risk_off", "mixed")
REGIME_ROW_KEYS = ROW_KEYS + ("regime",)

# Pooled floors are deliberately stricter than the per-symbol floors: a pooled row
# aggregates hundreds of symbols, so n is cheap and only a large t-stat and a real
# profit factor distinguish a family-wide edge from the average of many small ones.
# max_dd_r is looser (60R vs 20R) because the pooled equity curve concatenates every
# symbol's trades — its drawdown is not a position-sizing constraint.
POOLED_FLOORS = {"min_n": 500, "min_sessions": 60, "min_pf": 1.2, "min_wr": 0.45, "min_exp_r": 0.05,
                 "max_dd_r": 60.0, "min_stability": 0.6, "min_decay": 0.5, "min_t_stat": 3.0,
                 "min_exp_r_lb": 0.0}
POOLED_WATCH_FLOORS = {"min_n": 200, "min_pf": 1.05}
POOLED_SYMBOL = "*"
POOLED_SOURCE = "pooled"


def _pooled_failures(m: FitMetrics) -> list[str]:
    checks = {
        "min_n": m.n >= POOLED_FLOORS["min_n"], "min_sessions": m.sessions >= POOLED_FLOORS["min_sessions"],
        "min_pf": m.pf >= POOLED_FLOORS["min_pf"], "min_wr": m.wr >= POOLED_FLOORS["min_wr"],
        "min_exp_r": m.exp_r >= POOLED_FLOORS["min_exp_r"], "max_dd_r": m.max_dd_r <= POOLED_FLOORS["max_dd_r"],
        "min_stability": m.stability >= POOLED_FLOORS["min_stability"],
        "min_decay": m.decay >= POOLED_FLOORS["min_decay"],
        "min_t_stat": m.t_stat >= POOLED_FLOORS["min_t_stat"],
        "min_exp_r_lb": m.exp_r_lb >= POOLED_FLOORS["min_exp_r_lb"],
    }
    return [k for k, ok in checks.items() if not ok]


def verdict_pooled(m: FitMetrics) -> tuple[str, list[str]]:
    """PASS / WATCH / FAIL against `POOLED_FLOORS` (mirrors `score.verdict.verdict`)."""
    failed = _pooled_failures(m)
    if not failed:
        return "PASS", []
    if m.n >= POOLED_WATCH_FLOORS["min_n"] and m.exp_r > 0 and m.pf >= POOLED_WATCH_FLOORS["min_pf"]:
        return "WATCH", failed
    return "FAIL", failed


def trades_from_frame(df: pd.DataFrame) -> list[SimTrade]:
    """Rebuild `SimTrade`s from a trades Parquet frame (extra columns ignored)."""
    if df is None or df.empty:
        return []
    return [SimTrade(**{**r, "entry_ts": pd.Timestamp(r["entry_ts"]), "exit_ts": pd.Timestamp(r["exit_ts"])})
            for r in df[list(TRADE_COLUMNS)].to_dict("records")]


def _window_end(trades: list[SimTrade]) -> str:
    return str(max(t.exit_ts for t in trades))[:10] if trades else ""


def regime_rows(symbol: str, family: str, side: str, trades: list[SimTrade], n_windows: int, computed_at: str,
                latest_params: Optional[dict] = None, window_end: str = "") -> list[dict[str, Any]]:
    """One `build_row`-shaped dict (plus `regime`) per regime in `REGIMES` with ≥ 1 trade."""
    by_regime: dict[str, list[SimTrade]] = {}
    for t in trades:
        if t.regime in REGIMES:
            by_regime.setdefault(t.regime, []).append(t)
    out = []
    for regime in REGIMES:
        sub = by_regime.get(regime)
        if not sub:
            continue
        row = build_row(symbol, family, side, sub, n_windows, latest_params or {}, computed_at,
                        window_end or _window_end(sub))
        out.append({**row, "regime": regime})
    return out


def pooled_rows(all_trades: Optional[pd.DataFrame], computed_at: str) -> list[dict[str, Any]]:
    """One pooled row per (family, side, regime) over the whole universe's trades."""
    trades = trades_from_frame(all_trades)
    groups: dict[tuple[str, str, str], list[SimTrade]] = {}
    for t in trades:
        if t.regime in REGIMES:
            groups.setdefault((t.family, t.side, t.regime), []).append(t)
    out = []
    for (family, side, regime), sub in sorted(groups.items()):
        n_windows = len({t.window_id for t in sub})
        row = build_row(POOLED_SYMBOL, family, side, sub, n_windows, {}, computed_at, _window_end(sub),
                        source=POOLED_SOURCE, verdict_fn=verdict_pooled)
        out.append({**row, "regime": regime})
    return out
