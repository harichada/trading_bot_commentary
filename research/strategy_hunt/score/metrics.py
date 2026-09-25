"""Stage A style metrics plus stability, decay, t-stat, and bootstrap lower bound."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from research.strategy_hunt.models import FitMetrics, SimTrade

SCRATCH_R = 0.05
MIN_WINDOW_TRADES = 3
MIN_QUALIFYING_WINDOWS = 4
DECAY_WINDOWS = 4
MIN_BOOTSTRAP_TRADES = 10


def _window_means(trades: list[SimTrade]) -> dict[int, float]:
    by: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        by[t.window_id].append(t.r_multiple)
    return {w: float(np.mean(rs)) for w, rs in by.items() if len(rs) >= MIN_WINDOW_TRADES}


def compute_metrics(trades: list[SimTrade], n_windows: int, bootstrap: int = 200, seed: int = 0) -> FitMetrics:
    if not trades:
        return FitMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    r = np.array([t.r_multiple for t in trades], dtype=float)
    exp_r = float(r.mean())
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    pf = float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0)
    decided = r[np.abs(r) >= SCRATCH_R]
    wr = float((decided > 0).mean()) if decided.size else 0.0
    ordered = np.array([t.r_multiple for t in sorted(trades, key=lambda t: (t.exit_ts, t.entry_ts))])
    cum = np.cumsum(ordered)
    dd = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:] - cum
    max_dd_r = float(dd.max()) if dd.size else 0.0
    sessions = len({t.entry_ts.normalize() for t in trades})
    means = _window_means(trades)
    if len(means) >= MIN_QUALIFYING_WINDOWS:
        stability = float(np.mean([m > 0 for m in means.values()]))
        last = [means[w] for w in sorted(means)[-DECAY_WINDOWS:]]
        decay = float(np.mean(last) / exp_r) if exp_r > 0 else 0.0
    else:
        stability, decay = 0.0, 0.0
    t_stat = float(exp_r / (r.std(ddof=1) / np.sqrt(r.size))) if r.size >= 2 and r.std(ddof=1) > 0 else 0.0
    if r.size >= MIN_BOOTSTRAP_TRADES:
        rng = np.random.default_rng(seed)
        means_bs = rng.choice(r, size=(bootstrap, r.size), replace=True).mean(axis=1)
        exp_r_lb = float(np.percentile(means_bs, 5))
    else:
        exp_r_lb = 0.0
    return FitMetrics(int(r.size), sessions, exp_r, exp_r_lb, t_stat, pf, wr, max_dd_r, stability, decay)


def per_regime(trades: list[SimTrade]) -> dict[str, dict]:
    by: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        by[t.regime].append(t.r_multiple)
    return {k: {"n": len(v), "exp_r": float(np.mean(v))} for k, v in sorted(by.items())}
