"""Stage A style metrics plus stability, decay, t-stat, and bootstrap lower bound."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

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


def _session_means(trades: list[SimTrade]) -> np.ndarray:
    by: dict[Any, list[float]] = defaultdict(list)
    for t in trades:
        by[t.entry_ts.normalize()].append(t.r_multiple)
    return np.array([float(np.mean(rs)) for rs in by.values()], dtype=float)


def _t_stat_session(trades: list[SimTrade]) -> float:
    """Same-session trades (often across symbols) are correlated, not independent --
    a naive per-trade t-stat overstates significance on a pooled row. Group by
    session (`entry_ts.normalize()`), treat each session's mean r as one observation,
    and t-test on those `S` observations instead of the `n` raw trades."""
    m_s = _session_means(trades)
    s = m_s.size
    if s < 2:
        return 0.0
    std_s = float(m_s.std(ddof=1))
    if std_s == 0:
        return 0.0
    return float(m_s.mean() / (std_s / np.sqrt(s)))


def _recovery(net_r: float, max_dd_r: float) -> float:
    """Net R over max drawdown R. Unlike a raw `max_dd_r <= 60R` cap, this stays
    meaningful regardless of pooled trade count -- a 13k-trade pooled row's 639R
    drawdown can still be a real edge if net R clears it several times over."""
    if net_r > 0 and max_dd_r > 0:
        return net_r / max_dd_r
    if net_r > 0 and max_dd_r == 0:
        return float("inf")
    return 0.0


def compute_metrics(trades: list[SimTrade], n_windows: int, bootstrap: int = 200, seed: int = 0) -> FitMetrics:
    if not trades:
        return FitMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    r = np.array([t.r_multiple for t in trades], dtype=float)
    exp_r = float(r.mean())
    net_r = float(r.sum())
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
    t_stat_session = _t_stat_session(trades)
    recovery = _recovery(net_r, max_dd_r)
    return FitMetrics(int(r.size), sessions, exp_r, exp_r_lb, t_stat, pf, wr, max_dd_r, stability, decay,
                      t_stat_session, recovery)


def per_regime(trades: list[SimTrade]) -> dict[str, dict]:
    by: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        by[t.regime].append(t.r_multiple)
    return {k: {"n": len(v), "exp_r": float(np.mean(v))} for k, v in sorted(by.items())}
