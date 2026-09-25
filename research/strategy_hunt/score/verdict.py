"""PASS / WATCH / FAIL against locked floors."""
from __future__ import annotations

from research.strategy_hunt.models import FitMetrics

FLOORS = {"min_n": 150, "min_sessions": 10, "min_pf": 1.30, "min_wr": 0.48, "min_exp_r": 0.05,
          "max_dd_r": 20.0, "min_stability": 0.6, "min_decay": 0.5, "min_t_stat": 2.0, "min_exp_r_lb": 0.0}
WATCH_FLOORS = {"min_n": 60, "min_pf": 1.1}


def _pass_failures(m: FitMetrics) -> list[str]:
    checks = {
        "min_n": m.n >= FLOORS["min_n"], "min_sessions": m.sessions >= FLOORS["min_sessions"],
        "min_pf": m.pf >= FLOORS["min_pf"], "min_wr": m.wr >= FLOORS["min_wr"],
        "min_exp_r": m.exp_r >= FLOORS["min_exp_r"], "max_dd_r": m.max_dd_r <= FLOORS["max_dd_r"],
        "min_stability": m.stability >= FLOORS["min_stability"], "min_decay": m.decay >= FLOORS["min_decay"],
        "min_t_stat": m.t_stat >= FLOORS["min_t_stat"], "min_exp_r_lb": m.exp_r_lb >= FLOORS["min_exp_r_lb"],
    }
    return [k for k, ok in checks.items() if not ok]


def verdict(m: FitMetrics) -> tuple[str, list[str]]:
    failed = _pass_failures(m)
    if not failed:
        return "PASS", []
    if m.n >= WATCH_FLOORS["min_n"] and m.exp_r > 0 and m.pf >= WATCH_FLOORS["min_pf"]:
        return "WATCH", failed
    return "FAIL", failed
