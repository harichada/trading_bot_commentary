"""v-learning-loop-2026-10-22: the ONE copy of the Stage A promotion floors.

RESEARCH-LOCKED, do not loosen. Both the trade-count floor AND the session
floor are required (the old `research/shadow_short_resolver.py` gate accepted
either, which let 10 sessions of 12 trades "pass" the sample floor).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

MIN_TRADES = 150
MIN_SESSIONS = 10
MIN_PF = 1.30
MIN_WR = 0.48
MIN_EXPECTANCY_R = 0.05
MAX_DD_PCT = 0.06
MAX_LOSING_DAY_R = 2.0

PASS = "PASS"
FAIL = "FAIL"


@dataclass(frozen=True)
class StageAResult:
    verdict: str
    gates: Dict[str, bool] = field(default_factory=dict)

    @property
    def failures(self) -> List[str]:
        return [k for k, ok in self.gates.items() if not ok]


def sample_gate(n_trades: int, n_sessions: int) -> bool:
    """n >= 150 AND sessions >= 10."""
    return int(n_trades or 0) >= MIN_TRADES and int(n_sessions or 0) >= MIN_SESSIONS


def evaluate(n_trades: int, n_sessions: int, profit_factor: Optional[float], win_rate: float,
             expectancy_r: float, max_dd_pct: float, max_losing_day_r: float) -> StageAResult:
    """All floors. `max_losing_day_r` is the magnitude (positive) of the worst
    day in R; `profit_factor=None` means no losses (treated as passing PF)."""
    pf_ok = profit_factor is None or profit_factor == float("inf") or profit_factor >= MIN_PF
    if profit_factor is None and int(n_trades or 0) == 0:
        pf_ok = False
    gates = {
        f"n>={MIN_TRADES}_and_sessions>={MIN_SESSIONS}": sample_gate(n_trades, n_sessions),
        f"pf>={MIN_PF}": bool(pf_ok),
        f"wr>={MIN_WR}": float(win_rate or 0.0) >= MIN_WR,
        f"exp>={MIN_EXPECTANCY_R}R": float(expectancy_r or 0.0) >= MIN_EXPECTANCY_R,
        f"dd<={MAX_DD_PCT * 100:g}%": float(max_dd_pct or 0.0) <= MAX_DD_PCT,
        f"max_losing_day<={MAX_LOSING_DAY_R}R": abs(float(max_losing_day_r or 0.0)) <= MAX_LOSING_DAY_R,
    }
    return StageAResult(PASS if all(gates.values()) else FAIL, gates)


def floors() -> Tuple[Tuple[str, float], ...]:
    return (("min_trades", MIN_TRADES), ("min_sessions", MIN_SESSIONS), ("min_pf", MIN_PF),
            ("min_wr", MIN_WR), ("min_exp_r", MIN_EXPECTANCY_R), ("max_dd_pct", MAX_DD_PCT),
            ("max_losing_day_r", MAX_LOSING_DAY_R))
