"""Research-side domain types. All frozen."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

VERDICTS = ("PASS", "WATCH", "FAIL")
TRADE_COLUMNS = ("symbol", "family", "side", "entry_ts", "exit_ts", "entry_px", "stop_px", "target_px", "exit_px",
                 "entry_fill", "exit_fill", "r_multiple", "pnl_pct", "cost_pct", "exit_reason", "hold_bars",
                 "hold_sessions", "tag", "regime", "window_id", "params_json")


@dataclass(frozen=True)
class SimTrade:
    symbol: str
    family: str
    side: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_px: float
    stop_px: float
    target_px: float
    exit_px: float
    entry_fill: float
    exit_fill: float
    r_multiple: float
    pnl_pct: float
    cost_pct: float
    exit_reason: str
    hold_bars: int
    hold_sessions: int
    tag: str
    regime: str
    window_id: int
    params_json: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: d[k] for k in TRADE_COLUMNS}


@dataclass(frozen=True)
class FitMetrics:
    n: int
    sessions: int
    exp_r: float
    exp_r_lb: float
    t_stat: float
    pf: float
    wr: float
    max_dd_r: float
    stability: float
    decay: float
    # v-playbook-desk-b1-5 task 9: last two, with defaults, so every existing positional
    # FitMetrics(...) construction (compute_metrics's early return, tests) keeps working.
    t_stat_session: float = 0.0
    recovery: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Window:
    window_id: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
