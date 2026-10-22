"""v-learning-loop-2026-10-22: rolling per-(family, side, regime) scorecard.

Pure: takes `LearnTrade`s, returns plain dicts. `write_scorecard` is the only
I/O and is called by `research/learning/nightly.py` when
ENABLE_LEARNING_SCORECARD is on. Report only — nothing in the live path reads it.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.learning import stage_a

Key = Tuple[str, str, str]
STOP_REASONS = ("stop_loss", "stop", "hard_stop")
# R per trade = 1% of the account is the risk unit the bot sizes to, so a
# drawdown of D R is ~D% of equity. Used only for the DD% Stage A gate.
DEFAULT_RISK_PCT_PER_R = 0.01


@dataclass(frozen=True)
class LearnTrade:
    family: str
    side: str
    regime: str
    r: float
    session: str            # ET date YYYY-MM-DD of the exit
    exit_ts: str            # ISO, sortable
    exit_reason: str = ""
    symbol: str = ""
    pnl: float = 0.0
    source: str = "live"

    @property
    def key(self) -> Key:
        return (self.family, self.side, self.regime)

    @property
    def is_stop(self) -> bool:
        er = (self.exit_reason or "").lower()
        return any(er.startswith(s) for s in STOP_REASONS)


def metrics(trades: Sequence[LearnTrade], risk_pct_per_r: float = DEFAULT_RISK_PCT_PER_R) -> Dict[str, Any]:
    ts = sorted(trades, key=lambda t: t.exit_ts)
    n = len(ts)
    rs = [float(t.r) for t in ts]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    gross_w, gross_l = sum(wins), -sum(losses)
    pf: Optional[float] = (gross_w / gross_l) if gross_l > 0 else (None if n == 0 else float("inf"))
    exp = (sum(rs) / n) if n else 0.0
    wr = (len(wins) / n) if n else 0.0
    sd = math.sqrt(sum((r - exp) ** 2 for r in rs) / (n - 1)) if n > 1 else 0.0
    t_stat = (exp / (sd / math.sqrt(n))) if n > 1 and sd > 0 else 0.0
    eq = peak = dd = 0.0
    for r in rs:
        eq += r
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    by_day: Dict[str, float] = defaultdict(float)
    for t in ts:
        by_day[t.session] += float(t.r)
    worst_day = min(by_day.values()) if by_day else 0.0
    max_losing_day_r = -worst_day if worst_day < 0 else 0.0
    dd_pct = dd * risk_pct_per_r
    res = stage_a.evaluate(n, len(by_day), pf, wr, exp, dd_pct, max_losing_day_r)
    return {
        "n": n, "sessions": len(by_day), "wins": len(wins), "losses": len(losses),
        "win_rate": round(wr, 4),
        "profit_factor": (None if pf is None else ("inf" if pf == float("inf") else round(pf, 3))),
        "expectancy_r": round(exp, 4), "total_r": round(sum(rs), 3), "t_stat": round(t_stat, 3),
        "max_dd_r": round(dd, 3), "max_dd_pct": round(dd_pct, 4),
        "max_losing_day_r": round(max_losing_day_r, 3),
        "pnl_usd": round(sum(float(t.pnl or 0.0) for t in ts), 2),
        "stage_a": res.verdict, "stage_a_failures": res.failures,
    }


def rolling_scorecard(trades: Iterable[LearnTrade], window: int = 150) -> Dict[str, Dict[str, Any]]:
    """`{"family|side|regime": metrics}` over the newest `window` trades per key,
    plus `"family|side|*"` roll-ups (regime-pooled), which is what Stage A
    promotion is actually decided on."""
    by_key: Dict[Key, List[LearnTrade]] = defaultdict(list)
    for t in trades:
        by_key[t.key].append(t)
        by_key[(t.family, t.side, "*")].append(t)
    out: Dict[str, Dict[str, Any]] = {}
    for k in sorted(by_key):
        recent = sorted(by_key[k], key=lambda t: t.exit_ts)[-window:]
        out["|".join(k)] = metrics(recent)
    return out


def write_scorecard(out_dir: Any, card: Dict[str, Any], date_str: str) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"scorecard-{date_str}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"date": date_str, "floors": dict(stage_a.floors()), "keys": card},
                              indent=1, default=str))
    tmp.replace(p)
    return p
