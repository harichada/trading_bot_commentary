"""Pure decision function for the per-symbol strategy-fit gate. The engine audits and acts on the result."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# v-strategy-fit-review2-2026-09-25 (Important 2): keys must be the exact
# `reasoning["strategy"]` values live strategies actually emit, or the gate
# silently governs nothing. Dropped "oversold_v2" and "news" (no strategy
# writes either); added mean_reversion_short, oversold_bounce_v2, and
# free_news_sentiment. tests/core/test_strategy_fit_gate.py greps
# strategies/*.py to keep every key anchored to a real literal.
LANE_MAP = {
    "mean_reversion": "mean_rev_intraday", "mean_reversion_short": "mean_rev_intraday",
    "oversold_bounce_v2": "mean_rev_intraday",
    "day_trade_momentum": "dt_momentum", "day_trade_momentum_short": "dt_momentum",
    "orb_contraction_rvol": "orb",
    "news_sentiment": "news_catalyst", "free_news_sentiment": "news_catalyst",
}
FIT_PREFIX = "fit:"


def family_for(strategy_name: str) -> Optional[str]:
    if strategy_name in LANE_MAP:
        return LANE_MAP[strategy_name]
    if strategy_name.startswith(FIT_PREFIX) and len(strategy_name) > len(FIT_PREFIX):
        return strategy_name[len(FIT_PREFIX):]
    return None


@dataclass(frozen=True)
class GateDecision:
    action: str
    reason: str
    governed: bool
    family: Optional[str] = None
    verdict: Optional[str] = None
    score: Optional[float] = None
    exp_r: Optional[float] = None
    stability: Optional[float] = None
    source: Optional[str] = None
    age_days: Optional[float] = None

    def audit_fields(self) -> dict:
        out = {}
        for k in ("family", "verdict", "score", "exp_r", "stability", "source", "age_days"):
            v = getattr(self, k)
            if v is None:
                continue
            out[k] = round(v, 3) if isinstance(v, float) else v
        return out


def _blocked(shadow: bool, reason: str, **fields) -> GateDecision:
    return GateDecision("shadow" if shadow else "skip", reason, True, **fields)


def _score(row) -> float:
    """Row score as a comparable float; NaN and garbage sort to -inf.

    v-strategy-fit-review2-2026-09-25 (Minor): `nan > x` is False in both
    directions, so an unguarded comparison made a NaN-scored competitor
    neither higher nor lower — harmless by luck here, but the priority rule
    must state "NaN is never higher" rather than rely on that."""
    try:
        v = float(getattr(row, "score", float("nan")))
    except (TypeError, ValueError):
        return float("-inf")
    return v if math.isfinite(v) else float("-inf")


def _higher_score(other, row) -> bool:
    return _score(other) > _score(row)


def evaluate_fit_gate(fit, strategy_name: str, side: str, symbol: str, enabled: bool, shadow: bool,
                      competing_families: Optional[list[str]] = None) -> GateDecision:
    if not enabled:
        return GateDecision("proceed", "gate_disabled", False)
    family = family_for(strategy_name)
    if family is None:
        return GateDecision("proceed", "fit_not_governed", False)
    age = fit.age_days()
    if fit.is_stale():
        reason = "fit_stale" if fit.summary().get("computed_at") else "fit_missing"
        return _blocked(shadow, reason, family=family, age_days=age)
    row = fit.lookup(symbol, family, side)
    if row is None:
        return _blocked(shadow, "fit_missing_row", family=family, age_days=age)
    fields = dict(family=family, verdict=row.verdict, score=row.score, exp_r=row.exp_r, stability=row.stability,
                  source=row.source, age_days=age)
    if row.verdict == "PASS":
        for other in competing_families or []:
            o = fit.lookup(symbol, other, side)
            if o is not None and o.verdict == "PASS" and _higher_score(o, row):
                return _blocked(shadow, "fit_lower_priority", **fields)
        return GateDecision("proceed", "fit_pass", True, **fields)
    if row.verdict == "WATCH":
        return _blocked(shadow, "fit_watch_shadow", **fields)
    return _blocked(shadow, "fit_fail", **fields)
