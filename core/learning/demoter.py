"""v-learning-loop-2026-10-22: auto-demote triggers (LIVE -> shadow). NEVER promotes.

Triggers, evaluated per (family, side) on the live tape:
  day_loss_gt_2r         a session within `lookback_days` lost more than 2R in total
  three_stops_in_session 3 consecutive ~-1R stop-outs (r <= -0.9, stop exit) in one session
  rolling30_pf_lt1_neg   last 30 trades: PF < 1.0 AND expectancy < 0 (needs >= MIN_ROLLING_N)

`apply_demotions` only ever moves a playbook key *down* to `suspended`
(size 0.0). The existing playbook gate (`core/desk/playbook.evaluate_playbook_gate`,
ENABLE_PLAYBOOK_GATE) is what actually skips the signal. The nightly playbook
updater rewrites the playbook from evidence each night, and the demoter re-applies
after it, so a demotion lapses once the trigger falls out of its window.
Re-enabling LIVE is always a human decision.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from core.desk.playbook import (Playbook, PlaybookEntry, STATUS_SUSPENDED, SUSPENDED_SIZE_MULT)
from core.learning.scorecard import LearnTrade

DAY_LOSS_R = 2.0
STOP_R = -0.9
STOP_STREAK = 3
ROLLING_N = 30
MIN_ROLLING_N = 10
LOOKBACK_DAYS = 10
REGIMES = ("risk_on", "risk_off", "mixed", "unknown")
REASON_PREFIX = "learning_demote:"

FamilySide = Tuple[str, str]


@dataclass(frozen=True)
class Demotion:
    family: str
    side: str
    trigger: str
    detail: str
    session: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {"family": self.family, "side": self.side, "trigger": self.trigger,
                "detail": self.detail, "session": self.session}


def _within(session: str, today: date, days: int) -> bool:
    try:
        d = date.fromisoformat(session[:10])
    except (TypeError, ValueError):
        return False
    return today - timedelta(days=days) <= d <= today


def evaluate_family_side(trades: Sequence[LearnTrade], today: date,
                         lookback_days: int = LOOKBACK_DAYS) -> List[Demotion]:
    if not trades:
        return []
    fam, side = trades[0].family, trades[0].side
    ts = sorted(trades, key=lambda t: t.exit_ts)
    out: List[Demotion] = []
    by_day: Dict[str, List[LearnTrade]] = defaultdict(list)
    for t in ts:
        by_day[t.session].append(t)
    for sess in sorted(by_day):
        if not _within(sess, today, lookback_days):
            continue
        day = by_day[sess]
        tot = sum(t.r for t in day)
        if tot < -DAY_LOSS_R:
            out.append(Demotion(fam, side, "day_loss_gt_2r", f"{tot:.2f}R over {len(day)} trades", sess))
        streak = best = 0
        for t in day:
            streak = streak + 1 if (t.is_stop and t.r <= STOP_R) else 0
            best = max(best, streak)
        if best >= STOP_STREAK:
            out.append(Demotion(fam, side, "three_stops_in_session", f"{best} consecutive -1R stops", sess))
    last = ts[-ROLLING_N:]
    if len(last) >= MIN_ROLLING_N and _within(last[-1].session, today, lookback_days * 3):
        rs = [t.r for t in last]
        gw, gl = sum(r for r in rs if r > 0), -sum(r for r in rs if r < 0)
        pf = gw / gl if gl > 0 else float("inf")
        exp = sum(rs) / len(rs)
        if pf < 1.0 and exp < 0:
            out.append(Demotion(fam, side, "rolling30_pf_lt1_neg",
                                f"n={len(rs)} pf={pf:.2f} exp={exp:+.3f}R", last[-1].session))
    return out


def evaluate(trades: Iterable[LearnTrade], today: date,
             lookback_days: int = LOOKBACK_DAYS) -> List[Demotion]:
    groups: Dict[FamilySide, List[LearnTrade]] = defaultdict(list)
    for t in trades:
        if t.source == "live":
            groups[(t.family, t.side)].append(t)
    out: List[Demotion] = []
    for k in sorted(groups):
        out.extend(evaluate_family_side(groups[k], today, lookback_days))
    return out


def apply_demotions(playbook: Playbook, demotions: Sequence[Demotion], now_iso: str) -> Playbook:
    """Suspend every regime key of each demoted (family, side). Monotone: a key
    can only go to `suspended`; nothing here can enable or upsize anything."""
    book = playbook
    seen = set()
    for d in demotions:
        if (d.family, d.side) in seen:
            continue
        seen.add((d.family, d.side))
        for regime in REGIMES:
            prev: Optional[PlaybookEntry] = book.entries.get((d.family, d.side, regime))
            if prev is not None and prev.status == STATUS_SUSPENDED:
                continue
            book = book.with_entry(PlaybookEntry(
                family=d.family, side=d.side, regime=regime, status=STATUS_SUSPENDED,
                size_mult=SUSPENDED_SIZE_MULT,
                exp_r_recent=prev.exp_r_recent if prev else 0.0,
                t_stat_recent=prev.t_stat_recent if prev else 0.0,
                n_recent=prev.n_recent if prev else 0,
                updated_at=now_iso, reason=f"{REASON_PREFIX}{d.trigger}"))
    return book
