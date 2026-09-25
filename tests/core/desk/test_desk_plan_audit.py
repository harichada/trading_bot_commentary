"""v-playbook-desk-b2-final-2026-09-25 (whole-branch review, M2): the 09:00 plan audit
reports how many trade rows the loss-streak inputs came from.

`losing_day_streak=0 prev_day_r=0.0` reads the same whether the tape really was flat or
`DbLogger.get_bot_trades_between` failed soft to `[]` (an unreachable Postgres, a bad
bind). `trades_read=<n>` separates those two readings in the `desk_plan` audit line.

A separate module from `test_desk_loop.py` (which is at its 400-line budget) with its own
minimal fakes; the engine is not constructible in tests.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

from core.loops import desk_steps as steps
from core.loops.desk_loop import DeskLoop

_CTX = SimpleNamespace(regime="risk_on", spy_change_pct=0.9, vix_level=14.5,
                       vix_change_pct=-3.0, allows_long=True, allows_short=True)
_NOW = datetime(2026, 9, 25, 9, 0, tzinfo=steps.ET_TZ)
_DATE = "2026-09-25"


class _DbLogger:
    def __init__(self, trades):
        self._trades = list(trades)

    async def get_bot_trades_between(self, start, end):
        return list(self._trades)


class _Engine:
    """Only what `_step_plan` reads, plus a recording `_audit`."""

    def __init__(self, trades=()):
        self.is_running = True
        self.market_state = {"breadth": {"advancing_percent": 62.0}}
        self.db_logger = _DbLogger(trades)
        self._strategy_fit = SimpleNamespace(is_stale=lambda: False)
        self.audits: list = []

    def _audit(self, component, symbol, action, reason, **details):
        self.audits.append((component, symbol, action, reason, details))


def _cfg(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        DESK_DATA_DIR=str(tmp_path / "desk"),
        DESK_CALENDAR_PATH=str(tmp_path / "calendar.json"),
        DESK_PLAN_TIME_ET="09:00", DESK_REFRESH_TIME_ET="09:35",
        DESK_JOURNAL_TIME_ET="16:05", DESK_IN_PLAY_N=5, DESK_SCAN_MAX_SYMBOLS=10,
        ENABLE_DESK_WATCHLIST=False, ENABLE_PINNED_WATCHLIST=False, PINNED_WATCHLIST=[],
        STRATEGY_HUNT_DATA_ROOT=str(tmp_path / "research"),
        ENABLE_STRATEGY_FIT_GATE=True, STRATEGY_FIT_SHADOW=True, MAX_DAILY_LOSS=0.05,
    )


def _plan_audit(tmp_path, trades) -> dict:
    engine = _Engine(trades)
    cfg = _cfg(tmp_path)
    loop = DeskLoop(engine, config_fn=lambda: cfg, context_reader=lambda symbol: _CTX)
    asyncio.run(loop._step_plan(cfg, _NOW, _DATE))
    return [a[4] for a in engine.audits if a[0] == "desk_plan"][0]


def test_plan_audit_reports_the_row_count_behind_the_streak(tmp_path):
    details = _plan_audit(tmp_path, [
        {"exit_time": "2026-09-24 15:00", "pnl_r": -1.2, "strategy": "mean_reversion"},
        {"exit_time": "2026-09-23 15:00", "pnl_r": 0.4, "strategy": "mean_reversion"},
    ])
    assert details["trades_read"] == 2
    assert details["losing_day_streak"] == 1        # 09-24 lost, 09-23 did not


def test_plan_audit_reports_zero_rows_when_the_tape_came_back_empty(tmp_path):
    details = _plan_audit(tmp_path, [])
    assert details["trades_read"] == 0
    assert details["losing_day_streak"] == 0        # indistinguishable without the count
