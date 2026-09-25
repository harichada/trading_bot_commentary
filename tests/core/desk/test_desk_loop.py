"""v-playbook-desk-b2-2026-09-25 (Task 5): the desk loop's three scheduled steps.

The engine is not constructible in tests, so the loop is driven against a fake
engine (`market_state`, `dynamic_watchlist`, a provider stub, a db_logger stub
and a recording `_audit`) and a fake config. Every test asserts the shadow
contract: the loop only reads the engine, writes plan/journal files, and — when
`ENABLE_DESK_WATCHLIST` is on — the watchlist. It never blocks anything and
never raises out of a step.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from core.desk.models import DailyPlan
from core.desk.store import load_plan, save_plan
from core.loops import desk_steps as steps
from core.loops.desk_loop import DeskLoop


def _et(y, m, d, hh, mm) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=steps.ET_TZ)


# ────────────────────────────────────────────────────────────────────
# fakes
# ────────────────────────────────────────────────────────────────────

class FakeProvider:
    """`get_market_data` for the symbols it knows; raises for the rest."""

    def __init__(self, frames: dict):
        self.frames = frames
        self.calls: list = []

    def get_market_data(self, symbol, **kw):
        self.calls.append(symbol)
        if symbol not in self.frames:
            raise RuntimeError(f"no data for {symbol}")
        return self.frames[symbol]


class FakeDbLogger:
    def __init__(self, trades=None, decisions=None):
        self._trades = trades or []
        self._decisions = decisions or []

    async def get_bot_trades_between(self, start, end):
        return list(self._trades)

    async def get_bot_decisions_between(self, start, end):
        return list(self._decisions)

    async def get_todays_bot_entries(self, symbols=None):
        return {}


class FakeEngine:
    def __init__(self, *, frames=None, trades=None, decisions=None, breadth=None,
                 watchlist=None, fit_stale=False):
        self.is_running = True
        self.market_state = {"breadth": breadth if breadth is not None
                             else {"advancing_percent": 62.0}}
        self.dynamic_watchlist = list(watchlist or [])
        self.data_provider = FakeProvider(frames or {})
        self.db_logger = FakeDbLogger(trades, decisions)
        self._strategy_fit = SimpleNamespace(is_stale=lambda: fit_stale)
        self.audits: list = []

    def _audit(self, component, symbol, action, reason, **details):
        self.audits.append((component, symbol, action, reason, details))


def _cfg(tmp_path, **over) -> SimpleNamespace:
    cfg = SimpleNamespace(
        DESK_DATA_DIR=str(tmp_path / "desk"),
        DESK_CALENDAR_PATH=str(tmp_path / "calendar.json"),
        DESK_PLAN_TIME_ET="09:00",
        DESK_REFRESH_TIME_ET="09:35",
        DESK_JOURNAL_TIME_ET="16:05",
        DESK_IN_PLAY_N=5,
        DESK_SCAN_MAX_SYMBOLS=10,
        ENABLE_DESK_WATCHLIST=False,
        ENABLE_PINNED_WATCHLIST=True,
        PINNED_WATCHLIST=["SPY"],
        STRATEGY_HUNT_DATA_ROOT=str(tmp_path / "research"),
        ENABLE_STRATEGY_FIT_GATE=True,
        STRATEGY_FIT_SHADOW=True,
        MAX_DAILY_LOSS=0.05,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


_CTX = SimpleNamespace(regime="risk_on", spy_change_pct=0.9, vix_level=14.5,
                       vix_change_pct=-3.0, allows_long=True, allows_short=True)


def _loop(engine, cfg, ctx=_CTX, now=None) -> DeskLoop:
    return DeskLoop(engine, config_fn=lambda: cfg,
                    now_fn=(lambda: now) if now else None,
                    context_reader=lambda symbol: ctx)


def _session(day: str, volume: float = 1000.0, rows: int = 4) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 09:30", periods=rows, freq="5min")
    return pd.DataFrame({"open": 10.0, "high": 10.5, "low": 9.8, "close": 10.1,
                         "volume": volume}, index=idx)


def _bars(today_volume: float = 3000.0) -> pd.DataFrame:
    days = ["2026-09-17", "2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23",
            "2026-09-24"]
    frames = [_session(d) for d in days]
    frames.append(_session("2026-09-25", volume=today_volume))
    return pd.concat(frames)


# ────────────────────────────────────────────────────────────────────
# plan step
# ────────────────────────────────────────────────────────────────────

def test_plan_step_builds_saves_and_audits(tmp_path):
    engine = FakeEngine(trades=[{"exit_time": "2026-09-24 15:00", "pnl_r": -1.2,
                                 "strategy": "mean_reversion"}])
    cfg = _cfg(tmp_path)
    loop = _loop(engine, cfg)
    asyncio.run(loop._step_plan(cfg, _et(2026, 9, 25, 9, 0), "2026-09-25"))

    plan = load_plan(cfg.DESK_DATA_DIR, "2026-09-25")
    assert plan is not None
    assert plan.regime == "risk_on"
    assert plan.max_trades == 6                       # risk_on
    assert plan.breadth_score == pytest.approx(0.62)
    # 5% daily loss cap x 1.0 (risk_on), halved after a -1.2R session
    assert plan.risk_budget_pct == pytest.approx(2.5)
    assert plan.built_at.startswith("2026-09-25T09:00")

    audits = [a for a in engine.audits if a[0] == "desk_plan"]
    assert len(audits) == 1 and audits[0][2] == "build"


def test_plan_step_keeps_an_existing_plan_on_restart(tmp_path):
    cfg = _cfg(tmp_path)
    existing = DailyPlan(date="2026-09-25", regime="risk_off", max_trades=2,
                         built_at="2026-09-25T09:00:00")
    save_plan(existing, cfg.DESK_DATA_DIR)
    engine = FakeEngine()
    loop = _loop(engine, cfg)
    asyncio.run(loop._step_plan(cfg, _et(2026, 9, 25, 10, 0), "2026-09-25"))

    plan = load_plan(cfg.DESK_DATA_DIR, "2026-09-25")
    assert plan.regime == "risk_off" and plan.built_at == "2026-09-25T09:00:00"
    assert [a for a in engine.audits if a[0] == "desk_plan"] == []


def test_plan_step_uses_the_fit_gate_state_for_the_sit_out_rule(tmp_path):
    engine = FakeEngine(fit_stale=True)
    cfg = _cfg(tmp_path, STRATEGY_FIT_SHADOW=False)
    loop = _loop(engine, cfg)
    asyncio.run(loop._step_plan(cfg, _et(2026, 9, 25, 9, 0), "2026-09-25"))
    plan = load_plan(cfg.DESK_DATA_DIR, "2026-09-25")
    assert plan.sit_out is True and plan.sit_out_reason == "fit_stale_gate_enforcing"


def test_plan_step_survives_a_broken_market_context(tmp_path):
    def boom(symbol):
        raise RuntimeError("indices cache down")

    engine = FakeEngine()
    cfg = _cfg(tmp_path)
    loop = DeskLoop(engine, config_fn=lambda: cfg, context_reader=boom)
    asyncio.run(loop._step_plan(cfg, _et(2026, 9, 25, 9, 0), "2026-09-25"))
    plan = load_plan(cfg.DESK_DATA_DIR, "2026-09-25")
    assert plan is not None and plan.regime == "unknown"


# ────────────────────────────────────────────────────────────────────
# refresh step
# ────────────────────────────────────────────────────────────────────

def _with_plan(tmp_path, **over):
    cfg = _cfg(tmp_path, **over)
    save_plan(DailyPlan(date="2026-09-25", regime="risk_on", max_trades=6,
                        long_ok=True, short_ok=True, built_at="2026-09-25T09:00:00"),
              cfg.DESK_DATA_DIR)
    return cfg


def test_refresh_ranks_in_play_and_saves_the_plan(tmp_path):
    cfg = _with_plan(tmp_path)
    engine = FakeEngine(frames={"NVDA": _bars(4500.0), "AMD": _bars(1200.0),
                                "SPY": _bars(1000.0)},
                        watchlist=["NVDA", "AMD"])
    loop = _loop(engine, cfg)
    asyncio.run(loop._step_refresh(cfg, _et(2026, 9, 25, 9, 35), "2026-09-25"))

    plan = load_plan(cfg.DESK_DATA_DIR, "2026-09-25")
    symbols = [r.symbol for r in plan.in_play]
    assert symbols[0] == "NVDA"                 # 4.5x opening volume ranks first
    assert set(symbols) == {"NVDA", "AMD", "SPY"}
    assert plan.refreshed_at.startswith("2026-09-25T09:35")
    assert plan.regime == "risk_on"             # morning decisions untouched

    ranks = [a for a in engine.audits if a[0] == "desk_inplay"]
    assert len(ranks) == 1
    assert ranks[0][2] == "rank" and ranks[0][4]["n"] == 3


def test_refresh_does_not_touch_the_watchlist_when_the_flag_is_off(tmp_path):
    cfg = _with_plan(tmp_path, ENABLE_DESK_WATCHLIST=False)
    engine = FakeEngine(frames={"NVDA": _bars()}, watchlist=["NVDA"])
    loop = _loop(engine, cfg)
    asyncio.run(loop._step_refresh(cfg, _et(2026, 9, 25, 9, 35), "2026-09-25"))
    assert engine.dynamic_watchlist == ["NVDA"]


def test_refresh_hands_the_watchlist_to_the_desk_when_the_flag_is_on(tmp_path):
    cfg = _with_plan(tmp_path, ENABLE_DESK_WATCHLIST=True)
    engine = FakeEngine(frames={"NVDA": _bars(4500.0), "AMD": _bars(1200.0)},
                        watchlist=["NVDA", "AMD"])
    loop = _loop(engine, cfg)
    asyncio.run(loop._step_refresh(cfg, _et(2026, 9, 25, 9, 35), "2026-09-25"))
    assert engine.dynamic_watchlist[0] == "NVDA"
    assert "SPY" in engine.dynamic_watchlist            # pinned is preserved


def test_refresh_keeps_scanning_when_one_symbol_fails(tmp_path):
    cfg = _with_plan(tmp_path)
    engine = FakeEngine(frames={"NVDA": _bars(4500.0)}, watchlist=["NVDA", "BROKEN"])
    loop = _loop(engine, cfg)
    asyncio.run(loop._step_refresh(cfg, _et(2026, 9, 25, 9, 35), "2026-09-25"))
    plan = load_plan(cfg.DESK_DATA_DIR, "2026-09-25")
    symbols = [r.symbol for r in plan.in_play]
    assert "NVDA" in symbols and "BROKEN" in symbols
    broken = [r for r in plan.in_play if r.symbol == "BROKEN"][0]
    assert "missing:today_bars" in broken.reasons


def test_refresh_without_a_plan_does_nothing(tmp_path):
    cfg = _cfg(tmp_path)
    engine = FakeEngine(frames={"NVDA": _bars()}, watchlist=["NVDA"])
    loop = _loop(engine, cfg)
    asyncio.run(loop._step_refresh(cfg, _et(2026, 9, 25, 9, 35), "2026-09-25"))
    assert load_plan(cfg.DESK_DATA_DIR, "2026-09-25") is None
    assert [a for a in engine.audits if a[0] == "desk_inplay"] == []


# ────────────────────────────────────────────────────────────────────
# journal step
# ────────────────────────────────────────────────────────────────────

def test_journal_step_writes_the_file_and_audits(tmp_path):
    cfg = _with_plan(tmp_path)
    engine = FakeEngine(
        trades=[{"exit_time": "2026-09-25 11:00", "pnl_r": 0.7, "symbol": "NVDA",
                 "side": "long", "strategy": "mean_reversion", "exit_reason": "target"}],
        decisions=[{"component": "desk_limit", "action": "shadow",
                    "reason": "desk_max_trades", "symbol": "NVDA"}],
    )
    loop = _loop(engine, cfg)
    asyncio.run(loop._step_journal(cfg, _et(2026, 9, 25, 16, 5), "2026-09-25"))

    path = tmp_path / "desk" / "journal" / "2026-09-25.md"
    text = path.read_text()
    assert "## Plan" in text and "## Trades" in text and "## What to Change" in text
    assert "desk_max_trades" in text
    writes = [a for a in engine.audits if a[0] == "desk_journal"]
    assert len(writes) == 1 and writes[0][2] == "write"


def test_journal_step_writes_even_without_a_plan(tmp_path):
    cfg = _cfg(tmp_path)
    engine = FakeEngine()
    loop = _loop(engine, cfg)
    asyncio.run(loop._step_journal(cfg, _et(2026, 9, 25, 16, 5), "2026-09-25"))
    assert (tmp_path / "desk" / "journal" / "2026-09-25.md").exists()


# ────────────────────────────────────────────────────────────────────
# schedule
# ────────────────────────────────────────────────────────────────────

def test_tick_runs_the_day_in_order(tmp_path):
    cfg = _cfg(tmp_path)
    engine = FakeEngine(frames={"NVDA": _bars(4500.0)}, watchlist=["NVDA"])
    loop = _loop(engine, cfg)
    asyncio.run(loop.tick(_et(2026, 9, 25, 16, 5)))
    components = [a[0] for a in engine.audits]
    assert components == ["desk_plan", "desk_inplay", "desk_journal"]


def test_each_step_fires_once_per_date(tmp_path):
    cfg = _cfg(tmp_path)
    engine = FakeEngine(frames={"NVDA": _bars()}, watchlist=["NVDA"])
    loop = _loop(engine, cfg)
    asyncio.run(loop.tick(_et(2026, 9, 25, 9, 0)))
    asyncio.run(loop.tick(_et(2026, 9, 25, 9, 1)))
    assert [a[0] for a in engine.audits] == ["desk_plan"]
    asyncio.run(loop.tick(_et(2026, 9, 28, 9, 0)))       # Monday
    assert [a[0] for a in engine.audits] == ["desk_plan", "desk_plan"]


def test_nothing_fires_before_the_scheduled_time(tmp_path):
    cfg = _cfg(tmp_path)
    engine = FakeEngine()
    loop = _loop(engine, cfg)
    asyncio.run(loop.tick(_et(2026, 9, 25, 8, 59)))
    assert engine.audits == []


def test_weekends_are_skipped(tmp_path):
    cfg = _cfg(tmp_path)
    engine = FakeEngine()
    loop = _loop(engine, cfg)
    asyncio.run(loop.tick(_et(2026, 9, 26, 16, 5)))      # Saturday
    assert engine.audits == []
    assert load_plan(cfg.DESK_DATA_DIR, "2026-09-26") is None


def test_a_failing_step_never_escapes_the_tick(tmp_path, caplog):
    cfg = _cfg(tmp_path)
    engine = FakeEngine()

    class Boom(FakeEngine):
        @property
        def market_state(self):
            raise RuntimeError("engine state exploded")

        @market_state.setter
        def market_state(self, value):
            pass

    loop = _loop(Boom(), cfg)
    with caplog.at_level("WARNING"):
        asyncio.run(loop.tick(_et(2026, 9, 25, 9, 0)))   # must not raise
    assert "desk_loop step failed" in caplog.text


def test_run_stops_when_the_engine_stops(tmp_path):
    cfg = _cfg(tmp_path)
    engine = FakeEngine()
    engine.is_running = False
    loop = _loop(engine, cfg)
    asyncio.run(loop.run())                              # returns immediately
    assert engine.audits == []


def test_a_broken_config_never_escapes_the_tick(tmp_path):
    def boom():
        raise RuntimeError("config exploded")

    engine = FakeEngine()
    loop = DeskLoop(engine, config_fn=boom)
    asyncio.run(loop.tick(_et(2026, 9, 25, 9, 0)))       # must not raise
    assert engine.audits == []
