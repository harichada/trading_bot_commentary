"""v-daily-loss-pause-entries-2026-11-05: the daily-loss circuit pauses NEW
entries only.

Incident 2026-11-05 (10:12 + 11:56 ET): the 5% emergency stop fired on an
account day P&L of ~-$7.5k (limit $1,250, 0 bot fills), set is_running=False,
every loop exited and the watchdog ran out of restarts.

The engine is not constructible in tests, so — as in
tests/core/desk/test_desk_hooks.py — the real unbound engine methods are
borrowed onto a small stand-in object and driven for real (the analysis
loop body runs actual cycles).
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import core.engine as engine_mod
from core.config import Config
from core.engine import TradingEngineWithCommentary as Engine
from core.models import SignalType, TradingMode
from risk.daily_loss_gate import DailyLossGate
from risk.manager import RiskManagerWithCommentary

_ENV = (
    "DAILY_LOSS_ACTION", "DAILY_LOSS_BOT_MANAGED_ONLY",
    "ENABLE_BOT_ONLY_PNL_CIRCUIT", "MAX_DAILY_LOSS", "BOT_CAPITAL_ALLOCATION",
    "BLOCK_NEW_ENTRIES", "ENABLE_DAILY_REALIZED_LOSS_HALT",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)
    from core.config import config_manager
    risk = config_manager.config.setdefault("risk", {})
    for k in ("daily_loss_action", "daily_loss_bot_managed_only"):
        if k in risk:
            monkeypatch.delitem(risk, k)
    # incident shape: account-wide measure, 5% limit on $25k = $1,250
    monkeypatch.setenv("ENABLE_BOT_ONLY_PNL_CIRCUIT", "0")
    monkeypatch.setenv("MAX_DAILY_LOSS", "0.05")
    yield


# ── stand-in engine ───────────────────────────────────────────────────

class _Bus:
    def __init__(self):
        self._ev = asyncio.Event()
        self._ev.set()           # every cadence wait returns immediately

    def get_wake_event(self):
        return self._ev

    def clear_wake_event(self):
        pass


def _pos(symbol, managed=True, upnl=0.0, side="long"):
    return SimpleNamespace(
        symbol=symbol, managed_by_bot=managed, is_external=not managed,
        is_manually_managed=False, is_long_term=False, unrealized_pnl=upnl,
        quantity=10, current_price=10.0, entry_price=10.0, side=side,
    )


class _Eng:
    _get_daily_loss_gate = Engine._get_daily_loss_gate
    _trip_daily_loss_pause = Engine._trip_daily_loss_pause
    _daily_loss_entries_paused = Engine._daily_loss_entries_paused
    _daily_loss_status = Engine._daily_loss_status
    _check_emergency_daily_loss = Engine._check_emergency_daily_loss
    _analysis_loop_body = Engine._analysis_loop_body
    _evaluate_trading_conditions = Engine._evaluate_trading_conditions
    _process_signal_with_commentary = Engine._process_signal_with_commentary
    _execute_real_trade = Engine._execute_real_trade
    _position_loop = Engine._position_loop

    def __init__(self, day_pnl=-7500.0, stop_after=12, positions=None):
        self.mode = TradingMode.LIVE
        self.is_running = True
        self._daily_loss_gate = DailyLossGate(state_path=None)
        self.circuit_halted = False
        self.circuit_halt_reason = None
        self.commentary = MagicMock()
        self.risk_manager = RiskManagerWithCommentary(25000.0, self.commentary)
        self.risk_manager.buying_power = 50000.0
        self.schwab_client = object()
        self.account_id = "acct"
        self.allow_premarket = False
        self.allow_afterhours = False
        self.manual_close_only = False
        self.positions = positions if positions is not None else {}
        self.simulated_positions = {}
        self.trade_history = []
        self.db_logger = None
        self.data_provider = None
        self.ml_predictor = None
        self._news_bus = _Bus()
        self._last_bp_check = datetime.now()
        self.brain = MagicMock()
        self.day_pnl = day_pnl
        self.stop_after = stop_after
        self.calls = {k: 0 for k in (
            "account_info", "analyze", "manage", "order_status",
            "update_prices", "exits")}
        self.closed = []
        self.audits = []

    # sync helpers
    def is_market_hours(self):
        return True, "regular"

    def _in_warmup_window(self):
        return False

    def _get_next_market_open(self):
        return "09:30"

    def _get_active_blackout(self):
        return None

    def _audit(self, component, symbol, action, reason, /, **details):
        self.audits.append((component, symbol, action, reason, details))

    def _compute_bot_daily_pnl(self):
        return 0.0          # 0 bot fills, as in the incident

    def _compute_bot_open_notional(self):
        return 0.0

    def _save_state(self):
        pass

    def _symbols_with_open_positions(self):
        return list(self.positions)

    # async helpers
    async def _get_real_account_info(self):
        self.calls["account_info"] += 1
        return {"day_pnl": self.day_pnl, "buying_power": 50000.0, "balance": 25000.0}

    async def sync_positions_with_schwab(self):
        pass

    async def get_schwab_positions(self):
        return []

    async def _update_position_prices(self):
        self.calls["update_prices"] += 1

    async def _assess_market_conditions(self):
        pass

    async def _analyze_premarket_gaps(self):
        pass

    async def _analyze_markets_with_commentary(self):
        self.calls["analyze"] += 1

    async def _manage_positions_with_commentary(self):
        # exits / stops / re-brackets live here and in _position_loop
        self.calls["manage"] += 1
        if self.calls["manage"] >= self.stop_after:
            self.is_running = False

    async def _check_order_status(self):
        self.calls["order_status"] += 1

    async def _update_real_positions(self):
        pass

    async def _check_buying_power(self, _):
        pass

    async def _close_position_with_commentary(self, position, reason):
        self.closed.append((position.symbol, reason))

    async def _evaluate_exits_with_cached_quote(self, pos):
        self.calls["exits"] += 1

    async def _shadow_meta_evaluate(self, signal):
        raise AssertionError("router went past the daily-loss gate")

    class price_book:  # noqa: N801
        @staticmethod
        async def register_interest(*a, **k):
            return None


def _signal(sym="NVDA", side=SignalType.BUY, strategy="day_trade_momentum"):
    return SimpleNamespace(
        symbol=sym, signal_type=side, confidence=0.9, strength=0.9,
        reasoning={"strategy": strategy}, entry_price=100.0,
        stop_loss=98.0, take_profit=104.0, position_size=10,
    )


def _run_loop(eng, monkeypatch):
    monkeypatch.setattr(engine_mod, "log_performance", lambda **k: None)
    asyncio.run(asyncio.wait_for(eng._analysis_loop_body(), timeout=60))


# ── config flags ──────────────────────────────────────────────────────

def test_daily_loss_action_defaults_to_pause_entries():
    assert Config().DAILY_LOSS_ACTION == "pause_entries"


def test_daily_loss_action_yaml_and_env_override(monkeypatch):
    from core.config import config_manager
    monkeypatch.setitem(config_manager.config.setdefault("risk", {}),
                        "daily_loss_action", "halt")
    assert Config().DAILY_LOSS_ACTION == "halt"
    monkeypatch.setenv("DAILY_LOSS_ACTION", "pause_entries")
    assert Config().DAILY_LOSS_ACTION == "pause_entries"     # env wins
    monkeypatch.setenv("DAILY_LOSS_ACTION", "HALT")
    assert Config().DAILY_LOSS_ACTION == "halt"


def test_unknown_daily_loss_action_falls_back_to_pause(monkeypatch):
    monkeypatch.setenv("DAILY_LOSS_ACTION", "flatten_everything")
    assert Config().DAILY_LOSS_ACTION == "pause_entries"


def test_bot_managed_only_flag_default_off_and_overrides(monkeypatch):
    assert Config().DAILY_LOSS_BOT_MANAGED_ONLY is False
    from core.config import config_manager
    monkeypatch.setitem(config_manager.config.setdefault("risk", {}),
                        "daily_loss_bot_managed_only", True)
    assert Config().DAILY_LOSS_BOT_MANAGED_ONLY is True
    monkeypatch.setenv("DAILY_LOSS_BOT_MANAGED_ONLY", "0")
    assert Config().DAILY_LOSS_BOT_MANAGED_ONLY is False


# ── the gate ──────────────────────────────────────────────────────────

def test_gate_latches_once_and_clears_at_et_rollover():
    now = {"t": datetime(2026, 11, 5, 10, 12)}
    gate = DailyLossGate(state_path=None, clock=lambda: now["t"])
    assert gate.is_paused() == (False, None)
    assert gate.trip("emergency_stop", pnl=-7500) is True
    assert gate.trip("daily_loss_limit") is False          # already latched
    assert gate.is_paused() == (True, "emergency_stop")
    now["t"] = datetime(2026, 11, 6, 9, 0)
    assert gate.is_paused() == (False, None)


def test_gate_survives_a_same_day_restart_only(tmp_path):
    path = tmp_path / "daily_loss_pause.json"
    day1 = lambda: datetime(2026, 11, 5, 10, 12)  # noqa: E731
    DailyLossGate(state_path=str(path), clock=day1).trip("emergency_stop")
    assert json.loads(path.read_text())["reason"] == "emergency_stop"
    assert DailyLossGate(state_path=str(path), clock=day1).is_paused() == (True, "emergency_stop")
    day2 = lambda: datetime(2026, 11, 6, 9, 0)  # noqa: E731
    assert DailyLossGate(state_path=str(path), clock=day2).is_paused() == (False, None)


# ── risk manager ──────────────────────────────────────────────────────

def test_check_trading_allowed_no_longer_stops_analysis_in_pause_mode():
    rm = RiskManagerWithCommentary(25000.0, MagicMock())
    rm.buying_power = 50000.0
    rm.schwab_daily_pnl = -7500.0
    rm.consecutive_losses = 1
    breached, _, pnl, limit = rm.daily_loss_limit_breach()
    assert breached and pnl == -7500.0 and limit == pytest.approx(1250.0)
    assert rm.check_trading_allowed() == (True, "Trading allowed")
    assert rm.consecutive_losses == 1                          # untouched


def test_check_trading_allowed_legacy_halt_still_refuses(monkeypatch):
    monkeypatch.setenv("DAILY_LOSS_ACTION", "halt")
    rm = RiskManagerWithCommentary(25000.0, MagicMock())
    rm.buying_power = 50000.0
    rm.schwab_daily_pnl = -7500.0
    ok, why = rm.check_trading_allowed()
    assert ok is False and "Daily loss limit exceeded" in why


def test_bot_managed_only_flag_measures_bot_pnl(monkeypatch):
    rm = RiskManagerWithCommentary(25000.0, MagicMock())
    rm.schwab_daily_pnl = -7500.0      # hands-off / manual losses
    rm.bot_daily_pnl = -100.0
    assert rm.daily_loss_limit_breach()[0] is True             # account measure
    monkeypatch.setenv("DAILY_LOSS_BOT_MANAGED_ONLY", "1")
    assert rm.daily_loss_uses_bot_pnl() is True
    assert rm.daily_loss_circuit_pnl() == (-100.0, "Bot daily loss")
    assert rm.daily_loss_limit_breach()[0] is False


def test_bot_daily_pnl_excludes_hands_off_and_manual_positions():
    stub = SimpleNamespace(
        trade_history=[
            {"symbol": "HQGE", "exit_time": datetime.now().isoformat(), "pnl": -5000},
            {"symbol": "NVDA", "exit_time": datetime.now().isoformat(), "pnl": -50},
        ],
        positions={
            "SPCX": _pos("SPCX", managed=True, upnl=-2000),
            "PINS": _pos("PINS", managed=False, upnl=-400),
            "AAPL": _pos("AAPL", managed=True, upnl=-25),
        },
    )
    assert Engine._compute_bot_daily_pnl(stub) == pytest.approx(-75.0)


# ── the incident, replayed through the real analysis loop body ────────

def test_circuit_trips_but_every_loop_keeps_ticking(monkeypatch):
    eng = _Eng(day_pnl=-7500.0, stop_after=12,
               positions={"AAPL": _pos("AAPL"), "HQGE": _pos("HQGE", managed=False)})
    _run_loop(eng, monkeypatch)

    # the loop ran every cycle we asked for — it was never halted
    assert eng.calls["manage"] == 12            # position management / exits each cycle
    assert eng.calls["analyze"] == 12           # market analysis never skipped
    assert eng.calls["order_status"] == 12      # order/stop/bracket tracking
    assert eng.calls["update_prices"] == 12
    assert eng.circuit_halted is False
    # P&L kept refreshing after the trip (cycles 5 and 10: sync + check)
    assert eng.calls["account_info"] >= 4
    assert eng.risk_manager.schwab_daily_pnl == -7500.0
    # tripped, latched, nothing closed
    paused, reason = eng._daily_loss_entries_paused()
    assert paused is True and reason in ("emergency_stop", "daily_loss_limit")
    assert eng.closed == []
    st = eng._daily_loss_status()
    assert st["entries_paused"] is True and st["entries_paused_reason"] == reason
    assert st["circuit_halted"] is False and st["daily_loss_action"] == "pause_entries"
    assert any(a[2] == "entries_paused" for a in eng.audits)


def test_emergency_stop_in_pause_mode_keeps_is_running(monkeypatch):
    eng = _Eng(positions={"AAPL": _pos("AAPL")})
    eng.risk_manager.schwab_daily_pnl = -7500.0
    out = asyncio.run(eng._check_emergency_daily_loss(-7500.0))
    assert out == "ok"
    assert eng.is_running is True
    assert eng._get_daily_loss_gate().is_paused() == (True, "emergency_stop")
    assert eng.closed == []


def test_emergency_false_alarm_still_aborts(monkeypatch):
    eng = _Eng(day_pnl=+50.0)
    out = asyncio.run(eng._check_emergency_daily_loss(-7500.0))
    assert out == "abort"
    assert eng._get_daily_loss_gate().is_paused() == (False, None)


@pytest.mark.parametrize("side,strategy", [
    (SignalType.BUY, "day_trade_momentum"),
    (SignalType.SELL, "day_trade_short"),
    (SignalType.BUY, "mean_reversion"),
    (SignalType.BUY, "swing_lane"),
    (SignalType.BUY, "orb"),
])
def test_every_new_entry_is_blocked_while_paused(side, strategy):
    eng = _Eng()
    eng._get_daily_loss_gate().trip("emergency_stop")
    asyncio.run(eng._process_signal_with_commentary(_signal(side=side, strategy=strategy), "HOLD", {}))
    skips = [a for a in eng.audits if a[2] == "skip"]
    assert skips and skips[-1][3] == "daily_loss_entries_paused"


def test_adds_to_an_existing_position_are_blocked_too():
    eng = _Eng(positions={"NVDA": _pos("NVDA")})
    eng._get_daily_loss_gate().trip("daily_loss_limit")
    asyncio.run(eng._process_signal_with_commentary(_signal("NVDA"), "HOLD", {}))
    assert eng.audits[-1][3] == "daily_loss_entries_paused"


def test_execute_real_trade_refuses_while_paused_without_touching_broker():
    eng = _Eng()
    eng.data_provider = SimpleNamespace(get_quote=lambda *_: (_ for _ in ()).throw(
        AssertionError("broker/quote touched")))
    eng._get_daily_loss_gate().trip("emergency_stop")
    assert asyncio.run(eng._execute_real_trade(_signal())) is False
    assert eng.audits[-1][:4] == ("order_entry", "NVDA", "skip", "daily_loss_entries_paused")


def test_router_trips_live_on_the_limit_even_between_refreshes():
    eng = _Eng()
    eng.risk_manager.schwab_daily_pnl = -1300.0     # past $1,250
    asyncio.run(eng._process_signal_with_commentary(_signal(), "HOLD", {}))
    assert eng._get_daily_loss_gate().is_paused() == (True, "daily_loss_limit")
    assert eng.audits[-1][3] == "daily_loss_entries_paused"


def test_position_loop_keeps_running_exits_while_paused():
    eng = _Eng(positions={"AAPL": _pos("AAPL"), "TSLA": _pos("TSLA", side="short")})
    eng._get_daily_loss_gate().trip("emergency_stop")

    async def main():
        task = asyncio.create_task(eng._position_loop())
        for _ in range(200):
            await asyncio.sleep(0.01)
            if eng.calls["exits"] >= 4:
                break
        eng.is_running = False
        await asyncio.wait_for(task, timeout=10)

    asyncio.run(main())
    assert eng.calls["exits"] >= 4


def test_heartbeat_predicate_stays_true_while_paused():
    eng = _Eng()
    asyncio.run(eng._check_emergency_daily_loss(-7500.0))
    assert eng._get_daily_loss_gate().is_paused()[0] is True
    assert eng.is_running is True       # heartbeat runs `while is_running()`


# ── hands-off positions are never touched ─────────────────────────────

def test_legacy_halt_still_halts_but_never_touches_hands_off(monkeypatch):
    monkeypatch.setenv("DAILY_LOSS_ACTION", "halt")
    eng = _Eng(day_pnl=-7500.0, stop_after=50, positions={
        "AAPL": _pos("AAPL"), "HQGE": _pos("HQGE"), "SPCX": _pos("SPCX")})
    _run_loop(eng, monkeypatch)
    assert eng.is_running is False
    assert eng.circuit_halted is True
    assert eng.calls["manage"] == 5             # broke out on the first P&L check
    closed = {s for s, _ in eng.closed}
    assert closed == {"AAPL"}                   # HQGE / SPCX never closed
    st = eng._daily_loss_status()
    assert st["circuit_halted"] is True and st["entries_paused"] is False


def test_pause_mode_closes_nothing_including_hands_off(monkeypatch):
    eng = _Eng(day_pnl=-7500.0, stop_after=6, positions={
        "HQGE": _pos("HQGE"), "SPCX": _pos("SPCX"), "AAPL": _pos("AAPL")})
    _run_loop(eng, monkeypatch)
    assert eng.closed == []


def test_legacy_halt_router_is_not_gated(monkeypatch):
    """In halt mode the new gate stays out of the way (legacy paths decide)."""
    monkeypatch.setenv("DAILY_LOSS_ACTION", "halt")
    eng = _Eng()
    eng._get_daily_loss_gate().trip("emergency_stop")
    assert eng._daily_loss_entries_paused() == (False, None)


# ── /api/status + watchdog ────────────────────────────────────────────

def test_api_status_exposes_entries_paused(monkeypatch):
    import api.routes as routes
    eng = _Eng()
    eng._get_daily_loss_gate().trip("emergency_stop")
    eng.start_time = datetime.now() - timedelta(minutes=3)
    monkeypatch.setattr(routes, "trading_engine", eng, raising=False)
    body = asyncio.run(routes.get_bot_status())
    assert body["bot_running"] is True
    assert body["entries_paused"] is True
    assert body["entries_paused_reason"] == "emergency_stop"
    assert body["circuit_halted"] is False


class _Stub(BaseHTTPRequestHandler):
    payload: dict = {}

    def do_GET(self):  # noqa: N802
        b = json.dumps(self.payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    do_POST = do_GET

    def log_message(self, *a):
        return


def _watchdog(payload, tmp_path):
    handler = type("_H", (_Stub,), {"payload": payload})
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        env = dict(os.environ)
        env.update({
            "TRADING_API_KEY": "sentinel", "TRADING_BOT_PORT": str(srv.server_address[1]),
            "WATCHDOG_HOST": "127.0.0.1", "WATCHDOG_CURL_TIMEOUT": "2",
            "WATCHDOG_REQUIRE_LIVE": "1",
            "WATCHDOG_LOG": str(tmp_path / "wd.log"), "WATCHDOG_STATE": str(tmp_path / "wd.state"),
            "WATCHDOG_LOCK": str(tmp_path / "wd.lock"), "WATCHDOG_LASTRUN": str(tmp_path / "wd.lastrun"),
        })
        return subprocess.run(["bash", "scripts/bot_watchdog.sh", "--dry-run"],
                              capture_output=True, text=True, timeout=60, env=env)
    finally:
        srv.shutdown()
        srv.server_close()


def test_watchdog_treats_entries_paused_as_healthy(tmp_path):
    done = _watchdog({"bot_running": True, "mode": "live", "entries_paused": True,
                      "entries_paused_reason": "emergency_stop"}, tmp_path)
    out = done.stdout + done.stderr
    assert done.returncode == 0
    assert "health ok" in out and "entries_paused=emergency_stop" in out
    assert "restart" not in out


def test_watchdog_never_restarts_a_legacy_circuit_halt(tmp_path):
    done = _watchdog({"bot_running": False, "mode": "live", "circuit_halted": True}, tmp_path)
    out = done.stdout + done.stderr
    assert done.returncode == 0
    assert "deliberately HALTED by the daily-loss circuit" in out
    assert "would restart" not in out and "health FAILED" not in out


def test_watchdog_still_restarts_a_real_stop(tmp_path):
    done = _watchdog({"bot_running": False, "mode": "live", "circuit_halted": False}, tmp_path)
    out = done.stdout + done.stderr
    assert "reason=bot_not_running" in out and done.returncode == 1
