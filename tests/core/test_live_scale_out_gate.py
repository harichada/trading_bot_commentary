"""v-live-scale-out-gate-2026-11-11: SMCI incident replay.

Incident (2026-11-11 ET, SMCI long 73 @ 29.88, stop 29.43, TP 30.78):
  * 10:43:49 the manage-positions loop's "sim-path mirror" of the +1R
    scale-out ran on the LIVE position with no ENABLE_LIVE_SCALE_OUT gate:
    position.quantity 73 → 37, scaled_out=True, audit partial_exit_1R —
    but NO sell was sent (the block is bookkeeping-only). Schwab still 73.
  * 10:51:43 the trail-replace (order_monitor.brackets.
    replace_bracket_with_new_stop) cancelled the 73-share OCO and placed a
    37-share one, sized from the desynced local qty → 36 shares naked until
    Hari closed them by hand.

The engine is not constructible in tests, so (as in test_daily_loss_pause)
the real unbound engine methods are borrowed onto a stand-in and driven
for real; the brackets functions are driven for real against a fake Schwab
client. Nothing here talks to a broker.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from analysis.scale_trail_manager import ScaleTrailManager
from core.engine import TradingEngineWithCommentary as Engine
from core.models import Position, TradingMode
from core.order_monitor import brackets as br
from core.order_monitor import broker_flat as bf
from core.position_state import PositionState

_ENV = ("BRACKET_REPLACE_BROKER_QTY_GUARD",)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)
    from core.config import config_manager
    trading = config_manager.config.setdefault("trading", {})
    # ENABLE_LIVE_SCALE_OUT must read as its default (False) unless a test
    # sets it explicitly; never mutate a real yaml file.
    for k in ("enable_live_scale_out", "bracket_replace_broker_qty_guard"):
        if k in trading:
            monkeypatch.delitem(trading, k)

    async def _no_sleep(*_a, **_k):
        return None
    monkeypatch.setattr(br.asyncio, "sleep", _no_sleep)
    yield
    # asyncio.run() leaves no current loop; later legacy tests call
    # asyncio.get_event_loop(), so hand them a fresh one.
    asyncio.set_event_loop(asyncio.new_event_loop())


def _set_live_scale_out(monkeypatch, on: bool):
    from core.config import config_manager
    monkeypatch.setitem(config_manager.config.setdefault("trading", {}),
                        "enable_live_scale_out", on)


def _smci(mode="live", qty=73):
    p = Position(
        symbol="SMCI", entry_price=29.88, quantity=qty, side="long",
        stop_loss=29.88, take_profit=30.7764, entry_time=datetime.now(),
        original_stop=29.43, mode=mode, managed_by_bot=True,
    )
    # breakeven already lifted at 10:30 ET
    p.breakeven_lifted = True
    p.peak_favorable_r = 1.1
    p.state = PositionState.AT_BREAKEVEN.value
    p.bracket_order_id = "1008631825378"
    p.broker_stop_price = 30.0268
    return p


# ── stand-in engine ───────────────────────────────────────────────────

class _Commentary:
    def __init__(self):
        self.items = []

    def add_commentary(self, c):
        self.items.append(c)


class _Eng:
    _manage_positions_with_commentary = Engine._manage_positions_with_commentary
    _evaluate_exits_with_cached_quote = Engine._evaluate_exits_with_cached_quote
    _live_scale_out_block_reason = Engine._live_scale_out_block_reason
    _audit_scale_out_skipped_once = Engine._audit_scale_out_skipped_once
    _ensure_position_lock = Engine._ensure_position_lock

    def __init__(self, mode, positions, price):
        self.mode = mode
        self.positions = positions if mode != TradingMode.SIMULATION_WITH_COMMENTARY else {}
        self.simulated_positions = positions if mode == TradingMode.SIMULATION_WITH_COMMENTARY else {}
        self.audits = []
        self.commentary = _Commentary()
        self.scale_trail = ScaleTrailManager()
        self.manual_close_only = False
        self.loss_alert_tracker = {}
        self.schwab_client = MagicMock(name="schwab_client")
        self.closes = []
        self._price = price
        self.price_book = SimpleNamespace(get_mark=lambda s: SimpleNamespace(
            is_stale=False, has_price=True, price=self._price))
        quote = {"last": price}
        self.data_provider = SimpleNamespace(
            get_quote=lambda s: quote,
            get_market_data=lambda s: pd.DataFrame(),  # no ATR → no trail
        )

    def _audit(self, *args, **kw):
        self.audits.append((args, kw))

    def _is_auto_managed(self, position):
        return True

    async def _promote_zombie_if_stalled(self, position):
        return False

    async def _provider_call(self, fn, *a, timeout=None, site="", **kw):
        return fn(*a, **kw)

    async def _last_known_price(self, symbol):
        return self._price

    async def _close_position_with_commentary(self, position, reason, *a, **k):
        self.closes.append((position.symbol, reason))

    async def _close_real_position(self, *a, **k):
        self.closes.append(("real", a))

    def __getattr__(self, name):
        # Anything else the loops reach for after the scale-out block is
        # irrelevant to this test; make it an inert awaitable/no-op.
        async def _noop(*a, **k):
            return None
        return _noop

    def audit_events(self, event):
        return [a for a in self.audits if len(a[0]) >= 3 and a[0][2] == event]


PRICE_1R = 30.40   # > 29.88 + 1×0.45 → +1R reached


# ── 1. the manage-positions mirror is gated in LIVE ─────────────────────

def test_manage_loop_live_does_not_cut_qty_flag_off(monkeypatch):
    """SMCI replay, path 2: after the fix local qty stays 73, scaled_out
    stays False, no partial_exit_1R audit, no sell, skip audited."""
    _set_live_scale_out(monkeypatch, False)
    pos = _smci()
    eng = _Eng(TradingMode.LIVE, {"SMCI": pos}, PRICE_1R)
    asyncio.run(eng._manage_positions_with_commentary())

    assert pos.quantity == 73
    assert pos.scaled_out is False
    assert pos.state != PositionState.AT_1R.value
    assert not eng.audit_events("partial_exit_1R")
    skips = eng.audit_events("scale_out_skipped")
    assert skips and skips[0][0][3] == "live_partials_not_armed"
    assert skips[0][1].get("path") == "manage_positions"
    eng.schwab_client.place_order.assert_not_called()
    eng.schwab_client.cancel_order.assert_not_called()
    assert eng.closes == []


def test_manage_loop_skip_audit_is_deduped(monkeypatch):
    _set_live_scale_out(monkeypatch, False)
    pos = _smci()
    eng = _Eng(TradingMode.LIVE, {"SMCI": pos}, PRICE_1R)
    for _ in range(3):
        asyncio.run(eng._manage_positions_with_commentary())
    assert len(eng.audit_events("scale_out_skipped")) == 1
    assert pos.quantity == 73


def test_manage_loop_live_tagged_position_blocked_even_if_engine_mode_differs(monkeypatch):
    """Belt and braces: a position tagged mode='live' is never cut by the
    bookkeeping-only partial, whatever the engine mode label says."""
    _set_live_scale_out(monkeypatch, False)
    pos = _smci(mode="live")
    eng = _Eng(TradingMode.SIMULATION_WITH_COMMENTARY, {"SMCI": pos}, PRICE_1R)
    asyncio.run(eng._manage_positions_with_commentary())
    assert pos.quantity == 73 and pos.scaled_out is False


def test_manage_loop_sim_still_scales_out(monkeypatch):
    """Simulation behaviour is unchanged: bookkeeping partial still fires."""
    _set_live_scale_out(monkeypatch, False)
    pos = _smci(mode="simulation")
    eng = _Eng(TradingMode.SIMULATION_WITH_COMMENTARY, {"SMCI": pos}, PRICE_1R)
    asyncio.run(eng._manage_positions_with_commentary())
    assert pos.quantity == 37
    assert pos.scaled_out is True
    assert eng.audit_events("partial_exit_1R")
    assert not eng.audit_events("scale_out_skipped")


def test_manage_loop_live_flag_on_keeps_legacy_bookkeeping(monkeypatch):
    """Same guard as the position loop: with ENABLE_LIVE_SCALE_OUT on the
    (bookkeeping-only) partial runs. Documents why the flag must stay OFF."""
    _set_live_scale_out(monkeypatch, True)
    pos = _smci()
    eng = _Eng(TradingMode.LIVE, {"SMCI": pos}, PRICE_1R)
    asyncio.run(eng._manage_positions_with_commentary())
    assert pos.quantity == 37
    eng.schwab_client.place_order.assert_not_called()


# ── 2. the position-loop path (already gated) still is ─────────────────

def test_position_loop_live_flag_off_no_sell_no_cut(monkeypatch):
    _set_live_scale_out(monkeypatch, False)
    pos = _smci()
    eng = _Eng(TradingMode.LIVE, {"SMCI": pos}, PRICE_1R)
    asyncio.run(eng._evaluate_exits_with_cached_quote(pos))
    assert pos.quantity == 73 and pos.scaled_out is False
    assert not eng.audit_events("partial_exit_1R")
    skips = eng.audit_events("scale_out_skipped")
    assert skips and skips[0][0][3] == "live_partials_not_armed"
    eng.schwab_client.place_order.assert_not_called()
    assert eng.closes == []


def test_position_loop_sim_still_scales_out(monkeypatch):
    _set_live_scale_out(monkeypatch, False)
    pos = _smci(mode="simulation")
    eng = _Eng(TradingMode.SIMULATION_WITH_COMMENTARY, {"SMCI": pos}, PRICE_1R)
    asyncio.run(eng._evaluate_exits_with_cached_quote(pos))
    assert pos.quantity == 37 and pos.scaled_out is True


def test_block_reason_helper():
    e = _Eng(TradingMode.LIVE, {}, 1.0)
    assert e._live_scale_out_block_reason(_smci()) == "live_partials_not_armed"
    s = _Eng(TradingMode.SIMULATION_WITH_COMMENTARY, {}, 1.0)
    assert s._live_scale_out_block_reason(_smci(mode="simulation")) is None
    assert s._live_scale_out_block_reason(_smci(mode="live")) == "live_partials_not_armed"


# ── 3. trail-replace sizes from broker qty, fails closed ───────────────

class _Resp:
    def __init__(self, status, payload=None, location=None):
        self.status_code = status
        self._p = payload or {}
        self.headers = {"Location": location} if location else {}

    def json(self):
        return self._p


class _FakeSchwab:
    """Records every order call. get_account/get_order are reads."""

    class Account:
        class Fields:
            POSITIONS = "positions"

    def __init__(self, broker_qty=73, account_ok=True, bracket_qty=73,
                 order_ok=True, short_qty=0):
        self.broker_qty = broker_qty
        self.short_qty = short_qty
        self.account_ok = account_ok
        self.bracket_qty = bracket_qty
        self.order_ok = order_ok
        self.cancelled = []
        self.placed = []

    def get_account(self, account_hash, fields=None):
        if self.account_ok is None:
            raise RuntimeError("schwab 502")
        if not self.account_ok:
            return _Resp(500)
        positions = []
        if self.broker_qty or self.short_qty:
            positions.append({"instrument": {"symbol": "SMCI"},
                              "longQuantity": self.broker_qty,
                              "shortQuantity": self.short_qty})
        positions.append({"instrument": {"symbol": "HQGE"},
                          "longQuantity": 5100, "shortQuantity": 0})
        return _Resp(200, {"securitiesAccount": {"positions": positions}})

    def get_order(self, order_id, account_hash):
        if not self.order_ok or self.bracket_qty is None:
            return _Resp(404)
        return _Resp(200, {"orderStrategyType": "OCO", "childOrderStrategies": [
            {"orderType": "LIMIT", "quantity": self.bracket_qty},
            {"orderType": "STOP_LIMIT", "quantity": self.bracket_qty,
             "stopPrice": 30.0268},
        ]})

    def cancel_order(self, order_id, account_hash):
        self.cancelled.append(order_id)
        return _Resp(200)

    def place_order(self, account_hash, order):
        self.placed.append(order)
        return _Resp(201, location="/orders/1008632442515")


def _oco_leg_qtys(order):
    return [c["orderLegCollection"][0]["quantity"]
            for c in order["childOrderStrategies"]]


def _bracket_engine(client, mode=TradingMode.LIVE):
    eng = SimpleNamespace(
        mode=mode, schwab_client=client, account_hash="HASH",
        account_id="ACCT", commentary=_Commentary(), audits=[],
    )
    eng._audit = lambda *a, **k: eng.audits.append((a, k))
    return eng


def _events(eng, event):
    return [a for a in eng.audits if len(a[0]) >= 3 and a[0][2] == event]


def test_smci_replace_before_fix_shape_would_undersize():
    """Reproduce the bug shape: local qty desynced to 37 while broker holds
    73. Legacy (guard off) sizing places a 37-share OCO → 36 naked."""
    import os
    os.environ["BRACKET_REPLACE_BROKER_QTY_GUARD"] = "0"
    try:
        client = _FakeSchwab(broker_qty=73, bracket_qty=73)
        eng = _bracket_engine(client)
        pos = _smci(qty=37)
        asyncio.run(br.replace_bracket_with_new_stop(eng, pos, 30.1768))
        assert client.cancelled == ["1008631825378"]
        assert _oco_leg_qtys(client.placed[0]) == [37, 37]   # the bug
    finally:
        os.environ.pop("BRACKET_REPLACE_BROKER_QTY_GUARD", None)


def test_smci_replace_sized_to_broker_qty():
    """After the fix: the same desync is sized to broker qty 73."""
    client = _FakeSchwab(broker_qty=73, bracket_qty=73)
    eng = _bracket_engine(client)
    pos = _smci(qty=37)
    asyncio.run(br.replace_bracket_with_new_stop(eng, pos, 30.1768))
    assert client.cancelled == ["1008631825378"]
    assert len(client.placed) == 1
    assert _oco_leg_qtys(client.placed[0]) == [73, 73]
    assert pos.quantity == 73
    assert pos.bracket_order_id == "1008632442515"
    assert _events(eng, "qty_desync")
    rep = _events(eng, "bracket_replaced")
    assert rep and rep[0][1]["qty"] == 73


def test_full_smci_sequence_after_fix(monkeypatch):
    """End to end: manage loop at +1R (gated) then trail replace → local
    stays 73, replacement 73 shares, matching broker."""
    _set_live_scale_out(monkeypatch, False)
    pos = _smci(qty=73)
    eng = _Eng(TradingMode.LIVE, {"SMCI": pos}, PRICE_1R)
    asyncio.run(eng._manage_positions_with_commentary())
    assert pos.quantity == 73
    client = _FakeSchwab(broker_qty=73, bracket_qty=73)
    beng = _bracket_engine(client)
    asyncio.run(br.replace_bracket_with_new_stop(beng, pos, 30.1768))
    assert _oco_leg_qtys(client.placed[0]) == [73, 73]
    assert not _events(beng, "qty_desync")


@pytest.mark.parametrize("account_ok", [False, None])
def test_replace_broker_qty_unreadable_keeps_old_oco(account_ok):
    client = _FakeSchwab(account_ok=account_ok)
    eng = _bracket_engine(client)
    pos = _smci(qty=37)
    asyncio.run(br.replace_bracket_with_new_stop(eng, pos, 30.1768))
    assert client.cancelled == [] and client.placed == []
    assert pos.bracket_order_id == "1008631825378"
    assert pos.quantity == 37
    sk = _events(eng, "bracket_replace_skipped")
    assert sk and sk[0][0][3] == "broker_qty_unavailable"


def test_replace_broker_more_than_local_and_old_oco_unreadable_keeps_old_oco():
    client = _FakeSchwab(broker_qty=73, bracket_qty=None)
    eng = _bracket_engine(client)
    pos = _smci(qty=37)
    asyncio.run(br.replace_bracket_with_new_stop(eng, pos, 30.1768))
    assert client.cancelled == [] and client.placed == []
    assert _events(eng, "bracket_replace_skipped")[0][0][3] == "broker_qty_ambiguous"


def test_replace_never_grows_onto_manual_shares():
    """Account holds 100 (94 manual + 6 bot); bot OCO protected 6 → new
    OCO stays 6 (not 100)."""
    client = _FakeSchwab(broker_qty=100, bracket_qty=6)
    eng = _bracket_engine(client)
    pos = _smci(qty=6)
    asyncio.run(br.replace_bracket_with_new_stop(eng, pos, 30.1768))
    assert _oco_leg_qtys(client.placed[0]) == [6, 6]
    assert pos.quantity == 6


def test_replace_broker_fewer_than_local_sizes_down():
    client = _FakeSchwab(broker_qty=50, bracket_qty=73)
    eng = _bracket_engine(client)
    pos = _smci(qty=73)
    asyncio.run(br.replace_bracket_with_new_stop(eng, pos, 30.1768))
    assert _oco_leg_qtys(client.placed[0]) == [50, 50]
    assert pos.quantity == 50


def test_replace_broker_flat_does_not_cancel():
    client = _FakeSchwab(broker_qty=0)
    eng = _bracket_engine(client)
    pos = _smci(qty=73)
    asyncio.run(br.replace_bracket_with_new_stop(eng, pos, 30.1768))
    assert client.cancelled == [] and client.placed == []
    assert _events(eng, "bracket_replace_skipped")[0][0][3] == "broker_flat_or_wrong_side"


def test_replace_short_position_never_gets_sell_oco():
    client = _FakeSchwab(broker_qty=0, short_qty=40)
    eng = _bracket_engine(client)
    pos = _smci(qty=40)
    pos.side = "short"
    asyncio.run(br.replace_bracket_with_new_stop(eng, pos, 30.1768))
    assert client.cancelled == [] and client.placed == []
    assert _events(eng, "bracket_replace_skipped")[0][0][3] == "short_replace_not_supported"


def test_replace_non_live_mode_is_legacy():
    client = _FakeSchwab(broker_qty=73)
    eng = _bracket_engine(client, mode=TradingMode.SIMULATION_WITH_COMMENTARY)
    pos = _smci(qty=37)
    asyncio.run(br.replace_bracket_with_new_stop(eng, pos, 30.1768))
    assert _oco_leg_qtys(client.placed[0]) == [37, 37]


# ── 4. re_bracket: read error is NOT "flat" ─────────────────────────────

def test_re_bracket_read_error_aborts_without_flat_handling(monkeypatch):
    called = []

    async def _flat(*a, **k):
        called.append(a)
    monkeypatch.setattr(br, "handle_broker_flat_detected", _flat)
    client = _FakeSchwab(account_ok=None)
    eng = _bracket_engine(client)
    pos = _smci(qty=73)
    asyncio.run(br.re_bracket_position(eng, pos))
    assert called == []
    assert client.placed == [] and client.cancelled == []
    assert pos.managed_by_bot is True
    assert _events(eng, "re_bracket_aborted")[0][0][3] == "broker_qty_unavailable"


def test_re_bracket_sizes_to_broker_qty(monkeypatch):
    client = _FakeSchwab(broker_qty=73)
    eng = _bracket_engine(client)
    pos = _smci(qty=37)
    pos.bracket_order_id = None
    asyncio.run(br.re_bracket_position(eng, pos))
    assert _oco_leg_qtys(client.placed[0]) == [73, 73]
    assert pos.quantity == 73


def test_re_bracket_broker_flat_still_routes_to_flat_handler(monkeypatch):
    called = []

    async def _flat(*a, **k):
        called.append(a)
    monkeypatch.setattr(br, "handle_broker_flat_detected", _flat)
    client = _FakeSchwab(broker_qty=0)
    eng = _bracket_engine(client)
    asyncio.run(br.re_bracket_position(eng, _smci(qty=73)))
    assert len(called) == 1 and client.placed == []


# ── 5. read helper + config ─────────────────────────────────────────────

def test_read_broker_position_qty_semantics():
    eng = _bracket_engine(_FakeSchwab(broker_qty=73))
    assert asyncio.run(bf.read_broker_position_qty(eng, "SMCI")) == 73
    assert asyncio.run(bf.read_broker_position_qty(eng, "NVDA")) == 0
    eng = _bracket_engine(_FakeSchwab(broker_qty=0, short_qty=40))
    assert asyncio.run(bf.read_broker_position_qty(eng, "SMCI")) == -40
    eng = _bracket_engine(_FakeSchwab(account_ok=False))
    assert asyncio.run(bf.read_broker_position_qty(eng, "SMCI")) is None
    eng = _bracket_engine(_FakeSchwab(account_ok=None))
    assert asyncio.run(bf.read_broker_position_qty(eng, "SMCI")) is None
    eng = _bracket_engine(None)
    assert asyncio.run(bf.read_broker_position_qty(eng, "SMCI")) is None


def test_config_defaults(monkeypatch):
    from core.config import Config
    assert Config().ENABLE_LIVE_SCALE_OUT is False
    assert Config().BRACKET_REPLACE_BROKER_QTY_GUARD is True
    monkeypatch.setenv("BRACKET_REPLACE_BROKER_QTY_GUARD", "off")
    assert Config().BRACKET_REPLACE_BROKER_QTY_GUARD is False
