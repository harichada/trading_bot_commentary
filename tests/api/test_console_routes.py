"""Tests for the Risk Console API (api/console.py):
`GET /api/console`, `PUT /api/console/{key}`, `POST /api/console/block-entries`,
`DELETE /api/console/pending/{pending_id}`, `GET /api/console/audit`.

TDD: written BEFORE api/console.py exists. Client construction follows
`_make_client` in tests/api/test_desk_routes.py (import api.routes first,
delenv TRADING_API_KEY, DESK_DATA_DIR -> tmp_path, trading_engine = None).
`FakeManager` is copied from tests/core/test_settings_console_store.py
(Task 2). `api.console._manager` / `_cfg` / `_now` are monkeypatched per
test so no route ever falls through to the real
`core.config.config_manager`.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from core.settings_console import KNOBS

ET = ZoneInfo("America/New_York")


class FakeManager:
    def __init__(self, data=None):
        self.data = dict(data or {})
        self.updates = []

    def get(self, key, default=None):
        return self.data.get(key, default)

    def update(self, key, value):
        self.data[key] = value
        self.updates.append((key, value))


def _fake_cfg(tmp_path):
    return SimpleNamespace(
        DESK_DATA_DIR=str(tmp_path),
        BLOCK_NEW_ENTRIES=False,
        SETTINGS_PROMOTE_TIME_ET="09:25",
        RISK_PER_TRADE_PCT=0.01,
        BOT_CAPITAL_ALLOCATION=5000.0,
        MAX_POSITIONS=5,
        MAX_DAILY_LOSS=0.03,
        SWING_MAX_HOLD_SESSIONS=10,
        ORB_LIVE_ENTRIES_ENABLED=True,
        DAY_TRADE_LIVE_ENTRIES_ENABLED=True,
        DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED=False,
        MEAN_REV_LIVE_ENTRIES_ENABLED=True,
        FIT_LANE_LIVE_ENTRIES_ENABLED=("mr_20d", "mom_12_1"),
        FIT_LANE_LIVE_ALLOWLIST=("mr_20d",),
        FIT_LANES_ENABLED=("mr_20d",),
        ENABLE_PLAYBOOK_GATE=True,
        ENABLE_DESK_LIMITS=False,
    )


def _mon(h=10, m=0):  # Monday 2026-09-28, 10:00 ET
    return datetime(2026, 9, 28, h, m, tzinfo=ET)


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    for knob in KNOBS:
        if knob.env_name:
            monkeypatch.delenv(knob.env_name, raising=False)


@pytest.fixture
def manager():
    return FakeManager()


@pytest.fixture
def client(tmp_path, monkeypatch, manager):
    import api.console as console
    import api.routes as routes

    monkeypatch.delenv("TRADING_API_KEY", raising=False)
    monkeypatch.setattr(routes, "trading_engine", None)
    monkeypatch.setenv("DESK_DATA_DIR", str(tmp_path))

    cfg = _fake_cfg(tmp_path)
    monkeypatch.setattr(console, "_manager", lambda: manager)
    monkeypatch.setattr(console, "_cfg", lambda: cfg)
    monkeypatch.setattr(console, "_now", lambda: _mon())

    return TestClient(routes.app)


def test_get_console_snapshot(client):
    body = client.get("/api/console").json()
    assert body["ok"] is True and body["block_new_entries"] is False
    keys = {k["key"] for k in body["knobs"]}
    assert "risk_per_trade_pct" in keys and "fit_lanes_enabled" in keys
    fit = next(k for k in body["knobs"] if k["key"] == "fit_lanes_enabled")
    assert fit["restart_required"] is True


def test_put_tighten_applies(client, manager):
    r = client.put("/api/console/risk_per_trade_pct", json={"value": 0.005, "reason": "cut"})
    assert r.status_code == 200 and r.json()["result"]["status"] == "applied"
    assert manager.data["trading.risk_per_trade_pct"] == 0.005


def test_put_loosen_pending_and_cancel(client):
    r = client.put("/api/console/risk_per_trade_pct", json={"value": 0.02, "reason": "more"})
    assert r.status_code == 200 and r.json()["result"]["status"] == "pending"
    pid = r.json()["result"]["pending_id"]
    snap = client.get("/api/console").json()
    knob = next(k for k in snap["knobs"] if k["key"] == "risk_per_trade_pct")
    assert knob["pending"]["id"] == pid
    assert client.delete(f"/api/console/pending/{pid}").status_code == 200
    assert client.delete(f"/api/console/pending/{pid}").status_code == 404


def test_put_validation_400(client):
    assert client.put("/api/console/risk_per_trade_pct", json={"value": 2, "reason": ""}).status_code == 400


def test_put_unknown_404(client):
    assert client.put("/api/console/nope", json={"value": 1, "reason": ""}).status_code == 404


def test_put_env_shadowed_409(client, monkeypatch):
    monkeypatch.setenv("RISK_PER_TRADE_PCT", "0.01")
    r = client.put("/api/console/risk_per_trade_pct", json={"value": 0.005, "reason": ""})
    assert r.status_code == 409 and "env_shadowed" in r.json()["error"]
    snap = client.get("/api/console").json()
    assert next(k for k in snap["knobs"] if k["key"] == "risk_per_trade_pct")["env_shadowed"] is True


def test_restart_required_409(client):
    assert client.put("/api/console/fit_lanes_enabled", json={"value": "mr_20d", "reason": ""}).status_code == 409


def test_block_unblock(client, manager):
    assert client.post("/api/console/block-entries", json={"block": True, "reason": "x"}).status_code == 200
    assert manager.data["trading.block_new_entries"] is True
    assert client.post("/api/console/block-entries", json={"block": False, "reason": "x"}).status_code == 400
    assert client.post("/api/console/block-entries", json={"block": False, "reason": "x", "confirm": "YES"}).status_code == 200
    assert manager.data["trading.block_new_entries"] is False


def test_audit_endpoint(client):
    client.put("/api/console/risk_per_trade_pct", json={"value": 0.005, "reason": "cut"})
    rows = client.get("/api/console/audit?limit=5").json()["rows"]
    assert rows and rows[-1]["key"] == "risk_per_trade_pct"


def test_reason_is_bounded(client):
    r = client.put("/api/console/risk_per_trade_pct", json={"value": 0.005, "reason": "x" * 5000})
    assert r.status_code == 200
    rows = client.get("/api/console/audit?limit=1").json()["rows"]
    assert len(rows[-1]["reason"]) == 500
