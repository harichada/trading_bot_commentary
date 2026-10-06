"""v-algorithms-page-2026-10-06: /api/algorithms lists every registered strategy
with live/shadow status and recent decisions from the in-memory DecisionBook."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from strategies.decision_book import DecisionBook


@pytest.fixture
def client(monkeypatch):
    import api.routes as routes
    import api.algorithms as alg
    monkeypatch.delenv("TRADING_API_KEY", raising=False)
    strategies = [SimpleNamespace(name="fit:rsi_turn"), SimpleNamespace(name="fit:mr_20d"),
                  SimpleNamespace(name="day_trade_momentum"), SimpleNamespace(name="momentum")]
    engine = SimpleNamespace(strategies=strategies, mode=SimpleNamespace(value="live"), is_running=True)
    monkeypatch.setattr(routes, "trading_engine", engine)
    cfg = SimpleNamespace(FIT_LANE_LIVE_ENTRIES_ENABLED=("mr_20d",), DAY_TRADE_LIVE_ENTRIES_ENABLED=True,
                          DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED=False, ORB_LIVE_ENTRIES_ENABLED=False,
                          MEAN_REV_LIVE_ENTRIES_ENABLED=False, BLOCK_NEW_ENTRIES=False)
    monkeypatch.setattr(alg, "_cfg", lambda: cfg)
    DecisionBook.instance().clear()
    DecisionBook.instance().record("fit:rsi_turn", "MU", "skip", "no_entry_now", close=100.5)
    DecisionBook.instance().record("fit:rsi_turn", "TSLA", "skip", "no_entry_now", close=250.0, extra="x")
    DecisionBook.instance().record("day_trade_momentum", "NVDA", "signal_buy", "pullback", close=180.0)
    DecisionBook.instance().record("ghost", "AAPL", "skip", "old", close=1.0)
    return TestClient(routes.app)


def test_algorithms_listing(client):
    body = client.get("/api/algorithms").json()
    assert body["ok"] and body["mode"] == "live" and body["n_live"] == 3 and body["n_shadow"] == 1
    by = {a["name"]: a for a in body["algorithms"]}
    assert by["fit:rsi_turn"]["status"] == "shadow" and by["fit:mr_20d"]["status"] == "live"
    assert by["day_trade_momentum"]["status"] == "live" and by["momentum"]["status"] == "live"
    assert by["fit:rsi_turn"]["today_total"] == 2 and by["fit:rsi_turn"]["by_action"] == {"skip": 2}
    assert by["fit:rsi_turn"]["recent"][0]["symbol"] == "TSLA" and by["fit:rsi_turn"]["recent"][0]["details"] == {"extra": "x"}
    assert "ghost" in by and by["ghost"]["status"] == "unknown"       # decisions outlive registration
    assert [a["status"] for a in body["algorithms"]][:3] == ["live", "live", "live"]   # live first


def test_algorithm_detail_and_bounds(client):
    body = client.get("/api/algorithms/fit:rsi_turn?limit=1").json()
    assert body["ok"] and len(body["recent"]) == 1 and body["top_reasons"][0]["reason"] == "no_entry_now"
    assert client.get("/api/algorithms/nope").json()["recent"] == []


def test_listing_never_500_without_engine(monkeypatch):
    import api.routes as routes
    monkeypatch.delenv("TRADING_API_KEY", raising=False)
    monkeypatch.setattr(routes, "trading_engine", None)
    body = TestClient(routes.app).get("/api/algorithms").json()
    assert body["ok"] and body["mode"] == "" and isinstance(body["algorithms"], list)
