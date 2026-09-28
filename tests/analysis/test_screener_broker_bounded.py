"""v-wedge-fix-2026-09-28 fix round 2 (Important 2): the screener's quote call.

`StockScreener._get_quote_data` called `self.client.get_quote(symbol)` — a raw
Schwab call with no timeout, invoked once per candidate symbol while scanning.
It is now bounded; a timeout degrades to "no quote for this symbol", which is
what the caller already does on any quote failure.
"""
from __future__ import annotations

import time

import pytest

from analysis.screener import StockScreener
from core import provider_calls


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class GoodClient:
    def get_quote(self, symbol):
        return _Response({symbol: {"quote": {
            "lastPrice": 100.0, "highPrice": 101.0, "lowPrice": 99.0,
            "openPrice": 100.0, "totalVolume": 1_000_000,
            "bidPrice": 99.9, "askPrice": 100.1,
        }}})


class HangingClient:
    def __init__(self):
        self.calls = 0

    def get_quote(self, symbol):
        self.calls += 1
        time.sleep(30)
        return _Response({})


@pytest.fixture(autouse=True)
def _tight(monkeypatch):
    provider_calls.reset_state()
    monkeypatch.setattr(provider_calls, "broker_timeout", lambda: 0.2)
    # the screener passes the quote budget explicitly (3 s live); tighten it too
    from core import config as _cfg
    monkeypatch.setattr(_cfg.Config, "QUOTE_FETCH_TIMEOUT_SEC", property(lambda self: 0.2))
    yield
    provider_calls.reset_state()


def test_a_hung_quote_call_returns_no_quote_quickly():
    client = HangingClient()
    screener = StockScreener(schwab_client=client, commentary_system=None)
    started = time.monotonic()
    assert screener._get_quote_data("NVDA") is None
    assert time.monotonic() - started < 0.8
    assert client.calls == 1


def test_a_good_quote_still_comes_back():
    screener = StockScreener(schwab_client=GoodClient(), commentary_system=None)
    out = screener._get_quote_data("NVDA")
    assert out is not None
    assert out["symbol"] == "NVDA"
    assert out["last"] == pytest.approx(100.0)
    assert out["volatility"] == pytest.approx(2.0)
