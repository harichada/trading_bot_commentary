"""v-playbook-desk-b2-final-2026-09-25 (I1): `SchwabDataProvider.get_market_data`
must route a *minute* request to a minute endpoint.

Whole-branch review finding I1: the routing only reached
`get_price_history_every_five_minutes` when `period == 1`, so every 10-day
5-minute request — `core/engine.py::_fit_bars_source` and
`core/loops/desk_steps.py::fetch_today_bars5`, both of which ask for
`period_type="day", period=10, frequency_type="minute", frequency=5` — silently
received *daily* candles. The strategy-fit lane and the desk's in-play features
were computing 5-minute features on daily bars.

Offline: a fake schwab-py client records which price-history helper was reached;
nothing touches Schwab.
"""
from __future__ import annotations

import pandas as pd

from data_providers.schwab import (DAILY_HISTORY_CALL, MINUTE_HISTORY_CALLS,
                                   MINUTE_MAX_PERIOD, SchwabDataProvider,
                                   history_call_name)

# The two shapes live code asks for (engine `_fit_bars_source`, desk `fetch_today_bars5`).
_MINUTE_SHAPE = dict(period_type="day", period=10, frequency_type="minute", frequency=5)
_DAILY_SHAPE = dict(period_type="year", period=1, frequency_type="daily", frequency=1)


class _Response:
    status_code = 200

    def __init__(self, candles):
        self._candles = candles

    def json(self):
        return {"candles": self._candles}


def _candles(n: int = 3, start_ms: int = 1_758_800_000_000, step_ms: int = 300_000):
    return [{"datetime": start_ms + i * step_ms, "open": 1.0, "high": 1.0,
             "low": 1.0, "close": 1.0, "volume": 100} for i in range(n)]


class _FakeClient:
    """Records which price-history helper the provider reached for."""

    def __init__(self):
        self.calls: list[str] = []

    def _answer(self, name: str):
        self.calls.append(name)
        return _Response(_candles())

    def get_price_history_every_minute(self, symbol, **kw):
        return self._answer("get_price_history_every_minute")

    def get_price_history_every_five_minutes(self, symbol, **kw):
        return self._answer("get_price_history_every_five_minutes")

    def get_price_history_every_ten_minutes(self, symbol, **kw):
        return self._answer("get_price_history_every_ten_minutes")

    def get_price_history_every_day(self, symbol, **kw):
        return self._answer("get_price_history_every_day")


class _Commentary:
    def __init__(self):
        self.entries: list = []

    def add_commentary(self, entry):
        self.entries.append(entry)


def _provider():
    client = _FakeClient()
    return SchwabDataProvider(client, _Commentary()), client


# ── the pure routing helper ─────────────────────────────────────────

def test_ten_day_five_minute_shape_routes_to_the_five_minute_helper():
    assert history_call_name(**_MINUTE_SHAPE) == MINUTE_HISTORY_CALLS[5]


def test_one_day_five_minute_shape_still_routes_to_the_five_minute_helper():
    assert history_call_name("day", 1, "minute", 5) == MINUTE_HISTORY_CALLS[5]


def test_one_minute_shape_routes_to_the_minute_helper():
    assert history_call_name("day", 10, "minute", 1) == MINUTE_HISTORY_CALLS[1]


def test_daily_shape_routes_to_the_daily_helper():
    assert history_call_name(**_DAILY_SHAPE) == DAILY_HISTORY_CALL


def test_a_period_beyond_schwabs_minute_limit_falls_back_to_daily():
    """Schwab's `periodType=day` supports period 1–5,10 with minute bars; a longer
    minute request has no minute endpoint, so the daily one is the honest answer."""
    assert history_call_name("day", MINUTE_MAX_PERIOD + 1, "minute", 5) == DAILY_HISTORY_CALL


def test_an_unsupported_minute_frequency_falls_back_to_daily():
    assert history_call_name("day", 10, "minute", 7) == DAILY_HISTORY_CALL


def test_an_unreadable_period_is_read_as_the_single_day_default():
    assert history_call_name("day", None, "minute", 5) == MINUTE_HISTORY_CALLS[5]


# ── the provider call ───────────────────────────────────────────────

def test_get_market_data_calls_the_five_minute_endpoint_for_the_desk_shape():
    provider, client = _provider()
    out = provider.get_market_data("NVDA", **_MINUTE_SHAPE)
    assert client.calls == ["get_price_history_every_five_minutes"]
    assert isinstance(out, pd.DataFrame) and not out.empty


def test_get_market_data_calls_the_daily_endpoint_for_the_daily_shape():
    provider, client = _provider()
    provider.get_market_data("NVDA", **_DAILY_SHAPE)
    assert client.calls == ["get_price_history_every_day"]


def test_the_cache_never_serves_daily_bars_to_a_minute_request():
    """The cache key must carry the resolved endpoint: a daily-fallback response
    cached under a minute request's key would hand daily candles to the desk."""
    provider, client = _provider()
    provider.get_market_data("NVDA", period_type="day", period=30,
                             frequency_type="minute", frequency=5)   # daily fallback
    provider.get_market_data("NVDA", **_MINUTE_SHAPE)                # must still call out
    assert client.calls == ["get_price_history_every_day",
                            "get_price_history_every_five_minutes"]


def test_the_cache_still_serves_a_repeated_identical_request():
    provider, client = _provider()
    provider.get_market_data("NVDA", **_MINUTE_SHAPE)
    provider.get_market_data("NVDA", **_MINUTE_SHAPE)
    assert client.calls == ["get_price_history_every_five_minutes"]
