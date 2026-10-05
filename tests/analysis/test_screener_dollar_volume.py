"""v-dollar-volume-source-2026-10-05: the screener's dollar-volume leaders
source. MU on 2026-10-01 traded $48.5B on 44M shares and +2.8% versus the
prior close: not a share-count leader, not a 3% gainer, never a candidate.
These tests pin the ranking, the wiring and the failure behaviour."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from analysis import screener as sc
from analysis.screener import StockScreener, rank_by_dollar_volume


def _q(symbol, price, volume, pct=0.0):
    return {"symbol": symbol, "regularMarketPrice": {"raw": price},
            "regularMarketVolume": {"raw": volume}, "regularMarketChangePercent": {"raw": pct},
            "regularMarketChange": {"raw": price * pct / 100}, "shortName": symbol}


# The 2026-10-01 tape, roughly: MU is 7th by shares and 1st by dollars.
TAPE = [_q("NU", 1.6 * 1e9 / 123.1e6, 123.1e6, 4.9), _q("INTC", 119.6, 99.5e6, -0.19),
        _q("NVDA", 231.3, 93.4e6, 1.09), _q("SPCX", 150.0, 60e6, -2.0), _q("CTVA", 70.0, 55e6, 3.0),
        _q("NKE", 72.0, 50e6, -1.0), _q("MU", 1097.0, 44.2e6, 3.03), _q("ACN", 213.0, 25.8e6, 15.9),
        _q("ZERO", 0.0, 10e6), _q("", 10.0, 10e6)]


def test_rank_by_dollar_volume_puts_mu_first_and_drops_bad_rows():
    movers = [sc._parse_yahoo_quote(q, "yahoo_dollar_volume") for q in TAPE]
    movers = [m for m in movers if m is not None]
    ranked = rank_by_dollar_volume(movers, 5)
    assert [m["symbol"] for m in ranked][:3] == ["MU", "NVDA", "INTC"]
    assert ranked[0]["dollar_volume"] == pytest.approx(1097.0 * 44.2e6)
    assert len(ranked) == 5
    assert all(m["symbol"] != "ZERO" for m in ranked)
    # input dicts are not mutated
    assert "dollar_volume" not in movers[0]


def test_rank_dedupes_and_handles_garbage():
    movers = [{"symbol": "mu", "last": 1000, "volume": 1e6}, {"symbol": "MU", "last": 1000, "volume": 9e6},
              {"symbol": "X", "last": "n/a", "volume": 1}, {"symbol": "Y", "last": float("nan"), "volume": 1}]
    ranked = rank_by_dollar_volume(movers, 10)
    assert [m["symbol"] for m in ranked] == ["MU"] and ranked[0]["dollar_volume"] == 1e9
    assert rank_by_dollar_volume(movers, 0) == []


class _Resp:
    def __init__(self, code=200, quotes=None):
        self.status_code = code
        self._quotes = quotes or []
    def json(self):
        return {"finance": {"result": [{"quotes": self._quotes}]}}


@pytest.fixture
def screener(monkeypatch):
    s = StockScreener(schwab_client=None, commentary_system=None)
    monkeypatch.setattr(s, "_get_quote_data", lambda sym: {"bid": 1.0, "ask": 1.001})
    return s


def test_dollar_volume_source_pulls_wide_and_keeps_top_n(screener, monkeypatch):
    calls = []
    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(params)
        return _Resp(200, TAPE)
    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    from core.config import Config as _Cfg
    monkeypatch.setattr(_Cfg, "DOLLAR_VOLUME_SCAN_COUNT", property(lambda self: 250))
    monkeypatch.setattr(_Cfg, "DOLLAR_VOLUME_TOP_N", property(lambda self: 3))
    leaders = asyncio.run(screener._get_yahoo_dollar_volume_leaders())
    assert calls[0]["scrIds"] == "most_actives" and calls[0]["count"] == 250
    assert [m["symbol"] for m in leaders] == ["MU", "NVDA", "INTC"]
    assert all(m["source"] == "yahoo_dollar_volume" for m in leaders)
    assert all("bid" in m for m in leaders)          # survivors are quote-enriched
    # cached: a second call does not hit the network
    asyncio.run(screener._get_yahoo_dollar_volume_leaders())
    assert len(calls) == 1


def test_dollar_volume_source_http_error_returns_empty(screener, monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(429))
    assert asyncio.run(screener._get_yahoo_dollar_volume_leaders()) == []


def test_dollar_volume_source_network_error_returns_empty(screener, monkeypatch):
    import requests
    def boom(*a, **k):
        raise ConnectionError("down")
    monkeypatch.setattr(requests, "get", boom)
    assert asyncio.run(screener._get_yahoo_dollar_volume_leaders()) == []


def test_most_active_still_uses_shared_fetcher(screener, monkeypatch):
    import requests
    seen = {}
    def fake_get(url, params=None, headers=None, timeout=None):
        seen.update(params)
        return _Resp(200, TAPE[:3])
    monkeypatch.setattr(requests, "get", fake_get)
    movers = asyncio.run(screener._get_yahoo_most_active())
    assert seen["count"] == 30 and seen["scrIds"] == "most_actives"
    assert [m["symbol"] for m in movers] == ["NU", "INTC", "NVDA"]
    assert all(m["source"] == "yahoo_most_active" for m in movers)


def test_dollar_volume_tag_gets_the_mover_relax():
    import inspect
    src = inspect.getsource(StockScreener)
    i = src.index("_is_mover_source = _source in (")
    assert "'yahoo_dollar_volume'" in src[i:i + 200]
    assert "ENABLE_DOLLAR_VOLUME_SOURCE" in src


def test_config_knobs_defaults_and_clamps(monkeypatch):
    from core.config import Config, config_manager
    cfg = Config()
    monkeypatch.delenv("ENABLE_DOLLAR_VOLUME_SOURCE", raising=False)
    trading = config_manager.config.setdefault("trading", {})
    for k in ("enable_dollar_volume_source", "dollar_volume_scan_count", "dollar_volume_top_n"):
        monkeypatch.delitem(trading, k, raising=False)
    assert cfg.ENABLE_DOLLAR_VOLUME_SOURCE is True
    assert cfg.DOLLAR_VOLUME_SCAN_COUNT == 250 and cfg.DOLLAR_VOLUME_TOP_N == 30
    monkeypatch.setitem(trading, "dollar_volume_scan_count", 9999)
    monkeypatch.setitem(trading, "dollar_volume_top_n", 1)
    assert cfg.DOLLAR_VOLUME_SCAN_COUNT == 250 and cfg.DOLLAR_VOLUME_TOP_N == 5
    monkeypatch.setenv("ENABLE_DOLLAR_VOLUME_SOURCE", "0")
    assert cfg.ENABLE_DOLLAR_VOLUME_SOURCE is False
