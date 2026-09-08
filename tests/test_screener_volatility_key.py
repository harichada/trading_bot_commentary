"""v-screener-volatility-keyerror: regression test.

2026-07-31 log showed `Screener error: 'volatility'` every cycle. Root
cause: screen_stocks builds its commentary payload with a hard
`m['volatility']` access, but the volatility key only exists on movers
whose Schwab quote enrichment succeeded. With the Schwab token expired,
Yahoo-sourced movers were never enriched, the KeyError aborted the whole
cycle, and the screener returned [] — killing the Yahoo-only fallback
exactly when it was needed.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from analysis.screener import StockScreener


def _mover_without_volatility(symbol: str = "RKLB") -> dict:
    """A Yahoo-shaped mover that passes _is_tradeable but was never
    enriched by _get_quote_data — so it has no 'volatility' key."""
    return {
        "symbol": symbol,
        "description": "Rocket Lab",
        "last": 50.0,
        "change": 5.0,
        "percent_change": 11.0,
        "volume": 5_000_000,
        "direction": "up",
        "source": "yahoo_most_active",
    }


@pytest.fixture
def screener(monkeypatch):
    s = StockScreener(schwab_client=MagicMock(), commentary_system=MagicMock())

    async def _yahoo_active():
        return [_mover_without_volatility()]

    async def _empty(*args, **kwargs):
        return []

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(s, "_get_yahoo_most_active", _yahoo_active)
    monkeypatch.setattr(s, "_get_yahoo_top_movers", _empty)
    monkeypatch.setattr(s, "_get_index_movers", _empty)
    monkeypatch.setattr(s, "_get_volatile_stocks", _empty)
    monkeypatch.setattr(s, "_refresh_quality_data", _noop)
    # Schwab down → every quote enrichment fails.
    monkeypatch.setattr(s, "_get_quote_data", lambda sym: None)
    return s


def test_screener_survives_unenriched_movers(screener):
    """A mover lacking 'volatility' must not abort the whole cycle."""
    result = asyncio.run(screener.screen_stocks())
    assert [m["symbol"] for m in result] == ["RKLB"], (
        "screen_stocks returned no movers — KeyError('volatility') "
        "aborted the cycle instead of degrading gracefully"
    )


def test_screener_commentary_uses_safe_volatility_default(screener):
    """Commentary payload should report avg_volatility=0 for unenriched
    movers rather than raising."""
    asyncio.run(screener.screen_stocks())
    commentary = screener.commentary.add_commentary
    assert commentary.called, "commentary was never emitted"
    payload = commentary.call_args[0][0]
    assert payload.data["avg_volatility"] == 0
    assert payload.data["top_gainer"] == "RKLB"
