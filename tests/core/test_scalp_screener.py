# tests/core/test_scalp_screener.py
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from core.scalp_screener import ScalpScreener, SymbolScore

FAKE_QUOTES = {
    "AAPL": {"prev_close": 100.0, "open": 103.0, "volume": 5_000_000, "avg_daily_volume": 2_000_000, "price": 103.0},
    "TINY": {"prev_close": 1.0,   "open": 1.05,  "volume": 100_000,   "avg_daily_volume": 50_000,    "price": 1.05},   # fails min_price
    "HUGE": {"prev_close": 500.0, "open": 505.0, "volume": 1_000_000, "avg_daily_volume": 800_000,   "price": 505.0},  # fails max_price (>500)
    "FLAT": {"prev_close": 50.0,  "open": 50.1,  "volume": 200_000,   "avg_daily_volume": 200_000,   "price": 50.1},   # fails min_avg_volume
    "NVOL": {"prev_close": 50.0,  "open": 55.0,  "volume": 0,         "avg_daily_volume": 1_000_000, "price": 55.0},   # pre-market volume=0 → use avg_daily_volume
}

def test_gap_pct_calculated_correctly():
    # AAPL: (103 - 100) / 100 * 100 = 3.0%
    screener = ScalpScreener(["AAPL"], top_n=1)
    scores = screener._score_quote(FAKE_QUOTES["AAPL"])
    assert abs(scores.gap_pct - 3.0) < 0.01

def test_filters_remove_out_of_range_symbols():
    screener = ScalpScreener(["TINY", "HUGE", "FLAT", "AAPL"], top_n=5)
    passing = [s for s in screener._filter([
        screener._score_quote(FAKE_QUOTES[sym], sym) for sym in ["TINY","HUGE","FLAT","AAPL"]
    ])]
    syms = [s.symbol for s in passing]
    assert "AAPL" in syms
    assert "TINY" not in syms
    assert "HUGE" not in syms

def test_zero_volume_uses_avg_volume_for_rvol():
    screener = ScalpScreener(["NVOL"], top_n=1)
    score = screener._score_quote(FAKE_QUOTES["NVOL"], "NVOL")
    # rvol = avg_volume / avg_volume = 1.0 when today's volume is 0
    assert score.rvol == pytest.approx(1.0, abs=0.1)

def test_composite_score_range():
    screener = ScalpScreener(["AAPL"], top_n=1)
    score = screener._score_quote(FAKE_QUOTES["AAPL"], "AAPL")
    assert 0.0 <= score.composite <= 1.0

def test_select_returns_top_n_symbols():
    # Only AAPL passes filters
    screener = ScalpScreener(["AAPL", "TINY", "FLAT"], top_n=5)
    # Mock _fetch_quotes
    with patch.object(screener, "_fetch_quotes", return_value=FAKE_QUOTES):
        import asyncio
        result = asyncio.run(screener.select())
    assert result == ["AAPL"]

def test_missing_symbol_skipped_gracefully():
    screener = ScalpScreener(["AAPL", "INVALID_XYZ"], top_n=2)
    with patch.object(screener, "_fetch_quotes", return_value={"AAPL": FAKE_QUOTES["AAPL"]}):
        import asyncio
        result = asyncio.run(screener.select())
    assert "AAPL" in result
    assert "INVALID_XYZ" not in result


def test_fetch_quotes_returns_gap_and_volume():
    """_fetch_quotes with a patched yf.download returns expected quote fields."""
    screener = ScalpScreener(["AAPL"], top_n=1)
    # Single-symbol path: yfinance returns flat (non-MultiIndex) columns
    idx = pd.date_range("2026-11-17", periods=2, freq="B")
    fake_df = pd.DataFrame({
        "Open":   [100.0, 103.0],
        "High":   [101.0, 104.0],
        "Low":    [99.0,  102.0],
        "Close":  [100.5, 103.5],
        "Volume": [2_000_000, 3_000_000],
    }, index=idx)
    with patch("yfinance.download", return_value=fake_df):
        import asyncio
        quotes = asyncio.run(screener._fetch_quotes())
    assert "AAPL" in quotes
    q = quotes["AAPL"]
    assert q["prev_close"] == pytest.approx(100.5, abs=0.01)
    assert q["open"] == pytest.approx(103.0, abs=0.01)
    assert q["volume"] == pytest.approx(3_000_000, abs=1)


def test_zero_prev_close_gives_zero_gap():
    screener = ScalpScreener(["X"], top_n=1)
    score = screener._score_quote(
        {"prev_close": 0.0, "open": 5.0, "volume": 1_000_000, "avg_daily_volume": 1_000_000, "price": 5.0}, "X"
    )
    assert score.gap_pct == 0.0


def test_zero_hist_avg_gives_rvol_one():
    screener = ScalpScreener(["X"], top_n=1)
    score = screener._score_quote(
        {"prev_close": 10.0, "open": 11.0, "volume": 0, "avg_daily_volume": 0, "price": 11.0}, "X"
    )
    assert score.rvol == pytest.approx(1.0, abs=0.01)


def test_news_bus_enrichment_affects_composite():
    """When news_bus is provided, composite uses 0.4/0.4/0.2 weights and news_score > 0."""
    screener = ScalpScreener(["AAPL"], top_n=1)

    class FakeNewsBus:
        async def get_aggregate_sentiment(self, symbol):
            return 0.8  # strong positive

    with patch.object(screener, "_fetch_quotes", return_value={"AAPL": FAKE_QUOTES["AAPL"]}):
        import asyncio
        scores = asyncio.run(screener.score(news_bus=FakeNewsBus()))

    assert len(scores) == 1
    assert scores[0].news_score == pytest.approx(0.8, abs=0.01)
    # AAPL: gap_pct=3.0% → gap_score=0.3; rvol=5M/2M=2.5 → rvol_score=0.5
    # composite = 0.3*0.4 + 0.5*0.4 + 0.8*0.2 = 0.12 + 0.20 + 0.16 = 0.48
    assert scores[0].composite == pytest.approx(0.48, abs=0.02)


def test_fetch_quotes_multi_symbol_path():
    """_fetch_quotes with MultiIndex columns (multi-symbol yfinance format) returns all symbols."""
    import asyncio
    import numpy as np

    syms = ["AAPL", "MSFT"]
    screener = ScalpScreener(syms, top_n=2)

    def mock_download_multi(tickers, **kwargs):
        sym_list = tickers if isinstance(tickers, list) else tickers.split()
        metrics = ["Open", "High", "Low", "Close", "Volume"]
        # group_by="ticker" → first level = ticker, second level = metric
        idx = pd.MultiIndex.from_product([sym_list, metrics])
        data = {col: [100.0 if col[1] != "Volume" else 1_000_000] * 2
                for col in idx}
        df = pd.DataFrame(data, index=pd.date_range("2026-01-09", periods=2, freq="B"))
        df.columns = idx
        return df

    with patch("yfinance.download", side_effect=mock_download_multi):
        quotes = asyncio.run(screener._fetch_quotes())

    assert "AAPL" in quotes
    assert "MSFT" in quotes
    for sym in syms:
        q = quotes[sym]
        assert "prev_close" in q
        assert "open" in q
        assert "volume" in q
        assert "avg_daily_volume" in q
        assert q["price"] == pytest.approx(100.0, abs=0.01)
