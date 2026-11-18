# tests/core/test_scalp_screener.py
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from core.scalp_screener import ScalpScreener, SymbolScore

FAKE_QUOTES = {
    "AAPL": {"prev_close": 100.0, "open": 103.0, "volume": 5_000_000, "avg_volume": 2_000_000, "price": 103.0},
    "TINY": {"prev_close": 1.0,   "open": 1.05,  "volume": 100_000,   "avg_volume": 50_000,    "price": 1.05},   # fails min_price
    "HUGE": {"prev_close": 500.0, "open": 505.0, "volume": 1_000_000, "avg_volume": 800_000,   "price": 505.0},  # fails max_price
    "FLAT": {"prev_close": 50.0,  "open": 50.1,  "volume": 200_000,   "avg_volume": 1_000_000, "price": 50.1},   # fails min_avg_volume
    "NVOL": {"prev_close": 50.0,  "open": 55.0,  "volume": 0,         "avg_volume": 1_000_000, "price": 55.0},   # pre-market volume=0 → use avg_volume
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
