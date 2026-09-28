"""v-wedge-fix-2026-09-28 fix round 1 (Important 4): the quality gate fails closed.

`_passes_quality_filter` used to pass a symbol through whenever its 60-day
daily bars were missing. That was written when the bars came from an in-process
yfinance call that either worked or raised. Now they come from the isolated
subprocess pool, which legitimately returns nothing while it is cold or after a
child was killed — and "no data" must NARROW the universe, not open it. A cold
pool that widened the universe would admit exactly the un-vetted names the
trend and relative-strength gates exist to keep out.
"""
from __future__ import annotations

import pandas as pd
import pytest

from analysis.screener import LEVERAGED_ETF_BLOCKLIST, StockScreener


@pytest.fixture
def screener():
    return StockScreener(schwab_client=None, commentary_system=None)


def _frame(closes) -> pd.DataFrame:
    return pd.DataFrame({"Close": [float(c) for c in closes]})


def _healthy_series(n: int = 60, start: float = 100.0) -> list:
    # Rising series: last close above its own SMA50 and a positive 10d return.
    return [start + i for i in range(n)]


def test_missing_quality_data_fails_the_gate(screener):
    assert screener._quality_data_cache == {}
    assert screener._passes_quality_filter("NVDA") is False
    assert screener._quality_filter_rejects.get("NVDA", 0) >= 1


def test_an_empty_frame_fails_the_gate(screener):
    screener._quality_data_cache["NVDA"] = pd.DataFrame()
    assert screener._passes_quality_filter("NVDA") is False


def test_too_little_history_fails_the_gate(screener):
    """Under 50 closes there is no SMA50 to judge — that is missing data."""
    screener._quality_data_cache["NVDA"] = _frame(_healthy_series(30))
    assert screener._passes_quality_filter("NVDA") is False


def test_a_missing_close_column_fails_the_gate(screener):
    screener._quality_data_cache["NVDA"] = pd.DataFrame({"Open": [1.0] * 60})
    assert screener._passes_quality_filter("NVDA") is False


def test_an_evaluation_error_fails_the_gate(screener):
    """Unparseable closes are indistinguishable from no closes: fail closed."""
    screener._quality_data_cache["NVDA"] = pd.DataFrame(
        {"Close": ["n/a"] * 60}
    )
    assert screener._passes_quality_filter("NVDA") is False


def test_a_healthy_symbol_with_data_still_passes(screener):
    screener._quality_data_cache["NVDA"] = _frame(_healthy_series())
    screener._quality_data_cache["SPY"] = _frame(_healthy_series())
    assert screener._passes_quality_filter("NVDA") is True


def test_the_blocklist_is_still_hard(screener):
    symbol = next(iter(LEVERAGED_ETF_BLOCKLIST))
    screener._quality_data_cache[symbol] = _frame(_healthy_series())
    assert screener._passes_quality_filter(symbol) is False


def test_the_trend_gate_still_rejects_a_downtrend(screener):
    falling = list(range(160, 100, -1))          # 60 closes, below SMA50
    screener._quality_data_cache["XYZ"] = _frame(falling)
    screener._quality_data_cache["SPY"] = _frame(_healthy_series())
    assert screener._passes_quality_filter("XYZ") is False


def test_catastrophic_relative_weakness_still_rejects(screener):
    # Flat-ish symbol above its SMA50, SPY up ~20% over the 10d window.
    sym = _healthy_series(60)
    sym[-11:] = [sym[-11]] * 11
    spy = _healthy_series(49) + [200.0 + 5 * i for i in range(11)]
    screener._quality_data_cache["XYZ"] = _frame(sym)
    screener._quality_data_cache["SPY"] = _frame(spy)
    assert screener._passes_quality_filter("XYZ") is False
