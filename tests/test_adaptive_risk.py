"""
Test suite for the adaptive risk module.

The tests exercise the core logic and validate edge cases such as:

* Zero equity → no position
* Extremely high ATR → position capped by ``max_stop_pct``
* Loss‑rate threshold enforcement
* Draw‑down protection

The tests are intentionally lightweight and use `pytest`.
"""

import pytest
from datetime import datetime
from typing import List

from trading_bot_commentary.adaptive_risk import (
    AdaptiveRiskParams,
    AdaptiveRiskManager,
    TradePosition,
    StrategySignal,
)

# Helper to create a dummy StrategySignal

def dummy_signal(signal_type: str) -> StrategySignal:
    return StrategySignal(
        symbol="TEST",
        signal_type=signal_type,
        strength=1.0,
        strategy_name="dummy",
        timestamp=datetime.utcnow(),
        entry_price=100,
        stop_loss=95,
        take_profit=110,
        position_size=10,
        metadata={},
    )

@pytest.fixture
def manager():
    return AdaptiveRiskManager(AdaptiveRiskParams(
        risk_per_trade=0.01,
        atr_multiplier=2.0,
        max_stop_pct=0.04,
        max_drawdown_pct=0.10,
        loss_window=5,
        loss_rate_threshold=0.5,
    ))


def test_zero_equity(manager):
    pos = manager(
        symbol="AAPL",
        equity=0.0,
        price=150.0,
        atr=1.0,
        start_equity=1000.0,
        recent_trades=[],
    )
    assert pos is None


def test_high_atr_capped_by_max_stop_pct(manager):
    pos = manager(
        symbol="AAPL",
        equity=1000.0,
        price=100.0,
        atr=50.0,  # huge ATR
        start_equity=1000.0,
        recent_trades=[dummy_signal("BUY")],
    )
    assert pos is not None
    # Max stop pct 4% of price = 4.0
    # Stop price should be price - 4.0
    assert abs(pos.stop_loss - 96.0) < 1e-6


def test_loss_rate_threshold(manager):
    # 3 losses in last 5 trades, threshold 0.5 -> 3/5=0.6 > 0.5, should block
    trades = [dummy_signal("SELL") for _ in range(3)] + [dummy_signal("BUY") for _ in range(2)]
    pos = manager(
        symbol="AAPL",
        equity=1000.0,
        price=100.0,
        atr=1.0,
        start_equity=1000.0,
        recent_trades=trades,
    )
    assert pos is None


def test_drawdown_protection(manager):
    # equity fell below 90% of start, threshold 10% -> should block
    pos = manager(
        symbol="AAPL",
        equity=900.0,  # 10% drawdown
        price=100.0,
        atr=1.0,
        start_equity=1000.0,
        recent_trades=[dummy_signal("BUY")],
    )
    assert pos is None


def test_successful_position(manager):
    pos = manager(
        symbol="AAPL",
        equity=1000.0,
        price=100.0,
        atr=1.0,
        start_equity=1000.0,
        recent_trades=[dummy_signal("BUY")],
    )
    assert pos is not None
    assert pos.quantity > 0
    assert pos.stop_loss < pos.entry_price
    assert pos.take_profit > pos.entry_price

"""
End of test file.
"""
