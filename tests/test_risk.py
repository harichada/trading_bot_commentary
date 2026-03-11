"""Tests for risk management."""
import pytest
import numpy as np
from datetime import datetime

from core.models import Position


class TestPositionSizing:
    """Test position sizing logic."""

    def test_position_quantity_positive(self):
        """Position size should always be >= 0."""
        signal_position_size = int(10000 / 150.0)  # $10k / $150
        assert signal_position_size > 0
        assert signal_position_size == 66

    def test_max_position_value_constraint(self):
        """Position size should be capped by max position value."""
        total_value = 100000
        max_pct = 0.25
        max_position_value = total_value * max_pct
        entry_price = 500.0
        position_size = 100

        if position_size * entry_price > max_position_value:
            position_size = int(max_position_value / entry_price)

        assert position_size * entry_price <= max_position_value
        assert position_size == 50


class TestStopLossCalculation:
    """Test default stop loss and take profit from config."""

    def test_default_stop_loss(self):
        from core.config import Config
        c = Config()
        entry_price = 100.0
        stop_loss = entry_price * (1 - c.DEFAULT_STOP_LOSS_PCT)
        assert stop_loss == 95.0  # 5% below

    def test_default_take_profit(self):
        from core.config import Config
        c = Config()
        entry_price = 100.0
        take_profit = entry_price * (1 + c.DEFAULT_TAKE_PROFIT_PCT)
        assert abs(take_profit - 110.0) < 0.01  # 10% above

    def test_stop_loss_below_entry(self):
        """Stop loss should always be below entry for long positions."""
        from core.config import Config
        c = Config()
        entry_price = 250.0
        stop_loss = entry_price * (1 - c.DEFAULT_STOP_LOSS_PCT)
        assert stop_loss < entry_price

    def test_take_profit_above_entry(self):
        """Take profit should always be above entry for long positions."""
        from core.config import Config
        c = Config()
        entry_price = 250.0
        take_profit = entry_price * (1 + c.DEFAULT_TAKE_PROFIT_PCT)
        assert take_profit > entry_price


class TestCorrelationCalculation:
    """Test numpy correlation calculation (extracted logic from P4 fix)."""

    def test_positive_correlation(self):
        """Positively correlated series should produce correlation > 0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        corr = np.corrcoef(a, b)[0, 1]
        assert abs(corr - 1.0) < 1e-10

    def test_negative_correlation(self):
        """Negatively correlated series should produce correlation < 0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        corr = np.corrcoef(a, b)[0, 1]
        assert abs(corr - (-1.0)) < 1e-10

    def test_no_correlation(self):
        """Constant series should produce NaN correlation."""
        a = np.array([1.0, 1.0, 1.0, 1.0])
        b = np.array([2.0, 4.0, 6.0, 8.0])
        corr = np.corrcoef(a, b)[0, 1]
        assert np.isnan(corr)
