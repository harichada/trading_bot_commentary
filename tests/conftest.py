"""
Pytest Configuration and Fixtures
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

    # Generate realistic price data
    base_price = 150.0
    returns = np.random.normal(0.001, 0.02, 100)
    prices = base_price * np.cumprod(1 + returns)

    # Add some volatility clustering
    volatility = np.abs(returns) * 0.5 + 0.01

    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
        'high': prices * (1 + volatility),
        'low': prices * (1 - volatility),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    return data


@pytest.fixture
def trending_up_data() -> pd.DataFrame:
    """Generate upward trending price data"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

    # Strong uptrend
    trend = np.linspace(0, 0.5, 100)  # 50% gain
    noise = np.random.normal(0, 0.01, 100)
    base_price = 100.0
    prices = base_price * (1 + trend + noise)

    volatility = 0.015

    data = pd.DataFrame({
        'open': prices * (1 - volatility / 2),
        'high': prices * (1 + volatility),
        'low': prices * (1 - volatility),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    return data


@pytest.fixture
def trending_down_data() -> pd.DataFrame:
    """Generate downward trending price data"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

    # Strong downtrend
    trend = np.linspace(0, -0.3, 100)  # 30% loss
    noise = np.random.normal(0, 0.01, 100)
    base_price = 150.0
    prices = base_price * (1 + trend + noise)

    volatility = 0.015

    data = pd.DataFrame({
        'open': prices * (1 + volatility / 2),
        'high': prices * (1 + volatility),
        'low': prices * (1 - volatility),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    return data


@pytest.fixture
def range_bound_data() -> pd.DataFrame:
    """Generate range-bound (sideways) price data"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

    # Oscillating around a mean
    base_price = 100.0
    oscillation = 5 * np.sin(np.linspace(0, 4 * np.pi, 100))
    noise = np.random.normal(0, 1, 100)
    prices = base_price + oscillation + noise

    volatility = 0.02

    data = pd.DataFrame({
        'open': prices * (1 - volatility / 2),
        'high': prices * (1 + volatility),
        'low': prices * (1 - volatility),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    return data


@pytest.fixture
def sample_portfolio() -> Dict[str, Any]:
    """Sample portfolio state"""
    return {
        'account_balance': 100000.0,
        'buying_power': 50000.0,
        'cash': 50000.0,
        'positions': {
            'AAPL': {
                'symbol': 'AAPL',
                'quantity': 100,
                'entry_price': 150.0,
                'current_price': 155.0,
                'market_value': 15500.0,
                'unrealized_pnl': 500.0,
                'sector': 'Technology'
            },
            'GOOGL': {
                'symbol': 'GOOGL',
                'quantity': 50,
                'entry_price': 140.0,
                'current_price': 145.0,
                'market_value': 7250.0,
                'unrealized_pnl': 250.0,
                'sector': 'Technology'
            }
        },
        'daily_pnl': 750.0,
        'total_pnl': 2500.0
    }


@pytest.fixture
def empty_portfolio() -> Dict[str, Any]:
    """Empty portfolio state"""
    return {
        'account_balance': 100000.0,
        'buying_power': 100000.0,
        'cash': 100000.0,
        'positions': {},
        'daily_pnl': 0.0,
        'total_pnl': 0.0
    }


@pytest.fixture
def sample_signal() -> Dict[str, Any]:
    """Sample trading signal"""
    return {
        'symbol': 'AAPL',
        'action': 'buy',
        'confidence': 0.75,
        'strategy': 'momentum',
        'price': 150.0,
        'stop_loss': 147.0,
        'take_profit': 156.0
    }


@pytest.fixture
def high_confidence_signal() -> Dict[str, Any]:
    """High confidence trading signal"""
    return {
        'symbol': 'NVDA',
        'action': 'buy',
        'confidence': 0.90,
        'strategy': 'consensus',
        'price': 500.0,
        'stop_loss': 485.0,
        'take_profit': 530.0
    }


@pytest.fixture
def low_confidence_signal() -> Dict[str, Any]:
    """Low confidence trading signal"""
    return {
        'symbol': 'TSLA',
        'action': 'sell',
        'confidence': 0.35,
        'strategy': 'mean_reversion',
        'price': 250.0,
        'stop_loss': 260.0,
        'take_profit': 235.0
    }
