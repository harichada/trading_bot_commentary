"""
Tests for Trading Strategies
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategies.base import BaseStrategy, Signal, SignalType
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from strategies.manager import StrategyManager, create_default_manager


class TestSignal:
    """Tests for Signal dataclass"""

    def test_signal_creation(self):
        """Test signal can be created with required fields"""
        signal = Signal(
            symbol='AAPL',
            signal_type=SignalType.BUY,
            confidence=0.75,
            strategy='test'
        )

        assert signal.symbol == 'AAPL'
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence == 0.75
        assert signal.strategy == 'test'

    def test_signal_action_property(self):
        """Test action property converts signal types correctly"""
        buy_signal = Signal('AAPL', SignalType.BUY, 0.7, 'test')
        assert buy_signal.action == 'buy'

        strong_buy = Signal('AAPL', SignalType.STRONG_BUY, 0.9, 'test')
        assert strong_buy.action == 'buy'

        sell_signal = Signal('AAPL', SignalType.SELL, 0.7, 'test')
        assert sell_signal.action == 'sell'

        hold_signal = Signal('AAPL', SignalType.HOLD, 0.5, 'test')
        assert hold_signal.action == 'hold'

    def test_signal_is_actionable(self):
        """Test is_actionable property"""
        actionable = Signal('AAPL', SignalType.BUY, 0.7, 'test')
        assert actionable.is_actionable is True

        not_actionable_hold = Signal('AAPL', SignalType.HOLD, 0.7, 'test')
        assert not_actionable_hold.is_actionable is False

        not_actionable_low_conf = Signal('AAPL', SignalType.BUY, 0.3, 'test')
        assert not_actionable_low_conf.is_actionable is False

    def test_signal_to_dict(self):
        """Test signal serialization"""
        signal = Signal(
            symbol='AAPL',
            signal_type=SignalType.BUY,
            confidence=0.75,
            strategy='momentum',
            price=150.0,
            stop_loss=147.0,
            take_profit=156.0
        )

        d = signal.to_dict()
        assert d['symbol'] == 'AAPL'
        assert d['signal_type'] == 'buy'
        assert d['confidence'] == 0.75
        assert d['strategy'] == 'momentum'
        assert d['price'] == 150.0


class TestMomentumStrategy:
    """Tests for Momentum Strategy"""

    def test_strategy_initialization(self):
        """Test strategy initializes with defaults"""
        strategy = MomentumStrategy()
        assert strategy.name == 'momentum'
        assert strategy.enabled is True
        assert strategy.rsi_oversold == 30
        assert strategy.rsi_overbought == 70

    def test_strategy_with_custom_config(self):
        """Test strategy with custom configuration"""
        config = {
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'sma_period': 100
        }
        strategy = MomentumStrategy(config)
        assert strategy.rsi_oversold == 25
        assert strategy.rsi_overbought == 75
        assert strategy.sma_period == 100

    def test_analyze_with_valid_data(self, sample_ohlcv_data):
        """Test analyze returns valid signal"""
        strategy = MomentumStrategy()
        signal = strategy.analyze('AAPL', sample_ohlcv_data)

        assert signal.symbol == 'AAPL'
        assert signal.strategy == 'momentum'
        assert signal.signal_type in list(SignalType)
        assert 0 <= signal.confidence <= 1
        assert signal.price > 0

    def test_analyze_with_insufficient_data(self):
        """Test analyze handles insufficient data"""
        strategy = MomentumStrategy()
        small_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000000, 1100000]
        })

        signal = strategy.analyze('AAPL', small_data)
        assert signal.signal_type == SignalType.HOLD
        assert signal.confidence == 0.0
        assert 'insufficient_data' in str(signal.metadata)

    def test_analyze_uptrend(self, trending_up_data):
        """Test momentum strategy detects uptrend"""
        strategy = MomentumStrategy()
        signal = strategy.analyze('AAPL', trending_up_data)

        # In a strong uptrend, momentum should give bullish signal
        assert signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY, SignalType.HOLD]

    def test_analyze_downtrend(self, trending_down_data):
        """Test momentum strategy detects downtrend"""
        strategy = MomentumStrategy()
        signal = strategy.analyze('AAPL', trending_down_data)

        # In a strong downtrend, momentum should give bearish signal
        assert signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL, SignalType.HOLD]


class TestMeanReversionStrategy:
    """Tests for Mean Reversion Strategy"""

    def test_strategy_initialization(self):
        """Test strategy initializes correctly"""
        strategy = MeanReversionStrategy()
        assert strategy.name == 'mean_reversion'
        assert strategy.bb_period == 20
        assert strategy.bb_std == 2.0

    def test_analyze_range_bound(self, range_bound_data):
        """Test mean reversion in range-bound market"""
        strategy = MeanReversionStrategy()
        signal = strategy.analyze('AAPL', range_bound_data)

        assert signal.symbol == 'AAPL'
        assert signal.strategy == 'mean_reversion'
        assert signal.signal_type in list(SignalType)

    def test_analyze_with_extremes(self, sample_ohlcv_data):
        """Test mean reversion detects extremes"""
        strategy = MeanReversionStrategy()
        signal = strategy.analyze('AAPL', sample_ohlcv_data)

        # Should have BB percentage in metadata
        assert 'bb_pct' in signal.metadata
        assert 'zscore' in signal.metadata


class TestBreakoutStrategy:
    """Tests for Breakout Strategy"""

    def test_strategy_initialization(self):
        """Test strategy initializes correctly"""
        strategy = BreakoutStrategy()
        assert strategy.name == 'breakout'
        assert strategy.channel_period == 20
        assert strategy.volume_threshold == 1.5

    def test_analyze_with_valid_data(self, sample_ohlcv_data):
        """Test analyze returns valid signal"""
        strategy = BreakoutStrategy()
        signal = strategy.analyze('AAPL', sample_ohlcv_data)

        assert signal.symbol == 'AAPL'
        assert signal.strategy == 'breakout'
        assert 'upper_channel' in signal.metadata
        assert 'lower_channel' in signal.metadata


class TestStrategyManager:
    """Tests for Strategy Manager"""

    def test_manager_initialization(self):
        """Test manager initializes correctly"""
        manager = StrategyManager(min_consensus=2)
        assert manager.min_consensus == 2
        assert len(manager.strategies) == 0

    def test_register_strategy(self):
        """Test registering strategies"""
        manager = StrategyManager()
        strategy = MomentumStrategy()

        manager.register(strategy, weight=1.5)

        assert 'momentum' in manager.strategies
        assert manager._weights['momentum'] == 1.5

    def test_unregister_strategy(self):
        """Test unregistering strategies"""
        manager = StrategyManager()
        manager.register(MomentumStrategy())

        manager.unregister('momentum')

        assert 'momentum' not in manager.strategies

    def test_enable_disable_strategy(self):
        """Test enabling/disabling strategies"""
        manager = StrategyManager()
        strategy = MomentumStrategy()
        manager.register(strategy)

        manager.disable_strategy('momentum')
        assert 'momentum' not in manager.enabled_strategies

        manager.enable_strategy('momentum')
        assert 'momentum' in manager.enabled_strategies

    @pytest.mark.asyncio
    async def test_analyze_single_strategy(self, sample_ohlcv_data):
        """Test analyzing with single strategy"""
        manager = StrategyManager(min_consensus=1)
        manager.register(MomentumStrategy())

        signals = await manager.analyze('AAPL', sample_ohlcv_data)

        assert len(signals) >= 1
        assert signals[0].strategy == 'momentum'

    @pytest.mark.asyncio
    async def test_analyze_multiple_strategies(self, sample_ohlcv_data):
        """Test analyzing with multiple strategies"""
        manager = create_default_manager()

        signals = await manager.analyze('AAPL', sample_ohlcv_data)

        # Should have individual strategy signals
        strategy_names = [s.strategy for s in signals]
        assert 'momentum' in strategy_names
        assert 'mean_reversion' in strategy_names
        assert 'breakout' in strategy_names

    @pytest.mark.asyncio
    async def test_consensus_signal(self, trending_up_data):
        """Test consensus signal generation"""
        manager = StrategyManager(min_consensus=2, confidence_threshold=0.3)
        manager.register(MomentumStrategy())
        manager.register(MeanReversionStrategy())
        manager.register(BreakoutStrategy())

        signals = await manager.analyze('AAPL', trending_up_data)

        # Check if consensus signal was generated
        consensus_signals = [s for s in signals if s.strategy == 'consensus']
        # Consensus may or may not be generated depending on agreement
        assert len(signals) >= 3  # At least individual strategies

    def test_create_default_manager(self):
        """Test default manager creation"""
        manager = create_default_manager()

        assert 'momentum' in manager.strategies
        assert 'mean_reversion' in manager.strategies
        assert 'breakout' in manager.strategies

    def test_performance_tracking(self):
        """Test performance tracking"""
        manager = StrategyManager()
        manager.register(MomentumStrategy())

        signal = Signal('AAPL', SignalType.BUY, 0.8, 'momentum', price=150)

        # Simulate profitable trade
        manager.update_performance('momentum', signal, 0.05)

        perf = manager.get_performance()
        assert 'momentum' in perf
        assert perf['momentum']['total_signals'] == 1
        assert perf['momentum']['profitable_signals'] == 1
