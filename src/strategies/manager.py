"""
Strategy Manager

Coordinates multiple strategies and provides consensus signals.
"""

import logging
from typing import Dict, List, Any, Optional, Type
from datetime import datetime
import pandas as pd
import asyncio

from .base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class StrategyManager:
    """
    Manages multiple trading strategies and provides consensus signals.

    Features:
    - Register/unregister strategies
    - Run all strategies in parallel
    - Generate consensus signals based on strategy agreement
    - Track strategy performance
    - Dynamic strategy weighting

    Usage:
        manager = StrategyManager()
        manager.register(MomentumStrategy())
        manager.register(MeanReversionStrategy())

        signals = await manager.analyze('AAPL', data)
    """

    def __init__(self, min_consensus: int = 2, confidence_threshold: float = 0.5):
        """
        Initialize strategy manager.

        Args:
            min_consensus: Minimum number of strategies that must agree
            confidence_threshold: Minimum confidence for actionable signals
        """
        self._strategies: Dict[str, BaseStrategy] = {}
        self._weights: Dict[str, float] = {}
        self.min_consensus = min_consensus
        self.confidence_threshold = confidence_threshold

        # Performance tracking
        self._signal_history: List[Dict[str, Any]] = []
        self._max_history = 1000

    def register(self, strategy: BaseStrategy, weight: float = 1.0):
        """
        Register a strategy.

        Args:
            strategy: Strategy instance to register
            weight: Weight for consensus voting (default 1.0)
        """
        self._strategies[strategy.name] = strategy
        self._weights[strategy.name] = weight
        logger.info(f"Registered strategy: {strategy.name} (weight: {weight})")

    def unregister(self, name: str):
        """Unregister a strategy by name"""
        if name in self._strategies:
            del self._strategies[name]
            del self._weights[name]
            logger.info(f"Unregistered strategy: {name}")

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name"""
        return self._strategies.get(name)

    @property
    def strategies(self) -> List[str]:
        """Get list of registered strategy names"""
        return list(self._strategies.keys())

    @property
    def enabled_strategies(self) -> List[str]:
        """Get list of enabled strategy names"""
        return [name for name, s in self._strategies.items() if s.enabled]

    def enable_strategy(self, name: str):
        """Enable a strategy"""
        if name in self._strategies:
            self._strategies[name].enable()

    def disable_strategy(self, name: str):
        """Disable a strategy"""
        if name in self._strategies:
            self._strategies[name].disable()

    def set_weight(self, name: str, weight: float):
        """Set strategy weight"""
        if name in self._strategies:
            self._weights[name] = weight

    async def analyze(self, symbol: str, data: pd.DataFrame) -> List[Signal]:
        """
        Analyze symbol with all enabled strategies.

        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data

        Returns:
            List of signals (individual + consensus)
        """
        signals: List[Signal] = []

        # Run enabled strategies
        for name, strategy in self._strategies.items():
            if not strategy.enabled:
                continue

            try:
                signal = strategy.analyze(symbol, data)
                signals.append(signal)

                # Store in history
                self._add_to_history(signal)

            except Exception as e:
                logger.error(f"Error in strategy {name}: {e}")

        # Generate consensus signal if enough strategies agree
        if len(signals) >= self.min_consensus:
            consensus = self._calculate_consensus(symbol, signals)
            if consensus:
                signals.append(consensus)

        return signals

    def _calculate_consensus(self, symbol: str, signals: List[Signal]) -> Optional[Signal]:
        """Calculate consensus signal from individual signals"""
        if not signals:
            return None

        # Count weighted votes
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0

        actionable_signals = [s for s in signals if s.is_actionable]

        for signal in actionable_signals:
            weight = self._weights.get(signal.strategy, 1.0)
            total_weight += weight

            if signal.action == 'buy':
                buy_score += signal.confidence * weight
            elif signal.action == 'sell':
                sell_score += signal.confidence * weight

        if total_weight == 0:
            return None

        # Normalize scores
        buy_score /= total_weight
        sell_score /= total_weight

        # Determine consensus
        signal_type = SignalType.HOLD
        confidence = 0.0

        if buy_score > sell_score and buy_score >= self.confidence_threshold:
            # Count agreeing strategies
            agreeing = sum(1 for s in actionable_signals if s.action == 'buy')
            if agreeing >= self.min_consensus:
                confidence = buy_score
                signal_type = SignalType.STRONG_BUY if confidence > 0.7 else SignalType.BUY

        elif sell_score > buy_score and sell_score >= self.confidence_threshold:
            agreeing = sum(1 for s in actionable_signals if s.action == 'sell')
            if agreeing >= self.min_consensus:
                confidence = sell_score
                signal_type = SignalType.STRONG_SELL if confidence > 0.7 else SignalType.SELL

        if signal_type == SignalType.HOLD:
            return None

        # Calculate consensus stop/target from individual signals
        buy_signals = [s for s in actionable_signals if s.action == 'buy']
        sell_signals = [s for s in actionable_signals if s.action == 'sell']

        relevant_signals = buy_signals if signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else sell_signals

        stop_losses = [s.stop_loss for s in relevant_signals if s.stop_loss]
        take_profits = [s.take_profit for s in relevant_signals if s.take_profit]
        prices = [s.price for s in relevant_signals if s.price]

        avg_stop = sum(stop_losses) / len(stop_losses) if stop_losses else None
        avg_target = sum(take_profits) / len(take_profits) if take_profits else None
        avg_price = sum(prices) / len(prices) if prices else 0

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            strategy='consensus',
            price=avg_price,
            stop_loss=avg_stop,
            take_profit=avg_target,
            metadata={
                'buy_score': round(buy_score, 3),
                'sell_score': round(sell_score, 3),
                'agreeing_strategies': [s.strategy for s in actionable_signals if s.action == signal_type.value.replace('strong_', '')],
                'total_strategies': len(actionable_signals),
                'individual_signals': [s.to_dict() for s in signals]
            }
        )

    def _add_to_history(self, signal: Signal):
        """Add signal to history"""
        self._signal_history.append({
            'timestamp': datetime.now().isoformat(),
            'signal': signal.to_dict()
        })

        if len(self._signal_history) > self._max_history:
            self._signal_history = self._signal_history[-self._max_history:]

    def update_performance(self, strategy_name: str, signal: Signal, outcome: float):
        """Update strategy performance"""
        if strategy_name in self._strategies:
            self._strategies[strategy_name].update_performance(signal, outcome)

    def get_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all strategies"""
        return {
            name: strategy.performance
            for name, strategy in self._strategies.items()
        }

    def get_signal_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent signal history"""
        return self._signal_history[-limit:]

    def optimize_weights(self):
        """
        Optimize strategy weights based on recent performance.

        Uses simple performance-based weighting.
        """
        performances = self.get_performance()

        # Calculate weight based on win rate and average return
        for name, perf in performances.items():
            win_rate = perf.get('win_rate', 0.5)
            avg_return = perf.get('avg_return', 0)

            # Score combines win rate and profitability
            score = (win_rate * 0.6) + (max(0, avg_return) * 0.4)

            # Map to weight (0.5 to 2.0 range)
            new_weight = 0.5 + (score * 1.5)
            self._weights[name] = new_weight

            logger.info(f"Updated {name} weight to {new_weight:.2f}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'strategies': self.strategies,
            'enabled': self.enabled_strategies,
            'weights': self._weights.copy(),
            'min_consensus': self.min_consensus,
            'confidence_threshold': self.confidence_threshold,
            'performance': self.get_performance()
        }


def create_default_manager(config: Optional[Dict[str, Any]] = None) -> StrategyManager:
    """
    Create a strategy manager with default strategies.

    Args:
        config: Optional configuration dict

    Returns:
        Configured StrategyManager
    """
    from .momentum import MomentumStrategy
    from .mean_reversion import MeanReversionStrategy
    from .breakout import BreakoutStrategy

    config = config or {}

    manager = StrategyManager(
        min_consensus=config.get('min_consensus', 2),
        confidence_threshold=config.get('confidence_threshold', 0.5)
    )

    # Register default strategies
    manager.register(MomentumStrategy(config.get('momentum', {})), weight=1.0)
    manager.register(MeanReversionStrategy(config.get('mean_reversion', {})), weight=1.0)
    manager.register(BreakoutStrategy(config.get('breakout', {})), weight=1.0)

    return manager
