"""
Trading Strategies Module

Modular strategy system with pluggable strategies.
"""

from .base import BaseStrategy, Signal, SignalType
from .manager import StrategyManager
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy

__all__ = [
    'BaseStrategy', 'Signal', 'SignalType',
    'StrategyManager',
    'MomentumStrategy', 'MeanReversionStrategy', 'BreakoutStrategy'
]
