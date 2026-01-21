"""
Base Strategy Interface

Abstract base class for all trading strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


class SignalType(Enum):
    """Signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class Signal:
    """Trading signal from a strategy"""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    strategy: str
    timestamp: datetime = field(default_factory=datetime.now)
    price: float = 0.0
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def action(self) -> str:
        """Convert signal type to action string"""
        if self.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return "buy"
        elif self.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            return "sell"
        return "hold"

    @property
    def is_actionable(self) -> bool:
        """Check if signal should be acted upon"""
        return self.signal_type != SignalType.HOLD and self.confidence >= 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'action': self.action,
            'confidence': self.confidence,
            'strategy': self.strategy,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'metadata': self.metadata
        }


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All strategies must implement the analyze() method.

    Usage:
        class MyStrategy(BaseStrategy):
            def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
                # Your analysis logic
                return Signal(...)

    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy.

        Args:
            name: Strategy identifier
            config: Strategy-specific configuration
        """
        self.name = name
        self.config = config or {}
        self._enabled = True
        self._performance: Dict[str, Any] = {
            'total_signals': 0,
            'profitable_signals': 0,
            'win_rate': 0.0,
            'avg_return': 0.0
        }

    @property
    def enabled(self) -> bool:
        """Check if strategy is enabled"""
        return self._enabled

    def enable(self):
        """Enable strategy"""
        self._enabled = True

    def disable(self):
        """Disable strategy"""
        self._enabled = False

    @abstractmethod
    def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        """
        Analyze market data and generate a signal.

        Args:
            symbol: The stock symbol
            data: DataFrame with OHLCV data and indicators

        Returns:
            Signal object with recommendation
        """
        pass

    def update_performance(self, signal: Signal, outcome: float):
        """
        Update strategy performance metrics.

        Args:
            signal: The original signal
            outcome: The realized return (positive for profit)
        """
        self._performance['total_signals'] += 1

        if outcome > 0:
            self._performance['profitable_signals'] += 1

        # Update win rate
        total = self._performance['total_signals']
        wins = self._performance['profitable_signals']
        self._performance['win_rate'] = wins / total if total > 0 else 0.0

        # Update average return (exponential moving average)
        alpha = 0.1
        self._performance['avg_return'] = (
            alpha * outcome +
            (1 - alpha) * self._performance['avg_return']
        )

    @property
    def performance(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self._performance.copy()

    def get_required_columns(self) -> List[str]:
        """
        Get list of required DataFrame columns.

        Override in subclass if specific columns are needed.
        """
        return ['open', 'high', 'low', 'close', 'volume']

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that required data is present"""
        required = self.get_required_columns()
        missing = [col for col in required if col not in data.columns]

        if missing:
            return False

        if data.empty or len(data) < self.min_periods:
            return False

        return True

    @property
    def min_periods(self) -> int:
        """Minimum number of periods required for analysis"""
        return 20  # Default, override in subclass

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.

        Override in subclass to add strategy-specific indicators.
        """
        df = data.copy()

        # Basic indicators that most strategies use
        if 'close' in df.columns:
            # Simple moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()

            # Exponential moving averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()

            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # RSI
            df['rsi'] = self._calculate_rsi(df['close'])

            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

            # Average True Range
            df['atr'] = self._calculate_atr(df)

            # Volume indicators
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self._enabled})"
