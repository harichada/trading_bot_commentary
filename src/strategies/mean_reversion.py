"""
Mean Reversion Strategy

Identifies overbought/oversold conditions for reversion to mean.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from .base import BaseStrategy, Signal, SignalType


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy.

    Identifies when price deviates significantly from its mean and
    trades the expected reversion.

    Uses Bollinger Bands, RSI, and Z-score to identify extremes.

    Signals:
    - BUY: Price near/below lower BB + RSI oversold + negative Z-score
    - SELL: Price near/above upper BB + RSI overbought + positive Z-score
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="mean_reversion", config=config)

        # Strategy parameters
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2.0)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.zscore_threshold = self.config.get('zscore_threshold', 2.0)
        self.lookback = self.config.get('lookback', 50)

    @property
    def min_periods(self) -> int:
        return max(self.bb_period, self.lookback) + 10

    def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        """Generate trading signal based on mean reversion indicators"""
        if not self.validate_data(data):
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                strategy=self.name,
                metadata={'reason': 'insufficient_data'}
            )

        df = self.calculate_indicators(data)
        current = df.iloc[-1]

        close = current['close']
        rsi = current.get('rsi', 50)
        bb_upper = current.get('bb_upper', close * 1.02)
        bb_lower = current.get('bb_lower', close * 0.98)
        bb_middle = current.get('bb_middle', close)
        zscore = current.get('zscore', 0)
        bb_pct = current.get('bb_pct', 0.5)

        # Calculate signals
        signals = []
        weights = []

        # Bollinger Band position signal
        if bb_pct < 0:  # Below lower band
            signals.append(1)  # Strong buy signal
            weights.append(0.35)
        elif bb_pct > 1:  # Above upper band
            signals.append(-1)  # Strong sell signal
            weights.append(0.35)
        elif bb_pct < 0.2:
            signals.append(0.7)
            weights.append(0.3)
        elif bb_pct > 0.8:
            signals.append(-0.7)
            weights.append(0.3)
        else:
            signals.append(0)
            weights.append(0.15)

        # RSI signal
        if rsi < self.rsi_oversold:
            signals.append(1)
            weights.append(0.3)
        elif rsi > self.rsi_overbought:
            signals.append(-1)
            weights.append(0.3)
        elif rsi < 40:
            signals.append(0.5)
            weights.append(0.2)
        elif rsi > 60:
            signals.append(-0.5)
            weights.append(0.2)
        else:
            signals.append(0)
            weights.append(0.1)

        # Z-score signal
        if zscore < -self.zscore_threshold:
            signals.append(1)
            weights.append(0.25)
        elif zscore > self.zscore_threshold:
            signals.append(-1)
            weights.append(0.25)
        elif zscore < -1:
            signals.append(0.5)
            weights.append(0.15)
        elif zscore > 1:
            signals.append(-0.5)
            weights.append(0.15)
        else:
            signals.append(0)
            weights.append(0.1)

        # Distance from mean signal
        mean_distance = (close - bb_middle) / bb_middle if bb_middle > 0 else 0
        if abs(mean_distance) > 0.03:  # More than 3% from mean
            signals.append(-1 if mean_distance > 0 else 1)
            weights.append(0.1)

        # Calculate weighted score
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(signals, weights)) / total_weight

        # Determine signal type
        signal_type = SignalType.HOLD
        confidence = abs(weighted_score)

        if weighted_score > 0.35:
            signal_type = SignalType.STRONG_BUY if weighted_score > 0.65 else SignalType.BUY
        elif weighted_score < -0.35:
            signal_type = SignalType.STRONG_SELL if weighted_score < -0.65 else SignalType.SELL

        # For mean reversion, target is the mean, stop is further from mean
        atr = current.get('atr', close * 0.02)

        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            target_price = bb_middle  # Revert to mean
            stop_loss = close - (2 * atr)
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            target_price = bb_middle
            stop_loss = close + (2 * atr)
        else:
            target_price = None
            stop_loss = None

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=min(confidence, 1.0),
            strategy=self.name,
            price=close,
            target_price=target_price,
            stop_loss=stop_loss,
            take_profit=target_price,
            metadata={
                'rsi': round(rsi, 2),
                'bb_pct': round(bb_pct, 3),
                'zscore': round(zscore, 3),
                'bb_upper': round(bb_upper, 2),
                'bb_lower': round(bb_lower, 2),
                'bb_middle': round(bb_middle, 2),
                'distance_from_mean_pct': round(mean_distance * 100, 2),
                'weighted_score': round(weighted_score, 3),
                'signals': {
                    'bb_signal': 'oversold' if bb_pct < 0.2 else 'overbought' if bb_pct > 0.8 else 'neutral',
                    'rsi_signal': 'oversold' if rsi < self.rsi_oversold else 'overbought' if rsi > self.rsi_overbought else 'neutral',
                    'zscore_signal': 'oversold' if zscore < -self.zscore_threshold else 'overbought' if zscore > self.zscore_threshold else 'neutral'
                }
            }
        )

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion specific indicators"""
        df = super().calculate_indicators(data)

        # Z-score
        mean = df['close'].rolling(window=self.lookback).mean()
        std = df['close'].rolling(window=self.lookback).std()
        df['zscore'] = (df['close'] - mean) / std.replace(0, np.inf)

        # Bollinger Band percentage (0 = at lower, 1 = at upper)
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            bb_range = df['bb_upper'] - df['bb_lower']
            df['bb_pct'] = (df['close'] - df['bb_lower']) / bb_range.replace(0, np.inf)

        # Deviation from mean
        df['deviation'] = df['close'] - mean

        return df
