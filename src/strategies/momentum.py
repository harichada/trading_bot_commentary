"""
Momentum Strategy

Identifies and trades with the prevailing market trend.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from .base import BaseStrategy, Signal, SignalType


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy.

    Uses RSI, MACD, and moving average crossovers to identify
    momentum shifts and trend continuation.

    Signals:
    - BUY: RSI oversold + MACD bullish crossover + price above SMA
    - SELL: RSI overbought + MACD bearish crossover + price below SMA
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="momentum", config=config)

        # Strategy parameters
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        self.sma_period = self.config.get('sma_period', 50)
        self.lookback = self.config.get('lookback', 14)

    @property
    def min_periods(self) -> int:
        return max(self.sma_period, self.macd_slow + self.macd_signal) + 10

    def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        """Generate trading signal based on momentum indicators"""
        if not self.validate_data(data):
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                strategy=self.name,
                metadata={'reason': 'insufficient_data'}
            )

        # Calculate indicators
        df = self.calculate_indicators(data)

        # Get latest values
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current

        close = current['close']
        rsi = current.get('rsi', 50)
        macd = current.get('macd', 0)
        macd_signal_val = current.get('macd_signal', 0)
        macd_hist = current.get('macd_hist', 0)
        prev_macd_hist = previous.get('macd_hist', 0)
        sma = current.get(f'sma_{self.sma_period}', close)

        # Calculate individual signals
        signals = []
        weights = []

        # RSI signal
        if rsi < self.rsi_oversold:
            signals.append(1)  # Bullish
            weights.append(0.3)
        elif rsi > self.rsi_overbought:
            signals.append(-1)  # Bearish
            weights.append(0.3)
        else:
            signals.append(0)
            weights.append(0.1)

        # MACD crossover signal
        if macd_hist > 0 and prev_macd_hist <= 0:
            signals.append(1)  # Bullish crossover
            weights.append(0.35)
        elif macd_hist < 0 and prev_macd_hist >= 0:
            signals.append(-1)  # Bearish crossover
            weights.append(0.35)
        else:
            # MACD histogram direction
            if macd_hist > prev_macd_hist:
                signals.append(0.5)
            elif macd_hist < prev_macd_hist:
                signals.append(-0.5)
            else:
                signals.append(0)
            weights.append(0.2)

        # Price vs SMA signal
        price_distance = (close - sma) / sma if sma > 0 else 0
        if close > sma:
            signals.append(min(1, price_distance * 10))
            weights.append(0.25)
        else:
            signals.append(max(-1, price_distance * 10))
            weights.append(0.25)

        # Momentum signal (price change)
        if len(df) >= self.lookback:
            momentum = (close - df['close'].iloc[-self.lookback]) / df['close'].iloc[-self.lookback]
            if momentum > 0.02:  # 2% gain
                signals.append(min(1, momentum * 20))
            elif momentum < -0.02:
                signals.append(max(-1, momentum * 20))
            else:
                signals.append(0)
            weights.append(0.1)

        # Calculate weighted score
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(signals, weights)) / total_weight

        # Determine signal type and confidence
        signal_type = SignalType.HOLD
        confidence = abs(weighted_score)

        if weighted_score > 0.3:
            signal_type = SignalType.STRONG_BUY if weighted_score > 0.6 else SignalType.BUY
        elif weighted_score < -0.3:
            signal_type = SignalType.STRONG_SELL if weighted_score < -0.6 else SignalType.SELL

        # Calculate target and stop
        atr = current.get('atr', close * 0.02)
        stop_loss = close - (2 * atr) if signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else close + (2 * atr)
        take_profit = close + (3 * atr) if signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else close - (3 * atr)

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=min(confidence, 1.0),
            strategy=self.name,
            price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'rsi': round(rsi, 2),
                'macd': round(macd, 4),
                'macd_hist': round(macd_hist, 4),
                'sma': round(sma, 2),
                'weighted_score': round(weighted_score, 3),
                'signals': {
                    'rsi_signal': 'oversold' if rsi < self.rsi_oversold else 'overbought' if rsi > self.rsi_overbought else 'neutral',
                    'macd_signal': 'bullish' if macd_hist > 0 else 'bearish',
                    'trend_signal': 'bullish' if close > sma else 'bearish'
                }
            }
        )

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-specific indicators"""
        df = super().calculate_indicators(data)

        # Add strategy-specific SMA if not present
        if f'sma_{self.sma_period}' not in df.columns:
            df[f'sma_{self.sma_period}'] = df['close'].rolling(window=self.sma_period).mean()

        # Rate of Change
        df['roc'] = df['close'].pct_change(periods=self.lookback) * 100

        # Momentum indicator
        df['momentum'] = df['close'] - df['close'].shift(self.lookback)

        return df
