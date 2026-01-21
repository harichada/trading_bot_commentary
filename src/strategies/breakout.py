"""
Breakout Strategy

Identifies and trades price breakouts from consolidation patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from .base import BaseStrategy, Signal, SignalType


class BreakoutStrategy(BaseStrategy):
    """
    Breakout trading strategy.

    Identifies consolidation patterns and trades breakouts with
    volume confirmation.

    Uses:
    - Support/resistance levels from recent highs/lows
    - Volume surge detection
    - ATR for volatility expansion
    - Donchian channels for breakout levels

    Signals:
    - BUY: Price breaks above resistance with volume surge
    - SELL: Price breaks below support with volume surge
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="breakout", config=config)

        # Strategy parameters
        self.channel_period = self.config.get('channel_period', 20)
        self.consolidation_period = self.config.get('consolidation_period', 10)
        self.volume_threshold = self.config.get('volume_threshold', 1.5)
        self.breakout_threshold = self.config.get('breakout_threshold', 0.005)  # 0.5%
        self.atr_period = self.config.get('atr_period', 14)
        self.atr_multiplier = self.config.get('atr_multiplier', 1.5)

    @property
    def min_periods(self) -> int:
        return max(self.channel_period, self.consolidation_period) + 20

    def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        """Generate trading signal based on breakout patterns"""
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
        previous = df.iloc[-2] if len(df) > 1 else current

        close = current['close']
        prev_close = previous['close']

        # Get indicator values
        upper_channel = current.get('donchian_upper', close * 1.02)
        lower_channel = current.get('donchian_lower', close * 0.98)
        middle_channel = current.get('donchian_middle', close)

        prev_upper = previous.get('donchian_upper', prev_close * 1.02)
        prev_lower = previous.get('donchian_lower', prev_close * 0.98)

        volume_ratio = current.get('volume_ratio', 1.0)
        atr = current.get('atr', close * 0.02)
        atr_ratio = current.get('atr_ratio', 1.0)
        consolidation = current.get('consolidation', 0)

        # Detect breakout
        breakout_up = close > prev_upper and prev_close <= prev_upper
        breakout_down = close < prev_lower and prev_close >= prev_lower

        # Calculate signals
        signals = []
        weights = []

        # Breakout detection
        if breakout_up:
            signals.append(1)
            weights.append(0.4)
        elif breakout_down:
            signals.append(-1)
            weights.append(0.4)
        else:
            # Price position in channel
            channel_range = upper_channel - lower_channel
            if channel_range > 0:
                channel_position = (close - lower_channel) / channel_range
                if channel_position > 0.9:
                    signals.append(0.5)  # Near top, potential breakout
                elif channel_position < 0.1:
                    signals.append(-0.5)  # Near bottom
                else:
                    signals.append(0)
            else:
                signals.append(0)
            weights.append(0.2)

        # Volume confirmation
        if volume_ratio > self.volume_threshold:
            if breakout_up:
                signals.append(1)
            elif breakout_down:
                signals.append(-1)
            else:
                signals.append(0)
            weights.append(0.3)
        else:
            signals.append(0)
            weights.append(0.15)

        # Volatility expansion (breakout often comes with ATR expansion)
        if atr_ratio > 1.2:  # Volatility expanding
            if breakout_up or breakout_down:
                signals.append(1 if breakout_up else -1)
            else:
                signals.append(0)
            weights.append(0.2)
        else:
            signals.append(0)
            weights.append(0.1)

        # Consolidation quality (tighter = better breakout potential)
        if consolidation < 0.03:  # Tight consolidation
            if breakout_up or breakout_down:
                signals.append(1 if breakout_up else -1)
                weights.append(0.15)
            else:
                signals.append(0)
                weights.append(0.05)
        else:
            signals.append(0)
            weights.append(0.05)

        # Calculate weighted score
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(signals, weights)) / total_weight

        # Determine signal type
        signal_type = SignalType.HOLD
        confidence = abs(weighted_score)

        # Breakout strategy requires stronger signals
        if weighted_score > 0.4:
            signal_type = SignalType.STRONG_BUY if weighted_score > 0.7 else SignalType.BUY
        elif weighted_score < -0.4:
            signal_type = SignalType.STRONG_SELL if weighted_score < -0.7 else SignalType.SELL

        # Calculate targets based on ATR and channel width
        channel_width = upper_channel - lower_channel

        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            target_price = close + channel_width  # Target = channel width extension
            stop_loss = lower_channel - (atr * 0.5)  # Stop below channel
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            target_price = close - channel_width
            stop_loss = upper_channel + (atr * 0.5)
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
                'upper_channel': round(upper_channel, 2),
                'lower_channel': round(lower_channel, 2),
                'channel_width_pct': round((channel_width / close) * 100, 2) if close > 0 else 0,
                'volume_ratio': round(volume_ratio, 2),
                'atr': round(atr, 2),
                'atr_ratio': round(atr_ratio, 2),
                'consolidation': round(consolidation * 100, 2),
                'breakout_up': breakout_up,
                'breakout_down': breakout_down,
                'weighted_score': round(weighted_score, 3),
                'signals': {
                    'breakout': 'bullish' if breakout_up else 'bearish' if breakout_down else 'none',
                    'volume': 'confirmed' if volume_ratio > self.volume_threshold else 'weak',
                    'volatility': 'expanding' if atr_ratio > 1.2 else 'contracting' if atr_ratio < 0.8 else 'normal'
                }
            }
        )

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate breakout-specific indicators"""
        df = super().calculate_indicators(data)

        # Donchian Channels
        df['donchian_upper'] = df['high'].rolling(window=self.channel_period).max()
        df['donchian_lower'] = df['low'].rolling(window=self.channel_period).min()
        df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2

        # Volume ratio (current vs average)
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1)

        # ATR ratio (current vs average)
        if 'atr' in df.columns:
            df['atr_sma'] = df['atr'].rolling(window=20).mean()
            df['atr_ratio'] = df['atr'] / df['atr_sma'].replace(0, np.inf)

        # Consolidation measure (range as % of price)
        high_range = df['high'].rolling(window=self.consolidation_period).max()
        low_range = df['low'].rolling(window=self.consolidation_period).min()
        df['consolidation'] = (high_range - low_range) / df['close']

        # Recent highs and lows for support/resistance
        df['recent_high'] = df['high'].rolling(window=self.channel_period).max()
        df['recent_low'] = df['low'].rolling(window=self.channel_period).min()

        return df
