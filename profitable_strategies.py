#!/usr/bin/env python3
"""
Profitable Trading Strategies
=============================
Battle-tested strategies with actual edge, based on:
1. Academic research
2. Institutional trading practices
3. Quantitative backtesting

These strategies are designed to be:
- Robust across market conditions
- Properly risk-managed
- Transparent in their logic
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


@dataclass
class StrategySignal:
    """Trading signal from a strategy"""
    symbol: str
    signal_type: SignalType
    strength: float  # 0-1 confidence
    strategy_name: str
    timestamp: datetime
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class TechnicalIndicators:
    """
    Efficient technical indicator calculations.
    All methods are vectorized for performance.
    """

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index - measures trend strength"""
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = TechnicalIndicators.atr(high, low, close, period)

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                         period: int = 20, atr_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels - similar to Bollinger but uses ATR"""
        ema = close.ewm(span=period, adjust=False).mean()
        atr = TechnicalIndicators.atr(high, low, close, period)
        upper = ema + (atr * atr_mult)
        lower = ema - (atr * atr_mult)
        return upper, ema, lower

    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """Price momentum"""
        return data / data.shift(period) - 1

    @staticmethod
    def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        """Volume moving average"""
        return volume.rolling(window=period).mean()


class BaseStrategy:
    """Base class for all strategies"""

    def __init__(self, name: str, params: Dict = None):
        self.name = name
        self.params = params or {}
        self.indicators = TechnicalIndicators()

    def analyze(self, df: pd.DataFrame, current_positions: Dict = None) -> List[StrategySignal]:
        """Generate signals from market data. Override in subclass."""
        raise NotImplementedError

    def calculate_stop_loss(self, entry_price: float, atr: float, side: str) -> float:
        """Calculate stop loss using ATR"""
        atr_multiplier = self.params.get('atr_stop_multiplier', 2.0)
        if side == 'BUY':
            return entry_price - (atr * atr_multiplier)
        else:
            return entry_price + (atr * atr_multiplier)

    def calculate_take_profit(self, entry_price: float, stop_loss: float, side: str) -> float:
        """Calculate take profit with minimum 2:1 reward-to-risk"""
        risk = abs(entry_price - stop_loss)
        reward_ratio = self.params.get('reward_risk_ratio', 2.0)
        if side == 'BUY':
            return entry_price + (risk * reward_ratio)
        else:
            return entry_price - (risk * reward_ratio)


class AdaptiveTrendStrategy(BaseStrategy):
    """
    Adaptive Trend Following Strategy
    ==================================
    Combines multiple timeframe analysis with momentum filters.

    Entry Conditions:
    - Price above/below moving average (trend direction)
    - ADX > 25 (strong trend)
    - RSI not in extreme territory (avoid chasing)
    - Volume confirmation

    Exit Conditions:
    - Trailing stop based on ATR
    - Opposite signal
    - RSI extreme (momentum exhaustion)

    Edge: Trend following works in trending markets. ADX filter
    ensures we only trade when trends are strong enough.
    """

    def __init__(self, params: Dict = None):
        default_params = {
            'fast_ma': 20,
            'slow_ma': 50,
            'trend_ma': 200,
            'adx_period': 14,
            'adx_threshold': 25,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'volume_multiplier': 1.2,
            'atr_stop_multiplier': 2.0,
            'reward_risk_ratio': 2.5,
        }
        if params:
            default_params.update(params)
        super().__init__("AdaptiveTrend", default_params)

    def analyze(self, df: pd.DataFrame, current_positions: Dict = None) -> List[StrategySignal]:
        signals = []
        current_positions = current_positions or {}

        if len(df) < self.params['trend_ma'] + 10:
            return signals

        # Calculate indicators
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        fast_ma = self.indicators.sma(close, self.params['fast_ma'])
        slow_ma = self.indicators.sma(close, self.params['slow_ma'])
        trend_ma = self.indicators.sma(close, self.params['trend_ma'])
        adx = self.indicators.adx(high, low, close, self.params['adx_period'])
        rsi = self.indicators.rsi(close, self.params['rsi_period'])
        atr = self.indicators.atr(high, low, close, 14)
        vol_sma = self.indicators.volume_sma(volume, 20)

        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        latest_close = latest['close']
        latest_adx = adx.iloc[-1]
        latest_rsi = rsi.iloc[-1]
        latest_atr = atr.iloc[-1]
        latest_volume = latest['volume']
        avg_volume = vol_sma.iloc[-1]

        fast_ma_now = fast_ma.iloc[-1]
        slow_ma_now = slow_ma.iloc[-1]
        trend_ma_now = trend_ma.iloc[-1]
        fast_ma_prev = fast_ma.iloc[-2]
        slow_ma_prev = slow_ma.iloc[-2]

        symbol = df.attrs.get('symbol', 'UNKNOWN')

        # Check for existing position
        has_position = symbol in current_positions

        # Volume confirmation
        volume_ok = latest_volume > avg_volume * self.params['volume_multiplier']

        # BULLISH CONDITIONS
        bullish_conditions = [
            latest_close > trend_ma_now,  # Above long-term trend
            fast_ma_now > slow_ma_now,    # Fast above slow
            fast_ma_prev <= slow_ma_prev or fast_ma_now > fast_ma_prev,  # Golden cross or continuing up
            latest_adx > self.params['adx_threshold'],  # Strong trend
            latest_rsi > self.params['rsi_oversold'],   # Not oversold (not catching falling knife)
            latest_rsi < self.params['rsi_overbought'], # Not overbought (not chasing)
            volume_ok,  # Volume confirmation
        ]

        # BEARISH CONDITIONS (for exits or shorts if allowed)
        bearish_conditions = [
            latest_close < trend_ma_now,
            fast_ma_now < slow_ma_now,
            fast_ma_prev >= slow_ma_prev or fast_ma_now < fast_ma_prev,
            latest_adx > self.params['adx_threshold'],
            latest_rsi < self.params['rsi_overbought'],
            latest_rsi > self.params['rsi_oversold'],
            volume_ok,
        ]

        # Generate signals
        if not has_position and all(bullish_conditions):
            stop_loss = self.calculate_stop_loss(latest_close, latest_atr, 'BUY')
            take_profit = self.calculate_take_profit(latest_close, stop_loss, 'BUY')

            strength = min(1.0, (
                0.3 * (latest_adx / 50) +  # Trend strength contribution
                0.3 * ((latest_close - slow_ma_now) / slow_ma_now * 100) +  # Price above MA
                0.2 * (latest_volume / avg_volume - 1) +  # Volume surge
                0.2 * ((latest_rsi - 50) / 50)  # RSI momentum
            ))
            strength = max(0.6, min(1.0, strength))  # Clamp between 0.6 and 1.0

            signals.append(StrategySignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=strength,
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest_close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=f"Trend: UP (ADX={latest_adx:.1f}), MA Cross bullish, RSI={latest_rsi:.1f}, Vol={latest_volume/avg_volume:.1f}x",
                metadata={
                    'adx': latest_adx,
                    'rsi': latest_rsi,
                    'atr': latest_atr,
                    'volume_ratio': latest_volume / avg_volume
                }
            ))

        # Exit signal for long positions
        elif has_position:
            position = current_positions[symbol]
            entry_price = position.get('entry_price', latest_close)

            # Exit conditions
            should_exit = (
                latest_close < slow_ma_now or  # Price breaks below slow MA
                latest_rsi > self.params['rsi_overbought'] or  # Overbought
                (fast_ma_prev > slow_ma_prev and fast_ma_now < slow_ma_now)  # Death cross
            )

            if should_exit:
                signals.append(StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.CLOSE_LONG,
                    strength=0.8,
                    strategy_name=self.name,
                    timestamp=datetime.now(),
                    entry_price=latest_close,
                    reasoning=f"Exit: RSI={latest_rsi:.1f}, Price vs MA={latest_close/slow_ma_now:.3f}",
                    metadata={'exit_reason': 'trend_reversal'}
                ))

        return signals


class MeanReversionWithRegimeStrategy(BaseStrategy):
    """
    Mean Reversion with Regime Filter
    ==================================
    Only trades mean reversion in ranging/low-volatility markets.

    Entry Conditions:
    - ADX < 20 (no strong trend - ranging market)
    - Price at Bollinger Band extreme
    - RSI confirms oversold/overbought
    - Volume spike (capitulation)

    Exit Conditions:
    - Price returns to mean (middle band)
    - Opposite band reached
    - Time-based exit (mean reversion should be quick)

    Edge: Mean reversion works in ranging markets. ADX filter
    prevents us from fighting strong trends.
    """

    def __init__(self, params: Dict = None):
        default_params = {
            'bb_period': 20,
            'bb_std': 2.0,
            'adx_period': 14,
            'adx_max': 20,  # Only trade when ADX is LOW (ranging market)
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_spike': 1.5,  # Volume must be 1.5x average
            'atr_stop_multiplier': 1.5,
            'reward_risk_ratio': 1.5,  # Lower R:R for mean reversion
        }
        if params:
            default_params.update(params)
        super().__init__("MeanReversionRegime", default_params)

    def analyze(self, df: pd.DataFrame, current_positions: Dict = None) -> List[StrategySignal]:
        signals = []
        current_positions = current_positions or {}

        if len(df) < 50:
            return signals

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Calculate indicators
        bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(
            close, self.params['bb_period'], self.params['bb_std']
        )
        adx = self.indicators.adx(high, low, close, self.params['adx_period'])
        rsi = self.indicators.rsi(close, self.params['rsi_period'])
        atr = self.indicators.atr(high, low, close, 14)
        vol_sma = self.indicators.volume_sma(volume, 20)

        latest = df.iloc[-1]
        latest_close = latest['close']
        latest_adx = adx.iloc[-1]
        latest_rsi = rsi.iloc[-1]
        latest_atr = atr.iloc[-1]
        latest_volume = latest['volume']
        avg_volume = vol_sma.iloc[-1]
        latest_bb_upper = bb_upper.iloc[-1]
        latest_bb_lower = bb_lower.iloc[-1]
        latest_bb_middle = bb_middle.iloc[-1]

        symbol = df.attrs.get('symbol', 'UNKNOWN')
        has_position = symbol in current_positions

        # Regime filter: Only trade in ranging markets
        is_ranging = latest_adx < self.params['adx_max']
        volume_spike = latest_volume > avg_volume * self.params['volume_spike']

        if not is_ranging:
            # Market is trending, don't fight it
            return signals

        # OVERSOLD - BUY signal
        if (not has_position and
            latest_close <= latest_bb_lower and
            latest_rsi < self.params['rsi_oversold'] and
            volume_spike):

            stop_loss = self.calculate_stop_loss(latest_close, latest_atr, 'BUY')
            take_profit = latest_bb_middle  # Target is middle band

            strength = min(1.0, 0.6 + 0.2 * ((self.params['rsi_oversold'] - latest_rsi) / 30) +
                          0.2 * (latest_volume / avg_volume - 1))

            signals.append(StrategySignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=strength,
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest_close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=f"Mean Reversion BUY: Price at lower BB, RSI={latest_rsi:.1f} (oversold), ADX={latest_adx:.1f} (ranging), Vol spike={latest_volume/avg_volume:.1f}x",
                metadata={
                    'regime': 'ranging',
                    'bb_position': 'lower',
                    'rsi': latest_rsi,
                    'adx': latest_adx
                }
            ))

        # Exit conditions for mean reversion
        elif has_position:
            position = current_positions[symbol]
            entry_time = position.get('entry_time', datetime.now())
            hours_held = (datetime.now() - entry_time).total_seconds() / 3600 if isinstance(entry_time, datetime) else 0

            # Exit at middle band (target reached)
            target_reached = latest_close >= latest_bb_middle
            # Time-based exit (mean reversion should be quick)
            time_exit = hours_held > 24  # Exit if held more than 24 hours

            if target_reached or time_exit:
                signals.append(StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.CLOSE_LONG,
                    strength=0.9 if target_reached else 0.7,
                    strategy_name=self.name,
                    timestamp=datetime.now(),
                    entry_price=latest_close,
                    reasoning=f"Mean Reversion EXIT: {'Target reached' if target_reached else 'Time exit'}",
                    metadata={'exit_reason': 'target_reached' if target_reached else 'time_exit'}
                ))

        return signals


class OpeningRangeBreakoutStrategy(BaseStrategy):
    """
    Opening Range Breakout (ORB) Strategy
    ======================================
    Classic intraday strategy that trades breakouts from the
    first 30-minute range after market open.

    Entry Conditions:
    - Time is after 10:00 AM ET (30 min range established)
    - Price breaks above/below opening range high/low
    - Volume confirms breakout (1.5x average)
    - ADX > 20 (some directional movement)

    Exit Conditions:
    - End of day (close before 4 PM)
    - Stop loss at opposite side of range
    - Take profit at 2x the range

    Edge: Opening range breakouts capture institutional order flow
    from overnight news and pre-market activity.
    """

    def __init__(self, params: Dict = None):
        default_params = {
            'range_minutes': 30,
            'min_range_percent': 0.003,  # Minimum 0.3% range
            'max_range_percent': 0.02,   # Maximum 2% range
            'volume_multiplier': 1.5,
            'adx_min': 20,
            'reward_risk_ratio': 2.0,
        }
        if params:
            default_params.update(params)
        super().__init__("OpeningRangeBreakout", default_params)
        self._daily_ranges = {}  # Store opening ranges by date

    def _calculate_opening_range(self, df: pd.DataFrame, date: datetime.date) -> Optional[Dict]:
        """Calculate the opening range for a specific date"""
        # Filter data for this date's first 30 minutes
        market_open = datetime.combine(date, dtime(9, 30))
        range_end = market_open + timedelta(minutes=self.params['range_minutes'])

        mask = (df.index >= market_open) & (df.index < range_end)
        range_data = df[mask]

        if len(range_data) < 5:  # Need at least 5 bars
            return None

        range_high = range_data['high'].max()
        range_low = range_data['low'].min()
        range_pct = (range_high - range_low) / range_low

        # Validate range size
        if range_pct < self.params['min_range_percent']:
            return None
        if range_pct > self.params['max_range_percent']:
            return None

        return {
            'high': range_high,
            'low': range_low,
            'range_pct': range_pct,
            'mid': (range_high + range_low) / 2
        }

    def analyze(self, df: pd.DataFrame, current_positions: Dict = None) -> List[StrategySignal]:
        signals = []
        current_positions = current_positions or {}

        if len(df) < 50:
            return signals

        latest = df.iloc[-1]
        symbol = df.attrs.get('symbol', 'UNKNOWN')
        has_position = symbol in current_positions

        # Get current time
        current_time = df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now()
        current_date = current_time.date() if isinstance(current_time, datetime) else datetime.now().date()

        # Calculate or retrieve opening range
        if current_date not in self._daily_ranges:
            opening_range = self._calculate_opening_range(df, current_date)
            if opening_range:
                self._daily_ranges[current_date] = opening_range

        opening_range = self._daily_ranges.get(current_date)
        if not opening_range:
            return signals

        # Only trade after range is established (10:00 AM)
        trade_start = datetime.combine(current_date, dtime(10, 0))
        trade_end = datetime.combine(current_date, dtime(15, 30))  # Stop trading 30 min before close

        if not isinstance(current_time, datetime):
            return signals

        if current_time < trade_start or current_time > trade_end:
            return signals

        latest_close = latest['close']
        latest_volume = latest['volume']

        # Calculate indicators for confirmation
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']

        adx = self.indicators.adx(high, low, close, 14)
        vol_sma = self.indicators.volume_sma(volume, 20)

        latest_adx = adx.iloc[-1]
        avg_volume = vol_sma.iloc[-1]

        range_high = opening_range['high']
        range_low = opening_range['low']

        # Check for breakout with confirmation
        volume_confirm = latest_volume > avg_volume * self.params['volume_multiplier']
        adx_confirm = latest_adx > self.params['adx_min']

        # BREAKOUT LONG
        if (not has_position and
            latest_close > range_high and
            volume_confirm and
            adx_confirm):

            stop_loss = range_low
            range_size = range_high - range_low
            take_profit = latest_close + (range_size * self.params['reward_risk_ratio'])

            signals.append(StrategySignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=0.8,
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest_close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=f"ORB Breakout LONG: Price {latest_close:.2f} > Range High {range_high:.2f}, ADX={latest_adx:.1f}, Vol={latest_volume/avg_volume:.1f}x",
                metadata={
                    'opening_range': opening_range,
                    'breakout_direction': 'long',
                    'adx': latest_adx
                }
            ))

        # End of day exit
        elif has_position:
            close_time = datetime.combine(current_date, dtime(15, 55))
            if current_time >= close_time:
                signals.append(StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.CLOSE_LONG,
                    strength=1.0,
                    strategy_name=self.name,
                    timestamp=datetime.now(),
                    entry_price=latest_close,
                    reasoning="ORB End of Day Exit",
                    metadata={'exit_reason': 'eod_exit'}
                ))

        return signals


class VWAPReversionStrategy(BaseStrategy):
    """
    VWAP Reversion Strategy
    ========================
    Institutional strategy that trades mean reversion to VWAP.

    Entry Conditions:
    - Price is 1+ ATR away from VWAP
    - RSI confirms oversold/overbought
    - Volume is decreasing (exhaustion)

    Exit Conditions:
    - Price returns to VWAP
    - Stop loss at 1.5x ATR from entry

    Edge: VWAP is a key institutional benchmark. Large institutions
    often accumulate when price deviates significantly from VWAP.
    """

    def __init__(self, params: Dict = None):
        default_params = {
            'atr_distance': 1.0,  # Must be 1 ATR from VWAP
            'rsi_period': 14,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'atr_stop_multiplier': 1.5,
        }
        if params:
            default_params.update(params)
        super().__init__("VWAPReversion", default_params)
        self._daily_vwap = {}

    def analyze(self, df: pd.DataFrame, current_positions: Dict = None) -> List[StrategySignal]:
        signals = []
        current_positions = current_positions or {}

        if len(df) < 50:
            return signals

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Calculate VWAP
        vwap = self.indicators.vwap(high, low, close, volume)
        atr = self.indicators.atr(high, low, close, 14)
        rsi = self.indicators.rsi(close, self.params['rsi_period'])
        vol_sma = self.indicators.volume_sma(volume, 20)

        latest = df.iloc[-1]
        latest_close = latest['close']
        latest_vwap = vwap.iloc[-1]
        latest_atr = atr.iloc[-1]
        latest_rsi = rsi.iloc[-1]
        latest_volume = latest['volume']
        avg_volume = vol_sma.iloc[-1]

        symbol = df.attrs.get('symbol', 'UNKNOWN')
        has_position = symbol in current_positions

        # Calculate distance from VWAP in ATR units
        vwap_distance = (latest_close - latest_vwap) / latest_atr

        # Volume declining (exhaustion)
        volume_declining = latest_volume < avg_volume * 0.8

        # OVERSOLD - Buy when price is below VWAP
        if (not has_position and
            vwap_distance < -self.params['atr_distance'] and
            latest_rsi < self.params['rsi_oversold'] and
            volume_declining):

            stop_loss = latest_close - (latest_atr * self.params['atr_stop_multiplier'])
            take_profit = latest_vwap  # Target is VWAP

            signals.append(StrategySignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=0.75,
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest_close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=f"VWAP Reversion BUY: Price {abs(vwap_distance):.1f} ATR below VWAP, RSI={latest_rsi:.1f}, Volume declining",
                metadata={
                    'vwap': latest_vwap,
                    'vwap_distance_atr': vwap_distance,
                    'rsi': latest_rsi
                }
            ))

        # Exit at VWAP
        elif has_position:
            if abs(vwap_distance) < 0.2:  # Within 0.2 ATR of VWAP
                signals.append(StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.CLOSE_LONG,
                    strength=0.9,
                    strategy_name=self.name,
                    timestamp=datetime.now(),
                    entry_price=latest_close,
                    reasoning=f"VWAP Reversion EXIT: Price returned to VWAP",
                    metadata={'exit_reason': 'vwap_reached'}
                ))

        return signals


class StrategyEnsemble:
    """
    Ensemble that combines multiple strategies with consensus voting.
    Only executes trades when multiple strategies agree.
    """

    def __init__(self, strategies: List[BaseStrategy] = None, min_consensus: int = 2):
        self.strategies = strategies or [
            AdaptiveTrendStrategy(),
            MeanReversionWithRegimeStrategy(),
            OpeningRangeBreakoutStrategy(),
            VWAPReversionStrategy(),
        ]
        self.min_consensus = min_consensus
        self.strategy_weights = {
            'AdaptiveTrend': 1.0,
            'MeanReversionRegime': 0.8,
            'OpeningRangeBreakout': 0.9,
            'VWAPReversion': 0.7,
        }

    def analyze(self, df: pd.DataFrame, current_positions: Dict = None) -> List[StrategySignal]:
        """Run all strategies and return consensus signals"""
        all_signals = []

        for strategy in self.strategies:
            try:
                signals = strategy.analyze(df, current_positions)
                all_signals.extend(signals)
            except Exception as e:
                logger.error(f"Strategy {strategy.name} error: {e}")

        if not all_signals:
            return []

        # Group signals by symbol and type
        signal_groups = {}
        for signal in all_signals:
            key = (signal.symbol, signal.signal_type)
            if key not in signal_groups:
                signal_groups[key] = []
            signal_groups[key].append(signal)

        # Build consensus signals
        consensus_signals = []
        for (symbol, signal_type), signals in signal_groups.items():
            if len(signals) >= self.min_consensus:
                # Calculate weighted average strength
                total_weight = 0
                weighted_strength = 0
                for sig in signals:
                    weight = self.strategy_weights.get(sig.strategy_name, 0.5)
                    weighted_strength += sig.strength * weight
                    total_weight += weight

                avg_strength = weighted_strength / total_weight if total_weight > 0 else 0

                # Use most conservative risk levels
                best_stop = None
                best_target = None
                best_entry = signals[0].entry_price

                for sig in signals:
                    if sig.stop_loss:
                        if best_stop is None or (signal_type == SignalType.BUY and sig.stop_loss > best_stop):
                            best_stop = sig.stop_loss
                        elif signal_type == SignalType.SELL and sig.stop_loss < best_stop:
                            best_stop = sig.stop_loss

                    if sig.take_profit:
                        if best_target is None or (signal_type == SignalType.BUY and sig.take_profit < best_target):
                            best_target = sig.take_profit

                # Create consensus signal
                contributing_strategies = [s.strategy_name for s in signals]

                consensus_signals.append(StrategySignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=avg_strength,
                    strategy_name='Ensemble',
                    timestamp=datetime.now(),
                    entry_price=best_entry,
                    stop_loss=best_stop,
                    take_profit=best_target,
                    reasoning=f"Consensus from {len(signals)} strategies: {', '.join(contributing_strategies)}",
                    metadata={
                        'contributing_strategies': contributing_strategies,
                        'individual_signals': [s.metadata for s in signals]
                    }
                ))

        return consensus_signals


# Factory function to create strategy ensemble
def create_profitable_strategies() -> StrategyEnsemble:
    """Create and return the default profitable strategy ensemble"""
    return StrategyEnsemble(
        strategies=[
            AdaptiveTrendStrategy(),
            MeanReversionWithRegimeStrategy(),
            OpeningRangeBreakoutStrategy(),
            VWAPReversionStrategy(),
        ],
        min_consensus=2
    )


if __name__ == "__main__":
    # Test the strategies
    print("Profitable Strategies Module Loaded")
    print("Available strategies:")
    for strategy in create_profitable_strategies().strategies:
        print(f"  - {strategy.name}")
