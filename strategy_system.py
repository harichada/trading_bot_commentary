#!/usr/bin/env python3
"""
Professional Trading Strategy System
Modular, configurable, and extensible trading strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import json
import numpy as np
import pandas as pd
from datetime import datetime
import ta
import logging

logger = logging.getLogger('StrategySystem')

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"

@dataclass
class StrategySignal:
    symbol: str
    signal_type: SignalType
    strength: float  # 0-1 confidence
    strategy_name: str
    timestamp: datetime
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    name: str
    enabled: bool = True
    weight: float = 1.0  # Weight in ensemble
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk_parameters: Dict[str, float] = field(default_factory=lambda: {
        'max_position_size': 0.1,  # 10% of portfolio
        'stop_loss_pct': 0.01,     # 1% stop loss (tightened for day trading)
        'take_profit_pct': 0.025,  # 2.5% take profit
        'max_positions': 5
    })

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
        self.parameters = config.parameters
        self.risk_parameters = config.risk_parameters
        
    @abstractmethod
    def analyze(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        """Analyze market data and generate signals"""
        pass
    
    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        pass
    
    @abstractmethod
    def get_required_lookback(self) -> int:
        """Return required lookback period in candles"""
        pass
    
    def calculate_position_size(self, signal: StrategySignal, portfolio_value: float) -> float:
        """Calculate position size based on risk parameters"""
        max_size = portfolio_value * self.risk_parameters['max_position_size']
        # Can be overridden for more sophisticated sizing
        return max_size
    
    def set_risk_levels(self, signal: StrategySignal, current_price: float) -> StrategySignal:
        """Set stop loss and take profit levels"""
        if signal.signal_type in [SignalType.BUY]:
            signal.stop_loss = current_price * (1 - self.risk_parameters['stop_loss_pct'])
            signal.take_profit = current_price * (1 + self.risk_parameters['take_profit_pct'])
        elif signal.signal_type in [SignalType.SELL]:
            signal.stop_loss = current_price * (1 + self.risk_parameters['stop_loss_pct'])
            signal.take_profit = current_price * (1 - self.risk_parameters['take_profit_pct'])
        return signal

class MovingAverageCrossStrategy(BaseStrategy):
    """Classic moving average crossover strategy"""
    
    def get_required_indicators(self) -> List[str]:
        return ['sma_fast', 'sma_slow', 'volume_sma']
    
    def get_required_lookback(self) -> int:
        return self.parameters.get('slow_period', 50) + 10
    
    def analyze(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        signals = []
        
        fast_period = self.parameters.get('fast_period', 20)
        slow_period = self.parameters.get('slow_period', 50)
        
        # Calculate moving averages
        market_data['sma_fast'] = market_data['close'].rolling(fast_period).mean()
        market_data['sma_slow'] = market_data['close'].rolling(slow_period).mean()
        market_data['volume_sma'] = market_data['volume'].rolling(20).mean()
        
        latest = market_data.iloc[-1]
        prev = market_data.iloc[-2]
        
        symbol = market_data.index.name or 'UNKNOWN'
        
        # Check for crossover
        if prev['sma_fast'] <= prev['sma_slow'] and latest['sma_fast'] > latest['sma_slow']:
            # Golden cross - buy signal
            if latest['volume'] > latest['volume_sma'] * 1.2:  # Volume confirmation
                signal = StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=0.8,
                    strategy_name=self.name,
                    timestamp=datetime.now(),
                    entry_price=latest['close'],
                    metadata={
                        'fast_ma': latest['sma_fast'],
                        'slow_ma': latest['sma_slow'],
                        'volume_ratio': latest['volume'] / latest['volume_sma']
                    }
                )
                signals.append(self.set_risk_levels(signal, latest['close']))
                
        elif prev['sma_fast'] >= prev['sma_slow'] and latest['sma_fast'] < latest['sma_slow']:
            # Death cross - sell signal
            signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.SELL if symbol not in current_positions else SignalType.CLOSE_LONG,
                strength=0.8,
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest['close'],
                metadata={
                    'fast_ma': latest['sma_fast'],
                    'slow_ma': latest['sma_slow']
                }
            )
            signals.append(self.set_risk_levels(signal, latest['close']))
        
        return signals

class RSIMomentumStrategy(BaseStrategy):
    """RSI-based momentum strategy with divergence detection"""
    
    def get_required_indicators(self) -> List[str]:
        return ['rsi', 'rsi_ma', 'price_ma']
    
    def get_required_lookback(self) -> int:
        return 100
    
    def analyze(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        signals = []
        
        rsi_period = self.parameters.get('rsi_period', 14)
        oversold = self.parameters.get('oversold_level', 30)
        overbought = self.parameters.get('overbought_level', 70)
        
        # Calculate RSI
        market_data['rsi'] = ta.momentum.RSIIndicator(
            close=market_data['close'], 
            window=rsi_period
        ).rsi()
        
        market_data['rsi_ma'] = market_data['rsi'].rolling(5).mean()
        market_data['price_ma'] = market_data['close'].rolling(20).mean()
        
        latest = market_data.iloc[-1]
        symbol = market_data.index.name or 'UNKNOWN'
        
        # Check for divergences
        if self._check_bullish_divergence(market_data):
            signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=0.9,
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest['close'],
                metadata={
                    'rsi': latest['rsi'],
                    'divergence': 'bullish'
                }
            )
            signals.append(self.set_risk_levels(signal, latest['close']))
            
        elif self._check_bearish_divergence(market_data):
            signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.SELL if symbol not in current_positions else SignalType.CLOSE_LONG,
                strength=0.9,
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest['close'],
                metadata={
                    'rsi': latest['rsi'],
                    'divergence': 'bearish'
                }
            )
            signals.append(self.set_risk_levels(signal, latest['close']))
        
        # Standard RSI signals
        elif latest['rsi'] < oversold and latest['rsi'] > latest['rsi_ma']:
            signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=0.7,
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest['close'],
                metadata={'rsi': latest['rsi']}
            )
            signals.append(self.set_risk_levels(signal, latest['close']))
            
        elif latest['rsi'] > overbought and latest['rsi'] < latest['rsi_ma']:
            signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.SELL if symbol not in current_positions else SignalType.CLOSE_LONG,
                strength=0.7,
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest['close'],
                metadata={'rsi': latest['rsi']}
            )
            signals.append(self.set_risk_levels(signal, latest['close']))
        
        return signals
    
    def _check_bullish_divergence(self, data: pd.DataFrame, lookback: int = 20) -> bool:
        """Check for bullish divergence between price and RSI"""
        recent_data = data.tail(lookback)
        
        # Find recent lows
        price_lows = recent_data['close'].rolling(5).min()
        rsi_lows = recent_data['rsi'].rolling(5).min()
        
        # Check if price made lower low but RSI made higher low
        if len(price_lows) > 10:
            recent_price_low = price_lows.iloc[-1]
            prev_price_low = price_lows.iloc[-10:-5].min()
            recent_rsi_low = rsi_lows.iloc[-1]
            prev_rsi_low = rsi_lows.iloc[-10:-5].min()
            
            if recent_price_low < prev_price_low and recent_rsi_low > prev_rsi_low:
                return True
        
        return False
    
    def _check_bearish_divergence(self, data: pd.DataFrame, lookback: int = 20) -> bool:
        """Check for bearish divergence between price and RSI"""
        recent_data = data.tail(lookback)
        
        # Find recent highs
        price_highs = recent_data['close'].rolling(5).max()
        rsi_highs = recent_data['rsi'].rolling(5).max()
        
        # Check if price made higher high but RSI made lower high
        if len(price_highs) > 10:
            recent_price_high = price_highs.iloc[-1]
            prev_price_high = price_highs.iloc[-10:-5].max()
            recent_rsi_high = rsi_highs.iloc[-1]
            prev_rsi_high = rsi_highs.iloc[-10:-5].max()
            
            if recent_price_high > prev_price_high and recent_rsi_high < prev_rsi_high:
                return True
        
        return False

class BollingerBandStrategy(BaseStrategy):
    """Bollinger Band squeeze and breakout strategy"""
    
    def get_required_indicators(self) -> List[str]:
        return ['bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'atr', 'rsi']
    
    def get_required_lookback(self) -> int:
        return 50
    
    def analyze(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        signals = []
        
        bb_period = self.parameters.get('bb_period', 20)
        bb_std = self.parameters.get('bb_std', 2)
        
        # Calculate Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close=market_data['close'],
            window=bb_period,
            window_dev=bb_std
        )
        
        market_data['bb_upper'] = bb.bollinger_hband()
        market_data['bb_lower'] = bb.bollinger_lband()
        market_data['bb_middle'] = bb.bollinger_mavg()
        market_data['bb_width'] = bb.bollinger_wband()
        
        # Calculate ATR for volatility
        market_data['atr'] = ta.volatility.AverageTrueRange(
            high=market_data['high'],
            low=market_data['low'],
            close=market_data['close']
        ).average_true_range()

        # Calculate RSI for mean reversion signals
        market_data['rsi'] = ta.momentum.RSIIndicator(
            close=market_data['close'],
            window=14
        ).rsi()

        latest = market_data.iloc[-1]
        prev = market_data.iloc[-2]
        symbol = market_data.index.name or 'UNKNOWN'
        
        # Detect squeeze (use .iloc[-1] to get scalar value from rolling mean)
        bb_width_avg = market_data['bb_width'].rolling(20).mean().iloc[-1]
        bb_squeeze = latest['bb_width'] < bb_width_avg * 0.8

        # Breakout signals
        if bb_squeeze:
            # Wait for breakout from squeeze
            vol_avg = market_data['volume'].rolling(20).mean().iloc[-1]
            if latest['close'] > latest['bb_upper'] and latest['volume'] > vol_avg * 1.5:
                signal = StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=0.85,
                    strategy_name=self.name,
                    timestamp=datetime.now(),
                    entry_price=latest['close'],
                    metadata={
                        'bb_squeeze': True,
                        'breakout': 'upper',
                        'bb_width': latest['bb_width']
                    }
                )
                signals.append(self.set_risk_levels(signal, latest['close']))
                
        # Mean reversion signals
        elif latest['close'] < latest['bb_lower'] and latest['rsi'] < 30:
            signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=0.75,
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest['close'],
                metadata={
                    'bb_position': 'below_lower',
                    'bb_lower': latest['bb_lower']
                }
            )
            signals.append(self.set_risk_levels(signal, latest['close']))
            
        elif latest['close'] > latest['bb_upper'] and latest['rsi'] > 70:
            signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.SELL if symbol not in current_positions else SignalType.CLOSE_LONG,
                strength=0.75,
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest['close'],
                metadata={
                    'bb_position': 'above_upper',
                    'bb_upper': latest['bb_upper']
                }
            )
            signals.append(self.set_risk_levels(signal, latest['close']))
        
        return signals

class MACDStrategy(BaseStrategy):
    """MACD momentum strategy with histogram analysis"""
    
    def get_required_indicators(self) -> List[str]:
        return ['macd', 'macd_signal', 'macd_histogram']
    
    def get_required_lookback(self) -> int:
        return 50
    
    def analyze(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        signals = []
        
        fast_period = self.parameters.get('fast_period', 12)
        slow_period = self.parameters.get('slow_period', 26)
        signal_period = self.parameters.get('signal_period', 9)
        
        # Calculate MACD
        macd = ta.trend.MACD(
            close=market_data['close'],
            window_slow=slow_period,
            window_fast=fast_period,
            window_sign=signal_period
        )
        
        market_data['macd'] = macd.macd()
        market_data['macd_signal'] = macd.macd_signal()
        market_data['macd_histogram'] = macd.macd_diff()
        
        latest = market_data.iloc[-1]
        prev = market_data.iloc[-2]
        symbol = market_data.index.name or 'UNKNOWN'
        
        # MACD crossover signals
        if prev['macd'] <= prev['macd_signal'] and latest['macd'] > latest['macd_signal']:
            # Bullish crossover
            signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=0.8,
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest['close'],
                metadata={
                    'macd': latest['macd'],
                    'signal': latest['macd_signal'],
                    'histogram': latest['macd_histogram']
                }
            )
            signals.append(self.set_risk_levels(signal, latest['close']))
            
        elif prev['macd'] >= prev['macd_signal'] and latest['macd'] < latest['macd_signal']:
            # Bearish crossover
            signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.SELL if symbol not in current_positions else SignalType.CLOSE_LONG,
                strength=0.8,
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest['close'],
                metadata={
                    'macd': latest['macd'],
                    'signal': latest['macd_signal'],
                    'histogram': latest['macd_histogram']
                }
            )
            signals.append(self.set_risk_levels(signal, latest['close']))
        
        return signals


class MomentumBreakoutStrategy(BaseStrategy):
    """Momentum breakout strategy for catching fast directional moves

    This strategy identifies:
    - Strong price momentum (rate of change)
    - Volume surge confirmation
    - Short-term breakouts from recent range
    - Relative strength acceleration
    """

    def get_required_indicators(self) -> List[str]:
        return ['roc', 'atr', 'volume_surge', 'range_high', 'range_low']

    def get_required_lookback(self) -> int:
        return 30

    def analyze(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        signals = []

        # Parameters
        roc_period = self.parameters.get('roc_period', 5)  # Short-term rate of change
        range_period = self.parameters.get('range_period', 10)  # Recent range lookback
        volume_surge_threshold = self.parameters.get('volume_surge', 1.5)  # 50% above average
        momentum_threshold = self.parameters.get('momentum_threshold', 1.5)  # 1.5% move

        # Calculate Rate of Change (momentum)
        market_data['roc'] = ((market_data['close'] - market_data['close'].shift(roc_period)) /
                              market_data['close'].shift(roc_period) * 100)

        # Calculate ATR for volatility-adjusted stops
        market_data['atr'] = ta.volatility.AverageTrueRange(
            high=market_data['high'],
            low=market_data['low'],
            close=market_data['close'],
            window=14
        ).average_true_range()

        # Volume surge detection
        market_data['volume_ma'] = market_data['volume'].rolling(20).mean()
        market_data['volume_surge'] = market_data['volume'] / market_data['volume_ma']

        # Recent range (for breakout detection)
        market_data['range_high'] = market_data['high'].rolling(range_period).max()
        market_data['range_low'] = market_data['low'].rolling(range_period).min()

        # Short-term momentum (last 3 bars)
        market_data['short_momentum'] = (market_data['close'] - market_data['close'].shift(3)) / market_data['close'].shift(3) * 100

        latest = market_data.iloc[-1]
        prev = market_data.iloc[-2]
        symbol = market_data.index.name or 'UNKNOWN'

        # Bullish momentum breakout conditions:
        # 1. Strong positive momentum (ROC > threshold)
        # 2. Price breaking above recent range high
        # 3. Volume surge confirmation
        # 4. Consecutive up bars (momentum continuation)

        bullish_momentum = latest['roc'] > momentum_threshold
        breakout_up = latest['close'] > prev['range_high']
        volume_confirmed = latest['volume_surge'] > volume_surge_threshold
        consecutive_up = (latest['close'] > latest['open'] and
                         prev['close'] > prev['open'])
        short_term_strong = latest['short_momentum'] > 1.0  # 1% move in 3 bars

        if bullish_momentum and volume_confirmed and (breakout_up or short_term_strong):
            # Calculate strength based on momentum magnitude and volume
            strength = min(0.95, 0.7 + (latest['roc'] / 10) + (latest['volume_surge'] - 1) * 0.1)

            signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=strength,
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest['close'],
                metadata={
                    'roc': round(latest['roc'], 2),
                    'volume_surge': round(latest['volume_surge'], 2),
                    'breakout': breakout_up,
                    'short_momentum': round(latest['short_momentum'], 2),
                    'atr': round(latest['atr'], 2)
                }
            )
            # Use ATR-based stops for momentum trades (tighter)
            signal.stop_loss = latest['close'] - (latest['atr'] * 1.5)
            signal.take_profit = latest['close'] + (latest['atr'] * 3)  # 2:1 reward/risk
            signals.append(signal)

        # Bearish momentum (for exits or shorts)
        bearish_momentum = latest['roc'] < -momentum_threshold
        breakout_down = latest['close'] < prev['range_low']

        if bearish_momentum and volume_confirmed and (breakout_down or latest['short_momentum'] < -1.0):
            strength = min(0.95, 0.7 + (abs(latest['roc']) / 10) + (latest['volume_surge'] - 1) * 0.1)

            signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.SELL if symbol not in current_positions else SignalType.CLOSE_LONG,
                strength=strength,
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest['close'],
                metadata={
                    'roc': round(latest['roc'], 2),
                    'volume_surge': round(latest['volume_surge'], 2),
                    'breakout': breakout_down
                }
            )
            signals.append(signal)

        return signals


class SimplePriceActionStrategy(BaseStrategy):
    """
    Simple price action strategy - generates more signals

    Triggers on:
    - 3 consecutive up/down candles
    - Price above/below short-term moving average
    - Any momentum in the direction

    This is more aggressive to generate trades for testing.
    """

    def get_required_indicators(self) -> List[str]:
        return ['sma_short', 'sma_medium']

    def get_required_lookback(self) -> int:
        return 20

    def analyze(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        signals = []

        # Short-term indicators
        market_data['sma_short'] = market_data['close'].rolling(5).mean()
        market_data['sma_medium'] = market_data['close'].rolling(15).mean()

        # Candle direction (1 = up, -1 = down)
        market_data['candle_dir'] = np.where(
            market_data['close'] > market_data['open'], 1, -1
        )

        latest = market_data.iloc[-1]
        prev1 = market_data.iloc[-2]
        prev2 = market_data.iloc[-3]
        symbol = market_data.index.name or 'UNKNOWN'

        # Bullish conditions:
        # 1. Price above short SMA
        # 2. Short SMA above medium SMA (uptrend)
        # 3. Last 2 candles are green OR price momentum positive
        price_above_sma = latest['close'] > latest['sma_short']
        short_above_medium = latest['sma_short'] > latest['sma_medium']
        recent_green = (latest['candle_dir'] == 1) and (prev1['candle_dir'] == 1)
        price_momentum = (latest['close'] - prev2['close']) / prev2['close'] * 100

        if price_above_sma and (short_above_medium or recent_green) and price_momentum > 0.2:
            signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=0.65,  # Lower strength for simple strategy
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest['close'],
                metadata={
                    'price_momentum': round(price_momentum, 2),
                    'above_sma': price_above_sma,
                    'trend_aligned': short_above_medium
                }
            )
            signals.append(self.set_risk_levels(signal, latest['close']))

        # Bearish conditions (for exits or shorts)
        price_below_sma = latest['close'] < latest['sma_short']
        short_below_medium = latest['sma_short'] < latest['sma_medium']
        recent_red = (latest['candle_dir'] == -1) and (prev1['candle_dir'] == -1)

        if price_below_sma and (short_below_medium or recent_red) and price_momentum < -0.2:
            signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.SELL if symbol not in current_positions else SignalType.CLOSE_LONG,
                strength=0.65,
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest['close'],
                metadata={
                    'price_momentum': round(price_momentum, 2),
                    'below_sma': price_below_sma
                }
            )
            signals.append(signal)

        return signals


class VolumeProfileStrategy(BaseStrategy):
    """Volume profile and market structure strategy"""

    def get_required_indicators(self) -> List[str]:
        return ['vwap', 'volume_ma', 'price_levels']

    def get_required_lookback(self) -> int:
        return 100

    def analyze(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        signals = []

        # Calculate VWAP
        market_data['vwap'] = (market_data['close'] * market_data['volume']).cumsum() / market_data['volume'].cumsum()
        market_data['volume_ma'] = market_data['volume'].rolling(20).mean()
        
        # Find support/resistance levels based on volume
        price_levels = self._find_volume_levels(market_data)
        
        latest = market_data.iloc[-1]
        symbol = market_data.index.name or 'UNKNOWN'
        
        # Check for breakouts with volume
        for level in price_levels:
            if abs(latest['close'] - level) / level < 0.005:  # Near level
                if latest['volume'] > latest['volume_ma'] * 2:  # High volume
                    if latest['close'] > level and market_data.iloc[-2]['close'] < level:
                        # Breakout above resistance
                        signal = StrategySignal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            strength=0.85,
                            strategy_name=self.name,
                            timestamp=datetime.now(),
                            entry_price=latest['close'],
                            metadata={
                                'breakout_level': level,
                                'volume_ratio': latest['volume'] / latest['volume_ma'],
                                'vwap': latest['vwap']
                            }
                        )
                        signals.append(self.set_risk_levels(signal, latest['close']))
                        break
        
        return signals
    
    def _find_volume_levels(self, data: pd.DataFrame, bins: int = 20) -> List[float]:
        """Find price levels with high volume"""
        # Create price bins
        price_range = data['high'].max() - data['low'].min()
        bin_size = price_range / bins
        
        volume_profile = {}
        
        for _, row in data.iterrows():
            # Distribute volume across price range in candle
            price_levels_in_candle = np.arange(row['low'], row['high'], bin_size)
            volume_per_level = row['volume'] / len(price_levels_in_candle)
            
            for price in price_levels_in_candle:
                bin_price = round(price / bin_size) * bin_size
                volume_profile[bin_price] = volume_profile.get(bin_price, 0) + volume_per_level
        
        # Find high volume nodes (top 20%)
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        top_levels = [level for level, _ in sorted_levels[:int(len(sorted_levels) * 0.2)]]
        
        return sorted(top_levels)

class StrategyManager:
    """Manages multiple trading strategies"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_configs: Dict[str, StrategyConfig] = {}
        self._load_default_strategies()
    
    def _load_default_strategies(self):
        """Load default strategy configurations"""
        default_configs = {
            'ma_cross': StrategyConfig(
                name='MA Crossover',
                enabled=True,
                weight=1.0,
                parameters={'fast_period': 9, 'slow_period': 21}
            ),
            'rsi_momentum': StrategyConfig(
                name='RSI Momentum',
                enabled=True,
                weight=0.8,
                parameters={'rsi_period': 14, 'oversold_level': 35, 'overbought_level': 65}
            ),
            'bollinger_bands': StrategyConfig(
                name='Bollinger Bands',
                enabled=True,
                weight=0.9,
                parameters={'bb_period': 20, 'bb_std': 1.8}  # Tighter than 2.0 but not as loose as 1.5
            ),
            'macd': StrategyConfig(
                name='MACD',
                enabled=True,
                weight=0.7,
                parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
            ),
            'momentum_breakout': StrategyConfig(
                name='Momentum Breakout',
                enabled=True,
                weight=1.0,
                parameters={
                    'roc_period': 5,
                    'range_period': 10,
                    'volume_surge': 1.3,         # 30% above avg (was 1.1 too loose, 1.5 too tight)
                    'momentum_threshold': 0.8    # 0.8% move (was 0.5 too loose, 1.0 too tight)
                }
            ),
            'simple_price_action': StrategyConfig(
                name='Simple Price Action',
                enabled=True,
                weight=0.5,  # Low weight - consensus filter will gate it in live trading
                parameters={}
            ),
            'volume_profile': StrategyConfig(
                name='Volume Profile',
                enabled=False,
                weight=0.6,
                parameters={}
            )
        }
        
        # Initialize strategies
        strategy_classes = {
            'ma_cross': MovingAverageCrossStrategy,
            'rsi_momentum': RSIMomentumStrategy,
            'bollinger_bands': BollingerBandStrategy,
            'macd': MACDStrategy,
            'momentum_breakout': MomentumBreakoutStrategy,
            'simple_price_action': SimplePriceActionStrategy,
            'volume_profile': VolumeProfileStrategy
        }
        
        for key, config in default_configs.items():
            self.strategy_configs[key] = config
            if config.enabled:
                self.strategies[key] = strategy_classes[key](config)
    
    def add_strategy(self, key: str, strategy: BaseStrategy, config: StrategyConfig):
        """Add a new strategy"""
        self.strategies[key] = strategy
        self.strategy_configs[key] = config
    
    def remove_strategy(self, key: str):
        """Remove a strategy"""
        if key in self.strategies:
            del self.strategies[key]
            del self.strategy_configs[key]
    
    def enable_strategy(self, key: str):
        """Enable a strategy"""
        if key in self.strategy_configs:
            self.strategy_configs[key].enabled = True
            # Re-initialize if needed
            if key not in self.strategies:
                # Would need strategy class mapping here
                pass
    
    def disable_strategy(self, key: str):
        """Disable a strategy"""
        if key in self.strategy_configs:
            self.strategy_configs[key].enabled = False
            if key in self.strategies:
                del self.strategies[key]
    
    def update_strategy_config(self, key: str, config: StrategyConfig):
        """Update strategy configuration"""
        if key in self.strategies:
            self.strategy_configs[key] = config
            self.strategies[key].config = config
            self.strategies[key].parameters = config.parameters
            self.strategies[key].risk_parameters = config.risk_parameters
    
    def analyze_all(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        """Run all enabled strategies and collect signals"""
        all_signals = []
        
        for key, strategy in self.strategies.items():
            if self.strategy_configs[key].enabled:
                try:
                    signals = strategy.analyze(market_data, current_positions)
                    # Apply strategy weight to signal strength
                    for signal in signals:
                        signal.strength *= self.strategy_configs[key].weight
                    all_signals.extend(signals)
                except Exception as e:
                    logger.error(f"Error in strategy {key}: {e}")
        
        return all_signals
    
    def get_consensus_signal(self, signals: List[StrategySignal]) -> Optional[StrategySignal]:
        """Get consensus signal from multiple strategy signals"""
        if not signals:
            return None

        # Group signals by type
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type in [SignalType.SELL, SignalType.CLOSE_LONG]]

        # Calculate weighted consensus (requires 2+ strategies agreeing)
        # No single-signal bypasses - always require consensus
        if buy_signals and len(buy_signals) >= 2:  # Require at least 2 strategies to agree
            avg_strength = sum(s.strength for s in buy_signals) / len(buy_signals)
            if avg_strength > 0.60:  # Consensus threshold
                # Create consensus signal
                consensus = StrategySignal(
                    symbol=buy_signals[0].symbol,
                    signal_type=SignalType.BUY,
                    strength=avg_strength,
                    strategy_name='Consensus',
                    timestamp=datetime.now(),
                    entry_price=buy_signals[0].entry_price,
                    metadata={
                        'strategies': [s.strategy_name for s in buy_signals],
                        'individual_strengths': {s.strategy_name: s.strength for s in buy_signals}
                    }
                )
                # Use most conservative risk levels
                consensus.stop_loss = min(s.stop_loss for s in buy_signals if s.stop_loss)
                consensus.take_profit = min(s.take_profit for s in buy_signals if s.take_profit)
                return consensus
        
        # Sell consensus (requires 2+ strategies agreeing)
        if sell_signals and len(sell_signals) >= 2:
            avg_strength = sum(s.strength for s in sell_signals) / len(sell_signals)
            if avg_strength > 0.60:  # Consensus threshold
                consensus = StrategySignal(
                    symbol=sell_signals[0].symbol,
                    signal_type=sell_signals[0].signal_type,
                    strength=avg_strength,
                    strategy_name='Consensus',
                    timestamp=datetime.now(),
                    entry_price=sell_signals[0].entry_price,
                    metadata={
                        'strategies': [s.strategy_name for s in sell_signals],
                        'individual_strengths': {s.strategy_name: s.strength for s in sell_signals}
                    }
                )
                return consensus

        return None
    
    def save_configs(self, filepath: str):
        """Save strategy configurations to file"""
        configs = {
            key: {
                'name': config.name,
                'enabled': config.enabled,
                'weight': config.weight,
                'parameters': config.parameters,
                'risk_parameters': config.risk_parameters
            }
            for key, config in self.strategy_configs.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(configs, f, indent=2)
    
    def load_configs(self, filepath: str):
        """Load strategy configurations from file"""
        with open(filepath, 'r') as f:
            configs = json.load(f)
        
        for key, config_data in configs.items():
            config = StrategyConfig(
                name=config_data['name'],
                enabled=config_data['enabled'],
                weight=config_data['weight'],
                parameters=config_data['parameters'],
                risk_parameters=config_data['risk_parameters']
            )
            self.strategy_configs[key] = config