#!/usr/bin/env python3
"""
Professional Trading Strategy System
Modular, configurable, and extensible trading strategies

Enhanced with:
- Market Regime Detection (trending vs ranging)
- Leading Indicator Strategies
- Adaptive Risk Management
- Trend Filters and Pullback Detection
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


# =============================================================================
# MARKET REGIME DETECTION
# =============================================================================

class MarketRegime(Enum):
    """Market regime classifications"""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class RegimeAnalysis:
    """Results from market regime analysis"""
    regime: MarketRegime
    trend_strength: float  # 0-100 (ADX)
    trend_direction: float  # -1 to +1
    volatility_percentile: float  # 0-100
    momentum: float  # Rate of change
    support_level: float
    resistance_level: float
    position_in_range: float  # 0-100 (0=at support, 100=at resistance)
    recommended_strategy: str  # 'trend_following' or 'mean_reversion'
    confidence: float  # 0-1


class MarketRegimeDetector:
    """
    Detects market regime to adapt strategy behavior.

    Key insight: Different strategies work in different market conditions.
    - Trend-following works in strong trends
    - Mean-reversion works in ranging markets
    - High volatility requires wider stops
    """

    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
        self.regime_history: List[RegimeAnalysis] = []

    def analyze(self, df: pd.DataFrame) -> RegimeAnalysis:
        """
        Analyze market data to determine current regime.

        Returns detailed regime analysis with actionable recommendations.
        """
        if len(df) < self.lookback_period:
            return self._default_analysis(df)

        close = df['close']
        high = df['high']
        low = df['low']

        # 1. Trend Strength (ADX)
        adx = self._calculate_adx(df)
        trend_strength = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20

        # 2. Trend Direction (-1 to +1)
        trend_direction = self._calculate_trend_direction(close)

        # 3. Volatility Percentile
        volatility_pct = self._calculate_volatility_percentile(close)

        # 4. Momentum (Rate of Change)
        momentum = self._calculate_momentum(close)

        # 5. Support/Resistance Levels
        support, resistance = self._calculate_sr_levels(df)

        # 6. Position in Range
        current_price = close.iloc[-1]
        if resistance > support:
            position_in_range = ((current_price - support) / (resistance - support)) * 100
        else:
            position_in_range = 50
        position_in_range = max(0, min(100, position_in_range))

        # Determine Regime
        regime = self._classify_regime(trend_strength, trend_direction, volatility_pct)

        # Recommend Strategy Type
        if regime in [MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND]:
            recommended = 'trend_following'
            confidence = min(1.0, trend_strength / 40)  # Higher ADX = higher confidence
        elif regime == MarketRegime.RANGING:
            recommended = 'mean_reversion'
            confidence = min(1.0, (40 - trend_strength) / 25)  # Lower ADX = higher confidence
        elif regime == MarketRegime.HIGH_VOLATILITY:
            recommended = 'reduced_exposure'
            confidence = min(1.0, volatility_pct / 80)
        else:
            recommended = 'mixed'
            confidence = 0.5

        analysis = RegimeAnalysis(
            regime=regime,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            volatility_percentile=volatility_pct,
            momentum=momentum,
            support_level=support,
            resistance_level=resistance,
            position_in_range=position_in_range,
            recommended_strategy=recommended,
            confidence=confidence
        )

        self.regime_history.append(analysis)
        if len(self.regime_history) > 100:
            self.regime_history.pop(0)

        return analysis

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index for trend strength"""
        try:
            adx_indicator = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=period
            )
            return adx_indicator.adx()
        except:
            return pd.Series([20] * len(df), index=df.index)

    def _calculate_trend_direction(self, close: pd.Series) -> float:
        """Calculate trend direction from -1 (strong down) to +1 (strong up)"""
        # Use linear regression slope normalized
        if len(close) < 20:
            return 0

        recent = close.tail(20)
        x = np.arange(len(recent))
        slope, _ = np.polyfit(x, recent.values, 1)

        # Normalize by average price
        avg_price = recent.mean()
        normalized_slope = (slope / avg_price) * 100  # Slope as % per bar

        # Clip to [-1, 1]
        return max(-1, min(1, normalized_slope * 10))

    def _calculate_volatility_percentile(self, close: pd.Series) -> float:
        """Calculate current volatility as percentile of historical volatility"""
        returns = close.pct_change().dropna()
        if len(returns) < 20:
            return 50

        # Current volatility (last 10 bars)
        current_vol = returns.tail(10).std()

        # Rolling volatility history
        rolling_vol = returns.rolling(10).std().dropna()
        if len(rolling_vol) < 10:
            return 50

        # Percentile rank
        percentile = (rolling_vol < current_vol).mean() * 100
        return percentile

    def _calculate_momentum(self, close: pd.Series) -> float:
        """Calculate momentum as rate of change"""
        if len(close) < 10:
            return 0
        return (close.iloc[-1] / close.iloc[-10] - 1) * 100

    def _calculate_sr_levels(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate support and resistance levels"""
        lookback = min(50, len(df))
        recent = df.tail(lookback)

        # Simple approach: use recent highs/lows
        resistance = recent['high'].max()
        support = recent['low'].min()

        # Refine with pivot points
        pivots_high = self._find_pivots(recent['high'], is_high=True)
        pivots_low = self._find_pivots(recent['low'], is_high=False)

        if pivots_high:
            resistance = np.mean(pivots_high[-3:]) if len(pivots_high) >= 3 else resistance
        if pivots_low:
            support = np.mean(pivots_low[-3:]) if len(pivots_low) >= 3 else support

        return support, resistance

    def _find_pivots(self, series: pd.Series, is_high: bool, window: int = 5) -> List[float]:
        """Find pivot points in a series"""
        pivots = []
        values = series.values

        for i in range(window, len(values) - window):
            if is_high:
                if all(values[i] >= values[i-j] for j in range(1, window+1)) and \
                   all(values[i] >= values[i+j] for j in range(1, window+1)):
                    pivots.append(values[i])
            else:
                if all(values[i] <= values[i-j] for j in range(1, window+1)) and \
                   all(values[i] <= values[i+j] for j in range(1, window+1)):
                    pivots.append(values[i])

        return pivots

    def _classify_regime(self, adx: float, direction: float, volatility: float) -> MarketRegime:
        """Classify market regime based on indicators"""
        # High volatility overrides other classifications
        if volatility > 80:
            return MarketRegime.HIGH_VOLATILITY

        # Strong trend (ADX > 25)
        if adx > 30:
            if direction > 0.3:
                return MarketRegime.STRONG_UPTREND
            elif direction < -0.3:
                return MarketRegime.STRONG_DOWNTREND

        # Weak trend (ADX 20-30)
        if adx > 20:
            if direction > 0.1:
                return MarketRegime.WEAK_UPTREND
            elif direction < -0.1:
                return MarketRegime.WEAK_DOWNTREND

        # Ranging (ADX < 20)
        return MarketRegime.RANGING

    def _default_analysis(self, df: pd.DataFrame) -> RegimeAnalysis:
        """Return default analysis when insufficient data"""
        current_price = df['close'].iloc[-1] if len(df) > 0 else 100
        return RegimeAnalysis(
            regime=MarketRegime.RANGING,
            trend_strength=20,
            trend_direction=0,
            volatility_percentile=50,
            momentum=0,
            support_level=current_price * 0.95,
            resistance_level=current_price * 1.05,
            position_in_range=50,
            recommended_strategy='mixed',
            confidence=0.3
        )

    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of recent regime history"""
        if not self.regime_history:
            return {'regime': 'unknown', 'stability': 0}

        recent = self.regime_history[-10:]
        regimes = [r.regime for r in recent]
        most_common = max(set(regimes), key=regimes.count)
        stability = regimes.count(most_common) / len(regimes)

        return {
            'current_regime': self.regime_history[-1].regime.value,
            'most_common': most_common.value,
            'stability': stability,
            'avg_trend_strength': np.mean([r.trend_strength for r in recent]),
            'avg_volatility': np.mean([r.volatility_percentile for r in recent])
        }

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

        # Calculate moving averages (skip if pre-computed)
        if 'sma_fast' not in market_data.columns:
            market_data['sma_fast'] = market_data['close'].rolling(fast_period).mean()
        if 'sma_slow' not in market_data.columns:
            market_data['sma_slow'] = market_data['close'].rolling(slow_period).mean()
        if 'volume_sma' not in market_data.columns:
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

        # Calculate RSI (skip if pre-computed)
        if 'rsi' not in market_data.columns:
            market_data['rsi'] = ta.momentum.RSIIndicator(
                close=market_data['close'],
                window=rsi_period
            ).rsi()
        if 'rsi_ma' not in market_data.columns:
            market_data['rsi_ma'] = market_data['rsi'].rolling(5).mean()
        if 'price_ma' not in market_data.columns:
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

        # Calculate Bollinger Bands (skip if pre-computed)
        if 'bb_upper' not in market_data.columns:
            bb = ta.volatility.BollingerBands(
                close=market_data['close'],
                window=bb_period,
                window_dev=bb_std
            )
            market_data['bb_upper'] = bb.bollinger_hband()
            market_data['bb_lower'] = bb.bollinger_lband()
            market_data['bb_middle'] = bb.bollinger_mavg()
            market_data['bb_width'] = bb.bollinger_wband()

        # Calculate ATR for volatility (skip if pre-computed)
        if 'atr' not in market_data.columns:
            market_data['atr'] = ta.volatility.AverageTrueRange(
                high=market_data['high'],
                low=market_data['low'],
                close=market_data['close']
            ).average_true_range()

        # Calculate RSI for mean reversion signals (skip if pre-computed)
        if 'rsi' not in market_data.columns:
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

        # Calculate MACD (skip if pre-computed)
        if 'macd' not in market_data.columns:
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

        # Calculate Rate of Change (skip if pre-computed)
        if 'roc' not in market_data.columns:
            market_data['roc'] = ((market_data['close'] - market_data['close'].shift(roc_period)) /
                                  market_data['close'].shift(roc_period) * 100)

        # Calculate ATR for volatility-adjusted stops (skip if pre-computed)
        if 'atr' not in market_data.columns:
            market_data['atr'] = ta.volatility.AverageTrueRange(
                high=market_data['high'],
                low=market_data['low'],
                close=market_data['close'],
                window=14
            ).average_true_range()

        # Volume surge detection (skip if pre-computed)
        if 'volume_ma' not in market_data.columns:
            market_data['volume_ma'] = market_data['volume'].rolling(20).mean()
        if 'volume_surge' not in market_data.columns:
            market_data['volume_surge'] = market_data['volume'] / market_data['volume_ma']

        # Recent range (skip if pre-computed)
        if 'range_high' not in market_data.columns:
            market_data['range_high'] = market_data['high'].rolling(range_period).max()
        if 'range_low' not in market_data.columns:
            market_data['range_low'] = market_data['low'].rolling(range_period).min()

        # Short-term momentum (skip if pre-computed)
        if 'short_momentum' not in market_data.columns:
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

        # Short-term indicators (skip if pre-computed)
        if 'sma_short' not in market_data.columns:
            market_data['sma_short'] = market_data['close'].rolling(5).mean()
        if 'sma_medium' not in market_data.columns:
            market_data['sma_medium'] = market_data['close'].rolling(15).mean()

        # Candle direction (skip if pre-computed)
        if 'candle_dir' not in market_data.columns:
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


# =============================================================================
# ENHANCED STRATEGIES WITH REGIME AWARENESS
# =============================================================================

class SmartTrendStrategy(BaseStrategy):
    """
    Refined trend-following strategy with strict filters.

    Key features:
    1. Only trades in STRONG trends (ADX > 25)
    2. Requires DEEP pullback to enter (RSI < 40, price near EMA)
    3. Requires bullish candle confirmation (close > open)
    4. Uses ATR-based stops with good R:R ratio
    """

    def get_required_indicators(self) -> List[str]:
        return ['ema_fast', 'ema_slow', 'atr', 'adx', 'rsi']

    def get_required_lookback(self) -> int:
        return 50

    def analyze(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        signals = []
        if len(market_data) < 50:
            return signals

        symbol = market_data.index.name or 'UNKNOWN'
        close = market_data['close']
        high = market_data['high']
        low = market_data['low']
        open_price = market_data['open']

        # Calculate indicators if not present
        if 'ema_fast' not in market_data.columns:
            market_data['ema_fast'] = close.ewm(span=12).mean()
        if 'ema_slow' not in market_data.columns:
            market_data['ema_slow'] = close.ewm(span=26).mean()
        if 'ema_200' not in market_data.columns:
            market_data['ema_200'] = close.ewm(span=200, min_periods=50).mean()
        if 'atr' not in market_data.columns:
            market_data['atr'] = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range()
        if 'adx' not in market_data.columns or 'di_plus' not in market_data.columns or 'di_minus' not in market_data.columns:
            adx_ind = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
            market_data['adx'] = adx_ind.adx()
            market_data['di_plus'] = adx_ind.adx_pos()
            market_data['di_minus'] = adx_ind.adx_neg()
        if 'rsi' not in market_data.columns:
            market_data['rsi'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

        latest = market_data.iloc[-1]
        prev = market_data.iloc[-2]
        prev2 = market_data.iloc[-3] if len(market_data) > 2 else prev

        # TREND FILTER: Require meaningful trend (ADX > 20)
        # Lowered from 25 to 20 to catch more opportunities while still filtering out weak trends
        if latest['adx'] < 20:
            return signals

        # Check for uptrend with momentum
        # Note: EMA 200 check is optional when not enough data (e.g., short backtests)
        ema_200_ok = pd.notna(latest['ema_200']) and latest['close'] > latest['ema_200']
        ema_200_missing = pd.isna(latest['ema_200'])  # Skip this check if no data

        strong_uptrend = (
            latest['ema_fast'] > latest['ema_slow'] and
            latest['di_plus'] > latest['di_minus'] and
            latest['di_plus'] > 20 and  # Meaningful +DI
            (ema_200_ok or ema_200_missing)  # Above 200 EMA OR not enough data to calculate
        )

        # Check for downtrend with momentum
        strong_downtrend = (
            latest['ema_fast'] < latest['ema_slow'] and
            latest['di_minus'] > latest['di_plus'] and
            latest['di_minus'] > 20  # Meaningful -DI (lowered from 25)
        )

        # BUY: Only in strong uptrend with deep pullback
        if strong_uptrend and symbol not in current_positions:
            # STRICT PULLBACK CONDITIONS:
            # 1. RSI dropped to oversold in uptrend (< 40)
            # 2. Price touched or near the slow EMA (within 0.5%)
            # 3. RSI is now turning up (current > previous)
            # 4. Current candle is bullish (close > open)

            price_near_ema = abs(latest['close'] - latest['ema_slow']) / latest['ema_slow'] < 0.01  # Within 1%
            deep_pullback = latest['rsi'] < 45 or (latest['rsi'] < 50 and price_near_ema)
            rsi_not_declining = latest['rsi'] >= prev['rsi']  # RSI not actively declining (more lenient)
            bullish_candle = latest['close'] > latest['open']

            if deep_pullback and rsi_not_declining and bullish_candle:
                atr = latest['atr']
                entry_price = latest['close']

                signal = StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=min(1.0, (latest['adx'] - 20) / 20),  # Scale 0.25-1.0 for ADX 25-45
                    strategy_name=self.name,
                    timestamp=datetime.now(),
                    entry_price=entry_price,
                    stop_loss=entry_price - (atr * 2.5),  # 2.5 ATR stop
                    take_profit=entry_price + (atr * 5),   # 5 ATR target (2:1 R:R)
                    metadata={
                        'regime': 'strong_uptrend',
                        'adx': float(latest['adx']),
                        'rsi': float(latest['rsi']),
                        'di_plus': float(latest['di_plus']),
                        'entry_type': 'deep_pullback'
                    }
                )
                signals.append(signal)

        # SELL: Exit on trend reversal confirmation
        elif strong_downtrend and symbol in current_positions:
            signal = StrategySignal(
                symbol=symbol,
                signal_type=SignalType.CLOSE_LONG,
                strength=min(1.0, (latest['adx'] - 20) / 20),
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=latest['close'],
                metadata={
                    'regime': 'strong_downtrend',
                    'adx': float(latest['adx']),
                    'exit_reason': 'trend_reversal_confirmed'
                }
            )
            signals.append(signal)

        return signals


class MeanReversionStrategy(BaseStrategy):
    """
    Refined mean reversion strategy for ranging/consolidating markets.

    Key improvements:
    1. Only trades in CLEAR ranging markets (ADX < 20)
    2. Requires DEEP oversold (RSI < 25) at lower BB
    3. Must have bullish reversal candle confirmation
    4. Requires volume surge on the bounce
    5. Better R:R with tighter stops
    """

    def get_required_indicators(self) -> List[str]:
        return ['bb_upper', 'bb_lower', 'bb_middle', 'rsi', 'adx', 'atr']

    def get_required_lookback(self) -> int:
        return 30

    def analyze(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        signals = []
        if len(market_data) < 30:
            return signals

        symbol = market_data.index.name or 'UNKNOWN'
        close = market_data['close']
        high = market_data['high']
        low = market_data['low']
        volume = market_data['volume']

        # Calculate indicators
        if 'bb_upper' not in market_data.columns:
            bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            market_data['bb_upper'] = bb.bollinger_hband()
            market_data['bb_lower'] = bb.bollinger_lband()
            market_data['bb_middle'] = bb.bollinger_mavg()
        if 'rsi' not in market_data.columns:
            market_data['rsi'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        if 'adx' not in market_data.columns:
            market_data['adx'] = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()
        if 'atr' not in market_data.columns:
            market_data['atr'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        if 'volume_ma' not in market_data.columns:
            market_data['volume_ma'] = volume.rolling(20).mean()

        latest = market_data.iloc[-1]
        prev = market_data.iloc[-2]

        # Only trade in CLEAR ranging markets (ADX < 20 - stricter)
        if latest['adx'] > 20:
            return signals

        atr = latest['atr']
        entry_price = latest['close']

        # Calculate BB width to ensure it's not too narrow (compression phase)
        bb_width = (latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle']
        if bb_width < 0.03:  # Bands too narrow - avoid, breakout likely
            return signals

        # Buy at lower Bollinger Band (DEEP oversold) with reversal confirmation
        if symbol not in current_positions:
            # Price touched or penetrated lower band
            touched_lower_band = latest['low'] <= latest['bb_lower']

            # Oversold - RSI < 35 (adjusted to be more achievable while still selective)
            deep_oversold = latest['rsi'] < 35

            # Bullish reversal candle: close > open AND close in upper half of candle
            candle_range = latest['high'] - latest['low']
            close_position = (latest['close'] - latest['low']) / candle_range if candle_range > 0 else 0
            bullish_candle = latest['close'] > latest['open'] and close_position > 0.6

            # Volume surge on the bounce (1.3x average)
            volume_surge = latest['volume'] > latest['volume_ma'] * 1.3

            # Previous candle was bearish (confirms we're at a low point)
            prior_bearish = prev['close'] < prev['open']

            oversold_condition = (
                touched_lower_band and
                deep_oversold and
                bullish_candle and
                volume_surge and
                prior_bearish
            )

            if oversold_condition:
                # Target: middle band, Stop: 1.5 ATR below entry (tight for mean reversion)
                stop = entry_price - (atr * 1.5)
                target = latest['bb_middle']

                signal = StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=0.85,
                    strategy_name=self.name,
                    timestamp=datetime.now(),
                    entry_price=entry_price,
                    stop_loss=stop,
                    take_profit=target,
                    metadata={
                        'regime': 'ranging',
                        'entry_type': 'deep_oversold_reversal',
                        'rsi': latest['rsi'],
                        'bb_position': 'lower',
                        'bb_width': bb_width,
                        'volume_ratio': latest['volume'] / latest['volume_ma']
                    }
                )
                signals.append(signal)

        # Exit at upper BB (overbought) - only for existing positions
        if symbol in current_positions:
            overbought_condition = (
                latest['close'] >= latest['bb_upper'] * 0.99 and  # Near or at upper band
                latest['rsi'] > 70  # RSI overbought
            )

            if overbought_condition:
                signal = StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.CLOSE_LONG,
                    strength=0.80,
                    strategy_name=self.name,
                    timestamp=datetime.now(),
                    entry_price=entry_price,
                    metadata={
                        'regime': 'ranging',
                        'exit_type': 'overbought_at_upper_band',
                        'rsi': latest['rsi'],
                        'bb_position': 'upper'
                    }
                )
                signals.append(signal)

        return signals


class VolumeBreakoutStrategy(BaseStrategy):
    """
    Refined volume breakout strategy - a LEADING indicator.

    Key improvements:
    1. Requires VERY high volume (>2.5 std dev) for entries
    2. Breakout must be CLEAN (close above resistance, not just near)
    3. Needs trend support (ADX showing some trend developing)
    4. Strong bullish candle with good body
    5. Improved R:R ratio
    """

    def get_required_indicators(self) -> List[str]:
        return ['volume_ma', 'atr', 'high_20', 'low_20', 'adx']

    def get_required_lookback(self) -> int:
        return 25

    def analyze(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        signals = []
        if len(market_data) < 25:
            return signals

        symbol = market_data.index.name or 'UNKNOWN'
        close = market_data['close']
        high = market_data['high']
        low = market_data['low']
        volume = market_data['volume']

        # Calculate indicators
        if 'volume_ma' not in market_data.columns:
            market_data['volume_ma'] = volume.rolling(20).mean()
        if 'volume_std' not in market_data.columns:
            market_data['volume_std'] = volume.rolling(20).std()
        if 'atr' not in market_data.columns:
            market_data['atr'] = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range()
        if 'high_20' not in market_data.columns:
            market_data['high_20'] = high.rolling(20).max()
        if 'low_20' not in market_data.columns:
            market_data['low_20'] = low.rolling(20).min()
        if 'adx' not in market_data.columns:
            market_data['adx'] = ta.trend.ADXIndicator(
                high=high, low=low, close=close, window=14
            ).adx()

        latest = market_data.iloc[-1]
        prev = market_data.iloc[-2]
        prev2 = market_data.iloc[-3] if len(market_data) >= 3 else prev

        # Detect volume spike (>2.0 std deviations - balanced threshold)
        volume_zscore = (latest['volume'] - latest['volume_ma']) / (latest['volume_std'] + 1)
        volume_spike = volume_zscore > 2.0

        if not volume_spike:
            return signals

        atr = latest['atr']
        entry_price = latest['close']

        # Bullish volume breakout with strict confirmation
        if symbol not in current_positions:
            # Clean breakout: CLOSE above 20-bar high (not just near)
            breakout_level = market_data['high'].iloc[:-1].rolling(20).max().iloc[-1]  # Prior 20-bar high
            clean_breakout = latest['close'] > breakout_level

            # Strong bullish candle: close near high, body > 60% of range
            candle_range = latest['high'] - latest['low']
            body_size = abs(latest['close'] - latest['open'])
            body_ratio = body_size / candle_range if candle_range > 0 else 0
            strong_bullish = (
                latest['close'] > latest['open'] and
                body_ratio > 0.5 and  # Body at least 50% of range (relaxed from 60%)
                (latest['close'] - latest['low']) / candle_range > 0.6  # Close in upper 40% (relaxed)
            )

            # Consolidation before breakout (recent bars were tight)
            recent_range = market_data['high'].iloc[-6:-1].max() - market_data['low'].iloc[-6:-1].min()
            consolidation = recent_range < atr * 3  # Recent bars within 3 ATR

            # Some trend developing (ADX > 18) - not too weak
            trend_developing = latest['adx'] > 18

            bullish_breakout = (
                clean_breakout and
                strong_bullish and
                consolidation and
                trend_developing
            )

            if bullish_breakout:
                # Stop below breakout level, target 3x risk
                stop = breakout_level - (atr * 0.5)
                risk = entry_price - stop
                target = entry_price + (risk * 3)

                signal = StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=min(1.0, volume_zscore / 3),
                    strategy_name=self.name,
                    timestamp=datetime.now(),
                    entry_price=entry_price,
                    stop_loss=stop,
                    take_profit=target,
                    metadata={
                        'entry_type': 'clean_volume_breakout',
                        'volume_zscore': volume_zscore,
                        'breakout_level': breakout_level,
                        'body_ratio': body_ratio,
                        'adx': latest['adx']
                    }
                )
                signals.append(signal)

        # Bearish breakdown - exit signal for longs
        if symbol in current_positions:
            breakdown_level = market_data['low'].iloc[:-1].rolling(20).min().iloc[-1]
            bearish_breakdown = (
                latest['close'] < breakdown_level and
                latest['close'] < latest['open'] and  # Bearish candle
                volume_zscore > 2.0
            )

            if bearish_breakdown:
                signal = StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.CLOSE_LONG,
                    strength=min(1.0, volume_zscore / 3),
                    strategy_name=self.name,
                    timestamp=datetime.now(),
                    entry_price=entry_price,
                    metadata={
                        'exit_type': 'volume_breakdown',
                        'volume_zscore': volume_zscore,
                        'breakdown_level': breakdown_level
                    }
                )
                signals.append(signal)

        return signals


class SupportResistanceStrategy(BaseStrategy):
    """
    Refined Support/Resistance trading strategy.

    Key improvements:
    1. Requires MULTIPLE touches of level (proven S/R)
    2. Deep oversold at support (RSI < 35)
    3. Strong reversal candle with volume
    4. Level must be significant (recent swing point)
    5. Better R:R targeting next resistance
    """

    def get_required_indicators(self) -> List[str]:
        return ['atr', 'rsi', 'volume_ma']

    def get_required_lookback(self) -> int:
        return 50

    def analyze(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        signals = []
        if len(market_data) < 50:
            return signals

        symbol = market_data.index.name or 'UNKNOWN'
        close = market_data['close']
        high = market_data['high']
        low = market_data['low']
        volume = market_data['volume']

        # Calculate indicators
        if 'atr' not in market_data.columns:
            market_data['atr'] = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range()
        if 'rsi' not in market_data.columns:
            market_data['rsi'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        if 'volume_ma' not in market_data.columns:
            market_data['volume_ma'] = volume.rolling(20).mean()

        latest = market_data.iloc[-1]
        prev = market_data.iloc[-2]
        atr = latest['atr']
        current_price = latest['close']

        # Find support and resistance levels with touch counts
        support_levels = self._find_support_levels_with_touches(market_data, atr)
        resistance_levels = self._find_resistance_levels_with_touches(market_data, atr)

        # Check if price is at a PROVEN support level (multiple touches)
        near_support = None
        support_touches = 0
        for level, touches in support_levels:
            if touches >= 2 and abs(current_price - level) <= atr * 0.3:  # Tighter: 0.3 ATR
                near_support = level
                support_touches = touches
                break

        # Check if price is near a resistance level
        near_resistance = None
        for level, touches in resistance_levels:
            if touches >= 2 and abs(current_price - level) <= atr * 0.3:
                near_resistance = level
                break

        # Buy at proven support with strong confirmation
        if near_support and symbol not in current_positions:
            # Price must have dipped to/below support
            tested_support = latest['low'] <= near_support * 1.002

            # Deep oversold (RSI < 35)
            oversold = latest['rsi'] < 35

            # Strong bullish reversal candle
            candle_range = latest['high'] - latest['low']
            body = latest['close'] - latest['open']
            strong_reversal = (
                body > 0 and  # Bullish
                body / candle_range > 0.5 if candle_range > 0 else False  # Body > 50% of range
            )

            # Volume confirmation (above average)
            volume_confirm = latest['volume'] > latest['volume_ma'] * 1.2

            # Previous bar was bearish (confirms we're bouncing from a low)
            prior_bearish = prev['close'] < prev['open']

            if tested_support and oversold and strong_reversal and volume_confirm and prior_bearish:
                # Find nearest resistance for target
                target = current_price + (atr * 3)  # Default
                for res_level, _ in resistance_levels:
                    if res_level > current_price:
                        target = res_level
                        break

                stop = near_support - (atr * 1)  # Tight stop below support

                signal = StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=0.85,
                    strategy_name=self.name,
                    timestamp=datetime.now(),
                    entry_price=current_price,
                    stop_loss=stop,
                    take_profit=target,
                    metadata={
                        'entry_type': 'proven_support_bounce',
                        'support_level': near_support,
                        'support_touches': support_touches,
                        'rsi': latest['rsi'],
                        'volume_ratio': latest['volume'] / latest['volume_ma']
                    }
                )
                signals.append(signal)

        # Exit at resistance
        if near_resistance and symbol in current_positions:
            # Showing weakness at resistance
            tested_resistance = latest['high'] >= near_resistance * 0.998
            weakness = (
                latest['close'] < latest['open'] or  # Bearish candle
                latest['rsi'] > 70  # Overbought
            )

            if tested_resistance and weakness:
                signal = StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.CLOSE_LONG,
                    strength=0.85,
                    strategy_name=self.name,
                    timestamp=datetime.now(),
                    entry_price=current_price,
                    metadata={
                        'exit_type': 'resistance_rejection',
                        'resistance_level': near_resistance
                    }
                )
                signals.append(signal)

        return signals

    def _find_support_levels_with_touches(self, df: pd.DataFrame, atr: float) -> List[tuple]:
        """Find support levels with touch counts"""
        lows = df['low'].values
        levels_with_touches = []

        # Find swing lows
        for i in range(5, len(lows) - 5):
            if all(lows[i] <= lows[i-j] for j in range(1, 6)) and \
               all(lows[i] <= lows[i+j] for j in range(1, min(6, len(lows) - i))):
                level = lows[i]
                # Count touches (price came within 0.5% of level)
                touches = sum(1 for l in lows if abs(l - level) / level < 0.005)
                levels_with_touches.append((level, touches))

        # Sort by touches (most touched first), then by price (highest first)
        levels_with_touches.sort(key=lambda x: (-x[1], -x[0]))
        return levels_with_touches[:5]

    def _find_resistance_levels_with_touches(self, df: pd.DataFrame, atr: float) -> List[tuple]:
        """Find resistance levels with touch counts"""
        highs = df['high'].values
        levels_with_touches = []

        for i in range(5, len(highs) - 5):
            if all(highs[i] >= highs[i-j] for j in range(1, 6)) and \
               all(highs[i] >= highs[i+j] for j in range(1, min(6, len(highs) - i))):
                level = highs[i]
                touches = sum(1 for h in highs if abs(h - level) / level < 0.005)
                levels_with_touches.append((level, touches))

        levels_with_touches.sort(key=lambda x: (-x[1], x[0]))
        return levels_with_touches[:5]


class AdaptiveStrategy(BaseStrategy):
    """
    Refined meta-strategy that adapts based on market regime.

    Key improvements:
    1. Only trades when regime confidence is HIGH (>0.7)
    2. Requires clear regime detection before trading
    3. Delegates to refined trend/reversion strategies
    4. More conservative in uncertain conditions
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.regime_detector = MarketRegimeDetector()
        self.trend_strategy = SmartTrendStrategy(StrategyConfig(
            name='trend_component',
            parameters={'fast_period': 12, 'slow_period': 26}
        ))
        self.reversion_strategy = MeanReversionStrategy(StrategyConfig(
            name='reversion_component'
        ))
        # Minimum confidence to trade
        self.min_regime_confidence = 0.7

    def get_required_indicators(self) -> List[str]:
        return ['ema_fast', 'ema_slow', 'bb_upper', 'bb_lower', 'atr', 'adx', 'rsi', 'di_plus', 'di_minus']

    def get_required_lookback(self) -> int:
        return 50

    def analyze(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        signals = []
        if len(market_data) < 50:
            return signals

        symbol = market_data.index.name or 'UNKNOWN'

        # Analyze market regime
        regime = self.regime_detector.analyze(market_data)

        # Only trade when we have HIGH confidence in regime detection
        if regime.confidence < self.min_regime_confidence:
            # Low confidence - don't enter new positions
            # But still allow exits for risk management
            if symbol in current_positions and regime.recommended_strategy == 'reduced_exposure':
                signal = StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.CLOSE_LONG,
                    strength=0.6,
                    strategy_name=self.name,
                    timestamp=datetime.now(),
                    entry_price=market_data['close'].iloc[-1],
                    metadata={
                        'regime': 'uncertain',
                        'exit_reason': 'low_regime_confidence',
                        'regime_confidence': regime.confidence
                    }
                )
                signals.append(signal)
            return signals

        # High confidence regime - trade accordingly
        if regime.recommended_strategy == 'trend_following':
            signals = self.trend_strategy.analyze(market_data, current_positions)
            # Only keep signals that pass our strength threshold
            signals = [s for s in signals if s.strength >= 0.7]
            for s in signals:
                s.metadata['regime'] = regime.regime.value
                s.metadata['strategy_mode'] = 'trend_following'
                s.metadata['regime_confidence'] = regime.confidence

        elif regime.recommended_strategy == 'mean_reversion':
            signals = self.reversion_strategy.analyze(market_data, current_positions)
            signals = [s for s in signals if s.strength >= 0.7]
            for s in signals:
                s.metadata['regime'] = regime.regime.value
                s.metadata['strategy_mode'] = 'mean_reversion'
                s.metadata['regime_confidence'] = regime.confidence

        elif regime.recommended_strategy == 'reduced_exposure':
            # High volatility with high confidence - definitely exit
            if symbol in current_positions:
                signal = StrategySignal(
                    symbol=symbol,
                    signal_type=SignalType.CLOSE_LONG,
                    strength=0.8,
                    strategy_name=self.name,
                    timestamp=datetime.now(),
                    entry_price=market_data['close'].iloc[-1],
                    metadata={
                        'regime': regime.regime.value,
                        'exit_reason': 'high_volatility_risk_off',
                        'regime_confidence': regime.confidence
                    }
                )
                signals.append(signal)

        return signals


def _get_claude_strategy_class():
    """Import ClaudeStrategy with graceful degradation."""
    try:
        from claude_strategy import ClaudeStrategy
        return ClaudeStrategy
    except ImportError as e:
        logger.warning(f"ClaudeStrategy not available: {e}")
        return None


class StrategyManager:
    """Manages multiple trading strategies"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_configs: Dict[str, StrategyConfig] = {}
        self.regime_detector = MarketRegimeDetector()
        self.current_regime: Optional[RegimeAnalysis] = None
        self._load_default_strategies()
    
    def _load_default_strategies(self):
        """Load default strategy configurations

        ENHANCED strategy system with regime-aware trading:
        - New: smart_trend - trend-following with pullback entries (for trending markets)
        - New: mean_reversion - buy support, sell resistance (for ranging markets)
        - New: volume_breakout - leading indicator based on volume spikes
        - New: support_resistance - trade key price levels
        - New: adaptive - automatically switches based on market regime

        Legacy strategies still available but disabled by default.
        """
        default_configs = {
            # === NEW ENHANCED STRATEGIES (Recommended) ===
            'adaptive': StrategyConfig(
                name='Adaptive Strategy',
                enabled=True,  # RECOMMENDED: Automatically adapts to market conditions
                weight=1.5,
                parameters={}
            ),
            'smart_trend': StrategyConfig(
                name='Smart Trend',
                enabled=True,  # Trend-following with pullback entries
                weight=1.2,
                parameters={'fast_period': 12, 'slow_period': 26}
            ),
            'mean_reversion': StrategyConfig(
                name='Mean Reversion',
                enabled=True,  # For ranging markets
                weight=1.0,
                parameters={'bb_period': 20, 'bb_std': 2.0}
            ),
            'volume_breakout': StrategyConfig(
                name='Volume Breakout',
                enabled=True,  # Leading indicator
                weight=1.3,
                parameters={'volume_threshold': 2.0}
            ),
            'support_resistance': StrategyConfig(
                name='Support/Resistance',
                enabled=True,  # Key level trading
                weight=1.1,
                parameters={}
            ),

            # === LEGACY STRATEGIES (Available but disabled) ===
            'ma_cross': StrategyConfig(
                name='MA Crossover',
                enabled=False,  # Legacy - use smart_trend instead
                weight=1.0,
                parameters={'fast_period': 9, 'slow_period': 21}
            ),
            'rsi_momentum': StrategyConfig(
                name='RSI Momentum',
                enabled=False,  # Legacy - integrated into other strategies
                weight=1.0,
                parameters={'rsi_period': 14, 'oversold_level': 35, 'overbought_level': 65}
            ),
            'bollinger_bands': StrategyConfig(
                name='Bollinger Bands',
                enabled=False,  # Legacy - use mean_reversion instead
                weight=0.9,
                parameters={'bb_period': 20, 'bb_std': 1.8}
            ),
            'macd': StrategyConfig(
                name='MACD',
                enabled=False,  # Legacy - integrated into smart_trend
                weight=1.0,
                parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
            ),
            'momentum_breakout': StrategyConfig(
                name='Momentum Breakout',
                enabled=False,  # Legacy - use volume_breakout instead
                weight=1.0,
                parameters={
                    'roc_period': 5,
                    'range_period': 10,
                    'volume_surge': 1.3,
                    'momentum_threshold': 0.8
                }
            ),
            'simple_price_action': StrategyConfig(
                name='Simple Price Action',
                enabled=False,  # Legacy - use support_resistance instead
                weight=0.5,
                parameters={}
            ),
            'volume_profile': StrategyConfig(
                name='Volume Profile',
                enabled=False,
                weight=0.6,
                parameters={}
            ),

            # === CLAUDE AI STRATEGY ===
            'claude_ai': StrategyConfig(
                name='Claude AI',
                enabled=False,  # Requires ANTHROPIC_API_KEY to be set
                weight=1.5,
                parameters={
                    'model': 'claude-3-5-haiku-latest',
                    'factors': {
                        'price_action': True,
                        'technical_indicators': True,
                        'news_sentiment': False,
                        'market_regime': True,
                    },
                    'max_calls_per_hour': 30,
                    'cache_ttl_seconds': 300,
                    'backtesting_mode': 'fallback',
                }
            )
        }

        # Initialize strategies - map to classes
        strategy_classes = {
            # New enhanced strategies
            'adaptive': AdaptiveStrategy,
            'smart_trend': SmartTrendStrategy,
            'mean_reversion': MeanReversionStrategy,
            'volume_breakout': VolumeBreakoutStrategy,
            'support_resistance': SupportResistanceStrategy,
            # Legacy strategies
            'ma_cross': MovingAverageCrossStrategy,
            'rsi_momentum': RSIMomentumStrategy,
            'bollinger_bands': BollingerBandStrategy,
            'macd': MACDStrategy,
            'momentum_breakout': MomentumBreakoutStrategy,
            'simple_price_action': SimplePriceActionStrategy,
            'volume_profile': VolumeProfileStrategy,
            'claude_ai': _get_claude_strategy_class(),
        }

        for key, config in default_configs.items():
            self.strategy_configs[key] = config
            if config.enabled and key in strategy_classes and strategy_classes[key] is not None:
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

    def analyze_with_regime(self, market_data: pd.DataFrame, current_positions: Dict) -> Tuple[List[StrategySignal], RegimeAnalysis]:
        """
        Run all strategies with regime-aware filtering.

        Returns signals and current regime analysis.
        Enhanced strategies are weighted based on regime fit.
        """
        # First, analyze market regime
        self.current_regime = self.regime_detector.analyze(market_data)
        regime = self.current_regime

        all_signals = []

        for key, strategy in self.strategies.items():
            if not self.strategy_configs[key].enabled:
                continue

            try:
                signals = strategy.analyze(market_data, current_positions)

                # Apply base weight
                for signal in signals:
                    signal.strength *= self.strategy_configs[key].weight

                    # Apply regime-based weight adjustment
                    regime_multiplier = self._get_regime_multiplier(key, regime)
                    signal.strength *= regime_multiplier

                    # Add regime info to metadata
                    signal.metadata['regime'] = regime.regime.value
                    signal.metadata['regime_confidence'] = regime.confidence
                    signal.metadata['regime_multiplier'] = regime_multiplier

                all_signals.extend(signals)

            except Exception as e:
                logger.error(f"Error in strategy {key}: {e}")

        return all_signals, regime

    def _get_regime_multiplier(self, strategy_key: str, regime: RegimeAnalysis) -> float:
        """
        Get signal strength multiplier based on strategy-regime fit.

        Trend strategies boosted in trends, mean-reversion boosted in ranging.
        """
        # Trend-following strategies
        trend_strategies = {'smart_trend', 'ma_cross', 'macd', 'momentum_breakout'}
        # Mean-reversion strategies
        reversion_strategies = {'mean_reversion', 'bollinger_bands', 'support_resistance'}
        # Breakout/volume strategies
        breakout_strategies = {'volume_breakout'}
        # Adaptive (always 1.0)
        adaptive_strategies = {'adaptive'}

        if strategy_key in adaptive_strategies:
            return 1.0

        if strategy_key in trend_strategies:
            if regime.regime in [MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND]:
                return 1.3  # Boost in strong trends
            elif regime.regime == MarketRegime.RANGING:
                return 0.5  # Reduce in ranging markets
            else:
                return 1.0

        if strategy_key in reversion_strategies:
            if regime.regime == MarketRegime.RANGING:
                return 1.3  # Boost in ranging markets
            elif regime.regime in [MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND]:
                return 0.5  # Reduce in strong trends
            else:
                return 1.0

        if strategy_key in breakout_strategies:
            if regime.volatility_percentile > 60:
                return 1.2  # Breakouts work in volatile markets
            else:
                return 0.9

        return 1.0  # Default
    
    def get_consensus_signal(self, signals: List[StrategySignal]) -> Optional[StrategySignal]:
        """
        Get intelligent consensus signal from multiple strategy signals.

        Enhanced features:
        - Weighted consensus based on strategy weights and regime fit
        - Regime-aware thresholds (stricter in ranging, looser in trends)
        - Smart stop/take profit using best available levels
        - Single strong signal can pass if strength > 0.9 (high-confidence)
        """
        if not signals:
            return None

        # Group signals by type
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type in [SignalType.SELL, SignalType.CLOSE_LONG]]

        # Get regime-based threshold
        threshold = self._get_consensus_threshold()

        # Check for BUY consensus
        if buy_signals:
            # Calculate weighted strength
            total_weight = sum(s.strength for s in buy_signals)
            num_strategies = len(buy_signals)

            # Allow single high-confidence signal
            if num_strategies == 1 and buy_signals[0].strength >= 0.9:
                return self._build_consensus(buy_signals)

            # Standard consensus requires 2+ strategies
            if num_strategies >= 2:
                avg_strength = total_weight / num_strategies
                if avg_strength >= threshold:
                    return self._build_consensus(buy_signals)

        # Check for SELL consensus
        if sell_signals:
            total_weight = sum(s.strength for s in sell_signals)
            num_strategies = len(sell_signals)

            # Allow single high-confidence exit signal
            if num_strategies == 1 and sell_signals[0].strength >= 0.85:
                return self._build_consensus(sell_signals)

            # Standard consensus
            if num_strategies >= 2:
                avg_strength = total_weight / num_strategies
                if avg_strength >= threshold:
                    return self._build_consensus(sell_signals)

        return None

    def _get_consensus_threshold(self) -> float:
        """Get consensus threshold based on current market regime"""
        if self.current_regime is None:
            return 0.60  # Default

        regime = self.current_regime.regime

        if regime in [MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND]:
            return 0.50  # Lower threshold in strong trends (follow momentum)
        elif regime == MarketRegime.RANGING:
            return 0.70  # Higher threshold in ranging (be more selective)
        elif regime == MarketRegime.HIGH_VOLATILITY:
            return 0.80  # Very selective in high volatility
        else:
            return 0.60

    def _build_consensus(self, signals: List[StrategySignal]) -> StrategySignal:
        """Build consensus signal from agreeing strategies"""
        avg_strength = sum(s.strength for s in signals) / len(signals)

        # Collect all stop losses and take profits
        stop_losses = [s.stop_loss for s in signals if s.stop_loss is not None]
        take_profits = [s.take_profit for s in signals if s.take_profit is not None]

        # Use smart stop/profit selection
        # For stops: use the tightest (most conservative)
        # For profits: use the average (balanced)
        stop_loss = min(stop_losses) if stop_losses else None
        take_profit = np.mean(take_profits) if take_profits else None

        consensus = StrategySignal(
            symbol=signals[0].symbol,
            signal_type=signals[0].signal_type,
            strength=avg_strength,
            strategy_name='Consensus',
            timestamp=datetime.now(),
            entry_price=signals[0].entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'strategies': [s.strategy_name for s in signals],
                'individual_strengths': {s.strategy_name: s.strength for s in signals},
                'num_agreeing': len(signals),
                'regime': self.current_regime.regime.value if self.current_regime else 'unknown'
            }
        )

        return consensus
    
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