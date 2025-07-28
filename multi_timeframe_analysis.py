#!/usr/bin/env python3
"""
Multi-Timeframe Analysis System
Analyzes markets across multiple timeframes for better trading decisions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger('MultiTimeframeAnalysis')

class Timeframe(Enum):
    M1 = "1min"
    M5 = "5min"
    M15 = "15min"
    M30 = "30min"
    H1 = "1hour"
    H4 = "4hour"
    D1 = "1day"
    W1 = "1week"
    MN1 = "1month"
    
    @property
    def minutes(self) -> int:
        """Get timeframe in minutes"""
        mapping = {
            self.M1: 1,
            self.M5: 5,
            self.M15: 15,
            self.M30: 30,
            self.H1: 60,
            self.H4: 240,
            self.D1: 1440,
            self.W1: 10080,
            self.MN1: 43200
        }
        return mapping[self]
    
    def resample_freq(self) -> str:
        """Get pandas resample frequency"""
        mapping = {
            self.M1: '1T',
            self.M5: '5T',
            self.M15: '15T',
            self.M30: '30T',
            self.H1: '1H',
            self.H4: '4H',
            self.D1: '1D',
            self.W1: '1W',
            self.MN1: '1M'
        }
        return mapping[self]

@dataclass
class TimeframeAnalysis:
    """Analysis results for a single timeframe"""
    timeframe: Timeframe
    trend: str  # 'bullish', 'bearish', 'neutral'
    trend_strength: float  # 0-1
    support_levels: List[float]
    resistance_levels: List[float]
    momentum: float  # -1 to 1
    volatility: float
    volume_profile: str  # 'increasing', 'decreasing', 'steady'
    key_levels: List[float]
    signals: List[Dict[str, Any]]
    
@dataclass
class MultiTimeframeSignal:
    """Combined signal from multiple timeframes"""
    symbol: str
    primary_timeframe: Timeframe
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1
    confidence: float  # 0-1
    entry_price: float
    stop_loss: float
    take_profit: List[float]  # Multiple targets
    timeframe_alignment: Dict[Timeframe, str]  # Alignment of each timeframe
    key_levels: Dict[str, float]  # Important price levels
    metadata: Dict[str, Any] = field(default_factory=dict)

class TimeframeAnalyzer(ABC):
    """Base class for timeframe-specific analysis"""
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> TimeframeAnalysis:
        """Analyze data for specific patterns and signals"""
        pass

class TrendAnalyzer(TimeframeAnalyzer):
    """Analyzes trend direction and strength"""
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend using multiple methods"""
        if len(data) < 50:
            return {'trend': 'neutral', 'strength': 0}
        
        # Moving average analysis
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        ema_20 = data['close'].ewm(span=20).mean()
        
        # Trend direction
        current_price = data['close'].iloc[-1]
        ma_score = 0
        
        if current_price > sma_20.iloc[-1]:
            ma_score += 1
        if current_price > sma_50.iloc[-1]:
            ma_score += 1
        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            ma_score += 1
        if current_price > ema_20.iloc[-1]:
            ma_score += 1
        
        # ADX for trend strength
        adx = self._calculate_adx(data)
        
        # Determine trend
        if ma_score >= 3:
            trend = 'bullish'
        elif ma_score <= 1:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        # Trend strength (0-1)
        strength = min(adx / 50, 1.0) * (abs(ma_score - 2) / 2)
        
        return {
            'trend': trend,
            'strength': strength,
            'adx': adx,
            'ma_alignment': ma_score / 4
        }
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Directional movements
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        pos_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=data.index)
        neg_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=data.index)
        
        pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(period).mean()
        
        return adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 0

class SupportResistanceAnalyzer(TimeframeAnalyzer):
    """Identifies support and resistance levels"""
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Find support and resistance levels"""
        if len(data) < 20:
            return {'support': [], 'resistance': []}
        
        # Method 1: Local extrema
        highs = data['high'].rolling(10, center=True).max() == data['high']
        lows = data['low'].rolling(10, center=True).min() == data['low']
        
        resistance_levels = data.loc[highs, 'high'].values[-5:]  # Last 5 highs
        support_levels = data.loc[lows, 'low'].values[-5:]  # Last 5 lows
        
        # Method 2: Volume-weighted levels
        volume_levels = self._find_volume_levels(data)
        
        # Method 3: Fibonacci levels
        fib_levels = self._calculate_fibonacci_levels(data)
        
        # Combine and cluster levels
        all_resistance = list(resistance_levels) + [l for l in volume_levels if l > data['close'].iloc[-1]]
        all_support = list(support_levels) + [l for l in volume_levels if l < data['close'].iloc[-1]]
        
        # Cluster nearby levels
        resistance = self._cluster_levels(all_resistance)
        support = self._cluster_levels(all_support)
        
        return {
            'support': sorted(support, reverse=True)[:3],  # Top 3 support levels
            'resistance': sorted(resistance)[:3],  # Top 3 resistance levels
            'fibonacci': fib_levels
        }
    
    def _find_volume_levels(self, data: pd.DataFrame, bins: int = 20) -> List[float]:
        """Find price levels with high volume"""
        price_range = data['high'].max() - data['low'].min()
        bin_size = price_range / bins
        
        volume_profile = {}
        
        for _, row in data.iterrows():
            # Distribute volume across price range in candle
            price_levels = np.arange(row['low'], row['high'], bin_size)
            volume_per_level = row['volume'] / len(price_levels) if len(price_levels) > 0 else row['volume']
            
            for price in price_levels:
                bin_price = round(price / bin_size) * bin_size
                volume_profile[bin_price] = volume_profile.get(bin_price, 0) + volume_per_level
        
        # Get high volume nodes
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        return [level for level, _ in sorted_levels[:5]]
    
    def _calculate_fibonacci_levels(self, data: pd.DataFrame) -> List[float]:
        """Calculate Fibonacci retracement levels"""
        recent_high = data['high'].iloc[-50:].max()
        recent_low = data['low'].iloc[-50:].min()
        
        diff = recent_high - recent_low
        
        # Fibonacci ratios
        ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        levels = [recent_low + ratio * diff for ratio in ratios]
        
        return levels
    
    def _cluster_levels(self, levels: List[float], threshold: float = 0.01) -> List[float]:
        """Cluster nearby price levels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                # New cluster
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clusters.append(np.mean(current_cluster))
        return clusters

class MomentumAnalyzer(TimeframeAnalyzer):
    """Analyzes momentum indicators"""
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum using multiple indicators"""
        if len(data) < 20:
            return {'momentum': 0, 'rsi': 50, 'macd': 0}
        
        # RSI
        rsi = self._calculate_rsi(data['close'])
        
        # MACD
        macd, signal, histogram = self._calculate_macd(data['close'])
        
        # Stochastic
        stoch_k, stoch_d = self._calculate_stochastic(data)
        
        # Rate of Change
        roc = ((data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]) * 100
        
        # Combined momentum score (-1 to 1)
        momentum_score = 0
        
        # RSI contribution
        if rsi > 70:
            momentum_score += 0.25
        elif rsi < 30:
            momentum_score -= 0.25
        else:
            momentum_score += (rsi - 50) / 200  # Normalized contribution
        
        # MACD contribution
        if histogram > 0:
            momentum_score += 0.25
        else:
            momentum_score -= 0.25
        
        # Stochastic contribution
        if stoch_k > 80:
            momentum_score += 0.25
        elif stoch_k < 20:
            momentum_score -= 0.25
        
        # ROC contribution
        momentum_score += np.clip(roc / 20, -0.25, 0.25)
        
        return {
            'momentum': np.clip(momentum_score, -1, 1),
            'rsi': rsi,
            'macd_histogram': histogram,
            'stochastic': stoch_k,
            'roc': roc
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[float, float, float]:
        """Calculate MACD"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        return macd.iloc[-1], signal.iloc[-1], histogram.iloc[-1]
    
    def _calculate_stochastic(self, data: pd.DataFrame, period: int = 14) -> Tuple[float, float]:
        """Calculate Stochastic oscillator"""
        lowest_low = data['low'].rolling(period).min()
        highest_high = data['high'].rolling(period).max()
        
        k = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(3).mean()
        
        return k.iloc[-1], d.iloc[-1]

class MultiTimeframeAnalyzer:
    """Main multi-timeframe analysis system"""
    
    def __init__(self, timeframes: List[Timeframe] = None):
        self.timeframes = timeframes or [
            Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1
        ]
        
        self.trend_analyzer = TrendAnalyzer()
        self.sr_analyzer = SupportResistanceAnalyzer()
        self.momentum_analyzer = MomentumAnalyzer()
        
    def analyze(self, symbol: str, data: Dict[Timeframe, pd.DataFrame]) -> Dict[Timeframe, TimeframeAnalysis]:
        """Analyze multiple timeframes"""
        analyses = {}
        
        for timeframe in self.timeframes:
            if timeframe in data and len(data[timeframe]) > 0:
                analysis = self._analyze_timeframe(timeframe, data[timeframe])
                analyses[timeframe] = analysis
        
        return analyses
    
    def _analyze_timeframe(self, timeframe: Timeframe, data: pd.DataFrame) -> TimeframeAnalysis:
        """Analyze a single timeframe"""
        # Trend analysis
        trend_result = self.trend_analyzer.analyze(data)
        
        # Support/Resistance
        sr_result = self.sr_analyzer.analyze(data)
        
        # Momentum
        momentum_result = self.momentum_analyzer.analyze(data)
        
        # Volume analysis
        volume_profile = self._analyze_volume_profile(data)
        
        # Volatility
        volatility = data['close'].pct_change().std() * np.sqrt(252)
        
        # Generate signals
        signals = self._generate_timeframe_signals(
            data, trend_result, sr_result, momentum_result
        )
        
        # Key levels (combine support/resistance with other levels)
        key_levels = sorted(
            sr_result['support'] + sr_result['resistance'] + sr_result.get('fibonacci', [])
        )
        
        return TimeframeAnalysis(
            timeframe=timeframe,
            trend=trend_result['trend'],
            trend_strength=trend_result['strength'],
            support_levels=sr_result['support'],
            resistance_levels=sr_result['resistance'],
            momentum=momentum_result['momentum'],
            volatility=volatility,
            volume_profile=volume_profile,
            key_levels=key_levels[:5],  # Top 5 key levels
            signals=signals
        )
    
    def _analyze_volume_profile(self, data: pd.DataFrame) -> str:
        """Analyze volume trend"""
        if len(data) < 20:
            return 'steady'
        
        recent_volume = data['volume'].iloc[-10:].mean()
        previous_volume = data['volume'].iloc[-20:-10].mean()
        
        ratio = recent_volume / previous_volume if previous_volume > 0 else 1
        
        if ratio > 1.2:
            return 'increasing'
        elif ratio < 0.8:
            return 'decreasing'
        else:
            return 'steady'
    
    def _generate_timeframe_signals(self, data: pd.DataFrame, trend: Dict, 
                                   sr: Dict, momentum: Dict) -> List[Dict[str, Any]]:
        """Generate signals for a timeframe"""
        signals = []
        current_price = data['close'].iloc[-1]
        
        # Trend continuation signal
        if trend['trend'] == 'bullish' and trend['strength'] > 0.6:
            if momentum['momentum'] > 0.3:
                signals.append({
                    'type': 'trend_continuation',
                    'direction': 'buy',
                    'strength': trend['strength'] * momentum['momentum']
                })
        
        # Support bounce signal
        if sr['support']:
            nearest_support = min(sr['support'], key=lambda x: abs(x - current_price))
            if (current_price - nearest_support) / nearest_support < 0.01:  # Within 1%
                if momentum['rsi'] < 40:
                    signals.append({
                        'type': 'support_bounce',
                        'direction': 'buy',
                        'level': nearest_support,
                        'strength': 0.7
                    })
        
        # Resistance rejection signal
        if sr['resistance']:
            nearest_resistance = min(sr['resistance'], key=lambda x: abs(x - current_price))
            if (nearest_resistance - current_price) / current_price < 0.01:  # Within 1%
                if momentum['rsi'] > 60:
                    signals.append({
                        'type': 'resistance_rejection',
                        'direction': 'sell',
                        'level': nearest_resistance,
                        'strength': 0.7
                    })
        
        # Momentum divergence signal
        if abs(momentum['macd_histogram']) > 0:
            # Check for divergence (simplified)
            price_trend = 'up' if data['close'].iloc[-1] > data['close'].iloc[-10] else 'down'
            macd_trend = 'up' if momentum['macd_histogram'] > 0 else 'down'
            
            if price_trend != macd_trend:
                signals.append({
                    'type': 'divergence',
                    'direction': 'sell' if price_trend == 'up' else 'buy',
                    'strength': 0.6
                })
        
        return signals
    
    def generate_multi_timeframe_signal(self, symbol: str, 
                                      analyses: Dict[Timeframe, TimeframeAnalysis],
                                      primary_timeframe: Timeframe = Timeframe.H1) -> Optional[MultiTimeframeSignal]:
        """Generate combined signal from multiple timeframe analyses"""
        if primary_timeframe not in analyses:
            return None
        
        primary_analysis = analyses[primary_timeframe]
        
        # Check timeframe alignment
        alignment = self._check_timeframe_alignment(analyses)
        alignment_score = self._calculate_alignment_score(alignment)
        
        # Need minimum alignment to generate signal
        if alignment_score < 0.5:
            return None
        
        # Determine signal type
        bullish_count = sum(1 for a in alignment.values() if a == 'bullish')
        bearish_count = sum(1 for a in alignment.values() if a == 'bearish')
        
        if bullish_count > bearish_count and bullish_count >= len(analyses) * 0.6:
            signal_type = 'buy'
        elif bearish_count > bullish_count and bearish_count >= len(analyses) * 0.6:
            signal_type = 'sell'
        else:
            signal_type = 'hold'
        
        if signal_type == 'hold':
            return None
        
        # Calculate entry, stop loss, and targets
        current_price = self._get_current_price(analyses)
        
        if signal_type == 'buy':
            entry_price = current_price
            stop_loss = self._calculate_stop_loss_buy(analyses, current_price)
            take_profits = self._calculate_take_profits_buy(analyses, current_price)
        else:
            entry_price = current_price
            stop_loss = self._calculate_stop_loss_sell(analyses, current_price)
            take_profits = self._calculate_take_profits_sell(analyses, current_price)
        
        # Calculate signal strength and confidence
        strength = self._calculate_signal_strength(analyses, primary_timeframe)
        confidence = alignment_score * strength
        
        # Get key levels
        key_levels = self._consolidate_key_levels(analyses)
        
        return MultiTimeframeSignal(
            symbol=symbol,
            primary_timeframe=primary_timeframe,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profits,
            timeframe_alignment=alignment,
            key_levels=key_levels,
            metadata={
                'analyses_count': len(analyses),
                'primary_trend': primary_analysis.trend,
                'primary_momentum': primary_analysis.momentum
            }
        )
    
    def _check_timeframe_alignment(self, analyses: Dict[Timeframe, TimeframeAnalysis]) -> Dict[Timeframe, str]:
        """Check trend alignment across timeframes"""
        alignment = {}
        
        for timeframe, analysis in analyses.items():
            # Consider both trend and momentum
            if analysis.trend == 'bullish' and analysis.momentum > 0.2:
                alignment[timeframe] = 'bullish'
            elif analysis.trend == 'bearish' and analysis.momentum < -0.2:
                alignment[timeframe] = 'bearish'
            else:
                alignment[timeframe] = 'neutral'
        
        return alignment
    
    def _calculate_alignment_score(self, alignment: Dict[Timeframe, str]) -> float:
        """Calculate how well timeframes align (0-1)"""
        if not alignment:
            return 0
        
        # Count alignments
        trend_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        for trend in alignment.values():
            trend_counts[trend] += 1
        
        # Calculate score based on majority alignment
        max_count = max(trend_counts.values())
        total_count = len(alignment)
        
        # Penalize neutral counts
        if trend_counts['neutral'] > total_count * 0.5:
            return 0.3
        
        return max_count / total_count
    
    def _get_current_price(self, analyses: Dict[Timeframe, TimeframeAnalysis]) -> float:
        """Get current price from smallest timeframe"""
        # Get the smallest timeframe available
        sorted_timeframes = sorted(analyses.keys(), key=lambda x: x.minutes)
        
        if sorted_timeframes:
            # Would need actual price data here
            # For now, return a placeholder
            return 100.0
        
        return 0
    
    def _calculate_stop_loss_buy(self, analyses: Dict[Timeframe, TimeframeAnalysis], 
                                entry_price: float) -> float:
        """Calculate stop loss for buy signal"""
        # Use support levels from multiple timeframes
        all_supports = []
        
        for analysis in analyses.values():
            all_supports.extend(analysis.support_levels)
        
        # Filter supports below entry price
        valid_supports = [s for s in all_supports if s < entry_price]
        
        if valid_supports:
            # Use the highest support below entry
            stop_loss = max(valid_supports)
        else:
            # Default to 2% below entry
            stop_loss = entry_price * 0.98
        
        return stop_loss
    
    def _calculate_take_profits_buy(self, analyses: Dict[Timeframe, TimeframeAnalysis],
                                   entry_price: float) -> List[float]:
        """Calculate take profit targets for buy signal"""
        # Use resistance levels from multiple timeframes
        all_resistances = []
        
        for analysis in analyses.values():
            all_resistances.extend(analysis.resistance_levels)
        
        # Filter resistances above entry price
        valid_resistances = [r for r in all_resistances if r > entry_price]
        valid_resistances = sorted(valid_resistances)
        
        if valid_resistances:
            # Use first 3 resistance levels as targets
            targets = valid_resistances[:3]
        else:
            # Default targets at 1%, 2%, 3%
            targets = [
                entry_price * 1.01,
                entry_price * 1.02,
                entry_price * 1.03
            ]
        
        return targets
    
    def _calculate_stop_loss_sell(self, analyses: Dict[Timeframe, TimeframeAnalysis],
                                 entry_price: float) -> float:
        """Calculate stop loss for sell signal"""
        # Use resistance levels from multiple timeframes
        all_resistances = []
        
        for analysis in analyses.values():
            all_resistances.extend(analysis.resistance_levels)
        
        # Filter resistances above entry price
        valid_resistances = [r for r in all_resistances if r > entry_price]
        
        if valid_resistances:
            # Use the lowest resistance above entry
            stop_loss = min(valid_resistances)
        else:
            # Default to 2% above entry
            stop_loss = entry_price * 1.02
        
        return stop_loss
    
    def _calculate_take_profits_sell(self, analyses: Dict[Timeframe, TimeframeAnalysis],
                                    entry_price: float) -> List[float]:
        """Calculate take profit targets for sell signal"""
        # Use support levels from multiple timeframes
        all_supports = []
        
        for analysis in analyses.values():
            all_supports.extend(analysis.support_levels)
        
        # Filter supports below entry price
        valid_supports = [s for s in all_supports if s < entry_price]
        valid_supports = sorted(valid_supports, reverse=True)
        
        if valid_supports:
            # Use first 3 support levels as targets
            targets = valid_supports[:3]
        else:
            # Default targets at -1%, -2%, -3%
            targets = [
                entry_price * 0.99,
                entry_price * 0.98,
                entry_price * 0.97
            ]
        
        return targets
    
    def _calculate_signal_strength(self, analyses: Dict[Timeframe, TimeframeAnalysis],
                                  primary_timeframe: Timeframe) -> float:
        """Calculate overall signal strength"""
        primary_analysis = analyses[primary_timeframe]
        
        # Base strength from primary timeframe
        strength = primary_analysis.trend_strength * 0.4
        
        # Add momentum contribution
        strength += abs(primary_analysis.momentum) * 0.3
        
        # Add higher timeframe confirmation
        higher_timeframes = [tf for tf in analyses.keys() if tf.minutes > primary_timeframe.minutes]
        if higher_timeframes:
            higher_alignment = sum(
                1 for tf in higher_timeframes
                if analyses[tf].trend == primary_analysis.trend
            ) / len(higher_timeframes)
            strength += higher_alignment * 0.3
        
        return min(strength, 1.0)
    
    def _consolidate_key_levels(self, analyses: Dict[Timeframe, TimeframeAnalysis]) -> Dict[str, float]:
        """Consolidate key levels from all timeframes"""
        all_levels = []
        
        for analysis in analyses.values():
            all_levels.extend(analysis.key_levels)
        
        # Cluster nearby levels
        clustered = self._cluster_levels(all_levels)
        
        # Create dictionary with descriptive names
        key_levels = {}
        current_price = self._get_current_price(analyses)
        
        # Sort levels
        above_price = sorted([l for l in clustered if l > current_price])
        below_price = sorted([l for l in clustered if l < current_price], reverse=True)
        
        # Name the levels
        for i, level in enumerate(above_price[:3]):
            key_levels[f'resistance_{i+1}'] = level
        
        for i, level in enumerate(below_price[:3]):
            key_levels[f'support_{i+1}'] = level
        
        return key_levels
    
    def _cluster_levels(self, levels: List[float], threshold: float = 0.005) -> List[float]:
        """Cluster nearby price levels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clusters.append(np.mean(current_cluster))
        return clusters

class TimeframeDataManager:
    """Manages data for multiple timeframes"""
    
    def __init__(self):
        self.data: Dict[str, Dict[Timeframe, pd.DataFrame]] = {}
        
    def update_data(self, symbol: str, timeframe: Timeframe, new_data: pd.DataFrame):
        """Update data for a symbol and timeframe"""
        if symbol not in self.data:
            self.data[symbol] = {}
        
        self.data[symbol][timeframe] = new_data
    
    def resample_data(self, symbol: str, base_timeframe: Timeframe, 
                     base_data: pd.DataFrame, target_timeframes: List[Timeframe]):
        """Resample data to multiple timeframes"""
        if symbol not in self.data:
            self.data[symbol] = {}
        
        # Store base timeframe
        self.data[symbol][base_timeframe] = base_data
        
        # Resample to other timeframes
        for target_tf in target_timeframes:
            if target_tf.minutes > base_timeframe.minutes:
                # Can resample up
                resampled = self._resample_ohlcv(base_data, target_tf.resample_freq())
                self.data[symbol][target_tf] = resampled
    
    def _resample_ohlcv(self, data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample OHLCV data to different timeframe"""
        resampled = data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def get_aligned_data(self, symbol: str, timeframes: List[Timeframe], 
                        lookback_periods: Dict[Timeframe, int]) -> Dict[Timeframe, pd.DataFrame]:
        """Get aligned data for multiple timeframes"""
        if symbol not in self.data:
            return {}
        
        aligned_data = {}
        
        for timeframe in timeframes:
            if timeframe in self.data[symbol]:
                df = self.data[symbol][timeframe]
                lookback = lookback_periods.get(timeframe, 100)
                aligned_data[timeframe] = df.tail(lookback)
        
        return aligned_data