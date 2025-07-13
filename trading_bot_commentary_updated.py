#!/usr/bin/env python3
"""
Trading Bot with Real-Time Commentary System
Explains its thinking process in detail while analyzing markets
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from urllib.parse import urlparse, parse_qs
import warnings
warnings.filterwarnings('ignore')

# Core Dependencies
import numpy as np
import pandas as pd
from scipy import stats
import talib

# Make sure pandas is configured to handle the fillna deprecation warning
pd.set_option('mode.copy_on_write', True)

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap

# FastAPI and WebSocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# System imports
import sys
import threading
from queue import Queue
import pickle
from pathlib import Path
import requests
from textblob import TextBlob
import base64
from types import SimpleNamespace
from collections import deque
import yaml
import traceback
from functools import wraps
import time
from contextlib import asynccontextmanager

# Rich console for terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.layout import Layout
    from rich.live import Live
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# Schwab Integration
try:
    from schwab import auth, client as schwab_client
    from schwab.streaming import StreamClient
except ImportError:
    print("Installing schwab-py...")
    os.system("pip install schwab-py")
    from schwab import auth, client as schwab_client
    from schwab.streaming import StreamClient

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import TimeSeriesSplit

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

# Try to import advanced libraries
try:
    import pywt  # For wavelet transforms
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False
    print("PyWavelets not available, using basic features only")

import feedparser
import yfinance as yf
from datetime import datetime, timedelta
import hashlib
from typing import List, Dict, Optional
import asyncio
import re
from bs4 import BeautifulSoup
import aiohttp

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# ============================================================================
# ADVANCED TRADING COMPONENTS
# ============================================================================

# Note: These components are moved to after CommentarySystem class definition
# to avoid circular import issues

class AnomalyDetector:
    """Detects anomalies in market data and trading behavior"""

    def __init__(self, commentary_system):
        self.commentary = commentary_system
        self.last_price = {}
        self.last_volume = {}

    def check_data_anomalies(self, symbol: str, data: pd.DataFrame) -> List[Dict]:
        """Check for anomalies in OHLCV data"""
        anomalies = []

        # Check for large price gaps
        if self.last_price.get(symbol):
            gap = abs(data['Open'].iloc[0] - self.last_price[symbol]) / self.last_price[symbol]
            if gap > 0.1:  # 10% gap
                anomalies.append({
                    'type': 'price_gap',
                    'message': f"Large price gap of {gap:.1%} detected for {symbol}",
                    'level': 'high'
                })

        # Check for extreme volume
        avg_volume = data['Volume'].mean()
        if data['Volume'].iloc[-1] > avg_volume * 5:
            anomalies.append({
                'type': 'volume_spike',
                'message': f"Unusual volume spike detected for {symbol}",
                'level': 'medium'
            })

        # Update last known values
        self.last_price[symbol] = data['Close'].iloc[-1]
        self.last_volume[symbol] = data['Volume'].iloc[-1]

        # Add commentary for anomalies
        for anomaly in anomalies:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.ANOMALY,
                symbol=symbol,
                title=f"Anomaly Detected: {anomaly['type']}",
                message=anomaly['message'],
                importance=8 if anomaly['level'] == 'high' else 6
            ))

        return anomalies

class BehavioralAnalyzer:
    """Analyzes trading behavior and psychological patterns"""
    
    def __init__(self, commentary_system):
        self.commentary = commentary_system
        self.trade_patterns = []
        self.emotional_state = {
            'confidence': 0.5,
            'fear': 0.0,
            'greed': 0.0,
            'patience': 0.5
        }
        
    def analyze_trading_patterns(self, trades: List[Dict]) -> Dict[str, Any]:
        """Analyze trading behavior for psychological patterns"""
        if not trades:
            return {}
        
        df = pd.DataFrame(trades)
        
        # Calculate time between trades
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            df['time_between_trades'] = df['timestamp'].diff().dt.total_seconds() / 3600  # hours
        
        # Detect patterns
        patterns = {
            'overtrading': self._detect_overtrading(df),
            'revenge_trading': self._detect_revenge_trading(df),
            'risk_tolerance_changes': self._detect_risk_changes(df),
            'timing_patterns': self._analyze_timing(df),
            'position_sizing_patterns': self._analyze_position_sizing(df)
        }
        
        # Update emotional state
        self._update_emotional_state(patterns, df)
        
        return patterns
    
    def _detect_overtrading(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect overtrading patterns"""
        if 'time_between_trades' not in df.columns:
            return {}
        
        avg_time_between = df['time_between_trades'].mean()
        recent_trades = df.tail(10)
        recent_avg_time = recent_trades['time_between_trades'].mean()
        
        return {
            'is_overtrading': recent_avg_time < avg_time_between * 0.5,
            'avg_time_between_trades': avg_time_between,
            'recent_avg_time': recent_avg_time,
            'trade_frequency_increase': recent_avg_time < avg_time_between * 0.7
        }
    
    def _detect_revenge_trading(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect revenge trading after losses"""
        if len(df) < 3:
            return {}
        
        # Look for increased position sizes after losses
        df['pnl'] = df.get('pnl', 0)
        df['position_size'] = df.get('position_size', 1)
        
        revenge_patterns = []
        for i in range(1, len(df)):
            prev_trade = df.iloc[i-1]
            current_trade = df.iloc[i]
            
            # Check if previous trade was a loss and current position is larger
            if (prev_trade['pnl'] < 0 and 
                current_trade['position_size'] > prev_trade['position_size'] * 1.5):
                revenge_patterns.append({
                    'index': i,
                    'previous_loss': prev_trade['pnl'],
                    'size_increase': current_trade['position_size'] / prev_trade['position_size']
                })
        
        return {
            'revenge_trading_detected': len(revenge_patterns) > 0,
            'revenge_patterns': revenge_patterns,
            'total_revenge_instances': len(revenge_patterns)
        }
    
    def _detect_risk_changes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect changes in risk tolerance"""
        if len(df) < 10:
            return {}
        
        # Calculate rolling average of position sizes
        df['position_size'] = df.get('position_size', 1)
        df['rolling_avg_size'] = df['position_size'].rolling(window=5).mean()
        
        recent_avg = df['rolling_avg_size'].tail(5).mean()
        earlier_avg = df['rolling_avg_size'].head(5).mean()
        
        risk_change = (recent_avg - earlier_avg) / (earlier_avg + 1e-8)
        
        return {
            'risk_increasing': risk_change > 0.2,
            'risk_decreasing': risk_change < -0.2,
            'risk_change_percent': risk_change * 100,
            'recent_avg_position_size': recent_avg,
            'earlier_avg_position_size': earlier_avg
        }
    
    def _analyze_timing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trading timing patterns"""
        if 'timestamp' not in df.columns:
            return {}
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Find most active trading hours
        hour_counts = df['hour'].value_counts()
        most_active_hour = hour_counts.index[0] if len(hour_counts) > 0 else None
        
        # Check for weekend trading (should be minimal)
        weekend_trades = len(df[df['day_of_week'] >= 5])
        
        return {
            'most_active_hour': most_active_hour,
            'weekend_trades': weekend_trades,
            'trading_hours_distribution': hour_counts.to_dict()
        }
    
    def _analyze_position_sizing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze position sizing patterns"""
        if 'position_size' not in df.columns:
            return {}
        
        df['position_size'] = df.get('position_size', 1)
        
        # Check for consistency in position sizing
        size_std = df['position_size'].std()
        size_mean = df['position_size'].mean()
        size_cv = size_std / (size_mean + 1e-8)  # Coefficient of variation
        
        return {
            'position_size_consistency': size_cv < 0.5,  # Low CV = more consistent
            'size_coefficient_variation': size_cv,
            'avg_position_size': size_mean,
            'position_size_std': size_std
        }
    
    def _update_emotional_state(self, patterns: Dict, df: pd.DataFrame):
        """Update emotional state based on patterns"""
        # Update confidence based on recent performance
        if len(df) > 0:
            recent_pnl = df.tail(5).get('pnl', pd.Series([0])).sum()
            if recent_pnl > 0:
                self.emotional_state['confidence'] = min(1.0, self.emotional_state['confidence'] + 0.1)
            else:
                self.emotional_state['confidence'] = max(0.0, self.emotional_state['confidence'] - 0.1)
        
        # Update fear/greed based on patterns
        if patterns.get('overtrading', {}).get('is_overtrading', False):
            self.emotional_state['fear'] = min(1.0, self.emotional_state['fear'] + 0.2)
        
        if patterns.get('revenge_trading', {}).get('revenge_trading_detected', False):
            self.emotional_state['greed'] = min(1.0, self.emotional_state['greed'] + 0.3)
        
        # Decay emotions over time
        self.emotional_state['fear'] *= 0.95
        self.emotional_state['greed'] *= 0.95
    
    def get_behavioral_recommendations(self) -> List[str]:
        """Get recommendations based on behavioral analysis"""
        recommendations = []
        
        if self.emotional_state['fear'] > 0.7:
            recommendations.append("High fear detected - consider reducing position sizes")
        
        if self.emotional_state['greed'] > 0.7:
            recommendations.append("High greed detected - review risk management")
        
        if self.emotional_state['confidence'] < 0.3:
            recommendations.append("Low confidence - consider taking a trading break")
        
        return recommendations


class DataValidator:
    """Validates data quality and detects anomalies"""
    
    def __init__(self, commentary_system):
        self.commentary = commentary_system
        self.data_quality_history = {}
        
    def validate_market_data(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate market data quality"""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'quality_score': 1.0
        }
        
        if data.empty:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Empty dataset")
            validation_results['quality_score'] = 0.0
            return validation_results
        
        # Check for missing values
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            validation_results['issues'].append(f"Missing data: {missing_data.to_dict()}")
            validation_results['quality_score'] *= 0.8
        
        # Check for price anomalies
        if 'close' in data.columns:
            price_changes = data['close'].pct_change().abs()
            extreme_changes = price_changes > 0.5  # 50% price changes
            
            if extreme_changes.sum() > 0:
                validation_results['issues'].append(f"Extreme price changes detected: {extreme_changes.sum()} instances")
                validation_results['quality_score'] *= 0.7
        
        # Check for volume anomalies
        if 'volume' in data.columns:
            zero_volume = (data['volume'] == 0).sum()
            if zero_volume > len(data) * 0.1:  # More than 10% zero volume
                validation_results['issues'].append(f"High number of zero volume periods: {zero_volume}")
                validation_results['quality_score'] *= 0.9
        
        # Check for stale data
        if 'timestamp' in data.columns:
            latest_time = pd.to_datetime(data['timestamp'].max())
            current_time = pd.Timestamp.now()
            time_diff = (current_time - latest_time).total_seconds() / 3600  # hours
            
            if time_diff > 24:  # Data older than 24 hours
                validation_results['issues'].append(f"Stale data: {time_diff:.1f} hours old")
                validation_results['quality_score'] *= 0.5
        
        # Check for price gaps
        if 'close' in data.columns and len(data) > 1:
            gaps = []
            for i in range(1, len(data)):
                gap = abs(data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
                if gap > 0.1:  # 10% gap
                    gaps.append(gap)
            
            if gaps:
                validation_results['issues'].append(f"Price gaps detected: {len(gaps)} instances")
                validation_results['quality_score'] *= 0.8
        
        # Update history
        self.data_quality_history[symbol] = {
            'timestamp': datetime.now(),
            'quality_score': validation_results['quality_score'],
            'issues': validation_results['issues']
        }
        
        # Alert on poor data quality
        if validation_results['quality_score'] < 0.7:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title="Data Quality Issues",
                message=f"Data quality score: {validation_results['quality_score']:.2f}. "
                       f"Issues: {', '.join(validation_results['issues'])}",
                data=validation_results,
                importance=6
            ))
        
        return validation_results
    
    def cross_validate_sources(self, symbol: str, data_sources: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Cross-validate data from multiple sources"""
        if len(data_sources) < 2:
            return {'is_consistent': True, 'discrepancies': []}
        
        # Compare closing prices across sources
        closing_prices = {}
        for source, data in data_sources.items():
            if not data.empty and 'close' in data.columns:
                closing_prices[source] = data['close'].iloc[-1]
        
        if len(closing_prices) < 2:
            return {'is_consistent': True, 'discrepancies': []}
        
        prices = list(closing_prices.values())
        mean_price = np.mean(prices)
        max_deviation = max(abs(price - mean_price) / mean_price for price in prices)
        
        discrepancies = []
        if max_deviation > 0.01:  # 1% deviation
            for source, price in closing_prices.items():
                deviation = abs(price - mean_price) / mean_price
                if deviation > 0.01:
                    discrepancies.append({
                        'source': source,
                        'price': price,
                        'deviation': deviation
                    })
        
        return {
            'is_consistent': max_deviation <= 0.01,
            'max_deviation': max_deviation,
            'discrepancies': discrepancies,
            'mean_price': mean_price
        }


class AdvancedExitManager:
    """Advanced exit strategies with dynamic adjustments"""
    
    def __init__(self, commentary_system):
        self.commentary = commentary_system
        self.position_tracking = {}
        
    def initialize_trailing_stop(self, symbol: str, entry_price: float, 
                                initial_stop: float, trailing_percent: float = 0.02):
        """Initialize trailing stop for a position"""
        self.position_tracking[symbol] = {
            'entry_price': entry_price,
            'initial_stop': initial_stop,
            'current_stop': initial_stop,
            'trailing_percent': trailing_percent,
            'highest_price': entry_price,
            'lowest_price': entry_price
        }
    
    def update_trailing_stop(self, symbol: str, current_price: float, side: str = 'long') -> float:
        """Update trailing stop based on current price"""
        if symbol not in self.position_tracking:
            return None
        
        tracking = self.position_tracking[symbol]
        
        if side == 'long':
            # Update highest price for long positions
            if current_price > tracking['highest_price']:
                tracking['highest_price'] = current_price
                new_stop = current_price * (1 - tracking['trailing_percent'])
                if new_stop > tracking['current_stop']:
                    tracking['current_stop'] = new_stop
                    
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.TECHNICAL,
                        symbol=symbol,
                        title="Trailing Stop Updated",
                        message=f"Trailing stop moved up to ${new_stop:.2f} (was ${tracking['current_stop']:.2f})",
                        data={'new_stop': new_stop, 'current_price': current_price},
                        importance=5
                    ))
        else:
            # Update lowest price for short positions
            if current_price < tracking['lowest_price']:
                tracking['lowest_price'] = current_price
                new_stop = current_price * (1 + tracking['trailing_percent'])
                if new_stop < tracking['current_stop']:
                    tracking['current_stop'] = new_stop
                    
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.TECHNICAL,
                        symbol=symbol,
                        title="Trailing Stop Updated",
                        message=f"Trailing stop moved down to ${new_stop:.2f} (was ${tracking['current_stop']:.2f})",
                        data={'new_stop': new_stop, 'current_price': current_price},
                        importance=5
                    ))
        
        return tracking['current_stop']
    
    def should_partial_exit(self, position, current_price: float, 
                           indicators: Dict[str, float]) -> Tuple[bool, float]:
        """Determine if partial exit should be taken"""
        # Exit 50% if profit target is hit
        if position.side == 'long':
            profit_target = position.entry_price * 1.05  # 5% profit target
            if current_price >= profit_target:
                return True, 0.5  # Exit 50%
        else:
            profit_target = position.entry_price * 0.95  # 5% profit target
            if current_price <= profit_target:
                return True, 0.5  # Exit 50%
        
        # Exit 25% if RSI is overbought/oversold
        rsi = indicators.get('rsi', 50)
        if position.side == 'long' and rsi > 80:
            return True, 0.25  # Exit 25%
        elif position.side == 'short' and rsi < 20:
            return True, 0.25  # Exit 25%
        
        return False, 0.0
    
    def time_based_exit(self, position, max_hold_days: int = 5) -> bool:
        """Exit position based on time held"""
        days_held = (datetime.now() - position.entry_time).days
        return days_held >= max_hold_days
    
    def volatility_based_exit(self, position, current_volatility: float, 
                             entry_volatility: float) -> bool:
        """Exit position based on volatility changes"""
        volatility_change = abs(current_volatility - entry_volatility) / entry_volatility
        return volatility_change > 0.5  # 50% change in volatility


class AlternativeDataIntegrator:
    """Integrates alternative data sources for enhanced analysis"""
    
    def __init__(self, commentary_system):
        self.commentary = commentary_system
        self.options_flow_cache = {}
        self.insider_trading_cache = {}
        self.social_sentiment_cache = {}
        self.economic_calendar_cache = {}
        
    async def get_options_flow(self, symbol: str) -> Dict[str, Any]:
        """Get unusual options activity"""
        try:
            # This would integrate with options data providers
            # For now, return simulated data
            unusual_activity = {
                'high_volume_calls': np.random.choice([True, False], p=[0.3, 0.7]),
                'high_volume_puts': np.random.choice([True, False], p=[0.2, 0.8]),
                'large_call_sweeps': np.random.randint(0, 5),
                'large_put_sweeps': np.random.randint(0, 3),
                'put_call_ratio': np.random.uniform(0.5, 2.0),
                'unusual_volume': np.random.choice([True, False], p=[0.4, 0.6])
            }
            
            self.options_flow_cache[symbol] = {
                'timestamp': datetime.now(),
                'data': unusual_activity
            }
            
            return unusual_activity
            
        except Exception as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title="Options Flow Error",
                message=f"Failed to fetch options flow data: {str(e)}",
                importance=4
            ))
            return {}
    
    async def get_insider_trading(self, symbol: str) -> List[Dict]:
        """Get recent insider trading activity"""
        try:
            # Simulate insider trading data
            insider_trades = []
            if np.random.random() < 0.3:  # 30% chance of insider activity
                insider_trades = [{
                    'insider': 'CEO',
                    'transaction_type': np.random.choice(['BUY', 'SELL']),
                    'shares': np.random.randint(1000, 10000),
                    'price': np.random.uniform(50, 200),
                    'date': datetime.now() - timedelta(days=np.random.randint(1, 30))
                }]
            
            self.insider_trading_cache[symbol] = {
                'timestamp': datetime.now(),
                'trades': insider_trades
            }
            
            return insider_trades
            
        except Exception as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title="Insider Trading Error",
                message=f"Failed to fetch insider trading data: {str(e)}",
                importance=4
            ))
            return []
    
    async def get_social_sentiment(self, symbol: str) -> Dict[str, float]:
        """Get social media sentiment"""
        try:
            # Simulate social sentiment data
            sentiment_data = {
                'reddit_sentiment': np.random.uniform(-1, 1),
                'twitter_sentiment': np.random.uniform(-1, 1),
                'stocktwits_sentiment': np.random.uniform(-1, 1),
                'overall_sentiment': np.random.uniform(-1, 1),
                'sentiment_volume': np.random.randint(100, 1000)
            }
            
            self.social_sentiment_cache[symbol] = {
                'timestamp': datetime.now(),
                'data': sentiment_data
            }
            
            return sentiment_data
            
        except Exception as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title="Social Sentiment Error",
                message=f"Failed to fetch social sentiment data: {str(e)}",
                importance=4
            ))
            return {}
    
    async def get_economic_calendar(self) -> List[Dict]:
        """Get upcoming economic events"""
        try:
            # Simulate economic calendar
            events = [
                {
                    'event': 'FOMC Meeting',
                    'date': datetime.now() + timedelta(days=7),
                    'impact': 'HIGH',
                    'description': 'Federal Reserve interest rate decision'
                },
                {
                    'event': 'Non-Farm Payrolls',
                    'date': datetime.now() + timedelta(days=3),
                    'impact': 'HIGH',
                    'description': 'Employment data release'
                }
            ]
            
            self.economic_calendar_cache = {
                'timestamp': datetime.now(),
                'events': events
            }
            
            return events
            
        except Exception as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=None,
                title="Economic Calendar Error",
                message=f"Failed to fetch economic calendar: {str(e)}",
                importance=4
            ))
            return []
    
    def analyze_alternative_signals(self, symbol: str) -> Dict[str, Any]:
        """Analyze all alternative data sources for trading signals"""
        signals = {
            'options_bullish': False,
            'options_bearish': False,
            'insider_bullish': False,
            'insider_bearish': False,
            'social_bullish': False,
            'social_bearish': False,
            'overall_signal': 'NEUTRAL'
        }
        
        # Analyze options flow
        if symbol in self.options_flow_cache:
            options_data = self.options_flow_cache[symbol]['data']
            if options_data.get('high_volume_calls', False) and options_data.get('put_call_ratio', 1) < 0.8:
                signals['options_bullish'] = True
            elif options_data.get('high_volume_puts', False) and options_data.get('put_call_ratio', 1) > 1.5:
                signals['options_bearish'] = True
        
        # Analyze insider trading
        if symbol in self.insider_trading_cache:
            insider_trades = self.insider_trading_cache[symbol]['trades']
            recent_buys = sum(1 for trade in insider_trades if trade['transaction_type'] == 'BUY')
            recent_sells = sum(1 for trade in insider_trades if trade['transaction_type'] == 'SELL')
            
            if recent_buys > recent_sells:
                signals['insider_bullish'] = True
            elif recent_sells > recent_buys:
                signals['insider_bearish'] = True
        
        # Analyze social sentiment
        if symbol in self.social_sentiment_cache:
            sentiment_data = self.social_sentiment_cache[symbol]['data']
            if sentiment_data.get('overall_sentiment', 0) > 0.3:
                signals['social_bullish'] = True
            elif sentiment_data.get('overall_sentiment', 0) < -0.3:
                signals['social_bearish'] = True
        
        # Determine overall signal
        bullish_count = sum([signals['options_bullish'], signals['insider_bullish'], signals['social_bullish']])
        bearish_count = sum([signals['options_bearish'], signals['insider_bearish'], signals['social_bearish']])
        
        if bullish_count > bearish_count:
            signals['overall_signal'] = 'BULLISH'
        elif bearish_count > bullish_count:
            signals['overall_signal'] = 'BEARISH'
        
        return signals


class MarketNeutralStrategies:
    """Implements market neutral trading strategies"""
    
    def __init__(self, commentary_system):
        self.commentary = commentary_system
        self.pairs_cache = {}
        self.sector_weights = {}
        
    def find_pairs_trading_opportunities(self, symbols: List[str], 
                                       historical_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Find pairs trading opportunities"""
        opportunities = []
        
        # Calculate correlations between all pairs
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                if symbol1 in historical_data and symbol2 in historical_data:
                    df1 = historical_data[symbol1]
                    df2 = historical_data[symbol2]
                    
                    if len(df1) > 60 and len(df2) > 60:
                        # Calculate correlation
                        returns1 = df1['close'].pct_change().dropna()
                        returns2 = df2['close'].pct_change().dropna()
                        
                        # Align the data
                        common_dates = returns1.index.intersection(returns2.index)
                        if len(common_dates) > 30:
                            corr = returns1.loc[common_dates].corr(returns2.loc[common_dates])
                            
                            if abs(corr) > 0.8:  # High correlation
                                # Calculate spread
                                price1 = df1['close'].iloc[-1]
                                price2 = df2['close'].iloc[-1]
                                
                                # Calculate z-score of the spread
                                spread = price1 - price2
                                spread_series = df1['close'] - df2['close']
                                spread_mean = spread_series.mean()
                                spread_std = spread_series.std()
                                z_score = (spread - spread_mean) / (spread_std + 1e-8)
                                
                                if abs(z_score) > 2:  # Significant deviation
                                    opportunity = {
                                        'symbol1': symbol1,
                                        'symbol2': symbol2,
                                        'correlation': corr,
                                        'z_score': z_score,
                                        'spread': spread,
                                        'signal': 'SHORT_LONG' if z_score > 2 else 'LONG_SHORT',
                                        'confidence': min(abs(z_score) / 3, 1.0)
                                    }
                                    opportunities.append(opportunity)
        
        return opportunities
    
    def calculate_sector_weights(self, symbols: List[str]) -> Dict[str, float]:
        """Calculate sector weights for sector rotation strategy"""
        # This would integrate with a sector classification service
        # For now, use simulated sector data
        sectors = {
            'TECH': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'FINANCE': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'HEALTHCARE': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
            'ENERGY': ['XOM', 'CVX', 'COP', 'EOG', 'SLB']
        }
        
        sector_counts = {}
        for symbol in symbols:
            for sector, sector_symbols in sectors.items():
                if symbol in sector_symbols:
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    break
        
        total_symbols = len(symbols)
        if total_symbols > 0:
            sector_weights = {sector: count/total_symbols for sector, count in sector_counts.items()}
        else:
            sector_weights = {}
        
        self.sector_weights = sector_weights
        return sector_weights
    
    def generate_sector_rotation_signals(self, sector_performance: Dict[str, float]) -> Dict[str, str]:
        """Generate sector rotation signals based on performance"""
        signals = {}
        
        if not sector_performance:
            return signals
        
        # Sort sectors by performance
        sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
        
        # Top performing sector gets overweight signal
        if sorted_sectors:
            signals[sorted_sectors[0][0]] = 'OVERWEIGHT'
        
        # Bottom performing sector gets underweight signal
        if len(sorted_sectors) > 1:
            signals[sorted_sectors[-1][0]] = 'UNDERWEIGHT'
        
        # Middle sectors get neutral
        for sector, _ in sorted_sectors[1:-1]:
            signals[sector] = 'NEUTRAL'
        
        return signals
    
    def calculate_statistical_arbitrage_signals(self, symbol: str, 
                                              historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical arbitrage signals"""
        if len(historical_data) < 60:
            return {}
        
        # Calculate moving averages
        ma_20 = historical_data['close'].rolling(window=20).mean()
        ma_50 = historical_data['close'].rolling(window=50).mean()
        
        current_price = historical_data['close'].iloc[-1]
        current_ma20 = ma_20.iloc[-1]
        current_ma50 = ma_50.iloc[-1]
        
        # Calculate z-score of price deviation from moving average
        price_deviation = current_price - current_ma20
        deviation_std = (historical_data['close'] - ma_20).std()
        z_score = price_deviation / (deviation_std + 1e-8)
        
        # Generate signals
        signal = 'NEUTRAL'
        if z_score > 2:
            signal = 'SHORT'  # Price too high, expect reversion
        elif z_score < -2:
            signal = 'LONG'   # Price too low, expect reversion
        
        return {
            'signal': signal,
            'z_score': z_score,
            'price_deviation': price_deviation,
            'confidence': min(abs(z_score) / 3, 1.0),
            'current_price': current_price,
            'ma20': current_ma20,
            'ma50': current_ma50
        }


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: str = "Config().yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file or create default"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            config = self._get_default_config()
            self._save_config(config)
            return config
    
    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'schwab': {
                'api_key': os.getenv("SCHWAB_API_KEY", "your_api_key"),
                'app_secret': os.getenv("SCHWAB_APP_SECRET", "your_app_secret"),
                'callback_url': "https://127.0.0.1",
                'token_path': "token_1.json",
                'account_number': os.getenv("SCHWAB_ACCOUNT_NUMBER", "")
            },
            'trading': {
                'max_risk_per_trade': 0.02,
                'min_risk_reward_ratio': 2.0,
                'max_daily_loss': 0.05,
                'max_consecutive_losses': 3,
                'position_size_kelly_fraction': 0.25,
                'max_positions': 5,
                'reserve_cash_percent': 0.1,
                'min_position_size': 1,
                'max_position_value': 10000,
                'min_buying_power': 100
            },
            'commentary': {
                'enabled': True,
                'detail_level': "verbose",
                'max_history': 100
            },
            'technical_analysis': {
                'timeframes': ['1min', '5min', '15min', '1hour', '1day'],
                'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786]
            },
            'order_management': {
                'auto_cancel_existing_orders': True,
                'require_close_confirmation': True,
                'confirm_only_losses': False,
                'confirm_threshold_percent': 5
            },
            'paths': {
                'trade_journal': "trade_journal.json",
                'model': "trading_model.pkl",
                'log': "trading_bot.log",
                'commentary_log': "trading_commentary.json",
                'backtest_results': "backtest_results.json"
            },
            'backtesting': {
                'enabled': True,
                'start_date': "2023-01-01",
                'end_date': "2024-01-01",
                'initial_capital': 100000,
                'commission': 0.005,
                'slippage': 0.001
            },
            'performance_metrics': {
                'calculate_sharpe': True,
                'calculate_sortino': True,
                'calculate_max_drawdown': True,
                'risk_free_rate': 0.02
            },
            'error_recovery': {
                'max_retries': 3,
                'retry_delay': 1.0,
                'circuit_breaker_enabled': True,
                'max_daily_loss_threshold': 0.10,
                'emergency_stop_loss': 0.15
            }
        }
    
    def _save_config(self, config: dict):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def _validate_config(self):
        """Validate configuration values"""
        required_sections = ['schwab', 'trading', 'commentary', 'technical_analysis']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def update(self, key: str, value):
        """Update configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self._save_config(self.config)

# Initialize configuration
config_manager = ConfigManager()

# ============================================================================
# ENHANCED ERROR RECOVERY AND CIRCUIT BREAKERS
# ============================================================================

class CircuitBreaker:
    """Circuit breaker pattern for risk management"""
    
    def __init__(self, max_daily_loss: float = 0.10, emergency_stop: float = 0.15):
        self.max_daily_loss = max_daily_loss
        self.emergency_stop = emergency_stop
        self.daily_pnl = 0.0
        self.is_tripped = False
        self.trip_time = None
        self.reset_time = None
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L and check circuit breaker"""
        self.daily_pnl += pnl
        daily_loss_ratio = abs(min(0, self.daily_pnl)) / 100000  # Assuming 100k account
        
        if daily_loss_ratio >= self.emergency_stop:
            self._trip_circuit_breaker("EMERGENCY_STOP", daily_loss_ratio)
        elif daily_loss_ratio >= self.max_daily_loss:
            self._trip_circuit_breaker("MAX_DAILY_LOSS", daily_loss_ratio)
    
    def _trip_circuit_breaker(self, reason: str, loss_ratio: float):
        """Trip the circuit breaker"""
        self.is_tripped = True
        self.trip_time = datetime.now()
        self.reset_time = self.trip_time + timedelta(hours=24)
        logger.critical(f"Circuit breaker tripped: {reason} - Loss ratio: {loss_ratio:.2%}")
    
    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        if not self.is_tripped:
            return True
        
        if datetime.now() >= self.reset_time:
            self._reset_circuit_breaker()
            return True
        
        return False
    
    def _reset_circuit_breaker(self):
        """Reset the circuit breaker"""
        self.is_tripped = False
        self.trip_time = None
        self.reset_time = None
        logger.info("Circuit breaker reset")

class ErrorRecovery:
    """Enhanced error recovery with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.error_counts = {}
    
    async def retry_async(self, func, *args, **kwargs):
        """Retry async function with exponential backoff"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed: {e}")
                    raise last_exception
    
    def retry_sync(self, func, *args, **kwargs):
        """Retry sync function with exponential backoff"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed: {e}")
                    raise last_exception

def error_handler(func):
    """Decorator for error handling and recovery"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Add error recovery logic here
            raise
    return wrapper

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

class Config:
    """Central configuration for the trading bot"""
    # Use config manager instead of hardcoded values
    def __init__(self):
        self.manager = config_manager
    
    @property
    def SCHWAB_API_KEY(self):
        return self.manager.get('schwab.api_key')
    
    @property
    def SCHWAB_APP_SECRET(self):
        return self.manager.get('schwab.app_secret')
    
    @property
    def SCHWAB_CALLBACK_URL(self):
        return self.manager.get('schwab.callback_url')
    
    @property
    def SCHWAB_TOKEN_PATH(self):
        return Path(self.manager.get('schwab.token_path'))
    
    @property
    def SCHWAB_ACCOUNT_NUMBER(self):
        return self.manager.get('schwab.account_number')
    
    @property
    def MAX_RISK_PER_TRADE(self):
        return self.manager.get('trading.max_risk_per_trade')
    
    @property
    def MIN_RISK_REWARD_RATIO(self):
        return self.manager.get('trading.min_risk_reward_ratio')
    
    @property
    def MAX_DAILY_LOSS(self):
        return self.manager.get('trading.max_daily_loss')
    
    @property
    def MAX_CONSECUTIVE_LOSSES(self):
        return self.manager.get('trading.max_consecutive_losses')
    
    @property
    def POSITION_SIZE_KELLY_FRACTION(self):
        return self.manager.get('trading.position_size_kelly_fraction')
    
    @property
    def COMMENTARY_ENABLED(self):
        return self.manager.get('commentary.enabled')
    
    @property
    def COMMENTARY_DETAIL_LEVEL(self):
        return self.manager.get('commentary.detail_level')
    
    @property
    def MAX_COMMENTARY_HISTORY(self):
        return self.manager.get('commentary.max_history')
    
    @property
    def TIMEFRAMES(self):
        return self.manager.get('technical_analysis.timeframes')
    
    @property
    def FIBONACCI_LEVELS(self):
        return self.manager.get('technical_analysis.fibonacci_levels')
    
    @property
    def TRADE_JOURNAL_PATH(self):
        return Path(self.manager.get('paths.trade_journal'))
    
    @property
    def MODEL_PATH(self):
        return Path(self.manager.get('paths.model'))
    
    @property
    def LOG_PATH(self):
        return Path(self.manager.get('paths.log'))
    
    @property
    def COMMENTARY_LOG_PATH(self):
        return Path(self.manager.get('paths.commentary_log'))
    
    @property
    def AUTO_CANCEL_EXISTING_ORDERS(self):
        return self.manager.get('order_management.auto_cancel_existing_orders')
    
    @property
    def MAX_POSITIONS(self):
        return self.manager.get('trading.max_positions')
    
    @property
    def RESERVE_CASH_PERCENT(self):
        return self.manager.get('trading.reserve_cash_percent')
    
    @property
    def MIN_POSITION_SIZE(self):
        return self.manager.get('trading.min_position_size')
    
    @property
    def MAX_POSITION_VALUE(self):
        return self.manager.get('trading.max_position_value')
    
    @property
    def MIN_BUYING_POWER(self):
        return self.manager.get('trading.min_buying_power')
    
    @property
    def REQUIRE_CLOSE_CONFIRMATION(self):
        return self.manager.get('order_management.require_close_confirmation')
    
    @property
    def CONFIRM_ONLY_LOSSES(self):
        return self.manager.get('order_management.confirm_only_losses')
    
    @property
    def CONFIRM_THRESHOLD_PERCENT(self):
        return self.manager.get('order_management.confirm_threshold_percent')

# Initialize configuration
config = Config()
# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure logging"""
    logger = logging.getLogger('TradingBot')
    logger.setLevel(logging.DEBUG)
    
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        str(Config().LOG_PATH), 
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# ============================================================================
# COMPREHENSIVE BACKTESTING FRAMEWORK
# ============================================================================

@dataclass
class BacktestResult:
    """Results from backtesting"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    equity_curve: List[float]
    trade_history: List[Dict]
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float

class BacktestEngine:
    """Comprehensive backtesting engine"""
    
    def __init__(self, config: dict):
        self.config = config
        self.initial_capital = config.get('backtesting', {}).get('initial_capital', 100000)
        self.commission = config.get('backtesting', {}).get('commission', 0.005)
        self.slippage = config.get('backtesting', {}).get('slippage', 0.001)
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trade_history = []
        self.equity_curve = [self.initial_capital]
        self.current_date = None
    
    def run_backtest(self, data: Dict[str, pd.DataFrame], strategy, 
                    start_date: str, end_date: str) -> BacktestResult:
        """Run backtest on historical data"""
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Prepare data
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Get all unique dates
        all_dates = set()
        for symbol, df in data.items():
            df['date'] = pd.to_datetime(df.index)
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            all_dates.update(df['date'].dt.date)
        
        all_dates = sorted(list(all_dates))
        
        # Run simulation
        for date in all_dates:
            self.current_date = date
            self._process_date(date, data, strategy)
            self._update_equity_curve()
        
        # Calculate results
        return self._calculate_results(start_dt, end_dt)
    
    def _process_date(self, date, data: Dict[str, pd.DataFrame], strategy):
        """Process a single date"""
        # Update positions with current prices
        for symbol, position in self.positions.items():
            if symbol in data:
                df = data[symbol]
                current_price = df[df['date'].dt.date == date]['close'].iloc[0]
                position['current_price'] = current_price
                position['unrealized_pnl'] = (
                    current_price - position['entry_price']
                ) * position['quantity']
        
        # Generate signals
        for symbol, df in data.items():
            if symbol in df and df[df['date'].dt.date == date].shape[0] > 0:
                current_data = df[df['date'].dt.date == date].iloc[0]
                signal = strategy.generate_signal(current_data)
                
                if signal:
                    self._execute_signal(signal, current_data)
    
    def _execute_signal(self, signal, current_data):
        """Execute a trading signal"""
        symbol = signal.symbol
        price = current_data['close']
        
        # Apply slippage
        if signal.signal_type == SignalType.BUY:
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)
        
        # Calculate position size
        position_value = signal.position_size * execution_price
        commission_cost = position_value * self.commission
        
        if signal.signal_type == SignalType.BUY:
            if self.current_capital >= position_value + commission_cost:
                self.positions[symbol] = {
                    'entry_price': execution_price,
                    'quantity': signal.position_size,
                    'entry_date': self.current_date,
                    'current_price': execution_price,
                    'unrealized_pnl': 0
                }
                self.current_capital -= (position_value + commission_cost)
                self._record_trade(symbol, 'BUY', execution_price, signal.position_size, commission_cost)
        
        elif signal.signal_type == SignalType.SELL and symbol in self.positions:
            position = self.positions[symbol]
            exit_value = position['quantity'] * execution_price
            commission_cost = exit_value * self.commission
            pnl = (execution_price - position['entry_price']) * position['quantity'] - commission_cost
            
            self.current_capital += (exit_value - commission_cost)
            del self.positions[symbol]
            self._record_trade(symbol, 'SELL', execution_price, position['quantity'], commission_cost, pnl)
    
    def _record_trade(self, symbol: str, side: str, price: float, quantity: int, 
                     commission: float, pnl: float = 0):
        """Record a trade"""
        trade = {
            'date': self.current_date,
            'symbol': symbol,
            'side': side,
            'price': price,
            'quantity': quantity,
            'commission': commission,
            'pnl': pnl,
            'capital': self.current_capital
        }
        self.trade_history.append(trade)
    
    def _update_equity_curve(self):
        """Update equity curve"""
        total_value = self.current_capital
        for position in self.positions.values():
            total_value += position['unrealized_pnl']
        self.equity_curve.append(total_value)
    
    def _calculate_results(self, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
        days = (end_date - start_date).days
        annualized_return = (1 + total_return) ** (365 / days) - 1
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown, max_drawdown_duration = self._calculate_max_drawdown(equity_series)
        
        # Trade metrics
        trades_df = pd.DataFrame(self.trade_history)
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else float('inf')
            
            average_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            average_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            largest_win = winning_trades['pnl'].max() if len(winning_trades) > 0 else 0
            largest_loss = losing_trades['pnl'].min() if len(losing_trades) > 0 else 0
        else:
            win_rate = profit_factor = average_win = average_loss = largest_win = largest_loss = 0
        
        # Consecutive trades
        consecutive_wins, consecutive_losses = self._calculate_consecutive_trades(trades_df)
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades_df),
            winning_trades=len(winning_trades) if len(trades_df) > 0 else 0,
            losing_trades=len(losing_trades) if len(trades_df) > 0 else 0,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            equity_curve=self.equity_curve,
            trade_history=self.trade_history,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=equity_series.iloc[-1]
        )
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0
        risk_free_rate = self.config.get('performance_metrics', {}).get('risk_free_rate', 0.02) / 252
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0
        risk_free_rate = self.config.get('performance_metrics', {}).get('risk_free_rate', 0.02) / 252
        excess_returns = returns - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calculate duration
        peak_idx = equity_series.idxmax()
        bottom_idx = drawdown.idxmin()
        if peak_idx < bottom_idx:
            max_drawdown_duration = (bottom_idx - peak_idx).days
        else:
            max_drawdown_duration = 0
        
        return max_drawdown, max_drawdown_duration
    
    def _calculate_consecutive_trades(self, trades_df: pd.DataFrame) -> Tuple[int, int]:
        """Calculate consecutive wins and losses"""
        if len(trades_df) == 0:
            return 0, 0
        
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for pnl in trades_df['pnl']:
            if pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        return max_consecutive_wins, max_consecutive_losses

# ============================================================================
# ENHANCED PERFORMANCE METRICS
# ============================================================================

class PerformanceAnalyzer:
    """Comprehensive performance analysis"""
    
    def __init__(self, config: dict):
        self.config = config
        self.metrics_history = []
    
    def calculate_metrics(self, positions: List['Position'], trade_history: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not trade_history:
            return self._get_empty_metrics()
        
        trades_df = pd.DataFrame(trade_history)
        returns = trades_df['pnl'].sum() / 100000  # Assuming 100k account
        
        metrics = {
            'total_return': returns,
            'total_trades': len(trades_df),
            'win_rate': self._calculate_win_rate(trades_df),
            'profit_factor': self._calculate_profit_factor(trades_df),
            'average_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0,
            'average_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0,
            'largest_win': trades_df['pnl'].max(),
            'largest_loss': trades_df['pnl'].min(),
            'sharpe_ratio': self._calculate_sharpe_ratio(trades_df),
            'sortino_ratio': self._calculate_sortino_ratio(trades_df),
            'max_drawdown': self._calculate_max_drawdown(trades_df),
            'calmar_ratio': self._calculate_calmar_ratio(trades_df),
            'var_95': self._calculate_var(trades_df, 0.95),
            'cvar_95': self._calculate_cvar(trades_df, 0.95),
            'kelly_criterion': self._calculate_kelly_criterion(trades_df),
            'expectancy': self._calculate_expectancy(trades_df),
            'risk_reward_ratio': self._calculate_risk_reward_ratio(trades_df)
        }
        
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        return metrics
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics structure"""
        return {
            'total_return': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'kelly_criterion': 0.0,
            'expectancy': 0.0,
            'risk_reward_ratio': 0.0
        }
    
    def _calculate_win_rate(self, trades_df: pd.DataFrame) -> float:
        """Calculate win rate"""
        if len(trades_df) == 0:
            return 0.0
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        return winning_trades / len(trades_df)
    
    def _calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """Calculate profit factor"""
        winning_trades = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        losing_trades = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        return winning_trades / losing_trades if losing_trades > 0 else float('inf')
    
    def _calculate_sharpe_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio"""
        if len(trades_df) == 0:
            return 0.0
        returns = trades_df['pnl'] / 100000  # Normalize to account size
        risk_free_rate = self.Config().get('performance_metrics.risk_free_rate', 0.02) / 252
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
    
    def _calculate_sortino_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate Sortino ratio"""
        if len(trades_df) == 0:
            return 0.0
        returns = trades_df['pnl'] / 100000
        risk_free_rate = self.Config().get('performance_metrics.risk_free_rate', 0.02) / 252
        excess_returns = returns - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0
    
    def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if len(trades_df) == 0:
            return 0.0
        cumulative_pnl = trades_df['pnl'].cumsum()
        peak = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - peak) / 100000  # Normalize to account size
        return abs(drawdown.min())
    
    def _calculate_calmar_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate Calmar ratio"""
        if len(trades_df) == 0:
            return 0.0
        total_return = trades_df['pnl'].sum() / 100000
        max_drawdown = self._calculate_max_drawdown(trades_df)
        return total_return / max_drawdown if max_drawdown > 0 else 0
    
    def _calculate_var(self, trades_df: pd.DataFrame, confidence: float) -> float:
        """Calculate Value at Risk"""
        if len(trades_df) == 0:
            return 0.0
        returns = trades_df['pnl'] / 100000
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, trades_df: pd.DataFrame, confidence: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(trades_df) == 0:
            return 0.0
        returns = trades_df['pnl'] / 100000
        var = self._calculate_var(trades_df, confidence)
        return returns[returns <= var].mean()
    
    def _calculate_kelly_criterion(self, trades_df: pd.DataFrame) -> float:
        """Calculate Kelly Criterion"""
        if len(trades_df) == 0:
            return 0.0
        win_rate = self._calculate_win_rate(trades_df)
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 1
        
        if avg_loss == 0:
            return 0.0
        
        return (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win > 0 else 0
    
    def _calculate_expectancy(self, trades_df: pd.DataFrame) -> float:
        """Calculate expectancy per trade"""
        if len(trades_df) == 0:
            return 0.0
        win_rate = self._calculate_win_rate(trades_df)
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
        
        return win_rate * avg_win - (1 - win_rate) * avg_loss
    
    def _calculate_risk_reward_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate risk-reward ratio"""
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 1
        
        return avg_win / avg_loss if avg_loss > 0 else 0

# ============================================================================
# TRADING MODES WITH COMMENTARY
# ============================================================================

class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"
    SIMULATION_WITH_COMMENTARY = "simulation_commentary"


class CommentaryType(Enum):
    MARKET_ANALYSIS = "market_analysis"
    SIGNAL_GENERATION = "signal_generation"
    RISK_ASSESSMENT = "risk_assessment"
    DECISION = "decision"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    PSYCHOLOGY = "psychology"
    WARNING = "warning"
    OPPORTUNITY = "opportunity"
    ANOMALY = "anomaly"

class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

class NewsImpact(Enum):
    BREAKING = "breaking"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class CommentaryType(Enum):
    MARKET_ANALYSIS = "market_analysis"
    SIGNAL_GENERATION = "signal_generation"
    RISK_ASSESSMENT = "risk_assessment"
    DECISION = "decision"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    PSYCHOLOGY = "psychology"
    WARNING = "warning"
    OPPORTUNITY = "opportunity"
    ANOMALY = "anomaly"
# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TradingSignal:
    symbol: str
    signal_type: SignalType
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: int
    reasoning: Dict[str, Any]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Position:
    symbol: str
    entry_price: float
    current_price: float
    quantity: int
    side: str
    stop_loss: float
    take_profit: float
    entry_time: datetime
    unrealized_pnl: float = 0
    reasoning: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str
    indicators: Dict[str, float] = field(default_factory=dict)

@dataclass
class NewsItem:
    id: str
    symbol: str
    headline: str
    summary: str
    source: str
    url: str
    published_time: datetime
    sentiment_score: float = 0.0
    sentiment_confidence: float = 0.0
    impact: NewsImpact = NewsImpact.LOW
    relevance_score: float = 0.0
    
    def age_hours(self) -> float:
        return (datetime.now() - self.published_time).total_seconds() / 3600

# ============================================================================
# BACKTEST-SPECIFIC COMPONENTS
# ============================================================================

class BacktestStrategy(ABC):
    """Base class for backtest-compatible strategies"""
    
    @abstractmethod
    def generate_signal(self, symbol: str, data: pd.DataFrame, current_index: int) -> Optional[TradingSignal]:
        """Generate signal for backtesting"""
        pass

class BacktestBreakoutStrategy(BacktestStrategy):
    """Breakout strategy for backtesting"""
    
    def generate_signal(self, symbol: str, data: pd.DataFrame, current_index: int) -> Optional[TradingSignal]:
        if current_index < 50:  # Need history for indicators
            return None
            
        # Get current and historical data
        historical_data = data.iloc[:current_index+1]
        current_bar = data.iloc[current_index]
        
        # Calculate indicators
        close_prices = historical_data['Close'].values
        high_prices = historical_data['High'].values
        low_prices = historical_data['Low'].values
        volume_values = historical_data['Volume'].values
        
        # Simple resistance calculation
        resistance = float(high_prices[-20:].max())
        support = float(low_prices[-20:].min())
        
        # Get current values as floats
        current_close = float(current_bar['Close'])
        current_volume = float(current_bar['Volume'])
        avg_volume = float(volume_values[-20:].mean())
        
        # Check for breakout
        if current_close > resistance and current_volume > avg_volume * 1.5:
            stop_loss = current_close * 0.98
            take_profit = current_close * 1.05
            
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=0.8,
                entry_price=current_close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=0,
                reasoning={
                    'strategy': 'breakout',
                    'resistance_level': resistance,
                    'volume_surge': True
                },
                confidence=0.75
            )
        
        return None

class BacktestMeanReversionStrategy(BacktestStrategy):
    """Mean reversion strategy for backtesting"""
    
    def generate_signal(self, symbol: str, data: pd.DataFrame, current_index: int) -> Optional[TradingSignal]:
        if current_index < 50:
            return None
        
        historical_data = data.iloc[:current_index+1]
        current_bar = data.iloc[current_index]
        
        # Calculate indicators
        close_prices = historical_data['Close'].values
        
        # RSI
        rsi = float(self._calculate_rsi(close_prices))
        
        # Bollinger Bands
        sma20 = float(historical_data['Close'].iloc[-20:].mean())
        std20 = float(historical_data['Close'].iloc[-20:].std())
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        
        # Get current close as float
        current_close = float(current_bar['Close'])
        
        # --- News Sentiment Integration ---
        news_positive = 0
        news_negative = 0
        news_score = 0.0
        if 'news_positive_count' in current_bar and 'news_negative_count' in current_bar and 'news_sentiment_score' in current_bar:
            news_positive = current_bar['news_positive_count']
            news_negative = current_bar['news_negative_count']
            news_score = current_bar['news_sentiment_score']
        # Only allow mean reversion BUY if recent news is not negative
        if news_negative > news_positive:
            return None
        # Generate signals
        if rsi < 30 and current_close < bb_lower:
            stop_loss = current_close * 0.98
            take_profit = sma20
            base_confidence = 0.65
            if news_positive > 0 and news_score > 0.8:
                base_confidence += 0.1
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=0.7,
                entry_price=current_close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=0,
                reasoning={
                    'strategy': 'mean_reversion',
                    'rsi': rsi,
                    'bb_position': 'below_lower',
                    'news_positive': news_positive,
                    'news_negative': news_negative,
                    'news_score': news_score
                },
                confidence=base_confidence
            )
        return None
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

# ============================================================================
# FIXED BACKTEST ENGINE
# ============================================================================

class FixedBacktestEngine:
    """Fixed backtesting engine that works with the data structure"""
    
    def __init__(self, config: dict):
        self.config = config
        self.initial_capital = config.get('backtesting', {}).get('initial_capital', 100000)
        self.commission = config.get('backtesting', {}).get('commission', 0.005)
        self.slippage = config.get('backtesting', {}).get('slippage', 0.001)
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trade_history = []
        self.equity_curve = [self.initial_capital]
        self.current_date = None

    def walk_forward_validation(self, data: Dict[str, pd.DataFrame], strategy_class, start_date: str, end_date: str, window_size: int = 252, step_size: int = 63) -> List[Dict[str, Any]]:
        """
        Walk-forward validation for robust out-of-sample evaluation.
        window_size: number of bars in each training window (default: 1 year of trading days)
        step_size: number of bars to step forward each iteration (default: 1 quarter)
        Returns a list of result dicts for each walk.
        """
        results = []
        for symbol, df in data.items():
            df = df.sort_index()
            n = len(df)
            for start in range(0, n - window_size, step_size):
                train_start = start
                train_end = start + window_size
                test_start = train_end
                test_end = min(test_start + step_size, n)
                if test_end <= test_start:
                    break
                train_df = df.iloc[train_start:train_end]
                test_df = df.iloc[test_start:test_end]
                # Train strategy (if needed)
                strategy = strategy_class()  # stateless for now
                # Run backtest on test window
                test_data = {symbol: test_df}
                test_engine = FixedBacktestEngine(self.config)
                result = test_engine.run_backtest(test_data, strategy, str(test_df.index[0].date()), str(test_df.index[-1].date()))
                results.append({
                    'symbol': symbol,
                    'train_start': str(train_df.index[0].date()),
                    'train_end': str(train_df.index[-1].date()),
                    'test_start': str(test_df.index[0].date()),
                    'test_end': str(test_df.index[-1].date()),
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'total_trades': result.total_trades
                })
        return results
    
    def run_backtest(self, data: Dict[str, pd.DataFrame], strategy, 
                    start_date: str, end_date: str) -> BacktestResult:
        """Run backtest on historical data"""
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Convert string dates to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Process each symbol
        for symbol, df in data.items():
            # Filter data by date range
            mask = (df.index >= start_dt) & (df.index <= end_dt)
            filtered_df = df.loc[mask]
            
            if len(filtered_df) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                continue
            
            # Process each bar
            for i in range(50, len(filtered_df)):
                self.current_date = filtered_df.index[i]
                
                # Check for exit conditions on existing positions
                self._check_exits(symbol, filtered_df.iloc[i])
                
                # Generate new signals
                signal = strategy.generate_signal(symbol, filtered_df, i)
                
                if signal and symbol not in self.positions:
                    self._execute_signal(signal, filtered_df.iloc[i])
                
                # Update equity
                self._update_equity()
        
        # Close all remaining positions
        for symbol in list(self.positions.keys()):
            if symbol in data:
                last_price = data[symbol]['Close'].iloc[-1]
                self._close_position(symbol, last_price, 'end_of_backtest')
        
        # Calculate final results
        return self._calculate_results(start_dt, end_dt)
    
    def _check_exits(self, symbol: str, current_bar):
        """Check exit conditions for existing positions"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        current_price = current_bar['Close']
        
        # Check stop loss
        if current_price <= position['stop_loss']:
            self._close_position(symbol, current_price, 'stop_loss')
        # Check take profit
        elif current_price >= position['take_profit']:
            self._close_position(symbol, current_price, 'take_profit')
    
    def _execute_signal(self, signal: TradingSignal, current_bar):
        """Execute a trading signal"""
        # Calculate position size based on risk
        risk_amount = self.current_capital * 0.02  # 2% risk
        risk_per_share = abs(signal.entry_price - signal.stop_loss)
        
        if risk_per_share > 0:
            position_size = int(risk_amount / risk_per_share)
        else:
            return
        
        # Check if we have enough capital
        required_capital = position_size * signal.entry_price * (1 + self.commission)
        
        if required_capital > self.current_capital:
            position_size = int(self.current_capital * 0.95 / (signal.entry_price * (1 + self.commission)))
        
        if position_size == 0:
            return
        
        # Apply slippage
        execution_price = signal.entry_price * (1 + self.slippage)
        commission_cost = position_size * execution_price * self.commission
        
        # Create position
        self.positions[signal.symbol] = {
            'entry_price': execution_price,
            'quantity': position_size,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'entry_date': self.current_date
        }
        
        # Update capital
        self.current_capital -= (position_size * execution_price + commission_cost)
        
        # Record trade
        self._record_trade(signal.symbol, 'BUY', execution_price, position_size, commission_cost)
    
    def _close_position(self, symbol: str, exit_price: float, reason: str):
        """Close a position"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        
        # Apply slippage
        execution_price = exit_price * (1 - self.slippage)
        commission_cost = position['quantity'] * execution_price * self.commission
        
        # Calculate PnL
        gross_pnl = (execution_price - position['entry_price']) * position['quantity']
        net_pnl = gross_pnl - commission_cost
        
        # Update capital
        self.current_capital += (position['quantity'] * execution_price - commission_cost)
        
        # Record trade
        self._record_trade(symbol, 'SELL', execution_price, position['quantity'], commission_cost, net_pnl)
        
        # Remove position
        del self.positions[symbol]
    
    def _record_trade(self, symbol: str, side: str, price: float, quantity: int, 
                     commission: float, pnl: float = 0):
        """Record a trade"""
        self.trade_history.append({
            'date': self.current_date,
            'symbol': symbol,
            'side': side,
            'price': price,
            'quantity': quantity,
            'commission': commission,
            'pnl': pnl,
            'capital': self.current_capital
        })
    
    def _update_equity(self):
        """Update equity curve"""
        # Calculate total portfolio value
        total_value = self.current_capital
        
        # Add unrealized P&L from open positions
        # (This is simplified - in real backtesting you'd use current market prices)
        
        self.equity_curve.append(total_value)
    
    def _calculate_results(self, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        equity_series = pd.Series(self.equity_curve)
        
        # Calculate returns
        if len(equity_series) > 1:
            returns = equity_series.pct_change().dropna()
            total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
        else:
            returns = pd.Series([0])
            total_return = 0
        
        # Annualized return
        days = (end_date - start_date).days
        annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1 if days > 0 else 0
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns) if len(returns) > 1 else 0
        sortino_ratio = self._calculate_sortino_ratio(returns) if len(returns) > 1 else 0
        max_drawdown, max_drawdown_duration = self._calculate_max_drawdown(equity_series)
        
        # Trade metrics
        trades_df = pd.DataFrame(self.trade_history)
        
        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            
            if len(losing_trades) > 0 and len(winning_trades) > 0:
                profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum())
            else:
                profit_factor = float('inf') if len(winning_trades) > 0 else 0
            
            average_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            average_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            largest_win = winning_trades['pnl'].max() if len(winning_trades) > 0 else 0
            largest_loss = losing_trades['pnl'].min() if len(losing_trades) > 0 else 0
        else:
            win_rate = profit_factor = average_win = average_loss = largest_win = largest_loss = 0
        
        # Consecutive trades
        consecutive_wins, consecutive_losses = self._calculate_consecutive_trades(trades_df)
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades_df),
            winning_trades=len(winning_trades) if 'winning_trades' in locals() else 0,
            losing_trades=len(losing_trades) if 'losing_trades' in locals() else 0,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            equity_curve=self.equity_curve,
            trade_history=self.trade_history,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=equity_series.iloc[-1] if len(equity_series) > 0 else self.initial_capital
        )
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        return np.sqrt(252) * returns.mean() / returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        return np.sqrt(252) * returns.mean() / downside_returns.std()
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        if len(equity_series) < 2:
            return 0, 0
            
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Duration calculation
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i in range(len(drawdown)):
            if drawdown.iloc[i] < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                drawdown_start = None
                current_duration = 0
        
        return abs(max_drawdown), max_duration
    
    def _calculate_consecutive_trades(self, trades_df: pd.DataFrame) -> Tuple[int, int]:
        """Calculate consecutive wins and losses"""
        if len(trades_df) == 0 or 'pnl' not in trades_df.columns:
            return 0, 0
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for pnl in trades_df['pnl']:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        return max_consecutive_wins, max_consecutive_losses

# ============================================================================
# WEB INTERFACE WITH COMMENTARY DISPLAY
# ============================================================================

DASHBOARD_HTML_WITH_COMMENTARY = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot with Schwab Positions</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #0a0a0a;
            color: #fff;
        }
        .settings-row {
            display: flex;
            align-items: center;
            gap: 20px;
            flex-wrap: wrap;
        }

        .toggle-group {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .container {
            display: flex;
            height: 100vh;
        }
        .commentary-panel {
            flex: 1;
            background: #111;
            border-right: 1px solid #333;
            overflow-y: auto;
            padding: 20px;
            max-width: 600px;
        }
        .dashboard-panel {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
        }
        .commentary-item {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            animation: slideIn 0.3s ease-out;
        }
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        .commentary-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .commentary-title {
            font-weight: bold;
            font-size: 16px;
        }
        .commentary-time {
            color: #666;
            font-size: 12px;
        }
        .commentary-message {
            line-height: 1.6;
            margin-bottom: 10px;
            white-space: pre-line;
        }
        .commentary-data {
            background: #0a0a0a;
            padding: 10px;
            border-radius: 4px;
            font-size: 14px;
            font-family: monospace;
        }
        .confidence-bar {
            height: 4px;
            background: #333;
            border-radius: 2px;
            margin-top: 10px;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: #4a9eff;
            transition: width 0.3s ease;
        }
        
        /* Commentary type colors */
        .commentary-market_analysis { border-left: 4px solid #4a9eff; }
        .commentary-signal_generation { border-left: 4px solid #ffd700; }
        .commentary-risk_assessment { border-left: 4px solid #ff69b4; }
        .commentary-decision { border-left: 4px solid #22c55e; }
        .commentary-warning { border-left: 4px solid #ef4444; }
        .commentary-opportunity { border-left: 4px solid #00ff88; }
        .commentary-technical { border-left: 4px solid #6495ed; }
        .commentary-psychology { border-left: 4px solid #9370db; }
        
        /* Dashboard styles */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #333;
        }
        .card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .card h3 {
            margin: 0 0 15px 0;
            color: #4a9eff;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
        }
        .positive { color: #22c55e; }
        .negative { color: #ef4444; }
        .neutral { color: #999; }
        .button {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            margin: 0 5px;
        }
        .button-primary { background: #4a9eff; color: white; }
        .button-danger { background: #ef4444; color: white; }
        .button-secondary { background: #333; color: white; }
        
        .filter-buttons {
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .filter-button {
            padding: 5px 15px;
            border: 1px solid #333;
            background: transparent;
            color: #999;
            border-radius: 20px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }
        .filter-button.active {
            background: #4a9eff;
            color: white;
            border-color: #4a9eff;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            color: #fff;
        }
        th, td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #333;
        }
        th {
            color: #4a9eff;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        td {
            font-size: 14px;
        }
        .toggle-container {
            display: inline-flex;
            align-items: center;
            margin-right: 20px;
            position: relative;
        }
        .toggle-slider {
            width: 50px;
            height: 24px;
            background: #333;
            border-radius: 12px;
            margin-left: 10px;
            position: relative;
            cursor: pointer;
            transition: background 0.3s;
        }
        .toggle-slider::after {
            content: '';
            position: absolute;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: transform 0.3s;
        }
        #mode-toggle {
            display: none;
        }
        #mode-toggle:checked + .toggle-slider {
            background: #ef4444;
        }
        #mode-toggle:checked + .toggle-slider::after {
            transform: translateX(26px);
        }
        #mode-label {
            font-weight: bold;
            color: #4a9eff;
            margin-right: 10px;
        }
        
        /* Position badges */
        .position-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 8px;
        }
        .badge-real {
            background: #4a9eff;
            color: white;
        }
        .badge-simulated {
            background: #666;
            color: white;
        }
        
        /* Stats row */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: #222;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .stat-label {
            font-size: 12px;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        /* Modal styles for close confirmation */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }
        .modal-content {
            background: #1a1a1a;
            border: 2px solid #4a9eff;
            border-radius: 10px;
            padding: 30px;
            max-width: 500px;
            width: 90%;
            animation: modalFadeIn 0.3s ease-out;
        }
        @keyframes modalFadeIn {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }
        .modal-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #4a9eff;
        }
        .modal-details {
            margin: 20px 0;
            line-height: 1.6;
        }
        .modal-details p {
            margin: 8px 0;
        }
        .modal-buttons {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
            margin-top: 30px;
        }
        .modal-button {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
        }
        .modal-confirm {
            background: #22c55e;
            color: white;
        }
        .modal-confirm:hover {
            background: #16a34a;
        }
        .modal-cancel {
            background: #ef4444;
            color: white;
        }
        .modal-cancel:hover {
            background: #dc2626;
        }
        .close-button {
            padding: 5px 10px;
            background: #ef4444;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
            font-weight: bold;
            transition: background 0.2s;
        }
        .close-button:hover {
            background: #dc2626;
        }
        .close-button:disabled {
            background: #666;
            cursor: not-allowed;
            opacity: 0.5;
        }
        /* Ticker tape styles */
        .ticker-tape {
            background: #0a0a0a;
            border-bottom: 1px solid #333;
            padding: 10px 20px;
            overflow-x: auto;
            overflow-y: hidden;
            position: relative;
            scrollbar-width: thin;
            scrollbar-color: #333 #0a0a0a;
        }
        
        .ticker-tape::-webkit-scrollbar {
            height: 6px;
        }
        
        .ticker-tape::-webkit-scrollbar-track {
            background: #0a0a0a;
        }
        
        .ticker-tape::-webkit-scrollbar-thumb {
            background: #333;
            border-radius: 3px;
        }
        
        .ticker-tape::-webkit-scrollbar-thumb:hover {
            background: #555;
        }

        .ticker-content {
            display: flex;
            gap: 30px;
            align-items: center;
            min-width: 100%;
        }

        .ticker-item {
            display: flex;
            align-items: center;
            font-size: 14px;
            white-space: nowrap;
            transition: background 0.3s ease;
            padding: 5px 10px;
            border-radius: 4px;
        }

        .ticker-item:hover {
            background: #1a1a1a;
        }

        .ticker-symbol {
            font-weight: bold;
            color: #4a9eff;
            margin-right: 8px;
        }

        .ticker-price {
            color: #fff;
            margin-right: 8px;
            font-family: monospace;
            min-width: 80px;
        }

        .ticker-change {
            font-weight: 600;
            min-width: 80px;
        }

        .ticker-volatility {
            color: #ffd700;
            margin-left: 8px;
            font-size: 12px;
        }

        /* Price update animation */
        @keyframes priceFlash {
            0% { background-color: transparent; }
            50% { background-color: rgba(74, 158, 255, 0.3); }
            100% { background-color: transparent; }
        }

        .price-updated {
            animation: priceFlash 0.5s ease-out;
        }
    </style>
</head>
<body>
    <!-- Add this right after <body> -->
    <div class="ticker-tape" id="ticker-tape">
        <div class="ticker-content" id="ticker-content">
            <!-- Ticker items will be inserted here -->
        </div>
    </div>
    <div class="container">
        <div class="commentary-panel">
            <h2>Live Trading Commentary</h2>
            <div class="filter-buttons">
                <button class="filter-button active" onclick="filterCommentary('all')">All</button>
                <button class="filter-button" onclick="filterCommentary('decision')">Decisions</button>
                <button class="filter-button" onclick="filterCommentary('opportunity')">Opportunities</button>
                <button class="filter-button" onclick="filterCommentary('warning')">Warnings</button>
                <button class="filter-button" onclick="filterCommentary('technical')">Technical</button>
            </div>
            <div id="commentary-list"></div>
        </div>
        
        <div class="dashboard-panel">
            <div class="header">
                <h1>Trading Bot Dashboard</h1>
                <div class="settings-row">
                    <div class="toggle-group">
                        <label class="toggle-container">
                            <span id="mode-label">Simulation Mode</span>
                            <input type="checkbox" id="mode-toggle" onchange="toggleTradingMode()">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    
                    <div class="toggle-group">
                        <label class="toggle-container">
                            <span id="confirm-label">Confirmations ON</span>
                            <input type="checkbox" id="confirm-toggle" checked onchange="toggleConfirmations()">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <div class="toggle-group">
                        <label class="toggle-container">
                            <span id="close-mode-label">Manual Close Only</span>
                            <input type="checkbox" id="close-mode-toggle" checked onchange="toggleCloseMode()">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <button class="button button-secondary" onclick="refreshPositions()">🔄 Refresh</button>
                    <button class="button button-primary" onclick="startTrading()">Start Commentary</button>
                    <button class="button button-danger" onclick="stopTrading()">Stop</button>
                </div>
            </div>
            
            <div class="card">
                <h3>Account Overview</h3>
                <div class="stats-row">
                    <div class="stat-card">
                        <div class="stat-value" id="balance">$0</div>
                        <div class="stat-label">Total Value</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="buying-power">$0</div>
                        <div class="stat-label">Buying Power</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="daily-pnl">$0</div>
                        <div class="stat-label">Day P&L</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="cash">$0</div>
                        <div class="stat-label">Cash</div>
                    </div>
                </div>
                <div class="metric" id="margin-warning" style="display: none;">
                    <span style="color: #ef4444;">⚠️ MARGIN CALL ACTIVE</span>
                </div>
            </div>
            
            <!-- Real Schwab Positions -->
            <div class="card">
                <h3>Schwab Account Positions <span class="position-badge badge-real">REAL</span></h3>
                <table id="real-positions-table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Shares</th>
                            <th>Avg Cost</th>
                            <th>Current</th>
                            <th>Market Value</th>
                            <th>Day P&L</th>
                            <th>Total P&L</th>
                            <th>% Change</th>
                            <th>Actions</th> <!-- NEW COLUMN -->
                        </tr>
                    </thead>
                    <tbody id="real-positions-body">
                        <tr><td colspan="8" style="text-align: center; color: #666;">No real positions or Schwab not connected</td></tr>
                    </tbody>
                </table>
            </div>
            
            <!-- Simulated Positions -->
            <div class="card">
                <h3>Simulated Positions <span class="position-badge badge-simulated">SIM</span></h3>
                <table id="simulated-positions-table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Qty</th>
                            <th>Entry</th>
                            <th>Current</th>
                            <th>Stop Loss</th>
                            <th>Target</th>
                            <th>P&L</th>
                            <th>Actions</th> <!-- NEW COLUMN -->
                        </tr>
                    </thead>
                    <tbody id="simulated-positions-body">
                        <tr><td colspan="7" style="text-align: center; color: #666;">No simulated positions</td></tr>
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                <h3>Recent Bot Trades</h3>
                <table id="trades-table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Entry Time</th>
                            <th>Exit Time</th>
                            <th>P&L</th>
                            <th>Reason</th>
                        </tr>
                    </thead>
                    <tbody id="trades-body">
                        <tr><td colspan="5" style="text-align: center; color: #666;">No trades yet</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let currentFilter = 'all';
        let allCommentary = [];
        
        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
            };
            
            ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'commentary') {
                addCommentary(data.data);
            } else if (data.type === 'dashboard_update') {
                updateDashboard(data.data);
            } else if (data.type === 'close_confirmation_request') {
                showCloseConfirmation(data.data);
            }
        };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected');
                setTimeout(connectWebSocket, 5000);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        function addCommentary(commentary) {
            allCommentary.unshift(commentary);
            if (allCommentary.length > 100) {
                allCommentary = allCommentary.slice(0, 100);
            }
            
            if (currentFilter === 'all' || commentary.type === currentFilter) {
                const list = document.getElementById('commentary-list');
                const item = createCommentaryItem(commentary);
                list.insertBefore(item, list.firstChild);
                
                while (list.children.length > 50) {
                    list.removeChild(list.lastChild);
                }
            }
        }
        
        function createCommentaryItem(commentary) {
            const div = document.createElement('div');
            div.className = `commentary-item commentary-${commentary.type}`;
            
            let html = `
                <div class="commentary-header">
                    <div class="commentary-title">${commentary.title}</div>
                    <div class="commentary-time">${new Date(commentary.timestamp).toLocaleTimeString()}</div>
                </div>
                <div class="commentary-message">${commentary.message}</div>
            `;
            
            if (commentary.data && Object.keys(commentary.data).length > 0) {
                html += '<div class="commentary-data">';
                for (const [key, value] of Object.entries(commentary.data)) {
                    if (typeof value === 'object' && value !== null) {
                        html += `${key}: ${JSON.stringify(value, null, 2)}<br>`;
                    } else if (typeof value === 'number') {
                        html += `${key}: ${value.toFixed ? value.toFixed(2) : value}<br>`;
                    } else {
                        html += `${key}: ${value}<br>`;
                    }
                }
                html += '</div>';
            }
            
            if (commentary.confidence) {
                html += `
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${commentary.confidence * 100}%"></div>
                    </div>
                `;
            }
            
            div.innerHTML = html;
            return div;
        }
        
        function filterCommentary(filter) {
            currentFilter = filter;
            
            document.querySelectorAll('.filter-button').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            
            const list = document.getElementById('commentary-list');
            list.innerHTML = '';
            
            const filtered = currentFilter === 'all' 
                ? allCommentary 
                : allCommentary.filter(c => c.type === currentFilter);
            
            filtered.slice(0, 50).forEach(commentary => {
                list.appendChild(createCommentaryItem(commentary));
            });
        }
        
        function updateDashboard(data) {
            if (data.account) {
                document.getElementById('balance').textContent = `$${data.account.balance.toLocaleString()}`;
                document.getElementById('buying-power').textContent = `$${data.account.buying_power.toLocaleString()}`;
                document.getElementById('cash').textContent = `$${(data.account.cash || 0).toLocaleString()}`;
                
                const dailyPnl = document.getElementById('daily-pnl');
                dailyPnl.textContent = `$${data.account.daily_pnl.toFixed(2)}`;
                dailyPnl.className = data.account.daily_pnl >= 0 ? 'stat-value positive' : 'stat-value negative';
                
                const marginWarning = document.getElementById('margin-warning');
                marginWarning.style.display = data.account.margin_call ? 'block' : 'none';
            }
            
            // Update real positions from Schwab
            if (data.real_positions) {
                const tbody = document.getElementById('real-positions-body');
                if (data.real_positions.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: #666;">No real positions or Schwab not connected</td></tr>';
                } else {
                    tbody.innerHTML = data.real_positions.map(pos => `
                        <tr>
                            <td style="font-weight: bold;">${pos.symbol}</td>
                            <td>${pos.quantity > 0 ? pos.quantity : Math.abs(pos.quantity)} ${pos.quantity < 0 ? '(short)' : ''}</td>
                            <td>$${pos.entry_price.toFixed(2)}</td>
                            <td>$${pos.current_price.toFixed(2)}</td>
                            <td>$${pos.market_value.toLocaleString()}</td>
                            <td class="${pos.day_pnl >= 0 ? 'positive' : 'negative'}">
                                $${pos.day_pnl.toFixed(2)}
                            </td>
                            <td class="${pos.unrealized_pnl >= 0 ? 'positive' : 'negative'}">
                                $${pos.unrealized_pnl.toFixed(2)}
                            </td>
                            <td class="${pos.pnl_percent >= 0 ? 'positive' : 'negative'}">
                                ${pos.pnl_percent >= 0 ? '+' : ''}${pos.pnl_percent.toFixed(2)}%
                            </td>
                            <td>
                                <button class="close-button" onclick="requestClosePosition('${pos.symbol}', 'real')">
                                    Close
                                </button>
                            </td>
                        </tr>
                    `).join('');
                }
            }
            
            // Update simulated positions
            if (data.simulated_positions) {
                const tbody = document.getElementById('simulated-positions-body');
                if (data.simulated_positions.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: #666;">No simulated positions</td></tr>';
                } else {
                    tbody.innerHTML = data.simulated_positions.map(pos => `
                        <tr>
                            <td style="font-weight: bold;">${pos.symbol}</td>
                            <td>${pos.quantity}</td>
                            <td>$${pos.entry_price.toFixed(2)}</td>
                            <td>$${pos.current_price.toFixed(2)}</td>
                            <td>$${pos.stop_loss ? pos.stop_loss.toFixed(2) : 'N/A'}</td>
                            <td>$${pos.take_profit ? pos.take_profit.toFixed(2) : 'N/A'}</td>
                            <td class="${pos.unrealized_pnl >= 0 ? 'positive' : 'negative'}">
                                $${pos.unrealized_pnl.toFixed(2)}
                            </td>
                        </tr>
                    `).join('');
                }
            }
            
            // Update simulated positions
            if (data.simulated_positions) {
                const tbody = document.getElementById('simulated-positions-body');
                if (data.simulated_positions.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: #666;">No simulated positions</td></tr>';
                } else {
                    tbody.innerHTML = data.simulated_positions.map(pos => `
                        <tr>
                            <td style="font-weight: bold;">${pos.symbol}</td>
                            <td>${pos.quantity}</td>
                            <td>$${pos.entry_price.toFixed(2)}</td>
                            <td>$${pos.current_price.toFixed(2)}</td>
                            <td>$${pos.stop_loss ? pos.stop_loss.toFixed(2) : 'N/A'}</td>
                            <td>$${pos.take_profit ? pos.take_profit.toFixed(2) : 'N/A'}</td>
                            <td class="${pos.unrealized_pnl >= 0 ? 'positive' : 'negative'}">
                                $${pos.unrealized_pnl.toFixed(2)}
                            </td>
                            <td>
                                <button class="close-button" onclick="requestClosePosition('${pos.symbol}', 'simulated')">
                                    Close
                                </button>
                            </td>
                        </tr>
                    `).join('');
                }
            }
            if (data.screener) {
                updateTickerTape(data.screener);
            }   
        }
        
        async function requestClosePosition(symbol, positionType) {
            // Disable all close buttons temporarily
            document.querySelectorAll('.close-button').forEach(btn => btn.disabled = true);
            
            try {
                const response = await fetch('/api/request-close-position', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        symbol: symbol,
                        position_type: positionType,
                        reason: 'manual_close'
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    console.log(`Close requested for ${symbol}`);
                    
                    // Show notification
                    const notification = document.createElement('div');
                    notification.style.cssText = `
                        position: fixed;
                        top: 20px;
                        right: 20px;
                        background: #4a9eff;
                        color: white;
                        padding: 15px 20px;
                        border-radius: 5px;
                        font-weight: bold;
                        z-index: 10000;
                        animation: slideIn 0.3s ease-out;
                    `;
                    notification.textContent = `Close request sent for ${symbol}`;
                    document.body.appendChild(notification);
                    
                    setTimeout(() => {
                        notification.remove();
                    }, 3000);
                } else {
                    const error = await response.json();
                    alert(`Failed to close position: ${error.message}`);
                }
            } finally {
                // Re-enable buttons after a short delay
                setTimeout(() => {
                    document.querySelectorAll('.close-button').forEach(btn => btn.disabled = false);
                }, 1000);
            }
        }
        async function startTrading() {
            const response = await fetch('/api/start', { method: 'POST' });
            if (response.ok) {
                console.log('Trading started');
            }
        }
        
        async function stopTrading() {
            const response = await fetch('/api/stop', { method: 'POST' });
            if (response.ok) {
                console.log('Trading stopped');
            }
        }
        
        async function toggleTradingMode() {
            const toggle = document.getElementById('mode-toggle');
            const label = document.getElementById('mode-label');
            const isLive = toggle.checked;
            
            label.textContent = isLive ? 'LIVE TRADING' : 'Simulation Mode';
            label.style.color = isLive ? '#ef4444' : '#4a9eff';
            
            if (isLive) {
                const confirmed = confirm('⚠️ WARNING: Switching to LIVE TRADING mode. Real money will be at risk. Are you sure?');
                if (!confirmed) {
                    toggle.checked = false;
                    label.textContent = 'Simulation Mode';
                    label.style.color = '#4a9eff';
                    return;
                }
            }
            
            const response = await fetch('/api/toggle-mode', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode: isLive ? 'live' : 'simulation'})
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log('Mode changed to:', data.mode);
            }
        }
        
        async function refreshPositions() {
            const response = await fetch('/api/refresh-positions', { method: 'POST' });
            if (response.ok) {
                const data = await response.json();
                console.log(`Refreshed ${data.count} positions`);
            }
        }
        async function toggleCloseMode() {
            const toggle = document.getElementById('close-mode-toggle');
            const label = document.getElementById('close-mode-label');
            const isManualOnly = toggle.checked;
            
            label.textContent = isManualOnly ? 'Manual Close Only' : 'Auto Close Enabled';
            
            const response = await fetch('/api/toggle-close-mode', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({manual_only: isManualOnly})
            });
            
            if (response.ok) {
                console.log('Close mode:', isManualOnly ? 'manual' : 'auto');
            }
        }
        // Close confirmation functions
        function showCloseConfirmation(request) {
            const modalHTML = `
                <div class="modal-overlay" id="close-confirmation-modal">
                    <div class="modal-content">
                        <div class="modal-title">⚠️ Confirm Position Close</div>
                        <div class="modal-details">
                            <p><strong>Symbol:</strong> ${request.symbol}</p>
                            <p><strong>Quantity:</strong> ${request.quantity} shares</p>
                            <p><strong>Entry Price:</strong> $${request.entry_price.toFixed(2)}</p>
                            <p><strong>Current Price:</strong> $${request.current_price.toFixed(2)}</p>
                            <p><strong>P&L:</strong> <span class="${request.pnl >= 0 ? 'positive' : 'negative'}">$${request.pnl.toFixed(2)} (${request.roi.toFixed(1)}%)</span></p>
                            <p><strong>Reason:</strong> ${request.reason.replace(/_/g, ' ')}</p>
                            <hr style="margin: 20px 0; border-color: #333;">
                            <p style="font-size: 16px;">${request.message}</p>
                        </div>
                        <div class="modal-buttons">
                            <button class="modal-button modal-cancel" onclick="confirmClose('${request.request_id}', false)">
                                Cancel (Keep Position)
                            </button>
                            <button class="modal-button modal-confirm" onclick="confirmClose('${request.request_id}', true)">
                                Confirm Close
                            </button>
                        </div>
                    </div>
                </div>
            `;
            
            document.body.insertAdjacentHTML('beforeend', modalHTML);
            
            // Auto-cancel after 30 seconds
            setTimeout(() => {
                const modal = document.getElementById('close-confirmation-modal');
                if (modal) {
                    confirmClose(request.request_id, false);
                }
            }, 30000);
        }

        async function confirmClose(requestId, confirmed) {
            // Remove modal
            const modal = document.getElementById('close-confirmation-modal');
            if (modal) {
                modal.remove();
            }
            
            // Send confirmation to backend
            const response = await fetch('/api/confirm-close', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    request_id: requestId,
                    confirmed: confirmed
                })
            });
            
            if (response.ok) {
                console.log(`Close ${confirmed ? 'confirmed' : 'cancelled'}`);
            }
        }
        async function toggleConfirmations() {
            const toggle = document.getElementById('confirm-toggle');
            const label = document.getElementById('confirm-label');
            const isEnabled = toggle.checked;
            
            label.textContent = isEnabled ? 'Confirmations ON' : 'Confirmations OFF';
            
            const response = await fetch('/api/toggle-confirmations', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enabled: isEnabled})
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log('Confirmations:', data.enabled ? 'enabled' : 'disabled');
                
                // Show a temporary notification
                const notification = document.createElement('div');
                notification.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: ${isEnabled ? '#22c55e' : '#ef4444'};
                    color: white;
                    padding: 15px 20px;
                    border-radius: 5px;
                    font-weight: bold;
                    z-index: 10000;
                    animation: slideIn 0.3s ease-out;
                `;
                notification.textContent = `Close confirmations ${isEnabled ? 'enabled' : 'disabled'}`;
                document.body.appendChild(notification);
                
                setTimeout(() => {
                    notification.remove();
                }, 3000);
            }
        }
        // Add this function to update the ticker tape
        function updateTickerTape(screenerData) {
            if (!screenerData || screenerData.length === 0) return;
            
            const tickerContent = document.getElementById('ticker-content');
            
            // Create ticker HTML
            let tickerHTML = '';
            
            // Duplicate the list for seamless scrolling
            //const items = [...screenerData, ...screenerData];
            
            screenerData.forEach(stock => {
                const changeClass = stock.change >= 0 ? 'positive' : 'negative';
                const changeSymbol = stock.change >= 0 ? '+' : '';
                
                tickerHTML += `
                    <div class="ticker-item">
                        <span class="ticker-symbol">${stock.symbol}</span>
                        <span class="ticker-price">$${stock.last.toFixed(2)}</span>
                        <span class="ticker-change ${changeClass}">${changeSymbol}${stock.change.toFixed(2)}%</span>
                        <span class="ticker-volatility">⚡${stock.volatility.toFixed(1)}%</span>
                    </div>
                `;
            });
            
            tickerContent.innerHTML = tickerHTML;
        }

        // Connect on load
        connectWebSocket();
    </script>
</body>
</html>
"""
# ============================================================================
# FASTAPI APPLICATION WITH COMMENTARY
# ============================================================================

app = FastAPI(title="Trading Bot with Commentary API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
trading_engine = None
connection_manager = ConnectionManager()

@app.post("/api/toggle-mode")
async def toggle_trading_mode(request: dict):
    global trading_engine
    
    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}
    
    new_mode = request.get('mode', 'simulation')
    
    if new_mode == 'live':
        trading_engine.mode = TradingMode.LIVE
        # Add safety check
        if not trading_engine.schwab_client:
            return {"status": "error", "message": "Cannot switch to live mode - Schwab not connected"}
    else:
        trading_engine.mode = TradingMode.SIMULATION_WITH_COMMENTARY
    
    trading_engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.DECISION,
        symbol=None,
        title=f"🔄 Mode Changed",
        message=f"Switched to {'LIVE TRADING' if new_mode == 'live' else 'SIMULATION'} mode",
        importance=10
    ))
    
    return {"status": "success", "mode": trading_engine.mode.value}

@app.post("/api/toggle-confirmations")
async def toggle_confirmations(request: dict):
    """Toggle close confirmation requirement"""
    global trading_engine
    
    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}
    
    enabled = request.get('enabled', True)
    trading_engine.require_confirmations = enabled
    
    # Add commentary about the change
    trading_engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.DECISION,
        symbol=None,
        title=f"⚙️ Confirmation Settings Changed",
        message=f"Close confirmations {'enabled' if enabled else 'disabled'}",
        importance=7
    ))
    
    return {"status": "success", "enabled": enabled}

@app.get("/")
async def get_dashboard():
    return HTMLResponse(content=DASHBOARD_HTML_WITH_COMMENTARY)

@app.post("/api/start")
async def start_trading():
    global trading_engine, connection_manager
    
    if not trading_engine:
        trading_engine = TradingEngineWithCommentary(connection_manager=connection_manager)
        
        # Subscribe to commentary updates
        async def broadcast_commentary(commentary: TradingCommentary):
            await connection_manager.broadcast({
                'type': 'commentary',
                'data': commentary.to_dict()
            })
        
        trading_engine.commentary.subscribe(
            lambda c: asyncio.create_task(broadcast_commentary(c))
        )
    
    if not trading_engine.is_running:
        threading.Thread(
            target=lambda: asyncio.run(trading_engine.start())
        ).start()
        
        return {"status": "success", "message": "Trading with commentary started"}
    
    return {"status": "info", "message": "Already running"}

@app.post("/api/stop")
async def stop_trading():
    if trading_engine:
        trading_engine.is_running = False
        return {"status": "success", "message": "Trading stopped"}
    return {"status": "error", "message": "Not running"}

@app.post("/api/refresh-positions")
async def refresh_positions():
    """Manually refresh positions from Schwab"""
    if not trading_engine or not trading_engine.schwab_client:
        return {"status": "error", "message": "Schwab not connected"}
    
    positions = await trading_engine.get_schwab_positions()
    
    return {
        "status": "success",
        "positions": positions,
        "count": len(positions)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    
    try:
        while True:
            await asyncio.sleep(1)
            
            if trading_engine:
                # Get simulated positions
                sim_positions_data = []
                if trading_engine.mode == TradingMode.SIMULATION_WITH_COMMENTARY:
                    for symbol, pos in trading_engine.simulated_positions.items():
                        if pos is not None:
                            sim_positions_data.append({
                                'symbol': pos.symbol,
                                'quantity': pos.quantity,
                                'entry_price': pos.entry_price,
                                'current_price': pos.current_price,
                                'unrealized_pnl': pos.unrealized_pnl,
                                'type': 'simulated'
                            })
                
                # Get real Schwab positions
                real_positions_data = []
                if trading_engine.schwab_client:
                    schwab_positions = await trading_engine.get_schwab_positions()
                    for pos in schwab_positions:
                        real_positions_data.append({
                            'symbol': pos['symbol'],
                            'quantity': pos['quantity'],
                            'entry_price': pos['average_price'],
                            'current_price': pos['current_price'],
                            'unrealized_pnl': pos['total_pnl'],
                            'day_pnl': pos['day_pnl'],
                            'pnl_percent': pos['pnl_percent'],
                            'market_value': pos['market_value'],
                            'type': 'real'
                        })
                
                # Get account info
                account_info = await trading_engine._get_real_account_info()
                # Get screener data
                screener_data = []
                if trading_engine.screener and trading_engine.screener.top_movers:
                    screener_data = [{
                        'symbol': m['symbol'],
                        'last': m.get('last', 0),
                        'change': m.get('percent_change', 0),
                        'volume': m.get('volume', 0),
                        'volatility': m.get('volatility', 0),
                        'high': m.get('high', 0),
                        'low': m.get('low', 0)
                    } for m in trading_engine.screener.top_movers[:10]]
                
                await websocket.send_json({
                    'type': 'dashboard_update',
                    'data': {
                        'account': {
                            'balance': account_info.get('balance', trading_engine.risk_manager.account_balance),
                            'buying_power': account_info.get('buying_power', trading_engine.risk_manager.buying_power),
                            'daily_pnl': trading_engine.risk_manager.daily_pnl,
                            'margin_call': trading_engine.risk_manager.margin_call,
                            'cash': account_info.get('cash', 0)
                        },
                        'simulated_positions': sim_positions_data,
                        'real_positions': real_positions_data,
                        'trades': trading_engine.trade_history[-5:] if trading_engine.trade_history else [],
                        'screener': screener_data
                    }
                })
                
                
            
    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket)

@app.post("/api/confirm-close")
async def confirm_close(request: dict):
    """Handle user confirmation for closing positions"""
    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}
    
    request_id = request.get('request_id')
    confirmed = request.get('confirmed', False)
    
    if hasattr(trading_engine, 'pending_close_requests'):
        if request_id in trading_engine.pending_close_requests:
            trading_engine.pending_close_requests[request_id]['confirmed'] = confirmed
            return {"status": "success", "confirmed": confirmed}
    
    return {"status": "error", "message": "Invalid or expired request"}

@app.post("/api/request-close-position")
async def request_close_position(request: dict):
    """Handle manual position close request"""
    global trading_engine
    
    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}
    
    symbol = request.get('symbol')
    position_type = request.get('position_type', 'real')
    
    # Add commentary about manual close request
    trading_engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.DECISION,
        symbol=symbol,
        title=f"🔧 Manual Close Requested",
        message=f"User requested to close {symbol} position",
        importance=8
    ))
    
    # Trigger the close
    await trading_engine.close_position_manually(symbol, position_type)

    return {"status": "success", "message": f"Close process initiated for {symbol}"}

@app.post("/api/toggle-close-mode")
async def toggle_close_mode(request: dict):
    """Toggle between manual and automatic position closing"""
    global trading_engine
    
    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}
    
    manual_only = request.get('manual_only', True)
    trading_engine.manual_close_only = manual_only
    
    trading_engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.DECISION,
        symbol=None,
        title=f"⚙️ Close Mode Changed",
        message=f"Position closing: {'Manual only' if manual_only else 'Automatic (stop loss, targets)'}",
        importance=7
    ))
    
    return {"status": "success", "manual_only": manual_only}

@app.post("/api/reset-brain")
async def reset_brain():
    if trading_engine:
        trading_engine.reset_brain_state()
        return {"status": "success", "message": "Brain state reset"}
    return {"status": "error", "message": "Engine not running"}

@app.post("/api/run-backtest")
async def run_backtest(request: dict):
    """Run backtest on historical data using Schwab API"""
    try:
        start_date = request.get('start_date', '2023-01-01')
        end_date = request.get('end_date', '2024-01-01')
        symbols = request.get('symbols', ['AAPL', 'MSFT', 'GOOGL'])
        strategy_name = request.get('strategy', 'breakout')
        
        # Check if we have Schwab connection
        if not trading_engine or not trading_engine.schwab_client:
            return {"status": "error", "message": "Schwab not connected. Please authenticate first."}

        # Initialize backtest engine
        backtest_engine = FixedBacktestEngine(config_manager.config)
        
        # Get historical data from Schwab
        data = {}
        data_provider = trading_engine.data_provider

        for symbol in symbols:
            try:
                # Use Schwab to get daily data for backtesting
                df = data_provider.get_market_data(
                    symbol,
                    period_type='year',
                    period=2,  # 2 years of data
                    frequency_type='minute',
                    frequency=1
                )

                if not df.empty and len(df) > 50:
                    # Filter by date range
                    df = df[(df.index >= pd.to_datetime(start_date)) &
                           (df.index <= pd.to_datetime(end_date))]

                    if len(df) > 50:  # Still have enough data after filtering
                        data[symbol] = df
                        logger.info(f"Downloaded {len(df)} bars for {symbol} from Schwab")

            except Exception as e:
                logger.error(f"Error downloading data for {symbol}: {e}")
        
        if not data:
            return {"status": "error", "message": "No data available for backtesting"}
        
        # Create backtest strategy
        if strategy_name == 'breakout':
            strategy = BacktestBreakoutStrategy()
        elif strategy_name == 'mean_reversion':
            strategy = BacktestMeanReversionStrategy()
        else:
            strategy = BacktestBreakoutStrategy()
        
        # Run backtest
        result = backtest_engine.run_backtest(data, strategy, start_date, end_date)
        
        # Format results (rest of the code remains the same)
        response_data = {
            "status": "success",
            "message": "Backtest completed successfully",
            "results": {
                'total_return': f"{result.total_return:.2%}",
                'annualized_return': f"{result.annualized_return:.2%}",
                'sharpe_ratio': f"{result.sharpe_ratio:.2f}",
                'sortino_ratio': f"{result.sortino_ratio:.2f}",
                'max_drawdown': f"{result.max_drawdown:.2%}",
                'win_rate': f"{result.win_rate:.2%}",
                'profit_factor': f"{result.profit_factor:.2f}" if result.profit_factor != float('inf') else "N/A",
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'average_win': f"${result.average_win:.2f}",
                'average_loss': f"${result.average_loss:.2f}",
                'largest_win': f"${result.largest_win:.2f}",
                'largest_loss': f"${result.largest_loss:.2f}",
                'consecutive_wins': result.consecutive_wins,
                'consecutive_losses': result.consecutive_losses,
                'initial_capital': f"${result.initial_capital:,.2f}",
                'final_capital': f"${result.final_capital:,.2f}"
            },
            'equity_curve': result.equity_curve[-100:],
            'trade_count_by_symbol': {}
        }

        # Count trades by symbol
        for trade in result.trade_history:
            symbol = trade.get('symbol', 'UNKNOWN')
            if symbol not in response_data['trade_count_by_symbol']:
                response_data['trade_count_by_symbol'][symbol] = 0
            response_data['trade_count_by_symbol'][symbol] += 1

        # Save results
        save_data = {
            'config': request,
            'results': response_data['results'],
            'equity_curve': result.equity_curve,
            'trade_history': [
                {
                    'date': trade['date'].isoformat() if isinstance(trade['date'], pd.Timestamp) else str(trade['date']),
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'price': float(trade['price']),
                    'quantity': int(trade['quantity']),
                    'pnl': float(trade.get('pnl', 0)),
                    'capital': float(trade['capital'])
                }
                for trade in result.trade_history
            ]
        }

        with open(config_manager.get('paths.backtest_results'), 'w') as f:
            json.dump(save_data, f, indent=2)

        return response_data

    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        return {"status": "error", "message": f"Backtest failed: {str(e)}"}

@app.get("/api/performance-metrics")
async def get_performance_metrics():
    """Get comprehensive performance metrics"""
    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}
    
    try:
        metrics = trading_engine.performance_analyzer.calculate_metrics(
            list(trading_engine.positions.values()),
            trading_engine.trade_history
        )
        
        return {
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {"status": "error", "message": f"Error calculating metrics: {str(e)}"}

@app.post("/api/update-config")
async def update_config(request: dict):
    """Update configuration values"""
    try:
        key = request.get('key')
        value = request.get('value')
        
        if not key or value is None:
            return {"status": "error", "message": "Key and value are required"}
        
        config_manager.update(key, value)
        
        return {
            "status": "success",
            "message": f"Configuration updated: {key} = {value}"
        }
    except Exception as e:
        logger.error(f"Config update error: {e}")
        return {"status": "error", "message": f"Config update failed: {str(e)}"}

@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    try:
        return {
            "status": "success",
            "config": config_manager.config
        }
    except Exception as e:
        logger.error(f"Config retrieval error: {e}")
        return {"status": "error", "message": f"Config retrieval failed: {str(e)}"}


@app.get("/api/advanced-analytics")
async def get_advanced_analytics():
    """Get advanced analytics and insights"""
    try:
        if not trading_engine:
            return {"status": "error", "message": "Trading engine not initialized"}
        
        # Get performance report
        performance_report = trading_engine.performance_analyzer.calculate_metrics(
            list(trading_engine.positions.values()),
            trading_engine.trade_history
        )
        
        # Get behavioral analysis
        behavioral_patterns = trading_engine.behavioral_analyzer.analyze_trading_patterns(
            trading_engine.trade_history
        )
        
        # Get behavioral recommendations
        behavioral_recommendations = trading_engine.behavioral_analyzer.get_behavioral_recommendations()
        
        # Get portfolio optimization data
        positions = list(trading_engine.positions.values())
        historical_data = {}  # This would be populated with actual data
        
        # NOTE: Portfolio optimization is not fully implemented in this version
        # portfolio_weights = trading_engine.portfolio_optimizer.optimize_weights(
        #     positions, historical_data
        # )
        
        return {
            "status": "success",
            "performance_metrics": performance_report,
            "behavioral_patterns": behavioral_patterns,
            "behavioral_recommendations": behavioral_recommendations,
            # "portfolio_optimization": {
            #     "current_weights": portfolio_weights,
            #     "recommendations": performance_report.get("recommendations", [])
            # },
            "emotional_state": trading_engine.behavioral_analyzer.emotional_state
        }
    except Exception as e:
        logger.error(f"Error getting advanced analytics: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/alternative-data/{symbol}")
async def get_alternative_data(symbol: str):
    """Get alternative data for a specific symbol"""
    try:
        if not trading_engine:
            return {"status": "error", "message": "Trading engine not initialized"}
        
        # Get options flow
        options_flow = await trading_engine.alternative_data_integrator.get_options_flow(symbol)
        
        # Get insider trading
        insider_trading = await trading_engine.alternative_data_integrator.get_insider_trading(symbol)
        
        # Get social sentiment
        social_sentiment = await trading_engine.alternative_data_integrator.get_social_sentiment(symbol)
        
        # Get economic calendar
        economic_calendar = await trading_engine.alternative_data_integrator.get_economic_calendar()
        
        # Analyze alternative signals
        alternative_signals = trading_engine.alternative_data_integrator.analyze_alternative_signals(symbol)
        
        return {
            "status": "success",
            "symbol": symbol,
            "options_flow": options_flow,
            "insider_trading": insider_trading,
            "social_sentiment": social_sentiment,
            "economic_calendar": economic_calendar,
            "alternative_signals": alternative_signals
        }
    except Exception as e:
        logger.error(f"Error getting alternative data for {symbol}: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/market-neutral-opportunities")
async def get_market_neutral_opportunities():
    """Get market neutral trading opportunities"""
    try:
        if not trading_engine:
            return {"status": "error", "message": "Trading engine not initialized"}
        
        # Get current symbols
        symbols = list(trading_engine.simulated_positions.keys()) + list(trading_engine.real_positions.keys())
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Default symbols
        
        # Get historical data (this would be populated with actual data)
        historical_data = {}
        
        # Find pairs trading opportunities
        pairs_opportunities = trading_engine.market_neutral_strategies.find_pairs_trading_opportunities(
            symbols, historical_data
        )
        
        # Calculate sector weights
        sector_weights = trading_engine.market_neutral_strategies.calculate_sector_weights(symbols)
        
        # Get statistical arbitrage signals
        stat_arb_signals = {}
        for symbol in symbols[:5]:  # Limit to first 5 symbols
            if symbol in historical_data:
                stat_arb_signals[symbol] = trading_engine.market_neutral_strategies.calculate_statistical_arbitrage_signals(
                    symbol, historical_data[symbol]
                )
        
        return {
            "status": "success",
            "pairs_trading_opportunities": pairs_opportunities,
            "sector_weights": sector_weights,
            "statistical_arbitrage_signals": stat_arb_signals
        }
    except Exception as e:
        logger.error(f"Error getting market neutral opportunities: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/data-quality/{symbol}")
async def get_data_quality(symbol: str):
    """Get data quality analysis for a symbol"""
    try:
        if not trading_engine:
            return {"status": "error", "message": "Trading engine not initialized"}
        
        # Get market data
        if trading_engine.data_provider:
            data = trading_engine.data_provider.get_market_data(symbol)
            
            # Validate data quality
            quality_report = trading_engine.data_validator.validate_market_data(symbol, data)
            
            return {
                "status": "success",
                "symbol": symbol,
                "data_quality": quality_report,
                "historical_quality": trading_engine.data_validator.data_quality_history.get(symbol, {})
            }
        else:
            return {"status": "error", "message": "Data provider not available"}
    except Exception as e:
        logger.error(f"Error getting data quality for {symbol}: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/optimize-portfolio")
async def optimize_portfolio():
    """Trigger portfolio optimization"""
    try:
        if not trading_engine:
            return {"status": "error", "message": "Trading engine not initialized"}
        
        positions = list(trading_engine.positions.values())
        historical_data = {}  # This would be populated with actual data
        
        # NOTE: Portfolio optimization not fully implemented
        # optimized_weights = trading_engine.portfolio_optimizer.optimize_weights(
        #     positions, historical_data
        # )
        
        return {
            "status": "success",
            # "optimized_weights": optimized_weights,
            "message": "Portfolio optimization completed"
        }
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/record-execution")
async def record_execution(request: dict):
    """Record trade execution for performance monitoring"""
    try:
        if not trading_engine:
            return {"status": "error", "message": "Trading engine not initialized"}
        
        # NOTE: performance_monitor not fully implemented
        # trading_engine.performance_monitor.record_execution(
        #     symbol=request["symbol"],
        #     expected_price=request["expected_price"],
        #     actual_price=request["actual_price"],
        #     signal_time=datetime.fromisoformat(request["signal_time"]),
        #     execution_time=datetime.fromisoformat(request["execution_time"]),
        #     order_size=request["order_size"]
        # )
        
        return {"status": "success", "message": "Execution recorded successfully"}
    except Exception as e:
        logger.error(f"Error recording execution: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║        Trading Bot with Live Commentary - Educational Mode           ║
    ║     Learn How Professional Trading Algorithms Make Decisions         ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nStarting Trading Bot in Commentary Mode...")
    print("The bot will explain its thinking process in real-time.")
    print("\n📊 Features:")
    print("  • Real-time market analysis with explanations")
    print("  • Detailed reasoning for every trading decision")
    print("  • Risk management explanations")
    print("  • Post-trade analysis and lessons")
    print("  • Works even during margin calls (simulation only)")
    
    print("\n🌐 Open http://localhost:8000 in your browser")
    print("📱 Click 'Start Commentary' to begin")
    print("❌ Press Ctrl+C to stop\n")
    
    import signal
    
    def shutdown_handler(signum, frame):
        print("\n\n🛑 Shutting down gracefully...")
        if trading_engine:
            trading_engine._save_state()
            trading_engine.brain.save_memories()
            print("✅ State saved successfully")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown_handler)

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
