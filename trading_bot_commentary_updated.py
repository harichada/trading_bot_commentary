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
# WEBSOCKET CONNECTION MANAGER
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections"""
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        if not self.active_connections:
            return
            
        message_str = json.dumps(message)
        disconnected = set()
        
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_text(message_str)
                except Exception as e:
                    logger.debug(f"Failed to send to connection: {e}")
                    disconnected.add(connection)
        
        if disconnected:
            async with self._lock:
                self.active_connections -= disconnected

# ============================================================================
# COMMENTARY SYSTEM
# ============================================================================

@dataclass
class TradingCommentary:
    """Represents a single commentary entry"""
    timestamp: datetime
    type: CommentaryType
    symbol: Optional[str]
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    importance: int = 5
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'type': self.type.value,
            'symbol': self.symbol,
            'title': self.title,
            'message': self.message,
            'data': self.data,
            'confidence': self.confidence,
            'importance': self.importance
        }

class CommentarySystem:
    """Manages trading commentary and explanations"""
    
    def __init__(self, max_history: int = 100):
        self.history: deque = deque(maxlen=max_history)
        self.subscribers: Set[callable] = set()
        self.console = console if RICH_AVAILABLE else None
        
    def add_commentary(self, commentary: TradingCommentary):
        """Add new commentary and notify subscribers"""
        self.history.append(commentary)
        self._display_commentary(commentary)
        self._notify_subscribers(commentary)
        self._save_to_file(commentary)
    
    def _display_commentary(self, commentary: TradingCommentary):
        """Display commentary in terminal with rich formatting"""
        # Handle case where commentary might be a dict
        if isinstance(commentary, dict):
            # Convert dict back to TradingCommentary object
            try:
                commentary_type = CommentaryType(commentary.get('type', 'MARKET_ANALYSIS'))
            except ValueError:
                # If the type is not valid, default to MARKET_ANALYSIS
                commentary_type = CommentaryType.MARKET_ANALYSIS
            
            commentary = TradingCommentary(
                timestamp=datetime.fromisoformat(commentary['timestamp']) if isinstance(commentary.get('timestamp'), str) else commentary.get('timestamp', datetime.now()),
                type=commentary_type,
                symbol=commentary.get('symbol'),
                title=commentary.get('title', ''),
                message=commentary.get('message', ''),
                data=commentary.get('data', {}),
                confidence=commentary.get('confidence'),
                importance=commentary.get('importance', 5)
            )
        
        if not self.console:
            print(f"\n[{commentary.timestamp.strftime('%H:%M:%S')}] {commentary.title}")
            print(f"  {commentary.message}")
            return
        
        color_map = {
            CommentaryType.MARKET_ANALYSIS: "cyan",
            CommentaryType.SIGNAL_GENERATION: "yellow",
            CommentaryType.RISK_ASSESSMENT: "magenta",
            CommentaryType.DECISION: "green",
            CommentaryType.WARNING: "red",
            CommentaryType.OPPORTUNITY: "bright_green",
            CommentaryType.TECHNICAL: "blue",
            CommentaryType.PSYCHOLOGY: "purple",
            CommentaryType.ANOMALY: "bright_red"
        }
        
        color = color_map.get(commentary.type, "white")
        
        title = f"[bold {color}]{commentary.title}[/bold {color}]"
        
        content = Text()
        content.append(commentary.message, style=f"{color}")
        
        if commentary.data:
            content.append("\n\n📊 Details:\n", style="bold")
            for key, value in commentary.data.items():
                if isinstance(value, (int, float)):
                    content.append(f"  • {key}: ", style="dim")
                    content.append(f"{value:.4f}" if isinstance(value, float) else str(value), style="bold")
                    content.append("\n")
                else:
                    content.append(f"  • {key}: {value}\n", style="dim")
        
        if commentary.confidence:
            conf_color = "green" if commentary.confidence > 0.7 else "yellow" if commentary.confidence > 0.4 else "red"
            content.append(f"\n💡 Confidence: ", style="dim")
            content.append(f"{commentary.confidence:.1%}", style=f"bold {conf_color}")
        
        panel = Panel(
            content,
            title=title,
            subtitle=f"[dim]{commentary.timestamp.strftime('%H:%M:%S')}[/dim]",
            border_style=color,
            expand=False
        )
        
        self.console.print(panel)
    
    def _notify_subscribers(self, commentary: TradingCommentary):
        """Notify all subscribers of new commentary"""
        for subscriber in self.subscribers:
            try:
                subscriber(commentary)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def _save_to_file(self, commentary: TradingCommentary):
        """Save commentary to file for later analysis"""
        try:
            commentary_path = str(Config().COMMENTARY_LOG_PATH)
            if Path(commentary_path).exists():
                with open(commentary_path, 'r') as f:
                    all_commentary = json.load(f)
            else:
                all_commentary = []
            
            all_commentary.append(commentary.to_dict())
            
            if len(all_commentary) > 1000:
                all_commentary = all_commentary[-1000:]
            
            with open(commentary_path, 'w') as f:
                json.dump(all_commentary, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving commentary: {e}")
    
    def subscribe(self, callback: callable):
        """Subscribe to commentary updates"""
        self.subscribers.add(callback)

# ============================================================================
# TRADING MEMORY AND LEARNING SYSTEM
# ============================================================================

@dataclass
class TradingMemory:
    """Represents a trading memory/lesson learned"""
    timestamp: datetime
    symbol: str
    pattern: str
    outcome: str
    lesson: str
    confidence_adjustment: float
    context: Dict[str, Any]

class TradingBrain:
    """Persistent memory and learning system - acts like a human trader's experience"""
    
    def __init__(self, memory_path: Path = Path("trading_brain.json")):
        self.memory_path = memory_path
        self.memories: List[TradingMemory] = []
        self.pattern_success_rates: Dict[str, Dict[str, float]] = {}
        self.symbol_behaviors: Dict[str, Dict[str, Any]] = {}
        self.emotional_state = {
            'confidence': 0.7,
            'risk_appetite': 0.5,
            'patience': 0.8,
            'fear_level': 0.2,
            'greed_level': 0.3,
            'last_update': datetime.now()
        }
        self._load_memories()
    
    def recover_confidence(self):
        """Gradually recover confidence over time"""
        # Handle both string and datetime objects
        if isinstance(self.emotional_state['last_update'], str):
            last_update = datetime.fromisoformat(self.emotional_state['last_update'])
        else:
            last_update = self.emotional_state['last_update']
        
        time_since_update = (datetime.now() - last_update).seconds
        
        # Recover 1% confidence per hour of no trading
        if time_since_update > 3600:
            recovery = (time_since_update / 3600) * 0.01
            self.emotional_state['confidence'] = min(0.7, self.emotional_state['confidence'] + recovery)
            self.emotional_state['fear_level'] = max(0.3, self.emotional_state['fear_level'] - recovery)
            self.emotional_state['last_update'] = datetime.now().isoformat()
    
    def _load_memories(self):
        """Load memories from disk"""
        if self.memory_path.exists():
            try:
                with open(self.memory_path, 'r') as f:
                    data = json.load(f)
                    self.pattern_success_rates = data.get('pattern_success_rates', {})
                    self.symbol_behaviors = data.get('symbol_behaviors', {})
                    self.emotional_state = data.get('emotional_state', self.emotional_state)
                    
                    # Reconstruct memories
                    for mem_data in data.get('memories', []):
                        self.memories.append(TradingMemory(
                            timestamp=datetime.fromisoformat(mem_data['timestamp']),
                            symbol=mem_data['symbol'],
                            pattern=mem_data['pattern'],
                            outcome=mem_data['outcome'],
                            lesson=mem_data['lesson'],
                            confidence_adjustment=mem_data['confidence_adjustment'],
                            context=mem_data['context']
                        ))
                logger.info(f"Loaded {len(self.memories)} trading memories")
            except Exception as e:
                logger.error(f"Error loading memories: {e}")
    
    def save_memories(self):
        """Save memories to disk"""
        self.emotional_state['last_update'] = self.emotional_state['last_update'].isoformat() if isinstance(self.emotional_state['last_update'], datetime) else self.emotional_state['last_update']
        data = {
            'memories': [
                {
                    'timestamp': mem.timestamp.isoformat(),
                    'symbol': mem.symbol,
                    'pattern': mem.pattern,
                    'outcome': mem.outcome,
                    'lesson': mem.lesson,
                    'confidence_adjustment': mem.confidence_adjustment,
                    'context': mem.context
                }
                for mem in self.memories[-1000:]  # Keep last 1000 memories
            ],
            'pattern_success_rates': self.pattern_success_rates,
            'symbol_behaviors': self.symbol_behaviors,
            'emotional_state': self.emotional_state
        }
        
        with open(self.memory_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def remember_trade(self, symbol: str, pattern: str, outcome: str, 
                      pnl_percent: float, context: Dict[str, Any]):
        """Remember a trade outcome and learn from it"""
        # Update pattern success rates
        if pattern not in self.pattern_success_rates:
            self.pattern_success_rates[pattern] = {'wins': 0, 'losses': 0, 'total': 0}
        
        self.pattern_success_rates[pattern]['total'] += 1
        if pnl_percent > 0:
            self.pattern_success_rates[pattern]['wins'] += 1
        else:
            self.pattern_success_rates[pattern]['losses'] += 1
        
        # Update symbol-specific behaviors
        if symbol not in self.symbol_behaviors:
            self.symbol_behaviors[symbol] = {
                'avg_holding_time': 0,
                'best_exit_timing': {},
                'volatility_profile': {},
                'success_rate': 0.5
            }
        
        # Generate human-like lesson
        lesson = self._generate_lesson(symbol, pattern, outcome, pnl_percent, context)
        
        # Adjust emotional state based on outcome
        self._adjust_emotional_state(pnl_percent)
        
        # Create memory
        memory = TradingMemory(
            timestamp=datetime.now(),
            symbol=symbol,
            pattern=pattern,
            outcome=outcome,
            lesson=lesson,
            confidence_adjustment=self._calculate_confidence_adjustment(pnl_percent),
            context=context
        )
        
        self.memories.append(memory)
        self.save_memories()
        
        return memory
    
    def _generate_lesson(self, symbol: str, pattern: str, outcome: str, 
                        pnl_percent: float, context: Dict[str, Any]) -> str:
        """Generate human-like lessons from trades"""
        if pnl_percent > 2:
            lessons = [
                f"Nice! {symbol} responds well to {pattern} patterns. Should have held longer though...",
                f"Good call on {symbol}. Next time, use a trailing stop to capture more upside.",
                f"{symbol} momentum was stronger than expected. Mental note: be more patient with winners."
            ]
        elif pnl_percent > 0:
            lessons = [
                f"Small win on {symbol}, but a win is a win. Maybe tighten the entry next time.",
                f"Profitable, but left money on the table. {symbol} had more room to run.",
                f"Decent trade. Should trust my analysis more when the setup is this clean."
            ]
        elif pnl_percent > -1:
            lessons = [
                f"Small loss on {symbol}. The setup was right but timing was off.",
                f"Stopped out for a small loss. Good risk management, bad entry timing.",
                f"{symbol} didn't follow through. Need to watch for false breakouts here."
            ]
        else:
            lessons = [
                f"Ouch. {symbol} completely reversed. Need to respect the trend more.",
                f"Bad read on {symbol}. Market conditions changed - should've adapted.",
                f"Lesson learned: don't fight the tape on {symbol} when volume is weak."
            ]
        
        return np.random.choice(lessons)
    
    def _adjust_emotional_state(self, pnl_percent: float):
        """Adjust emotional state based on trade outcomes - like a real trader"""
        # Confidence adjustment
        if pnl_percent > 0:
            self.emotional_state['confidence'] = min(0.95, self.emotional_state['confidence'] + 0.02)
            self.emotional_state['greed_level'] = min(0.8, self.emotional_state['greed_level'] + 0.05)
        else:
            self.emotional_state['confidence'] = max(0.3, self.emotional_state['confidence'] - 0.03)
            self.emotional_state['fear_level'] = min(0.8, self.emotional_state['fear_level'] + 0.05)
        
        # Patience adjustment based on recent outcomes
        recent_trades = self.memories[-5:] if len(self.memories) >= 5 else self.memories
        if recent_trades:
            avg_outcome = np.mean([m.confidence_adjustment for m in recent_trades])
            if avg_outcome < -0.02:
                self.emotional_state['patience'] = min(0.95, self.emotional_state['patience'] + 0.1)
                self.emotional_state['risk_appetite'] = max(0.2, self.emotional_state['risk_appetite'] - 0.1)
        
        self.emotional_state['last_update'] = datetime.now()
    
    def _calculate_confidence_adjustment(self, pnl_percent: float) -> float:
        """Calculate how much to adjust confidence based on outcome"""
        if pnl_percent > 3:
            return 0.05
        elif pnl_percent > 1:
            return 0.02
        elif pnl_percent > 0:
            return 0.01
        elif pnl_percent > -1:
            return -0.01
        elif pnl_percent > -2:
            return -0.02
        else:
            return -0.05
    
    def get_pattern_confidence(self, pattern: str) -> float:
        """Get confidence in a pattern based on history"""
        if pattern not in self.pattern_success_rates:
            return 0.5
        
        stats = self.pattern_success_rates[pattern]
        if stats['total'] < 3:
            return 0.5  # Not enough data
        
        win_rate = stats['wins'] / stats['total']
        # Adjust by emotional state
        return win_rate * self.emotional_state['confidence']
    
    def should_take_trade(self, symbol: str, pattern: str, base_confidence: float) -> Tuple[bool, str]:
        """Decide if we should take a trade based on memory and emotional state"""
        # Recover confidence gradually
        self.recover_confidence()

        # Lower threshold during low confidence periods
        if self.emotional_state['confidence'] < 0.5:
            threshold = 0.3  # Much lower bar
        else:
            threshold = 0.6 * self.emotional_state['confidence']
        pattern_confidence = self.get_pattern_confidence(pattern)
        
        # Check recent memories for this symbol
        recent_symbol_memories = [m for m in self.memories[-20:] if m.symbol == symbol]
        if recent_symbol_memories:
            recent_success = sum(1 for m in recent_symbol_memories if 'win' in m.outcome.lower())
            recent_rate = recent_success / len(recent_symbol_memories)
            
            if recent_rate < 0.2 and len(recent_symbol_memories) >= 3:
                return False, f"I've been wrong on {symbol} lately. Sitting this one out."
        
        # Emotional state check
        if self.emotional_state['fear_level'] > 0.9:
            if base_confidence < 0.6:
                return False, "I'm feeling cautious after recent losses. Need a stronger setup."
        
        if self.emotional_state['greed_level'] > 0.7:
            if base_confidence < 0.6:
                return False, "Getting greedy here. Need to calm down and wait for quality setups."
        
        # Confidence threshold
        combined_confidence = (base_confidence + pattern_confidence) / 2
        threshold = 0.6 * self.emotional_state['confidence']
        
        if combined_confidence < threshold:
            return False, f"Not confident enough. Setup confidence: {combined_confidence:.1%}"
        
        return True, "Looks good based on my experience"


class DynamicExitManager:
    """Manages exits like a human day trader - adapts based on price action"""
    
    def __init__(self, trading_brain: TradingBrain, commentary_system):
        self.brain = trading_brain
        self.commentary = commentary_system
        self.exit_trackers: Dict[str, Dict[str, Any]] = {}
    
    def initialize_position_tracking(self, symbol: str, entry_price: float, 
                                   initial_stop: float, initial_target: float):
        """Start tracking a position for dynamic exit management"""
        self.exit_trackers[symbol] = {
            'entry_price': entry_price,
            'current_stop': initial_stop,
            'current_target': initial_target,
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'time_in_profit': 0,
            'time_in_loss': 0,
            'momentum_shifts': 0,
            'partial_exits': [],
            'trailing_activated': False,
            'scalp_target_hit': False,
            'last_check': datetime.now()
        }
    
    async def evaluate_exit(self, position, current_price: float, 
                          indicators: Dict[str, float]) -> Tuple[bool, str, float]:
        """
        Evaluate if we should exit - returns (should_exit, reason, exit_portion)
        exit_portion: 1.0 for full exit, 0.5 for half, etc.
        """
        symbol = position.symbol
        tracker = self.exit_trackers.get(symbol, {})
        
        if not tracker:
            return False, "", 0
        
        # Update tracker
        tracker['highest_price'] = max(tracker['highest_price'], current_price)
        tracker['lowest_price'] = min(tracker['lowest_price'], current_price)
        
        # Time tracking
        time_elapsed = datetime.now() - tracker['last_check']
        if current_price > position.entry_price:
            tracker['time_in_profit'] += time_elapsed.total_seconds()
        else:
            tracker['time_in_loss'] += time_elapsed.total_seconds()
        tracker['last_check'] = datetime.now()
        
        pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
        
        # 1. Quick Scalp Exit - Take quick profits
        if pnl_percent > 0.5 and not tracker['scalp_target_hit']:
            if tracker['time_in_profit'] < 300:  # Less than 5 minutes
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=symbol,
                    title=f"💰 Quick Scalp Opportunity",
                    message=f"Up {pnl_percent:.1f}% quickly. Taking half off the table like a smart day trader would.",
                    importance=8
                ))
                tracker['scalp_target_hit'] = True
                tracker['partial_exits'].append({'price': current_price, 'portion': 0.5})
                return True, "quick_scalp", 0.5

        # Progressive profit taking
        if pnl_percent > 1.5 and not tracker.get('scaled_out'):
            tracker['scaled_out'] = True
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=symbol,
                title=f"💰 Scaling Out 50%",
                message=f"Taking half off at {pnl_percent:.1f}% profit",
                importance=8
            ))
            return True, "scale_out_half", 0.5

        if pnl_percent > 3.0:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=symbol,
                title=f"🎯 Extended Target Hit",
                message=f"Full exit at {pnl_percent:.1f}% profit",
                importance=9
            ))
            return True, "take_profits_extended", 1.0

        # 2. Momentum Failure Exit
        if pnl_percent > 0.3:
            # Check if momentum is dying
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            
            if position.side == 'long' and macd < macd_signal and rsi < 50:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=symbol,
                    title=f"📉 Momentum Dying",
                    message=f"I'm up {pnl_percent:.1f}% but momentum is fading. Time to book profits before it reverses.",
                    data={'rsi': rsi, 'macd_cross': 'bearish'},
                    importance=8
                ))
                return True, "momentum_fade", 1.0
        
        # 3. Time-Based Trailing Stop
        if pnl_percent > 1.0 and not tracker['trailing_activated']:
            new_stop = position.entry_price * 1.002  # Move stop to breakeven + 0.2%
            tracker['current_stop'] = new_stop
            tracker['trailing_activated'] = True
            
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=symbol,
                title=f"🛡️ Protecting Profits",
                message=f"Moving stop to breakeven + 0.2%. Can't let a winner turn into a loser!",
                importance=7
            ))
        
        # 4. Dynamic Trailing Stop based on ATR
        if tracker['trailing_activated'] and pnl_percent > 1.5:
            atr = indicators.get('atr', current_price * 0.01)
            new_stop = current_price - (1.5 * atr)
            
            if new_stop > tracker['current_stop']:
                tracker['current_stop'] = new_stop
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=symbol,
                    title=f"📈 Trailing Stop Update",
                    message=f"Profit at {pnl_percent:.1f}%. Moving stop up to ${new_stop:.2f} to lock in more gains.",
                    importance=6
                ))
        
        # 5. Exhaustion Exit - When move is overextended
        if pnl_percent > 2.0:
            price_extension = ((current_price - tracker['lowest_price']) / tracker['lowest_price']) * 100
            if price_extension > 3.0 and indicators.get('rsi', 50) > 75:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.OPPORTUNITY,
                    symbol=symbol,
                    title=f"🎯 Exhaustion Top",
                    message=f"Up {pnl_percent:.1f}% and RSI screaming overbought. Taking profits here - pigs get slaughtered!",
                    confidence=0.9,
                    importance=9
                ))
                return True, "exhaustion", 1.0
        
        # 6. Time Decay Exit - Position not working out
        position_age_minutes = (datetime.now() - position.entry_time).total_seconds() / 60
        if position_age_minutes > 30 and abs(pnl_percent) < 0.3:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.PSYCHOLOGY,
                symbol=symbol,
                title=f"😴 Dead Money",
                message=f"This trade isn't working after {position_age_minutes:.0f} minutes. Moving on to better opportunities.",
                importance=7
            ))
            return True, "time_decay", 1.0
        
        # 7. Check against dynamic stop
        if current_price <= tracker['current_stop']:
            return True, "trailing_stop", 1.0
        
        # 8. Human-like Patience Limit
        if position_age_minutes > 15 and pnl_percent > 0.8:
            patience = self.brain.emotional_state['patience']
            if patience < 0.5 and np.random.random() > patience:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.PSYCHOLOGY,
                    symbol=symbol,
                    title=f"😤 Getting Impatient",
                    message=f"I've been in this trade for {position_age_minutes:.0f} minutes. Good enough profit at {pnl_percent:.1f}%, taking it.",
                    importance=7
                ))
                return True, "impatience", 1.0
        
        return False, "", 0

        # Short position stop loss (price went up)
        if position.side == 'short' and current_price >= position.stop_loss:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=position.symbol,
                title=f"🛑 Short Stop Loss Hit",
                message=f"Stopped out of short at ${current_price:.2f}",
                importance=9
            ))
            return True, "stop_loss_short"    

# ============================================================================
# SCHWAB DATA PROVIDER
# ============================================================================

class SchwabDataProvider:
    """Schwab data provider with commentary"""
    
    def __init__(self, client: schwab_client.Client, commentary_system):
        self.client = client
        self._cache = {}
        self._cache_timeout = 60
        self.commentary = commentary_system
        self.market_hours_cache = {}
        
    def get_market_data(self, symbol: str, period_type: str = 'day', 
                       period: int = 1, frequency_type: str = 'minute', 
                       frequency: int = 5) -> pd.DataFrame:
        """Get market data using Schwab API"""
        cache_key = f"{symbol}_{period_type}_{frequency_type}_{frequency}"
        
        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data
        
        try:
            # Use simplified calls for common scenarios
            if period_type == 'day' and period == 1 and frequency_type == 'minute':
                if frequency == 5:
                    response = self.client.get_price_history_every_five_minutes(symbol)
                elif frequency == 1:
                    response = self.client.get_price_history_every_minute(symbol)
                else:
                    response = self.client.get_price_history_every_day(symbol)
            elif frequency_type == 'daily':
                response = self.client.get_price_history_every_day(symbol)
            else:
                response = self.client.get_price_history_every_day(symbol)
            
            if response and hasattr(response, 'status_code') and response.status_code == 200:
                data = response.json()
                candles = data.get('candles', [])
                
                if not candles:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ No Price Data: {symbol}",
                        message="Schwab API returned empty candles array",
                        importance=5
                    ))
                    return pd.DataFrame()
                
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.TECHNICAL,
                    symbol=symbol,
                    title=f"📈 Data Retrieved: {symbol}",
                    message=f"Got {len(candles)} price candles from Schwab",
                    data={
                        'candle_count': len(candles),
                        'timeframe': f"{frequency}{frequency_type}"
                    },
                    importance=3
                ))
                
                try:
                    df = pd.DataFrame(candles)
                    if not df.empty and 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                        df.set_index('datetime', inplace=True)
                        df.rename(columns={
                            'open': 'Open',
                            'high': 'High',
                            'low': 'Low',
                            'close': 'Close',
                            'volume': 'Volume'
                        }, inplace=True)
                        
                        # Ensure all numeric columns are float64
                        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                        for col in numeric_columns:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
                        
                        # Drop any rows with NaN values
                        df = df.dropna()
                        
                        if df.empty:
                            self.commentary.add_commentary(TradingCommentary(
                                timestamp=datetime.now(),
                                type=CommentaryType.WARNING,
                                symbol=symbol,
                                title=f"⚠️ Data Quality Issue: {symbol}",
                                message="All data rows were invalid after cleaning",
                                importance=5
                            ))
                            return pd.DataFrame()
                        
                        self._cache[cache_key] = (df, time.time())
                        return df
                    else:
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.WARNING,
                            symbol=symbol,
                            title=f"⚠️ Data Format Issue: {symbol}",
                            message="Data structure is not as expected",
                            importance=5
                        ))
                        return pd.DataFrame()
                except Exception as e:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ Data Processing Error: {symbol}",
                        message=f"Error processing candle data: {str(e)}",
                        importance=6
                    ))
                    return pd.DataFrame()
            else:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title=f"⚠️ Data Fetch Failed: {symbol}",
                    message=f"API returned status code: {getattr(response, 'status_code', 'Unknown')}",
                    importance=6
                ))
            
            return pd.DataFrame()
            
        except Exception as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title=f"⚠️ Data Error: {symbol}",
                message=f"Error fetching data: {str(e)}",
                importance=7
            ))
            return pd.DataFrame()
    
    def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get real-time quote"""
        try:
            response = self.client.get_quote(symbol)
            assert response.status_code == 200, response.text
            data = response.json()
            
            quote_data = data.get(symbol, {})
            if 'quote' in quote_data:
                quote = quote_data['quote']
            else:
                quote = quote_data
            
            return {
                'bid': quote.get('bidPrice', 0),
                'ask': quote.get('askPrice', 0),
                'last': quote.get('lastPrice', 0),
                'volume': quote.get('totalVolume', 0),
                'high': quote.get('highPrice', 0),
                'low': quote.get('lowPrice', 0),
                'open': quote.get('openPrice', 0),
                'close': quote.get('closePrice', 0),
                'mark': quote.get('mark', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return {}
    
    def calculate_market_breadth(self) -> Dict[str, float]:
        """Calculate market breadth using Schwab data"""
        try:
            # Get movers data
            try:
                gainers = self._get_movers('$SPX.X', 'up')
                losers = self._get_movers('$SPX.X', 'down')
            except:
                gainers = []
                losers = []
       
            # Get VIX quote - handle index differently
            vix_value = 20  # Default
            try:                               
                # If still no VIX, try getting it from index endpoint
                if vix_value == 20:
                    try:
                        response = self.client.get_quotes('$VIX')
                        if response.status_code == 200:
                            data = response.json()
                            #logger.info(f" VIX Quote Details: {data}")
                            # Indices might be in a different structure
                            if '$VIX' in data:
                                vix_data = data['$VIX']
                                #print(vix_data)
                                if 'quote' in vix_data:
                                    vix_quote = vix_data['quote'].get('lastPrice', 20)
                                elif 'lastPrice' in vix_data:
                                    vix_quote = vix_data.get('lastPrice', 20)
                    except:
                        pass
            except Exception as e:
                logger.debug(f"VIX fetch failed, using default: {e}")
            
            # Calculate breadth metrics
            advance_decline = len(gainers) - len(losers) if gainers or losers else 0
            
            breadth_data = {
                'advance_decline': advance_decline,
                'new_highs': len([g for g in gainers if g.get('changePercent', 0) > 5]) if gainers else 0,
                'new_lows': len([l for l in losers if abs(l.get('changePercent', 0)) > 5]) if losers else 0,
                'vix': vix_quote,
                'put_call_ratio': 1.0  # Default, would need options data
            }
            
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="📊 Market Breadth Calculated",
                message="Market internals analyzed",
                data=breadth_data,
                importance=4
            ))
            
            return breadth_data
            
        except Exception as e:
            logger.error(f"Error calculating market breadth: {e}")
            return {
                'advance_decline': 0,
                'new_highs': 0,
                'new_lows': 0,
                'vix': 20,
                'put_call_ratio': 1.0
            }
    
    def _get_movers(self, index: str, direction: str) -> List[Dict]:
        """Get market movers"""
        try:
            response = self.client.get_movers(index, direction=direction, change='percent')
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return []

# ============================================================================
# STOCK SCREENER
# ============================================================================

class StockScreener:
    """Real-time stock screener using Schwab API"""
    
    def __init__(self, schwab_client, commentary_system):
        self.client = schwab_client
        self.commentary = commentary_system
        self.top_movers = []
        self.last_screen_time = None
        self.screen_interval = 120  # 2 minutes
        self._movers_cache = {}
        self._cache_timeout = 60  # 1 minute cache for API efficiency
        
    async def screen_stocks(self) -> List[Dict[str, Any]]:
        """Screen for top moving and volatile stocks"""
        try:
            # Get movers from major indices
            all_movers = []
            
            # Get S&P 500 movers
            sp500_movers = await self._get_index_movers('$SPX.X')
            all_movers.extend(sp500_movers)
            
            # Get NASDAQ movers
            nasdaq_movers = await self._get_index_movers('$COMPX')
            all_movers.extend(nasdaq_movers)
            
            # Get additional volatile stocks
            volatile_stocks = await self._get_volatile_stocks()
            all_movers.extend(volatile_stocks)
            
            # Remove duplicates and filter
            seen = set()
            unique_movers = []
            
            for mover in all_movers:
                if mover['symbol'] not in seen and self._is_tradeable(mover):
                    seen.add(mover['symbol'])
                    unique_movers.append(mover)
            
            # Sort by combined score (volatility + volume)
            scored_movers = []
            for mover in unique_movers:
                score = self._calculate_mover_score(mover)
                mover['score'] = score
                scored_movers.append(mover)
            
            # Sort and take top 10
            scored_movers.sort(key=lambda x: x['score'], reverse=True)
            self.top_movers = scored_movers[:10]
            
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="🔍 Stock Screener Update",
                message=f"Found {len(self.top_movers)} high-opportunity stocks",
                data={
                    'top_gainer': self.top_movers[0]['symbol'] if self.top_movers else None,
                    'avg_volatility': np.mean([m['volatility'] for m in self.top_movers]) if self.top_movers else 0
                },
                importance=6
            ))
            
            self.last_screen_time = datetime.now()
            return self.top_movers
            
        except Exception as e:
            logger.error(f"Screener error: {e}")
            return []
    
    async def _get_index_movers(self, index: str) -> List[Dict[str, Any]]:
        """Get movers for a specific index"""
        cache_key = f"movers_{index}"
        
        # Check cache
        if cache_key in self._movers_cache:
            cached_data, timestamp = self._movers_cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data
        
        movers = []
        
        try:
            # Get both gainers and losers
            for direction in ['up', 'down']:
                response = self.client.get_movers(index, direction=direction, change='percent')
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for item in data[:20]:  # Top 20 from each
                        if 'symbol' in item:
                            mover = {
                                'symbol': item['symbol'],
                                'description': item.get('description', ''),
                                'last': item.get('last', 0),
                                'change': item.get('change', 0),
                                'percent_change': item.get('changePercent', 0),
                                'volume': item.get('totalVolume', 0),
                                'direction': direction
                            }
                            
                            # Get detailed quote for more data
                            quote = self._get_quote_data(mover['symbol'])
                            if quote:
                                mover.update(quote)
                            
                            movers.append(mover)
        except Exception as e:
            logger.debug(f"Error getting movers for {index}: {e}")
        
        # Cache results
        self._movers_cache[cache_key] = (movers, time.time())
        return movers
    
    async def _get_volatile_stocks(self) -> List[Dict[str, Any]]:
        """Find additional volatile stocks using a pre-screened watchlist"""
        volatile_candidates = [
            'TSLA', 'NVDA', 'AMD', 'PLTR', 'COIN', 'MARA', 'RIOT', 
            'SQ', 'ROKU', 'SNAP', 'PINS', 'DKNG', 'PENN', 'FUBO',
            'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'FSR'
        ]
        
        volatile_stocks = []
        
        for symbol in volatile_candidates:
            try:
                quote = self._get_quote_data(symbol)
                if quote and quote.get('volume', 0) > 500000:
                    volatile_stocks.append(quote)
            except:
                continue
        
        return volatile_stocks
    
    def _get_quote_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed quote data for a symbol"""
        try:
            response = self.client.get_quote(symbol)
            if response.status_code == 200:
                data = response.json()
                quote_data = data.get(symbol, {})
                
                if 'quote' in quote_data:
                    quote = quote_data['quote']
                else:
                    quote = quote_data
                
                # Calculate volatility metrics
                high = quote.get('highPrice', 0)
                low = quote.get('lowPrice', 0)
                open_price = quote.get('openPrice', 1)
                last = quote.get('lastPrice', 0)
                
                # Intraday range percentage (best for day trading)
                volatility = ((high - low) / open_price * 100) if open_price > 0 else 0
                
                return {
                    'symbol': symbol,
                    'description': quote.get('description', ''),
                    'last': last,
                    'change': quote.get('netChange', 0),
                    'percent_change': quote.get('netPercentChangeInDouble', 0),
                    'volume': quote.get('totalVolume', 0),
                    'high': high,
                    'low': low,
                    'open': open_price,
                    'volatility': volatility,
                    'bid': quote.get('bidPrice', 0),
                    'ask': quote.get('askPrice', 0),
                    'spread': quote.get('askPrice', 0) - quote.get('bidPrice', 0)
                }
        except Exception as e:
            logger.debug(f"Quote error for {symbol}: {e}")
        
        return None
    
    def _is_tradeable(self, mover: Dict[str, Any]) -> bool:
        """Filter for tradeable stocks"""
        # Volume filter
        if mover.get('volume', 0) < 500000:
            return False
        
        # Price filter ($5-$500 for good liquidity)
        price = mover.get('last', 0)
        if price < 5 or price > 500:
            return False
        
        # Spread filter (less than 0.5% spread)
        if mover.get('spread', 0) / price > 0.005:
            return False
        
        # Exclude ETFs and special instruments
        symbol = mover.get('symbol', '')
        if any(etf in symbol for etf in ['SPY', 'QQQ', 'IWM', 'DIA', 'VXX', 'UVXY']):
            return False
        
        return True
    
    def _calculate_mover_score(self, mover: Dict[str, Any]) -> float:
        """Calculate a score for ranking movers"""
        # Volatility component (40% weight)
        volatility_score = min(mover.get('volatility', 0) / 5, 1) * 0.4
        
        # Volume component (30% weight) - normalized by average
        volume = mover.get('volume', 0)
        volume_score = min(volume / 10_000_000, 1) * 0.3
        
        # Price change component (30% weight)
        change_score = min(abs(mover.get('percent_change', 0)) / 10, 1) * 0.3
        
        return volatility_score + volume_score + change_score
    
    def get_watchlist_symbols(self) -> List[str]:
        """Get current top mover symbols for the watchlist"""
        return [mover['symbol'] for mover in self.top_movers[:5]]  # Top 5 for focused trading
# ============================================================================
# TECHNICAL ANALYZER WITH COMMENTARY
# ============================================================================

class TechnicalAnalyzerWithCommentary:
    """Technical analyzer that explains its analysis"""
    
    def __init__(self, commentary_system):
        self.commentary = commentary_system
    
    async def analyze_with_commentary(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Perform technical analysis with commentary"""
        if data is None or data.empty:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title=f"⚠️ No Data Available: {symbol}",
                message="No price data available for technical analysis",
                importance=5
            ))
            return {}
        
        # Check if we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title=f"⚠️ Missing Data Columns: {symbol}",
                message=f"Missing required columns: {', '.join(missing_columns)}",
                importance=5
            ))
            return {}
        
        if len(data) < 50:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title=f"⚠️ Insufficient Data: {symbol}",
                message=f"Only {len(data)} candles available, need at least 50 for analysis",
                importance=5
            ))
            return {}
        
        indicators = {}
        
        try:
            # Ensure data is numeric and handle any non-numeric values
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    # Convert to numeric, replacing any non-numeric with NaN
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    # Fill NaN values with forward fill, then backward fill
                    data[col] = data[col].ffill().bfill()
            
            # Drop any remaining rows with NaN values
            data = data.dropna()
            
            if len(data) < 50:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title=f"⚠️ Data Quality Issue: {symbol}",
                    message=f"After cleaning, only {len(data)} valid candles remain",
                    importance=5
                ))
                return {}
            
            # Convert to numpy arrays with proper type
            close = data['Close'].values.astype(np.float64)
            high = data['High'].values.astype(np.float64)
            low = data['Low'].values.astype(np.float64)
            volume = data['Volume'].values.astype(np.float64)
            
            # Moving averages
            try:
                indicators['sma_20'] = float(talib.SMA(close, timeperiod=20)[-1])
                indicators['sma_50'] = float(talib.SMA(close, timeperiod=50)[-1])
                indicators['ema_20'] = float(talib.EMA(close, timeperiod=20)[-1])
            except Exception as e:
                logger.debug(f"MA calculation error for {symbol}: {e}")
                indicators['sma_20'] = float(np.mean(close[-20:]))
                indicators['sma_50'] = float(np.mean(close[-50:])) if len(close) >= 50 else float(np.mean(close))
                indicators['ema_20'] = indicators['sma_20']  # Fallback to SMA
            
            # RSI
            try:
                rsi_values = talib.RSI(close, timeperiod=14)
                indicators['rsi'] = float(rsi_values[-1]) if not np.isnan(rsi_values[-1]) else 50.0
            except Exception as e:
                logger.debug(f"RSI calculation error for {symbol}: {e}")
                indicators['rsi'] = 50.0  # Neutral default
            
            # MACD
            try:
                macd, macd_signal, macd_hist = talib.MACD(close)
                indicators['macd'] = float(macd[-1]) if not np.isnan(macd[-1]) else 0.0
                indicators['macd_signal'] = float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else 0.0
                indicators['macd_histogram'] = float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else 0.0
            except Exception as e:
                logger.debug(f"MACD calculation error for {symbol}: {e}")
                indicators['macd'] = 0.0
                indicators['macd_signal'] = 0.0
                indicators['macd_histogram'] = 0.0
            
            # Bollinger Bands
            try:
                upper, middle, lower = talib.BBANDS(close, timeperiod=20)
                indicators['bb_upper'] = float(upper[-1]) if not np.isnan(upper[-1]) else close[-1] * 1.02
                indicators['bb_middle'] = float(middle[-1]) if not np.isnan(middle[-1]) else close[-1]
                indicators['bb_lower'] = float(lower[-1]) if not np.isnan(lower[-1]) else close[-1] * 0.98
            except Exception as e:
                logger.debug(f"BB calculation error for {symbol}: {e}")
                mean_price = float(np.mean(close[-20:]))
                std_price = float(np.std(close[-20:]))
                indicators['bb_middle'] = mean_price
                indicators['bb_upper'] = mean_price + (2 * std_price)
                indicators['bb_lower'] = mean_price - (2 * std_price)
            
            # ATR
            try:
                atr_values = talib.ATR(high, low, close, timeperiod=14)
                indicators['atr'] = float(atr_values[-1]) if not np.isnan(atr_values[-1]) else 0.0
            except Exception as e:
                logger.debug(f"ATR calculation error for {symbol}: {e}")
                # Simple ATR approximation
                tr = np.maximum(high[-14:] - low[-14:], np.abs(high[-14:] - close[-15:-1]))
                indicators['atr'] = float(np.mean(tr))
            
            # Volume indicators - handle potential issues
            try:
                # OBV can be problematic with large volumes, use relative OBV
                if len(close) > 0 and len(volume) > 0:
                    obv_values = talib.OBV(close, volume)
                    if len(obv_values) > 0 and not np.isnan(obv_values[-1]):
                        indicators['obv'] = float(obv_values[-1])
                    else:
                        # Calculate simple OBV manually
                        obv = 0
                        for i in range(1, len(close)):
                            if close[i] > close[i-1]:
                                obv += volume[i]
                            elif close[i] < close[i-1]:
                                obv -= volume[i]
                        indicators['obv'] = float(obv)
                else:
                    indicators['obv'] = 0.0
            except Exception as e:
                logger.debug(f"OBV calculation error for {symbol}: {e}")
                indicators['obv'] = float(np.sum(volume[-20:])) if len(volume) > 0 else 0.0
            
            # ADX
            try:
                adx_values = talib.ADX(high, low, close, timeperiod=14)
                indicators['adx'] = float(adx_values[-1]) if not np.isnan(adx_values[-1]) else 0.0
            except Exception as e:
                logger.debug(f"ADX calculation error for {symbol}: {e}")
                indicators['adx'] = 25.0  # Neutral trend strength
            
            # Support/Resistance
            pivot = (high[-1] + low[-1] + close[-1]) / 3
            indicators['pivot'] = float(pivot)
            indicators['resistance_1'] = float(2 * pivot - low[-1])
            indicators['support_1'] = float(2 * pivot - high[-1])
            
            # Add calculated metrics for ML
            indicators['returns'] = float((close[-1] - close[-2]) / close[-2]) if len(close) > 1 else 0.0
            indicators['volume_ratio'] = float(volume[-1] / np.mean(volume[-20:])) if len(volume) > 20 and np.mean(volume[-20:]) > 0 else 1.0
            indicators['high_low_ratio'] = float((high[-1] - low[-1]) / close[-1]) if close[-1] > 0 else 0.02
            
            # Ensure all indicators are float type
            for key, value in indicators.items():
                if isinstance(value, (np.floating, np.integer)):
                    indicators[key] = float(value)
                elif np.isnan(value) or np.isinf(value):
                    indicators[key] = 0.0
            
            # Interpret indicators
            interpretations = []
            
            # Trend analysis
            if close[-1] > indicators['sma_50']:
                interpretations.append("📈 Price above 50 SMA - Uptrend")
            else:
                interpretations.append("📉 Price below 50 SMA - Downtrend")
            
            # RSI analysis
            if indicators['rsi'] > 70:
                interpretations.append(f"🔥 RSI at {indicators['rsi']:.1f} - Overbought")
            elif indicators['rsi'] < 30:
                interpretations.append(f"❄️ RSI at {indicators['rsi']:.1f} - Oversold")
            else:
                interpretations.append(f"➖ RSI at {indicators['rsi']:.1f} - Neutral")
            
            # MACD analysis
            if indicators['macd'] > indicators['macd_signal']:
                interpretations.append("🟢 MACD above signal - Bullish momentum")
            else:
                interpretations.append("🔴 MACD below signal - Bearish momentum")
            
            # Bollinger Bands
            if close[-1] > indicators['bb_upper']:
                interpretations.append("📊 Price above upper Bollinger Band - Overbought")
            elif close[-1] < indicators['bb_lower']:
                interpretations.append("📊 Price below lower Bollinger Band - Oversold")
            
            # ADX trend strength
            if indicators['adx'] > 25:
                interpretations.append(f"💪 ADX at {indicators['adx']:.1f} - Strong trend")
            else:
                interpretations.append(f"😴 ADX at {indicators['adx']:.1f} - Weak trend")
            
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.TECHNICAL,
                symbol=symbol,
                title=f"📊 Technical Analysis: {symbol}",
                message="\n".join(interpretations),
                data={
                    'price': float(close[-1]),
                    'rsi': indicators['rsi'],
                    'sma_20': indicators['sma_20'],
                    'sma_50': indicators['sma_50'],
                    'volume_ratio': indicators['volume_ratio'],
                    'atr': indicators['atr'],
                    'adx': indicators['adx']
                },
                importance=5
            ))
            
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {e}", exc_info=True)
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title=f"⚠️ Technical Analysis Error: {symbol}",
                message=f"Could not complete technical analysis: {str(e)}",
                importance=7
            ))
            
            # Return basic indicators as fallback
            try:
                close_price = float(data['Close'].iloc[-1])
                indicators = {
                    'sma_20': close_price,
                    'sma_50': close_price,
                    'ema_20': close_price,
                    'rsi': 50.0,
                    'macd': 0.0,
                    'macd_signal': 0.0,
                    'macd_histogram': 0.0,
                    'bb_upper': close_price * 1.02,
                    'bb_middle': close_price,
                    'bb_lower': close_price * 0.98,
                    'atr': close_price * 0.01,
                    'obv': 0.0,
                    'adx': 25.0,
                    'pivot': close_price,
                    'resistance_1': close_price * 1.01,
                    'support_1': close_price * 0.99,
                    'returns': 0.0,
                    'volume_ratio': 1.0,
                    'high_low_ratio': 0.02
                }
            except:
                indicators = {}
        
        return indicators

class MarketRegimeDetector:
    """Detects market regimes for adaptive model behavior"""
    
    def __init__(self):
        self.regime_history = []
        self.current_regime = "normal"
        
    def detect_regime(self, market_data: Dict[str, float]) -> str:
        """Detect current market regime based on volatility and trend"""
        vix = market_data.get('vix', 20)
        trend_strength = market_data.get('trend_strength', 0)
        
        if vix > 30:
            regime = "high_volatility"
        elif vix < 15:
            regime = "low_volatility"
        elif abs(trend_strength) > 0.7:
            regime = "trending"
        else:
            regime = "normal"
            
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': regime,
            'vix': vix
        })
        
        self.current_regime = regime
        return regime

class ConceptDriftDetector:
    """Monitors for concept drift in model performance"""
    
    def __init__(self, warning_level: float = 0.95, drift_level: float = 0.99):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.error_rate = 0.0
        self.error_std = 0.0
        self.n_samples = 0
        self.drift_detected = False
        self.warning_detected = False
        
    def update(self, prediction_correct: bool) -> Tuple[bool, bool]:
        """Update drift detector with new prediction result"""
        error = 0 if prediction_correct else 1
        
        if self.n_samples == 0:
            self.error_rate = error
            self.error_std = 0
        else:
            # Update error rate with exponential weighted average
            alpha = 2 / (self.n_samples + 1)
            self.error_rate = alpha * error + (1 - alpha) * self.error_rate
            self.error_std = np.sqrt(self.error_rate * (1 - self.error_rate) / (self.n_samples + 1))
        
        self.n_samples += 1
        
        # Check for drift
        if self.n_samples > 30:  # Need minimum samples
            drift_threshold = self.error_rate + self.error_std * self.drift_level
            warning_threshold = self.error_rate + self.error_std * self.warning_level
            
            current_error_rate = error
            
            if current_error_rate > drift_threshold:
                self.drift_detected = True
            elif current_error_rate > warning_threshold:
                self.warning_detected = True
                
        return self.warning_detected, self.drift_detected

class AdvancedFeatureEngineer:
    """Multi-timeframe feature engineering with microstructure and wavelets"""
    
    def __init__(self):
        self.timeframes = ['1min', '5min', '15min', '1hour']
        self.feature_names = []
        
    def extract_multi_timeframe_features(self, data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Extract features from multiple timeframes"""
        all_features = []
        
        for tf in self.timeframes:
            if tf in data and not data[tf].empty:
                tf_features = self._extract_timeframe_features(data[tf], tf)
                all_features.extend(tf_features)
                
        return np.array(all_features)
    
    def _extract_timeframe_features(self, df: pd.DataFrame, timeframe: str) -> List[float]:
        """Extract features for a single timeframe"""
        features = []
        
        if len(df) < 50:
            return [0] * 26  # Return zeros if insufficient data
        
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        # Price-based features
        features.extend([
            # Returns at different intervals
            (close[-1] - close[-2]) / close[-2] if close[-2] != 0 else 0,
            (close[-1] - close[-5]) / close[-5] if close[-5] != 0 else 0,
            (close[-1] - close[-20]) / close[-20] if close[-20] != 0 else 0,
            
            # Moving averages
            close[-20:].mean(),
            close[-50:].mean() if len(close) >= 50 else close.mean(),
            
            # Price position
            (close[-1] - low[-20:].min()) / (high[-20:].max() - low[-20:].min() + 1e-10),
            
            # Volatility
            np.std(close[-20:]) / (close[-20:].mean() + 1e-10),
            
            # Volume features
            volume[-1] / (volume[-20:].mean() + 1e-10),
            np.std(volume[-20:]) / (volume[-20:].mean() + 1e-10),
        ])
        
        # Technical indicators
        features.extend(self._calculate_technical_indicators(close, high, low, volume))
        
        # Microstructure features
        features.extend(self._calculate_microstructure_features(df))
        
        # Statistical moments
        returns = np.diff(close[-20:]) / close[-21:-1]
        features.extend([
            stats.skew(returns),
            stats.kurtosis(returns),
            np.percentile(returns, 75) - np.percentile(returns, 25)  # IQR
        ])
        
        # Wavelet features if available
        if WAVELET_AVAILABLE:
            features.extend(self._extract_wavelet_features(close[-50:]))
        
        return features
    
    def _calculate_technical_indicators(self, close: np.ndarray, high: np.ndarray, 
                                      low: np.ndarray, volume: np.ndarray) -> List[float]:
        """Calculate technical indicators"""
        features = []
        
        # RSI
        rsi = self._calculate_rsi(close, 14)
        features.append(rsi)
        
        # MACD
        if len(close) >= 26:
            macd, signal, hist = self._calculate_macd(close)
            features.extend([macd, signal, hist])
        else:
            features.extend([0, 0, 0])
        
        # ATR
        atr = self._calculate_atr(high, low, close, 14)
        features.append(atr)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close)
        features.extend([
            (close[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10),
            (bb_upper - bb_lower) / (bb_middle + 1e-10)  # BB width
        ])
        
        return features
    
    def _calculate_microstructure_features(self, df: pd.DataFrame) -> List[float]:
        """Calculate market microstructure features"""
        features = []
        
        # Bid-ask spread proxy (high-low as proxy)
        spread = (df['High'] - df['Low']).iloc[-20:]
        features.extend([
            spread.mean(),
            spread.std(),
            spread.iloc[-1] / spread.mean() if spread.mean() > 0 else 1
        ])
        
        # Order flow imbalance proxy (using volume and price direction)
        price_direction = np.sign(df['Close'].diff())
        signed_volume = df['Volume'] * price_direction
        
        features.extend([
            signed_volume.iloc[-20:].sum() / (df['Volume'].iloc[-20:].sum() + 1e-10),
            signed_volume.iloc[-5:].sum() / (df['Volume'].iloc[-5:].sum() + 1e-10)
        ])
        
        return features
    
    def _extract_wavelet_features(self, signal: np.ndarray) -> List[float]:
        """Extract wavelet transform features"""
        features = []
        
        try:
            # Discrete Wavelet Transform
            coeffs = pywt.wavedec(signal, 'db4', level=3)
            
            # Energy of each level
            for i, coeff in enumerate(coeffs):
                energy = np.sum(coeff**2)
                features.append(energy)
            
            # Approximate to detail ratio
            if len(coeffs) > 1:
                approx_energy = np.sum(coeffs[0]**2)
                detail_energy = sum(np.sum(c**2) for c in coeffs[1:])
                features.append(approx_energy / (detail_energy + 1e-10))
            
        except Exception:
            features = [0] * 5  # Default features if wavelet fails
            
        return features
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100.0
            
        rs = up / down
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """Calculate MACD"""
        exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        
        return macd.iloc[-1], signal.iloc[-1], hist.iloc[-1]
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, 
                      close: np.ndarray, period: int = 14) -> float:
        """Calculate ATR"""
        if len(high) < period:
            return 0.0
            
        tr = np.maximum(high - low, 
                       np.abs(high - np.roll(close, 1)),
                       np.abs(low - np.roll(close, 1)))[1:]
        
        return np.mean(tr[-period:])
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, 
                                  period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]
            
        middle = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower

class HybridTradingModel:
    """Production-ready hybrid ML model with adaptive retraining"""
    
    def __init__(self, commentary_system=None):
        self.commentary = commentary_system
        self.feature_engineer = AdvancedFeatureEngineer()
        self.regime_detector = MarketRegimeDetector()
        self.drift_detector = ConceptDriftDetector()
        
        # Model ensemble
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                random_state=42,
                verbose=-1
            )
        }
        
        # Meta learner
        self.meta_model = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='soft',
            weights=[0.4, 0.4, 0.2]  # RF and XGB weighted more
        )
        
        self.scaler = RobustScaler()  # Robust to outliers
        self.is_trained = False
        self.last_training_time = None
        self.training_history = []
        self.feature_importance = {}
        
        # Performance tracking
        self.performance_metrics = {
            'accuracy': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'win_rate': []
        }
        
    def should_retrain(self, current_metrics: Dict[str, float]) -> bool:
        """Determine if model should be retrained based on multiple factors"""
        if not self.last_training_time:
            return True
            
        # Time-based retraining
        regime = self.regime_detector.current_regime
        if regime == "high_volatility":
            retrain_interval = timedelta(hours=6)  # Retrain every 6 hours
        elif regime == "trending":
            retrain_interval = timedelta(days=1)
        else:
            retrain_interval = timedelta(days=3)
            
        if datetime.now() - self.last_training_time > retrain_interval:
            return True
            
        # Performance-based retraining
        if len(self.performance_metrics['sharpe_ratio']) >= 5:
            recent_sharpe = np.mean(self.performance_metrics['sharpe_ratio'][-5:])
            if recent_sharpe < 0.5:
                return True
                
        if len(self.performance_metrics['accuracy']) >= 10:
            recent_accuracy = np.mean(self.performance_metrics['accuracy'][-10:])
            if recent_accuracy < 0.55:
                return True
                
        # Drift-based retraining
        if self.drift_detector.drift_detected:
            return True
            
        return False
    
    def train(self, training_data: Dict[str, pd.DataFrame], labels: pd.Series,
              market_conditions: Dict[str, float]):
        """Train the hybrid model with multi-timeframe data"""
        if self.commentary:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="🎯 Training Advanced ML Model",
                message="Extracting multi-timeframe features and training ensemble...",
                importance=8
            ))
        
        # Extract features for all samples
        X = []
        valid_indices = []
        
        for i in range(len(labels)):
            try:
                # Get historical data for each timeframe
                sample_data = {}
                for tf in self.feature_engineer.timeframes:
                    if tf in training_data:
                        # Get data up to index i
                        sample_data[tf] = training_data[tf].iloc[max(0, i-100):i+1]
                
                if all(tf in sample_data and len(sample_data[tf]) > 20 
                       for tf in ['1min', '5min']):  # Minimum required
                    features = self.feature_engineer.extract_multi_timeframe_features(sample_data)
                    X.append(features)
                    valid_indices.append(i)
            except Exception as e:
                continue
        
        if len(X) < 100:
            if self.commentary:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=None,
                    title="⚠️ Insufficient Training Data",
                    message=f"Only {len(X)} valid samples available",
                    importance=8
                ))
            return
        
        X = np.array(X)
        y = labels.iloc[valid_indices].values
        
        # Handle any remaining NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train individual models
            for name, model in self.models.items():
                model.fit(X_train, y_train)
                
            # Train meta model
            self.meta_model.fit(X_train, y_train)
            
            # Validate
            val_pred = self.meta_model.predict(X_val)
            accuracy = accuracy_score(y_val, val_pred)
            cv_scores.append(accuracy)
        
        # Final training on all data
        self.meta_model.fit(X_scaled, y)
        
        # Calculate feature importance
        self._calculate_feature_importance(X, y)
        
        # Update training metadata
        self.is_trained = True
        self.last_training_time = datetime.now()
        self.training_history.append({
            'timestamp': self.last_training_time,
            'samples': len(X),
            'cv_accuracy': np.mean(cv_scores),
            'regime': self.regime_detector.current_regime
        })
        
        if self.commentary:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="✅ Model Training Complete",
                message=f"Trained on {len(X)} samples with {np.mean(cv_scores):.1%} CV accuracy",
                data={
                    'models': list(self.models.keys()),
                    'features': len(X[0]),
                    'regime': self.regime_detector.current_regime
                },
                confidence=np.mean(cv_scores),
                importance=8
            ))
    
    def predict(self, market_data: Dict[str, pd.DataFrame], symbol: str) -> Tuple[int, Dict[str, Any]]:
        """Make prediction with confidence and explanation"""
        if not self.is_trained:
            return 0, {'error': 'Model not trained'}
        
        try:
            # Extract features
            features = self.feature_engineer.extract_multi_timeframe_features(market_data)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            X = features.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                pred = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled)[0]
                predictions[name] = pred
                probabilities[name] = max(prob)
            
            # Meta prediction
            final_prediction = self.meta_model.predict(X_scaled)[0]
            final_probability = self.meta_model.predict_proba(X_scaled)[0]
            confidence = max(final_probability)
            
            # Generate explanation
            explanation = self._generate_prediction_explanation(
                features, predictions, probabilities, final_prediction, confidence
            )
            
            # Update drift detector
            # Note: In production, you'd check actual outcome later
            self.drift_detector.update(True)  # Placeholder
            
            return int(final_prediction), explanation
            
        except Exception as e:
            if self.commentary:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title="⚠️ Prediction Error",
                    message=str(e),
                    importance=7
                ))
            return 0, {'error': str(e)}
    
    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """Calculate and store feature importance"""
        # Use RandomForest feature importance as primary
        rf_importance = self.models['rf'].feature_importances_
        
        # Get XGBoost importance
        xgb_importance = self.models['xgb'].feature_importances_
        
        # Average importances
        avg_importance = (rf_importance + xgb_importance) / 2
        
        # Store top features
        top_indices = np.argsort(avg_importance)[-20:][::-1]
        self.feature_importance = {
            f'feature_{i}': float(avg_importance[i]) 
            for i in top_indices
        }
    
    def _generate_prediction_explanation(self, features: np.ndarray, 
                                       predictions: Dict[str, int],
                                       probabilities: Dict[str, float],
                                       final_prediction: int,
                                       confidence: float) -> Dict[str, Any]:
        """Generate human-readable explanation for prediction"""
        explanation = {
            'prediction': int(final_prediction),
            'confidence': float(confidence),
            'model_votes': predictions,
            'model_confidences': probabilities,
            'regime': self.regime_detector.current_regime,
            'top_features': self.feature_importance,
            'signal_strength': 'strong' if confidence > 0.7 else 'moderate' if confidence > 0.55 else 'weak'
        }
        
        # Add regime-specific insights
        if self.regime_detector.current_regime == "high_volatility":
            explanation['note'] = "High volatility regime - position size should be reduced"
        elif self.regime_detector.current_regime == "trending":
            explanation['note'] = "Trending market - momentum strategies preferred"
            
        return explanation
    
    def update_performance(self, prediction_correct: bool, pnl: float):
        """Update model performance metrics"""
        self.performance_metrics['accuracy'].append(1 if prediction_correct else 0)
        
        # Update other metrics (simplified)
        if len(self.performance_metrics['accuracy']) > 100:
            # Keep only recent history
            for key in self.performance_metrics:
                self.performance_metrics[key] = self.performance_metrics[key][-100:]
    
    def save_model(self, path: Path):
        """Save model and metadata"""
        model_data = {
            'models': self.models,
            'meta_model': self.meta_model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path: Path):
        """Load model and metadata"""
        if path.exists():
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.models = model_data['models']
            self.meta_model = model_data['meta_model']
            self.scaler = model_data['scaler']
            self.feature_importance = model_data.get('feature_importance', {})
            self.training_history = model_data.get('training_history', [])
            
            if model_data.get('last_training_time'):
                self.last_training_time = datetime.fromisoformat(model_data['last_training_time'])
                
            self.is_trained = True

# ============================================================================
# ML DATA STRUCTURES
# ============================================================================

@dataclass
class MLTrainingRecord:
    """Record of training data for consistency"""
    timestamp: datetime
    symbol: str
    features: Dict[str, float]
    label: int
    market_conditions: Dict[str, Any]

@dataclass
class MLPrediction:
    """Structured ML prediction result"""
    signal: int  # 0 or 1
    confidence: float
    reasoning: Dict[str, Any]
    feature_importance: Dict[str, float]
    timestamp: datetime

# ============================================================================
# FEATURE ENGINEERING THAT MATCHES EXISTING TECHNICAL ANALYZER
# ============================================================================

class MLFeatureExtractor:
    """Extract features consistently using same data as TechnicalAnalyzer"""
    
    def __init__(self):
        # Define feature groups for organization and debugging
        self.feature_groups = {
            'price_momentum': ['returns_1', 'returns_5', 'returns_20', 'price_vs_sma20', 'price_vs_sma50'],
            'technical_indicators': ['rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position', 'bb_width'],
            'volume_analysis': ['volume_ratio', 'volume_std_ratio', 'volume_trend'],
            'volatility': ['atr_ratio', 'high_low_ratio', 'std_dev_ratio'],
            'support_resistance': ['pivot_distance', 'r1_distance', 's1_distance'],
            'market_microstructure': ['spread_proxy', 'volume_imbalance']
        }
        
        # Flattened feature list for consistent ordering
        self.feature_names = []
        for group_features in self.feature_groups.values():
            self.feature_names.extend(group_features)
    
    def extract_from_dataframe(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract features from OHLCV DataFrame - matches data available during live trading"""
        features = {}
        
        # Validate input
        if df is None or df.empty or len(df) < 50:
            # Return zero features if insufficient data
            return {name: 0.0 for name in self.feature_names}
        
        try:
            # Ensure numeric types
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any NaN rows
            df = df.dropna()
            
            if len(df) < 50:
                return {name: 0.0 for name in self.feature_names}
            
            # Extract price series
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            volume = df['Volume'].values
            
            # Price momentum features
            features['returns_1'] = (close[-1] - close[-2]) / close[-2] if close[-2] != 0 else 0
            features['returns_5'] = (close[-1] - close[-5]) / close[-5] if close[-5] != 0 else 0
            features['returns_20'] = (close[-1] - close[-20]) / close[-20] if close[-20] != 0 else 0
            features['price_vs_sma20'] = close[-1] / np.mean(close[-20:]) - 1
            features['price_vs_sma50'] = close[-1] / np.mean(close[-50:]) - 1 if len(close) >= 50 else features['price_vs_sma20']
            
            # Technical indicators (matching TechnicalAnalyzer calculations)
            features['rsi'] = self._calculate_rsi(close) / 100.0  # Normalize to 0-1
            
            # MACD
            macd, signal, hist = self._calculate_macd(close)
            features['macd'] = macd / close[-1] if close[-1] != 0 else 0  # Normalize by price
            features['macd_signal'] = signal / close[-1] if close[-1] != 0 else 0
            features['macd_hist'] = hist / close[-1] if close[-1] != 0 else 0
            
            # Bollinger Bands
            bb_middle = np.mean(close[-20:])
            bb_std = np.std(close[-20:])
            bb_upper = bb_middle + 2 * bb_std
            bb_lower = bb_middle - 2 * bb_std
            features['bb_position'] = (close[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10)
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0
            
            # Volume analysis
            features['volume_ratio'] = volume[-1] / (np.mean(volume[-20:]) + 1e-10)
            features['volume_std_ratio'] = np.std(volume[-20:]) / (np.mean(volume[-20:]) + 1e-10)
            features['volume_trend'] = (np.mean(volume[-5:]) - np.mean(volume[-20:])) / (np.mean(volume[-20:]) + 1e-10)
            
            # Volatility features
            atr = self._calculate_atr(high, low, close)
            features['atr_ratio'] = atr / close[-1] if close[-1] != 0 else 0
            features['high_low_ratio'] = (high[-1] - low[-1]) / close[-1] if close[-1] != 0 else 0
            features['std_dev_ratio'] = np.std(close[-20:]) / np.mean(close[-20:])
            
            # Support/Resistance
            pivot = (high[-1] + low[-1] + close[-1]) / 3
            features['pivot_distance'] = (close[-1] - pivot) / close[-1] if close[-1] != 0 else 0
            features['r1_distance'] = ((2 * pivot - low[-1]) - close[-1]) / close[-1] if close[-1] != 0 else 0
            features['s1_distance'] = (close[-1] - (2 * pivot - high[-1])) / close[-1] if close[-1] != 0 else 0
            
            # Market microstructure
            features['spread_proxy'] = (high[-1] - low[-1]) / close[-1] if close[-1] != 0 else 0
            price_direction = np.sign(np.diff(close[-20:]))
            signed_volume = volume[-19:] * price_direction
            features['volume_imbalance'] = np.sum(signed_volume) / (np.sum(volume[-19:]) + 1e-10)
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # Return zero features on error
            features = {name: 0.0 for name in self.feature_names}
        
        # Ensure all features are present and handle NaN/Inf
        for name in self.feature_names:
            if name not in features:
                features[name] = 0.0
            elif np.isnan(features[name]) or np.isinf(features[name]):
                features[name] = 0.0
        
        return features
    
    def extract_from_indicators(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """Extract features from pre-calculated indicators (for compatibility)"""
        features = {}
        
        # Map indicator values to our feature names
        mapping = {
            'rsi': 'rsi',
            'macd': 'macd',
            'macd_signal': 'macd_signal',
            'macd_histogram': 'macd_hist',
            'volume_ratio': 'volume_ratio',
            'atr': 'atr_ratio',
            'high_low_ratio': 'high_low_ratio',
            'returns': 'returns_1'
        }
        
        # Fill in what we can from indicators
        for indicator_name, feature_name in mapping.items():
            if indicator_name in indicators:
                if feature_name == 'rsi':
                    features[feature_name] = indicators[indicator_name] / 100.0
                elif feature_name in ['macd', 'macd_signal', 'macd_hist']:
                    # These should already be normalized in indicators
                    features[feature_name] = indicators[indicator_name]
                else:
                    features[feature_name] = indicators[indicator_name]
        
        # Fill missing features with defaults
        for name in self.feature_names:
            if name not in features:
                features[name] = 0.0
        
        return features
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI - matches TechnicalAnalyzer implementation"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100.0
        
        rs = up / down
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """Calculate MACD - matches TechnicalAnalyzer implementation"""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0
        
        exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        
        return macd.iloc[-1], signal.iloc[-1], hist.iloc[-1]
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate ATR"""
        if len(high) < period:
            return 0.0
        
        tr = np.maximum(high - low, 
                       np.abs(high - np.roll(close, 1)),
                       np.abs(low - np.roll(close, 1)))[1:]
        
        return np.mean(tr[-period:])

# ============================================================================
# PRODUCTION ML MODEL WITH PROPER INTEGRATION
# ============================================================================

class IntegratedMLModel:
    """ML model that integrates seamlessly with existing bot architecture"""
    
    def __init__(self, brain: 'TradingBrain' = None, commentary_system = None):
        self.brain = brain
        self.commentary = commentary_system
        self.feature_extractor = MLFeatureExtractor()
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Model metadata
        self.model_version = "2.0"
        self.last_training_time = None
        self.training_samples = 0
        self.feature_names = self.feature_extractor.feature_names
        self.performance_history = []
        
        # Paths
        self.model_path = Path("ml_model_integrated.pkl")
        self.training_data_path = Path("ml_training_data.json")
        
        # Load existing model
        self._load_model()
    
    def collect_training_data(self, data_provider, symbols: List[str], 
                            lookback_days: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Collect training data from historical market data"""
        if self.commentary:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="📊 Collecting Training Data",
                message=f"Gathering {lookback_days} days of data from {len(symbols)} symbols",
                importance=7
            ))
        
        all_features = []
        all_labels = []
        
        for symbol in symbols:
            try:
                # Get historical data
                df = data_provider.get_market_data(
                    symbol,
                    period_type='month',
                    period=1,
                    frequency_type='minute',
                    frequency=5
                )
                
                if df.empty or len(df) < 100:
                    continue
                
                # Generate training samples using sliding window
                for i in range(50, len(df) - 20):  # Need history and future
                    # Extract features from data up to point i
                    window_data = df.iloc[:i+1]
                    features = self.feature_extractor.extract_from_dataframe(window_data)
                    
                    # Create label: 1 if price increases by 0.3% in next 10 bars
                    current_price = df['Close'].iloc[i]
                    future_price = df['Close'].iloc[i + 10]
                    price_change = (future_price - current_price) / current_price
                    
                    # Label based on whether move was profitable
                    # 3-class labels: 2=BUY, 1=HOLD, 0=SELL
                    if price_change > 0.003:
                        label = 2  # BUY
                    elif price_change < -0.003:
                        label = 0  # SELL (SHORT)
                    else:
                        label = 1  # HOLD
                    
                    # Convert features to array
                    feature_array = [features[name] for name in self.feature_names]
                    all_features.append(feature_array)
                    all_labels.append(label)
                    
                    # Store for brain learning if available
                    if self.brain and label == 1:
                        self.brain.remember_trade(
                            symbol=symbol,
                            pattern="ml_training",
                            outcome="win" if price_change > 0.003 else "loss",
                            pnl_percent=price_change * 100,
                            context={'features': features, 'training': True}
                        )
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
                continue
        
        if self.commentary:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="✅ Training Data Collected",
                message=f"Collected {len(all_features)} training samples",
                importance=6
            ))
        
        return np.array(all_features), np.array(all_labels)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the ML model with proper validation"""
        if len(X) < 100:
            if self.commentary:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=None,
                    title="⚠️ Insufficient Training Data",
                    message=f"Need at least 100 samples, have {len(X)}",
                    importance=8
                ))
            return False
        
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train XGBoost model
            self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',  # Multi-class logloss
            use_label_encoder=False,
            early_stopping_rounds=10,
            num_class=3,  # 3 classes
            objective='multi:softprob'  # Multi-class classification
        )
            
            # Train with validation split
            split_idx = int(0.8 * len(X))
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Calculate validation accuracy
            val_pred = self.model.predict(X_val)
            accuracy = np.mean(val_pred == y_val)
            
            # Update metadata
            self.is_trained = True
            self.last_training_time = datetime.now()
            self.training_samples = len(X)
            
            # Save model
            self._save_model()
            
            if self.commentary:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.MARKET_ANALYSIS,
                    symbol=None,
                    title="✅ Model Training Complete",
                    message=f"Trained on {len(X)} samples\nValidation accuracy: {accuracy:.1%}",
                    data={
                        'samples': len(X),
                        'features': len(self.feature_names),
                        'accuracy': accuracy,
                        'model': 'XGBoost'
                    },
                    confidence=accuracy,
                    importance=8
                ))
            
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            if self.commentary:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=None,
                    title="❌ Training Failed",
                    message=str(e),
                    importance=9
                ))
            return False
    
    def predict(self, df: pd.DataFrame, symbol: str) -> MLPrediction:
        """Make prediction using same data format as training"""
        # Default prediction if not trained
        if not self.is_trained:
            return MLPrediction(
                signal=0,
                confidence=0.5,
                reasoning={'method': 'not_trained'},
                feature_importance={},
                timestamp=datetime.now()
            )
        
        try:
            # Extract features from dataframe
            features = self.feature_extractor.extract_from_dataframe(df)
            
            # Convert to array
            X = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
            
            # Scale
            X_scaled = self.scaler.transform(X)
            
            # Predict
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = float(max(probabilities))
            
            # Get feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance_dict = {
                    name: float(imp) 
                    for name, imp in zip(self.feature_names, self.model.feature_importances_)
                }
                # Get top 5 important features
                top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            else:
                top_features = []
            
            # Generate reasoning
            reasoning = {
                'signal': int(prediction),
                'confidence': confidence,
                'top_features': dict(top_features),
                'rsi': features.get('rsi', 0) * 100,  # Convert back to 0-100
                'volume_ratio': features.get('volume_ratio', 1),
                'bb_position': features.get('bb_position', 0.5)
            }
            # Map XGBoost output back to trading signals
            prediction_map = {0: -1, 1: 0, 2: 1}  # 0->SELL, 1->HOLD, 2->BUY
            mapped_prediction = prediction_map[int(prediction)]
            return MLPrediction(
                signal=int(prediction),
                confidence=confidence,
                reasoning=reasoning,
                feature_importance=dict(top_features) if top_features else {},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return MLPrediction(
                signal=0,
                confidence=0.5,
                reasoning={'error': str(e)},
                feature_importance={},
                timestamp=datetime.now()
            )
    
    def should_retrain(self, performance_metrics: Dict[str, float]) -> bool:
        """Determine if model needs retraining"""
        if not self.is_trained:
            return True
        
        # Time-based retraining
        if self.last_training_time:
            days_since_training = (datetime.now() - self.last_training_time).days
            if days_since_training > 7:  # Weekly retraining
                return True
        
        # Performance-based retraining
        if 'accuracy' in performance_metrics and performance_metrics['accuracy'] < 0.55:
            return True
        
        if 'sharpe_ratio' in performance_metrics and performance_metrics['sharpe_ratio'] < 0.5:
            return True
        
        return False
    
    def _save_model(self):
        """Save model and metadata"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_version': self.model_version,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_samples': self.training_samples,
            'performance_history': self.performance_history[-100:]  # Keep last 100
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def _load_model(self):
        """Load existing model"""
        if not self.model_path.exists():
            return
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Check version compatibility
            if model_data.get('model_version') != self.model_version:
                logger.warning(f"Model version mismatch, retraining needed")
                return
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data.get('feature_names', self.feature_names)
            self.training_samples = model_data.get('training_samples', 0)
            self.performance_history = model_data.get('performance_history', [])
            
            if model_data.get('last_training_time'):
                self.last_training_time = datetime.fromisoformat(model_data['last_training_time'])
            
            self.is_trained = True
            logger.info(f"Loaded ML model v{self.model_version} trained on {self.training_samples} samples")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")

# ============================================================================
# ENHANCED ML PREDICTOR FOR BOT INTEGRATION
# ============================================================================

class MLPredictorWithCommentary:
    """Drop-in replacement for existing ML predictor with proper integration"""
    
    def __init__(self, commentary_system):
        self.commentary = commentary_system
        self.model = None  # Will be set to IntegratedMLModel
        self.brain = None  # Will be set by trading engine
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the integrated ML model"""
        self.model = IntegratedMLModel(
            brain=self.brain,
            commentary_system=self.commentary
        )
        
        # Try to load existing model
        if not self.model.is_trained:
            logger.info("ML model not trained, will use fallback rules until training")
    
    async def predict_with_commentary(self, indicators: Dict[str, float], 
                                    symbol: str, 
                                    market_data: Optional[pd.DataFrame] = None) -> Tuple[int, Dict]:
        """
        Make prediction with detailed explanation
        
        Args:
            indicators: Pre-calculated indicators (for compatibility)
            symbol: Trading symbol
            market_data: Raw OHLCV DataFrame (preferred) or dict with DataFrames
        
        Returns:
            Tuple of (signal, explanation_dict)
        """
        # Handle different input formats
        df = None
        
        if isinstance(market_data, pd.DataFrame):
            # Direct DataFrame - best case
            df = market_data
        elif isinstance(market_data, dict):
            # Dict of DataFrames - extract 5min data
            df = market_data.get('5min', market_data.get('1min'))
        
        # If no market data but have indicators, use fallback
        if df is None or df.empty:
            return self._fallback_prediction(indicators, symbol)
        
        # Make ML prediction
        ml_prediction = self.model.predict(df, symbol)
        
        # Add commentary
        self._add_prediction_commentary(ml_prediction, symbol)
        
        # Convert to expected format
        explanation = {
            'prediction': ml_prediction.signal,
            'confidence': ml_prediction.confidence,
            'method': 'ml_model',
            'signal_strength': self._get_signal_strength(ml_prediction.confidence),
            **ml_prediction.reasoning
        }
        
        return ml_prediction.signal, explanation
    
    def _fallback_prediction(self, indicators: Dict[str, float], symbol: str) -> Tuple[int, Dict]:
        """Fallback prediction using simple rules when ML not available"""
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        
       # Simple rules
        if rsi < 30 and macd > macd_signal:
            signal = 1
            confidence = 0.65
            reasoning = "Oversold with bullish MACD crossover"
        elif rsi > 70 and macd < macd_signal:
            signal = -1  # SELL/SHORT signal
            confidence = 0.65
            reasoning = "Overbought with bearish MACD crossover"
        else:
            signal = 0
            confidence = 0.5
            reasoning = "No clear signal"
        
        if self.commentary:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=symbol,
                title=f"📊 Fallback Analysis: {symbol}",
                message=f"Using rule-based analysis: {reasoning}",
                data={'rsi': rsi, 'macd_vs_signal': macd > macd_signal},
                confidence=confidence,
                importance=5
            ))
        
        return signal, {
            'prediction': signal,
            'confidence': confidence,
            'method': 'fallback',
            'signal_strength': self._get_signal_strength(confidence),
            'reasoning': reasoning
        }
    
    def _add_prediction_commentary(self, prediction: MLPrediction, symbol: str):
        """Add commentary for ML prediction"""
        if not self.commentary:
            return
        
        # Build message
        if prediction.signal == 1:
            action = "BUY"
            emoji = "🟢"
        else:
            action = "HOLD"
            emoji = "⏸️"
        
        # Explain top features
        feature_explanation = []
        for feature, importance in prediction.feature_importance.items():
            if 'rsi' in feature and prediction.reasoning.get('rsi', 50) < 30:
                feature_explanation.append(f"RSI oversold ({prediction.reasoning['rsi']:.0f})")
            elif 'volume_ratio' in feature and prediction.reasoning.get('volume_ratio', 1) > 1.5:
                feature_explanation.append(f"Volume surge ({prediction.reasoning['volume_ratio']:.1f}x)")
            elif 'bb_position' in feature and prediction.reasoning.get('bb_position', 0.5) < 0.2:
                feature_explanation.append("Near lower Bollinger Band")
        
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.MARKET_ANALYSIS,
            symbol=symbol,
            title=f"{emoji} ML Signal: {action} {symbol}",
            message=f"Confidence: {prediction.confidence:.1%}\n" + 
                   f"Key factors: {', '.join(feature_explanation) if feature_explanation else 'Multiple technical factors'}",
            data={
                'signal': prediction.signal,
                'confidence': prediction.confidence,
                'top_features': prediction.feature_importance
            },
            confidence=prediction.confidence,
            importance=7
        ))
    
    def _get_signal_strength(self, confidence: float) -> str:
        """Convert confidence to signal strength"""
        if confidence > 0.75:
            return "strong"
        elif confidence > 0.60:
            return "moderate"
        else:
            return "weak"
    
    async def retrain_model(self, data_provider, symbols: List[str]):
        """Retrain the ML model with recent data"""
        if self.commentary:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="🔧 ML Model Retraining",
                message=f"Starting model retraining with {len(symbols)} symbols",
                importance=8
            ))
        
        # Collect training data
        X, y = self.model.collect_training_data(data_provider, symbols)
        
        if len(X) > 0:
            # Train model
            success = self.model.train(X, y)
            
            if success and self.brain:
                # Record in brain
                self.brain.remember_trade(
                    symbol="MODEL",
                    pattern="retraining",
                    outcome="completed",
                    pnl_percent=0,
                    context={
                        'samples': len(X),
                        'timestamp': datetime.now().isoformat()
                    }
                )
        else:
            if self.commentary:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=None,
                    title="⚠️ Retraining Failed",
                    message="Could not collect sufficient training data",
                    importance=8
                ))
    
    def set_brain(self, brain: 'TradingBrain'):
        """Set reference to brain for learning integration"""
        self.brain = brain
        if self.model:
            self.model.brain = brain


# ============================================================================
# RISK MANAGER WITH COMMENTARY
# ============================================================================

class RiskManagerWithCommentary:
    """Risk manager that explains its decisions"""
    
    def __init__(self, account_balance: float, commentary_system):
        self.account_balance = account_balance
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.positions = {}
        self.max_portfolio_heat = 0.06
        self.margin_call = False
        self.buying_power = account_balance * 0.5
        self.commentary = commentary_system
    
    async def calculate_position_size_with_commentary(self, signal, current_price: float) -> int:
        """Calculate position size with detailed explanation"""
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.RISK_ASSESSMENT,
            symbol=signal.symbol,
            title=f"💰 Calculating Position Size",
            message="Let me determine the appropriate position size based on risk management rules...",
            importance=7
        ))
        
        # Risk per share
        risk_per_share = abs(current_price - signal.stop_loss)
        if risk_per_share == 0:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"⚠️ Invalid Stop Loss",
                message="Stop loss is at the same price as entry. Cannot calculate position size.",
                importance=8
            ))
            return 0
        
        # Maximum risk amount
        max_risk_amount = self.account_balance * Config().MAX_RISK_PER_TRADE
        # Adjust based on recent performance
        if hasattr(self, 'trade_history') and len(self.trade_history) >= 5:
            recent_trades = self.trade_history[-5:]
            wins = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
            win_rate = wins / 5
            
            if win_rate >= 0.8:
                max_risk_amount *= 1.3
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.RISK_ASSESSMENT,
                    symbol=signal.symbol,
                    title=f"🔥 Hot Streak Adjustment",
                    message=f"Recent win rate {win_rate:.0%}, increasing position size",
                    importance=6
                ))
            elif win_rate <= 0.2:
                max_risk_amount *= 0.5
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.RISK_ASSESSMENT,
                    symbol=signal.symbol,
                    title=f"❄️ Cold Streak Protection",
                    message=f"Recent win rate {win_rate:.0%}, reducing position size",
                    importance=6
                ))
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.RISK_ASSESSMENT,
            symbol=signal.symbol,
            title=f"📊 Risk Calculation",
            message=f"With {Config().MAX_RISK_PER_TRADE:.1%} risk per trade on ${self.account_balance:,.0f} account",
            data={
                'max_risk_amount': max_risk_amount,
                'risk_per_share': risk_per_share,
                'stop_distance_percent': (risk_per_share / current_price) * 100
            },
            importance=6
        ))
        
        # Calculate position size
        position_size = int(max_risk_amount / risk_per_share)
        
        # Check against max position value
        position_value = position_size * current_price
        if position_value > Config().MAX_POSITION_VALUE:
            old_size = position_size
            position_size = int(Config().MAX_POSITION_VALUE / current_price)
            
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=signal.symbol,
                title=f"⚠️ Position Size Capped",
                message=f"Reduced from {old_size} to {position_size} shares (max ${Config().MAX_POSITION_VALUE})",
                importance=7
            ))
        
        # Check buying power
        if position_value > self.buying_power:
            old_size = position_size
            position_size = int(self.buying_power * 0.95 / current_price)
            
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=signal.symbol,
                title=f"⚠️ Position Size Adjusted",
                message=f"Reduced position from {old_size} to {position_size} shares due to buying power constraints",
                data={
                    'required_capital': old_size * current_price,
                    'available_buying_power': self.buying_power,
                    'adjusted_capital': position_size * current_price
                },
                importance=7
            ))
        
        # Ensure minimum position size
        if position_size < Config().MIN_POSITION_SIZE:
            position_size = 0
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"❌ Position Too Small",
                message=f"Calculated size below minimum ({Config().MIN_POSITION_SIZE} shares)",
                importance=8
            ))
        
        # Final position size
        if position_size > 0:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=signal.symbol,
                title=f"✅ Position Size Determined",
                message=f"Will trade {position_size} shares, risking ${position_size * risk_per_share:.2f}",
                data={
                    'shares': position_size,
                    'total_value': position_size * current_price,
                    'max_loss': position_size * risk_per_share,
                    'max_gain': position_size * (signal.take_profit - current_price),
                    'risk_reward_ratio': (signal.take_profit - current_price) / risk_per_share
                },
                confidence=0.9,
                importance=8
            ))
        
        return position_size
    
    def check_trading_allowed(self) -> Tuple[bool, str]:
        """Check if trading is allowed"""
        if self.margin_call:
            return False, "Margin call active - resolve before trading"
        
        if self.buying_power < 100:
            return False, f"Insufficient buying power: ${self.buying_power:.2f}"
        
        if self.daily_pnl <= -Config().MAX_DAILY_LOSS * self.account_balance:
            return False, "Daily loss limit exceeded"
        
        if self.consecutive_losses >= Config().MAX_CONSECUTIVE_LOSSES:
            return False, "Max consecutive losses reached"
        
        return True, "Trading allowed"

# ============================================================================
# TRADING STRATEGIES WITH COMMENTARY
# ============================================================================

class TradingStrategyWithCommentary(ABC):
    """Base class for strategies with commentary"""
    
    def __init__(self, commentary_system):
        self.commentary = commentary_system
    
    @abstractmethod
    async def generate_signal_with_commentary(self, market_data) -> Optional[TradingSignal]:
        pass

class BreakoutStrategyWithCommentary(TradingStrategyWithCommentary):
    """Breakout strategy with detailed explanations"""
    
    async def generate_signal_with_commentary(self, market_data) -> Optional[TradingSignal]:
        indicators = market_data.indicators
        
        try:
            # Check for breakout with proper type conversion
            resistance_1 = float(indicators.get('resistance_1', 0))
            support_1 = float(indicators.get('support_1', 0))
            volume_ratio = float(indicators.get('volume_ratio', 1))
            
            # Validate values
            if np.isnan(resistance_1) or resistance_1 <= 0 or np.isnan(support_1) or support_1 <= 0:
                return None
            
            if resistance_1 > 0 and market_data.close > resistance_1:
                # Check volume confirmation
                volume_surge = market_data.volume > volume_ratio * 1.5
                
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.OPPORTUNITY,
                    symbol=market_data.symbol,
                    title=f"🚀 Breakout Detected!",
                    message=f"Price broke above resistance at ${resistance_1:.2f} {symbol}. "
                           f"{'Volume confirms breakout!' if volume_surge else 'Volume is average.'}",
                    data={
                        'breakout_level': resistance_1,
                        'current_price': market_data.close,
                        'volume_surge': volume_surge,
                        'distance_from_resistance': ((market_data.close - resistance_1) / resistance_1) * 100
                    },
                    confidence=0.8 if volume_surge else 0.6,
                    importance=8
                ))
                
                # Calculate targets
                atr = indicators.get('atr', market_data.close * 0.02)
                stop_loss = market_data.close - (2 * atr)
                take_profit = market_data.close + 2 * (market_data.close - stop_loss)
                
                return TradingSignal(
                    symbol=market_data.symbol,
                    signal_type=SignalType.BUY,
                    strength=0.8,
                    entry_price=market_data.close,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=0,
                    reasoning={
                        'strategy': 'breakout',
                        'breakout_level': resistance_1,
                        'volume_confirmation': volume_surge
                    },
                    confidence=0.75
                )
        except Exception as e:
            logger.debug(f"Breakout strategy error for {market_data.symbol}: {e}")
        
        if indicators.get('volume_ratio', 1) < 1.5:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.TECHNICAL,
                symbol=market_data.symbol,
                title=f"📊 Low Volume",
                message="Volume too low for reliable entry",
                importance=5
            ))
        
        return None

class MeanReversionStrategyWithCommentary(TradingStrategyWithCommentary):
    """Mean reversion strategy with explanations"""
    
    async def generate_signal_with_commentary(self, market_data) -> Optional[TradingSignal]:
        indicators = market_data.indicators
        
        try:
            # Check for oversold conditions
            rsi = float(indicators.get('rsi', 50))
            bb_lower = float(indicators.get('bb_lower', 0))
            bb_middle = float(indicators.get('bb_middle', market_data.close))
            
            # Validate values
            if np.isnan(rsi) or np.isnan(bb_lower) or bb_lower <= 0:
                return None
            
            if rsi < 30 and market_data.close < bb_lower:
                distance_from_mean = ((bb_middle - market_data.close) / market_data.close) * 100
                
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.OPPORTUNITY,
                    symbol=market_data.symbol,
                    title=f"🔄 Mean Reversion Setup",
                    message=f"Oversold conditions detected - RSI at {rsi:.1f} and price below Bollinger Band. "
                           f"Price is {distance_from_mean:.1f}% below the mean.",
                    data={
                        'rsi': rsi,
                        'bollinger_position': 'below_lower_band',
                        'distance_from_mean': distance_from_mean,
                        'bb_lower': bb_lower,
                        'bb_middle': bb_middle
                    },
                    confidence=0.65,
                    importance=7
                ))
                
                stop_loss = market_data.close * 0.98
                take_profit = bb_middle
                
                return TradingSignal(
                    symbol=market_data.symbol,
                    signal_type=SignalType.BUY,
                    strength=0.7,
                    entry_price=market_data.close,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=0,
                    reasoning={
                        'strategy': 'mean_reversion',
                        'rsi': rsi,
                        'bb_position': 'below_lower_band',
                        'distance_from_mean': distance_from_mean
                    },
                    confidence=0.65
                )
        # Add short signal for overbought conditions
            elif rsi > 70 and market_data.close > bb_upper:
                distance_from_mean = ((market_data.close - bb_middle) / market_data.close) * 100
                
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.OPPORTUNITY,
                    symbol=market_data.symbol,
                    title=f"🔻 Short Setup - Mean Reversion",
                    message=f"Overbought conditions - RSI at {rsi:.1f} and price above upper Bollinger Band. "
                        f"Price is {distance_from_mean:.1f}% above the mean.",
                    data={
                        'rsi': rsi,
                        'bollinger_position': 'above_upper_band',
                        'distance_from_mean': distance_from_mean
                    },
                    confidence=0.65,
                    importance=7
                ))
                
                stop_loss = market_data.close * 1.02  # 2% above entry
                take_profit = bb_middle
                
                return TradingSignal(
                    symbol=market_data.symbol,
                    signal_type=SignalType.SELL,  # SELL = SHORT
                    strength=0.7,
                    entry_price=market_data.close,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=0,
                    reasoning={
                        'strategy': 'mean_reversion_short',
                        'rsi': rsi,
                        'bb_position': 'above_upper_band',
                        'distance_from_mean': distance_from_mean
                    },
                    confidence=0.65
                )
        except Exception as e:
            logger.debug(f"Mean reversion strategy error for {market_data.symbol}: {e}")
        
        return None

class MomentumStrategyWithCommentary(TradingStrategyWithCommentary):
    """Momentum strategy with explanations"""
    
    async def generate_signal_with_commentary(self, market_data) -> Optional[TradingSignal]:
        indicators = market_data.indicators
        
        try:
            # Check for momentum with proper type conversion
            macd = float(indicators.get('macd', 0))
            macd_signal = float(indicators.get('macd_signal', 0))
            rsi = float(indicators.get('rsi', 50))
            adx = float(indicators.get('adx', 0))
            
            # Validate values
            if np.isnan(macd) or np.isnan(macd_signal) or np.isnan(rsi) or np.isnan(adx):
                return None
            
            if macd > macd_signal and 50 < rsi < 70 and adx > 25:
                
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.OPPORTUNITY,
                    symbol=market_data.symbol,
                    title=f"📈 Momentum Building",
                    message=f"Strong momentum detected: MACD bullish crossover, RSI at {rsi:.1f} (healthy), "
                           f"ADX at {adx:.1f} (strong trend)",
                    data={
                        'macd_crossover': True,
                        'macd': macd,
                        'macd_signal': macd_signal,
                        'rsi': rsi,
                        'adx': adx,
                        'trend_strength': 'strong' if adx > 30 else 'moderate'
                    },
                    confidence=0.7,
                    importance=7
                ))
                
                stop_loss = market_data.close * 0.97
                take_profit = market_data.close * 1.06
                
                return TradingSignal(
                    symbol=market_data.symbol,
                    signal_type=SignalType.BUY,
                    strength=0.75,
                    entry_price=market_data.close,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=0,
                    reasoning={
                        'strategy': 'momentum',
                        'macd_bullish': True,
                        'rsi_healthy': True,
                        'trend_strong': adx > 25
                    },
                    confidence=0.7
                )
        except Exception as e:
            logger.debug(f"Momentum strategy error for {market_data.symbol}: {e}")
        
        return None

# ============================================================================
# ENHANCED TRADING ENGINE WITH COMMENTARY
# ============================================================================

class TradingEngineWithCommentary:
    """Trading engine that explains its decisions in real-time"""
    
    def __init__(self, mode: TradingMode = TradingMode.SIMULATION_WITH_COMMENTARY, connection_manager=None):
        self.mode = mode
        self.schwab_client = None
        self.data_provider = None
        self.positions = {}
        self.simulated_positions = {}
        self.trade_history = []
        self.is_running = False
        self.account_id = None
        self.account_hash = None
        self._last_bp_check = datetime.now() - timedelta(minutes=5)  # Force initial check
        # Commentary system
        self.commentary = CommentarySystem()
        # Initialize brain BEFORE ml_predictor
        self.brain = TradingBrain()
        # Initialize components
        self._init_schwab_client()
        # Order tracking
        self.pending_orders = {}
        self.order_id_to_symbol = {}
        self.last_order_check = datetime.now()
        
        # Risk manager
        self.risk_manager = RiskManagerWithCommentary(
            account_balance=100000,
            commentary_system=self.commentary
        )
        
        # Technical analyzer
        self.technical_analyzer = TechnicalAnalyzerWithCommentary(
            commentary_system=self.commentary
        )
        # ML predictor
        self.ml_predictor = MLPredictorWithCommentary(
            commentary_system=self.commentary
        )
        self.ml_predictor.set_brain(self.brain)
        # Stock screener
        self.screener = StockScreener(
            self.schwab_client,
            self.commentary
        ) if self.schwab_client else None
        
        # Dynamic watchlist
        self.dynamic_watchlist = ['NVDA', 'TSLA', 'PLTR']  # Default symbols
        self.last_screener_run = None
        # Trading strategies
        self.strategies = [
            BreakoutStrategyWithCommentary(self.commentary),
            MeanReversionStrategyWithCommentary(self.commentary),
            MomentumStrategyWithCommentary(self.commentary)
        ]
        self.strategies.append(FreeNewsSignalStrategy(self.commentary))
        
	    # Initialize brain and exit manager
        self.brain = TradingBrain()
        self.exit_manager = DynamicExitManager(self.brain, self.commentary)

	    # Load previous state
        self._load_state()

        # Connection manager for WebSocket
        self.connection_manager = connection_manager or ConnectionManager()
        # Confirmation settings
        self.require_confirmations = Config().REQUIRE_CLOSE_CONFIRMATION
        self.confirm_only_losses = Config().CONFIRM_ONLY_LOSSES
        self.confirm_threshold_percent = Config().CONFIRM_THRESHOLD_PERCENT
        self.manual_close_only = True  # NEW: Only close positions manually

        # Market state tracking
        self.market_state = {
            'vix': 0,
            'market_trend': 'neutral',
            'breadth': {},
            'economic_events': []
        }
        
        # Enhanced error recovery and circuit breakers
        self.circuit_breaker = CircuitBreaker()
        self.error_recovery = ErrorRecovery()
        self.error_counts = {}
        self.last_error_time = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        
        # Performance tracking
        self.performance_analyzer = PerformanceAnalyzer(config_manager.config)
        
        # Advanced components integration
        self.behavioral_analyzer = BehavioralAnalyzer(self.commentary)
        self.data_validator = DataValidator(self.commentary)
        self.advanced_exit_manager = AdvancedExitManager(self.commentary)
        self.alternative_data_integrator = AlternativeDataIntegrator(self.commentary)
        self.market_neutral_strategies = MarketNeutralStrategies(self.commentary)
        
        # Add initial commentary
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.MARKET_ANALYSIS,
            symbol=None,
            title="🚀 Trading Bot Initialized",
            message=f"Starting in {mode.value} mode with enhanced error recovery and circuit breakers. I'll explain my thinking process as I analyze markets.",
            importance=10
        ))
        self._cached_buying_power = 0
        self._last_bp_check = datetime.now()
        self.day_trades_count = 0
        self.performance_metrics = {
            'slippage': [],
            'fill_rate': {'attempts': 0, 'fills': 0}
        }
    # Add to rest brain - caution state:
    def reset_brain_state(self):
        """Reset brain to neutral confident state"""
        self.brain.emotional_state = {
            'confidence': 0.7,
            'risk_appetite': 0.6,
            'patience': 0.8,
            'fear_level': 0.2,
            'greed_level': 0.3,
            'last_update': datetime.now().isoformat()
        }
        # Clear recent bad memories
        self.brain.memories = self.brain.memories[:-10] if len(self.brain.memories) > 10 else []
        self.brain.consecutive_losses = 0
        self.brain.save_memories()
        
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.PSYCHOLOGY,
            symbol=None,
            title="🧠 Brain Reset",
            message="Cleared recent negative memories. Fresh start!",
            importance=8
        ))

    def _load_state(self):
        """Load previous trading state"""
        state_file = Path("trading_state.json")
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.trade_history = state.get('trade_history', [])
                    self.risk_manager.daily_pnl = state.get('daily_pnl', 0)
                    self.risk_manager.consecutive_losses = state.get('consecutive_losses', 0)
                    
                    # Add human-like morning routine commentary
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.PSYCHOLOGY,
                        symbol=None,
                        title="☕ Good Morning!",
                        message=f"Back at the desk. Yesterday's P&L: ${state.get('daily_pnl', 0):.2f}. "
                               f"Let's see what the market has for us today.",
                        importance=8
                    ))
            except Exception as e:
                logger.error(f"Error loading state: {e}")
    
    async def _execute_real_trade(self, signal) -> bool:
        """Execute real trade through Schwab"""
        
        if not self.schwab_client or not self.account_id:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"❌ Trade Execution Failed",
                message="Schwab client not properly initialized",
                importance=10
            ))
            return False
        
        try:
            # Validate OCO prices first
            if not self._validate_oco_prices(signal.symbol, signal.stop_loss, signal.take_profit):
                return False
            
            # Check buying power
            quote = self.data_provider.get_quote(signal.symbol)
            price = quote.get('ask', signal.entry_price) if signal.signal_type == SignalType.BUY else quote.get('bid', signal.entry_price)
            required = signal.position_size * price
                                    
            from schwab.orders.equities import equity_buy_market, equity_sell_market, equity_sell_short_market
            from schwab.orders.common import Duration, Session
            
            # Build order based on signal type
            if signal.signal_type == SignalType.BUY:
                order_builder = equity_buy_market(signal.symbol, signal.position_size)
            else:  # SHORT
                try:
                    order_builder = equity_sell_short_market(signal.symbol, signal.position_size)
                except AttributeError:
                    # Fallback if short function not available
                    order_builder = equity_sell_market(signal.symbol, signal.position_size)
                    order_builder.set_instruction('SELL_SHORT')
            
            # Set order parameters
            order_builder.set_duration(Duration.DAY)
            order_builder.set_session(Session.NORMAL)
            
            # Build the order
            order = order_builder.build()
            
            # Place the order
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=signal.symbol,
                title=f"📤 Placing Live Order",
                message=f"Sending {'BUY' if signal.signal_type == SignalType.BUY else 'SELL'} order for {signal.position_size} shares",
                importance=9
            ))
            
            response = self.schwab_client.place_order(self.account_hash, order)
            
            if response.status_code in [200, 201]:
                # Extract order ID from Location header
                order_id = response.headers.get('Location', '').split('/')[-1]
                
                # Track pending order
                self.pending_orders[order_id] = {
                    'symbol': signal.symbol,
                    'signal': signal,
                    'status': 'PENDING',
                    'placed_time': datetime.now()
                }
                self.order_id_to_symbol[order_id] = signal.symbol
                
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=signal.symbol,
                    title=f"✅ Order Submitted",
                    message=f"Order ID: {order_id} - Waiting for fill confirmation",
                    data={
                        'order_id': order_id,
                        'quantity': signal.position_size,
                        'expected_price': signal.entry_price
                    },
                    importance=9
                ))

                # Wait a moment and check order status immediately
                await asyncio.sleep(2)
                # Wait for fill verification
                if await self._verify_order_fill(order_id, signal):
                    await self._place_bracket_orders(signal, order_id)
                    return True
                else:
                    return False                
            else:
                # Parse rejection reason
                rejection = self._parse_order_rejection(response)
                
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"❌ Order Rejected: {rejection['reason']}",
                    message=rejection['message'],
                    data={'details': rejection['details']},
                    importance=10
                ))
                return False
                
        except Exception as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"❌ Order Exception",
                message=str(e),
                importance=10
            ))
            logger.error(f"Order execution error: {e}", exc_info=True)
            return False
    # Add these methods after _execute_real_trade in TradingEngineWithCommentary class:

    async def _check_existing_orders(self, symbol: str) -> Dict[str, Any]:
        """Check for existing orders on a symbol"""
        try:
            response = self.schwab_client.get_orders_for_account(
                self.account_hash,
                from_entered_datetime=datetime.now() - timedelta(days=1),
                status=self.schwab_client.Order.Status.WORKING  # Only active orders
            )
            
            if response.status_code == 200:
                orders = response.json()
                symbol_orders = []
                
                for order in orders:
                    # Check main order
                    order_symbol = self._extract_symbol_from_order(order)
                    if order_symbol == symbol:
                        symbol_orders.append(order)
                    
                    # Check child orders (OCO)
                    for child in order.get('childOrderStrategies', []):
                        child_symbol = self._extract_symbol_from_order(child)
                        if child_symbol == symbol:
                            symbol_orders.append(order)
                            break
                
                return {
                    'has_orders': len(symbol_orders) > 0,
                    'orders': symbol_orders,
                    'count': len(symbol_orders)
                }
        except Exception as e:
            logger.error(f"Error checking orders: {e}")
        
        return {'has_orders': False, 'orders': [], 'count': 0}

    def _extract_symbol_from_order(self, order: Dict) -> Optional[str]:
        """Extract symbol from order structure"""
        try:
            legs = order.get('orderLegCollection', [])
            if legs:
                return legs[0].get('instrument', {}).get('symbol')
        except:
            pass
        return None

    async def _cancel_existing_orders(self, symbol: str) -> bool:
        """Cancel existing orders for a symbol"""
        try:
            existing = await self._check_existing_orders(symbol)
            
            if existing['has_orders']:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=symbol,
                    title=f"🔄 Canceling Existing Orders",
                    message=f"Found {existing['count']} existing orders for {symbol}",
                    importance=7
                ))
                
                for order in existing['orders']:
                    order_id = order.get('orderId')
                    if order_id:
                        try:
                            response = self.schwab_client.cancel_order(self.account_id, order_id)
                            if response.status_code in [200, 201, 202]:
                                logger.info(f"Canceled order {order_id}")
                        except Exception as e:
                            logger.error(f"Failed to cancel order {order_id}: {e}")
                
                # Wait for cancellations to process
                await asyncio.sleep(1)
                return True
        
        except Exception as e:
            logger.error(f"Error canceling orders: {e}")
        
        return False
    '''
    async def _check_buying_power(self, required_amount: float) -> Tuple[bool, float]:
        """Check if we have sufficient buying power"""
        try:
            if hasattr(self, '_cached_buying_power'):
                # Use cached value if recent (within 30 seconds)
                if (datetime.now() - self._last_bp_check).total_seconds() < 30:
                    return self._cached_buying_power >= required_amount, self._cached_buying_power
            
            response = self.schwab_client.get_account(self.account_hash)
            logger.info(f" Checking Buying Power: {response.json()}")
            if response.status_code == 200:
                data = response.json()
                balances = data.get('securitiesAccount', {}).get('currentBalances', {})
                
                # Get appropriate buying power based on account type
                buying_power = balances.get('buyingPower', 0)
                
                # For margin accounts
                if 'dayTradingBuyingPower' in balances:
                    buying_power = min(buying_power, balances['dayTradingBuyingPower'])
                
                # Cache the result
                self._cached_buying_power = buying_power
                self._last_bp_check = datetime.now()
                
                # Update risk manager
                self.risk_manager.buying_power = buying_power
                
                return buying_power >= required_amount, buying_power
                
        except Exception as e:
            logger.error(f"Error checking buying power: {e}")
        
        return False, 0
    '''
    async def _check_buying_power(self, required_amount: float) -> Tuple[bool, float]:
        """Check if we have sufficient buying power"""
        try:
            # Check if we should use cache
            cache_age = (datetime.now() - self._last_bp_check).total_seconds()
        
            # Only use cache if we have a valid cached value AND it's recent
            if self._cached_buying_power > 0 and cache_age < 30:
                logger.debug(f"Using cached buying power: ${self._cached_buying_power:,.2f} (age: {cache_age:.1f}s)")
                return self._cached_buying_power >= required_amount, self._cached_buying_power
            
            # Fetch fresh data
            response = self.schwab_client.get_account(self.account_hash)
            logger.info(f" Buying power check: {response.json()}")
            if response.status_code != 200:
                logger.error(f"Failed to get account data: {response.status_code}")
                return False, 0
            
            data = response.json()
            
            # Log the structure for debugging
            logger.debug(f"Account data keys: {list(data.keys())}")
            
            # Navigate to the correct location
            account_data = data.get('securitiesAccount', {})
            
            # Try multiple possible locations for buying power
            buying_power = 0
            
            # Method 1: Check currentBalances
            if 'currentBalances' in account_data:
                balances = account_data['currentBalances']
                logger.debug(f"Balance fields available: {list(balances.keys())}")
                
                # Try different field names
                buying_power = (
                    balances.get('buyingPower', 0) or
                    balances.get('availableFunds', 0) or
                    balances.get('availableFundsNonMarginableTrade', 0) or
                    balances.get('buyingPowerNonMarginableTrade', 0)
                )
                
                # For margin accounts, check day trading buying power
                if 'dayTradingBuyingPower' in balances:
                    day_trading_bp = balances.get('dayTradingBuyingPower', 0)
                    if day_trading_bp > 0:
                        buying_power = min(buying_power, day_trading_bp)
            
            # Method 2: Check projectedBalances if currentBalances didn't work
            if buying_power == 0 and 'projectedBalances' in account_data:
                projected = account_data['projectedBalances']
                buying_power = (
                    projected.get('buyingPower', 0) or
                    projected.get('availableFunds', 0) or
                    projected.get('availableFundsNonMarginableTrade', 0)
                )
            
            # Method 3: Calculate from cash and margin if still 0
            if buying_power == 0 and 'currentBalances' in account_data:
                balances = account_data['currentBalances']
                cash = balances.get('cashBalance', 0) or balances.get('totalCash', 0)
                margin = balances.get('marginBalance', 0)
                
                # For cash accounts, buying power is usually just cash
                # For margin accounts, it could be cash + margin available
                account_type = account_data.get('type', 'CASH')
                if account_type == 'MARGIN':
                    buying_power = cash + margin
                else:
                    buying_power = cash
            
            # Log what we found
            logger.info(f"Buying power found: ${buying_power:,.2f} (required: ${required_amount:,.2f})")
            
            # Cache the result
            self._cached_buying_power = buying_power
            self._last_bp_check = datetime.now()
            
            # Update risk manager
            self.risk_manager.buying_power = buying_power
            
            return buying_power >= required_amount, buying_power
            
        except Exception as e:
            logger.error(f"Error checking buying power: {e}", exc_info=True)
            # Return cached value if available
            if hasattr(self, '_cached_buying_power') and self._cached_buying_power > 0:
                logger.warning(f"Using last known buying power: ${self._cached_buying_power:,.2f}")
                return self._cached_buying_power >= required_amount, self._cached_buying_power
        
        return False, 0
    def _parse_order_rejection(self, response) -> Dict[str, Any]:
        """Parse Schwab order rejection reasons"""
        try:
            if hasattr(response, 'text'):
                error_data = json.loads(response.text)
                error_message = str(error_data).lower()
                
                # Common rejection reasons
                if 'buying power' in error_message or 'insufficient funds' in error_message:
                    return {
                        'reason': 'INSUFFICIENT_FUNDS',
                        'message': 'Not enough buying power for this trade',
                        'details': error_data
                    }
                elif 'duplicate' in error_message:
                    return {
                        'reason': 'DUPLICATE_ORDER',
                        'message': 'A similar order already exists for this symbol',
                        'details': error_data
                    }
                elif 'market closed' in error_message or 'outside hours' in error_message:
                    return {
                        'reason': 'MARKET_CLOSED',
                        'message': 'Cannot place order - market is closed',
                        'details': error_data
                    }
                elif 'position' in error_message and 'exist' in error_message:
                    return {
                        'reason': 'POSITION_EXISTS',
                        'message': 'Position already exists in this symbol',
                        'details': error_data
                    }
                elif 'halted' in error_message:
                    return {
                        'reason': 'SYMBOL_HALTED',
                        'message': 'Trading is halted for this symbol',
                        'details': error_data
                    }
                elif 'invalid' in error_message and 'price' in error_message:
                    return {
                        'reason': 'INVALID_PRICE',
                        'message': 'Order price is invalid (check stop/limit prices)',
                        'details': error_data
                    }
        except:
            pass
        
        return {
            'reason': 'UNKNOWN',
            'message': f'Order rejected - Status: {response.status_code}',
            'details': response.text if hasattr(response, 'text') else None
        }

    def _validate_oco_prices(self, symbol: str, stop_price: float, take_profit: float) -> bool:
        """Validate OCO order prices before submission"""
        try:
            # Get current quote
            quote = self.data_provider.get_quote(symbol)
            if not quote:
                return True  # Skip validation if no quote
            
            current_price = quote.get('last', 0)
            bid = quote.get('bid', current_price)
            
            # For sell orders: stop must be below current bid
            if stop_price >= bid:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title=f"⚠️ Invalid Stop Price",
                    message=f"Stop ${stop_price:.2f} must be below bid ${bid:.2f}",
                    importance=8
                ))
                return False
            
            # Take profit should be above current price
            if take_profit <= current_price:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title=f"⚠️ Invalid Target Price",
                    message=f"Target ${take_profit:.2f} should be above current ${current_price:.2f}",
                    importance=8
                ))
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Price validation error: {e}")
            return True  # Allow order if validation fails
    
    async def _place_bracket_orders(self, signal, parent_order_id: str):
        """Place stop loss and take profit orders as OCO"""
        try:
            from schwab.orders.common import one_cancels_other, Duration, Session, OrderType
            from schwab.orders.equities import equity_sell_limit
            
            # Create take profit order (limit sell)
            take_profit_order = equity_sell_limit(
                signal.symbol, 
                signal.position_size, 
                signal.take_profit
            ).set_duration(Duration.GOOD_TILL_CANCEL).set_session(Session.NORMAL)
            
            # Create stop loss order (stop limit - using stop price slightly below limit)
            # For stop limit, we set both stop price and limit price
            stop_loss_order = (equity_sell_limit(
                signal.symbol,
                signal.position_size,
                signal.stop_loss * 0.995  # Limit price slightly below stop
            ).set_order_type(OrderType.STOP_LIMIT)
            .set_stop_price(signal.stop_loss)
            .set_duration(Duration.GOOD_TILL_CANCEL)
            .set_session(Session.NORMAL))
            
            # Create OCO order using helper function
            oco_order = one_cancels_other(take_profit_order, stop_loss_order)
            
            # Build and place the OCO order
            response = self.schwab_client.place_order(self.account_hash, oco_order.build())
            
            if response.status_code in [200, 201]:
                # Extract order ID from response
                order_id = response.headers.get('Location', '').split('/')[-1]
                
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.RISK_ASSESSMENT,
                    symbol=signal.symbol,
                    title=f"✅ OCO Bracket Order Placed",
                    message=f"Stop Loss: ${signal.stop_loss:.2f} | Take Profit: ${signal.take_profit:.2f}",
                    data={
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'order_type': 'OCO',
                        'order_id': order_id
                    },
                    importance=7
                ))
            else:
                rejection = self._parse_order_rejection(response)
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"⚠️ OCO Order Failed: {rejection['reason']}",
                    message=rejection['message'],
                    data={'details': rejection['details']},
                    importance=7
                ))
                
        except Exception as e:
            logger.error(f"Bracket order error: {e}", exc_info=True)
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"⚠️ Bracket Order Error", 
                message=f"Could not place stop/target orders: {str(e)}",
                importance=7
            ))
    def _validate_oco_prices(self, symbol: str, stop_price: float, take_profit: float) -> bool:
        """Validate OCO order prices before submission"""
        try:
            # Get current quote
            quote = self.data_provider.get_quote(symbol)
            if not quote:
                return True  # Skip validation if no quote
            
            current_price = quote.get('last', 0)
            bid = quote.get('bid', current_price)
            
            # For sell orders: stop must be below current bid
            if stop_price >= bid:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title=f"⚠️ Invalid Stop Price",
                    message=f"Stop ${stop_price:.2f} must be below bid ${bid:.2f}",
                    importance=8
                ))
                return False
            
            # Take profit should be above current price
            if take_profit <= current_price:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title=f"⚠️ Invalid Target Price",
                    message=f"Target ${take_profit:.2f} should be above current ${current_price:.2f}",
                    importance=8
                ))
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Price validation error: {e}")
            return True  # Allow order if validation fails
    async def _check_specific_order_status(self, order_id: str) -> str:
        """Check status of a specific order immediately"""
        try:
            response = self.schwab_client.get_order(self.account_hash, order_id)
            
            if response.status_code == 200:
                order_info = response.json()
                status = order_info.get('status', 'UNKNOWN')
                
                if order_id in self.pending_orders:
                    self.pending_orders[order_id]['status'] = status
                    
                    if status == 'REJECTED':
                        symbol = self.pending_orders[order_id]['symbol']
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.WARNING,
                            symbol=symbol,
                            title=f"❌ Order Rejected by Broker",
                            message=f"Order {order_id} was rejected. Not proceeding with position.",
                            data={'order_info': order_info},
                            importance=10
                        ))
                        # Remove from tracking
                        del self.pending_orders[order_id]
                        if order_id in self.order_id_to_symbol:
                            del self.order_id_to_symbol[order_id]
                    
                    elif status == 'FILLED':
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.DECISION,
                            symbol=self.pending_orders[order_id]['symbol'],
                            title=f"✅ Order Filled",
                            message=f"Order {order_id} has been filled",
                            importance=8
                        ))
                
                return status
        except Exception as e:
            logger.error(f"Error checking order {order_id}: {e}")
            return 'UNKNOWN'
        
    async def _verify_order_fill(self, order_id: str, signal, max_retries: int = 3) -> bool:
        """Verify order filled with retries"""
        for attempt in range(max_retries):
            status = await self._check_specific_order_status(order_id)
            if status == 'FILLED':
                return True
            elif status in ['REJECTED', 'CANCELED']:
                return False
            await asyncio.sleep(2)
        
        # Cancel unfilled order
        try:
            self.schwab_client.cancel_order(self.account_hash, order_id)
            del self.pending_orders[order_id]
        except:
            pass
        return False
    
    async def _check_order_status(self):
        """Check status of pending orders"""
        if not self.pending_orders or not self.schwab_client:
            return
        
        # Check orders every 10 seconds
        if (datetime.now() - self.last_order_check).total_seconds() < 10:
            return
        
        self.last_order_check = datetime.now()
        
        for order_id, order_data in list(self.pending_orders.items()):
            try:
                response = self.schwab_client.get_order(
                    self.account_hash,
                    order_id
                    #fields=[schwab_client.Orders.Fields.EXECUTION_LEGS]
                )
                
                if response.status_code == 200:
                    order_info = response.json()
                    logger.info(f"Existing Orders : {order_info}")
                    status = order_info.get('status', 'UNKNOWN')
                    
                    if status == 'FILLED':
                        # Get fill price
                        fill_price = signal.entry_price  # Default
                        if 'orderActivityCollection' in order_info:
                            for activity in order_info['orderActivityCollection']:
                                if activity.get('executionType') == 'FILL':
                                    legs = activity.get('executionLegs', [])
                                    if legs:
                                        fill_price = legs[0].get('price', signal.entry_price)
                                        expected = order_data['signal'].entry_price
                                        slippage = abs(fill_price - expected) / expected
                                        self.performance_metrics['slippage'].append(slippage)
                        # Create position
                        signal = order_data['signal']
                        position = Position(
                            symbol=signal.symbol,
                            entry_price=fill_price,
                            current_price=fill_price,
                            quantity=signal.position_size,
                            side='long' if signal.signal_type == SignalType.BUY else 'short',
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            entry_time=datetime.now(),
                            reasoning=signal.reasoning
                        )
                        self.positions[signal.symbol] = position
                        
                        # Initialize exit tracking
                        self.exit_manager.initialize_position_tracking(
                            signal.symbol,
                            fill_price,
                            signal.stop_loss,
                            signal.take_profit
                        )
                        
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.DECISION,
                            symbol=signal.symbol,
                            title=f"💰 Order Filled!",
                            message=f"Position opened at ${fill_price:.2f}",
                            data={
                                'order_id': order_id,
                                'fill_price': fill_price,
                                'quantity': signal.position_size
                            },
                            importance=9
                        ))
                        
                        del self.pending_orders[order_id]
                        
                    elif status in ['CANCELED', 'REJECTED', 'EXPIRED']:
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.WARNING,
                            symbol=order_data['symbol'],
                            title=f"❌ Order {status}",
                            message=f"Order {order_id} was {status.lower()}",
                            importance=8
                        ))
                        del self.pending_orders[order_id]
                        
            except Exception as e:
                logger.error(f"Error checking order {order_id}: {e}")
    
    async def _close_real_position(self, position, exit_portion: float = 1.0):
        """Close real position through Schwab"""
        if not self.schwab_client:
            return False
        
        try:
            from schwab.orders.equities import equity_sell_market
            from schwab.orders.common import Duration, Session
            
            quantity_to_sell = int(position.quantity * exit_portion)
            
            # Create market sell order
            from schwab.orders.equities import equity_sell_market, equity_buy_market, equity_buy_to_cover_market
            from schwab.orders.common import Duration, Session

            quantity_to_sell = int(position.quantity * exit_portion)

            # Create appropriate order based on position side
            if position.side == 'long':
                order_builder = equity_sell_market(position.symbol, quantity_to_sell)
            else:  # SHORT position - need to buy to cover
                try:
                    order_builder = equity_buy_to_cover_market(position.symbol, quantity_to_sell)
                except AttributeError:
                    # Fallback if buy to cover not available
                    order_builder = equity_buy_market(position.symbol, quantity_to_sell)
                    order_builder.set_instruction('BUY_TO_COVER')

            order_builder.set_duration(Duration.DAY)
            order_builder.set_session(Session.NORMAL)
            
            order = order_builder.build()
            
            response = self.schwab_client.place_order(self.account_hash, order)
            
            if response.status_code in [200, 201]:
                order_id = response.headers.get('Location', '').split('/')[-1]
                
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=position.symbol,
                    title=f"📤 Closing Position",
                    message=f"Selling {quantity_to_sell} shares at market",
                    importance=9
                ))
                
                # Track this as a closing order
                self.pending_orders[order_id] = {
                    'symbol': position.symbol,
                    'type': 'CLOSE',
                    'quantity': quantity_to_sell,
                    'status': 'PENDING',
                    'placed_time': datetime.now()
                }
                
                return True
            else:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=position.symbol,
                    title=f"❌ Close Order Failed",
                    message=f"Could not close position: {response.status_code}",
                    importance=10
                ))
                return False
                
        except Exception as e:
            logger.error(f"Position close error: {e}")
            return False
    async def _update_real_positions(self):
        """Update real positions from account"""
        if not self.schwab_client or self.mode != TradingMode.LIVE:
            return
        
        try:
            # Get account positions
            response = self.schwab_client.get_account(
                self.account_hash
                #fields=[schwab_client.Account.Fields.POSITIONS]
            )
            
            if response.status_code == 200:
                account_data = response.json()
                positions_data = account_data.get('securitiesAccount', {}).get('positions', [])
                
                # Update existing positions
                for pos_data in positions_data:
                    symbol = pos_data.get('instrument', {}).get('symbol')
                    if symbol and symbol in self.positions:
                        position = self.positions[symbol]
                        position.current_price = pos_data.get('marketValue', 0) / pos_data.get('longQuantity', 1)
                        position.unrealized_pnl = pos_data.get('unrealizedProfitLoss', 0)
                        
        except Exception as e:
            logger.error(f"Position update error: {e}")
    def _save_state(self):
        """Save current trading state"""
        state = {
            'trade_history': self.trade_history[-100:],  # Keep last 100 trades
            'daily_pnl': self.risk_manager.daily_pnl,
            'consecutive_losses': self.risk_manager.consecutive_losses,
            'last_save': datetime.now().isoformat()
        }
        
        with open("trading_state.json", 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def _init_schwab_client(self):
        """Initialize Schwab client with commentary"""
        try:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.TECHNICAL,
                symbol=None,
                title="🔌 Connecting to Schwab API",
                message="Attempting to establish connection with broker...",
                importance=7
            ))
            
            # Check credentials
            if not Config().SCHWAB_API_KEY or Config().SCHWAB_API_KEY == "your_api_key":
                raise ValueError("SCHWAB_API_KEY not set")
            if not Config().SCHWAB_APP_SECRET or Config().SCHWAB_APP_SECRET == "your_app_secret":
                raise ValueError("SCHWAB_APP_SECRET not set")
            
            # Check if token exists
            if Config().SCHWAB_TOKEN_PATH.exists():
                try:
                    # Fixed token loading
                    with open(Config().SCHWAB_TOKEN_PATH, 'r') as f:
                        token_file_data = json.load(f)
                        # Handle both formats
                        if 'token' in token_file_data:
                            token_data = token_file_data['token']
                        else:
                            token_data = token_file_data
                    
                    # Create client from token
                    self.schwab_client = auth.client_from_token_file(
                        Config().SCHWAB_TOKEN_PATH,
                        Config().SCHWAB_API_KEY,
                        Config().SCHWAB_APP_SECRET
                    )
                    
                    # Test if token is valid
                    response_hash = self.schwab_client.get_account_numbers()
                    response = self.schwab_client.get_accounts()
                    if response.status_code == 200:
                        self.data_provider = SchwabDataProvider(self.schwab_client, self.commentary)
                        accounts = response.json()
                        accounts_hash = response_hash.json()
                        #logger.info(f"Account Info: {accounts}")
                        if accounts:
                            self.account_id = Config().SCHWAB_ACCOUNT_NUMBER or accounts[0].get('securitiesAccount', {}).get('accountNumber')
                            self.account_hash = accounts_hash[0].get('hashValue')
                                                        
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.TECHNICAL,
                            symbol=None,
                            title="✅ Schwab Connected",
                            message="Successfully connected to Schwab API. Ready to trade!",
                            importance=8
                        ))
                        return
                    else:
                        logger.warning("Token expired or invalid")
                except Exception as e:
                    logger.warning(f"Token loading failed: {e}")
            
            # If we get here, need to authenticate
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=None,
                title="⚠️ Authentication Required",
                message="Schwab token not found or invalid. Running in simulation mode without real data.",
                importance=8
            ))
            
            # Create dummy data provider for simulation
            self.data_provider = DummyDataProvider(self.commentary)
            
        except Exception as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=None,
                title="⚠️ Broker Connection Issue",
                message=f"Could not connect to Schwab: {str(e)}. Running in simulation mode.",
                importance=8
            ))
            
            # Create dummy data provider for simulation
            self.data_provider = DummyDataProvider(self.commentary)
     
    async def start(self):
        """Start the trading engine with commentary"""
        self.is_running = True
        
        risk_per_trade = Config().MAX_RISK_PER_TRADE
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.MARKET_ANALYSIS,
            symbol=None,
            title="🎯 Trading Session Started",
            message="Beginning market analysis. I'll explain each step of my decision-making process.",
            data={
                'mode': self.mode.value,
                'account_balance': self.risk_manager.account_balance,
                'risk_per_trade': f"{risk_per_trade:.1%}"
            },
            importance=9
        ))
        
        # In start() method, after self.is_running = True
        if not self.ml_predictor.model.is_trained:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="🎯 Initial Model Training Required",
                message="Collecting historical data for first-time model training...",
                importance=9
            ))
            # Train on initial symbols
            await self.ml_predictor.retrain_model(
                self.data_provider,
                ['PLTR', 'NVDA', 'TSLA']
            )
        
        # Sync with existing Schwab positions on startup
        if self.mode == TradingMode.LIVE and self.schwab_client:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="🔄 Syncing with Schwab Account",
                message="Checking for existing positions in your account...",
                importance=8
            ))
            
            await self._update_and_track_real_positions()
            
            if self.positions:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.MARKET_ANALYSIS,
                    symbol=None,
                    title=f"📊 Found {len(self.positions)} Existing Positions",
                    message=f"Tracking positions: {', '.join(self.positions.keys())}",
                    importance=8
                ))

        analysis_count = 0
        
        while self.is_running:
            try:
                #Market hours check with commentary
                # if not self._is_market_open():
                #    if analysis_count == 0:  # Only show once
                #        self.commentary.add_commentary(TradingCommentary(
                #            timestamp=datetime.now(),
                #            type=CommentaryType.MARKET_ANALYSIS,
                #            symbol=None,
                #            title="🌙 Market Closed",
                #           message="Markets are currently closed. I'm waiting for the next session.",
                #            data={'next_open': self._get_next_market_open()},
                #            importance=3
                #        ))
                #    await asyncio.sleep(60)
                #    continue
                
                analysis_count += 1
                
                # Risk assessment with detailed explanation
                await self._assess_market_conditions()
                await self._analyze_premarket_gaps() 
                # Check if we should trade (with explanation)
                should_trade, reason = await self._evaluate_trading_conditions()
                
                if should_trade:
                    # Analyze markets with commentary
                    await self._analyze_markets_with_commentary()
                else:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=None,
                        title="🛑 Trading Paused",
                        message=f"Not trading right now: {reason}",
                        importance=6
                    ))
                
                # Manage existing positions with commentary
                await self._manage_positions_with_commentary()
                # Emergency stop check
                if self.risk_manager.daily_pnl < -self.risk_manager.account_balance * 0.03:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=None,
                        title="🚨 EMERGENCY STOP",
                        message=f"Daily loss exceeded 3% limit. Closing all positions.",
                        importance=10
                    ))
                    
                    # Close all positions
                    for symbol in list(self.positions.keys()):
                        position = self.positions[symbol]
                        await self._close_position_with_commentary(position, "emergency_stop")
                    
                    self.is_running = False
                    break
                # Check order status for real trades
                if self.mode == TradingMode.LIVE:
                    await self._check_order_status()
                    await self._update_real_positions()
                
                # After the position management section
                if analysis_count % 100 == 0:  # Every 100 cycles
                    asyncio.create_task(self.ml_predictor.retrain_model(
                        self.data_provider, 
                        ['TSLA', 'PLTR', 'NVDA']
                    ))

		        # Save state periodically
                if analysis_count % 10 == 0:  # Every 10 analysis cycles
                    self._save_state()
                    self.brain.save_memories()

                # Add retraining check:
                if analysis_count % 100 == 0:  # Every 100 cycles
                    asyncio.create_task(self.ml_predictor.retrain_model(
                        self.data_provider,
                        ['PLTR', 'NVDA', 'TSLA']
                    ))
                # Wait before next analysis
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=None,
                    title="❌ Error in Trading Loop",
                    message=f"Encountered an error: {str(e)}",
                    importance=9
                ))
                logger.error(f"Trading loop error: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _assess_market_conditions(self):
        """Assess overall market conditions with detailed commentary"""
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.MARKET_ANALYSIS,
            symbol=None,
            title="🔍 Assessing Market Conditions",
            message="Let me check the overall market health before making any trading decisions...",
            importance=7
        ))
        
        # Get market breadth
        if self.data_provider:
            breadth = self.data_provider.calculate_market_breadth()
            self.market_state['breadth'] = breadth
            self.market_state['vix'] = breadth.get('vix', 20)
            
            # Interpret VIX
            vix_interpretation = ""
            if breadth['vix'] < 15:
                vix_interpretation = "Low volatility - Markets are calm, good for trend following"
            elif breadth['vix'] < 25:
                vix_interpretation = "Normal volatility - Standard trading conditions"
            elif breadth['vix'] < 35:
                vix_interpretation = "Elevated volatility - Higher risk, but also opportunities"
            else:
                vix_interpretation = "High volatility - Extreme caution needed, reduce position sizes"
            
            # Interpret breadth
            advance_decline = breadth.get('advance_decline', 0)
            breadth_interpretation = ""
            if advance_decline > 1.5:
                breadth_interpretation = "Strong bullish breadth - More stocks advancing"
                self.market_state['market_trend'] = 'bullish'
            elif advance_decline < -1.5:
                breadth_interpretation = "Strong bearish breadth - More stocks declining"
                self.market_state['market_trend'] = 'bearish'
            else:
                breadth_interpretation = "Neutral market breadth"
                self.market_state['market_trend'] = 'neutral'
            
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="📊 Market Breadth Analysis",
                message=f"{vix_interpretation}\n{breadth_interpretation}",
                data={
                    'vix': breadth['vix'],
                    'advance_decline_ratio': advance_decline,
                    'new_highs': breadth.get('new_highs', 0),
                    'new_lows': breadth.get('new_lows', 0),
                    'put_call_ratio': breadth.get('put_call_ratio', 1.0)
                },
                confidence=0.8,
                importance=8
            ))
    
    async def _analyze_premarket_gaps(self):
        """Analyze pre-market gaps for fade opportunities"""
        if 'gaps' not in self.market_state:
            self.market_state['gaps'] = {}
        
        for symbol in self.dynamic_watchlist:
            try:
                quote = self.data_provider.get_quote(symbol)
                if not quote:
                    continue
                    
                # Get today's open and yesterday's close
                open_price = quote.get('open', 0)
                prev_close = quote.get('close', 0)  # This is previous day's close
                
                # Skip if prices are invalid
                if open_price <= 0 or prev_close <= 0:
                    continue
                    
                gap_percent = ((open_price - prev_close) / prev_close) * 100
                
                if abs(gap_percent) > 2:
                    self.market_state['gaps'][symbol] = {
                        'percent': gap_percent,
                        'direction': 'up' if gap_percent > 0 else 'down',
                        'fade_probability': 0.65 if abs(gap_percent) > 5 else 0.45,
                        'open': open_price,
                        'prev_close': prev_close
                    }
                    
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.MARKET_ANALYSIS,
                        symbol=symbol,
                        title=f"🌅 Gap Detected: {symbol}",
                        message=f"{gap_percent:.1f}% gap {self.market_state['gaps'][symbol]['direction']} (Open: ${open_price:.2f}, Prev Close: ${prev_close:.2f})",
                        importance=7
                    ))
            except Exception as e:
                logger.debug(f"Gap analysis error for {symbol}: {e}")

    async def _evaluate_trading_conditions(self) -> Tuple[bool, str]:
        """Evaluate if we should trade with detailed reasoning"""
        # In commentary mode, we always "trade" but don't execute real orders
        if self.mode == TradingMode.SIMULATION_WITH_COMMENTARY:
            if self.risk_manager.margin_call:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=None,
                    title="⚠️ Margin Call Active",
                    message="Account has a margin call! In real trading, I would stop here. "
                            "But in commentary mode, I'll continue analyzing to show you my thought process.",
                    data={
                        'margin_call': True,
                        'buying_power': self.risk_manager.buying_power
                    },
                    importance=10
                ))
            return True, "Simulation mode - always analyze"
        
        if self._is_news_blackout():
            return False, "Major economic news event"
        # For other modes, check actual conditions
        allowed, reason = self.risk_manager.check_trading_allowed()
        
        if not allowed:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=None,
                title="🚫 Trading Conditions Not Met",
                message=f"I'm not trading because: {reason}",
                data={
                    'daily_pnl': self.risk_manager.daily_pnl,
                    'consecutive_losses': self.risk_manager.consecutive_losses,
                    'buying_power': self.risk_manager.buying_power
                },
                importance=8
            ))
        
        return allowed, reason
    
    async def _analyze_markets_with_commentary(self):
        """Analyze markets with detailed commentary"""
        # Run screener if it's time
        if self.screener and (not self.last_screener_run or 
                            (datetime.now() - self.last_screener_run).total_seconds() > 120):
            try:
                await self.screener.screen_stocks()
                
                # Update watchlist with top movers
                screener_symbols = self.screener.get_watchlist_symbols()
                if screener_symbols:
                    # Combine with default symbols, keeping unique
                    self.dynamic_watchlist = list(set(screener_symbols + ['NVDA', 'TSLA', 'PLTR']))[:10]
                    
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.MARKET_ANALYSIS,
                        symbol=None,
                        title="📊 Watchlist Updated",
                        message=f"Now tracking: {', '.join(self.dynamic_watchlist)}",
                        importance=6
                    ))
                
                self.last_screener_run = datetime.now()
            except Exception as e:
                logger.error(f"Screener error: {e}")
        
        # Use dynamic watchlist
        watchlist = self.dynamic_watchlist
        
        # Get current positions (both tracked and from Schwab)
        positions_held = set(self.positions.keys())
        
        # Add Schwab positions if in live mode
        if self.mode == TradingMode.LIVE and self.schwab_client:
            schwab_positions = await self.get_schwab_positions()
            for pos in schwab_positions:
                positions_held.add(pos['symbol'])
        
        # Filter watchlist to exclude symbols we already have positions in
        symbols_to_analyze = [s for s in watchlist if s not in positions_held]
        
        if not symbols_to_analyze:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="📊 All Watchlist Symbols Have Positions",
                message=f"Currently holding positions in all watchlist symbols: {', '.join(positions_held)}",
                importance=5
            ))
            return
        
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.MARKET_ANALYSIS,
            symbol=None,
            title="📈 Scanning Watchlist",
            message=f"Analyzing {len(symbols_to_analyze)} symbols for trading opportunities. "
                f"Market trend: {self.market_state['market_trend']}\n"
                f"Skipping {', '.join(positions_held)} (already have positions)",
            importance=6
        ))
        
        for symbol in symbols_to_analyze:
            try:
                # Rate limiting check
                await asyncio.sleep(0.5)
                
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.MARKET_ANALYSIS,
                    symbol=symbol,
                    title=f"🔎 Analyzing {symbol}",
                    message="Fetching price data and calculating technical indicators...",
                    importance=4
                ))
                
                # Get market data
                if self.data_provider:
                    data = self.data_provider.get_market_data(symbol)
                    if data.empty:
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.WARNING,
                            symbol=symbol,
                            title=f"⚠️ No Data: {symbol}",
                            message="Could not retrieve price data for this symbol",
                            importance=5
                        ))
                        continue
                    
                    # Get quote
                    quote = self.data_provider.get_quote(symbol)
                    if not quote:
                        quote = {'last': data['Close'].iloc[-1]}
                    
                    # Technical analysis with commentary
                    indicators = await self.technical_analyzer.analyze_with_commentary(data, symbol)
                    
                    # Create market data object with proper type conversion
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        open=float(data['Open'].iloc[-1]),
                        high=float(data['High'].iloc[-1]),
                        low=float(data['Low'].iloc[-1]),
                        close=float(quote.get('last', data['Close'].iloc[-1])),
                        volume=int(quote.get('volume', data['Volume'].iloc[-1])),
                        timeframe='5min',
                        indicators=indicators
                    )
                    
                    # Validate market data
                    if np.isnan(market_data.close) or market_data.close <= 0:
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.WARNING,
                            symbol=symbol,
                            title=f"⚠️ Invalid Price Data: {symbol}",
                            message=f"Price data is invalid (close=${market_data.close}), skipping analysis",
                            importance=6
                        ))
                        continue
                    
                    ml_signal, ml_explanation = await self.ml_predictor.predict_with_commentary(
                        indicators, symbol, data
)
                    
                    # Check each strategy
                    for strategy in self.strategies:
                        signal = await strategy.generate_signal_with_commentary(market_data)
                        
                        if signal:
                            # Process the signal with full explanation
                            await self._process_signal_with_commentary(signal, ml_signal, ml_explanation)
                
            except Exception as e:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title=f"⚠️ Error Analyzing {symbol}",
                    message=f"Encountered error: {str(e)}",
                    importance=5
                ))
                logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
    
    async def _process_signal_with_commentary(self, signal, ml_signal, ml_explanation):
        """Process trading signal with detailed explanation"""
        # Check if already in position (both bot-tracked and real Schwab positions)
        if signal.symbol in self.positions or signal.symbol in self.simulated_positions:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=signal.symbol,
                title=f"📍 Already in Position",
                message=f"I already have a position in {signal.symbol}, skipping this signal.",
                importance=3
            ))
            return
        
        # CRITICAL: Check real Schwab positions before placing order
        if self.mode == TradingMode.LIVE and self.schwab_client:
            schwab_positions = await self.get_schwab_positions()
            existing_position = next((pos for pos in schwab_positions if pos['symbol'] == signal.symbol), None)
            
            if existing_position:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"🚫 Existing Schwab Position Found",
                    message=f"Found existing position: {existing_position['quantity']} shares at ${existing_position['average_price']:.2f}\n"
                        f"Current P&L: ${existing_position['total_pnl']:.2f} ({existing_position['pnl_percent']:.1f}%)",
                    data=existing_position,
                    importance=9
                ))
                
                # Add to bot tracking if not already tracked
                if signal.symbol not in self.positions:
                    position = Position(
                        symbol=signal.symbol,
                        entry_price=existing_position['average_price'],
                        current_price=existing_position['current_price'],
                        quantity=existing_position['quantity'],
                        side='long' if existing_position['quantity'] > 0 else 'short',
                        stop_loss=existing_position['average_price'] * 0.95,
                        take_profit=existing_position['average_price'] * 1.10,
                        entry_time=datetime.now(),
                        unrealized_pnl=existing_position['total_pnl'],
                        reasoning={'source': 'existing_schwab_position'}
                    )
                    self.positions[signal.symbol] = position
                    
                    # Initialize exit tracking
                    if hasattr(self, 'exit_manager'):
                        self.exit_manager.initialize_position_tracking(
                            signal.symbol,
                            position.entry_price,
                            position.stop_loss,
                            position.take_profit
                        )
                
                return
        
        # EARLY BUYING POWER CHECK for LIVE mode
        if self.mode == TradingMode.LIVE:
            # First check if we have ANY buying power before doing calculations
            has_power, available_bp = await self._check_buying_power(Config().MIN_BUYING_POWER)
            logger.info(f"Has buying power or not: {has_power} {available_bp}")
            if not has_power or available_bp < Config().MIN_BUYING_POWER:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"❌ No Buying Power",
                    message=f"Cannot trade - buying power is ${available_bp:.2f} (need at least ${Config().MIN_BUYING_POWER})",
                    importance=9
                ))
                return
            
            # Update risk manager with current buying power
            self.risk_manager.buying_power = available_bp
            # Check for existing orders
            if self.mode == TradingMode.LIVE:
                existing_orders = await self._check_existing_orders(signal.symbol)
                if existing_orders['has_orders']:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=signal.symbol,
                        title=f"⚠️ Existing Orders Found",
                        message=f"Found {existing_orders['count']} existing orders for {signal.symbol}",
                        importance=7
                    ))
                    
                    # Cancel existing orders if configured
                    if Config().AUTO_CANCEL_EXISTING_ORDERS:
                        await self._cancel_existing_orders(signal.symbol)
                    else:
                        return
        # Check with brain if we should take this trade
        should_trade, brain_reason = self.brain.should_take_trade(
            signal.symbol, 
            signal.reasoning.get('strategy', 'unknown'),
            signal.confidence
        )
        if not should_trade:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.PSYCHOLOGY,
                symbol=signal.symbol,
                title=f"🧠 Experience Says No",
                message=brain_reason,
                data={
                    'pattern_history': self.brain.get_pattern_confidence(signal.reasoning.get('strategy', 'unknown')),
                    'emotional_state': self.brain.emotional_state
                },
                importance=7
            ))
            return
	
	# Add more human-like pre-trade thoughts
        pre_trade_thoughts = [
            f"Alright, {signal.symbol} is setting up nicely. Let me check the level 2...",
            f"Interesting setup on {signal.symbol}. This reminds me of last week's trade.",
            f"{signal.symbol} looking ready. Volume confirms - let's do this.",
            f"Been watching {signal.symbol} all morning. Finally getting the entry I wanted."
        ]
        
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.PSYCHOLOGY,
            symbol=signal.symbol,
            title=f"💭 Pre-Trade Thoughts",
            message=np.random.choice(pre_trade_thoughts),
            importance=5
        ))

        # Evaluate the signal
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.SIGNAL_GENERATION,
            symbol=signal.symbol,
            title=f"🎯 Trading Signal Generated!",
            message=f"Strategy '{signal.reasoning.get('strategy', 'unknown')}' found a potential {signal.signal_type.name} opportunity",
            data={
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'risk_reward_ratio': (signal.take_profit - signal.entry_price) / (signal.entry_price - signal.stop_loss),
                'signal_strength': signal.strength,
                'ml_confirmation': ml_signal == 1
            },
            confidence=signal.confidence,
            importance=8
        ))
        
        # Check ML confirmation
        # Check ML confirmation (1 for BUY, -1 for SELL)
        expected_ml_signal = 1 if signal.signal_type == SignalType.BUY else -1
        if ml_signal != expected_ml_signal:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=signal.symbol,
                title=f"🤖 ML Model Disagrees",
                message="The machine learning model doesn't confirm this signal. "
                       "I typically wait for ML confirmation to increase win probability.",
                data=ml_explanation,
                importance=7
            ))
            
            if self.mode != TradingMode.SIMULATION_WITH_COMMENTARY:
                return
            else:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=signal.symbol,
                    title=f"📝 Simulation Override",
                    message="In simulation mode, I'll show you what would happen if we took this trade anyway.",
                    importance=5
                ))
        # Multi-timeframe confirmation
        try:
            daily_data = self.data_provider.get_market_data(signal.symbol, frequency_type='daily', frequency=1)
            if not daily_data.empty and len(daily_data) > 20:
                daily_trend = 'up' if daily_data['Close'].iloc[-1] > daily_data['Close'].iloc[-20] else 'down'
                
                if (signal.signal_type == SignalType.BUY and daily_trend == 'down') or \
                (signal.signal_type == SignalType.SELL and daily_trend == 'up'):
                    signal.confidence *= 0.7
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=signal.symbol,
                        title=f"⚠️ Against Daily Trend",
                        message=f"Signal is against daily trend ({daily_trend}), reducing confidence",
                        importance=6
                    ))
        except Exception as e:
            logger.error(f"Multi-timeframe check failed: {e}")
        
        # Check portfolio concentration
        if self.positions:
            total_value = sum(p.current_price * p.quantity for p in self.positions.values()) 
            max_position_value = total_value * 0.25  # 25% max
            
            if signal.position_size * signal.entry_price > max_position_value:
                signal.position_size = int(max_position_value / signal.entry_price)
        # Calculate position size with explanation
        position_size = await self.risk_manager.calculate_position_size_with_commentary(
            signal, signal.entry_price
        )
        
        if position_size == 0:
            return
        
        signal.position_size = position_size

	# After creating position, initialize exit tracking
        if self.mode == TradingMode.SIMULATION_WITH_COMMENTARY:
            self.exit_manager.initialize_position_tracking(
                signal.symbol,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit
            )
        
        # Final decision
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.DECISION,
            symbol=signal.symbol,
            title=f"✅ Trade Decision: {signal.signal_type.name}",
            message=f"After careful analysis, I've decided to {'BUY' if signal.signal_type.value == 1 else 'SELL'} {position_size} shares of {signal.symbol}",
            data={
                'total_cost': position_size * signal.entry_price,
                'max_risk': position_size * (signal.entry_price - signal.stop_loss),
                'max_reward': position_size * (signal.take_profit - signal.entry_price),
                'reasoning': signal.reasoning
            },
            confidence=signal.confidence,
            importance=10
        ))
        
        # Execute the trade based on mode
        position = None
        if self.mode == TradingMode.SIMULATION_WITH_COMMENTARY:
            # Create simulated position
            position = Position(
                symbol=signal.symbol,
                entry_price=signal.entry_price,
                current_price=signal.entry_price,
                quantity=signal.position_size,
                side='long' if signal.signal_type == SignalType.BUY else 'short',
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                entry_time=datetime.now(),
                reasoning=signal.reasoning
            )
            self.simulated_positions[signal.symbol] = position
            
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=signal.symbol,
                title=f"📋 Simulated Position Opened",
                message=f"Position tracked in simulation. I'll monitor it and explain my management decisions.",
                importance=7
            ))
            
        elif self.mode == TradingMode.LIVE:
            # Execute real trade
            success = await self._execute_real_trade(signal)
            if success:
                # Don't create position here - wait for order fill confirmation
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=signal.symbol,
                    title=f"📋 Order Placed Successfully",
                    message=f"Waiting for order fill confirmation before tracking position",
                    importance=8
                ))
            else:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"❌ Trade Execution Failed",
                    message=f"Could not execute trade for {signal.symbol}",
                    importance=10
                ))
        # Only initialize exit tracking if position was created
        if position and hasattr(self, 'exit_manager'):
            self.exit_manager.initialize_position_tracking(
                signal.symbol,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit
            )

   
    async def _manage_positions_with_commentary(self):
        """Manage positions with detailed commentary"""
        positions_to_check = self.simulated_positions if self.mode == TradingMode.SIMULATION_WITH_COMMENTARY else self.positions
        
        if not positions_to_check:
            return
        
        for symbol, position in list(positions_to_check.items()):
            try:
                # Check if position is None
                if position is None:
                    logger.warning(f"None position found for {symbol}, removing")
                    del positions_to_check[symbol]
                    continue
                
                # Get current price
                if self.data_provider:
                    quote = self.data_provider.get_quote(symbol)
                    if quote:
                        current_price = quote.get('last', position.current_price)
                    else:
                        # Simulate price movement for demo
                        current_price = position.current_price * (1 + np.random.randn() * 0.001)
                    
                    # Update position
                    old_price = position.current_price
                    position.current_price = current_price
                    if position.side == 'short':
                        position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                    else:  # long
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    
                    # Commentary on position status (only if significant change)
                    price_change_pct = ((current_price - old_price) / old_price) * 100
                    if abs(price_change_pct) > 0.1:  # Only comment on >0.1% moves
                        pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                        
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.MARKET_ANALYSIS,
                            symbol=symbol,
                            title=f"📊 Position Update: {symbol}",
                            message=f"Position {'profitable' if position.unrealized_pnl > 0 else 'losing'} "
                                f"by {abs(pnl_pct):.1f}%",
                            data={
                                'current_price': current_price,
                                'entry_price': position.entry_price,
                                'unrealized_pnl': position.unrealized_pnl,
                                'pnl_percentage': pnl_pct,
                                'distance_to_stop': ((current_price - position.stop_loss) / current_price) * 100,
                                'distance_to_target': ((position.take_profit - current_price) / current_price) * 100
                            },
                            importance=4
                        ))
                    
                    # Check exit conditions with reasoning
                    should_exit, exit_reason = await self._evaluate_exit_conditions(position, current_price)
                    
                    if should_exit:
                        await self._close_position_with_commentary(position, exit_reason)
                
            except Exception as e:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title=f"⚠️ Error Managing {symbol}",
                    message=str(e),
                    importance=6
                ))
    async def _evaluate_exit_conditions(self, position, current_price: float) -> Tuple[bool, str]:
        """Evaluate exit conditions with detailed reasoning"""
        # If manual close only mode, only check hard stops
        if self.manual_close_only and self.mode == TradingMode.LIVE:
            # Still check stop loss for safety
            if current_price <= position.stop_loss:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=position.symbol,
                    title=f"⚠️ Stop Loss Triggered",
                    message=f"Price hit stop loss at ${current_price:.2f}. Manual close required.",
                    importance=9
                ))
                # Don't auto-close, just warn
                return False, ""
            return False, ""
        # First check hard stops
        if current_price <= position.stop_loss:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=position.symbol,
                title=f"🛑 Stop Loss Hit",
                message=f"Stopped out at ${current_price:.2f}. Part of the game - on to the next one.",
                importance=9
            ))
            return True, "stop_loss"
        
        # Get current indicators for dynamic exit
        if self.data_provider:
            data = self.data_provider.get_market_data(position.symbol)
            if not data.empty:
                indicators = await self.technical_analyzer.analyze_with_commentary(data, position.symbol)
                
                # Use dynamic exit manager
                should_exit, reason, exit_portion = await self.exit_manager.evaluate_exit(
                    position, current_price, indicators
                )
                
                if should_exit:
                    if exit_portion < 1.0:
                        # Partial exit - reduce position
                        position.quantity = int(position.quantity * (1 - exit_portion))
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.DECISION,
                            symbol=position.symbol,
                            title=f"📊 Partial Exit",
                            message=f"Taking {exit_portion*100:.0f}% off the table. Letting the rest ride.",
                            importance=8
                        ))
                        if position.quantity > 0:
                            return False, ""  # Keep remaining position
                    
                    return True, reason
        
        return False, ""
    
    async def _close_position_with_commentary(self, position, reason: str):
        """Close position with detailed commentary"""
        # Debug log
        logger.info(f"Attempting to close position: {position.symbol}, mode: {self.mode}")
        # ADD CONFIRMATION HERE
        # Check if confirmation is needed
        if self.mode == TradingMode.LIVE and self.require_confirmations:
            # Calculate P&L percentage
            pnl = position.unrealized_pnl
            pnl_percent = abs((pnl / (position.entry_price * position.quantity)) * 100)
            
            # Determine if confirmation is needed based on settings
            needs_confirmation = True
            
            # Skip confirmation for profits if only confirming losses
            if self.confirm_only_losses and pnl > 0:
                needs_confirmation = False
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=position.symbol,
                    title=f"✅ Auto-closing Profit",
                    message=f"Confirmations only required for losses. Closing profitable position.",
                    importance=6
                ))
            
            # Skip confirmation if below threshold
            elif self.confirm_threshold_percent > 0 and pnl_percent < self.confirm_threshold_percent:
                needs_confirmation = False
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=position.symbol,
                    title=f"✅ Auto-closing Small Position",
                    message=f"P&L below {self.confirm_threshold_percent}% threshold. No confirmation needed.",
                    importance=6
                ))
            
            # Request confirmation if needed
            if needs_confirmation:
                if not await self._get_close_confirmation(position, reason):
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.DECISION,
                        symbol=position.symbol,
                        title=f"🚫 Close Cancelled",
                        message=f"User cancelled position close for {position.symbol}",
                        importance=7
                    ))
                    return
        pnl = position.unrealized_pnl
        roi = (pnl / (position.entry_price * position.quantity)) * 100
        
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.DECISION,
            symbol=position.symbol,
            title=f"{'💰' if pnl > 0 else '💸'} Position Closed: {position.symbol}",
            message=f"Closing position with {'profit' if pnl > 0 else 'loss'} of ${abs(pnl):.2f} ({abs(roi):.1f}%)",
            data={
                'entry_price': position.entry_price,
                'exit_price': position.current_price,
                'quantity': position.quantity,
            'pnl': pnl,
                'roi_percentage': roi,
                'exit_reason': reason,
                'holding_period': (datetime.now() - position.entry_time).total_seconds() / 3600
            },
            importance=9
        ))
        
	# Learn from this trade
        memory = self.brain.remember_trade(
            symbol=position.symbol,
            pattern=position.reasoning.get('strategy', 'unknown'),
            outcome='win' if pnl > 0 else 'loss',
            pnl_percent=roi,
            context={
                'exit_reason': reason,
                'holding_time': (datetime.now() - position.entry_time).total_seconds() / 60,
                'max_profit': getattr(position, 'max_unrealized_pnl', pnl),
                'market_conditions': self.market_state
            }
        )
        
        # Add the lesson learned
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.PSYCHOLOGY,
            symbol=position.symbol,
            title=f"📚 Lesson Learned",
            message=memory.lesson,
            importance=7
        ))
        
        # Track strategy performance
        strategy = position.reasoning.get('strategy', 'unknown')
        if not hasattr(self, 'strategy_performance'):
            self.strategy_performance = {}

        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {'wins': 0, 'losses': 0, 'total': 0}

        if pnl > 0:
            self.strategy_performance[strategy]['wins'] += 1
        else:
            self.strategy_performance[strategy]['losses'] += 1
        self.strategy_performance[strategy]['total'] += 1

        # Disable underperforming strategies
        if self.strategy_performance[strategy]['total'] > 20:
            win_rate = self.strategy_performance[strategy]['wins'] / self.strategy_performance[strategy]['total']
            if win_rate < 0.35:
                logger.warning(f"Disabling {strategy} - win rate {win_rate:.1%}")
        # Save state after each trade
        self._save_state()

        # Post-trade analysis
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.PSYCHOLOGY,
            symbol=position.symbol,
            title=f"📝 Post-Trade Analysis",
            message=self._generate_post_trade_analysis(position, reason, pnl),
            importance=6
        ))
        
        # Update risk manager
        self.risk_manager.daily_pnl += pnl
        if pnl < 0:
            self.risk_manager.consecutive_losses += 1
        else:
            self.risk_manager.consecutive_losses = 0
        
        # At the end of the method, before removing from positions dict:
        if self.mode == TradingMode.LIVE:
        # Close real position
            success = await self._close_real_position(position)
            if not success:
                return  # Don't remove from tracking if close failed
        
        # Remove from positions
        if self.mode == TradingMode.SIMULATION_WITH_COMMENTARY:
            del self.simulated_positions[position.symbol]
        else:
            del self.positions[position.symbol]
        
        # Update trade history
        self.trade_history.append({
            'symbol': position.symbol,
            'entry_time': position.entry_time.isoformat(),
            'exit_time': datetime.now().isoformat(),
            'entry_price': position.entry_price,
            'exit_price': position.current_price,
            'quantity': position.quantity,
            'pnl': pnl,
            'exit_reason': reason,
            'reasoning': position.reasoning
        })
    
    async def _get_close_confirmation(self, position, reason: str) -> bool:
        """Get user confirmation before closing position"""
        # Initialize pending requests dict if it doesn't exist
        if not hasattr(self, 'pending_close_requests'):
            self.pending_close_requests = {}
        # Store pending close request
        close_request_id = f"close_{position.symbol}_{int(time.time())}"
        self.pending_close_requests = getattr(self, 'pending_close_requests', {})
        
        # Create confirmation request
        self.pending_close_requests[close_request_id] = {
            'position': position,
            'reason': reason,
            'timestamp': datetime.now(),
            'confirmed': None
        }
        
        # Send confirmation request to UI
        pnl = position.unrealized_pnl
        roi = (pnl / (position.entry_price * position.quantity)) * 100
        # Log that we're requesting confirmation
        logger.info(f"Requesting close confirmation for {position.symbol}")
        # Send confirmation request to UI
        try:
            await self.connection_manager.broadcast({
                'type': 'close_confirmation_request',
                'data': {
                    'request_id': close_request_id,
                    'symbol': position.symbol,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'pnl': pnl,
                    'roi': roi,
                    'reason': reason,
                    'message': f"Close {position.symbol} with {'profit' if pnl > 0 else 'loss'} of ${abs(pnl):.2f} ({abs(roi):.1f}%)?"
                }
            })
        except Exception as e:
            logger.error(f"Error broadcasting close confirmation: {e}")
            return False
        
        # Wait for user response (with timeout)
        timeout = 30  # 30 seconds timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if close_request_id in self.pending_close_requests:
                if self.pending_close_requests[close_request_id]['confirmed'] is not None:
                    confirmed = self.pending_close_requests[close_request_id]['confirmed']
                    del self.pending_close_requests[close_request_id]
                    return confirmed
            await asyncio.sleep(0.1)
        
        # Timeout - default to not closing
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.WARNING,
            symbol=position.symbol,
            title=f"⏱️ Close Confirmation Timeout",
            message=f"No response received for closing {position.symbol}. Position remains open.",
            importance=8
        ))
        
        if close_request_id in self.pending_close_requests:
            del self.pending_close_requests[close_request_id]
        
        return False
    def _generate_post_trade_analysis(self, position, exit_reason: str, pnl: float) -> str:
        """Generate insightful post-trade analysis"""
        analysis = []
        
        if pnl > 0:
            analysis.append("✅ What went right:")
            analysis.append(f"- The {position.reasoning.get('strategy', 'strategy')} correctly identified the opportunity")
            analysis.append(f"- Entry timing was good at ${position.entry_price:.2f}")
            if exit_reason == "take_profit":
                analysis.append("- The trade reached its full potential as planned")
        else:
            analysis.append("❌ What could be improved:")
            if exit_reason == "stop_loss":
                analysis.append("- The market moved against our analysis")
                analysis.append("- Consider if the stop was too tight or entry was premature")
            analysis.append("- Review if market conditions changed after entry")
        
        analysis.append("\n💡 Key lessons:")
        analysis.append("- Risk management worked as designed")
        analysis.append(f"- The {abs((position.current_price - position.entry_price) / position.entry_price * 100):.1f}% move took {(datetime.now() - position.entry_time).total_seconds() / 3600:.1f} hours")
        
        return "\n".join(analysis)
    
    async def _get_real_account_info(self) -> Dict[str, float]:
        """Get real account information from Schwab"""
        if not self.schwab_client or not self.account_id:
            return {}
    
        try:
            response = self.schwab_client.get_account(self.account_hash)
            if response.status_code == 200:
                data = response.json()
                account = data.get('securitiesAccount', {})
                balances = account.get('currentBalances', {})
            
                return {
                    'balance': balances.get('liquidationValue', 0),
                    'buying_power': balances.get('buyingPower', 0),
                    'day_trades_remaining': account.get('roundTrips', 3),
                    'cash': balances.get('cashBalance', 0)
                }
        except Exception as e:
            logger.error(f"Account info error: {e}")
    
        return {}

    def _is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        
        # Pre-market: 4:00 AM - 9:30 AM EST
        pre_market_start = now.replace(hour=4, minute=0, second=0)
    
        # After-hours end: 8:00 PM EST
        after_hours_end = now.replace(hour=20, minute=0, second=0)
    
        return pre_market_start <= now < after_hours_end
    
    def _get_next_market_open(self) -> str:
        """Get next market open time"""
        now = datetime.now()
        
        if now.weekday() < 5 and now.hour < 9 or (now.hour == 9 and now.minute < 30):
            next_open = now.replace(hour=9, minute=30, second=0)
        elif now.weekday() >= 4:
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            next_open = now + timedelta(days=days_until_monday)
            next_open = next_open.replace(hour=9, minute=30, second=0)
        else:
            next_open = now + timedelta(days=1)
            next_open = next_open.replace(hour=9, minute=30, second=0)
        
        return next_open.strftime("%Y-%m-%d %H:%M:%S")

    # Add these methods inside the TradingEngineWithCommentary class:

    async def get_schwab_positions(self) -> List[Dict]:
        """Get real positions from Schwab account"""
        if not self.schwab_client or not self.account_id:
            return []
        
        try:
            response = self.schwab_client.get_account(
                self.account_hash,
                fields=[self.schwab_client.Account.Fields.POSITIONS]
            )
            
            if response.status_code == 200:
                data = response.json()
                account_data = data.get('securitiesAccount', {})
                positions_data = account_data.get('positions', [])
                
                schwab_positions = []
                
                for pos in positions_data:
                    instrument = pos.get('instrument', {})
                    symbol = instrument.get('symbol', 'UNKNOWN')
                    
                    # Extract position details
                    long_quantity = pos.get('longQuantity', 0)
                    short_quantity = pos.get('shortQuantity', 0)
                    quantity = long_quantity - short_quantity
                    
                    if quantity == 0:
                        continue
                    
                    # Get prices
                    average_price = pos.get('averagePrice', 0)
                    market_value = pos.get('marketValue', 0)
                    
                    # Calculate current price
                    current_price = market_value / quantity if quantity != 0 else 0
                    
                    # Get P&L
                    day_pnl = pos.get('currentDayProfitLoss', 0)
                    total_pnl = pos.get('unrealizedProfitLoss', 0)
                    
                    # Calculate percentages
                    pnl_percent = (total_pnl / (average_price * abs(quantity))) * 100 if average_price > 0 and quantity != 0 else 0
                    
                    position_info = {
                        'symbol': symbol,
                        'quantity': quantity,
                        'side': 'long' if quantity > 0 else 'short',
                        'average_price': average_price,
                        'current_price': current_price,
                        'market_value': market_value,
                        'day_pnl': day_pnl,
                        'total_pnl': total_pnl,
                        'pnl_percent': pnl_percent,
                        'asset_type': instrument.get('assetType', 'EQUITY')
                    }
                    
                    schwab_positions.append(position_info)
                    
                    # Add commentary about the position
                    # self.commentary.add_commentary(TradingCommentary(
                    #     timestamp=datetime.now(),
                    #     type=CommentaryType.MARKET_ANALYSIS,
                    #     symbol=symbol,
                    #     title=f"📊 Existing Position: {symbol}",
                    #     message=f"Found {abs(quantity)} shares {'long' if quantity > 0 else 'short'}\n"
                    #         f"Avg Price: ${average_price:.2f}, Current: ${current_price:.2f}\n"
                    #         f"P&L: ${total_pnl:.2f} ({pnl_percent:.1f}%)",
                    #     data=position_info,
                    #     importance=6
                    # ))
                
                return schwab_positions
                
        except Exception as e:
            logger.error(f"Error fetching Schwab positions: {e}")
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=None,
                title="⚠️ Error Fetching Positions",
                message=f"Could not retrieve positions: {str(e)}",
                importance=7
            ))
        
        return []

    async def _update_and_track_real_positions(self):
        """Update and track real positions from Schwab account"""
        if not self.schwab_client or self.mode != TradingMode.LIVE:
            return
        
        try:
            schwab_positions = await self.get_schwab_positions()
            
            # Update existing tracked positions
            for pos_data in schwab_positions:
                symbol = pos_data['symbol']
                
                # Check if we're already tracking this position
                if symbol in self.positions:
                    # Update the tracked position with latest data
                    position = self.positions[symbol]
                    position.current_price = pos_data['current_price']
                    position.unrealized_pnl = pos_data['total_pnl']
                else:
                    # Create a new position object for positions not initiated by bot
                    position = Position(
                        symbol=symbol,
                        entry_price=pos_data['average_price'],
                        current_price=pos_data['current_price'],
                        quantity=pos_data['quantity'],
                        side='long' if pos_data['quantity'] > 0 else 'short',
                        stop_loss=pos_data['average_price'] * 0.95,  # Default 5% stop
                        take_profit=pos_data['average_price'] * 1.10,  # Default 10% target
                        entry_time=datetime.now(),  # We don't know actual entry time
                        unrealized_pnl=pos_data['total_pnl'],
                        reasoning={'source': 'existing_position', 'tracked_from': datetime.now().isoformat()}
                    )
                    
                    self.positions[symbol] = position
                    
                    # Initialize exit tracking for existing positions
                    if hasattr(self, 'exit_manager'):
                        self.exit_manager.initialize_position_tracking(
                            symbol,
                            position.entry_price,
                            position.stop_loss,
                            position.take_profit
                        )
                    
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.MARKET_ANALYSIS,
                        symbol=symbol,
                        title=f"📌 Tracking Existing Position",
                        message=f"Now tracking {symbol} position from your account",
                        data={
                            'shares': position.quantity,
                            'avg_price': position.entry_price,
                            'current_pnl': position.unrealized_pnl
                        },
                        importance=7
                    ))
                    
        except Exception as e:
            logger.error(f"Error updating real positions: {e}")
    
    def _is_news_blackout(self) -> bool:
        """Check if we're in news blackout period"""
        now = datetime.now()
        
        # Economic calendar blackouts (EST)
        blackouts = [
            # (hour, minute, duration_minutes)
            (8, 30, 15),   # CPI/Jobs
            (10, 0, 15),   # Consumer confidence
            (14, 0, 30),   # FOMC
            (14, 30, 15),  # Powell speaks
        ]
        
        for hour, minute, duration in blackouts:
            event_time = now.replace(hour=hour, minute=minute, second=0)
            if event_time <= now <= event_time + timedelta(minutes=duration):
                return True
        return False
# ============================================================================
# DUMMY DATA PROVIDER FOR SIMULATION
# ============================================================================

class DummyDataProvider:
    """Provides simulated market data when Schwab is not connected"""
    
    def __init__(self, commentary_system):
        self.commentary = commentary_system
        self._price_cache = {}
        self._last_prices = {}
    
    def get_market_data(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Generate simulated market data"""
        # Generate realistic-looking price data
        np.random.seed(hash(symbol) % 1000)  # Consistent data per symbol
        
        base_price = {
            'AAPL': 180,
            'MSFT': 420,
            'GOOGL': 140,
            'AMZN': 175,
            'NVDA': 1000,  # Added NVDA
            'TSLA': 250,
            'SPY': 450,
            'QQQ': 390
        }.get(symbol, 100)
        
        # Get or initialize last price
        if symbol not in self._last_prices:
            self._last_prices[symbol] = base_price
        
        # Generate 100 5-minute candles
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        
        # Random walk from last price
        returns = np.random.randn(100) * 0.002  # 0.2% volatility
        prices = self._last_prices[symbol] * np.exp(np.cumsum(returns))
        
        # Update last price
        self._last_prices[symbol] = prices[-1]
        
        # Create OHLCV data with proper float64 types
        data = pd.DataFrame(index=dates)
        data['Open'] = (prices * (1 + np.random.randn(100) * 0.0005)).astype(np.float64)
        data['High'] = (prices * (1 + np.abs(np.random.randn(100)) * 0.001)).astype(np.float64)
        data['Low'] = (prices * (1 - np.abs(np.random.randn(100)) * 0.001)).astype(np.float64)
        data['Close'] = prices.astype(np.float64)
        data['Volume'] = np.random.randint(1000000, 5000000, 100).astype(np.float64)
        
        return data
    
    def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get simulated quote"""
        if symbol not in self._last_prices:
            # Initialize with base price if not already done
            base_price = {
                'AAPL': 180,
                'MSFT': 420,
                'GOOGL': 140,
                'AMZN': 175,
                'NVDA': 1000,
                'TSLA': 250,
                'SPY': 450,
                'QQQ': 390
            }.get(symbol, 100)
            self._last_prices[symbol] = base_price
        
        last_price = self._last_prices.get(symbol, 100)
        
        return {
            'bid': last_price - 0.01,
            'ask': last_price + 0.01,
            'last': last_price,
            'volume': np.random.randint(1000000, 5000000),
            'high': last_price * 1.01,
            'low': last_price * 0.99,
            'open': last_price * 0.995,
            'close': last_price
        }
    
    def calculate_market_breadth(self) -> Dict[str, float]:
        """Generate simulated market breadth"""
        return {
            'advance_decline': np.random.uniform(-2, 2),
            'new_highs': np.random.randint(50, 200),
            'new_lows': np.random.randint(20, 100),
            'vix': np.random.uniform(12, 25),
            'put_call_ratio': np.random.uniform(0.7, 1.3)
        }

# ============================================================================
# FREE NEWS AGGREGATOR
# ============================================================================

class FreeNewsAggregator:
    """Aggregates news from free sources"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        
        # Free RSS feeds
        self.rss_feeds = {
            'yahoo': "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
            'google': "https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en",
            'investing': "https://www.investing.com/rss/news.rss",
            'marketwatch': "https://feeds.marketwatch.com/marketwatch/topstories/"
        }
    
    async def fetch_news(self, symbol: str, hours: int = 24) -> List[NewsItem]:
        """Fetch news from all free sources"""
        all_news = []
        
        # Yahoo Finance (most reliable free source)
        yahoo_news = await self._fetch_yahoo_news(symbol, hours)
        all_news.extend(yahoo_news)
        
        # RSS Feeds
        rss_news = await self._fetch_rss_news(symbol, hours)
        all_news.extend(rss_news)
        
        # MarketWatch scraping
        marketwatch_news = await self._fetch_marketwatch_news(symbol, hours)
        all_news.extend(marketwatch_news)
        
        # Remove duplicates
        unique_news = {}
        for item in all_news:
            key = item.headline[:50]  # Use first 50 chars as key
            if key not in unique_news:
                unique_news[key] = item
        
        # Sort by relevance and time
        sorted_news = sorted(
            unique_news.values(),
            key=lambda x: (x.relevance_score, -x.age_hours()),
            reverse=True
        )
        
        return sorted_news[:20]  # Top 20 news items
    
    async def _fetch_yahoo_news(self, symbol: str, hours: int) -> List[NewsItem]:
        """Fetch from Yahoo Finance"""
        try:
            # Use yfinance for news
            ticker = yf.Ticker(symbol)
            news_data = ticker.news
            
            news_items = []
            for article in news_data:
                # Check age
                pub_time = datetime.fromtimestamp(article.get('providerPublishTime', 0))
                if (datetime.now() - pub_time).total_seconds() / 3600 > hours:
                    continue
                
                # Create news item
                news_item = NewsItem(
                    id=hashlib.md5(article['link'].encode()).hexdigest()[:10],
                    symbol=symbol,
                    headline=article.get('title', ''),
                    summary=article.get('summary', ''),
                    source=article.get('publisher', 'Yahoo Finance'),
                    url=article.get('link', ''),
                    published_time=pub_time,
                    sentiment_score=0.0,  # Will be analyzed
                    sentiment_confidence=0.0,
                    impact=NewsImpact.MEDIUM,
                    relevance_score=0.8  # Yahoo pre-filters by ticker
                )
                
                news_items.append(news_item)
            
            # Also get RSS feed
            rss_url = self.rss_feeds['yahoo'].format(symbol=symbol)
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:10]:  # Last 10 entries
                # Parse time
                pub_time = datetime(*entry.published_parsed[:6])
                if (datetime.now() - pub_time).total_seconds() / 3600 > hours:
                    continue
                
                news_item = NewsItem(
                    id=hashlib.md5(entry.link.encode()).hexdigest()[:10],
                    symbol=symbol,
                    headline=entry.title,
                    summary=entry.get('summary', ''),
                    source='Yahoo Finance RSS',
                    url=entry.link,
                    published_time=pub_time,
                    sentiment_score=0.0,
                    sentiment_confidence=0.0,
                    impact=NewsImpact.MEDIUM,
                    relevance_score=0.8
                )
                
                news_items.append(news_item)
            
            return news_items
            
        except Exception as e:
            logger.error(f"Yahoo news error for {symbol}: {e}")
            return []
    
    async def _fetch_rss_news(self, symbol: str, hours: int) -> List[NewsItem]:
        """Fetch from various RSS feeds"""
        news_items = []
        
        # Google News RSS
        try:
            google_url = self.rss_feeds['google'].format(symbol=symbol)
            feed = feedparser.parse(google_url)
            
            for entry in feed.entries[:5]:  # Top 5 from Google
                # Extract actual source from Google News title
                title = entry.title
                source = 'Google News'
                if ' - ' in title:
                    parts = title.rsplit(' - ', 1)
                    title = parts[0]
                    source = parts[1]
                
                # Parse time
                pub_time = datetime(*entry.published_parsed[:6])
                if (datetime.now() - pub_time).total_seconds() / 3600 > hours:
                    continue
                
                # Check relevance
                if symbol.upper() not in title.upper():
                    continue
                
                news_item = NewsItem(
                    id=hashlib.md5(entry.link.encode()).hexdigest()[:10],
                    symbol=symbol,
                    headline=title,
                    summary=entry.get('summary', '')[:200],
                    source=source,
                    url=entry.link,
                    published_time=pub_time,
                    sentiment_score=0.0,
                    sentiment_confidence=0.0,
                    impact=NewsImpact.MEDIUM,
                    relevance_score=0.6
                )
                
                news_items.append(news_item)
                
        except Exception as e:
            logger.debug(f"Google News error: {e}")
        
        return news_items
    
    async def _fetch_marketwatch_news(self, symbol: str, hours: int) -> List[NewsItem]:
        """Scrape news from MarketWatch"""
        news_items = []
        
        try:
            # MarketWatch URLs
            urls = [
                f"https://www.marketwatch.com/investing/stock/{symbol}",
                f"https://www.marketwatch.com/investing/stock/{symbol}/news"
            ]
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                for url in urls:
                    try:
                        async with session.get(url, headers=headers) as response:
                            if response.status != 200:
                                continue
                            
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Find news articles
                            articles = soup.find_all('div', class_='article__content')
                            if not articles:
                                # Try alternative selectors
                                articles = soup.find_all('div', {'class': re.compile('element--article')})
                            
                            for article in articles[:10]:  # Get up to 10 articles
                                try:
                                    # Extract headline
                                    headline_elem = article.find(['h3', 'h2', 'a'], class_=re.compile('headline|title'))
                                    if not headline_elem:
                                        continue
                                    
                                    headline = headline_elem.get_text(strip=True)
                                    
                                    # Extract link
                                    link_elem = article.find('a', href=True)
                                    if link_elem:
                                        link = link_elem['href']
                                        if not link.startswith('http'):
                                            link = f"https://www.marketwatch.com{link}"
                                    else:
                                        continue
                                    
                                    # Extract time
                                    time_elem = article.find(['span', 'time'], class_=re.compile('time|date|timestamp'))
                                    if time_elem:
                                        time_text = time_elem.get_text(strip=True)
                                        pub_time = self._parse_time(time_text)
                                    else:
                                        pub_time = datetime.now()
                                    
                                    # Check age
                                    if (datetime.now() - pub_time).total_seconds() / 3600 > hours:
                                        continue
                                    
                                    # Extract summary
                                    summary_elem = article.find(['p', 'div'], class_=re.compile('summary|teaser|deck'))
                                    summary = summary_elem.get_text(strip=True) if summary_elem else ''
                                    
                                    # Check relevance
                                    if symbol.upper() not in headline.upper() and symbol.upper() not in summary.upper():
                                        relevance = 0.3
                                    else:
                                        relevance = 0.7
                                    
                                    news_item = NewsItem(
                                        id=hashlib.md5(link.encode()).hexdigest()[:10],
                                        symbol=symbol,
                                        headline=headline,
                                        summary=summary[:200],
                                        source='MarketWatch',
                                        url=link,
                                        published_time=pub_time,
                                        sentiment_score=0.0,
                                        sentiment_confidence=0.0,
                                        impact=NewsImpact.MEDIUM,
                                        relevance_score=relevance
                                    )
                                    
                                    news_items.append(news_item)
                                    
                                except Exception as e:
                                    logger.debug(f"Error parsing article: {e}")
                                    continue
                                    
                    except Exception as e:
                        logger.debug(f"MarketWatch scraping error for {url}: {e}")
                        
        except Exception as e:
            logger.error(f"MarketWatch general error: {e}")
        
        return news_items
    
    def _parse_time(self, time_text: str) -> datetime:
        """Parse various time formats from MarketWatch"""
        try:
            time_text = time_text.strip()
            
            # Handle relative times
            if 'ago' in time_text:
                if 'minute' in time_text:
                    minutes = int(re.search(r'(\d+)', time_text).group(1))
                    return datetime.now() - timedelta(minutes=minutes)
                elif 'hour' in time_text:
                    hours = int(re.search(r'(\d+)', time_text).group(1))
                    return datetime.now() - timedelta(hours=hours)
                elif 'day' in time_text:
                    days = int(re.search(r'(\d+)', time_text).group(1))
                    return datetime.now() - timedelta(days=days)
            
            # Handle "Today" or "Yesterday"
            if 'today' in time_text.lower():
                return datetime.now()
            elif 'yesterday' in time_text.lower():
                return datetime.now() - timedelta(days=1)
            
            # Try parsing absolute dates
            for fmt in ['%b. %d, %Y', '%B %d, %Y', '%m/%d/%Y']:
                try:
                    return datetime.strptime(time_text, fmt)
                except:
                    continue
                    
        except:
            pass
        
        return datetime.now()  # Default to now

# ============================================================================
# SIMPLIFIED NEWS STRATEGY
# ============================================================================

class FreeNewsSignalStrategy(TradingStrategyWithCommentary):
    """News strategy using only free sources"""
    from nltk.sentiment import SentimentIntensityAnalyzer
    def __init__(self, commentary_system):
        super().__init__(commentary_system)
        self.aggregator = FreeNewsAggregator()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()  # VADER only
        self.last_signal_time = {}
        self.profiles = {}
    
    async def generate_signal_with_commentary(self, market_data) -> Optional[TradingSignal]:
        """Generate signal from free news sources"""
        symbol = market_data.symbol
        
        # Check cooldown
        if symbol in self.last_signal_time:
            if (datetime.now() - self.last_signal_time[symbol]).total_seconds() < 3600:
                return None
        
        # Fetch news
        news_items = await self.aggregator.fetch_news(symbol, 24)
        
        if len(news_items) < 3:  # Need at least 3 articles
            return None
        
        # Analyze sentiment
        sentiments = []
        high_impact_news = []
        
        for item in news_items:
            # Simple sentiment analysis
            text = f"{item.headline} {item.summary}"
            scores = self.sentiment_analyzer.polarity_scores(text)
            item.sentiment_score = scores['compound']
            item.sentiment_confidence = abs(scores['compound'])
            
            # Detect high impact
            if any(keyword in text.lower() for keyword in ['earnings', 'beat', 'miss', 'sec', 'investigation']):
                item.impact = NewsImpact.HIGH
                high_impact_news.append(item)
            
            sentiments.append(item.sentiment_score)
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Generate signal if strong sentiment
        if abs(avg_sentiment) > 0.3 or high_impact_news:
            signal_type = SignalType.BUY if avg_sentiment > 0 else SignalType.SELL
            confidence = min(abs(avg_sentiment) + 0.3, 0.8)
            
            # Commentary
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.OPPORTUNITY,
                symbol=symbol,
                title=f"📰 News Signal: {signal_type.name} {symbol}",
                message=f"News sentiment: {'Positive' if avg_sentiment > 0 else 'Negative'} ({avg_sentiment:.2f})\n\n"
                       f"Headlines:\n" + 
                       "\n".join([f"• {item.headline}" for item in news_items[:3]]),
                data={
                    'sentiment': avg_sentiment,
                    'news_count': len(news_items),
                    'high_impact': len(high_impact_news) > 0,
                    'sources': list(set(item.source for item in news_items[:5]))
                },
                confidence=confidence,
                importance=8 if high_impact_news else 7
            ))
            
            # Calculate stops
            atr = market_data.indicators.get('atr', market_data.close * 0.02)
            
            if signal_type == SignalType.BUY:
                stop_loss = market_data.close - (2 * atr)
                take_profit = market_data.close + (4 * atr)
            else:
                stop_loss = market_data.close + (2 * atr)
                take_profit = market_data.close - (4 * atr)
            
            self.last_signal_time[symbol] = datetime.now()
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=abs(avg_sentiment),
                entry_price=market_data.close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=0,
                reasoning={
                    'strategy': 'free_news_sentiment',
                    'sentiment': avg_sentiment,
                    'news_count': len(news_items),
                    'sources': [item.source for item in news_items[:3]]
                },
                confidence=confidence
            )
        
        return None

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
    reason = request.get('reason', 'manual_close')
    
    # Find the position
    position = None
    if position_type == 'simulated':
        position = trading_engine.simulated_positions.get(symbol)
    else:
        position = trading_engine.positions.get(symbol)
    
    if not position:
        return {"status": "error", "message": f"Position not found for {symbol}"}
    
    # Add commentary about manual close request
    trading_engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.DECISION,
        symbol=symbol,
        title=f"🔧 Manual Close Requested",
        message=f"User requested to close {symbol} position",
        importance=8
    ))
    
    # Trigger the close (which will show confirmation if enabled)
    try:
        await trading_engine._close_position_with_commentary(position, reason)
        return {"status": "success", "message": f"Close process initiated for {symbol}"}
    except Exception as e:
        logger.error(f"Error closing position {symbol}: {e}")
        return {"status": "error", "message": str(e)}

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
    """Run backtest on historical data"""
    try:
        start_date = request.get('start_date', '2023-01-01')
        end_date = request.get('end_date', '2024-01-01')
        symbols = request.get('symbols', ['AAPL', 'MSFT', 'GOOGL'])
        strategy_name = request.get('strategy', 'breakout')
        
        # Initialize backtest engine
        backtest_engine = BacktestEngine(config_manager.config)
        
        # Get historical data
        data = {}
        for symbol in symbols:
            try:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not df.empty:
                    data[symbol] = df
            except Exception as e:
                logger.error(f"Error downloading data for {symbol}: {e}")
        
        if not data:
            return {"status": "error", "message": "No data available for backtesting"}
        
        # Create strategy instance
        if strategy_name == 'breakout':
            strategy = BreakoutStrategyWithCommentary(CommentarySystem())
        elif strategy_name == 'mean_reversion':
            strategy = MeanReversionStrategyWithCommentary(CommentarySystem())
        elif strategy_name == 'momentum':
            strategy = MomentumStrategyWithCommentary(CommentarySystem())
        else:
            strategy = BreakoutStrategyWithCommentary(CommentarySystem())
        
        # Run backtest
        result = backtest_engine.run_backtest(data, strategy, start_date, end_date)
        
        # Save results
        with open(config_manager.get('paths.backtest_results'), 'w') as f:
            json.dump({
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'total_trades': result.total_trades,
                'equity_curve': result.equity_curve,
                'trade_history': result.trade_history
            }, f, indent=2)
        
        return {
            "status": "success",
            "message": "Backtest completed successfully",
            "results": {
                'total_return': f"{result.total_return:.2%}",
                'annualized_return': f"{result.annualized_return:.2%}",
                'sharpe_ratio': f"{result.sharpe_ratio:.2f}",
                'sortino_ratio': f"{result.sortino_ratio:.2f}",
                'max_drawdown': f"{result.max_drawdown:.2%}",
                'win_rate': f"{result.win_rate:.2%}",
                'profit_factor': f"{result.profit_factor:.2f}",
                'total_trades': result.total_trades
            }
        }
    except Exception as e:
        logger.error(f"Backtest error: {e}")
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
        performance_report = trading_engine.performance_monitor.get_performance_report()
        
        # Get behavioral analysis
        behavioral_patterns = trading_engine.behavioral_analyzer.analyze_trading_patterns(
            trading_engine.trade_history
        )
        
        # Get behavioral recommendations
        behavioral_recommendations = trading_engine.behavioral_analyzer.get_behavioral_recommendations()
        
        # Get portfolio optimization data
        positions = list(trading_engine.simulated_positions.values()) + list(trading_engine.real_positions.values())
        historical_data = {}  # This would be populated with actual data
        
        portfolio_weights = trading_engine.portfolio_optimizer.optimize_weights(
            positions, historical_data
        )
        
        return {
            "status": "success",
            "performance_metrics": performance_report,
            "behavioral_patterns": behavioral_patterns,
            "behavioral_recommendations": behavioral_recommendations,
            "portfolio_optimization": {
                "current_weights": portfolio_weights,
                "recommendations": performance_report.get("recommendations", [])
            },
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
        
        positions = list(trading_engine.simulated_positions.values()) + list(trading_engine.real_positions.values())
        historical_data = {}  # This would be populated with actual data
        
        optimized_weights = trading_engine.portfolio_optimizer.optimize_weights(
            positions, historical_data
        )
        
        return {
            "status": "success",
            "optimized_weights": optimized_weights,
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
        
        trading_engine.performance_monitor.record_execution(
            symbol=request["symbol"],
            expected_price=request["expected_price"],
            actual_price=request["actual_price"],
            signal_time=datetime.fromisoformat(request["signal_time"]),
            execution_time=datetime.fromisoformat(request["execution_time"]),
            order_size=request["order_size"]
        )
        
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
