import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from core.models import CommentaryType
from core.commentary import TradingCommentary

logger = logging.getLogger('TradingBot')


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
