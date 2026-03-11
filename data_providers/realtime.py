import logging
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd

logger = logging.getLogger('TradingBot')


class RealTimeDataProvider:
    """Provides real-time market data using yfinance - works in both simulation and live modes"""

    def __init__(self, commentary_system, schwab_client=None):
        self.commentary = commentary_system
        self.schwab_client = schwab_client
        self._price_cache = {}
        self._cache_time = {}
        self._cache_ttl = 30  # Cache for 30 seconds to avoid rate limiting

    def get_market_data(self, symbol: str, period: str = "1d", interval: str = "5m", **kwargs) -> pd.DataFrame:
        """Get real market data from yfinance"""
        try:
            import yfinance as yf

            # Check cache first
            cache_key = f"{symbol}_{period}_{interval}"
            if cache_key in self._price_cache:
                cache_age = (datetime.now() - self._cache_time.get(cache_key, datetime.min)).total_seconds()
                if cache_age < self._cache_ttl:
                    return self._price_cache[cache_key]

            # Fetch real data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if not data.empty:
                # Ensure proper column names
                data.columns = [col.title() for col in data.columns]
                if 'Adj Close' in data.columns:
                    data = data.drop('Adj Close', axis=1, errors='ignore')

                # Cache the data
                self._price_cache[cache_key] = data
                self._cache_time[cache_key] = datetime.now()

                return data

        except Exception as e:
            logger.warning(f"Failed to get real data for {symbol}: {e}")

        # Return empty DataFrame on failure
        return pd.DataFrame()

    def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get real-time quote from yfinance"""
        try:
            import yfinance as yf

            # Check cache
            cache_key = f"quote_{symbol}"
            if cache_key in self._price_cache:
                cache_age = (datetime.now() - self._cache_time.get(cache_key, datetime.min)).total_seconds()
                if cache_age < 15:  # Quote cache for 15 seconds
                    return self._price_cache[cache_key]

            ticker = yf.Ticker(symbol)
            info = ticker.fast_info

            quote = {
                'bid': getattr(info, 'bid', 0) or info.last_price * 0.9999,
                'ask': getattr(info, 'ask', 0) or info.last_price * 1.0001,
                'last': info.last_price,
                'volume': info.last_volume or 0,
                'high': info.day_high or info.last_price,
                'low': info.day_low or info.last_price,
                'open': info.open or info.last_price,
                'close': info.previous_close or info.last_price,
                'market_cap': getattr(info, 'market_cap', 0) or 0
            }

            # Cache the quote
            self._price_cache[cache_key] = quote
            self._cache_time[cache_key] = datetime.now()

            return quote

        except Exception as e:
            logger.warning(f"Failed to get real quote for {symbol}: {e}")
            return None

    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Get quotes for multiple symbols efficiently"""
        quotes = {}
        try:
            import yfinance as yf

            # Batch download for efficiency
            tickers = yf.Tickers(' '.join(symbols))

            for symbol in symbols:
                try:
                    ticker = tickers.tickers.get(symbol)
                    if ticker:
                        info = ticker.fast_info
                        quotes[symbol] = {
                            'last': info.last_price,
                            'volume': info.last_volume or 0,
                            'high': info.day_high or info.last_price,
                            'low': info.day_low or info.last_price,
                            'open': info.open or info.last_price,
                            'change_pct': ((info.last_price - info.previous_close) / info.previous_close * 100) if info.previous_close else 0
                        }
                except Exception as e:
                    logger.debug(f"Failed to get quote for {symbol}: {e}")

        except Exception as e:
            logger.warning(f"Failed to get multiple quotes: {e}")

        return quotes

    def calculate_market_breadth(self) -> Dict[str, float]:
        """Calculate real market breadth from major indices and sectors"""
        try:
            import yfinance as yf

            # Check cache
            if 'market_breadth' in self._price_cache:
                cache_age = (datetime.now() - self._cache_time.get('market_breadth', datetime.min)).total_seconds()
                if cache_age < 60:  # Cache breadth for 1 minute
                    return self._price_cache['market_breadth']

            # Major market ETFs
            symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP']
            advances = 0
            declines = 0

            tickers = yf.Tickers(' '.join(symbols))
            for symbol in symbols:
                try:
                    ticker = tickers.tickers.get(symbol)
                    if ticker:
                        info = ticker.fast_info
                        if info.last_price > info.previous_close:
                            advances += 1
                        else:
                            declines += 1
                except Exception as e:
                    logger.debug(f"Failed to get ticker data for {symbol}: {e}")

            total = advances + declines
            breadth = {
                'advance_decline_ratio': advances / max(declines, 1),
                'advancing_percent': (advances / total * 100) if total > 0 else 50,
                'declining_percent': (declines / total * 100) if total > 0 else 50,
                'advances': advances,
                'declines': declines
            }

            self._price_cache['market_breadth'] = breadth
            self._cache_time['market_breadth'] = datetime.now()

            return breadth

        except Exception as e:
            logger.warning(f"Failed to calculate market breadth: {e}")
            return {
                'advance_decline_ratio': 1.0,
                'advancing_percent': 50,
                'declining_percent': 50,
                'advances': 0,
                'declines': 0
            }


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
