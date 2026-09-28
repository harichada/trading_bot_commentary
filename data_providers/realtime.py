import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from core.config import Config
from core.isolated_calls import run_isolated_sync
from yf_isolated import (
    yf_fast_quote,
    yf_history,
    yf_market_breadth,
    yf_multi_quotes,
)

logger = logging.getLogger('TradingBot')


class RealTimeDataProvider:
    """Provides real-time market data using yfinance - works in both simulation and live modes

    v-wedge-fix-2026-09-28 (Task 4, ruling 2): every method here used to call
    yfinance in-process and synchronously, with no timeout at all. This is the
    fallback provider the engine uses when Schwab is absent, and its calls sit
    on whatever loop or thread asked for them. yfinance rides curl_cffi, which
    can stall in a TLS write while holding the GIL and freeze the entire
    process (2026-09-28). Each public method now delegates to a module-level
    entry point in `yf_isolated` via the spawn-context pool, which can be
    killed. Failures degrade exactly as they did before: empty frame, None
    quote, empty dict, neutral breadth.
    """

    def _isolated(
        self, fn, *args, site: str, timeout: Optional[float] = None,
    ) -> Optional[Any]:
        """One bounded, killable yfinance call. None when it does not answer.

        v-wedge-fix-2026-09-28 fix round 1 (Minor 8): the budget must match the
        shape of the call. The engine's own helper allows 3 s for a quote and
        10 s for history; an inner call with a 10 s budget under an outer 3 s
        quote budget is decided by a timeout nobody asked for.
        """
        return run_isolated_sync(
            fn, *args,
            timeout=(
                Config().PROVIDER_HISTORY_TIMEOUT_SEC if timeout is None
                else timeout
            ),
            site=f"realtime_{site}",
        )

    def __init__(self, commentary_system, schwab_client=None):
        self.commentary = commentary_system
        self.schwab_client = schwab_client
        self._price_cache = {}
        self._cache_time = {}
        self._cache_ttl = 30  # Cache for 30 seconds to avoid rate limiting

    def get_market_data(self, symbol: str, period: str = "1d", interval: str = "5m", **kwargs) -> pd.DataFrame:
        """Get real market data from yfinance (isolated subprocess)"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{period}_{interval}"
            if cache_key in self._price_cache:
                cache_age = (datetime.now() - self._cache_time.get(cache_key, datetime.min)).total_seconds()
                if cache_age < self._cache_ttl:
                    return self._price_cache[cache_key]

            # Fetch real data — column titling and Adj Close removal happen in
            # the child (yf_isolated.yf_history) so only a DataFrame crosses.
            data = self._isolated(yf_history, symbol, period, interval,
                                  site="market_data")

            if data is not None and not data.empty:
                # Cache the data
                self._price_cache[cache_key] = data
                self._cache_time[cache_key] = datetime.now()

                return data

        except Exception as e:
            logger.warning(f"Failed to get real data for {symbol}: {e}")

        # Return empty DataFrame on failure
        return pd.DataFrame()

    def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get real-time quote from yfinance (isolated subprocess)"""
        try:
            # Check cache
            cache_key = f"quote_{symbol}"
            if cache_key in self._price_cache:
                cache_age = (datetime.now() - self._cache_time.get(cache_key, datetime.min)).total_seconds()
                if cache_age < 15:  # Quote cache for 15 seconds
                    return self._price_cache[cache_key]

            # The whole dict is assembled in the child: fast_info objects are
            # not picklable, and callers must never see a half-built quote.
            quote = self._isolated(
                yf_fast_quote, symbol, site="quote",
                timeout=Config().QUOTE_FETCH_TIMEOUT_SEC,
            )
            if quote is None:
                return None

            # Cache the quote
            self._price_cache[cache_key] = quote
            self._cache_time[cache_key] = datetime.now()

            return quote

        except Exception as e:
            logger.warning(f"Failed to get real quote for {symbol}: {e}")
            return None

    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Get quotes for multiple symbols efficiently (isolated subprocess)"""
        try:
            quotes = self._isolated(
                yf_multi_quotes, list(symbols), site="multi_quotes",
            )
            return quotes or {}
        except Exception as e:
            logger.warning(f"Failed to get multiple quotes: {e}")
            return {}

    def calculate_market_breadth(self) -> Dict[str, float]:
        """Calculate real market breadth from major indices and sectors"""
        try:
            # Check cache
            if 'market_breadth' in self._price_cache:
                cache_age = (datetime.now() - self._cache_time.get('market_breadth', datetime.min)).total_seconds()
                if cache_age < 60:  # Cache breadth for 1 minute
                    return self._price_cache['market_breadth']

            # Major market ETFs
            symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP']
            breadth = self._isolated(
                yf_market_breadth, symbols, site="market_breadth",
            )
            if breadth is None:
                raise ValueError("isolated breadth fetch returned nothing")

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
