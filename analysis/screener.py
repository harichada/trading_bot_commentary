import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from core.models import CommentaryType
from core.commentary import TradingCommentary

logger = logging.getLogger('TradingBot')


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
                title="\U0001f50d Stock Screener Update",
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
            except Exception as e:
                logger.debug(f"Failed to get quote for {symbol}: {e}")
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
