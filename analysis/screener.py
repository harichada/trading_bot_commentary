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

            # v-watchlist-size-2026-04-29: cap raised 10 → 30 so
            # get_watchlist_symbols can serve up to Config.WATCHLIST_SIZE
            # without re-running the screener.
            scored_movers.sort(key=lambda x: x['score'], reverse=True)
            self.top_movers = scored_movers[:30]

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
        """Find additional volatile stocks using a pre-screened watchlist.

        v-watchlist-size-2026-04-29: expanded candidate list 20 → 50 names
        and dropped the per-candidate volume gate from 500k → 100k. The
        general _is_tradeable filter still applies downstream; this
        function is the SOURCING step, not the filtering step.
        """
        volatile_candidates = [
            # Mega-cap tech / AI
            'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'AMD', 'TSLA',
            'AVGO', 'ORCL', 'CRM', 'ADBE', 'NFLX', 'INTC', 'QCOM', 'CSCO',
            # AI / semiconductor / cloud
            'PLTR', 'SMCI', 'ARM', 'MU', 'AMAT', 'LRCX', 'KLAC', 'MRVL',
            # Crypto / fintech
            'COIN', 'MARA', 'RIOT', 'SQ', 'PYPL', 'HOOD', 'SOFI',
            # EV / clean energy
            'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'PLUG', 'FCEL',
            # Consumer / social
            'ROKU', 'SNAP', 'PINS', 'DKNG', 'PENN', 'FUBO', 'CHWY', 'PTON',
            # Biotech volatility
            'MRNA', 'BNTX', 'NVAX',
            # Index / sector ETFs (will be filtered by _is_tradeable
            # if they're on the ETF block-list, but kept here for sourcing)
            'TQQQ', 'SQQQ',
        ]

        volatile_stocks = []
        for symbol in volatile_candidates:
            try:
                quote = self._get_quote_data(symbol)
                # Lower bar — sourcing stage. _is_tradeable does the real
                # liquidity/spread filtering with current market context.
                if quote and quote.get('volume', 0) > 100_000:
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
                    # Schwab field is netPercentChange (the legacy
                    # netPercentChangeInDouble name returned None, which made
                    # every ticker show 0.00% on the dashboard).
                    'percent_change': quote.get('netPercentChange',
                        quote.get('netPercentChangeInDouble', 0)),
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
        """Filter for tradeable stocks.

        v-watchlist-size-2026-04-29: relaxed thresholds so pre-market and
        low-vol periods don't shrink the watchlist to 5 symbols. Filters
        are still meaningful — just calibrated for actual intraday data
        rather than full-RTH peak liquidity.
        """
        # Volume filter — pre-market sees ~10% of regular-session volume,
        # so 250k is the equivalent gate for pre-market that 2.5M would
        # be at peak. Original 500k rejected most names before 09:30.
        if mover.get('volume', 0) < 250_000:
            return False

        # Price filter ($3-$1000 — wider than $5-$500. Lower bound catches
        # popular sub-$5 names (LCID, NIO sometimes); upper accommodates
        # NVDA/AVGO/SMCI without rejection.)
        price = mover.get('last', 0)
        if price < 3 or price > 1000:
            return False

        # Spread filter — was 0.5%, now 1% to allow mid-cap intraday.
        # In tight RTH spreads stay under 0.1%; pre-market 0.5-1% is normal.
        spread = mover.get('spread', 0)
        if price > 0 and spread / price > 0.01:
            return False

        # Exclude leveraged / inverse ETFs that don't behave like stocks
        # under our trade rules. SPY/QQQ/IWM are fine to trade if they
        # ever appear, but the legacy block-list leaves them out.
        symbol = mover.get('symbol', '')
        excluded = ('SPY', 'QQQ', 'IWM', 'DIA', 'VXX', 'UVXY',
                    'TQQQ', 'SQQQ', 'TSLL', 'NVDL', 'TMF', 'TLT')
        if symbol in excluded:
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

    def get_watchlist_symbols(self, limit: int = 20) -> List[str]:
        """Get current top mover symbols for the watchlist.

        v-watchlist-size-2026-04-29: previously hardcoded to 5 — too narrow,
        the bot would burn cycles on the same handful of names. Caller now
        passes Config.WATCHLIST_SIZE so the cap is centrally tunable.
        """
        return [mover['symbol'] for mover in self.top_movers[:limit]]
