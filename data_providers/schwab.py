import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from core.models import CommentaryType
from core.commentary import TradingCommentary

try:
    from schwab import auth, client as schwab_client
except ImportError:
    schwab_client = None

try:
    from circuit_breaker import api_circuit_breaker
except ImportError:
    def api_circuit_breaker(func):
        return func

logger = logging.getLogger('TradingBot')


class SchwabDataProvider:
    """Schwab data provider with commentary"""

    def __init__(self, client, commentary_system):
        self.client = client
        self._cache = {}
        self._cache_timeout = 60
        self.commentary = commentary_system
        self.market_hours_cache = {}

    @api_circuit_breaker
    def get_market_data(self, symbol: str, period_type: str = 'day',
                       period: int = 1, frequency_type: str = 'minute',
                       frequency: int = 5) -> pd.DataFrame:
        """Get market data using Schwab API with circuit breaker protection"""
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

    @api_circuit_breaker
    def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get real-time quote with circuit breaker protection"""
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
            except Exception as e:
                logger.debug(f"Failed to fetch movers: {e}")
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
                    except Exception as e:
                        logger.debug(f"VIX index endpoint failed: {e}")
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
        except Exception as e:
            logger.debug(f"Failed to get movers for {index} {direction}: {e}")
        return []
