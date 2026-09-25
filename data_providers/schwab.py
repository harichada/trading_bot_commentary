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


# ── v-playbook-desk-b2-final-2026-09-25 (whole-branch review, I1) ──
# Which schwab-py price-history helper serves a request shape. The old routing
# only reached a minute endpoint when `period == 1`, so every 10-day 5-minute
# request (`core/engine.py::_fit_bars_source`,
# `core/loops/desk_steps.py::fetch_today_bars5`) silently received *daily*
# candles. Route on `frequency_type` instead: Schwab's `periodType=day` supports
# period 1-5,10 with minute bars, and each schwab-py helper below asks for the
# full range the endpoint will give (~9 months of 5-minute bars, ~48 days of
# 1-minute bars) regardless of the `period` the caller passed, so `period` only
# decides whether a minute endpoint exists at all.
MINUTE_HISTORY_CALLS = {
    1: "get_price_history_every_minute",
    5: "get_price_history_every_five_minutes",
    10: "get_price_history_every_ten_minutes",
    15: "get_price_history_every_fifteen_minutes",
    30: "get_price_history_every_thirty_minutes",
}
DAILY_HISTORY_CALL = "get_price_history_every_day"
MINUTE_MAX_PERIOD = 10          # Schwab: periodType=day accepts period 1-5,10 for minute bars
DEFAULT_MINUTE_PERIOD = 1       # an unreadable period is read as the single-day default


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def history_call_name(period_type: str, period: Any, frequency_type: str,
                      frequency: Any) -> str:
    """The `schwab.client` method that serves this request shape.

    A minute request inside Schwab's minute window routes to the matching minute
    helper; everything else (a `daily` request, a minute window Schwab has no
    endpoint for, an unsupported minute frequency) routes to the daily helper —
    and that fallback is logged rather than silent, which was finding I1.
    """
    if str(frequency_type or "").strip().lower() != "minute":
        return DAILY_HISTORY_CALL
    days = _as_int(period, DEFAULT_MINUTE_PERIOD)
    name = MINUTE_HISTORY_CALLS.get(_as_int(frequency))
    if name is not None and days is not None and days <= MINUTE_MAX_PERIOD:
        return name
    logger.warning(
        "schwab: no minute endpoint for period_type=%r period=%r frequency=%r - "
        "falling back to daily candles", period_type, period, frequency)
    return DAILY_HISTORY_CALL


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
        # The resolved endpoint - not the requested shape - keys the cache: two
        # shapes that route to the same helper get the same candles, and a shape
        # that fell back to daily must never be served to a minute request (I1).
        call_name = history_call_name(period_type, period, frequency_type, frequency)
        cache_key = f"{symbol}_{call_name}"

        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data

        try:
            history_call = getattr(self.client, call_name, None)
            if not callable(history_call):
                logger.warning("schwab: client has no %s - falling back to %s",
                               call_name, DAILY_HISTORY_CALL)
                history_call = getattr(self.client, DAILY_HISTORY_CALL)
            response = history_call(symbol)

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
                        # Epoch milliseconds -> a *naive UTC* index (pandas attaches no
                        # timezone here). Every consumer that needs ET must localize UTC
                        # first and then convert - see core/engine.py::_fit_bars_source
                        # and core/loops/desk_steps.py::fetch_today_bars5.
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

            # v-same-basis-rs-2026-09-21: add netPercentChange (day % vs prior close)
            # so symbol day change uses same basis as SPY from MarketIndicesCache.
            # Schwab may return netPercentChange or netPercentChangeInDouble.
            net_change_pct = quote.get(
                'netPercentChange',
                quote.get('netPercentChangeInDouble', 0.0)
            )
            return {
                'bid': quote.get('bidPrice', 0),
                'ask': quote.get('askPrice', 0),
                'last': quote.get('lastPrice', 0),
                'volume': quote.get('totalVolume', 0),
                'high': quote.get('highPrice', 0),
                'low': quote.get('lowPrice', 0),
                'open': quote.get('openPrice', 0),
                'close': quote.get('closePrice', 0),
                'mark': quote.get('mark', 0),
                'netPercentChange': float(net_change_pct) if net_change_pct else 0.0,
            }
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return {}

    def calculate_market_breadth(self) -> Dict[str, float]:
        """Calculate market breadth using Schwab data"""
        try:
            # v-schwab-provider-movers-api-fix-2026-05-11: second copy of
            # the same Schwab movers bug fixed in analysis/screener.py:
            # old API used (`direction='up'/'down'`, `change='percent'`)
            # and old index name (`$SPX.X`). schwab-py migrated to
            # `sort_order=PERCENT_CHANGE_UP/DOWN` and index `$SPX`. This
            # call site was silently failing at DEBUG level every cycle,
            # so market-breadth metrics (advance/decline, new highs/lows)
            # have been zero or stale for weeks.
            try:
                gainers = self._get_movers('$SPX', 'PERCENT_CHANGE_UP')
                losers = self._get_movers('$SPX', 'PERCENT_CHANGE_DOWN')
            except Exception as e:
                logger.warning(f"Failed to fetch movers: {e}")
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

    def _get_movers(self, index: str, sort_order: str = 'PERCENT_CHANGE_UP') -> List[Dict]:
        """Get market movers.

        v-schwab-provider-movers-api-fix-2026-05-11: see
        calculate_market_breadth for context. Resolves the enum members
        from the client class so this is type-safe regardless of
        enforce_enums setting.
        """
        try:
            _Index = type(self.client).Movers.Index
            _SortOrder = type(self.client).Movers.SortOrder
            index_enum = getattr(_Index, index.lstrip('$'), None) or _Index(index)
            sort_enum = getattr(_SortOrder, sort_order, None) or _SortOrder(sort_order)
            response = self.client.get_movers(index_enum, sort_order=sort_enum)
            if response.status_code == 200:
                payload = response.json()
                # New API returns {'screeners': [...]} ; old returned bare list
                if isinstance(payload, dict):
                    return payload.get('screeners', []) or []
                return payload if isinstance(payload, list) else []
            logger.warning(
                "schwab_provider: get_movers(%s, %s) returned HTTP %s",
                index, sort_order, response.status_code,
            )
        except Exception as e:
            # Promote to WARNING — DEBUG-level let this break go silent.
            logger.warning(
                "schwab_provider: get_movers(%s, %s) raised %s: %s",
                index, sort_order, type(e).__name__, e,
            )
        return []
