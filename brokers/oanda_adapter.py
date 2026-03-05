"""OANDA broker adapter — forex trading via oandapyV20.

Self-contained: does NOT import gap_fade_app.  All OANDA interaction is
implemented directly against the ``oandapyV20`` SDK.

Environment variables
---------------------
OANDA_TOKEN         Bearer token (required)
OANDA_ACCOUNT_ID    Account ID, e.g. 101-001-12345-001 (required)
OANDA_ENVIRONMENT   'practice' (default) or 'live'
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from brokers.base import AbstractBroker, BrokerConfig, OrderResult, TickStreamer
from brokers.market_hours import ForexHours, MarketHoursPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Symbol conversion
# ---------------------------------------------------------------------------

def _to_oanda(symbol: str) -> str:
    """Convert slash format to OANDA underscore format.  EUR/USD → EUR_USD."""
    return symbol.replace('/', '_')


def _from_oanda(instrument: str) -> str:
    """Convert OANDA underscore format to slash.  EUR_USD → EUR/USD."""
    return instrument.replace('_', '/')


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FOREX_MAJORS_AND_CROSSES: List[str] = [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'NZD/USD', 'USD/CAD',
    'EUR/GBP', 'EUR/JPY', 'EUR/CHF', 'EUR/AUD', 'EUR/NZD', 'EUR/CAD',
    'GBP/JPY', 'GBP/CHF', 'GBP/AUD', 'GBP/NZD', 'GBP/CAD',
    'AUD/JPY', 'AUD/NZD', 'AUD/CAD',
    'NZD/JPY', 'NZD/CAD',
    'CHF/JPY', 'CAD/JPY',
]

OANDA_INTERVAL_MAP: Dict[str, str] = {
    '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
    '1h': 'H1', '4h': 'H4', '1d': 'D', '1Day': 'D',
}


# ---------------------------------------------------------------------------
# Env-based configuration (no dependency on gap_fade_app)
# ---------------------------------------------------------------------------

def _config_from_env() -> Optional[Dict[str, str]]:
    """Read OANDA credentials from environment.  Returns None if missing."""
    token = os.environ.get('OANDA_TOKEN', '').strip()
    account_id = os.environ.get('OANDA_ACCOUNT_ID', '').strip()
    if not token or not account_id:
        return None
    env = os.environ.get('OANDA_ENVIRONMENT', 'practice').strip().lower()
    return {
        'token': token,
        'account_id': account_id,
        'environment': env,
    }


# ---------------------------------------------------------------------------
# OandaTickStreamer — SSE-based
# ---------------------------------------------------------------------------

class OandaTickStreamer(TickStreamer):
    """Real-time price stream using OANDA's Server-Sent Events endpoint.

    ``oandapyV20.endpoints.pricing.PricingStream`` is a synchronous
    generator — we run it in ``run_in_executor`` and bridge ticks to an
    ``asyncio.Queue``.
    """

    THROTTLE_INTERVAL = 0.25  # seconds between price updates per symbol

    def __init__(
        self,
        symbols: List[str],
        api_token: str,
        account_id: str,
        environment: str = 'practice',
        on_tick: Optional[Callable] = None,
    ):
        self._symbols: List[str] = list(symbols)
        self._api_token = api_token
        self._account_id = account_id
        self._environment = environment
        self._on_tick = on_tick

        self._prices: Dict[str, float] = {}
        self._connected = False
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_emit: Dict[str, float] = {}

    # -- TickStreamer interface -----------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.get_event_loop().create_task(self._stream_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._connected = False

    async def add_symbols(self, symbols: List[str]) -> None:
        for s in symbols:
            if s not in self._symbols:
                self._symbols.append(s)
        # Must restart stream — OANDA SSE doesn't support dynamic subscription
        if self._running:
            await self.stop()
            await self.start()

    async def remove_symbols(self, symbols: List[str]) -> None:
        self._symbols = [s for s in self._symbols if s not in symbols]
        if self._running:
            await self.stop()
            if self._symbols:
                await self.start()

    @property
    def latest_prices(self) -> Dict[str, float]:
        return dict(self._prices)

    @property
    def symbols(self) -> List[str]:
        return list(self._symbols)

    @property
    def connected(self) -> bool:
        return self._connected

    # -- Internal ------------------------------------------------------------

    async def _stream_loop(self) -> None:
        """Run the SSE stream in a thread and dispatch ticks."""
        import oandapyV20
        import oandapyV20.endpoints.pricing as pricing

        loop = asyncio.get_event_loop()
        backoff = 3

        while self._running:
            try:
                api = oandapyV20.API(
                    access_token=self._api_token,
                    environment=self._environment,
                )
                instruments = ','.join(_to_oanda(s) for s in self._symbols)
                ep = pricing.PricingStream(
                    accountID=self._account_id,
                    params={'instruments': instruments},
                )

                def _blocking_stream():
                    """Synchronous generator consumer."""
                    for tick in api.request(ep):
                        if not self._running:
                            break
                        yield tick

                self._connected = True
                gen = _blocking_stream()

                while self._running:
                    tick = await loop.run_in_executor(None, next, gen, None)
                    if tick is None:
                        break
                    if tick.get('type') != 'PRICE':
                        continue

                    instrument = tick.get('instrument', '')
                    symbol = _from_oanda(instrument)
                    bid = float(tick.get('bids', [{}])[0].get('price', 0))
                    ask = float(tick.get('asks', [{}])[0].get('price', 0))
                    if bid <= 0 or ask <= 0:
                        continue
                    mid = (bid + ask) / 2
                    self._prices[symbol] = mid

                    # Throttle callback
                    now = time.monotonic()
                    last = self._last_emit.get(symbol, 0)
                    if now - last >= self.THROTTLE_INTERVAL and self._on_tick:
                        self._last_emit[symbol] = now
                        try:
                            self._on_tick(symbol, mid, 0)
                        except Exception:
                            logger.exception("on_tick callback error")

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("OANDA stream error — reconnecting in %ds", backoff)
                self._connected = False
                await asyncio.sleep(backoff)

        self._connected = False


# ---------------------------------------------------------------------------
# OandaBrokerAdapter
# ---------------------------------------------------------------------------

class OandaBrokerAdapter(AbstractBroker):
    """OANDA forex broker adapter via ``oandapyV20`` SDK.

    Lazy SDK init: ``self._api`` is created on first use via ``_get_api()``.
    """

    def __init__(self, config: Optional[BrokerConfig] = None):
        if config is None:
            config = BrokerConfig(broker_id='oanda', environment='paper')
        super().__init__(config)
        self._api = None
        self._env_config: Optional[Dict[str, str]] = None
        self._market_hours = ForexHours()

    # -- Lazy init -----------------------------------------------------------

    def _get_env(self) -> Optional[Dict[str, str]]:
        """Cached env config lookup."""
        if self._env_config is None:
            self._env_config = _config_from_env()
        return self._env_config

    def _get_api(self):
        """Lazy-init oandapyV20.API client."""
        if self._api is None:
            import oandapyV20
            env = self._get_env()
            if env is None:
                raise RuntimeError("OANDA not configured (set OANDA_TOKEN + OANDA_ACCOUNT_ID)")
            self._api = oandapyV20.API(
                access_token=env['token'],
                environment=env['environment'],
            )
        return self._api

    @property
    def _account_id(self) -> str:
        env = self._get_env()
        return env['account_id'] if env else ''

    # -- Connection / Account ------------------------------------------------

    def is_configured(self) -> bool:
        return self._get_env() is not None

    async def get_account(self) -> Optional[Dict]:
        import oandapyV20.endpoints.accounts as accounts

        try:
            api = self._get_api()
            ep = accounts.AccountDetails(accountID=self._account_id)
            resp = await asyncio.to_thread(api.request, ep)
            acct = resp.get('account', {})
            return {
                'equity': acct.get('NAV', '0'),
                'balance': acct.get('balance', '0'),
                'buying_power': acct.get('marginAvailable', '0'),
                'cash': acct.get('balance', '0'),
                'portfolio_value': acct.get('NAV', '0'),
                'unrealized_pl': acct.get('unrealizedPL', '0'),
                'broker_id': 'oanda',
                'raw': acct,
            }
        except Exception as e:
            logger.error("OANDA get_account failed: %s", e)
            return None

    async def get_positions(self) -> List[Dict]:
        import oandapyV20.endpoints.positions as positions_ep

        try:
            api = self._get_api()
            ep = positions_ep.OpenPositions(accountID=self._account_id)
            resp = await asyncio.to_thread(api.request, ep)

            result: List[Dict] = []
            for pos in resp.get('positions', []):
                instrument = pos.get('instrument', '')
                symbol = _from_oanda(instrument)

                # OANDA has separate long/short sub-objects
                long_units = int(pos.get('long', {}).get('units', '0'))
                short_units = int(pos.get('short', {}).get('units', '0'))

                if long_units > 0:
                    avg_price = pos.get('long', {}).get('averagePrice', '0')
                    result.append({
                        'symbol': symbol,
                        'qty': float(long_units),
                        'avg_entry_price': float(avg_price),
                        'side': 'long',
                        'broker_id': 'oanda',
                        'raw': pos,
                    })
                if short_units != 0:
                    avg_price = pos.get('short', {}).get('averagePrice', '0')
                    result.append({
                        'symbol': symbol,
                        'qty': float(abs(short_units)),
                        'avg_entry_price': float(avg_price),
                        'side': 'short',
                        'broker_id': 'oanda',
                        'raw': pos,
                    })

            return result
        except Exception as e:
            logger.error("OANDA get_positions failed: %s", e)
            return []

    # -- Order Execution -----------------------------------------------------

    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        timeout_sec: float = 30.0,
    ) -> OrderResult:
        import oandapyV20.endpoints.orders as orders_ep

        try:
            api = self._get_api()
            instrument = _to_oanda(symbol)

            # OANDA uses signed units: positive = buy, negative = sell
            signed_units = str(int(qty)) if side == 'buy' else str(-int(qty))

            if order_type == 'limit' and limit_price is not None:
                order_body = {
                    'order': {
                        'type': 'LIMIT',
                        'instrument': instrument,
                        'units': signed_units,
                        'price': f'{limit_price:.5f}',
                        'timeInForce': 'GTC',
                    }
                }
            else:
                order_body = {
                    'order': {
                        'type': 'MARKET',
                        'instrument': instrument,
                        'units': signed_units,
                    }
                }

            ep = orders_ep.OrderCreate(accountID=self._account_id, data=order_body)
            resp = await asyncio.to_thread(api.request, ep)

            # Market orders fill immediately
            if order_type != 'limit':
                fill = resp.get('orderFillTransaction', {})
                if fill:
                    return OrderResult(
                        order_id=fill.get('id', ''),
                        status='filled',
                        filled_qty=abs(float(fill.get('units', 0))),
                        filled_avg_price=float(fill.get('price', 0)),
                        symbol=symbol,
                        side=side,
                    )

            # Limit order — poll for fill
            order_create_tx = resp.get('orderCreateTransaction', {})
            order_id = order_create_tx.get('id', '')
            if not order_id:
                return OrderResult(
                    status='error', symbol=symbol, side=side,
                    error='No order ID in OANDA response',
                )

            return await self._poll_order_fill(order_id, symbol, side, timeout_sec)

        except Exception as e:
            logger.error("OANDA submit_order failed: %s", e)
            return OrderResult(
                status='error', symbol=symbol, side=side, error=str(e),
            )

    async def _poll_order_fill(
        self, order_id: str, symbol: str, side: str, timeout_sec: float,
    ) -> OrderResult:
        """Poll a pending order until filled or timeout."""
        import oandapyV20.endpoints.orders as orders_ep

        deadline = time.monotonic() + timeout_sec
        api = self._get_api()

        while time.monotonic() < deadline:
            try:
                ep = orders_ep.OrderDetails(
                    accountID=self._account_id, orderID=order_id,
                )
                resp = await asyncio.to_thread(api.request, ep)
                order = resp.get('order', {})
                state = order.get('state', '').upper()

                if state == 'FILLED':
                    return OrderResult(
                        order_id=order_id,
                        status='filled',
                        filled_qty=abs(float(order.get('filledUnits', order.get('units', 0)))),
                        filled_avg_price=float(order.get('price', order.get('averagePrice', 0))),
                        symbol=symbol,
                        side=side,
                    )
                if state in ('CANCELLED', 'TRIGGERED'):
                    return OrderResult(
                        order_id=order_id, status='cancelled',
                        symbol=symbol, side=side,
                    )
            except Exception as e:
                logger.warning("OANDA poll order %s error: %s", order_id, e)

            await asyncio.sleep(0.5)

        return OrderResult(
            order_id=order_id, status='timeout', symbol=symbol, side=side,
            error=f'Order {order_id} not filled within {timeout_sec}s',
        )

    async def place_stop_order(
        self,
        symbol: str,
        qty: float,
        stop_price: float,
        limit_offset_pct: float = 0.003,
        direction: str = 'short',
    ) -> Dict:
        import oandapyV20.endpoints.orders as orders_ep

        try:
            api = self._get_api()
            instrument = _to_oanda(symbol)

            # Stop order to close a position:
            # Closing a short → buy (positive units)
            # Closing a long  → sell (negative units)
            if direction == 'short':
                signed_units = str(int(qty))
                limit_price = stop_price * (1 + limit_offset_pct)
            else:
                signed_units = str(-int(qty))
                limit_price = stop_price * (1 - limit_offset_pct)

            order_body = {
                'order': {
                    'type': 'STOP',
                    'instrument': instrument,
                    'units': signed_units,
                    'price': f'{stop_price:.5f}',
                    'priceBound': f'{limit_price:.5f}',
                    'timeInForce': 'GTC',
                }
            }

            ep = orders_ep.OrderCreate(accountID=self._account_id, data=order_body)
            resp = await asyncio.to_thread(api.request, ep)

            order_tx = resp.get('orderCreateTransaction', {})
            return {
                'id': order_tx.get('id', ''),
                'status': 'new',
                'symbol': symbol,
                'stop_price': f'{stop_price:.5f}',
                'limit_price': f'{limit_price:.5f}',
            }

        except Exception as e:
            logger.error("OANDA place_stop_order failed: %s", e)
            return {'error': str(e)}

    async def cancel_order(self, order_id: str) -> bool:
        import oandapyV20.endpoints.orders as orders_ep

        try:
            api = self._get_api()
            ep = orders_ep.OrderCancel(
                accountID=self._account_id, orderID=order_id,
            )
            await asyncio.to_thread(api.request, ep)
            return True
        except Exception as e:
            logger.error("OANDA cancel_order %s failed: %s", order_id, e)
            return False

    async def get_order(self, order_id: str) -> Optional[Dict]:
        import oandapyV20.endpoints.orders as orders_ep

        try:
            api = self._get_api()
            ep = orders_ep.OrderDetails(
                accountID=self._account_id, orderID=order_id,
            )
            resp = await asyncio.to_thread(api.request, ep)
            order = resp.get('order', {})
            state = order.get('state', '').upper()

            status_map = {
                'PENDING': 'new',
                'FILLED': 'filled',
                'CANCELLED': 'cancelled',
                'TRIGGERED': 'triggered',
            }

            return {
                'id': order.get('id', order_id),
                'status': status_map.get(state, state.lower()),
                'symbol': _from_oanda(order.get('instrument', '')),
                'units': order.get('units', '0'),
                'price': order.get('price', '0'),
                'raw': order,
            }
        except Exception as e:
            logger.error("OANDA get_order %s failed: %s", order_id, e)
            return None

    async def replace_stop_order(
        self,
        old_order_id: str,
        symbol: str,
        qty: float,
        new_stop_price: float,
        limit_offset_pct: float = 0.003,
    ) -> Dict:
        cancelled = await self.cancel_order(old_order_id)
        if not cancelled:
            return {'error': f'Failed to cancel old order {old_order_id}'}

        return await self.place_stop_order(
            symbol, qty, new_stop_price, limit_offset_pct,
        )

    # -- Market Data ---------------------------------------------------------

    def create_tick_streamer(
        self,
        symbols: List[str],
        on_tick: Optional[Callable] = None,
    ) -> TickStreamer:
        env = self._get_env()
        if env is None:
            raise RuntimeError("OANDA not configured")
        return OandaTickStreamer(
            symbols=symbols,
            api_token=env['token'],
            account_id=env['account_id'],
            environment=env['environment'],
            on_tick=on_tick,
        )

    async def fetch_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1Day',
    ) -> Optional[pd.DataFrame]:
        import oandapyV20.endpoints.instruments as instruments_ep

        granularity = OANDA_INTERVAL_MAP.get(interval)
        if granularity is None:
            logger.warning("OANDA: interval '%s' not supported", interval)
            return None

        try:
            api = self._get_api()
            instrument = _to_oanda(symbol)

            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
            # Cap at current UTC time — OANDA rejects future timestamps
            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            if end_dt > now_utc:
                end_dt = now_utc

            params = {
                'granularity': granularity,
                'from': start_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'to': end_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'price': 'M',  # mid prices
            }

            ep = instruments_ep.InstrumentsCandles(
                instrument=instrument, params=params,
            )
            resp = await asyncio.to_thread(api.request, ep)

            candles = resp.get('candles', [])
            if not candles:
                logger.warning("OANDA: 0 candles for %s %s", symbol, interval)
                return None

            rows = []
            for c in candles:
                if not c.get('complete', True):
                    continue
                mid = c.get('mid', {})
                rows.append({
                    'datetime': c['time'],
                    'open': float(mid.get('o', 0)),
                    'high': float(mid.get('h', 0)),
                    'low': float(mid.get('l', 0)),
                    'close': float(mid.get('c', 0)),
                    'volume': int(c.get('volume', 0)),
                })

            if not rows:
                logger.warning("OANDA: no complete candles for %s %s", symbol, interval)
                return None

            import numpy as np
            df = pd.DataFrame(rows)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
            df = df.dropna()
            df.index.name = symbol

            logger.info("OANDA: fetched %d bars for %s (%s)", len(df), symbol, interval)
            return df

        except Exception as e:
            logger.error("OANDA fetch_bars failed for %s: %s", symbol, e)
            return None

    async def fetch_bars_multi(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = '1Day',
    ) -> Dict[str, pd.DataFrame]:
        # OANDA has no multi-symbol endpoint — sequential per-symbol
        result: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = await self.fetch_bars(symbol, start_date, end_date, interval)
            if df is not None and not df.empty:
                result[symbol] = df
        return result

    # -- Symbol Info ----------------------------------------------------------

    async def check_tradeable(self, symbol: str) -> Tuple[bool, bool]:
        import oandapyV20.endpoints.accounts as accounts_ep

        try:
            api = self._get_api()
            instrument = _to_oanda(symbol)
            ep = accounts_ep.AccountInstruments(
                accountID=self._account_id,
                params={'instruments': instrument},
            )
            resp = await asyncio.to_thread(api.request, ep)
            instruments = resp.get('instruments', [])
            if instruments:
                # Forex is always bidirectional
                return True, True
            return False, False
        except Exception as e:
            logger.error("OANDA check_tradeable %s failed: %s", symbol, e)
            return False, False

    async def get_tradeable_symbols(
        self,
        min_price: float = 1.0,
        max_price: float = 0.0,
    ) -> List[str]:
        import oandapyV20.endpoints.accounts as accounts_ep

        try:
            api = self._get_api()
            ep = accounts_ep.AccountInstruments(accountID=self._account_id)
            resp = await asyncio.to_thread(api.request, ep)
            instruments = resp.get('instruments', [])

            symbols = []
            for inst in instruments:
                if inst.get('type') == 'CURRENCY':
                    symbols.append(_from_oanda(inst['name']))
            return symbols if symbols else list(FOREX_MAJORS_AND_CROSSES)

        except Exception as e:
            logger.error("OANDA get_tradeable_symbols failed: %s", e)
            return list(FOREX_MAJORS_AND_CROSSES)

    # -- Snapshot Prices -----------------------------------------------------

    async def get_snapshot_prices(self, symbols: List[str]) -> Dict[str, float]:
        import oandapyV20.endpoints.pricing as pricing_ep

        try:
            api = self._get_api()
            instruments = ','.join(_to_oanda(s) for s in symbols)
            ep = pricing_ep.PricingInfo(
                accountID=self._account_id,
                params={'instruments': instruments},
            )
            resp = await asyncio.to_thread(api.request, ep)

            prices: Dict[str, float] = {}
            for p in resp.get('prices', []):
                instrument = p.get('instrument', '')
                symbol = _from_oanda(instrument)
                bids = p.get('bids', [])
                asks = p.get('asks', [])
                if bids and asks:
                    bid = float(bids[0].get('price', 0))
                    ask = float(asks[0].get('price', 0))
                    if bid > 0 and ask > 0:
                        prices[symbol] = (bid + ask) / 2
            return prices

        except Exception as e:
            logger.error("OANDA get_snapshot_prices failed: %s", e)
            return {}

    # -- Market Hours --------------------------------------------------------

    def get_market_hours(self) -> MarketHoursPolicy:
        return self._market_hours
