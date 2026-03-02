#!/usr/bin/env python3
"""
Gap Fade Trading Strategy — Standalone FastAPI App

Data-driven gap fade strategy: short gap-ups that occur on below-average volume.
Study of 3,754 Alpaca SIP events (5 years, 156 stocks) showed gap-ups on low volume
(<1x avg) fade 71-81% with +2.5% avg P&L. High-volume gaps (>3x) only fade 31%.

Sections:
  1. Imports & Config
  2. Alpaca Helpers
  3. Data Classes
  4. Scanner
  5. Strategy Engine
  6. Backtester
  7. Live Trader
  8. WebSocket & Broadcast
  9. FastAPI Endpoints
 10. Dashboard HTML
 11. Main
"""

# =============================================================================
# SECTION 1: IMPORTS & CONFIG
# =============================================================================

import asyncio
import json
import logging
import os
import re
import sqlite3
import subprocess
import time as _time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone, date
from itertools import groupby
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import aiohttp
import feedparser
import yfinance as yf

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn

logger = logging.getLogger('GapFadeApp')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

# App version from git tag (e.g. v5-loop-fixes -> v5.0)
def _get_version() -> str:
    try:
        tag = subprocess.check_output(
            ['git', 'describe', '--tags', '--abbrev=0'],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        # Extract version number: "v5-loop-fixes" -> "5"
        num = tag.lstrip('v').split('-')[0]
        return f'v{num}.0'
    except Exception:
        return 'v0.0'

APP_VERSION = _get_version()

# Eastern timezone offset (UTC-5 standard, UTC-4 DST)
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo('America/New_York')
except ImportError:
    import pytz
    ET = pytz.timezone('America/New_York')

# =============================================================================
# SECTION 2: ALPACA HELPERS
# =============================================================================

def _load_env_file():
    """Load credentials from .env file if env vars are missing."""
    env_path = os.path.join(os.path.dirname(__file__) or '.', '.env')
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value and not os.environ.get(key):
                    os.environ[key] = value
        logger.debug("Loaded credentials from .env")
    except Exception as e:
        logger.debug(f"Could not load .env: {e}")


def _get_alpaca_config() -> Optional[dict]:
    """Get Alpaca paper trading config from env vars."""
    _load_env_file()
    api_key = os.environ.get('ALPACA_API_KEY', '')
    secret_key = os.environ.get('ALPACA_SECRET_KEY', '')
    if not api_key or not secret_key:
        return None
    return {
        'api_key': api_key,
        'secret_key': secret_key,
        'base_url': 'https://paper-api.alpaca.markets',
        'data_url': 'https://data.alpaca.markets',
    }


def _alpaca_headers(cfg: dict) -> dict:
    return {
        'APCA-API-KEY-ID': cfg['api_key'],
        'APCA-API-SECRET-KEY': cfg['secret_key'],
    }


def fetch_alpaca_bars(symbol: str, start_date: str, end_date: str,
                      interval: str = '1Day', feed: str = 'iex') -> Optional[pd.DataFrame]:
    """Fetch historical bars from Alpaca Data API v2 with pagination."""
    cfg = _get_alpaca_config()
    if cfg is None:
        return None

    interval_map = {
        '1m': '1Min', '5m': '5Min', '10m': '10Min', '15m': '15Min',
        '30m': '30Min', '1h': '1Hour', '1d': '1Day', '1Day': '1Day',
        '1Min': '1Min',
    }
    timeframe = interval_map.get(interval, interval)
    headers = _alpaca_headers(cfg)

    start_rfc = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%dT00:00:00Z')
    end_rfc = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')

    all_bars = []
    page_token = None
    url = f'{cfg["data_url"]}/v2/stocks/{symbol}/bars'

    retries_429 = 0
    try:
        while True:
            params = {
                'timeframe': timeframe,
                'start': start_rfc,
                'end': end_rfc,
                'limit': 10000,
                'feed': feed,
                'adjustment': 'split',
            }
            if page_token:
                params['page_token'] = page_token

            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code == 429:
                retries_429 += 1
                if retries_429 > 5:
                    logger.warning(f"Alpaca rate limit exceeded for {symbol} after 5 retries")
                    return None
                _time.sleep(min(2 ** retries_429, 30))
                continue
            retries_429 = 0
            if resp.status_code != 200:
                logger.error(f"Alpaca API returned {resp.status_code}: {resp.text[:200]}")
                return None

            data = resp.json()
            bars = data.get('bars') or []
            all_bars.extend(bars)

            page_token = data.get('next_page_token')
            if not page_token:
                break

        if not all_bars:
            return None

        df = pd.DataFrame(all_bars)
        df['datetime'] = pd.to_datetime(df['t'])
        df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].dropna()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df.index.name = symbol
        return df

    except Exception as e:
        logger.error(f"Alpaca data fetch failed for {symbol}: {e}")
        return None


def fetch_alpaca_bars_multi(symbols: List[str], start_date: str, end_date: str,
                            interval: str = '1Day', feed: str = 'iex',
                            batch_size: int = 100) -> Dict[str, pd.DataFrame]:
    """Fetch daily bars for many symbols at once using Alpaca multi-stock bars endpoint.

    Returns {symbol: DataFrame} dict.  Much faster than per-symbol fetching.
    Batches `batch_size` symbols per request (Alpaca limit).
    """
    cfg = _get_alpaca_config()
    if cfg is None:
        return {}

    interval_map = {
        '1m': '1Min', '5m': '5Min', '10m': '10Min', '15m': '15Min',
        '30m': '30Min', '1h': '1Hour', '1d': '1Day', '1Day': '1Day',
    }
    timeframe = interval_map.get(interval, interval)
    headers = _alpaca_headers(cfg)
    start_rfc = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%dT00:00:00Z')
    end_rfc = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')

    url = f'{cfg["data_url"]}/v2/stocks/bars'
    results: Dict[str, pd.DataFrame] = {}
    total_batches = (len(symbols) + batch_size - 1) // batch_size

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        batch_num = i // batch_size
        batch_bars: Dict[str, list] = defaultdict(list)

        # Retry loop for timeouts
        for attempt in range(3):
            page_token = None
            if attempt > 0:
                batch_bars.clear()
                _time.sleep(2 ** attempt)

            try:
                while True:
                    params = {
                        'symbols': ','.join(batch),
                        'timeframe': timeframe,
                        'start': start_rfc,
                        'end': end_rfc,
                        'limit': 10000,
                        'feed': feed,
                        'adjustment': 'split',
                    }
                    if page_token:
                        params['page_token'] = page_token

                    resp = requests.get(url, headers=headers, params=params, timeout=45)
                    if resp.status_code == 429:
                        _time.sleep(min(2 ** (attempt + 1), 30))
                        continue
                    if resp.status_code != 200:
                        logger.warning(f"Multi-bar batch {batch_num} failed ({resp.status_code})")
                        break

                    data = resp.json()
                    bars_dict = data.get('bars') or {}
                    for sym, bars in bars_dict.items():
                        batch_bars[sym].extend(bars)

                    page_token = data.get('next_page_token')
                    if not page_token:
                        break

                # If we got data, don't retry
                if batch_bars:
                    break

            except requests.exceptions.Timeout:
                logger.warning(f"Multi-bar batch {batch_num}/{total_batches} timed out (attempt {attempt+1}/3)")
            except Exception as e:
                logger.warning(f"Multi-bar batch {batch_num} error: {e}")
                break  # Don't retry on non-timeout errors

        # Convert each symbol's bars to DataFrame
        for sym, bars in batch_bars.items():
            if not bars:
                continue
            try:
                df = pd.DataFrame(bars)
                df['datetime'] = pd.to_datetime(df['t'])
                df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low',
                                   'c': 'close', 'v': 'volume'}, inplace=True)
                df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].dropna()
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
                df.set_index('datetime', inplace=True)
                df.sort_index(inplace=True)
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                df.index.name = sym
                results[sym] = df
            except Exception:
                pass

        # Rate limit between batches
        if i + batch_size < len(symbols):
            _time.sleep(0.3)

        # Log progress every 20 batches
        if batch_num % 20 == 0 and batch_num > 0:
            logger.info(f"Multi-bar progress: batch {batch_num}/{total_batches}, {len(results)} symbols so far")

    logger.info(f"Multi-bar fetch: {len(results)} symbols with data out of {len(symbols)} requested")
    return results


def fetch_alpaca_snapshots(symbols: List[str]) -> dict:
    """Batch fetch latest quotes/trades via Alpaca snapshots endpoint.
    Returns {symbol: {latestTrade: {p, s, t}, dailyBar: {o,h,l,c,v}, prevDailyBar: {o,h,l,c,v}}}
    """
    cfg = _get_alpaca_config()
    if cfg is None:
        return {}

    headers = _alpaca_headers(cfg)
    results = {}

    # Batch 500 symbols per request (Alpaca supports ~2000 per URL, 500 is safe)
    for i in range(0, len(symbols), 500):
        batch = symbols[i:i+500]
        params = {
            'symbols': ','.join(batch),
            'feed': 'iex',
        }
        try:
            resp = requests.get(
                f'{cfg["data_url"]}/v2/stocks/snapshots',
                headers=headers, params=params, timeout=15
            )
            if resp.status_code == 200:
                results.update(resp.json())
            else:
                logger.warning(f"Snapshot batch failed ({resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Snapshot fetch error: {e}")

        if i + 500 < len(symbols):
            _time.sleep(0.1)

    return results


# Module-level cache for Alpaca assets list
_assets_cache: Dict[str, Any] = {'symbols': [], 'timestamp': 0.0}


def fetch_alpaca_assets(min_price: float = 1.0,
                        asset_class: str = 'us_equity') -> List[str]:
    """Fetch all active, tradeable symbols from Alpaca.

    Filters: active, tradeable, major exchanges (NYSE, NASDAQ, ARCA, AMEX, BATS).
    Cached for 24 hours.
    """
    now = _time.time()
    if _assets_cache['symbols'] and (now - _assets_cache['timestamp']) < 86400:
        return _assets_cache['symbols']

    cfg = _get_alpaca_config()
    if cfg is None:
        logger.warning("fetch_alpaca_assets: Alpaca not configured")
        return []

    headers = _alpaca_headers(cfg)
    valid_exchanges = {'NYSE', 'NASDAQ', 'ARCA', 'AMEX', 'BATS', 'NYSEARCA'}

    try:
        resp = requests.get(
            f'{cfg["base_url"]}/v2/assets',
            headers=headers,
            params={'status': 'active', 'asset_class': asset_class},
            timeout=60
        )
        if resp.status_code != 200:
            logger.error(f"Alpaca assets API returned {resp.status_code}: {resp.text[:200]}")
            return []

        assets = resp.json()
        symbols = []
        for a in assets:
            if not a.get('tradable', False):
                continue
            if a.get('status') != 'active':
                continue
            exchange = a.get('exchange', '')
            if exchange not in valid_exchanges:
                continue
            sym = a.get('symbol', '')
            # Skip symbols with special characters (warrants, units, etc.)
            if not sym or '/' in sym or '.' in sym or '-' in sym or len(sym) > 5:
                continue
            symbols.append(sym)

        symbols = sorted(set(symbols))
        _assets_cache['symbols'] = symbols
        _assets_cache['timestamp'] = now
        logger.info(f"Fetched {len(symbols)} tradeable assets from Alpaca")
        return symbols

    except Exception as e:
        logger.error(f"fetch_alpaca_assets failed: {e}")
        return []


def alpaca_check_shortable(symbol: str) -> Tuple[bool, bool]:
    """Check if symbol is shortable on Alpaca. Returns (shortable, easy_to_borrow)."""
    cfg = _get_alpaca_config()
    if cfg is None:
        return False, False

    headers = {**_alpaca_headers(cfg), 'Content-Type': 'application/json'}
    try:
        resp = requests.get(
            f'{cfg["base_url"]}/v2/assets/{symbol}',
            headers=headers, timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get('shortable', False), data.get('easy_to_borrow', False)
    except Exception as e:
        logger.warning(f"Shortable check failed for {symbol}: {e}")
    return False, False


@dataclass
class OrderResult:
    """Structured result from order submission + fill verification."""
    order_id: str = ''
    status: str = ''               # 'filled', 'partially_filled', 'rejected', 'cancelled', 'error', 'timeout'
    filled_qty: int = 0
    filled_avg_price: float = 0.0
    symbol: str = ''
    side: str = ''
    error: str = ''

    @property
    def is_filled(self) -> bool:
        return self.status == 'filled'

    @property
    def is_error(self) -> bool:
        return self.status in ('error', 'rejected', 'cancelled', 'timeout')


def alpaca_place_order(symbol: str, qty: int, side: str, order_type: str = 'market',
                       limit_price: float = None, stop_price: float = None,
                       time_in_force: str = 'day') -> dict:
    """Place an order on Alpaca paper account (sync, for backward compat).
    side: 'sell' for short entry, 'buy' for cover (buy-to-cover).
    """
    cfg = _get_alpaca_config()
    if cfg is None:
        return {'error': 'Alpaca not configured'}

    headers = {**_alpaca_headers(cfg), 'Content-Type': 'application/json'}
    payload = {
        'symbol': symbol,
        'qty': str(qty),
        'side': side,
        'type': order_type,
        'time_in_force': time_in_force,
    }
    if limit_price is not None:
        payload['limit_price'] = str(round(limit_price, 2))
    if stop_price is not None:
        payload['stop_price'] = str(round(stop_price, 2))

    try:
        resp = requests.post(
            f'{cfg["base_url"]}/v2/orders',
            headers=headers, json=payload, timeout=10
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            logger.info(f"Alpaca order placed: {side} {qty} {symbol} -> {data.get('id', '?')}")
            return {'id': data.get('id'), 'status': data.get('status'),
                    'symbol': symbol, 'qty': qty, 'side': side}
        else:
            err = resp.text[:200]
            logger.warning(f"Alpaca order failed ({resp.status_code}): {err}")
            return {'error': f"HTTP {resp.status_code}: {err}"}
    except Exception as e:
        logger.warning(f"Alpaca order error: {e}")
        return {'error': str(e)}


def alpaca_get_order(order_id: str) -> Optional[dict]:
    """Get order status from Alpaca by order ID."""
    cfg = _get_alpaca_config()
    if cfg is None:
        return None
    try:
        resp = requests.get(
            f'{cfg["base_url"]}/v2/orders/{order_id}',
            headers=_alpaca_headers(cfg), timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"Get order {order_id} failed: HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"Get order {order_id} error: {e}")
    return None


def alpaca_cancel_order(order_id: str) -> bool:
    """Cancel an open order on Alpaca. Returns True if successfully cancelled."""
    cfg = _get_alpaca_config()
    if cfg is None:
        return False
    try:
        resp = requests.delete(
            f'{cfg["base_url"]}/v2/orders/{order_id}',
            headers=_alpaca_headers(cfg), timeout=10
        )
        return resp.status_code in (200, 204)
    except Exception as e:
        logger.warning(f"Cancel order {order_id} error: {e}")
        return False


async def alpaca_submit_and_confirm(
    symbol: str, qty: int, side: str,
    order_type: str = 'market', limit_price: float = None,
    timeout_sec: float = 30.0, poll_interval: float = 0.5,
) -> OrderResult:
    """Submit order and poll until filled, rejected, or timeout.

    For limit orders that don't fill within timeout, cancels the order and
    returns a timeout result. Market orders typically fill within 1-2 polls.
    """
    result = OrderResult(symbol=symbol, side=side)

    # Submit order (runs sync HTTP in thread to not block event loop)
    order_resp = await asyncio.to_thread(
        alpaca_place_order, symbol, qty, side, order_type, limit_price
    )
    if 'error' in order_resp:
        result.status = 'error'
        result.error = order_resp['error']
        logger.error(f"Order submit failed: {side} {qty} {symbol}: {result.error}")
        return result

    order_id = order_resp.get('id', '')
    result.order_id = order_id
    logger.info(f"Order submitted: {side} {qty} {symbol} ({order_type}) id={order_id}")

    # Poll for fill
    deadline = _time.monotonic() + timeout_sec
    while _time.monotonic() < deadline:
        await asyncio.sleep(poll_interval)
        order_data = await asyncio.to_thread(alpaca_get_order, order_id)
        if order_data is None:
            continue

        status = order_data.get('status', '')

        if status == 'filled':
            result.status = 'filled'
            result.filled_qty = int(order_data.get('filled_qty', qty))
            result.filled_avg_price = float(order_data.get('filled_avg_price', 0))
            logger.info(f"Order FILLED: {side} {result.filled_qty} {symbol} "
                       f"@ ${result.filled_avg_price:.2f} (id={order_id})")
            return result

        if status in ('cancelled', 'canceled', 'expired', 'suspended'):
            result.status = 'cancelled'
            result.error = f"Order {status}"
            logger.warning(f"Order {status}: {side} {qty} {symbol} (id={order_id})")
            return result

        if status == 'rejected':
            result.status = 'rejected'
            result.error = f"Rejected: {order_data.get('reject_reason', 'unknown')}"
            logger.warning(f"Order REJECTED: {side} {qty} {symbol}: {result.error}")
            return result

        if status == 'partially_filled':
            result.filled_qty = int(order_data.get('filled_qty', 0))
            result.filled_avg_price = float(order_data.get('filled_avg_price', 0))
            # Keep polling — may fully fill

    # Timeout — cancel the order if it hasn't filled
    logger.warning(f"Order TIMEOUT after {timeout_sec}s: {side} {qty} {symbol} (id={order_id})")
    order_data = await asyncio.to_thread(alpaca_get_order, order_id)
    if order_data and order_data.get('status') == 'filled':
        result.status = 'filled'
        result.filled_qty = int(order_data.get('filled_qty', qty))
        result.filled_avg_price = float(order_data.get('filled_avg_price', 0))
        return result

    # Cancel unfilled order
    cancelled = await asyncio.to_thread(alpaca_cancel_order, order_id)
    filled_qty = int(order_data.get('filled_qty', 0)) if order_data else 0

    if filled_qty > 0:
        result.status = 'partially_filled'
        result.filled_qty = filled_qty
        result.filled_avg_price = float(order_data.get('filled_avg_price', 0))
        result.error = f"Partial fill: {filled_qty}/{qty} shares"
    else:
        result.status = 'timeout'
        result.error = f"No fill after {timeout_sec}s, order cancelled"

    return result


def alpaca_get_positions() -> List[dict]:
    """Get current positions on Alpaca paper account."""
    cfg = _get_alpaca_config()
    if cfg is None:
        return []

    headers = _alpaca_headers(cfg)
    try:
        resp = requests.get(
            f'{cfg["base_url"]}/v2/positions',
            headers=headers, timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"Get positions error: {e}")
    return []


def alpaca_place_stop_order(symbol: str, qty: int, stop_price: float,
                            limit_offset_pct: float = 0.003,
                            direction: str = 'short') -> dict:
    """Place a broker-side stop-limit order.

    For shorts: buy-to-cover stop (side='buy', limit above stop).
    For longs: sell stop (side='sell', limit below stop).

    Alpaca holds this order server-side and triggers it the instant price
    hits the stop — no polling delay.

    Returns {'id': order_id, ...} or {'error': ...}.
    """
    cfg = _get_alpaca_config()
    if cfg is None:
        return {'error': 'Alpaca not configured'}

    if direction == 'long':
        side = 'sell'
        limit_price = round(stop_price * (1 - limit_offset_pct), 2)
    else:
        side = 'buy'
        limit_price = round(stop_price * (1 + limit_offset_pct), 2)
    headers = {**_alpaca_headers(cfg), 'Content-Type': 'application/json'}
    payload = {
        'symbol': symbol,
        'qty': str(qty),
        'side': side,
        'type': 'stop_limit',
        'stop_price': str(round(stop_price, 2)),
        'limit_price': str(limit_price),
        'time_in_force': 'day',
    }

    try:
        resp = requests.post(
            f'{cfg["base_url"]}/v2/orders',
            headers=headers, json=payload, timeout=10
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            logger.info(f"Broker stop placed: {side.upper()} {qty} {symbol} stop=${stop_price:.2f} "
                       f"limit=${limit_price:.2f} -> id={data.get('id', '?')}")
            return {'id': data.get('id'), 'status': data.get('status'),
                    'symbol': symbol, 'stop_price': stop_price, 'limit_price': limit_price}
        else:
            err = resp.text[:200]
            logger.warning(f"Broker stop failed ({resp.status_code}): {err}")
            return {'error': f"HTTP {resp.status_code}: {err}"}
    except Exception as e:
        logger.warning(f"Broker stop error: {e}")
        return {'error': str(e)}


async def alpaca_replace_stop_order(old_order_id: str, symbol: str, qty: int,
                                     new_stop_price: float,
                                     limit_offset_pct: float = 0.003) -> dict:
    """Cancel old stop order and place a new one at a different price.

    Used when stop moves to breakeven after partial fill.
    Returns the new order result dict.
    """
    # Cancel old
    if old_order_id:
        await asyncio.to_thread(alpaca_cancel_order, old_order_id)

    # Place new
    result = await asyncio.to_thread(
        alpaca_place_stop_order, symbol, qty, new_stop_price, limit_offset_pct
    )
    return result


def alpaca_get_account() -> Optional[dict]:
    """Get Alpaca paper account info (equity, buying power, etc.)."""
    cfg = _get_alpaca_config()
    if cfg is None:
        return None

    headers = _alpaca_headers(cfg)
    try:
        resp = requests.get(
            f'{cfg["base_url"]}/v2/account',
            headers=headers, timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"Get account error: {e}")
    return None


class AlpacaTickStreamer:
    """Real-time trade stream via Alpaca WebSocket."""

    WS_URL = 'wss://stream.data.alpaca.markets/v2/iex'
    THROTTLE_SEC = 0.25

    def __init__(self, symbols, on_tick=None):
        if isinstance(symbols, str):
            symbols = [symbols]
        self.symbols = list(symbols)
        self.on_tick = on_tick
        self.latest_prices: Dict[str, float] = {}
        self.connected = False
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_push: Dict[str, float] = {}
        self._ws = None  # reference to live websocket for dynamic subscriptions

    async def add_symbols(self, symbols: List[str]):
        """Dynamically subscribe to additional symbols on the live connection."""
        new_syms = [s for s in symbols if s not in self.symbols]
        if not new_syms:
            return
        self.symbols.extend(new_syms)
        if self._ws and not self._ws.closed:
            await self._ws.send_json({'action': 'subscribe', 'trades': new_syms})
            logger.info(f"Alpaca stream: dynamically subscribed to {new_syms}")

    async def remove_symbols(self, symbols: List[str]):
        """Dynamically unsubscribe from symbols on the live connection."""
        rm_syms = [s for s in symbols if s in self.symbols]
        if not rm_syms:
            return
        self.symbols = [s for s in self.symbols if s not in rm_syms]
        if self._ws and not self._ws.closed:
            await self._ws.send_json({'action': 'unsubscribe', 'trades': rm_syms})
            logger.info(f"Alpaca stream: dynamically unsubscribed from {rm_syms}")

    async def start(self):
        cfg = _get_alpaca_config()
        if cfg is None:
            logger.warning("AlpacaTickStreamer: no Alpaca credentials")
            return
        self._task = asyncio.create_task(self._run(cfg))

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session and not self._session.closed:
            await self._session.close()
        self._ws = None
        self.connected = False

    async def _run(self, cfg: dict):
        while True:
            try:
                self._session = aiohttp.ClientSession()
                async with self._session.ws_connect(self.WS_URL) as ws:
                    self._ws = ws
                    await ws.receive_json()
                    await ws.send_json({
                        'action': 'auth',
                        'key': cfg['api_key'],
                        'secret': cfg['secret_key'],
                    })
                    auth_resp = await ws.receive_json()
                    if not any(m.get('msg') == 'authenticated' for m in auth_resp):
                        is_limit = any(m.get('code') == 406 for m in auth_resp)
                        wait = 60 if is_limit else 5
                        logger.error(f"Alpaca stream auth failed: {auth_resp} — retry in {wait}s")
                        self._ws = None
                        await self._session.close()
                        await asyncio.sleep(wait)
                        continue

                    await ws.send_json({
                        'action': 'subscribe',
                        'trades': self.symbols,
                    })
                    sub_resp = await ws.receive_json()
                    logger.info(f"Alpaca stream: subscribed to {len(self.symbols)} symbols")
                    self.connected = True

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            for item in json.loads(msg.data):
                                if item.get('T') == 't':
                                    sym = item.get('S', '')
                                    price = float(item['p'])
                                    size = int(item.get('s', 0))
                                    self.latest_prices[sym] = price
                                    if self.on_tick:
                                        now = _time.monotonic()
                                        last = self._last_push.get(sym, 0.0)
                                        if now - last >= self.THROTTLE_SEC:
                                            self._last_push[sym] = now
                                            await self.on_tick(sym, price, size)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Alpaca stream error: {e}, reconnecting in 3s...")
            finally:
                self.connected = False
                self._ws = None
                if self._session and not self._session.closed:
                    await self._session.close()
            await asyncio.sleep(3)


# =============================================================================
# SECTION 2a: POLYGON / MASSIVE.COM HELPERS
# =============================================================================

def _get_polygon_key() -> Optional[str]:
    """Get Polygon.io / Massive.com API key from env."""
    _load_env_file()
    return os.environ.get('POLYGON_API_KEY', '') or os.environ.get('MASSIVE_API_KEY', '') or None


def fetch_polygon_grouped_daily(date_str: str) -> List[dict]:
    """Fetch OHLCV for ALL US stocks for a single date via Polygon grouped daily.

    One API call returns every ticker (~11k+).
    Free tier: 5 req/min, EOD data.

    Returns list of dicts: [{symbol, date, open, high, low, close, volume}, ...]
    """
    api_key = _get_polygon_key()
    if not api_key:
        raise ValueError("POLYGON_API_KEY or MASSIVE_API_KEY not set in .env")

    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
    params = {'adjusted': 'true', 'apiKey': api_key}

    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 403:
        msg = resp.json().get('message', 'Forbidden')
        if 'today' in msg.lower():
            raise ValueError(f"Polygon free tier: no same-day data ({date_str}). Use yesterday or older.")
        raise ValueError(f"Polygon 403: {msg}")
    if resp.status_code == 429:
        raise ValueError("Polygon rate limit hit — try again in a minute")
    resp.raise_for_status()

    data = resp.json()
    if data.get('resultsCount', 0) == 0:
        logger.warning(f"Polygon grouped daily: 0 results for {date_str} (market closed?)")
        return []

    bars = []
    for r in data.get('results', []):
        sym = r.get('T', '')
        if not sym or len(sym) > 5:  # skip warrants, units, etc.
            continue
        bars.append({
            'symbol': sym,
            'date': date_str,
            'open': float(r.get('o', 0)),
            'high': float(r.get('h', 0)),
            'low': float(r.get('l', 0)),
            'close': float(r.get('c', 0)),
            'volume': int(r.get('v', 0)),
        })

    logger.info(f"Polygon grouped daily: {len(bars)} tickers for {date_str}")
    return bars


def fetch_polygon_daily_range(start_date: str, end_date: str,
                               progress_callback=None) -> Dict[str, List[dict]]:
    """Fetch grouped daily bars for a date range.

    Calls the grouped daily endpoint once per trading day.
    Free tier: 5 req/min, so we sleep 12s between calls to be safe.

    Returns {date_str: [bar_dicts]}
    """
    from datetime import timedelta
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    all_bars = {}
    current = start
    day_num = 0
    total_days = (end - start).days + 1

    while current <= end:
        ds = current.strftime('%Y-%m-%d')
        # Skip weekends
        if current.weekday() < 5:
            try:
                bars = fetch_polygon_grouped_daily(ds)
                if bars:
                    all_bars[ds] = bars
                    day_num += 1
                    if progress_callback:
                        progress_callback(day_num, total_days, ds, len(bars))
            except Exception as e:
                logger.warning(f"Polygon fetch error for {ds}: {e}")

            # Rate limit: 5 req/min on free tier → 12s between calls
            _time.sleep(12)

        current += timedelta(days=1)

    return all_bars


# =============================================================================
# SECTION 2b: LOCAL PRICE DATABASE
# =============================================================================

class PriceDB:
    """SQLite cache for daily bars — eliminates repeated Alpaca fetches."""

    DB_PATH = os.path.join(os.path.dirname(__file__) or '.', 'gap_fade_prices.db')

    def __init__(self):
        self._conn = sqlite3.connect(self.DB_PATH, check_same_thread=False)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.execute('PRAGMA synchronous=NORMAL')
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_bars (
                symbol TEXT NOT NULL,
                date   TEXT NOT NULL,
                open   REAL,
                high   REAL,
                low    REAL,
                close  REAL,
                volume REAL,
                PRIMARY KEY (symbol, date)
            ) WITHOUT ROWID
        ''')
        # Covering index for date-range scans (used by SQL gap scanner)
        self._conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_bars_date_symbol
            ON daily_bars (date, symbol)
        ''')
        # Persistent trade storage — all completed trades survive restarts
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol        TEXT NOT NULL,
                entry_price   REAL NOT NULL,
                exit_price    REAL NOT NULL,
                shares        INTEGER NOT NULL,
                pnl           REAL NOT NULL,
                pnl_pct       REAL NOT NULL,
                entry_time    TEXT NOT NULL,
                exit_time     TEXT NOT NULL,
                exit_reason   TEXT NOT NULL,
                holding_minutes INTEGER DEFAULT 0,
                side          TEXT DEFAULT 'short',
                gap_pct       REAL DEFAULT 0.0,
                vol_ratio     REAL DEFAULT 0.0,
                score         REAL DEFAULT 0.0,
                catalyst      TEXT DEFAULT '',
                UNIQUE(symbol, entry_time, exit_time, shares, exit_reason)
            )
        ''')
        self._conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_trades_entry_time
            ON trades (entry_time)
        ''')
        self._conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_trades_symbol_entry
            ON trades (symbol, entry_time)
        ''')
        # -- Journal entries (replaces JSONL files) --
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS journal_entries (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT NOT NULL,
                entry_type TEXT NOT NULL,
                source     TEXT NOT NULL,
                symbol     TEXT DEFAULT '',
                content    TEXT NOT NULL,
                data       TEXT DEFAULT '{}',
                llm_call   INTEGER DEFAULT 0,
                UNIQUE(timestamp, entry_type, source, symbol, content)
            )
        ''')
        self._conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_journal_ts
            ON journal_entries (timestamp)
        ''')
        self._conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_journal_type
            ON journal_entries (entry_type, timestamp)
        ''')
        # -- Market events (persists EventBus audit log) --
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS market_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                mono_time   REAL NOT NULL,
                event_type  TEXT NOT NULL,
                tier        INTEGER NOT NULL,
                symbol      TEXT DEFAULT '',
                description TEXT DEFAULT '',
                data        TEXT DEFAULT '{}',
                dedup_key   TEXT DEFAULT '',
                UNIQUE(timestamp, event_type, symbol, dedup_key)
            )
        ''')
        self._conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_events_ts
            ON market_events (timestamp)
        ''')
        self._conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_events_type
            ON market_events (event_type, timestamp)
        ''')
        # -- LLM call audit log --
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS llm_calls (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT NOT NULL,
                conversation_role TEXT NOT NULL,
                priority         TEXT NOT NULL,
                model            TEXT DEFAULT '',
                prompt_tokens    INTEGER DEFAULT 0,
                response_tokens  INTEGER DEFAULT 0,
                duration_ms      INTEGER DEFAULT 0,
                success          INTEGER DEFAULT 1,
                budget_5min      INTEGER DEFAULT 0,
                budget_1hr       INTEGER DEFAULT 0,
                response_summary TEXT DEFAULT '',
                UNIQUE(timestamp, conversation_role)
            )
        ''')
        self._conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_llm_ts
            ON llm_calls (timestamp)
        ''')
        self._conn.commit()
        # Migrate legacy JSONL journal files (one-time)
        self._migrate_jsonl_to_sqlite()
        # Update query planner statistics (fast on subsequent runs)
        self._conn.execute('ANALYZE')

    def upsert_bars(self, symbol: str, df: pd.DataFrame):
        """Insert or replace bars for a single symbol from a DataFrame."""
        if df is None or df.empty:
            return
        rows = []
        for idx, row in df.iterrows():
            dt = idx
            if hasattr(dt, 'strftime'):
                date_str = dt.strftime('%Y-%m-%d')
            else:
                date_str = str(dt)[:10]
            rows.append((symbol, date_str, float(row['open']), float(row['high']),
                         float(row['low']), float(row['close']), float(row['volume'])))
        self._conn.executemany(
            'INSERT OR REPLACE INTO daily_bars (symbol, date, open, high, low, close, volume) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)', rows
        )
        self._conn.commit()

    def upsert_bars_batch(self, dfs: Dict[str, pd.DataFrame]):
        """Bulk insert bars for many symbols at once (single transaction)."""
        rows = []
        for symbol, df in dfs.items():
            if df is None or df.empty:
                continue
            for idx, row in df.iterrows():
                dt = idx
                if hasattr(dt, 'strftime'):
                    date_str = dt.strftime('%Y-%m-%d')
                else:
                    date_str = str(dt)[:10]
                rows.append((symbol, date_str, float(row['open']), float(row['high']),
                             float(row['low']), float(row['close']), float(row['volume'])))
        if rows:
            self._conn.executemany(
                'INSERT OR REPLACE INTO daily_bars (symbol, date, open, high, low, close, volume) '
                'VALUES (?, ?, ?, ?, ?, ?, ?)', rows
            )
            self._conn.commit()
        logger.info(f"PriceDB: upserted {len(rows)} rows for {len(dfs)} symbols")

    def upsert_bars_dicts(self, bars: List[dict]):
        """Bulk insert from list of dicts (Polygon format).

        Each dict: {symbol, date, open, high, low, close, volume}
        """
        if not bars:
            return
        rows = [(b['symbol'], b['date'], b['open'], b['high'],
                 b['low'], b['close'], b['volume']) for b in bars]
        self._conn.executemany(
            'INSERT OR REPLACE INTO daily_bars (symbol, date, open, high, low, close, volume) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)', rows
        )
        self._conn.commit()
        logger.info(f"PriceDB: upserted {len(rows)} rows from Polygon data")

    def get_bars(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Get daily bars for one symbol in [start, end] date range."""
        cur = self._conn.execute(
            'SELECT date, open, high, low, close, volume FROM daily_bars '
            'WHERE symbol = ? AND date >= ? AND date <= ? ORDER BY date',
            (symbol, start, end)
        )
        rows = cur.fetchall()
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(np.float64)
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        df.index.name = symbol
        return df

    def get_bars_batch(self, symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        """Get daily bars for many symbols at once. Returns {symbol: DataFrame}."""
        results = {}
        # Use chunked IN queries for efficiency
        chunk_size = 500
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            placeholders = ','.join('?' * len(chunk))
            cur = self._conn.execute(
                f'SELECT symbol, date, open, high, low, close, volume FROM daily_bars '
                f'WHERE symbol IN ({placeholders}) AND date >= ? AND date <= ? ORDER BY symbol, date',
                chunk + [start, end]
            )
            rows = cur.fetchall()
            # Group by symbol
            for sym, group in groupby(rows, key=lambda r: r[0]):
                bar_rows = list(group)
                df = pd.DataFrame(bar_rows, columns=['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume'])
                df.drop(columns=['symbol'], inplace=True)
                df['datetime'] = pd.to_datetime(df['datetime'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(np.float64)
                df.set_index('datetime', inplace=True)
                df.sort_index(inplace=True)
                df.index.name = sym
                results[sym] = df
        return results

    def get_last_date(self, symbol: str) -> Optional[str]:
        """Get the most recent date stored for a symbol."""
        cur = self._conn.execute(
            'SELECT MAX(date) FROM daily_bars WHERE symbol = ?', (symbol,)
        )
        row = cur.fetchone()
        return row[0] if row and row[0] else None

    def get_last_dates_batch(self) -> Dict[str, str]:
        """Get the most recent date for every symbol in the DB."""
        cur = self._conn.execute(
            'SELECT symbol, MAX(date) FROM daily_bars GROUP BY symbol'
        )
        return {row[0]: row[1] for row in cur.fetchall()}

    def get_symbols(self) -> List[str]:
        """Get all symbols stored in the DB."""
        cur = self._conn.execute('SELECT DISTINCT symbol FROM daily_bars ORDER BY symbol')
        return [row[0] for row in cur.fetchall()]

    def get_stats(self) -> dict:
        """Get DB statistics: symbol count, date range, file size."""
        cur = self._conn.execute('SELECT COUNT(DISTINCT symbol), MIN(date), MAX(date), COUNT(*) FROM daily_bars')
        row = cur.fetchone()
        symbol_count = row[0] or 0
        min_date = row[1] or ''
        max_date = row[2] or ''
        total_rows = row[3] or 0
        try:
            size_bytes = os.path.getsize(self.DB_PATH)
        except OSError:
            size_bytes = 0
        return {
            'symbol_count': symbol_count,
            'min_date': min_date,
            'max_date': max_date,
            'total_rows': total_rows,
            'size_mb': round(size_bytes / (1024 * 1024), 1),
        }

    def scan_gaps_sql(self, symbols: List[str], start: str, end: str,
                      gap_threshold: float = 0.05, max_gap_pct: float = None,
                      min_price: float = 3.0,
                      min_avg_volume: int = 5000, vol_window: int = 20,
                      gap_down_config: dict = None) -> List[dict]:
        """Find gap days using pure SQL with window functions.

        Uses LAG() for prev_close and prev_volume (no look-ahead), and AVG()
        window for rolling average volume.

        For large universes (>2000 symbols), runs a single date-range scan
        instead of chunked IN-clause queries — ~15x faster on 12K+ symbols.

        gap_down_config: if set, also scan for gap-downs with keys:
            threshold (positive float), max_pct (positive float), vol_ratio_max (float)

        Returns list of gap dicts ready for backtester simulation.
        """
        # Fetch extra days before start for rolling avg volume warmup
        adj_start = (datetime.strptime(start, '%Y-%m-%d') - timedelta(days=60)).strftime('%Y-%m-%d')

        # --- Build gap filter clauses (shared by both paths) ---
        gap_up_where = "(open - prev_close) / prev_close >= ?"
        gap_up_params = [gap_threshold]
        if max_gap_pct:
            gap_up_where += " AND (open - prev_close) / prev_close <= ?"
            gap_up_params.append(max_gap_pct)

        if gap_down_config:
            gd_thresh = gap_down_config.get('threshold', 0.05)
            gd_max = gap_down_config.get('max_pct', 0.50)
            gap_filter = f"""
            (
                ({gap_up_where})
                OR
                (
                    (prev_close - open) / prev_close >= ?
                    AND (prev_close - open) / prev_close <= ?
                )
            )
            """
            gap_params = gap_up_params + [gd_thresh, gd_max]
        else:
            gap_filter = gap_up_where
            gap_params = gap_up_params

        # --- Choose scan strategy based on universe size ---
        # Large universe (>2000): single pass over date range, no IN clause
        # Small universe: chunked IN-clause for targeted scans
        use_full_scan = len(symbols) > 2000

        all_gaps = []

        if use_full_scan:
            # Build a set for post-filter (faster than 12K-element IN clause)
            symbol_set = set(symbols)
            sql = f"""
            WITH w AS (
                SELECT symbol, date, open, high, low, close, volume,
                       LAG(close)  OVER (PARTITION BY symbol ORDER BY date) AS prev_close,
                       LAG(volume) OVER (PARTITION BY symbol ORDER BY date) AS prev_volume,
                       AVG(volume) OVER (
                           PARTITION BY symbol ORDER BY date
                           ROWS BETWEEN {vol_window} PRECEDING AND 1 PRECEDING
                       ) AS avg_vol
                FROM daily_bars
                WHERE date >= ? AND date <= ?
            )
            SELECT symbol, date, open, high, low, close, volume,
                   prev_close, prev_volume, avg_vol
            FROM w
            WHERE date >= ?
              AND prev_close > 0
              AND open > 0
              AND {gap_filter}
              AND open >= ?
              AND avg_vol >= ?
            ORDER BY date, symbol
            """
            params = [adj_start, end, start] + gap_params + [min_price, min_avg_volume]
            cur = self._conn.execute(sql, params)
            for row in cur:
                sym = row[0]
                if sym not in symbol_set:
                    continue
                all_gaps.append(self._gap_row_to_dict(row))
        else:
            # Chunked approach for small symbol lists
            chunk_size = 500
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i:i + chunk_size]
                placeholders = ','.join('?' * len(chunk))
                sql = f"""
                WITH w AS (
                    SELECT symbol, date, open, high, low, close, volume,
                           LAG(close)  OVER (PARTITION BY symbol ORDER BY date) AS prev_close,
                           LAG(volume) OVER (PARTITION BY symbol ORDER BY date) AS prev_volume,
                           AVG(volume) OVER (
                               PARTITION BY symbol ORDER BY date
                               ROWS BETWEEN {vol_window} PRECEDING AND 1 PRECEDING
                           ) AS avg_vol
                    FROM daily_bars
                    WHERE symbol IN ({placeholders}) AND date >= ? AND date <= ?
                )
                SELECT symbol, date, open, high, low, close, volume,
                       prev_close, prev_volume, avg_vol
                FROM w
                WHERE date >= ?
                  AND prev_close > 0
                  AND open > 0
                  AND {gap_filter}
                  AND open >= ?
                  AND avg_vol >= ?
                ORDER BY date, symbol
                """
                params = chunk + [adj_start, end, start] + gap_params + [min_price, min_avg_volume]
                cur = self._conn.execute(sql, params)
                for row in cur:
                    all_gaps.append(self._gap_row_to_dict(row))

        return all_gaps

    # ── Trade persistence ─────────────────────────────────────────────

    def insert_trade(self, trade) -> bool:
        """INSERT OR IGNORE a TradeRecord (or dict) into the trades table.
        Returns True if a row was inserted, False if duplicate/ignored."""
        d = trade if isinstance(trade, dict) else asdict(trade)
        try:
            cur = self._conn.execute(
                '''INSERT OR IGNORE INTO trades
                   (symbol, entry_price, exit_price, shares, pnl, pnl_pct,
                    entry_time, exit_time, exit_reason, holding_minutes,
                    side, gap_pct, vol_ratio, score, catalyst)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (d['symbol'], d['entry_price'], d['exit_price'], d['shares'],
                 d['pnl'], d['pnl_pct'], d['entry_time'], d['exit_time'],
                 d['exit_reason'], d.get('holding_minutes', 0),
                 d.get('side', 'short'), d.get('gap_pct', 0.0),
                 d.get('vol_ratio', 0.0), d.get('score', 0.0),
                 d.get('catalyst', '')))
            self._conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            logger.error(f"PriceDB.insert_trade failed: {e}")
            return False

    def load_all_trades(self) -> List[dict]:
        """Load all trades ordered by trade_id (insertion order)."""
        try:
            cur = self._conn.execute(
                '''SELECT symbol, entry_price, exit_price, shares, pnl, pnl_pct,
                          entry_time, exit_time, exit_reason, holding_minutes,
                          side, gap_pct, vol_ratio, score, catalyst
                   FROM trades ORDER BY trade_id''')
            cols = ['symbol', 'entry_price', 'exit_price', 'shares', 'pnl', 'pnl_pct',
                    'entry_time', 'exit_time', 'exit_reason', 'holding_minutes',
                    'side', 'gap_pct', 'vol_ratio', 'score', 'catalyst']
            return [dict(zip(cols, row)) for row in cur]
        except Exception as e:
            logger.error(f"PriceDB.load_all_trades failed: {e}")
            return []

    def query_trades(self, start_date: str = None, end_date: str = None,
                     symbol: str = None, limit: int = 100, offset: int = 0) -> dict:
        """Paginated trade query. Returns {trades, total, limit, offset}."""
        try:
            where_clauses = []
            params: list = []
            if start_date:
                where_clauses.append('entry_time >= ?')
                params.append(start_date)
            if end_date:
                where_clauses.append('entry_time <= ?')
                params.append(end_date)
            if symbol:
                where_clauses.append('symbol = ?')
                params.append(symbol.upper())
            where_sql = (' WHERE ' + ' AND '.join(where_clauses)) if where_clauses else ''

            # Total count
            total = self._conn.execute(
                f'SELECT COUNT(*) FROM trades{where_sql}', params).fetchone()[0]

            # Paginated results (newest first)
            cur = self._conn.execute(
                f'''SELECT symbol, entry_price, exit_price, shares, pnl, pnl_pct,
                           entry_time, exit_time, exit_reason, holding_minutes,
                           side, gap_pct, vol_ratio, score, catalyst
                    FROM trades{where_sql}
                    ORDER BY trade_id DESC LIMIT ? OFFSET ?''',
                params + [limit, offset])
            cols = ['symbol', 'entry_price', 'exit_price', 'shares', 'pnl', 'pnl_pct',
                    'entry_time', 'exit_time', 'exit_reason', 'holding_minutes',
                    'side', 'gap_pct', 'vol_ratio', 'score', 'catalyst']
            trades = [dict(zip(cols, row)) for row in cur]
            return {'trades': trades, 'total': total, 'limit': limit, 'offset': offset}
        except Exception as e:
            logger.error(f"PriceDB.query_trades failed: {e}")
            return {'trades': [], 'total': 0, 'limit': limit, 'offset': offset}

    def count_trades(self) -> int:
        """Return total number of trades in the database."""
        try:
            return self._conn.execute('SELECT COUNT(*) FROM trades').fetchone()[0]
        except Exception:
            return 0

    # -- Journal entry persistence --

    def insert_journal(self, entry: dict):
        """Insert a journal entry. Uses INSERT OR IGNORE for idempotency."""
        try:
            self._conn.execute('''
                INSERT OR IGNORE INTO journal_entries
                (timestamp, entry_type, source, symbol, content, data, llm_call)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (entry['timestamp'], entry['entry_type'], entry['source'],
                  entry.get('symbol', ''), entry['content'],
                  json.dumps(entry.get('data', {}), default=str),
                  1 if entry.get('llm_call') else 0))
            self._conn.commit()
        except Exception as e:
            logger.error(f"Journal insert failed: {e}")

    def query_journal(self, date: str = '', n: int = 50, entry_type: str = '') -> list:
        """Query journal entries. If date given, filter by date prefix on timestamp."""
        try:
            where, params = [], []
            if date:
                where.append('timestamp LIKE ?')
                params.append(f'{date}%')
            if entry_type:
                where.append('entry_type = ?')
                params.append(entry_type)
            where_sql = (' WHERE ' + ' AND '.join(where)) if where else ''
            cur = self._conn.execute(
                f'''SELECT timestamp, entry_type, source, symbol, content, data, llm_call
                    FROM journal_entries{where_sql}
                    ORDER BY id DESC LIMIT ?''',
                params + [n])
            cols = ['timestamp', 'entry_type', 'source', 'symbol', 'content', 'data', 'llm_call']
            rows = []
            for row in cur:
                d = dict(zip(cols, row))
                try:
                    d['data'] = json.loads(d['data']) if d['data'] else {}
                except (json.JSONDecodeError, TypeError):
                    d['data'] = {}
                d['llm_call'] = bool(d['llm_call'])
                rows.append(d)
            rows.reverse()  # oldest first
            return rows
        except Exception as e:
            logger.error(f"Journal query failed: {e}")
            return []

    def get_journal_stats(self, date: str) -> dict:
        """Count journal entries by type for a date."""
        try:
            cur = self._conn.execute(
                '''SELECT entry_type, COUNT(*) FROM journal_entries
                   WHERE timestamp LIKE ? GROUP BY entry_type''',
                (f'{date}%',))
            counts = dict(cur.fetchall())
            total = sum(counts.values())
            return {'date': date, 'total': total, 'by_type': counts}
        except Exception as e:
            logger.error(f"Journal stats failed: {e}")
            return {'date': date, 'total': 0, 'by_type': {}}

    # -- Market event persistence --

    def insert_event(self, event):
        """Insert a MarketEvent into the database."""
        try:
            wall_ts = datetime.now(ET).strftime('%Y-%m-%dT%H:%M:%S')
            self._conn.execute('''
                INSERT OR IGNORE INTO market_events
                (timestamp, mono_time, event_type, tier, symbol, description, data, dedup_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (wall_ts, event.timestamp, event.event_type, event.tier,
                  event.symbol or '', event.description or '',
                  json.dumps(event.data or {}, default=str),
                  event.dedup_key or ''))
            self._conn.commit()
        except Exception as e:
            logger.error(f"Event insert failed: {e}")

    def query_events(self, date: str = '', n: int = 50, event_type: str = '') -> list:
        """Query market events."""
        try:
            where, params = [], []
            if date:
                where.append('timestamp LIKE ?')
                params.append(f'{date}%')
            if event_type:
                where.append('event_type = ?')
                params.append(event_type)
            where_sql = (' WHERE ' + ' AND '.join(where)) if where else ''
            cur = self._conn.execute(
                f'''SELECT timestamp, mono_time, event_type, tier, symbol,
                           description, data, dedup_key
                    FROM market_events{where_sql}
                    ORDER BY id DESC LIMIT ?''',
                params + [n])
            cols = ['timestamp', 'mono_time', 'event_type', 'tier', 'symbol',
                    'description', 'data', 'dedup_key']
            rows = []
            for row in cur:
                d = dict(zip(cols, row))
                try:
                    d['data'] = json.loads(d['data']) if d['data'] else {}
                except (json.JSONDecodeError, TypeError):
                    d['data'] = {}
                rows.append(d)
            rows.reverse()  # oldest first
            return rows
        except Exception as e:
            logger.error(f"Event query failed: {e}")
            return []

    # -- LLM call audit log --

    def insert_llm_call(self, call_data: dict):
        """Insert an LLM call log entry."""
        try:
            self._conn.execute('''
                INSERT OR IGNORE INTO llm_calls
                (timestamp, conversation_role, priority, model,
                 prompt_tokens, response_tokens, duration_ms,
                 success, budget_5min, budget_1hr, response_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (call_data['timestamp'], call_data['conversation_role'],
                  call_data['priority'], call_data.get('model', ''),
                  call_data.get('prompt_tokens', 0), call_data.get('response_tokens', 0),
                  call_data.get('duration_ms', 0), call_data.get('success', 1),
                  call_data.get('budget_5min', 0), call_data.get('budget_1hr', 0),
                  call_data.get('response_summary', '')[:200]))
            self._conn.commit()
        except Exception as e:
            logger.error(f"LLM call insert failed: {e}")

    def query_llm_calls(self, date: str = '', n: int = 50) -> list:
        """Query LLM call history."""
        try:
            where, params = [], []
            if date:
                where.append('timestamp LIKE ?')
                params.append(f'{date}%')
            where_sql = (' WHERE ' + ' AND '.join(where)) if where else ''
            cur = self._conn.execute(
                f'''SELECT timestamp, conversation_role, priority, model,
                           prompt_tokens, response_tokens, duration_ms,
                           success, budget_5min, budget_1hr, response_summary
                    FROM llm_calls{where_sql}
                    ORDER BY id DESC LIMIT ?''',
                params + [n])
            cols = ['timestamp', 'conversation_role', 'priority', 'model',
                    'prompt_tokens', 'response_tokens', 'duration_ms',
                    'success', 'budget_5min', 'budget_1hr', 'response_summary']
            rows = [dict(zip(cols, row)) for row in cur]
            rows.reverse()  # oldest first
            return rows
        except Exception as e:
            logger.error(f"LLM call query failed: {e}")
            return []

    def get_llm_stats(self, date: str) -> dict:
        """Total calls, avg duration, success rate for a date."""
        try:
            cur = self._conn.execute(
                '''SELECT COUNT(*), AVG(duration_ms), SUM(success),
                          SUM(prompt_tokens), SUM(response_tokens)
                   FROM llm_calls WHERE timestamp LIKE ?''',
                (f'{date}%',))
            row = cur.fetchone()
            total = row[0] or 0
            avg_ms = round(row[1] or 0, 1)
            successes = row[2] or 0
            return {
                'date': date,
                'total_calls': total,
                'avg_duration_ms': avg_ms,
                'success_rate': round(successes / total, 3) if total else 0,
                'total_prompt_tokens': row[3] or 0,
                'total_response_tokens': row[4] or 0,
            }
        except Exception as e:
            logger.error(f"LLM stats failed: {e}")
            return {'date': date, 'total_calls': 0, 'avg_duration_ms': 0,
                    'success_rate': 0, 'total_prompt_tokens': 0, 'total_response_tokens': 0}

    # -- JSONL migration (one-time) --

    def _migrate_jsonl_to_sqlite(self):
        """Migrate legacy JSONL journal files into journal_entries table."""
        import glob as _glob
        # Check if already migrated
        try:
            cur = self._conn.execute('SELECT COUNT(*) FROM journal_entries')
            if cur.fetchone()[0] > 0:
                return  # already have data, skip
        except Exception:
            return
        journal_dir = os.path.join(os.path.dirname(__file__) or '.', 'journals')
        files = sorted(_glob.glob(os.path.join(journal_dir, 'journal_*.jsonl')))
        if not files:
            return
        total_migrated = 0
        for fpath in files:
            try:
                with open(fpath, 'r') as f:
                    batch = []
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            batch.append((
                                entry.get('timestamp', ''),
                                entry.get('entry_type', ''),
                                entry.get('source', ''),
                                entry.get('symbol', ''),
                                entry.get('content', ''),
                                json.dumps(entry.get('data', {}), default=str),
                                1 if entry.get('llm_call') else 0,
                            ))
                        except json.JSONDecodeError:
                            continue
                    if batch:
                        self._conn.executemany('''
                            INSERT OR IGNORE INTO journal_entries
                            (timestamp, entry_type, source, symbol, content, data, llm_call)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', batch)
                        total_migrated += len(batch)
            except Exception as e:
                logger.warning(f"JSONL migration failed for {fpath}: {e}")
        if total_migrated:
            self._conn.commit()
            logger.info(f"Migrated {total_migrated} journal entries from {len(files)} JSONL files to SQLite")

    @staticmethod
    def _gap_row_to_dict(row) -> dict:
        """Convert a gap SQL row tuple to a dict."""
        sym, date_str, bar_open, bar_high, bar_low, bar_close, volume, \
            prev_close, prev_volume, avg_vol = row
        gap_pct = (bar_open - prev_close) / prev_close
        vol_ratio = prev_volume / avg_vol if avg_vol and avg_vol > 0 else 1.0
        direction = 'short' if gap_pct > 0 else 'long'
        return {
            'date': date_str,
            'symbol': sym,
            'open': float(bar_open),
            'high': float(bar_high),
            'low': float(bar_low),
            'close': float(bar_close),
            'prev_close': float(prev_close),
            'gap_pct': round(gap_pct, 4),
            'volume': int(volume),
            'avg_vol': int(avg_vol) if avg_vol else 0,
            'vol_ratio': round(vol_ratio, 2),
            'direction': direction,
        }


# Module-level singleton (created lazily)
_price_db: Optional[PriceDB] = None


def get_price_db() -> PriceDB:
    global _price_db
    if _price_db is None:
        _price_db = PriceDB()
    return _price_db


# =============================================================================
# SECTION 3: DATA CLASSES & CONFIG
# =============================================================================

# Universe from validated study
LARGE_CAP = [
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','JPM','JNJ',
    'V','PG','UNH','HD','MA','DIS','PYPL','BAC','INTC','VZ',
    'NFLX','ADBE','CRM','CMCSA','PFE','TMO','ABT','CSCO','PEP','AVGO',
    'ACN','COST','NKE','MRK','WMT','LLY','MCD','DHR','TXN','QCOM',
    'MDT','HON','UPS','LOW','MS','GS','BLK','ISRG','SCHW','AXP',
    'BA','CAT','DE','GE','LMT','RTX','MMM','IBM','ORCL','AMD',
    'NOW','SNOW','UBER','ABNB','COIN','SHOP','MELI','SE','SPOT',
    'ZM','DOCU','CRWD','ZS','DDOG','NET','MDB','TEAM','OKTA',
]

MID_CAP = [
    'ROKU','SNAP','PINS','ETSY','DASH','LYFT','HOOD','DKNG','AFRM','SOFI',
    'UPST','RBLX','U','PATH','BILL','HUBS','VEEV','GNRC',
    'ENPH','FSLR','SEDG','RUN','PLUG','CHPT','QS',
    'RIVN','LCID','XPEV','LI','NIO',
    'W','CHWY','PTON','BYND',
    'PLTR','IONQ','RKLB','JOBY','SPCE','ASTS',
    'AI','SOUN','CRSP','EDIT','NTLA','BEAM','IONS','REGN','VRTX','BIIB',
    'MRNA','BNTX',
]

SMALL_CAP = [
    'GME','AMC','KOSS','BB','NOK',
    'MARA','RIOT','HUT','BITF','CIFR','CLSK','IREN','WULF',
    'SMCI','RGTI','DNA','GEVO',
    'SKLZ','FUBO','LMND','ROOT',
    'SPWR','MAXN','ARRY','STEM',
    'LAZR','OUST','MVIS',
    'BILI','PDD','JD','BABA','BIDU',
]

ETFS = ['SPY', 'QQQ', 'IWM', 'ARKK', 'XBI']

UNIVERSE = sorted(set(LARGE_CAP + MID_CAP + SMALL_CAP + ETFS))


@dataclass
class GapFadeConfig:
    """All tunable parameters for the gap fade strategy."""
    # Gap detection (optimized: sweep found 7-10%/50%/3.0x best)
    gap_threshold: float = 0.07        # minimum gap-up % to consider (7%)
    max_gap_pct: float = 0.50          # maximum gap-up % to consider (50%) — filter out M&A/catalyst mega-gaps
    vol_ratio_max: float = 3.0         # only short when gap-day vol < this × avg vol
    min_avg_volume: int = 5_000         # minimum 20d avg daily volume (IEX ~2-3% of consolidated)
    min_price: float = 10.0            # minimum stock price (low-price stocks have wide spreads/slippage)

    # Position sizing
    initial_capital: float = 25_000
    risk_pct: float = 0.02             # risk 2% of equity per trade
    kelly_fraction: float = 0.25       # quarter Kelly
    max_positions: int = 5             # max concurrent positions
    thin_day_threshold: int = 10       # if fewer candidates than this, trade ALL of them

    # Stops and targets (optimized: 2.5% stop, 1/3 partial cover)
    stop_pct: float = 0.015            # 1.5% stop loss above entry (walk-forward optimal)
    partial_target_pct: float = 0.50   # cover fraction when price drops to midpoint
    partial_cover_frac: float = 0.33   # fraction of position to cover at partial target (0.33 = 1/3)
    bounce_entry_pct: float = 0.0      # wait for bounce above open before shorting (0 = disabled)
    # Full target = prev_close (full gap fill)

    # Adaptive stops — scale stop with gap size
    adaptive_stops: bool = True         # if True, stop = gap_pct * stop_gap_fraction (clamped)
    stop_gap_fraction: float = 0.25     # stop = gap_pct * this fraction (25% of gap)
    stop_min_pct: float = 0.015         # floor: 1.5% minimum stop
    stop_max_pct: float = 0.025         # ceiling: 2.5% maximum stop

    # Market regime filter — reduce/block entries on broad rally days
    regime_filter: bool = False
    regime_spy_gap_limit: float = 0.01   # SPY gap > 1% → halve max positions
    regime_spy_block_pct: float = 0.015  # SPY gap > 1.5% → block all entries
    regime_vix_threshold: float = 25.0   # VIX > 25 → halve max positions

    # Re-entry after stop-out
    reentry_enabled: bool = True
    reentry_cooldown_minutes: int = 30   # minutes to wait after stop before re-entry
    reentry_max_per_symbol: int = 1      # max re-entries per symbol per day
    reentry_stop_pct: float = 0.01       # tighter stop on re-entry (1%)
    reentry_trigger_pct: float = 0.0     # price must drop this % below original entry

    # Gap-down fading (longs)
    trade_gap_downs: bool = False
    gap_down_threshold: float = 0.05     # minimum gap-down % to consider (stored positive)
    gap_down_max_pct: float = 0.50       # maximum gap-down %
    gap_down_vol_ratio_max: float = 3.0  # max vol ratio for gap-down candidates

    # Entry cutoff — stop opening new positions after this time
    entry_cutoff_hour: int = 11
    entry_cutoff_min: int = 30

    # Minimum hold time before LLM can take profit (mechanical exits unaffected)
    min_hold_minutes: int = 15

    # Minimum profit thresholds before LLM can close (both must be below to block)
    min_profit_take_pct: float = 0.005   # 0.5% minimum unrealized P&L
    min_gap_fill_pct: float = 0.15       # 15% minimum gap fill

    # Time exits
    time_exit_hour: int = 15           # close remaining by 3:00 PM ET (study edge = full day)
    time_exit_min: int = 0
    eod_exit_hour: int = 15            # force close ALL by 3:50 PM ET
    eod_exit_min: int = 50

    # Circuit breakers
    daily_loss_limit: float = 0.02     # halt if daily P&L <= -2%
    max_consec_losses: int = 3         # pause after N consecutive losses
    max_drawdown: float = 0.05         # halt if drawdown >= 5%

    # Drawdown circuit breakers (backtest + live)
    dd_circuit_breaker: bool = False           # master toggle (off = backward compat)
    dd_tier1_threshold: float = 0.15           # 15% DD → reduce position size
    dd_tier1_scale: float = 0.50               # position size multiplier (0.5 = half)
    dd_tier2_threshold: float = 0.25           # 25% DD → minimal trading
    dd_tier2_scale: float = 0.25               # position size multiplier (0.25 = quarter)
    dd_tier2_max_positions: int = 1            # max concurrent positions in Tier 2
    dd_hard_stop: float = 0.0                  # permanent halt threshold (0 = disabled)

    # Scanner universe
    scan_universe: str = 'alpaca'      # 'alpaca' = all tradeable, 'study' = original 167, 'custom' = user list
    custom_symbols: str = ''           # comma-separated custom list (only used when scan_universe='custom')

    # Position cap
    max_notional: float = 50_000       # max $ value per position (caps compounding)

    # Backtest
    backtest_years: int = 3            # default lookback for backtests

    # Live order execution
    limit_orders_only: bool = True       # True = limit orders (less slippage), False = market orders (faster fill)
    limit_offset_pct: float = 0.001      # limit price offset from current (0.1% = slightly aggressive)

    # Realism adjustments (backtest accuracy)
    slippage_pct: float = 0.0015         # 0.15% slippage per side (entry + exit)
    borrow_rate_annual: float = 0.02     # 2% annualized short borrow fee
    max_pct_adv: float = 0.02           # max 2% of avg daily volume per position
    adverse_fill: bool = True            # assume adverse ordering on ambiguous bars
    adverse_fill_pct: float = 0.50       # probability of adverse fill when ambiguous (0=always favorable, 1=always adverse)

    # Catalyst detection
    catalyst_enabled: bool = True
    catalyst_skip_earnings: bool = False   # hard-skip earnings gaps (False = penalty only)
    catalyst_earnings_penalty: float = 40.0  # heavy penalty for earnings gaps (when not hard-skip)
    catalyst_news_penalty: float = 15.0    # score penalty for news-driven gaps
    catalyst_noise_bonus: float = 5.0      # score bonus for clean noise gaps

    # Automation
    auto_start: bool = True              # auto-start trading loop on app boot

    # Telegram alerts
    alert_telegram_enabled: bool = False
    alert_telegram_token: str = ''       # env TELEGRAM_BOT_TOKEN
    alert_telegram_chat_id: str = ''     # env TELEGRAM_CHAT_ID

    # LLM Supervisor
    llm_enabled: bool = False
    llm_url: str = 'http://localhost:11434'
    llm_model: str = 'gpt-oss:20b'
    llm_timeout: float = 30.0               # seconds per LLM call
    llm_max_failures: int = 5               # circuit breaker opens after N failures
    llm_circuit_reset: float = 120.0        # seconds before retrying after circuit opens
    llm_max_hold_overrides: int = 2          # max times LLM can override a stop per position
    llm_provider: str = 'ollama'             # 'ollama' or 'openai' (OpenAI-compatible: GPT, Groq, Together, Anthropic via proxy)
    llm_api_key: str = ''                    # API key for cloud providers (env LLM_API_KEY as fallback)

    # Strategy plugin
    active_strategy: str = 'classic_gap_fade'  # strategy ID from registry

    def __post_init__(self):
        if not self.llm_api_key:
            self.llm_api_key = os.environ.get('LLM_API_KEY', '')


@dataclass
class GapCandidate:
    """Scanner output — a potential gap fade candidate."""
    symbol: str
    gap_pct: float                     # e.g. 0.08 = 8% gap up
    prev_close: float
    premarket_price: float
    avg_vol_20d: float
    vol_ratio: float                   # premarket or gap-day vol / 20d avg
    shortable: bool
    easy_to_borrow: bool
    score: float = 0.0                 # ranking score
    catalyst: str = ''                 # 'earnings', 'fda', 'ma', 'offering', 'upgrade', 'downgrade', ''
    catalyst_detail: str = ''          # headline snippet or earnings info
    direction: str = 'short'           # 'short' (gap-up fade) or 'long' (gap-down fade)


@dataclass
class GapPosition:
    """Active position (short for gap-up fades, long for gap-down fades)."""
    symbol: str
    shares: int
    entry_price: float
    stop_price: float
    half_target: float                 # cover 50% here
    full_target: float                 # cover remaining here (= prev_close)
    prev_close: float
    partial_filled: bool = False       # has the 50% cover fired?
    entry_time: str = ''
    remaining_shares: int = 0
    # Live trading state
    closing: bool = False              # True while a cover order is pending (prevents duplicate exits)
    entry_order_id: str = ''           # Alpaca order ID for entry
    entry_fill_price: float = 0.0     # actual fill price from broker
    stop_order_id: str = ''           # Alpaca broker-side stop order ID
    direction: str = 'short'          # 'short' or 'long'
    llm_hold_overrides: int = 0       # how many times LLM has overridden stop on this position
    # Profit tracking (updated on every tick/check)
    high_water_pnl_pct: float = 0.0   # best unrealized P&L % seen
    high_water_price: float = 0.0     # price at high water mark
    high_water_time: str = ''         # time of high water mark
    partial_fill_time: str = ''       # when partial cover fired
    last_prices: str = ''             # last 5 prices as comma-separated string (for serialization)
    last_llm_profit_check: float = 0.0  # monotonic time of last LLM profit eval
    # Candidate metadata (copied from GapCandidate on open)
    gap_pct: float = 0.0
    vol_ratio: float = 0.0
    score: float = 0.0
    catalyst: str = ''
    strategy_id: str = ''                # which strategy opened this position

    def __post_init__(self):
        if self.remaining_shares == 0:
            self.remaining_shares = self.shares

    def update_tracking(self, price: float, now_str: str):
        """Update high water mark and price history."""
        # Calculate unrealized P&L %
        if self.direction == 'short':
            pnl_pct = (self.entry_price - price) / self.entry_price
        else:
            pnl_pct = (price - self.entry_price) / self.entry_price

        if pnl_pct > self.high_water_pnl_pct:
            self.high_water_pnl_pct = pnl_pct
            self.high_water_price = price
            self.high_water_time = now_str

        # Ring buffer of last 5 prices
        prices = self.last_prices.split(',') if self.last_prices else []
        prices.append(f'{price:.4f}')
        if len(prices) > 5:
            prices = prices[-5:]
        self.last_prices = ','.join(prices)

    def get_price_history(self) -> List[float]:
        """Return last 5 prices as list of floats."""
        if not self.last_prices:
            return []
        try:
            return [float(p) for p in self.last_prices.split(',') if p]
        except ValueError:
            return []

    def gap_fill_pct(self, current_price: float) -> float:
        """How much of the gap has been filled (0.0 = none, 1.0 = full)."""
        gap = abs(self.entry_price - self.prev_close)
        if gap < 0.001:
            return 0.0
        if self.direction == 'short':
            filled = self.entry_price - current_price
        else:
            filled = current_price - self.entry_price
        return max(0.0, filled / gap)


@dataclass
class DailyStats:
    """Circuit breaker tracking for the day."""
    date: str = ''
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    peak_equity: float = 0.0
    consecutive_losses: int = 0
    halted: bool = False
    halt_reason: str = ''


@dataclass
class TradeRecord:
    """Completed trade for logging."""
    symbol: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    entry_time: str
    exit_time: str
    exit_reason: str                   # 'stop', 'partial', 'full_target', 'time_exit', 'eod', 'manual'
    holding_minutes: int = 0
    side: str = 'short'                # 'short' or 'long' (for gap-down fading)
    # Candidate metadata (for learning from outcomes)
    gap_pct: float = 0.0
    vol_ratio: float = 0.0
    score: float = 0.0
    catalyst: str = ''


@dataclass
class StopOutRecord:
    """Tracks a stopped-out position for potential re-entry."""
    symbol: str
    stop_time: datetime
    original_entry: float
    prev_close: float
    gap_pct: float
    avg_vol_20d: float
    reentry_count: int = 0
    direction: str = 'short'


@dataclass
class MarketRegime:
    """Market context for filtering entries on broad rally/crash days."""
    spy_gap_pct: float = 0.0
    vix_level: float = 20.0
    position_reduction: int = 0    # 0=none, -999=block all entries
    note: str = ''


def fetch_market_regime_backtest(date_str: str, spy_data: Dict[str, dict],
                                  config: GapFadeConfig) -> Optional[MarketRegime]:
    """Compute market regime from pre-cached SPY daily bars for backtest.

    spy_data: dict keyed by date_str → {'prev_close': float, 'open': float}
    """
    if not config.regime_filter:
        return None

    spy = spy_data.get(date_str)
    if not spy or spy['prev_close'] <= 0:
        return None

    spy_gap = (spy['open'] - spy['prev_close']) / spy['prev_close']
    regime = MarketRegime(spy_gap_pct=spy_gap)

    if abs(spy_gap) >= config.regime_spy_block_pct:
        regime.position_reduction = -999
        regime.note = f'SPY gap {spy_gap:+.1%} >= block threshold {config.regime_spy_block_pct:.1%}'
    elif abs(spy_gap) >= config.regime_spy_gap_limit:
        regime.position_reduction = -1   # halve
        regime.note = f'SPY gap {spy_gap:+.1%} >= limit {config.regime_spy_gap_limit:.1%} — halve positions'

    return regime


async def fetch_market_regime_live(config: GapFadeConfig) -> Optional[MarketRegime]:
    """Fetch live market regime from Alpaca snapshot for SPY."""
    if not config.regime_filter:
        return None

    try:
        cfg = _get_alpaca_config()
        if cfg is None:
            return None

        headers = {
            'APCA-API-KEY-ID': cfg['api_key'],
            'APCA-API-SECRET-KEY': cfg['secret_key'],
        }
        async with aiohttp.ClientSession() as session:
            # SPY snapshot for prev_close vs current
            url = f'{cfg["data_url"]}/v2/stocks/SPY/snapshot'
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

            prev_close = float(data.get('prevDailyBar', {}).get('c', 0))
            current = float(data.get('latestTrade', {}).get('p', 0))

            if prev_close <= 0 or current <= 0:
                return None

            spy_gap = (current - prev_close) / prev_close
            regime = MarketRegime(spy_gap_pct=spy_gap)

            if abs(spy_gap) >= config.regime_spy_block_pct:
                regime.position_reduction = -999
                regime.note = f'SPY gap {spy_gap:+.1%} — blocking entries'
            elif abs(spy_gap) >= config.regime_spy_gap_limit:
                regime.position_reduction = -1
                regime.note = f'SPY gap {spy_gap:+.1%} — halving positions'

            return regime
    except Exception as e:
        logger.warning(f"fetch_market_regime_live failed: {e}")
        return None


# =============================================================================
# SECTION 3b: CATALYST DETECTION
# =============================================================================

CATALYST_PATTERNS = {
    'earnings': [
        'earnings', 'quarterly results', 'beats estimates', 'misses estimates',
        'eps', 'revenue', 'guidance', 'fiscal q', 'quarterly report',
        'profit', 'net income', 'earnings call', 'beats expectations',
        'earnings surprise', 'reports q',
    ],
    'fda': [
        'fda approv', 'clinical trial', 'phase 3', 'phase 2', 'pdufa',
        'drug approv', 'fda clear', 'new drug application', 'breakthrough therapy',
        'fda accept', 'fda reject', 'complete response',
    ],
    'ma': [
        'acquir', 'merger', 'buyout', 'takeover', 'tender offer',
        'acquisition', 'merge with', 'deal to buy', 'agreed to buy',
        'purchase agreement',
    ],
    'offering': [
        'offering', 'dilut', 'equity raise', 'capital raise',
        'secondary offering', 'shelf offering', 'public offering',
        'stock offering', 'share sale', 'at-the-market',
    ],
    'upgrade': [
        'upgrade', 'price target raised', 'overweight', 'outperform',
        'buy rating', 'price target increase', 'raises target',
        'initiates coverage', 'bullish',
    ],
    'downgrade': [
        'downgrade', 'price target lowered', 'underweight', 'underperform',
        'sell rating', 'price target cut', 'lowers target', 'bearish',
    ],
}


def _classify_headlines(headlines: List[str]) -> Tuple[str, str]:
    """Classify a list of headlines into a catalyst type.

    Returns (catalyst_type, detail) where catalyst_type is one of the
    CATALYST_PATTERNS keys or '' for noise.
    """
    if not headlines:
        return '', ''

    combined = ' '.join(headlines).lower()
    # Check in priority order: earnings > fda > ma > offering > upgrade > downgrade
    for cat in ('earnings', 'fda', 'ma', 'offering', 'upgrade', 'downgrade'):
        for keyword in CATALYST_PATTERNS[cat]:
            if keyword in combined:
                # Find the headline that matched for the detail
                for h in headlines:
                    if keyword in h.lower():
                        detail = h[:120] if len(h) > 120 else h
                        return cat, detail
                return cat, headlines[0][:120]
    return '', ''


async def _fetch_alpaca_news(symbols: List[str]) -> Dict[str, List[str]]:
    """Fetch recent news headlines from Alpaca News API.

    Returns {symbol: [headline1, headline2, ...]}
    """
    cfg = _get_alpaca_config()
    if not cfg:
        return {}

    headers = {
        'APCA-API-KEY-ID': cfg['api_key'],
        'APCA-API-SECRET-KEY': cfg['secret_key'],
    }
    since = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime('%Y-%m-%dT%H:%M:%SZ')
    sym_str = ','.join(symbols)
    url = f"{cfg['data_url']}/v1beta1/news?symbols={sym_str}&start={since}&limit=50&sort=desc"

    result: Dict[str, List[str]] = defaultdict(list)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=2.5)) as resp:
                if resp.status != 200:
                    logger.debug(f"Alpaca news API returned {resp.status}")
                    return {}
                data = await resp.json()
                for article in data.get('news', []):
                    headline = article.get('headline', '')
                    for sym in article.get('symbols', []):
                        if sym in symbols:
                            result[sym].append(headline)
    except Exception as e:
        logger.debug(f"Alpaca news fetch failed: {e}")
    return dict(result)


async def _check_earnings_batch(symbols: List[str]) -> Dict[str, str]:
    """Check if any symbols reported earnings in the last 2 days using yfinance.

    Returns {symbol: 'earnings YYYY-MM-DD'} for symbols with RECENT PAST earnings.
    Only matches earnings dates within the last 2 calendar days — NOT future dates.
    The old code matched any date >= cutoff, which caught upcoming earnings weeks away
    and caused nearly every stock to be classified as 'earnings' during earnings season.
    """
    result: Dict[str, str] = {}
    now = datetime.now(timezone.utc)
    cutoff_past = now - timedelta(days=2)

    def _check_one(sym: str) -> Optional[Tuple[str, str]]:
        try:
            ticker = yf.Ticker(sym)
            dates = ticker.earnings_dates
            if dates is not None and len(dates) > 0:
                for dt in dates.index:
                    dt_aware = dt.to_pydatetime()
                    if dt_aware.tzinfo is None:
                        dt_aware = dt_aware.replace(tzinfo=timezone.utc)
                    # Only match earnings that ALREADY HAPPENED (past 2 days)
                    # Not future earnings — those don't explain today's gap
                    if cutoff_past <= dt_aware <= now:
                        return sym, f"earnings {dt_aware.strftime('%Y-%m-%d')}"
        except Exception:
            pass
        return None

    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, _check_one, sym) for sym in symbols]
    try:
        done = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=3.0)
        for item in done:
            if isinstance(item, tuple) and item is not None:
                result[item[0]] = item[1]
    except asyncio.TimeoutError:
        logger.debug("yfinance earnings check timed out")
    return result


async def _fetch_google_news_batch(symbols: List[str]) -> Dict[str, List[str]]:
    """Fallback: fetch headlines from Google News RSS for uncovered symbols.

    Returns {symbol: [headline1, ...]}
    """
    result: Dict[str, List[str]] = defaultdict(list)

    async def _fetch_one(session: aiohttp.ClientSession, sym: str):
        url = f"https://news.google.com/rss/search?q={sym}+stock&hl=en-US&gl=US&ceid=US:en"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                if resp.status != 200:
                    return
                text = await resp.text()
                feed = feedparser.parse(text)
                for entry in feed.entries[:5]:
                    result[sym].append(entry.get('title', ''))
        except Exception:
            pass

    try:
        async with aiohttp.ClientSession() as session:
            tasks = [_fetch_one(session, sym) for sym in symbols]
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2.0)
    except asyncio.TimeoutError:
        logger.debug("Google News RSS fetch timed out")
    return dict(result)


async def detect_catalysts(candidates: List[GapCandidate], config: GapFadeConfig):
    """Main catalyst detection: classify each candidate's gap as catalyst-driven or noise.

    Modifies candidates in-place (sets catalyst and catalyst_detail fields).
    Three tiers: Alpaca News (fast) + yfinance earnings (parallel) -> Google News (fallback).
    Hard timeout of 3.0s.
    """
    if not candidates:
        return

    symbols = [c.symbol for c in candidates]
    sym_map = {c.symbol: c for c in candidates}

    try:
        # Tier 1: Alpaca News + yfinance earnings in parallel
        alpaca_task = _fetch_alpaca_news(symbols)
        earnings_task = _check_earnings_batch(symbols)
        news_results, earnings_results = await asyncio.wait_for(
            asyncio.gather(alpaca_task, earnings_task), timeout=3.0
        )

        # Apply earnings results first (highest priority)
        for sym, detail in earnings_results.items():
            if sym in sym_map:
                sym_map[sym].catalyst = 'earnings'
                sym_map[sym].catalyst_detail = detail

        # Apply Alpaca news results for symbols without earnings catalyst
        for sym, headlines in news_results.items():
            if sym in sym_map and not sym_map[sym].catalyst:
                cat, detail = _classify_headlines(headlines)
                if cat:
                    sym_map[sym].catalyst = cat
                    sym_map[sym].catalyst_detail = detail

        # Tier 2: Google News fallback for symbols still uncovered
        uncovered = [s for s in symbols if not sym_map[s].catalyst and s not in news_results]
        if uncovered:
            google_results = await _fetch_google_news_batch(uncovered)
            for sym, headlines in google_results.items():
                if sym in sym_map and not sym_map[sym].catalyst:
                    cat, detail = _classify_headlines(headlines)
                    if cat:
                        sym_map[sym].catalyst = cat
                        sym_map[sym].catalyst_detail = detail

    except asyncio.TimeoutError:
        logger.warning("Catalyst detection hit 3.0s hard timeout — proceeding with partial results")
    except Exception as e:
        logger.warning(f"Catalyst detection error: {e}")


# =============================================================================
# SECTION 4: SCANNER
# =============================================================================

class GapScanner:
    """Pre-market gap detection and filtering."""

    def __init__(self, config: GapFadeConfig, universe: List[str] = None):
        self.config = config
        self._explicit_universe = universe  # None = use config.scan_universe
        self._avg_volumes: Dict[str, float] = {}     # 20d avg volume cache
        self._prev_closes: Dict[str, float] = {}     # previous day close cache
        self._shortable_cache: Dict[str, Tuple[bool, bool]] = {}
        self._cache_date: str = ''

    def _resolve_universe(self) -> List[str]:
        """Resolve the scan universe based on config or explicit override."""
        if self._explicit_universe is not None:
            return self._explicit_universe

        mode = getattr(self.config, 'scan_universe', 'study')
        if mode == 'alpaca':
            assets = fetch_alpaca_assets(min_price=self.config.min_price)
            if assets:
                return assets
            logger.warning("Alpaca assets fetch failed, falling back to study universe")
            return UNIVERSE
        elif mode == 'custom':
            custom = getattr(self.config, 'custom_symbols', '')
            if custom:
                return [s.strip().upper() for s in custom.split(',') if s.strip()]
            return UNIVERSE
        else:  # 'study' or default
            return UNIVERSE

    def _fetch_volume_for_candidates(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch 20d avg volume only for symbols that have a gap. Much faster than caching all."""
        today = datetime.now(ET).strftime('%Y-%m-%d')
        end = datetime.now(ET)
        start = end - timedelta(days=45)
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')

        avg_volumes = {}
        need_api = []
        db = get_price_db()

        for sym in symbols:
            # Use in-memory cache if available today
            if self._cache_date == today and sym in self._avg_volumes:
                avg_volumes[sym] = self._avg_volumes[sym]
                continue
            # Try local DB
            df = db.get_bars(sym, start_str, end_str)
            if df is not None and len(df) >= 5:
                vol_20 = df['volume'].tail(20).mean()
                avg_volumes[sym] = vol_20
                self._avg_volumes[sym] = vol_20
            else:
                need_api.append(sym)

        # Fall back to API for symbols not in DB
        for i, sym in enumerate(need_api):
            try:
                df = fetch_alpaca_bars(sym, start_str, end_str, '1Day', 'iex')
                if df is not None and len(df) >= 5:
                    vol_20 = df['volume'].tail(20).mean()
                    avg_volumes[sym] = vol_20
                    self._avg_volumes[sym] = vol_20
                    # Cache into DB
                    db.upsert_bars(sym, df)
            except Exception:
                pass

            # Rate limit
            if (i + 1) % 80 == 0:
                _time.sleep(1)

        self._cache_date = today
        return avg_volumes

    def _ensure_volume_cache(self):
        """Fetch 30d daily bars to compute 20d avg volume (cached per day).
        Used for study/custom universes where the set is small enough to pre-cache.
        """
        universe = self._resolve_universe()
        today = datetime.now(ET).strftime('%Y-%m-%d')
        if self._cache_date == today and self._avg_volumes:
            return

        logger.info(f"Building volume cache for {len(universe)} symbols...")
        end = datetime.now(ET)
        start = end - timedelta(days=45)
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')

        for i, sym in enumerate(universe):
            try:
                df = fetch_alpaca_bars(sym, start_str, end_str, '1Day', 'iex')
                if df is not None and len(df) >= 5:
                    vol_20 = df['volume'].tail(20).mean()
                    self._avg_volumes[sym] = vol_20
                    self._prev_closes[sym] = float(df['close'].iloc[-1])
            except Exception:
                pass

            # Rate limit
            if (i + 1) % 80 == 0:
                _time.sleep(1)

        self._cache_date = today
        logger.info(f"Volume cache built: {len(self._avg_volumes)} symbols")

    def scan_premarket(self) -> Tuple[List[GapCandidate], int]:
        """Scan for gap-up candidates using Alpaca snapshots.

        Uses snapshot-first approach for large universes: snapshot all symbols first,
        then fetch volume data only for gap candidates.

        Returns (candidates, universe_size).
        """
        universe = self._resolve_universe()
        universe_size = len(universe)
        is_large = universe_size > 500  # Alpaca universe mode

        if not is_large:
            # Small universe: pre-cache volume then snapshot (original flow)
            self._ensure_volume_cache()

        # Fetch snapshots in batch
        logger.info(f"Fetching snapshots for {universe_size} symbols ({self.config.scan_universe} mode)...")
        snapshots = fetch_alpaca_snapshots(universe)
        if not snapshots:
            logger.warning("No snapshot data returned")
            return [], universe_size

        # Phase 1: Find gap candidates from snapshots (no volume filter yet for large universe)
        raw_candidates = []
        for sym, snap in snapshots.items():
            try:
                latest_trade = snap.get('latestTrade', {})
                daily_bar = snap.get('dailyBar', {})

                if not latest_trade or not daily_bar:
                    continue

                current_price = float(latest_trade.get('p', 0))
                # During pre-market: dailyBar = yesterday, prevDailyBar = 2 days ago
                # During market hours: dailyBar = today, prevDailyBar = yesterday
                # Detect which by comparing dailyBar date to today
                import datetime as _dt
                bar_date = daily_bar.get('t', '')[:10]  # 'YYYY-MM-DD'
                today_str = _dt.date.today().strftime('%Y-%m-%d')
                if bar_date == today_str:
                    # Market hours: dailyBar is today, use prevDailyBar for yesterday's close
                    prev_bar = snap.get('prevDailyBar', {})
                    prev_close = float(prev_bar.get('c', 0)) if prev_bar else 0
                else:
                    # Pre-market: dailyBar is yesterday
                    prev_close = float(daily_bar.get('c', 0))

                if prev_close <= 0 or current_price <= 0:
                    continue

                gap_pct = (current_price - prev_close) / prev_close

                # Determine direction
                is_gap_up = gap_pct >= self.config.gap_threshold and \
                            (self.config.max_gap_pct <= 0 or gap_pct <= self.config.max_gap_pct)
                is_gap_down = False
                if self.config.trade_gap_downs and gap_pct < 0:
                    abs_gap = abs(gap_pct)
                    is_gap_down = abs_gap >= self.config.gap_down_threshold and \
                                  abs_gap <= self.config.gap_down_max_pct

                if not is_gap_up and not is_gap_down:
                    continue

                if current_price < self.config.min_price:
                    continue

                direction = 'short' if is_gap_up else 'long'

                # For small universe, check pre-cached volume
                if not is_large:
                    avg_vol = self._avg_volumes.get(sym, 0)
                    if avg_vol < self.config.min_avg_volume:
                        continue
                    # daily_bar is yesterday's completed bar; use its volume as proxy
                    yesterday_vol = float(daily_bar.get('v', 0))
                    vol_ratio = yesterday_vol / avg_vol if avg_vol > 0 and yesterday_vol > 0 else 0.5
                else:
                    avg_vol = 0
                    vol_ratio = 0.5  # placeholder until we fetch volume

                raw_candidates.append(GapCandidate(
                    symbol=sym,
                    gap_pct=gap_pct,
                    prev_close=prev_close,
                    premarket_price=current_price,
                    avg_vol_20d=avg_vol,
                    vol_ratio=vol_ratio,
                    shortable=True,
                    easy_to_borrow=True,
                    score=0.0,
                    direction=direction,
                ))
            except Exception as e:
                logger.debug(f"Snapshot parse error for {sym}: {e}")

        # Phase 2: For large universe, fetch volume only for gap candidates
        if is_large and raw_candidates:
            gap_symbols = [c.symbol for c in raw_candidates]
            logger.info(f"Found {len(gap_symbols)} gap candidates, fetching volume data...")
            avg_volumes = self._fetch_volume_for_candidates(gap_symbols)

            # Apply volume filters
            filtered = []
            for c in raw_candidates:
                avg_vol = avg_volumes.get(c.symbol, 0)
                if avg_vol < self.config.min_avg_volume:
                    continue
                c.avg_vol_20d = avg_vol

                # Recalculate vol_ratio with actual data
                snap = snapshots.get(c.symbol, {})
                daily_bar = snap.get('dailyBar', {})
                today_vol = float(daily_bar.get('v', 0)) if daily_bar else 0
                c.vol_ratio = today_vol / avg_vol if avg_vol > 0 and today_vol > 0 else 0.5
                filtered.append(c)
            raw_candidates = filtered

        # Filter by volume ratio (different limits for gap-ups vs gap-downs)
        def _vol_ok_live(c):
            if c.direction == 'long':
                return c.vol_ratio <= self.config.gap_down_vol_ratio_max
            return c.vol_ratio <= self.config.vol_ratio_max
        candidates = [c for c in raw_candidates if _vol_ok_live(c)]

        # Check shortability for short candidates (longs don't need shortability)
        shorts = [c for c in candidates if c.direction == 'short']
        longs = [c for c in candidates if c.direction == 'long']
        shorts.sort(key=lambda c: c.gap_pct, reverse=True)
        for c in shorts[:20]:
            shortable, etb = alpaca_check_shortable(c.symbol)
            c.shortable = shortable
            c.easy_to_borrow = etb
            _time.sleep(0.15)
        shorts = [c for c in shorts if c.shortable]
        candidates = shorts + longs

        # Score: bigger abs(gap) = higher priority (larger gaps fade more)
        for c in candidates:
            c.score = abs(c.gap_pct) * 100
            if c.easy_to_borrow:
                c.score += 5
            if c.vol_ratio < 0.5:
                c.score += 10  # very low volume = stronger signal

        candidates.sort(key=lambda c: c.score, reverse=True)
        logger.info(f"Scanner found {len(candidates)} gap candidates from {universe_size} symbols")
        return candidates, universe_size

    def scan_historical(self, symbol: str, start_date: str, end_date: str) -> List[dict]:
        """Find historical gap-up days for backtesting (single symbol)."""
        # Fetch extra days for rolling avg volume
        adj_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=45)).strftime('%Y-%m-%d')
        df = fetch_alpaca_bars(symbol, adj_start, end_date, '1Day', 'iex')
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        gd_cfg = None
        if self.config.trade_gap_downs:
            gd_cfg = {
                'threshold': self.config.gap_down_threshold,
                'max_pct': self.config.gap_down_max_pct,
                'vol_ratio_max': self.config.gap_down_vol_ratio_max,
            }
        return self._find_gaps_in_df(df, symbol, after_date=start_dt,
                                     gap_threshold=self.config.gap_threshold,
                                     max_gap_pct=self.config.max_gap_pct if self.config.max_gap_pct > 0 else None,
                                     min_price=self.config.min_price,
                                     min_avg_volume=self.config.min_avg_volume,
                                     gap_down_config=gd_cfg)

    def scan_historical_batch(self, symbols: List[str], start_date: str,
                              end_date: str, progress_callback=None) -> List[dict]:
        """Find historical gap-up days for ALL symbols.

        Fast path: SQL window functions on local PriceDB (~1-2s for 11k symbols).
        Slow path: fetch missing symbols from API, then SQL scan.
        Returns list of gap dicts sorted by date.
        """
        adj_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=45)).strftime('%Y-%m-%d')
        db = get_price_db()

        # Check which symbols are missing from DB
        db_symbols = set(db.get_symbols())
        missing_symbols = [s for s in symbols if s not in db_symbols]

        if missing_symbols:
            logger.info(f"PriceDB has {len(db_symbols)} symbols, "
                        f"fetching {len(missing_symbols)} missing from API...")
            if progress_callback:
                progress_callback(0, len(missing_symbols))
            api_dfs = fetch_alpaca_bars_multi(missing_symbols, adj_start, end_date, '1Day', 'iex')
            if api_dfs:
                db.upsert_bars_batch(api_dfs)
                logger.info(f"Cached {len(api_dfs)} new symbols into PriceDB")
        else:
            logger.info(f"PriceDB has all {len(symbols)} symbols cached")

        # Fast SQL gap scan — no DataFrame construction
        logger.info(f"SQL gap scan for {len(symbols)} symbols...")
        t0 = _time.time()
        gd_cfg = None
        if self.config.trade_gap_downs:
            gd_cfg = {
                'threshold': self.config.gap_down_threshold,
                'max_pct': self.config.gap_down_max_pct,
                'vol_ratio_max': self.config.gap_down_vol_ratio_max,
            }
        all_gaps = db.scan_gaps_sql(
            symbols, start_date, end_date,
            gap_threshold=self.config.gap_threshold,
            max_gap_pct=self.config.max_gap_pct if self.config.max_gap_pct > 0 else None,
            min_price=self.config.min_price,
            min_avg_volume=self.config.min_avg_volume,
            gap_down_config=gd_cfg,
        )
        elapsed = _time.time() - t0
        logger.info(f"SQL gap scan: {len(all_gaps)} gaps in {elapsed:.1f}s")
        return all_gaps

    @staticmethod
    def _find_gaps_in_df(df: Optional[pd.DataFrame], symbol: str,
                         after_date: datetime = None, min_bars: int = 5,
                         gap_threshold: float = None, max_gap_pct: float = None,
                         min_price: float = None,
                         min_avg_volume: int = None,
                         gap_down_config: dict = None) -> List[dict]:
        """Extract gap days from a DataFrame of daily bars.

        Detects gap-ups (and gap-downs if gap_down_config provided).
        Shared logic for both single-symbol and batch scanning.
        """
        if df is None or len(df) < min_bars:
            return []

        opens = df['open'].values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values
        dates = df.index

        # Use a shorter window if not enough bars for 20-day rolling
        window = min(20, len(df) - 1)
        avg_vol = pd.Series(volumes).rolling(window).mean().values

        gaps = []
        for i in range(1, len(df)):
            # Skip bars before the requested start date
            if after_date is not None:
                bar_dt = dates[i].to_pydatetime() if hasattr(dates[i], 'to_pydatetime') else dates[i]
                if hasattr(bar_dt, 'replace'):
                    bar_dt = bar_dt.replace(tzinfo=None)
                if bar_dt < after_date:
                    continue

            prev_c = closes[i-1]
            if prev_c <= 0 or opens[i] <= 0:
                continue
            gap = (opens[i] - prev_c) / prev_c

            # Determine if this qualifies as a gap-up or gap-down
            is_gap_up = (gap_threshold is None or gap >= gap_threshold) and \
                        (max_gap_pct is None or gap <= max_gap_pct) and gap > 0
            is_gap_down = False
            if gap_down_config and gap < 0:
                gd_thresh = gap_down_config.get('threshold', 0.05)
                gd_max = gap_down_config.get('max_pct', 0.50)
                abs_gap = abs(gap)
                is_gap_down = abs_gap >= gd_thresh and abs_gap <= gd_max

            if not is_gap_up and not is_gap_down:
                continue

            if min_price is not None and opens[i] < min_price:
                continue

            # Use avg volume computed up to the PREVIOUS day (no look-ahead)
            prev_avg_vol = avg_vol[i-1] if i >= 1 and not np.isnan(avg_vol[i-1]) else (
                avg_vol[i] if not np.isnan(avg_vol[i]) else volumes[i-1])
            if min_avg_volume is not None and prev_avg_vol < min_avg_volume:
                continue

            # vol_ratio uses PREVIOUS day's volume (known at pre-market)
            vol_ratio = volumes[i-1] / prev_avg_vol if prev_avg_vol > 0 else 1.0

            direction = 'short' if gap > 0 else 'long'

            gaps.append({
                'date': dates[i].strftime('%Y-%m-%d') if hasattr(dates[i], 'strftime') else str(dates[i])[:10],
                'symbol': symbol,
                'open': float(opens[i]),
                'high': float(highs[i]),
                'low': float(lows[i]),
                'close': float(closes[i]),
                'prev_close': float(prev_c),
                'gap_pct': round(gap, 4),
                'volume': int(volumes[i]),
                'avg_vol': int(prev_avg_vol),
                'vol_ratio': round(vol_ratio, 2),
                'direction': direction,
            })

        return gaps


# =============================================================================
# SECTION 5: STRATEGY ENGINE
# =============================================================================

def compute_adaptive_stop_pct(config: GapFadeConfig, gap_pct: float) -> float:
    """Compute stop-loss % based on gap size when adaptive_stops is enabled.

    Larger gaps get wider stops (proportional to gap_pct * stop_gap_fraction),
    clamped to [stop_min_pct, stop_max_pct]. Falls back to fixed config.stop_pct
    when adaptive_stops is disabled.
    """
    if not config.adaptive_stops:
        return config.stop_pct
    return max(config.stop_min_pct, min(config.stop_max_pct,
                                         abs(gap_pct) * config.stop_gap_fraction))


def _direction_pnl(direction: str, entry: float, exit_price: float, shares: int) -> float:
    """Compute P&L for a trade given its direction."""
    if direction == 'long':
        return (exit_price - entry) * shares
    return (entry - exit_price) * shares


def _stop_hit(direction: str, bar_high: float, bar_low: float, stop_price: float) -> bool:
    """Check if stop price was hit given direction."""
    if direction == 'long':
        return bar_low <= stop_price  # long stop is below entry
    return bar_high >= stop_price     # short stop is above entry


def _target_hit(direction: str, price: float, target: float) -> bool:
    """Check if target price was hit given direction."""
    if direction == 'long':
        return price >= target        # long target is above entry
    return price <= target            # short target is below entry


class GapFadeEngine:
    """Entry/exit logic, position sizing, and circuit breakers."""

    # From study data: vol < 1x gap fade statistics
    STUDY_WIN_RATE = 0.709
    STUDY_AVG_WIN = 2.5      # %
    STUDY_AVG_LOSS = -3.0    # % (with 3% stop)

    def __init__(self, config: GapFadeConfig, backtest_mode: bool = False):
        self.config = config
        self.backtest_mode = backtest_mode  # skip circuit breakers in backtest
        self.positions: Dict[str, GapPosition] = {}
        self.daily_stats = DailyStats()
        self.equity = config.initial_capital
        self.peak_equity = config.initial_capital
        self._stopped_today: Dict[str, StopOutRecord] = {}  # re-entry tracking
        self.trade_log: List[TradeRecord] = []
        self.all_trade_log: List[TradeRecord] = []   # persists across resets
        self._effective_max_positions: int = config.max_positions  # updated per scan

    def effective_max_positions(self, n_candidates: int) -> int:
        """If fewer candidates than thin_day_threshold, trade all of them."""
        if n_candidates < self.config.thin_day_threshold:
            return n_candidates
        return self.config.max_positions

    def compute_kelly_size(self) -> float:
        """Compute optimal position size using Kelly criterion.
        Uses study data: p=0.709 win rate with vol<1x filter.
        Returns fraction of equity to risk per trade.
        """
        p = self.STUDY_WIN_RATE
        q = 1 - p
        b = abs(self.STUDY_AVG_WIN / self.STUDY_AVG_LOSS)  # win/loss ratio
        kelly = (p * b - q) / b if b > 0 else 0
        return max(0, kelly * self.config.kelly_fraction)

    def compute_position_size(self, entry_price: float, stop_price: float) -> int:
        """Compute number of shares to short based on risk and Kelly sizing."""
        risk_per_share = abs(stop_price - entry_price)
        if risk_per_share <= 0:
            return 0

        # Kelly-adjusted risk
        kelly_risk = self.compute_kelly_size()
        risk_frac = min(self.config.risk_pct, kelly_risk) if kelly_risk > 0 else self.config.risk_pct

        dollar_risk = self.equity * risk_frac
        shares = int(dollar_risk / risk_per_share)

        # Per-position cap: divide equity by effective max so all slots fit without leverage
        eff_max = self._effective_max_positions
        open_count = len(self.positions)
        slots = max(1, eff_max - open_count)
        per_slot_equity = self.equity / max(1, eff_max)
        max_shares = int(per_slot_equity / entry_price) if entry_price > 0 else 0
        shares = min(shares, max_shares)

        # Absolute notional cap (also per-slot)
        if entry_price > 0 and self.config.max_notional > 0:
            notional_limit = min(self.config.max_notional, per_slot_equity)
            max_shares_notional = int(notional_limit / entry_price)
            shares = min(shares, max_shares_notional)

        return max(0, shares)

    def can_reenter(self, symbol: str, current_price: float,
                    current_time: datetime) -> Tuple[bool, str]:
        """Check if a stopped-out symbol qualifies for re-entry."""
        if not self.config.reentry_enabled:
            return False, "re-entry disabled"
        rec = self._stopped_today.get(symbol)
        if rec is None:
            return False, "no stop-out record"
        if rec.reentry_count >= self.config.reentry_max_per_symbol:
            return False, f"max re-entries ({self.config.reentry_max_per_symbol}) reached"
        elapsed = (current_time - rec.stop_time).total_seconds() / 60
        if elapsed < self.config.reentry_cooldown_minutes:
            return False, f"cooldown ({elapsed:.0f}m < {self.config.reentry_cooldown_minutes}m)"
        # Trigger: price must move favorably past original entry by trigger_pct
        if self.config.reentry_trigger_pct > 0:
            if rec.direction == 'long':
                # Long re-entry: price must rise above original entry
                trigger_price = rec.original_entry * (1 + self.config.reentry_trigger_pct)
                if current_price < trigger_price:
                    return False, f"price ${current_price:.2f} < trigger ${trigger_price:.2f}"
            else:
                # Short re-entry: price must drop below original entry
                trigger_price = rec.original_entry * (1 - self.config.reentry_trigger_pct)
                if current_price > trigger_price:
                    return False, f"price ${current_price:.2f} > trigger ${trigger_price:.2f}"
        return True, "re-entry eligible"

    def should_enter(self, candidate: GapCandidate) -> Tuple[bool, str]:
        """Check if we should enter a new short. Returns (ok, reason)."""
        # Catalyst-driven gaps: never short into earnings or M&A
        if candidate.catalyst in ('earnings', 'ma'):
            return False, f"catalyst-driven gap ({candidate.catalyst})"

        # Entry timing guards (live mode only)
        if not self.backtest_mode:
            now = datetime.now(ET)
            # Pre-market guard — no entries before 9:30 AM
            if now.hour < 9 or (now.hour == 9 and now.minute < 30):
                return False, f"market not open yet ({now.strftime('%H:%M')} ET, opens 9:30)"
            # Entry cutoff — no new positions after configured time
            cutoff = now.replace(hour=self.config.entry_cutoff_hour,
                                 minute=self.config.entry_cutoff_min, second=0, microsecond=0)
            if now >= cutoff:
                return False, f"past entry cutoff ({self.config.entry_cutoff_hour}:{self.config.entry_cutoff_min:02d})"

        # Already in this symbol?
        if candidate.symbol in self.positions:
            return False, "already in position"

        # Re-entry cap — prevent churning stopped-out symbols
        if candidate.symbol in self._stopped_today:
            rec = self._stopped_today[candidate.symbol]
            if rec.reentry_count >= self.config.reentry_max_per_symbol:
                return False, f"re-entry cap reached for {candidate.symbol} ({rec.reentry_count}/{self.config.reentry_max_per_symbol})"

        # Max positions (applies in both modes) — uses effective max for thin days
        eff_max = self._effective_max_positions
        if len(self.positions) >= eff_max:
            return False, f"max positions ({eff_max}) reached"

        # Volume ratio check (applies in both modes; different limit for gap-downs)
        vol_limit = self.config.gap_down_vol_ratio_max if candidate.direction == 'long' else self.config.vol_ratio_max
        if candidate.vol_ratio > vol_limit:
            return False, f"vol ratio {candidate.vol_ratio:.1f} > {vol_limit}"

        # Circuit breakers — skip in backtest mode (measure raw edge)
        if not self.backtest_mode:
            if self.daily_stats.halted:
                return False, f"halted: {self.daily_stats.halt_reason}"

            if self.daily_stats.consecutive_losses >= self.config.max_consec_losses:
                return False, f"consecutive losses ({self.daily_stats.consecutive_losses}) >= limit"

            if self.equity > 0:
                daily_return = self.daily_stats.pnl / self.config.initial_capital
                if daily_return <= -self.config.daily_loss_limit:
                    self.daily_stats.halted = True
                    self.daily_stats.halt_reason = f"daily loss limit ({daily_return:.1%})"
                    return False, self.daily_stats.halt_reason

            dd = (self.peak_equity - self.equity) / self.config.initial_capital if self.config.initial_capital > 0 else 0
            if dd >= self.config.max_drawdown:
                self.daily_stats.halted = True
                self.daily_stats.halt_reason = f"max drawdown ({dd:.1%})"
                return False, self.daily_stats.halt_reason

        return True, "ok"

    def open_position(self, candidate: GapCandidate, entry_price: float,
                      entry_time: str) -> Optional[GapPosition]:
        """Create a new position. Direction-aware: short for gap-ups, long for gap-downs."""
        direction = candidate.direction
        # Slippage: adverse fill (short=lower, long=higher)
        slip = self.config.slippage_pct if self.backtest_mode else 0
        if direction == 'long':
            fill_price = entry_price * (1 + slip)  # long gets worse (higher) fill
        else:
            fill_price = entry_price * (1 - slip)  # short gets worse (lower) fill

        eff_stop_pct = compute_adaptive_stop_pct(self.config, candidate.gap_pct)
        if direction == 'long':
            stop_price = fill_price * (1 - eff_stop_pct)  # long stop below entry
            half_target = (fill_price + candidate.prev_close) / 2
            full_target = candidate.prev_close  # gap fill = back up to prev close
        else:
            stop_price = fill_price * (1 + eff_stop_pct)  # short stop above entry
            half_target = (fill_price + candidate.prev_close) / 2
            full_target = candidate.prev_close  # gap fill = back down to prev close

        shares = self.compute_position_size(fill_price, stop_price)
        # Liquidity cap: max % of average daily volume
        if candidate.avg_vol_20d > 0 and self.config.max_pct_adv > 0:
            max_liq = int(candidate.avg_vol_20d * self.config.max_pct_adv)
            shares = min(shares, max_liq)
        if shares <= 0:
            return None

        pos = GapPosition(
            symbol=candidate.symbol,
            shares=shares,
            entry_price=fill_price,
            stop_price=stop_price,
            half_target=half_target,
            full_target=full_target,
            prev_close=candidate.prev_close,
            entry_time=entry_time,
            remaining_shares=shares,
            direction=direction,
            gap_pct=candidate.gap_pct,
            vol_ratio=candidate.vol_ratio,
            score=candidate.score,
            catalyst=candidate.catalyst,
        )
        self.positions[candidate.symbol] = pos
        self.daily_stats.trades += 1
        return pos

    def check_exits(self, symbol: str, price: float, high: float,
                    current_time: datetime) -> List[TradeRecord]:
        """Check exit conditions for a position. Returns list of trades closed."""
        if symbol not in self.positions:
            return []

        pos = self.positions[symbol]
        trades = []
        now_str = current_time.strftime('%Y-%m-%d %H:%M')
        # Slippage: buy-to-cover gets worse (higher) fill in backtest
        slip = self.config.slippage_pct if self.backtest_mode else 0

        # Parse entry time for holding duration
        try:
            entry_dt = datetime.strptime(pos.entry_time, '%Y-%m-%d %H:%M')
            if current_time.tzinfo and not entry_dt.tzinfo:
                entry_dt = entry_dt.replace(tzinfo=current_time.tzinfo)
            holding_min = int((current_time - entry_dt).total_seconds() / 60)
        except (ValueError, TypeError):
            holding_min = 0

        d = pos.direction
        # Exit slippage: adverse fill direction depends on position direction
        # Short: covering = buy, slippage makes fill higher
        # Long: selling = sell, slippage makes fill lower
        exit_slip = lambda p: p * (1 + slip) if d == 'short' else p * (1 - slip)

        def _make_trade(exit_px, reason, shares_out):
            fp = exit_slip(exit_px)
            pnl = _direction_pnl(d, pos.entry_price, fp, shares_out)
            pnl_pct = pnl / (pos.entry_price * shares_out) if pos.entry_price > 0 else 0
            return TradeRecord(
                symbol=symbol, entry_price=pos.entry_price, exit_price=fp,
                shares=shares_out, pnl=pnl, pnl_pct=pnl_pct,
                entry_time=pos.entry_time, exit_time=now_str,
                exit_reason=reason, holding_minutes=holding_min,
                side=d,
                gap_pct=pos.gap_pct, vol_ratio=pos.vol_ratio,
                score=pos.score, catalyst=pos.catalyst,
            )

        # 1. Stop loss — direction-aware check
        if _stop_hit(d, high, price, pos.stop_price):
            trade = _make_trade(pos.stop_price, 'stop', pos.remaining_shares)
            trades.append(trade)
            self._record_trade(trade)
            # Save stop-out record for potential re-entry
            if self.config.reentry_enabled:
                prev = self._stopped_today.get(symbol)
                cnt = (prev.reentry_count if prev else 0)
                self._stopped_today[symbol] = StopOutRecord(
                    symbol=symbol, stop_time=current_time,
                    original_entry=pos.entry_price,
                    prev_close=pos.prev_close,
                    gap_pct=pos.gap_pct,
                    avg_vol_20d=0.0,
                    reentry_count=cnt,
                    direction=d,
                )
            del self.positions[symbol]
            return trades

        # 2. Partial profit — cover fraction when price hits midpoint target
        if not pos.partial_filled and _target_hit(d, price, pos.half_target):
            cover_shares = max(1, int(pos.remaining_shares * self.config.partial_cover_frac))
            if cover_shares > 0:
                trade = _make_trade(price, 'partial', cover_shares)
                trades.append(trade)
                self._record_trade(trade)
                pos.remaining_shares -= cover_shares
                pos.partial_filled = True
                pos.partial_fill_time = datetime.now(ET).strftime('%H:%M:%S')
                # Move stop to breakeven after partial fill
                pos.stop_price = pos.entry_price

        # 3. Full target — cover remaining when price reaches prev_close
        if _target_hit(d, price, pos.full_target) and pos.remaining_shares > 0:
            trade = _make_trade(price, 'full_target', pos.remaining_shares)
            trades.append(trade)
            self._record_trade(trade)
            del self.positions[symbol]
            return trades

        # 4. Time exit
        et_hour = current_time.hour if current_time.tzinfo else current_time.hour
        et_min = current_time.minute
        if (et_hour > self.config.time_exit_hour or
            (et_hour == self.config.time_exit_hour and et_min >= self.config.time_exit_min)):
            if pos.remaining_shares > 0:
                trade = _make_trade(price, 'time_exit', pos.remaining_shares)
                trades.append(trade)
                self._record_trade(trade)
                del self.positions[symbol]
                return trades

        # 5. EOD exit
        if (et_hour > self.config.eod_exit_hour or
            (et_hour == self.config.eod_exit_hour and et_min >= self.config.eod_exit_min)):
            if pos.remaining_shares > 0:
                trade = _make_trade(price, 'eod', pos.remaining_shares)
                trades.append(trade)
                self._record_trade(trade)
                del self.positions[symbol]
                return trades

        return trades

    def evaluate_exit(self, symbol: str, price: float, high: float,
                      current_time: datetime) -> Optional[dict]:
        """Evaluate if a position should exit WITHOUT mutating state.

        Returns None if no exit, or a dict with:
          {'reason': str, 'shares': int, 'is_full_close': bool, 'trigger_price': float}

        The live trader calls this, then places the order, then calls
        confirm_exit() with the actual fill price to finalize.
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        if pos.closing:
            return None  # already has a pending cover order

        et_hour = current_time.hour
        et_min = current_time.minute

        d = pos.direction

        # 1. Stop loss — direction-aware check
        if _stop_hit(d, high, price, pos.stop_price):
            return {'reason': 'stop', 'shares': pos.remaining_shares,
                    'is_full_close': True, 'trigger_price': pos.stop_price}

        # 2. Partial profit — cover fraction at midpoint target
        if not pos.partial_filled and _target_hit(d, price, pos.half_target):
            cover_shares = max(1, int(pos.remaining_shares * self.config.partial_cover_frac))
            if cover_shares > 0:
                return {'reason': 'partial', 'shares': cover_shares,
                        'is_full_close': False, 'trigger_price': price}

        # 3. Full target — cover remaining at prev_close
        if _target_hit(d, price, pos.full_target) and pos.remaining_shares > 0:
            return {'reason': 'full_target', 'shares': pos.remaining_shares,
                    'is_full_close': True, 'trigger_price': price}

        # 4. Time exit
        if (et_hour > self.config.time_exit_hour or
            (et_hour == self.config.time_exit_hour and et_min >= self.config.time_exit_min)):
            if pos.remaining_shares > 0:
                return {'reason': 'time_exit', 'shares': pos.remaining_shares,
                        'is_full_close': True, 'trigger_price': price}

        # 5. EOD exit
        if (et_hour > self.config.eod_exit_hour or
            (et_hour == self.config.eod_exit_hour and et_min >= self.config.eod_exit_min)):
            if pos.remaining_shares > 0:
                return {'reason': 'eod', 'shares': pos.remaining_shares,
                        'is_full_close': True, 'trigger_price': price}

        return None

    def confirm_exit(self, symbol: str, exit_shares: int, fill_price: float,
                     reason: str, current_time: datetime) -> Optional[TradeRecord]:
        """Finalize an exit after the cover order has been confirmed filled.

        Mutates position state and records the trade. Call only after broker
        confirms the fill.
        """
        if symbol not in self.positions:
            logger.warning(f"confirm_exit: {symbol} not in positions")
            return None

        pos = self.positions[symbol]
        now_str = current_time.strftime('%Y-%m-%d %H:%M')

        try:
            entry_dt = datetime.strptime(pos.entry_time, '%Y-%m-%d %H:%M')
            if current_time.tzinfo and not entry_dt.tzinfo:
                entry_dt = entry_dt.replace(tzinfo=current_time.tzinfo)
            holding_min = int((current_time - entry_dt).total_seconds() / 60)
        except (ValueError, TypeError):
            holding_min = 0

        # Use actual entry fill price if available, fall back to position entry_price
        entry_px = pos.entry_fill_price if pos.entry_fill_price > 0 else pos.entry_price
        pnl = _direction_pnl(pos.direction, entry_px, fill_price, exit_shares)
        pnl_pct = pnl / (entry_px * exit_shares) if entry_px > 0 else 0

        trade = TradeRecord(
            symbol=symbol, entry_price=entry_px, exit_price=fill_price,
            shares=exit_shares, pnl=pnl, pnl_pct=pnl_pct,
            entry_time=pos.entry_time, exit_time=now_str,
            exit_reason=reason, holding_minutes=holding_min,
            side=pos.direction,
            gap_pct=pos.gap_pct, vol_ratio=pos.vol_ratio,
            score=pos.score, catalyst=pos.catalyst,
        )
        self._record_trade(trade)

        pos.remaining_shares -= exit_shares
        pos.closing = False

        if reason == 'partial':
            pos.partial_filled = True
            pos.partial_fill_time = current_time.strftime('%H:%M:%S')
            pos.stop_price = entry_px  # move stop to breakeven

        if pos.remaining_shares <= 0:
            del self.positions[symbol]

        return trade

    def force_close_all(self, prices: Dict[str, float], reason: str = 'eod') -> List[TradeRecord]:
        """Force close all positions at current prices (backtest mode)."""
        trades = []
        now = datetime.now(ET)
        now_str = now.strftime('%Y-%m-%d %H:%M')
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            price = prices.get(sym, pos.entry_price)
            pnl = _direction_pnl(pos.direction, pos.entry_price, price, pos.remaining_shares)
            pnl_pct = pnl / (pos.entry_price * pos.remaining_shares) if pos.entry_price > 0 and pos.remaining_shares > 0 else 0
            try:
                entry_dt = datetime.strptime(pos.entry_time, '%Y-%m-%d %H:%M')
                if now.tzinfo and not entry_dt.tzinfo:
                    entry_dt = entry_dt.replace(tzinfo=now.tzinfo)
                holding_min = int((now - entry_dt).total_seconds() / 60)
            except (ValueError, TypeError):
                holding_min = 0
            trades.append(TradeRecord(
                symbol=sym, entry_price=pos.entry_price, exit_price=price,
                shares=pos.remaining_shares, pnl=pnl, pnl_pct=pnl_pct,
                entry_time=pos.entry_time, exit_time=now_str,
                exit_reason=reason, holding_minutes=holding_min, side=pos.direction,
                gap_pct=pos.gap_pct, vol_ratio=pos.vol_ratio,
                score=pos.score, catalyst=pos.catalyst,
            ))
            self._record_trade(trades[-1])
            del self.positions[sym]
        return trades

    def _record_trade(self, trade: TradeRecord):
        """Update stats after a trade closes."""
        self.daily_stats.pnl += trade.pnl
        self.equity += trade.pnl
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        if trade.pnl > 0:
            self.daily_stats.wins += 1
            self.daily_stats.consecutive_losses = 0
        else:
            self.daily_stats.losses += 1
            self.daily_stats.consecutive_losses += 1

        self.trade_log.append(trade)
        self.all_trade_log.append(trade)

        # Persist to SQLite (live trades only — backtests stay out)
        if not self.backtest_mode:
            try:
                get_price_db().insert_trade(trade)
            except Exception as e:
                logger.error(f"SQLite trade persist failed (non-fatal): {e}")

    def reset_daily(self):
        """Reset daily stats for new trading day."""
        self.daily_stats = DailyStats(
            date=datetime.now(ET).strftime('%Y-%m-%d'),
            peak_equity=self.equity,
        )
        self.trade_log = []
        self._stopped_today = {}

    def get_metrics(self) -> dict:
        """Compute performance metrics from all trades."""
        trades = self.all_trade_log
        if not trades:
            return {'total_trades': 0}

        pnls = [t.pnl for t in trades]
        pnl_pcts = [t.pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 9999.99

        # Max drawdown from cumulative P&L
        cum_pnl = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum_pnl)
        drawdowns = peak - cum_pnl
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0
        max_dd_pct = max_dd / self.config.initial_capital if self.config.initial_capital > 0 else 0

        # Sharpe (daily, annualized rough estimate)
        if len(pnl_pcts) > 1:
            sharpe = np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(252) if np.std(pnl_pcts) > 0 else 0
        else:
            sharpe = 0

        avg_holding = np.mean([t.holding_minutes for t in trades]) if trades else 0

        return {
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(win_rate, 4),
            'total_pnl': round(total_pnl, 2),
            'return_pct': round(total_pnl / self.config.initial_capital * 100, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown': round(max_dd, 2),
            'max_drawdown_pct': round(max_dd_pct * 100, 2),
            'sharpe': round(sharpe, 2),
            'avg_holding_min': round(avg_holding, 1),
            'final_equity': round(self.equity, 2),
        }


# =============================================================================
# SECTION 6: BACKTESTER
# =============================================================================

# Use vectorbt-powered backtester if available, else keep original
try:
    from gap_fade_backtester import VbtGapFadeBacktester as _VbtBacktester
    _USE_VBT_BACKTESTER = True
except ImportError:
    _USE_VBT_BACKTESTER = False


class GapFadeBacktester:
    """Backtest gap fade strategy using 1-minute Alpaca bars."""

    def __init__(self, config: GapFadeConfig = None, strategy_id: str = ''):
        self.config = config or GapFadeConfig()
        self.progress = 0.0
        self.status = 'idle'
        self._cancel = False
        self.result = None
        self.strategy = None
        self.strategy_id = strategy_id
        if strategy_id:
            self._load_strategy(strategy_id)

    def _load_strategy(self, strategy_id: str, strategy_config: dict = None):
        """Load a strategy for backtest use."""
        try:
            from gap_fade_strategies import GapFadeStrategyRegistry
            if GapFadeStrategyRegistry.has_strategy(strategy_id):
                self.strategy = GapFadeStrategyRegistry.create_strategy(strategy_id, strategy_config)
                self.strategy_id = strategy_id
                logger.info(f"Backtest strategy: {self.strategy.name} ({strategy_id})")
            else:
                logger.warning(f"Backtest strategy '{strategy_id}' not found, using engine defaults")
                self.strategy = None
                self.strategy_id = ''
        except Exception as e:
            logger.warning(f"Failed to load backtest strategy '{strategy_id}': {e}")
            self.strategy = None
            self.strategy_id = ''

    async def run(self, symbol: str = None, symbols: List[str] = None,
                  start_date: str = None, end_date: str = None,
                  config: GapFadeConfig = None,
                  progress_callback=None, **kwargs) -> dict:
        """Run backtest. use_1min=True for detailed 1-min bar simulation (slow).

        kwargs:
            strategy_id: str — strategy to use (default: engine defaults)
            strategy_config: dict — strategy-specific config overrides
        """
        if config:
            self.config = config

        # Load strategy if specified
        strategy_id = kwargs.get('strategy_id', self.strategy_id or '')
        strategy_config = kwargs.get('strategy_config')
        if strategy_id:
            self._load_strategy(strategy_id, strategy_config)

        self.status = 'running'
        self._cancel = False
        self.progress = 0
        self.bt_log: List[dict] = []  # detailed decision log

        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=self.config.backtest_years * 365)).strftime('%Y-%m-%d')

        syms = symbols or ([symbol] if symbol else UNIVERSE)
        logger.info(f"BACKTEST START: strategy={strategy_id or 'engine'}, "
                     f"symbols={len(syms)}, range={start_date} to {end_date}")

        engine = GapFadeEngine(self.config, backtest_mode=True)
        scanner = GapScanner(self.config, syms)
        all_gap_days = []
        raw_gap_count = 0  # before vol filter

        async def _log(level: str, msg: str, data: dict = None):
            """Append to bt_log and broadcast to dashboard."""
            entry = {'level': level, 'msg': msg, 'data': data or {}}
            self.bt_log.append(entry)
            await broadcast({'type': 'bt_log', 'entry': entry})

        # Seed random for reproducible adverse fill resolution
        import random
        random.seed(42)

        # Step 1: Find gap days — use batch API for large universes
        total_symbols = len(syms)
        max_gap_label = f', max_gap <= {self.config.max_gap_pct:.0%}' if self.config.max_gap_pct > 0 else ''
        await _log('info', f'Scanning {total_symbols} symbols for gap-ups >= {self.config.gap_threshold:.0%}{max_gap_label}, '
                   f'vol_ratio <= {self.config.vol_ratio_max}x, stop = {self.config.stop_pct:.0%}')

        # Vol filter helper (different limits for gap-ups vs gap-downs)
        def _vol_ok(g):
            if g.get('direction') == 'long':
                if not self.config.trade_gap_downs:
                    return False  # reject gap-downs when feature disabled
                return g['vol_ratio'] <= self.config.gap_down_vol_ratio_max
            return g['vol_ratio'] <= self.config.vol_ratio_max

        use_batch = total_symbols > 50  # batch API for large universes

        if use_batch:
            # Batch mode: fetch all symbols' daily bars in multi-symbol requests
            await _log('info', f'Using batch API for {total_symbols} symbols...')
            if progress_callback:
                await progress_callback(2, f"Fetching daily bars for {total_symbols} symbols (batch)...")

            def _batch_progress(done, total):
                pct = (done / total) * 25 if total > 0 else 0
                self.progress = pct

            all_gap_days = await asyncio.to_thread(
                scanner.scan_historical_batch,
                syms, start_date, end_date,
                progress_callback=_batch_progress,
            )
            raw_gap_count = len(all_gap_days)

            if progress_callback:
                await progress_callback(25, f"Found {raw_gap_count} raw gaps, filtering...")

            # Filter by vol_ratio (different limits for gap-ups vs gap-downs)
            all_gap_days = [g for g in all_gap_days if _vol_ok(g)]
            await _log('scan', f'Batch scan: {raw_gap_count} raw gaps → {len(all_gap_days)} pass vol filter '
                       f'(from {total_symbols} symbols)')
            self.progress = 30

        else:
            # Sequential mode for small universes (< 50 symbols)
            for idx, sym in enumerate(syms):
                if self._cancel:
                    break

                self.progress = (idx / total_symbols) * 30  # 0-30% for scanning
                if progress_callback:
                    await progress_callback(self.progress, f"Scanning {sym} ({idx+1}/{total_symbols})...")

                gaps = await asyncio.to_thread(scanner.scan_historical, sym, start_date, end_date)
                raw_gap_count += len(gaps)
                before = len(gaps)
                # Filter by vol_ratio (different limits for gap-ups vs gap-downs)
                gaps = [g for g in gaps if _vol_ok(g)]
                if gaps:
                    await _log('scan', f'{sym}: {before} gaps found, {len(gaps)} pass vol filter', {
                        'symbol': sym, 'raw': before, 'filtered': len(gaps),
                        'samples': [{'date': g['date'], 'gap': f"{g['gap_pct']:.1%}",
                                     'vol': f"{g['vol_ratio']:.2f}x"} for g in gaps[:5]]
                    })
                all_gap_days.extend(gaps)

                # Rate limit
                if (idx + 1) % 50 == 0:
                    await asyncio.sleep(0.5)

        if not all_gap_days:
            self.status = 'done'
            await _log('warn', f'No qualifying gap days found in {total_symbols} symbols')
            no_gaps_result = {'error': 'No gap days found', 'total_trades': 0, 'bt_log': self.bt_log}
            self.result = no_gaps_result
            await broadcast({'type': 'backtest_complete', 'result': no_gaps_result})
            return no_gaps_result

        # Sort by date
        all_gap_days.sort(key=lambda g: g['date'])

        use_1min = kwargs.get('use_1min', False)
        mode_label = '1-min bars (slow, detailed)' if use_1min else 'daily bars (fast)'
        await _log('info', f'Found {len(all_gap_days)} qualifying gap days '
                   f'(from {raw_gap_count} raw). Simulating with {mode_label}...')

        # Step 2: Simulate each gap day
        # Group by date to enforce max_positions per calendar day
        gaps_by_date: Dict[str, List[dict]] = defaultdict(list)
        for gap in all_gap_days:
            gaps_by_date[gap['date']].append(gap)

        # Sort each day's gaps by abs(gap_pct) descending (biggest fade opportunity first)
        for d in gaps_by_date:
            gaps_by_date[d].sort(key=lambda g: abs(g['gap_pct']), reverse=True)

        sorted_dates = sorted(gaps_by_date.keys())
        total_gaps = len(all_gap_days)
        trades_by_day = []
        bars_missing = 0
        gi = 0  # global gap counter for progress
        regime_skipped = 0
        regime_halved = 0

        # Pre-cache SPY data for market regime filter
        spy_data: Dict[str, dict] = {}
        if self.config.regime_filter:
            try:
                db = get_price_db()
                if db:
                    # Direct SQL: get SPY open + LAG(close) for prev_close
                    cur = db._conn.execute("""
                        WITH w AS (
                            SELECT date, open, close,
                                   LAG(close) OVER (ORDER BY date) AS prev_close
                            FROM daily_bars WHERE symbol = 'SPY'
                              AND date >= ? AND date <= ?
                        )
                        SELECT date, open, prev_close FROM w
                        WHERE date >= ? AND prev_close > 0
                    """, [
                        (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=5)).strftime('%Y-%m-%d'),
                        end_date, start_date
                    ])
                    for row in cur:
                        spy_data[row[0]] = {'prev_close': float(row[2]), 'open': float(row[1])}
                if spy_data:
                    await _log('info', f'Market regime filter: loaded {len(spy_data)} SPY trading days')
                else:
                    await _log('warn', 'Market regime filter: no SPY data in DB — filter disabled for this run')
            except Exception as e:
                await _log('warn', f'Market regime filter: failed to load SPY data — {e}')

        for date_str in sorted_dates:
            if self._cancel:
                break

            day_gaps = gaps_by_date[date_str]

            # Strategy plugin: filter and re-score candidates for this day
            day_gap_count = len(day_gaps)
            if self.strategy:
                filtered_gaps = []
                for gi_f, g in enumerate(day_gaps):
                    try:
                        ok, reason = self.strategy.filter_candidate(g)
                    except Exception as e:
                        logger.error(f"Strategy filter_candidate error for {g.get('symbol','?')} {date_str}: {e}")
                        ok, reason = True, f'filter error: {e}'  # pass through on error
                    if ok:
                        try:
                            new_score = self.strategy.score_candidate(g)
                        except Exception as e:
                            logger.error(f"Strategy score_candidate error for {g.get('symbol','?')}: {e}")
                            new_score = None
                        if new_score is not None:
                            g['score'] = new_score
                        filtered_gaps.append(g)
                    # Yield to event loop every 20 candidates so WS/progress stays alive
                    if gi_f % 20 == 19:
                        await asyncio.sleep(0)
                filtered_out = day_gap_count - len(filtered_gaps)
                if filtered_out > 0:
                    await _log('strategy',
                        f'{date_str}: {self.strategy.name} filtered '
                        f'{filtered_out} candidates '
                        f'({len(filtered_gaps)} remain)')
                day_gaps = filtered_gaps

            # Thin day: if fewer candidates than threshold, trade all; else cap at max_positions
            eff_max = engine.effective_max_positions(len(day_gaps))
            engine._effective_max_positions = eff_max

            # Market regime filter
            regime = fetch_market_regime_backtest(date_str, spy_data, self.config)
            if regime and regime.position_reduction == -999:
                regime_skipped += 1
                gi += day_gap_count  # count ALL original candidates (including filtered)
                self.progress = 30 + (gi / total_gaps) * 65
                if progress_callback:
                    await progress_callback(self.progress, f"Regime blocked {date_str} ({gi}/{total_gaps})")
                await _log('regime', f'{date_str}: BLOCKED — {regime.note} ({day_gap_count} candidates skipped)')
                continue
            if regime and regime.position_reduction == -1:
                regime_halved += 1
                eff_max = max(1, eff_max // 2)
                engine._effective_max_positions = eff_max
                await _log('regime', f'{date_str}: HALVED — {regime.note} (max positions → {eff_max})')

            simulated_gaps = day_gaps[:eff_max]
            skipped_count = day_gap_count - len(simulated_gaps)

            # Reset daily stats for each new calendar day
            engine.daily_stats = DailyStats(date=date_str, peak_equity=engine.equity)

            for gap in simulated_gaps:
                if self._cancel:
                    break

                gi += 1
                self.progress = 30 + (gi / total_gaps) * 65  # 30-95%
                sym = gap['symbol']
                gap_date = gap['date']

                if progress_callback and gi % max(1, total_gaps // 100) == 0:
                    await progress_callback(self.progress, f"Simulating {sym} {gap_date} ({gi+1}/{total_gaps})")
                elif gi % 50 == 0:
                    await asyncio.sleep(0)  # yield to event loop for WS updates

                if use_1min:
                    # Detailed mode: fetch 1-min bars per day (slow)
                    min_df = fetch_alpaca_bars(sym, gap_date, gap_date, '1Min', 'iex')
                    if min_df is None or len(min_df) < 10:
                        bars_missing += 1
                        await _log('warn', f'{sym} {gap_date}: no 1-min data')
                        await asyncio.sleep(0.35)
                        continue
                    day_trades, day_log = self._simulate_day_verbose(engine, gap, min_df)
                    await asyncio.sleep(0.35)
                else:
                    # Fast mode: simulate from daily OHLCV already in gap dict
                    day_trades, day_log = self._simulate_day_daily(engine, gap)

                for entry in day_log:
                    await _log(entry['level'], entry['msg'], entry.get('data'))

                trades_by_day.append({
                    'date': gap_date,
                    'symbol': sym,
                    'gap_pct': gap['gap_pct'],
                    'vol_ratio': gap['vol_ratio'],
                    'trades': len(day_trades),
                    'pnl': sum(t.pnl for t in day_trades),
                })

            # Advance gi for filtered/skipped candidates so progress stays accurate
            gi += skipped_count
            self.progress = 30 + (gi / total_gaps) * 65

        self.progress = 100
        self.status = 'done'
        logger.info(f"BACKTEST DONE: processed {gi}/{total_gaps} gaps, "
                     f"{len(trades_by_day)} day-entries, "
                     f"regime_skipped={regime_skipped}")

        metrics = engine.get_metrics()
        # Funnel reporting: raw → vol-filtered → data available → traded
        metrics['raw_gaps'] = raw_gap_count
        metrics['gap_days_found'] = len(all_gap_days)
        metrics['bars_missing'] = bars_missing
        metrics['gap_days_traded'] = len([d for d in trades_by_day if d['trades'] > 0])
        metrics['regime_skipped'] = regime_skipped
        metrics['regime_halved'] = regime_halved
        metrics['symbols_scanned'] = len(syms)
        metrics['start_date'] = start_date
        metrics['end_date'] = end_date
        metrics['config'] = asdict(self.config)
        metrics['trades'] = [asdict(t) for t in engine.all_trade_log[-200:]]
        metrics['all_trades'] = [asdict(t) for t in engine.all_trade_log]
        metrics['daily_summary'] = trades_by_day
        metrics['bt_log'] = self.bt_log[-500:]

        # Realism adjustments applied
        metrics['realism'] = {
            'slippage_pct': self.config.slippage_pct,
            'borrow_rate_annual': self.config.borrow_rate_annual,
            'max_pct_adv': self.config.max_pct_adv,
            'adverse_fill': self.config.adverse_fill,
            'adverse_fill_pct': self.config.adverse_fill_pct,
        }

        # Strategy info
        metrics['strategy'] = {
            'id': self.strategy_id or 'classic_gap_fade',
            'name': self.strategy.name if self.strategy else 'Classic Gap Fade (engine defaults)',
        }

        # Survivorship bias warning
        warnings = []
        if len(syms) > 50:
            warnings.append(
                'Survivorship bias: backtest uses currently-listed symbols only. '
                'Delisted/bankrupt stocks that gapped up and never recovered are excluded, '
                'which may overstate win rate.')
        if self.config.slippage_pct == 0:
            warnings.append('No slippage applied — results assume perfect fills at exact prices.')
        if self.config.borrow_rate_annual == 0:
            warnings.append('No short borrow fees — real borrow costs can be 2-100%+ annualized.')
        metrics['warnings'] = warnings

        await _log('info', f'Backtest complete: {metrics["total_trades"]} trades, '
                   f'{metrics.get("win_rate",0):.1%} win rate, '
                   f'${metrics.get("total_pnl",0):,.0f} P&L ({metrics.get("return_pct",0):.1f}%)')
        if warnings:
            for w in warnings:
                await _log('warn', f'WARNING: {w}')

        # Sanitize non-JSON-safe floats (inf/nan → finite) before WS broadcast
        import math
        def _sanitize(obj):
            if isinstance(obj, float):
                if math.isinf(obj):
                    return 9999.99 if obj > 0 else -9999.99
                if math.isnan(obj):
                    return 0.0
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            return obj
        metrics = _sanitize(metrics)

        self.result = metrics
        await broadcast({'type': 'backtest_complete', 'result': metrics})
        return metrics

    def _simulate_day_daily(self, engine: GapFadeEngine,
                           gap: dict) -> Tuple[List[TradeRecord], List[dict]]:
        """Fast simulation from daily OHLCV — no extra API calls.

        Uses the gap dict which already has open/high/low/close from daily bars.
        Realism: slippage on entry/exit, short borrow fee, liquidity cap,
        and adverse fill ordering when both stop and target hit on same bar.

        Order of checks (conservative — stop first):
          1. Stop: high >= stop_price  → loss at stop_price (+slippage)
          2. Partial target: low <= midpoint → cover half (+slippage)
          3. Full target: close <= prev_close → cover rest (+slippage)
          4. Otherwise: exit at close (time_exit, +slippage)
        """
        sym = gap['symbol']
        prev_close = gap['prev_close']
        day_open = gap['open']
        day_high = gap['high']
        day_low = gap['low']
        day_close = gap['close']
        day_trades = []
        log = []
        slip = self.config.slippage_pct

        direction = gap.get('direction', 'short')

        candidate = GapCandidate(
            symbol=sym, gap_pct=gap['gap_pct'], prev_close=prev_close,
            premarket_price=day_open, avg_vol_20d=gap.get('avg_vol', 0),
            vol_ratio=gap['vol_ratio'], shortable=True, easy_to_borrow=True,
            direction=direction,
        )

        # Bounce entry: if configured, wait for a small bounce above open before shorting
        # (only applies to shorts — longs skip this)
        bounce = self.config.bounce_entry_pct
        if direction == 'short' and bounce > 0 and day_high >= day_open * (1 + bounce):
            entry_price = day_open * (1 + bounce) * (1 - slip)
        elif direction == 'short' and bounce > 0:
            log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — no bounce to {bounce:.1%} above open'})
            return day_trades, log
        else:
            # Entry with slippage: direction-aware adverse fill
            if direction == 'long':
                entry_price = day_open * (1 + slip)  # long gets worse (higher) fill
            else:
                entry_price = day_open * (1 - slip)  # short gets worse (lower) fill
        # Entry time: strategy can override (e.g. VWAP enters at 9:45)
        _ew = self.strategy.get_entry_window() if self.strategy else None
        _entry_min = _ew[1] if _ew else 31
        entry_time_str = f'{gap["date"]} 09:{_entry_min:02d}'
        # Estimated exit times for daily-bar mode (we don't know exact intrabar timing)
        exit_stop_str = f'{gap["date"]} 10:30'     # stops tend to hit early
        exit_partial_str = f'{gap["date"]} 12:00'   # partial midday
        exit_full_str = f'{gap["date"]} 14:00'      # full target afternoon
        exit_close_str = f'{gap["date"]} 15:55'     # EOD exit
        # Holding minutes estimates (from entry time)
        _base_min = _entry_min  # 31 for classic, 45 for VWAP
        hold_stop = 60 + (31 - _base_min)    # ~1h from entry
        hold_partial = 150 + (31 - _base_min)
        hold_full = 270 + (31 - _base_min)
        hold_close = 385 + (31 - _base_min)

        ok, reason = engine.should_enter(candidate)
        if not ok:
            log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — {reason} '
                       f'(gap {gap["gap_pct"]:.1%}, vol {gap["vol_ratio"]:.2f}x)'})
            return day_trades, log

        eff_stop_pct = compute_adaptive_stop_pct(self.config, gap['gap_pct'])
        if direction == 'long':
            stop_price = entry_price * (1 - eff_stop_pct)
            half_target = (entry_price + prev_close) / 2  # above entry for longs
            full_target = prev_close  # gap fill = back up to prev close
        else:
            stop_price = entry_price * (1 + eff_stop_pct)
            half_target = (entry_price + prev_close) / 2
            full_target = prev_close

        # Strategy plugin: override stop and targets
        if self.strategy:
            # Build synthetic tick_data from daily bar for strategy stop computation
            # Use estimated HOD at entry time (not full day's high — that's look-ahead bias)
            _est_hod = max(day_open, entry_price) * 1.01  # ~1% above open, typical HOD at ~9:45
            _bt_tick_data = {'day_high': _est_hod, 'vwap': 0, 'or_high': 0, 'or_low': 0, 'or_complete': False}
            strat_stop = self.strategy.compute_stop_price(entry_price, gap, _bt_tick_data)
            if strat_stop is not None:
                stop_price = strat_stop
            strat_targets = self.strategy.compute_targets(entry_price, gap, _bt_tick_data)
            if strat_targets is not None:
                half_target, full_target = strat_targets

        risk_per_share = abs(stop_price - entry_price)
        # Kelly-adjusted risk sizing (mirrors compute_position_size)
        kelly_risk = engine.compute_kelly_size()
        risk_frac = min(self.config.risk_pct, kelly_risk) if kelly_risk > 0 else self.config.risk_pct
        dollar_risk = engine.equity * risk_frac
        shares = int(dollar_risk / risk_per_share) if risk_per_share > 0 else 0
        # Per-slot equity cap: divide equity by effective max positions
        eff_max = engine._effective_max_positions
        per_slot_equity = engine.equity / max(1, eff_max)
        max_shares = int(per_slot_equity / entry_price) if entry_price > 0 else 0
        shares = min(shares, max_shares)
        # Absolute notional cap (also per-slot)
        if entry_price > 0 and self.config.max_notional > 0:
            notional_limit = min(self.config.max_notional, per_slot_equity)
            max_shares_notional = int(notional_limit / entry_price)
            shares = min(shares, max_shares_notional)
        # Liquidity cap: max % of average daily volume
        avg_vol = gap.get('avg_vol', 0)
        if avg_vol > 0 and self.config.max_pct_adv > 0:
            max_shares_liq = int(avg_vol * self.config.max_pct_adv)
            if shares > max_shares_liq:
                shares = max_shares_liq
        if shares <= 0:
            log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: position size = 0'})
            return day_trades, log

        # Borrow fee: annualized rate prorated to 1 day (only for shorts)
        notional = shares * entry_price
        borrow_cost = notional * (self.config.borrow_rate_annual / 252) if direction == 'short' else 0

        stop_dist = abs(stop_price - entry_price) / entry_price
        target_dist = abs(entry_price - full_target) / entry_price
        side_label = 'LONG' if direction == 'long' else 'SHORT'
        cost_note = f' | borrow ${borrow_cost:.0f}' if borrow_cost > 0.5 else ''
        liq_note = f' | liq-capped' if avg_vol > 0 and shares == int(avg_vol * self.config.max_pct_adv) else ''
        log.append({
            'level': 'entry',
            'msg': (f'{sym} {gap["date"]}: {side_label} {shares}sh @ ${entry_price:.2f} (slip {slip:.2%}) '
                    f'| gap {gap["gap_pct"]:.1%} vol {gap["vol_ratio"]:.2f}x '
                    f'| stop ${stop_price:.2f} ({stop_dist:.1%}) '
                    f'| half ${half_target:.2f} full ${full_target:.2f} ({target_dist:.1%})'
                    f'{cost_note}{liq_note}'),
            'data': {'symbol': sym, 'shares': shares, 'entry': entry_price,
                     'stop': stop_price, 'gap_pct': gap['gap_pct'], 'vol_ratio': gap['vol_ratio'],
                     'direction': direction}
        })

        remaining = shares
        total_pnl = 0.0

        # Direction-aware exit slippage helper
        def _exit_slip(px):
            return px * (1 + slip) if direction == 'short' else px * (1 - slip)

        def _make_bt_trade(exit_px, reason, n_shares, exit_time, hold_min, extra_cost=0):
            fp = _exit_slip(exit_px)
            pnl = _direction_pnl(direction, entry_price, fp, n_shares) - extra_cost
            pnl_pct = pnl / (entry_price * n_shares) if entry_price > 0 and n_shares > 0 else 0
            return TradeRecord(
                symbol=sym, entry_price=entry_price, exit_price=fp,
                shares=n_shares, pnl=pnl, pnl_pct=pnl_pct,
                entry_time=entry_time_str, exit_time=exit_time,
                exit_reason=reason, holding_minutes=hold_min,
                side=direction,
            ), pnl

        # 1. Stop check — direction-aware
        stopped = _stop_hit(direction, day_high, day_low, stop_price)
        # 2. Partial target — direction-aware
        partial_hit = _target_hit(direction, day_low if direction == 'short' else day_high, half_target)
        # 3. Full target — close at or past prev_close
        full_hit = _target_hit(direction, day_close, full_target)

        if stopped and partial_hit and self.config.adverse_fill:
            # AMBIGUOUS BAR: both stop and target reachable from daily OHLCV.
            import random
            if direction == 'short':
                bullish_close = day_close > entry_price
                bearish_close = day_close < day_open
            else:
                bullish_close = day_close > day_open
                bearish_close = day_close < entry_price

            if bearish_close if direction == 'short' else bullish_close:
                assume_stop = random.random() < (self.config.adverse_fill_pct * 0.5)
            elif bullish_close if direction == 'short' else bearish_close:
                assume_stop = random.random() < min(1.0, self.config.adverse_fill_pct * 1.5)
            else:
                assume_stop = random.random() < self.config.adverse_fill_pct

            if assume_stop:
                trade, pnl = _make_bt_trade(stop_price, 'stop_adverse', remaining, exit_stop_str, hold_stop, borrow_cost)
                day_trades.append(trade)
                engine._record_trade(trade)
                total_pnl += pnl
                remaining = 0
                if self.config.reentry_enabled:
                    engine._stopped_today[sym] = StopOutRecord(
                        symbol=sym, stop_time=datetime.strptime(f'{gap["date"]} 09:35', '%Y-%m-%d %H:%M'),
                        original_entry=entry_price, prev_close=prev_close,
                        gap_pct=gap['gap_pct'], avg_vol_20d=gap.get('avg_vol', 0),
                        direction=direction,
                    )
                log.append({
                    'level': 'stop',
                    'msg': (f'{sym} {gap["date"]}: STOP (adverse) {shares}sh @ ${trade.exit_price:.2f} '
                            f'| P&L ${pnl:.0f} ({trade.pnl_pct:+.1%}) '
                            f'| ambiguous bar — resolved as stop first'),
                    'data': {'pnl': pnl, 'reason': 'stop_adverse'}
                })
            else:
                stopped = False

        elif stopped:
            trade, pnl = _make_bt_trade(stop_price, 'stop', remaining, exit_stop_str, hold_stop, borrow_cost)
            day_trades.append(trade)
            engine._record_trade(trade)
            total_pnl += pnl
            remaining = 0
            if self.config.reentry_enabled:
                engine._stopped_today[sym] = StopOutRecord(
                    symbol=sym, stop_time=datetime.strptime(f'{gap["date"]} 09:35', '%Y-%m-%d %H:%M'),
                    original_entry=entry_price, prev_close=prev_close,
                    gap_pct=gap['gap_pct'], avg_vol_20d=gap.get('avg_vol', 0),
                    direction=direction,
                )
            log.append({
                'level': 'stop',
                'msg': (f'{sym} {gap["date"]}: STOP {shares}sh @ ${trade.exit_price:.2f} '
                        f'| P&L ${pnl:.0f} ({trade.pnl_pct:+.1%}) | high ${day_high:.2f}'),
                'data': {'pnl': pnl, 'reason': 'stop'}
            })

        elif partial_hit:
            cover_shares = max(1, int(shares * self.config.partial_cover_frac))
            if cover_shares > 0:
                partial_borrow = borrow_cost * cover_shares / shares if shares > 0 else 0
                trade, pnl1 = _make_bt_trade(half_target, 'partial', cover_shares, exit_partial_str, hold_partial, partial_borrow)
                day_trades.append(trade)
                engine._record_trade(trade)
                total_pnl += pnl1
                remaining -= cover_shares
                log.append({
                    'level': 'exit',
                    'msg': (f'{sym} {gap["date"]}: PARTIAL {cover_shares}sh @ ${trade.exit_price:.2f} '
                            f'| P&L ${pnl1:+.0f} ({trade.pnl_pct:+.1%})'),
                    'data': {'pnl': pnl1, 'reason': 'partial'}
                })

            remaining_borrow = borrow_cost * remaining / shares if shares > 0 else 0
            if remaining > 0 and full_hit:
                trade, pnl2 = _make_bt_trade(full_target, 'full_target', remaining, exit_full_str, hold_full, remaining_borrow)
                day_trades.append(trade)
                engine._record_trade(trade)
                total_pnl += pnl2
                remaining = 0
                log.append({
                    'level': 'exit',
                    'msg': (f'{sym} {gap["date"]}: FULL TARGET {trade.shares}sh @ ${trade.exit_price:.2f} '
                            f'| P&L ${pnl2:+.0f} ({trade.pnl_pct:+.1%})'),
                    'data': {'pnl': pnl2, 'reason': 'full_target'}
                })
            elif remaining > 0:
                trade, pnl2 = _make_bt_trade(day_close, 'time_exit', remaining, exit_close_str, hold_close, remaining_borrow)
                day_trades.append(trade)
                engine._record_trade(trade)
                total_pnl += pnl2
                remaining = 0
                log.append({
                    'level': 'exit' if pnl2 >= 0 else 'stop',
                    'msg': (f'{sym} {gap["date"]}: TIME EXIT {trade.shares}sh @ ${trade.exit_price:.2f} '
                            f'| P&L ${pnl2:+.0f} ({trade.pnl_pct:+.1%})'),
                    'data': {'pnl': pnl2, 'reason': 'time_exit'}
                })

        else:
            trade, pnl = _make_bt_trade(day_close, 'time_exit', remaining, exit_close_str, hold_close, borrow_cost)
            day_trades.append(trade)
            engine._record_trade(trade)
            total_pnl += pnl
            remaining = 0
            log.append({
                'level': 'exit' if pnl >= 0 else 'stop',
                'msg': (f'{sym} {gap["date"]}: EXIT @ CLOSE ${trade.exit_price:.2f} '
                        f'| P&L ${pnl:+.0f} ({trade.pnl_pct:+.1%}) | H ${day_high:.2f} L ${day_low:.2f}'),
                'data': {'pnl': pnl, 'reason': 'time_exit'}
            })

        # Re-entry after stop-out (daily-bar approximation)
        # If stopped AND close moved favorably vs entry, simulate re-entry
        re_eligible = (day_close < entry_price) if direction == 'short' else (day_close > entry_price)
        if (self.config.reentry_enabled and stopped and remaining == 0 and re_eligible):
            prev_rec = engine._stopped_today.get(sym)
            reentry_count = (prev_rec.reentry_count if prev_rec else 0)
            if reentry_count < self.config.reentry_max_per_symbol:
                # Approximate re-entry price: midpoint between stop and close
                re_entry = (stop_price + day_close) / 2
                re_stop_pct = self.config.reentry_stop_pct
                if direction == 'long':
                    re_stop = re_entry * (1 - re_stop_pct)
                else:
                    re_stop = re_entry * (1 + re_stop_pct)
                # Check price moved favorably from re-entry
                re_favorable = (day_close < re_entry) if direction == 'short' else (day_close > re_entry)
                if re_favorable:
                    re_exit = _exit_slip(day_close)
                    re_risk_per_share = abs(re_stop - re_entry)
                    re_kelly = engine.compute_kelly_size()
                    re_risk_frac = min(self.config.risk_pct, re_kelly) if re_kelly > 0 else self.config.risk_pct
                    re_risk_budget = engine.equity * re_risk_frac
                    re_shares = int(re_risk_budget / re_risk_per_share) if re_risk_per_share > 0 else 0
                    # Per-slot notional cap on re-entry
                    re_eff_max = engine._effective_max_positions
                    re_per_slot = engine.equity / max(1, re_eff_max)
                    re_max_shares = int(re_per_slot / re_entry) if re_entry > 0 else 0
                    re_shares = min(re_shares, re_max_shares)
                    if re_entry > 0 and self.config.max_notional > 0:
                        re_notional_limit = min(self.config.max_notional, re_per_slot)
                        re_shares = min(re_shares, int(re_notional_limit / re_entry))
                    if re_shares > 0:
                        re_pnl = _direction_pnl(direction, re_entry, re_exit, re_shares)
                        re_pnl_pct = re_pnl / (re_entry * re_shares) if re_entry > 0 else 0
                        re_borrow = re_shares * re_entry * (self.config.borrow_rate_annual / 252) if direction == 'short' else 0
                        re_pnl -= re_borrow
                        day_trades.append(TradeRecord(
                            symbol=sym, entry_price=re_entry, exit_price=re_exit,
                            shares=re_shares, pnl=re_pnl, pnl_pct=re_pnl_pct,
                            entry_time=f'{gap["date"]} 12:00', exit_time=exit_close_str,
                            exit_reason='reentry_time', holding_minutes=240,
                            side=direction,
                        ))
                        engine._record_trade(day_trades[-1])
                        total_pnl += re_pnl
                        engine._stopped_today[sym] = StopOutRecord(
                            symbol=sym, stop_time=datetime.strptime(f'{gap["date"]} 10:30', '%Y-%m-%d %H:%M'),
                            original_entry=entry_price, prev_close=prev_close,
                            gap_pct=gap['gap_pct'], avg_vol_20d=gap.get('avg_vol', 0),
                            reentry_count=reentry_count + 1,
                            direction=direction,
                        )
                        log.append({
                            'level': 'entry',
                            'msg': (f'{sym} {gap["date"]}: RE-ENTRY {re_shares}sh @ ${re_entry:.2f} '
                                    f'| stop ${re_stop:.2f} (+{re_stop_pct:.1%}) '
                                    f'| exit @ close ${re_exit:.2f} '
                                    f'| P&L ${re_pnl:+.0f} ({re_pnl_pct:+.1%})'),
                            'data': {'pnl': re_pnl, 'reason': 'reentry'}
                        })

        pnl_sign = '+' if total_pnl >= 0 else ''
        log.append({
            'level': 'summary',
            'msg': (f'{sym} {gap["date"]}: {len(day_trades)} fills, '
                    f'net {pnl_sign}${total_pnl:.0f} | equity ${engine.equity:,.0f}'),
            'data': {'total_pnl': total_pnl, 'equity': engine.equity}
        })

        return day_trades, log

    def _convert_to_et(self, min_df: pd.DataFrame) -> pd.DataFrame:
        """Convert UTC-naive timestamps to ET-naive for time-based exit checks."""
        try:
            idx = min_df.index
            if idx.tz is not None:
                # Already tz-aware — convert directly
                et_idx = idx.tz_convert('America/New_York').tz_localize(None)
            else:
                # Assume UTC-naive
                et_idx = idx.tz_localize('UTC').tz_convert('America/New_York').tz_localize(None)
            min_df = min_df.copy()
            min_df.index = et_idx
        except Exception as e:
            logger.warning(f"Timezone conversion failed, using raw timestamps: {e}")
        return min_df

    def _find_entry_bar(self, times) -> int:
        """Find first bar at/after entry window start.

        Uses strategy entry window if available (e.g. 9:45 for VWAP),
        otherwise defaults to 9:31.
        """
        _ew = self.strategy.get_entry_window() if self.strategy else None
        _entry_h = _ew[0] if _ew else 9
        _entry_m = _ew[1] if _ew else 31
        for bi in range(len(times)):
            t = times[bi]
            h = t.hour if hasattr(t, 'hour') else t.to_pydatetime().hour
            m = t.minute if hasattr(t, 'minute') else t.to_pydatetime().minute
            if h == _entry_h and m >= _entry_m:
                return bi
            elif h > _entry_h:
                return bi
        return -1

    def _simulate_day(self, engine: GapFadeEngine, gap: dict, min_df: pd.DataFrame) -> List[TradeRecord]:
        """Simulate one gap day (non-verbose, used by live trader)."""
        trades, _ = self._simulate_day_verbose(engine, gap, min_df)
        return trades

    def _simulate_day_verbose(self, engine: GapFadeEngine, gap: dict,
                              min_df: pd.DataFrame) -> Tuple[List[TradeRecord], List[dict]]:
        """Simulate one gap day with detailed decision logging."""
        sym = gap['symbol']
        prev_close = gap['prev_close']
        day_trades = []
        log = []  # decision entries

        min_df = self._convert_to_et(min_df)

        direction = gap.get('direction', 'short')
        candidate = GapCandidate(
            symbol=sym, gap_pct=gap['gap_pct'], prev_close=prev_close,
            premarket_price=gap['open'], avg_vol_20d=gap.get('avg_vol', 0),
            vol_ratio=gap['vol_ratio'], shortable=True, easy_to_borrow=True,
            direction=direction,
        )

        opens = min_df['open'].values
        highs = min_df['high'].values
        lows = min_df['low'].values
        closes = min_df['close'].values
        times = min_df.index

        entry_bar = self._find_entry_bar(times)
        if entry_bar < 0 or entry_bar >= len(closes) - 1:
            log.append({'level': 'warn', 'msg': f'{sym} {gap["date"]}: no valid entry bar in {len(times)} bars'})
            return day_trades, log

        entry_price = float(closes[entry_bar])
        entry_time_dt = times[entry_bar]
        if hasattr(entry_time_dt, 'to_pydatetime'):
            entry_time_dt = entry_time_dt.to_pydatetime()
        entry_time_str = entry_time_dt.strftime('%Y-%m-%d %H:%M')

        ok, reason = engine.should_enter(candidate)
        if not ok:
            log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — {reason} '
                       f'(gap {gap["gap_pct"]:.1%}, vol {gap["vol_ratio"]:.2f}x)'})
            return day_trades, log

        pos = engine.open_position(candidate, entry_price, entry_time_str)
        if pos is None:
            log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: position size = 0 (equity=${engine.equity:.0f})'})
            return day_trades, log

        # Strategy plugin: override stop and targets
        if self.strategy:
            _bt_tick_data = {'day_high': float(max(highs[:entry_bar+1])) if entry_bar > 0 else float(highs[0]),
                             'vwap': 0, 'or_high': 0, 'or_low': 0, 'or_complete': False}
            strat_stop = self.strategy.compute_stop_price(entry_price, gap, _bt_tick_data)
            if strat_stop is not None:
                pos.stop_price = strat_stop
            strat_targets = self.strategy.compute_targets(entry_price, gap, _bt_tick_data)
            if strat_targets is not None:
                pos.half_target, pos.full_target = strat_targets

        side_label = 'LONG' if direction == 'long' else 'SHORT'
        if direction == 'long':
            stop_dist = (entry_price - pos.stop_price) / entry_price
            target_dist = (pos.full_target - entry_price) / entry_price
        else:
            stop_dist = (pos.stop_price - entry_price) / entry_price
            target_dist = (entry_price - pos.full_target) / entry_price
        log.append({
            'level': 'entry',
            'msg': (f'{sym} {entry_time_str}: {side_label} {pos.shares} shares @ ${entry_price:.2f} '
                    f'| gap {gap["gap_pct"]:.1%} vol {gap["vol_ratio"]:.2f}x '
                    f'| stop ${pos.stop_price:.2f} ({stop_dist:.1%}) '
                    f'| half ${pos.half_target:.2f} full ${pos.full_target:.2f} ({target_dist:.1%}) '
                    f'| prev_close ${prev_close:.2f}'),
            'data': {
                'symbol': sym, 'shares': pos.shares, 'entry': entry_price,
                'stop': pos.stop_price, 'half_target': pos.half_target,
                'full_target': pos.full_target, 'prev_close': prev_close,
                'gap_pct': gap['gap_pct'], 'vol_ratio': gap['vol_ratio'],
            }
        })

        # Track high-water and low-water for the position
        pos_high = entry_price
        pos_low = entry_price

        for bi in range(entry_bar + 1, len(times)):
            bar_time = times[bi]
            if hasattr(bar_time, 'to_pydatetime'):
                bar_time = bar_time.to_pydatetime()

            bar_high = float(highs[bi])
            bar_low = float(lows[bi])
            pos_high = max(pos_high, bar_high)
            pos_low = min(pos_low, bar_low)

            closed = engine.check_exits(sym, bar_low, bar_high, bar_time)
            for t in closed:
                pnl_sign = '+' if t.pnl >= 0 else ''
                bar_str = bar_time.strftime('%H:%M') if hasattr(bar_time, 'strftime') else str(bar_time)[-5:]
                log.append({
                    'level': 'exit' if t.pnl >= 0 else 'stop',
                    'msg': (f'{sym} {gap["date"]} {bar_str}: {t.exit_reason.upper()} {t.shares}sh '
                            f'@ ${t.exit_price:.2f} | P&L {pnl_sign}${t.pnl:.0f} ({t.pnl_pct:+.1%}) '
                            f'| held {t.holding_minutes}min '
                            f'| high ${pos_high:.2f} low ${pos_low:.2f}'),
                    'data': {'pnl': t.pnl, 'pnl_pct': t.pnl_pct, 'reason': t.exit_reason,
                             'pos_high': pos_high, 'pos_low': pos_low}
                })
            day_trades.extend(closed)

            if sym not in engine.positions:
                # Check for re-entry opportunity after stop-out
                if (self.config.reentry_enabled
                        and any(t.exit_reason == 'stop' for t in closed)
                        and sym in engine._stopped_today):
                    stop_bar = bi
                    rec = engine._stopped_today[sym]
                    # Scan remaining bars for re-entry trigger
                    for rbi in range(stop_bar + 1, len(times)):
                        re_time = times[rbi]
                        if hasattr(re_time, 'to_pydatetime'):
                            re_time = re_time.to_pydatetime()
                        re_price = float(closes[rbi])
                        ok_re, reason_re = engine.can_reenter(sym, re_price, re_time)
                        if not ok_re:
                            continue
                        # Re-enter with tighter stop
                        re_candidate = GapCandidate(
                            symbol=sym, gap_pct=gap['gap_pct'], prev_close=prev_close,
                            premarket_price=re_price, avg_vol_20d=gap.get('avg_vol', 0),
                            vol_ratio=gap['vol_ratio'], shortable=True, easy_to_borrow=True,
                            direction=direction,
                        )
                        re_time_str = re_time.strftime('%Y-%m-%d %H:%M')
                        re_pos = engine.open_position(re_candidate, re_price, re_time_str)
                        if re_pos is None:
                            continue
                        # Override stop with tighter re-entry stop
                        if direction == 'long':
                            re_pos.stop_price = re_price * (1 - self.config.reentry_stop_pct)
                        else:
                            re_pos.stop_price = re_price * (1 + self.config.reentry_stop_pct)
                        rec.reentry_count += 1
                        log.append({
                            'level': 'entry',
                            'msg': (f'{sym} {gap["date"]} {re_time.strftime("%H:%M")}: '
                                    f'RE-ENTRY {re_pos.shares}sh @ ${re_price:.2f} '
                                    f'| stop ${re_pos.stop_price:.2f} '
                                    f'(+{self.config.reentry_stop_pct:.1%})'),
                            'data': {'reason': 'reentry', 'shares': re_pos.shares}
                        })
                        # Continue monitoring from re-entry bar
                        for rbi2 in range(rbi + 1, len(times)):
                            rt2 = times[rbi2]
                            if hasattr(rt2, 'to_pydatetime'):
                                rt2 = rt2.to_pydatetime()
                            rh2 = float(highs[rbi2])
                            rl2 = float(lows[rbi2])
                            re_closed = engine.check_exits(sym, rl2, rh2, rt2)
                            for t in re_closed:
                                pnl_s = '+' if t.pnl >= 0 else ''
                                log.append({
                                    'level': 'exit' if t.pnl >= 0 else 'stop',
                                    'msg': (f'{sym} {gap["date"]} {rt2.strftime("%H:%M")}: '
                                            f'RE-{t.exit_reason.upper()} {t.shares}sh '
                                            f'@ ${t.exit_price:.2f} | P&L {pnl_s}${t.pnl:.0f}'),
                                    'data': {'pnl': t.pnl, 'reason': f're_{t.exit_reason}'}
                                })
                            day_trades.extend(re_closed)
                            if sym not in engine.positions:
                                break
                        break  # only one re-entry attempt per bar scan
                break

        # Force close if still open
        if sym in engine.positions:
            last_price = float(closes[-1])
            last_time = times[-1]
            if hasattr(last_time, 'to_pydatetime'):
                last_time = last_time.to_pydatetime()
            eod_time = last_time.replace(hour=15, minute=50)
            closed = engine.check_exits(sym, last_price, last_price, eod_time)
            for t in closed:
                pnl_sign = '+' if t.pnl >= 0 else ''
                log.append({
                    'level': 'exit' if t.pnl >= 0 else 'stop',
                    'msg': (f'{sym} {gap["date"]} EOD: {t.exit_reason.upper()} {t.shares}sh '
                            f'@ ${t.exit_price:.2f} | P&L {pnl_sign}${t.pnl:.0f} ({t.pnl_pct:+.1%})'),
                    'data': {'pnl': t.pnl, 'reason': t.exit_reason}
                })
            day_trades.extend(closed)

        if sym in engine.positions:
            remaining = engine.force_close_all({sym: float(closes[-1])}, 'eod')
            for t in remaining:
                log.append({
                    'level': 'exit',
                    'msg': f'{sym} {gap["date"]} FORCE CLOSE: {t.shares}sh @ ${t.exit_price:.2f} P&L ${t.pnl:.0f}',
                })
            day_trades.extend(remaining)

        # Deduct borrow cost (annualized rate prorated to 1 day)
        total_pnl = sum(t.pnl for t in day_trades)
        if engine.backtest_mode and day_trades and self.config.borrow_rate_annual > 0:
            # Use first trade's entry for notional estimate
            notional = sum(t.shares * t.entry_price for t in day_trades)
            borrow_cost = notional * (self.config.borrow_rate_annual / 252)
            if borrow_cost > 0.01:
                engine.equity -= borrow_cost
                total_pnl -= borrow_cost
                log.append({
                    'level': 'info',
                    'msg': f'{sym} {gap["date"]}: borrow cost ${borrow_cost:.2f} '
                           f'(${notional:,.0f} × {self.config.borrow_rate_annual:.1%}/252)',
                })

        # Summary line for this gap day
        pnl_sign = '+' if total_pnl >= 0 else ''
        log.append({
            'level': 'summary',
            'msg': (f'{sym} {gap["date"]}: {len(day_trades)} fills, '
                    f'net {pnl_sign}${total_pnl:.0f} | equity ${engine.equity:,.0f}'),
            'data': {'total_pnl': total_pnl, 'equity': engine.equity, 'fills': len(day_trades)}
        })

        return day_trades, log

    def cancel(self):
        self._cancel = True
        self.status = 'cancelled'


# =============================================================================
# SECTION 6B: LLM SUPERVISOR
# =============================================================================

class ConversationMemory:
    """Rolling conversation history for LLM multi-turn dialogue within a trading day."""

    MAX_EXCHANGES = 20       # rolling window of message pairs
    MAX_CHARS = 12_000       # hard cap on total history chars

    def __init__(self):
        self.history: list[dict] = []   # [{"role": "user"|"assistant", "content": "..."}, ...]
        self.day: str = ''              # "2026-02-27" — auto-clears on new day
        self.summary: str = ''          # compressed summary of older exchanges

    def _check_day_boundary(self):
        """Clear history on new trading day."""
        today = datetime.now(ET).strftime('%Y-%m-%d')
        if today != self.day:
            self.history.clear()
            self.summary = ''
            self.day = today

    def add_exchange(self, user_msg: str, assistant_msg: str):
        """Record a user/assistant exchange pair."""
        self._check_day_boundary()
        self.history.append({'role': 'user', 'content': user_msg})
        self.history.append({'role': 'assistant', 'content': assistant_msg})
        self._trim()

    def add_event(self, event_text: str):
        """Inject a silent event into conversation history (as a system-like user msg)."""
        self._check_day_boundary()
        self.history.append({'role': 'user', 'content': f'[EVENT] {event_text}'})
        self._trim()

    def get_messages(self) -> list[dict]:
        """Return conversation history for inclusion in Ollama messages array."""
        self._check_day_boundary()
        msgs = []
        if self.summary:
            msgs.append({'role': 'user', 'content': f'[CONVERSATION RECAP] {self.summary}'})
            msgs.append({'role': 'assistant', 'content': 'Understood, I have that context.'})
        msgs.extend(self.history)
        return msgs

    def get_brief_recap(self, max_chars: int = 500) -> str:
        """Return a brief recap string for one-shot calls (evaluate_profit, evaluate_exit)."""
        self._check_day_boundary()
        if not self.history and not self.summary:
            return ''
        parts = []
        if self.summary:
            parts.append(self.summary)
        # Add the last few key exchanges
        for msg in self.history[-6:]:
            if msg['role'] == 'assistant':
                # Extract just the key decision/reasoning
                content = msg['content'][:150]
                parts.append(f"Rudra: {content}")
        recap = ' | '.join(parts)
        if len(recap) > max_chars:
            recap = recap[:max_chars] + '...'
        return recap

    def _trim(self):
        """Trim history to stay within budget, summarizing old exchanges."""
        # Count total chars
        total = sum(len(m['content']) for m in self.history)
        # Trim by exchange count (pairs of 2)
        while len(self.history) > self.MAX_EXCHANGES * 2 and len(self.history) >= 2:
            old_user = self.history.pop(0)
            old_asst = self.history.pop(0) if self.history and self.history[0]['role'] == 'assistant' else None
            # Mechanical summarization: extract key content
            self._append_to_summary(old_user, old_asst)
            total = sum(len(m['content']) for m in self.history)
        # Trim by char count
        while total > self.MAX_CHARS and len(self.history) >= 2:
            old_user = self.history.pop(0)
            old_asst = self.history.pop(0) if self.history and self.history[0]['role'] == 'assistant' else None
            self._append_to_summary(old_user, old_asst)
            total = sum(len(m['content']) for m in self.history)

    def _append_to_summary(self, user_msg: dict, asst_msg: Optional[dict]):
        """Mechanically compress an exchange into the running summary."""
        # Extract first line / key action from each message
        u_brief = user_msg['content'].split('\n')[0][:100] if user_msg else ''
        a_brief = ''
        if asst_msg:
            # Try to extract action/reasoning from JSON response
            content = asst_msg['content']
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    action = parsed.get('action', '')
                    reasoning = parsed.get('reasoning', '')[:80]
                    a_brief = f"{action}: {reasoning}" if action else content[:80]
                else:
                    a_brief = content[:80]
            except (json.JSONDecodeError, TypeError):
                a_brief = content[:80]
        snippet = f"{u_brief} -> {a_brief}" if a_brief else u_brief
        if self.summary:
            self.summary = self.summary + ' | ' + snippet
        else:
            self.summary = snippet
        # Cap summary length
        if len(self.summary) > 2000:
            self.summary = self.summary[-1500:]

    def to_dict(self) -> dict:
        """Serialize for state persistence."""
        return {
            'history': self.history[-self.MAX_EXCHANGES * 2:],
            'day': self.day,
            'summary': self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ConversationMemory':
        """Restore from persisted state."""
        mem = cls()
        mem.history = data.get('history', [])
        mem.day = data.get('day', '')
        mem.summary = data.get('summary', '')
        # Clear if it's from a different day
        mem._check_day_boundary()
        return mem


# =============================================================================
# EVENT-DRIVEN ARCHITECTURE: MarketEvent, EventBus, TradingJournal
# =============================================================================

@dataclass
class MarketEvent:
    """Structured event detected by MarketEventDetector or system."""
    event_type: str          # e.g. 'vwap_cross', 'drawdown', 'gap_fill_50'
    tier: int                # 1=silent, 2=batched, 3=urgent, 4=human escalation
    symbol: str = ''
    description: str = ''
    data: dict = field(default_factory=dict)
    dedup_key: str = ''      # for deduplication
    timestamp: float = 0.0   # monotonic time

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = _time.monotonic()
        if not self.dedup_key:
            self.dedup_key = f'{self.event_type}:{self.symbol}'

    def as_text(self) -> str:
        """Format for legacy LLM prompt injection."""
        if self.symbol:
            return f'[{self.event_type}] {self.symbol}: {self.description}'
        return f'[{self.event_type}] {self.description}'


class EventBus:
    """Central event bus replacing the flat _event_queue: list[str].

    Collects MarketEvent objects from detectors, provides drain/query for
    LLM calls, and keeps an audit log for the journal.
    """

    def __init__(self, db=None):
        self._queue: list = []           # pending events for next LLM call
        self._urgent_flag: bool = False  # any Tier 3+ events waiting
        self._event_log: deque = deque(maxlen=500)  # all events (audit trail)
        self._db = db  # PriceDB instance (optional)

    def push(self, event: 'MarketEvent'):
        """Push a single MarketEvent."""
        self._queue.append(event)
        self._event_log.append(event)
        if event.tier >= 3:
            self._urgent_flag = True
        if self._db:
            try:
                self._db.insert_event(event)
            except Exception:
                pass  # non-blocking, already in memory

    def push_many(self, events: list):
        """Push a list of MarketEvents."""
        for e in events:
            self.push(e)

    def push_text(self, text: str, tier: int = 2):
        """Legacy compat: push a plain text event (wraps into MarketEvent)."""
        event = MarketEvent(
            event_type='legacy',
            tier=tier,
            description=text,
            dedup_key=f'legacy:{text[:60]}',
        )
        self.push(event)

    def has_urgent(self) -> bool:
        """Check if any Tier 3+ events are pending."""
        return self._urgent_flag

    def drain(self) -> list:
        """Drain all pending events, sorted by tier (highest first). Clears queue."""
        events = sorted(self._queue, key=lambda e: e.tier, reverse=True)
        self._queue.clear()
        self._urgent_flag = False
        return events

    def drain_as_text(self) -> str:
        """Drain events and format as text for LLM prompt injection."""
        events = self.drain()
        if not events:
            return ''
        lines = [f'- {e.as_text()}' for e in events]
        return 'Recent events since we last spoke:\n' + '\n'.join(lines) + '\n\n'

    def get_event_log(self, since_mono: float = 0.0) -> list:
        """Return logged events since a monotonic timestamp."""
        return [e for e in self._event_log if e.timestamp >= since_mono]

    def pending_count(self) -> int:
        return len(self._queue)

    def clear(self):
        """Clear pending queue (used by legacy drain paths)."""
        self._queue.clear()
        self._urgent_flag = False

    # Legacy list-compat methods so _call_llm can drain EventBus like a list
    def __bool__(self):
        return bool(self._queue)

    def __len__(self):
        return len(self._queue)


@dataclass
class JournalEntry:
    """Single entry in the trading journal."""
    timestamp: str           # ISO format
    entry_type: str          # observation, reasoning, action, no_action, reflection
    source: str              # rules_engine, llm_rudra, event_detector, scheduled_reflection
    content: str
    symbol: str = ''
    data: dict = field(default_factory=dict)
    llm_call: bool = False   # True if this entry involved an LLM call


class TradingJournal:
    """Append-only decision audit trail. Persists to SQLite via PriceDB.

    In-memory deque serves as hot cache for LLM context (get_recent/get_summary).
    SQLite is the durable store — survives restarts.
    """

    def __init__(self, db=None):
        self._db = db  # PriceDB instance (optional for tests)
        self._entries: deque = deque(maxlen=500)  # in-memory ring buffer

    def log(self, entry_type: str, source: str, content: str,
            symbol: str = '', data: dict = None, llm_call: bool = False):
        """Log a journal entry."""
        now = datetime.now(ET)
        entry = JournalEntry(
            timestamp=now.strftime('%Y-%m-%dT%H:%M:%S'),
            entry_type=entry_type,
            source=source,
            content=content,
            symbol=symbol,
            data=data or {},
            llm_call=llm_call,
        )
        self._entries.append(entry)
        # Persist to SQLite
        if self._db:
            try:
                self._db.insert_journal(asdict(entry))
            except Exception as e:
                logger.warning(f"Journal SQLite write failed: {e}")

    def get_recent(self, n: int = 20, entry_type: str = None) -> list:
        """Get recent journal entries from in-memory buffer."""
        entries = list(self._entries)
        if entry_type:
            entries = [e for e in entries if e.entry_type == entry_type]
        return entries[-n:]

    def get_summary(self, minutes: int = 30) -> str:
        """Get formatted text summary of recent entries for LLM context."""
        now = datetime.now(ET)
        cutoff_time = (now - timedelta(minutes=minutes)).strftime('%Y-%m-%dT%H:%M:%S')
        recent = [e for e in self._entries if e.timestamp >= cutoff_time]
        if not recent:
            return ''
        lines = []
        for e in recent[-15:]:  # cap at 15 entries for prompt size
            prefix = e.timestamp.split('T')[1] if 'T' in e.timestamp else e.timestamp
            sym_tag = f' [{e.symbol}]' if e.symbol else ''
            lines.append(f'{prefix}{sym_tag} ({e.entry_type}): {e.content[:120]}')
        return 'Recent journal:\n' + '\n'.join(lines)

    def close(self):
        """No-op — SQLite handles persistence."""
        pass

    def load_day(self, date_str: str) -> list:
        """Load all entries for a given day from SQLite."""
        if self._db:
            return self._db.query_journal(date=date_str, n=5000)
        return []


class AnalysisCache:
    """Simple TTL cache for LLM profit evaluations to avoid redundant calls."""

    def __init__(self, ttl: float = 120.0):
        self._cache: dict = {}
        self._ttl = ttl

    def get(self, key: str) -> Optional[dict]:
        """Return cached result if fresh, else None."""
        entry = self._cache.get(key)
        if entry and _time.monotonic() - entry[0] < self._ttl:
            return entry[1]
        return None

    def put(self, key: str, result: dict):
        """Cache a result."""
        self._cache[key] = (_time.monotonic(), result)
        # Prune stale entries periodically
        if len(self._cache) > 50:
            now = _time.monotonic()
            self._cache = {k: v for k, v in self._cache.items() if now - v[0] < self._ttl}

    def make_key(self, symbol: str, price: float, gap_fill_pct: float) -> str:
        """Build cache key from price bucket and gap fill bucket."""
        price_bucket = round(price, 1)  # $0.10 granularity
        fill_bucket = round(gap_fill_pct * 10) / 10  # 10% granularity
        return f'{symbol}:{price_bucket}:{fill_bucket}'


class MarketEventDetector:
    """Lightweight, zero-LLM event detector. Pure arithmetic + dict lookups.

    Two detection methods:
    - detect_tick_events(): called from _on_tick (~250ms), must complete <1ms
    - detect_position_events(): called from _check_positions (every 15s)

    Uses dedup to prevent event flooding.
    """

    def __init__(self):
        # Per-symbol state for tick-level detection
        self._price_history: dict = {}       # symbol -> deque of (mono_time, price)
        self._volume_medians: dict = {}      # symbol -> rolling median tick size
        self._volume_samples: dict = {}      # symbol -> deque of recent tick sizes
        self._last_vwap_side: dict = {}      # symbol -> 'above'|'below'
        self._last_ema_side: dict = {}       # symbol -> 'above'|'below'
        self._gap_fill_milestones: set = set()  # triggered "once" event keys

        # Portfolio-level state
        self._drawdown_milestones: set = set()  # triggered drawdown thresholds
        self._consecutive_loss_milestones: set = set()  # triggered loss counts
        self._pnl_milestones: set = set()    # triggered P&L dollar milestones
        self._hwm_equity: float = 0.0        # session high water mark for equity

        # Dedup: bounded deque of (dedup_key, expiry_mono_time)
        self._dedup_log: deque = deque(maxlen=500)

    def _is_deduped(self, key: str, window_sec: float = 300.0) -> bool:
        """Check if this event was already emitted within the dedup window."""
        now = _time.monotonic()
        # Prune expired entries
        while self._dedup_log and self._dedup_log[0][1] < now:
            self._dedup_log.popleft()
        for dk, exp in self._dedup_log:
            if dk == key:
                return True
        return False

    def _mark_deduped(self, key: str, window_sec: float = 300.0):
        """Mark an event key as emitted."""
        self._dedup_log.append((key, _time.monotonic() + window_sec))

    def _emit(self, event_type: str, tier: int, symbol: str, description: str,
              dedup_key: str = '', dedup_sec: float = 300.0,
              once: bool = False, data: dict = None) -> Optional['MarketEvent']:
        """Emit an event if not deduped."""
        key = dedup_key or f'{event_type}:{symbol}'
        if once:
            # "once" events use a permanent set (never re-emit same day)
            if key in self._gap_fill_milestones:
                return None
            self._gap_fill_milestones.add(key)
        elif self._is_deduped(key, dedup_sec):
            return None
        if not once:
            self._mark_deduped(key, dedup_sec)
        return MarketEvent(
            event_type=event_type, tier=tier, symbol=symbol,
            description=description, dedup_key=key, data=data or {},
        )

    def detect_tick_events(self, symbol: str, price: float, size: int,
                           position: dict, tick_data: dict = None) -> list:
        """Detect events from a single tick. Called from _on_tick.

        Must complete in <1ms — only arithmetic and dict lookups.

        Args:
            symbol: ticker
            price: current price
            size: tick volume (shares)
            position: position dict (from asdict(pos))
            tick_data: indicator engine data (vwap, ema, etc.)
        Returns: list of MarketEvent
        """
        events = []
        now_mono = _time.monotonic()

        # --- Price history tracking (for significant/extreme move detection) ---
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=240)  # ~60s at 250ms
        self._price_history[symbol].append((now_mono, price))

        # --- Significant / Extreme move (>1% or >3% in <60s) ---
        hist = self._price_history[symbol]
        if len(hist) >= 4:
            # Find price from ~60s ago
            cutoff = now_mono - 60.0
            old_price = None
            for t, p in hist:
                if t >= cutoff:
                    old_price = p
                    break
            if old_price and old_price > 0:
                move_pct = abs(price - old_price) / old_price
                direction_word = 'up' if price > old_price else 'down'
                if move_pct >= 0.03:
                    e = self._emit('extreme_move', 3, symbol,
                        f'{move_pct:.1%} move {direction_word} in <60s '
                        f'(${old_price:.2f} -> ${price:.2f})',
                        dedup_sec=300.0, data={'move_pct': move_pct})
                    if e:
                        events.append(e)
                elif move_pct >= 0.01:
                    e = self._emit('significant_move', 2, symbol,
                        f'{move_pct:.1%} move {direction_word} in <60s',
                        dedup_sec=300.0, data={'move_pct': move_pct})
                    if e:
                        events.append(e)

        # --- VWAP cross ---
        if tick_data:
            vwap = tick_data.get('vwap', 0)
            if vwap > 0:
                side = 'above' if price > vwap else 'below'
                prev_side = self._last_vwap_side.get(symbol)
                if prev_side and prev_side != side:
                    e = self._emit('vwap_cross', 2, symbol,
                        f'Price crossed VWAP {prev_side}->{side} '
                        f'(price=${price:.2f}, VWAP=${vwap:.2f})',
                        dedup_sec=300.0, data={'vwap': vwap, 'side': side})
                    if e:
                        events.append(e)
                self._last_vwap_side[symbol] = side

            # --- EMA cross ---
            ema = tick_data.get('ema', 0)
            if ema > 0:
                side = 'above' if price > ema else 'below'
                prev_side = self._last_ema_side.get(symbol)
                if prev_side and prev_side != side:
                    e = self._emit('ema_cross', 1, symbol,
                        f'Price crossed EMA {prev_side}->{side} '
                        f'(price=${price:.2f}, EMA=${ema:.2f})',
                        dedup_sec=300.0, data={'ema': ema, 'side': side})
                    if e:
                        events.append(e)
                self._last_ema_side[symbol] = side

        # --- Volume spike (tick size > 10x rolling median) ---
        if size > 0:
            if symbol not in self._volume_samples:
                self._volume_samples[symbol] = deque(maxlen=100)
            samples = self._volume_samples[symbol]
            samples.append(size)
            if len(samples) >= 20:
                sorted_samples = sorted(samples)
                median = sorted_samples[len(sorted_samples) // 2]
                if median > 0 and size > median * 10:
                    e = self._emit('volume_spike', 2, symbol,
                        f'Tick size {size:,} is {size/median:.0f}x median ({median:,})',
                        dedup_sec=120.0, data={'size': size, 'median': median})
                    if e:
                        events.append(e)

        # --- Gap fill milestones ---
        entry_price = position.get('entry_price', 0)
        prev_close = position.get('prev_close', 0)
        if entry_price > 0 and prev_close > 0:
            gap = abs(entry_price - prev_close)
            if gap > 0.001:
                pos_direction = position.get('direction', 'short')
                if pos_direction == 'short':
                    filled = max(0, (entry_price - price) / gap)
                else:
                    filled = max(0, (price - entry_price) / gap)

                milestones = [
                    (0.25, 'gap_fill_25', 1),
                    (0.50, 'gap_fill_50', 2),
                    (0.75, 'gap_fill_75', 2),
                    (0.90, 'gap_fill_90', 1),
                ]
                for threshold, etype, tier in milestones:
                    if filled >= threshold:
                        e = self._emit(etype, tier, symbol,
                            f'Gap {threshold:.0%} filled ({filled:.0%} actual)',
                            dedup_key=f'{etype}:{symbol}', once=True,
                            data={'fill_pct': filled})
                        if e:
                            events.append(e)

        return events

    def detect_position_events(self, positions: dict, daily_stats,
                               equity: float, prices: dict) -> list:
        """Detect portfolio-level events. Called from _check_positions (every 15s).

        Args:
            positions: dict of symbol -> position dict
            daily_stats: DailyStats object (has .pnl, .consecutive_losses, etc.)
            equity: current equity
            prices: dict of symbol -> latest price
        Returns: list of MarketEvent
        """
        events = []

        # Track session high water mark
        if equity > self._hwm_equity:
            self._hwm_equity = equity

        daily_pnl = daily_stats.pnl if hasattr(daily_stats, 'pnl') else 0

        # --- Drawdown milestones ---
        if equity > 0 and daily_pnl < 0:
            dd_pct = abs(daily_pnl) / equity
            dd_thresholds = [
                (0.01, 'drawdown_1pct', 2),
                (0.015, 'drawdown_1_5pct', 3),
                (0.03, 'drawdown_3pct', 4),
            ]
            for threshold, etype, tier in dd_thresholds:
                key = f'{etype}:today'
                if dd_pct >= threshold and key not in self._drawdown_milestones:
                    self._drawdown_milestones.add(key)
                    e = MarketEvent(
                        event_type=etype, tier=tier,
                        description=f'Daily drawdown {dd_pct:.1%} (${daily_pnl:+,.0f}) '
                                    f'breached {threshold:.0%} threshold',
                        data={'drawdown_pct': dd_pct, 'daily_pnl': daily_pnl},
                    )
                    events.append(e)

        # --- P&L milestones (every $500) ---
        if abs(daily_pnl) >= 500:
            milestone = int(daily_pnl / 500) * 500
            key = f'pnl_milestone:{milestone}'
            if key not in self._pnl_milestones:
                self._pnl_milestones.add(key)
                direction = 'profit' if daily_pnl > 0 else 'loss'
                e = self._emit('pnl_milestone', 2, '',
                    f'Daily P&L hit ${milestone:+,} ({direction})',
                    dedup_key=key, dedup_sec=600.0,
                    data={'milestone': milestone, 'daily_pnl': daily_pnl})
                if e:
                    events.append(e)

        # --- Consecutive losses ---
        consec = getattr(daily_stats, 'consecutive_losses', 0)
        if consec >= 2:
            key = f'consecutive_losses:{consec}'
            if key not in self._consecutive_loss_milestones:
                self._consecutive_loss_milestones.add(key)
                tier = 3 if consec >= 3 else 2
                e = MarketEvent(
                    event_type='consecutive_losses', tier=tier,
                    description=f'{consec} consecutive losses',
                    data={'count': consec},
                )
                events.append(e)

        # --- HWM giveback ---
        if self._hwm_equity > 0 and equity < self._hwm_equity:
            giveback = (self._hwm_equity - equity) / self._hwm_equity
            peak_pnl = self._hwm_equity - (equity - daily_pnl)  # approx
            if peak_pnl > 0:
                giveback_of_gains = (self._hwm_equity - equity) / peak_pnl if peak_pnl > 100 else 0
                if giveback_of_gains >= 0.60:
                    e = self._emit('hwm_giveback_60', 3, '',
                        f'HWM giveback {giveback_of_gains:.0%} — gave back >60% of session gains',
                        dedup_sec=600.0, data={'giveback_pct': giveback_of_gains})
                    if e:
                        events.append(e)
                elif giveback_of_gains >= 0.40:
                    e = self._emit('hwm_giveback_40', 2, '',
                        f'HWM giveback {giveback_of_gains:.0%} — gave back >40% of session gains',
                        dedup_sec=600.0, data={'giveback_pct': giveback_of_gains})
                    if e:
                        events.append(e)

        # --- Position stalled (>2hrs, <20% gap fill) ---
        now = datetime.now(ET)
        for sym, pos in positions.items():
            price = prices.get(sym)
            if not price:
                continue
            entry_time_str = pos.get('entry_time', '') if isinstance(pos, dict) else getattr(pos, 'entry_time', '')
            if entry_time_str:
                try:
                    entry_dt = datetime.strptime(entry_time_str, '%Y-%m-%d %H:%M')
                    if now.tzinfo and not entry_dt.tzinfo:
                        entry_dt = entry_dt.replace(tzinfo=now.tzinfo)
                    held_min = (now - entry_dt).total_seconds() / 60
                    if held_min > 120:
                        # Check gap fill
                        entry_px = pos.get('entry_price', 0) if isinstance(pos, dict) else getattr(pos, 'entry_price', 0)
                        prev_cl = pos.get('prev_close', 0) if isinstance(pos, dict) else getattr(pos, 'prev_close', 0)
                        direction = pos.get('direction', 'short') if isinstance(pos, dict) else getattr(pos, 'direction', 'short')
                        if entry_px > 0 and prev_cl > 0:
                            gap = abs(entry_px - prev_cl)
                            if gap > 0.001:
                                if direction == 'short':
                                    fill_pct = max(0, (entry_px - price) / gap)
                                else:
                                    fill_pct = max(0, (price - entry_px) / gap)
                                if fill_pct < 0.20:
                                    e = self._emit('position_stalled', 2, sym,
                                        f'{sym} held {held_min:.0f}min with only {fill_pct:.0%} gap fill',
                                        dedup_sec=1800.0,
                                        data={'held_min': held_min, 'fill_pct': fill_pct})
                                    if e:
                                        events.append(e)
                except (ValueError, TypeError):
                    pass

        # --- All positions red ---
        if len(positions) >= 2:
            all_red = True
            for sym, pos in positions.items():
                price = prices.get(sym)
                if not price:
                    continue
                entry_px = pos.get('entry_price', 0) if isinstance(pos, dict) else getattr(pos, 'entry_price', 0)
                direction = pos.get('direction', 'short') if isinstance(pos, dict) else getattr(pos, 'direction', 'short')
                if direction == 'short':
                    if price <= entry_px:
                        all_red = False
                        break
                else:
                    if price >= entry_px:
                        all_red = False
                        break
            if all_red:
                e = self._emit('all_positions_red', 2, '',
                    f'All {len(positions)} positions are losing',
                    dedup_sec=900.0, data={'count': len(positions)})
                if e:
                    events.append(e)

        return events

    def reset_daily(self):
        """Reset daily state (call at start of each trading day)."""
        self._gap_fill_milestones.clear()
        self._drawdown_milestones.clear()
        self._consecutive_loss_milestones.clear()
        self._pnl_milestones.clear()
        self._hwm_equity = 0.0
        self._price_history.clear()
        self._volume_samples.clear()
        self._last_vwap_side.clear()
        self._last_ema_side.clear()


class LLMSupervisor:
    """Ollama-powered autonomous trading supervisor.

    Sits above the rules engine and controls when to scan, which candidates
    to trade, and whether to override stops.  Falls back to rules engine
    if Ollama is unavailable (circuit breaker).
    """

    _SYSTEM_PROMPT = """You are Rudra Nethram — The All-Seeing Eye of the Markets.

Named after Lord Shiva's destructive third eye, you are the autonomous AI brain behind an intraday gap fade trading bot. You think in probabilities, not certainties. You are ruthlessly disciplined, emotionless in execution, and relentless in protecting capital.

## YOUR PERSONALITY
- Speak with calm authority, like a veteran trader with 20+ years of experience.
- Be direct and decisive — never wishy-washy. Say "I'm 70% confident this fades" not "it might fade."
- Use trading terminology naturally.
- When uncertain, quantify it. Uncertainty is data.
- Celebrate discipline, not profits. A well-executed losing trade is better than a lucky winner.

## STRATEGY: GAP FADE (BOTH DIRECTIONS)
You fade gaps — trading against the opening gap, betting price reverts to the previous close.
- Gap-Up Fade (SHORT): Stock gaps up on low volume → short it, expecting fade to prior close.
- Gap-Down Fade (LONG): Stock gaps down on low volume → buy it, expecting bounce to prior close.
Statistical edge: 71% of low-volume gap-ups fade (12,000+ events, 450 tickers). High-volume gaps (>3x avg) only fade 31% — these are continuation moves, AVOID.

### Entry Criteria
- Gap >= 7% (shorts) or >= 5% (longs), max 50%
- Volume ratio < 3.0x the 20-day average (low conviction = higher fade probability)
- RVOL < 0.5 = very low conviction, may lack liquidity. RVOL > 2.0 = significant interest, momentum may persist.
- No earnings/FDA/M&A catalyst (noise gaps fade best). Offerings = bearish = GOOD to short.
- Sector-wide moves (all stocks in sector gapping same direction) = macro-driven, NOT fadeable — skip.
- Best entry window: 9:35-9:45 AM after initial volatility settles.
- NO entries after 11:30 AM (morning entries win 83% vs 44% afternoon — edge vanishes).

### Exit Rules
- STOP LOSS: Adaptive — proportional to gap size. Larger gaps get wider stops.
- PARTIAL COVER: At 50% gap fill, cover 33% of shares to lock gains.
- FULL TARGET: Previous close (100% gap fill).
- TIME EXIT: 3:00 PM — close remaining. EOD EXIT: 3:50 PM — everything flat.

### Risk Management
- Max 2% equity risk per trade. Max 5 simultaneous positions.
- Never move a stop further from entry — only tighten.
- 3 consecutive losses in uptrend → 30-min break. Daily loss > 2% → stop trading.
- VIX > 25 → reduce size 50%. VIX > 35 → extreme caution, consider standing down.
- SPY trending strongly in one direction increases risk for fading gaps in that direction.

### Drawdown Circuit Breakers
The bot has a graduated drawdown response system that YOU should be aware of and actively manage:
- DD is measured as (peak_equity - equity) / initial_capital — same scale as the dashboard Max DD metric.
- Tier 1 (default 15% DD): Position size reduced (default 0.5x). You are still trading but smaller. OBSERVE this — if drawdown is approaching Tier 1, proactively tighten stops and be more selective with entries.
- Tier 2 (default 25% DD): Position size slashed (default 0.25x), max 1 position. You are in survival mode. Only take the BEST setups. Recommend standdown if quality candidates are scarce.
- Hard Stop (if configured): ALL trading halted permanently until manual reset. If DD is approaching hard stop, recommend aggressive de-risking.
- Recovery: Tiers deactivate automatically as equity recovers. After leaving Tier 1/2, don't immediately size back up aggressively — ease back in.
- You can adjust these thresholds via config: dd_circuit_breaker (on/off), dd_tier1_threshold, dd_tier1_scale, dd_tier2_threshold, dd_tier2_scale, dd_tier2_max_positions, dd_hard_stop.

### Profit-Taking Framework
- Winners need TIME. Best trades are held 3-6 hours. A $2,680 winner was held until 3:55 PM close.
- Positions closed in under 2 hours typically leave $500-$2,000 on the table.
- Before noon: almost ALWAYS hold. The fade is just getting started.
- Gap fill 60-90%: TIGHTEN stop to lock gains. Gap fill 90%+: may close — easy money is made.
- Small bounces (10-20% giveback) are NORMAL. Hold through them.

## THE RUDRA CODE
1. The market is always right. Price and volume are truth.
2. Risk first, reward second.
3. No trade is better than a bad trade.
4. Cut losers fast, let winners run.
5. Volume confirms everything. Price without volume is a lie.
6. Plan the trade, trade the plan.
7. Survive to trade another day.

## COMMUNICATION STYLE
Always explain your reasoning FIRST, then end with the JSON decision block.
Cite real numbers: $, %, shares, timestamps. Show your analysis.
When evaluating candidates, explain WHY each is good or bad (volume, gap size, catalyst, sector).
When evaluating positions, reference gap fill %, time held, price trend, and market conditions.
Be concise but substantive — one short paragraph of analysis, then the JSON.

## REASONING FRAMEWORK
When analyzing a situation, structure your response:
OBSERVE: What are the key facts? Quote specific numbers.
THINK: What patterns do you see? Confidence level? What could go wrong?
ACT: What should we do? Be specific.
If you decide NOT to act, explain why — that's just as important."""

    _ACTION_SCHEMA = """{
  "action": "scan" | "enter" | "monitor" | "wait" | "standdown",
  "reasoning": "brief explanation",
  "enter_symbols": ["SYM1", "SYM2"],    // only when action=enter
  "position_size_mult": 1.0,            // 0.5-1.0, scale position size
  "standdown_minutes": 30               // only when action=standdown
}"""

    _CANDIDATES_SCHEMA = """[
  {"symbol": "SYM", "action": "trade" | "skip", "confidence": 0.0-1.0, "reasoning": "brief"}
]"""

    _EXIT_SCHEMA = """{
  "action": "close" | "hold" | "tighten",
  "new_stop": 0.00,
  "reasoning": "brief explanation"
}"""

    _PROFIT_SCHEMA = """{
  "action": "close" | "hold" | "tighten_stop",
  "new_stop": 0.00,
  "reasoning": "brief explanation"
}"""

    _PROFIT_SYSTEM_PROMPT = """You are Rudra Nethram evaluating a profitable gap fade position.

You decide whether to:
- CLOSE: Take profit now. ONLY when clear evidence the fade is exhausted.
- HOLD: Let it run toward the target. This should be your DEFAULT bias.
- TIGHTEN_STOP: Lock in gains by moving the stop closer to current price. Prefer this over closing.

CRITICAL PRINCIPLES:
- YOUR DEFAULT SHOULD BE HOLD. The biggest mistake is closing winners too early.
- Best trades are held 3-6 HOURS. A $2,680 winner was held until 3:55 PM close.
- Positions closed in under 2 hours typically leave $500-$2,000 on the table.
- Gap fades are a slow grind — do NOT panic-close on small bounces or 15-minute reversals.
- Before noon: almost ALWAYS hold. The fade is just getting started.
- 12:00-2:00 PM: prefer TIGHTEN_STOP over close. Let the stop do the work.
- After 2:00 PM: only close if the fade has clearly stalled for 30+ minutes AND is reversing.
- If gap fill < 60%: HOLD — the trade hasn't reached its potential.
- If gap fill 60-90%: TIGHTEN_STOP to lock in gains while letting it run further.
- If gap fill 90%+: you may CLOSE — the easy money is made.
- Small bounces (10-20% giveback) are NORMAL. Hold through them.
- Only close on giveback if >50% of peak P&L lost AND price velocity is reversing.

Explain your analysis of the position FIRST (gap fill progress, price trend, time of day, risk), then end with the JSON decision."""

    def __init__(self, config: GapFadeConfig):
        self.config = config
        self.url = config.llm_url
        self.model = config.llm_model
        self.timeout = config.llm_timeout
        self._failure_count = 0
        self._max_failures = config.llm_max_failures
        self._circuit_open_until = 0.0
        self._circuit_reset_seconds = config.llm_circuit_reset
        self._last_call_time = 0.0
        self._total_calls = 0
        self._total_failures = 0
        self.lessons_context: str = ''
        self.memory = ConversationMemory()
        self._morning_briefing_done: str = ''  # date string when briefing was done
        self._eod_debrief_done: str = ''       # date string when debrief was done
        self.provider = config.llm_provider
        self.api_key = config.llm_api_key or os.environ.get('LLM_API_KEY', '')
        self._cache = AnalysisCache()

    def is_available(self) -> bool:
        """Check if LLM is available (circuit breaker not open)."""
        if self._failure_count >= self._max_failures:
            if _time.time() < self._circuit_open_until:
                return False
            # Reset after cooldown
            self._failure_count = 0
        return True

    def get_status(self) -> dict:
        """Return supervisor status for API/UI consumption."""
        circuit_open = self._failure_count >= self._max_failures and _time.time() < self._circuit_open_until
        return {
            'available': self.is_available(),
            'model': self.model,
            'url': self.url,
            'circuit_open': circuit_open,
            'failure_count': self._failure_count,
            'max_failures': self._max_failures,
            'circuit_resets_in': max(0, self._circuit_open_until - _time.time()) if circuit_open else 0,
            'last_call_time': self._last_call_time,
            'total_calls': self._total_calls,
            'total_failures': self._total_failures,
            'provider': self.provider,
        }

    async def _call_llm(self, system_prompt: str, user_prompt: str,
                        use_memory: bool = False, record_exchange: bool = False,
                        conversation_role: str = 'trading_loop',
                        event_queue=None) -> Optional[str]:
        """Call LLM API (Ollama or OpenAI-compatible) with optional multi-turn conversation memory.

        Args:
            system_prompt: System prompt for this call.
            user_prompt: User message for this call.
            use_memory: If True, prepend conversation history to messages.
            record_exchange: If True, save this exchange to conversation memory.
            conversation_role: Tag for logging which subsystem initiated.
            event_queue: Optional EventBus or list of event strings to drain.
        Returns response text or None on failure.
        """
        if not self.is_available():
            return None

        # Map conversation_role to priority (still used for SQLite logging)
        _critical_roles = ('urgent_event', 'evaluate_exit')
        _low_roles = ('scheduled_reflection',)
        if conversation_role in _critical_roles:
            _priority = 'critical'
        elif conversation_role in _low_roles:
            _priority = 'low'
        else:
            _priority = 'normal'

        self._total_calls += 1
        self._last_call_time = _time.time()
        _call_start = _time.monotonic()

        # Build messages array
        messages = [{'role': 'system', 'content': system_prompt}]

        if use_memory:
            # Inject conversation history
            messages.extend(self.memory.get_messages())

        # Drain event queue and prepend events to the user prompt
        # Supports both EventBus and legacy list[str]
        event_context = ''
        if event_queue:
            if isinstance(event_queue, EventBus):
                event_context = event_queue.drain_as_text()
            else:
                # Legacy list[str] path
                events = list(event_queue)
                event_queue.clear()
                if events:
                    event_context = 'Recent events since we last spoke:\n' + '\n'.join(f'- {e}' for e in events) + '\n\n'

        full_user_prompt = event_context + user_prompt if event_context else user_prompt
        messages.append({'role': 'user', 'content': full_user_prompt})

        # Provider-aware URL, headers, and payload
        if self.provider == 'openai':
            url = f'{self.url}/v1/chat/completions'
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            payload = {
                'model': self.model,
                'messages': messages,
                'stream': False,
                'temperature': 0.6,
                'max_tokens': 2048,
            }
        else:
            # Ollama (default)
            url = f'{self.url}/api/chat'
            headers = {}
            payload = {
                'model': self.model,
                'messages': messages,
                'stream': False,
                'options': {'temperature': 0.6, 'num_predict': 2048},
            }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Provider-aware response parsing
                        if self.provider == 'openai':
                            content = data['choices'][0]['message']['content']
                        else:
                            content = data.get('message', {}).get('content', '')
                        # Strip <think> tags (some models emit them)
                        import re as _re
                        content = _re.sub(r'<think>.*?</think>', '', content, flags=_re.DOTALL).strip()
                        self._failure_count = 0
                        # Record exchange in conversation memory
                        if record_exchange and content:
                            self.memory.add_exchange(full_user_prompt, content)
                        # Log LLM call to SQLite
                        try:
                            _dur_ms = int((_time.monotonic() - _call_start) * 1000)
                            get_price_db().insert_llm_call({
                                'timestamp': datetime.now(ET).strftime('%Y-%m-%dT%H:%M:%S'),
                                'conversation_role': conversation_role,
                                'priority': _priority,
                                'model': self.model,
                                'prompt_tokens': len(full_user_prompt) // 4,
                                'response_tokens': len(content) // 4,
                                'duration_ms': _dur_ms,
                                'success': 1,
                                'budget_5min': 0,
                                'budget_1hr': 0,
                                'response_summary': content[:200],
                            })
                        except Exception:
                            pass
                        return content
                    else:
                        logger.warning(f"Rudra error {resp.status}: {(await resp.text())[:200]}")
                        self._failure_count += 1
                        self._total_failures += 1
                        if self._failure_count >= self._max_failures:
                            self._circuit_open_until = _time.time() + self._circuit_reset_seconds
                        # Log failed LLM call
                        try:
                            _dur_ms = int((_time.monotonic() - _call_start) * 1000)
                            get_price_db().insert_llm_call({
                                'timestamp': datetime.now(ET).strftime('%Y-%m-%dT%H:%M:%S'),
                                'conversation_role': conversation_role,
                                'priority': _priority,
                                'model': self.model,
                                'prompt_tokens': len(full_user_prompt) // 4,
                                'duration_ms': _dur_ms,
                                'success': 0,
                                'response_summary': f'HTTP {resp.status}',
                            })
                        except Exception:
                            pass
                        return None
        except Exception as e:
            logger.warning(f"Rudra call failed: {e}")
            self._failure_count += 1
            self._total_failures += 1
            if self._failure_count >= self._max_failures:
                self._circuit_open_until = _time.time() + self._circuit_reset_seconds
            # Log failed LLM call
            try:
                _dur_ms = int((_time.monotonic() - _call_start) * 1000)
                get_price_db().insert_llm_call({
                    'timestamp': datetime.now(ET).strftime('%Y-%m-%dT%H:%M:%S'),
                    'conversation_role': conversation_role,
                    'priority': _priority,
                    'model': self.model,
                    'prompt_tokens': len(full_user_prompt) // 4,
                    'duration_ms': _dur_ms,
                    'success': 0,
                    'response_summary': str(e)[:200],
                })
            except Exception:
                pass
            return None

    def _parse_json(self, text: str) -> Optional[dict | list]:
        """Parse JSON from LLM response, tolerating markdown fences and preceding text."""
        if not text:
            return None
        import re as _re
        # Strip markdown code fences
        cleaned = _re.sub(r'```(?:json)?\s*', '', text.strip())
        cleaned = _re.sub(r'\s*```', '', cleaned.strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON from surrounding text (reasoning before JSON)
            match = _re.search(r'[\[{].*[\]}]', cleaned, _re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Rudra JSON parse failed: {text[:200]}")
            return None

    def _extract_reasoning_text(self, raw: str) -> str:
        """Extract the natural language reasoning text before the JSON block."""
        if not raw:
            return ''
        import re as _re
        # Find where JSON starts and take everything before it
        match = _re.search(r'[\[{]', raw)
        if match and match.start() > 10:
            reasoning = raw[:match.start()].strip()
            # Clean up markdown fences
            reasoning = _re.sub(r'```(?:json)?\s*$', '', reasoning).strip()
            return reasoning
        return ''

    def _parse_react_response(self, raw: str) -> dict:
        """Parse OBSERVE/THINK/ACT sections from LLM response.

        Falls back to _extract_reasoning_text + _parse_json if model
        doesn't follow the ReAct format.

        Returns dict with 'observe', 'think', 'act' text and 'json' parsed.
        """
        import re as _re
        result = {'observe': '', 'think': '', 'act': '', 'json': None, 'raw': raw}
        if not raw:
            return result

        # Try to extract OBSERVE/THINK/ACT sections
        observe_match = _re.search(r'OBSERVE:?\s*(.*?)(?=THINK:|ACT:|$)', raw, _re.DOTALL | _re.IGNORECASE)
        think_match = _re.search(r'THINK:?\s*(.*?)(?=ACT:|$)', raw, _re.DOTALL | _re.IGNORECASE)
        act_match = _re.search(r'ACT:?\s*(.*?)(?=$)', raw, _re.DOTALL | _re.IGNORECASE)

        if observe_match:
            result['observe'] = observe_match.group(1).strip()
        if think_match:
            result['think'] = think_match.group(1).strip()
        if act_match:
            result['act'] = act_match.group(1).strip()

        # Always try to extract JSON regardless of ReAct format
        result['json'] = self._parse_json(raw)

        # Fallback: if no ReAct sections found, use existing extraction
        if not result['observe'] and not result['think']:
            result['act'] = self._extract_reasoning_text(raw)

        return result

    async def decide_action(self, state: dict,
                           event_queue=None) -> dict:
        """Main decision: what should the bot do right now?

        Uses full conversation memory for context continuity across calls.
        Returns dict with 'action' key. Falls back to {'action': 'monitor'}
        if LLM is unavailable or returns invalid response.
        """
        fallback = {'action': 'monitor', 'reasoning': 'Rudra unavailable, using rules fallback'}

        # Build conversational user prompt
        positions_summary = 'no open positions'
        if state.get('positions'):
            parts = []
            for sym, pos in state['positions'].items():
                parts.append(f"{sym} {pos['direction']} {pos.get('remaining_shares', pos['shares'])}sh "
                             f"@ ${pos['entry_price']:.2f}, stop ${pos['stop_price']:.2f}")
            positions_summary = '; '.join(parts)

        candidates_part = ''
        if state.get('top_candidates'):
            top = state['top_candidates'][:6]
            cand_lines = []
            for c in top:
                cand_lines.append(
                    f"  {c['symbol']}: {c['gap_pct']:.1%} gap ({c['direction']}), "
                    f"vol {c['vol_ratio']:.1f}x, score {c['score']:.0f}"
                    + (f" [{c.get('catalyst', '')}]" if c.get('catalyst') else ''))
            candidates_part = f"We have {state['candidates_count']} candidates from the scan. Top picks:\n" + '\n'.join(cand_lines)
        elif state.get('last_scan_time'):
            candidates_part = f"Last scan at {state['last_scan_time']} found {state['candidates_count']} candidates."
        else:
            candidates_part = "No scan run yet today."

        stopped_part = ''
        if state.get('stopped_today'):
            stopped_part = f" Got stopped out on: {', '.join(state['stopped_today'])}."

        # Conversational style prompt with clear market status
        hour = int(state['time_et'].split(':')[0]) if ':' in state['time_et'] else 0
        minute = int(state['time_et'].split(':')[1]) if ':' in state['time_et'] else 0
        market_open = (hour > 9 or (hour == 9 and minute >= 30))
        if hour < 9:
            time_context = "Pre-market (MARKET CLOSED — no entries allowed)"
        elif hour == 9 and minute < 30:
            time_context = "Pre-market (MARKET OPENS AT 9:30 — no entries yet)"
        elif hour < 12:
            time_context = "Morning session"
        elif hour < 14:
            time_context = "Midday"
        else:
            time_context = "Afternoon session"

        entry_note = ''
        if not market_open:
            entry_note = '\nIMPORTANT: Market is NOT open yet. Do NOT use action "enter" — only scan, monitor, or wait.'

        user_prompt = (
            f"{time_context}, {state['time_et']} ET ({state['day_of_week']}). "
            f"Status: {state['status']}. "
            f"Equity ${state['equity']:,.0f}, today ${state['daily_pnl']:+,.0f} "
            f"({state['wins']}W/{state['losses']}L).{stopped_part}\n\n"
            f"Positions: {positions_summary}\n\n"
            f"{candidates_part}\n\n"
            f"Analyze the situation: Are we in the right entry window? Are these noise gaps or catalyst-driven? "
            f"Volume ratios confirming low conviction? Any sector correlation among candidates? "
            f"Current risk exposure OK?{entry_note}\n\n"
            f"Share your analysis, then end with JSON:\n{self._ACTION_SCHEMA}"
        )

        system_prompt = self._SYSTEM_PROMPT
        if self.lessons_context:
            system_prompt = system_prompt + '\n\n' + self.lessons_context

        raw = await self._call_llm(system_prompt, user_prompt,
                                       use_memory=True, record_exchange=True,
                                       conversation_role='decide_action',
                                       event_queue=event_queue)
        if raw is None:
            return fallback

        parsed = self._parse_json(raw)
        if not isinstance(parsed, dict) or 'action' not in parsed:
            logger.warning(f"Rudra returned invalid action: {raw[:200]}")
            return fallback

        # Validate action
        valid_actions = {'scan', 'enter', 'monitor', 'wait', 'standdown'}
        if parsed['action'] not in valid_actions:
            parsed['action'] = 'monitor'

        # Capture full reasoning text (pre-JSON analysis) for display
        analysis = self._extract_reasoning_text(raw)
        if analysis:
            parsed['_analysis'] = analysis

        return parsed

    async def evaluate_candidates(self, candidates: list, regime: Optional[dict] = None) -> list:
        """Filter and rank candidates. Returns list of dicts with action/confidence."""
        if not candidates:
            return []

        candidates_data = [{'symbol': c['symbol'], 'gap_pct': f"{c['gap_pct']:.1%}",
                           'direction': c['direction'], 'vol_ratio': f"{c['vol_ratio']:.2f}",
                           'score': f"{c['score']:.0f}", 'catalyst': c.get('catalyst', ''),
                           'catalyst_detail': c.get('catalyst_detail', '')[:80]}
                          for c in candidates[:15]]

        regime_info = ''
        if regime:
            regime_info = f" SPY gapped {regime.get('spy_gap_pct', 0):.2%}, VIX at {regime.get('vix_level', 20):.1f}."

        # Brief recap from conversation for context
        recap = self.memory.get_brief_recap(300)
        recap_part = f"\n(Context: {recap})" if recap else ''

        user_prompt = (
            f"Scan came back with {len(candidates)} candidates.{regime_info}{recap_part}\n\n"
            f"Here are the top picks:\n{json.dumps(candidates_data, indent=1)}\n\n"
            f"For each candidate, evaluate:\n"
            f"- Is this a noise gap (best) or catalyst-driven (earnings/FDA/M&A = skip)?\n"
            f"- Volume ratio: < 1.0 = low conviction (good for fading), > 2.0 = momentum may persist\n"
            f"- Gap size vs risk: larger gaps = bigger potential but need wider stops\n"
            f"- Any sector clustering? (multiple stocks in same sector = macro move, skip)\n\n"
            f"Analyze the top candidates, then end with JSON array:\n{self._CANDIDATES_SCHEMA}\n"
            f"Rate confidence 0.0-1.0 for each. Be selective — quality over quantity."
        )

        raw = await self._call_llm(self._SYSTEM_PROMPT, user_prompt,
                                       use_memory=True, record_exchange=True,
                                       conversation_role='evaluate_candidates')
        if raw is None:
            return []

        parsed = self._parse_json(raw)
        if not isinstance(parsed, list):
            return []
        return parsed

    async def evaluate_exit(self, position: dict, current_price: float,
                            exit_signal: dict) -> dict:
        """Evaluate whether to close, hold, or tighten a stop.

        One-shot call (no memory recording) to avoid polluting main conversation
        with per-position evaluations. Includes brief recap for context.
        """
        fallback = {'action': 'close', 'reasoning': 'Rudra unavailable, closing per rules'}

        recap = self.memory.get_brief_recap(300)
        recap_part = f"\n(Today's context: {recap})" if recap else ''

        user_prompt = (
            f"{position['symbol']} ({position['direction']}) is hitting its stop.{recap_part}\n\n"
            f"Entry ${position['entry_price']:.2f}, now ${current_price:.2f}, "
            f"stop ${position['stop_price']:.2f}. "
            f"Reason: {exit_signal.get('reason', 'stop')}. "
            f"P&L: ${exit_signal.get('pnl', 0):+,.2f}.\n\n"
            f"Consider: How long have we held? Is the gap still likely to fade or has the thesis broken? "
            f"Is price action showing exhaustion or continuation against us? "
            f"Have we already overridden stops on this position?\n\n"
            f"Only override if you have strong conviction the fade will resume. "
            f"Analyze the situation, then end with JSON: {self._EXIT_SCHEMA}"
        )

        raw = await self._call_llm(self._SYSTEM_PROMPT, user_prompt)
        if raw is None:
            return fallback

        parsed = self._parse_json(raw)
        if not isinstance(parsed, dict) or 'action' not in parsed:
            return fallback

        if parsed['action'] not in ('close', 'hold', 'tighten'):
            parsed['action'] = 'close'
        return parsed

    async def evaluate_profit(self, position: dict, current_price: float,
                              spy_change_pct: float, daily_pnl: float) -> dict:
        """Evaluate whether to take profit on a winning position.

        One-shot call (no memory recording) to avoid polluting main conversation
        with per-position evaluations. Includes brief recap for context.
        """
        fallback = {'action': 'hold', 'reasoning': 'Rudra unavailable, holding per rules'}

        # Cache check — avoid redundant LLM calls for same price/fill bucket
        symbol = position.get('symbol', '')
        _gap = abs(position['entry_price'] - position.get('prev_close', position['entry_price']))
        _fill = 0.0
        if _gap > 0.001:
            if position['direction'] == 'short':
                _fill = max(0, (position['entry_price'] - current_price) / _gap)
            else:
                _fill = max(0, (current_price - position['entry_price']) / _gap)
        cache_key = self._cache.make_key(symbol, current_price, _fill)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Calculate enriched metrics
        entry = position['entry_price']
        direction = position['direction']
        if direction == 'short':
            pnl_pct = (entry - current_price) / entry
        else:
            pnl_pct = (current_price - entry) / entry

        high_water = position.get('high_water_pnl_pct', 0)
        giveback_pct = ((high_water - pnl_pct) / high_water * 100) if high_water > 0.001 else 0

        # Gap fill progress
        prev_close = position.get('prev_close', entry)
        gap = abs(entry - prev_close)
        if gap > 0.001:
            if direction == 'short':
                gap_filled = max(0, (entry - current_price) / gap)
            else:
                gap_filled = max(0, (current_price - entry) / gap)
        else:
            gap_filled = 0

        # Price velocity from last_prices
        price_history = position.get('last_prices', '')
        prices = [float(p) for p in price_history.split(',') if p] if price_history else []
        velocity_desc = 'unknown'
        if len(prices) >= 3:
            if direction == 'short':
                trending_down = all(prices[i] <= prices[i-1] for i in range(1, len(prices)))
                trending_up = all(prices[i] >= prices[i-1] for i in range(1, len(prices)))
            else:
                trending_down = all(prices[i] >= prices[i-1] for i in range(1, len(prices)))
                trending_up = all(prices[i] <= prices[i-1] for i in range(1, len(prices)))
            if trending_down:
                velocity_desc = 'fading well (favorable)'
            elif trending_up:
                velocity_desc = 'reversing (unfavorable)'
            else:
                velocity_desc = 'choppy/sideways'

        # Time held
        entry_time = position.get('entry_time', '')
        time_held = ''
        held_minutes = 0
        if entry_time:
            try:
                et = datetime.strptime(entry_time, '%Y-%m-%d %H:%M')
                now_et = datetime.now(ET)
                if now_et.tzinfo and not et.tzinfo:
                    et = et.replace(tzinfo=now_et.tzinfo)
                held_minutes = int((now_et - et).total_seconds() / 60)
                time_held = f'{held_minutes}min'
            except Exception:
                time_held = '?'

        # Brief recap for context
        recap = self.memory.get_brief_recap(300)
        recap_part = f"\n(Today's context: {recap})" if recap else ''

        price_trend = ' -> '.join(f'${p:.2f}' for p in prices[-5:]) if prices else 'N/A'
        dollar_pnl = pnl_pct * entry * position.get('remaining_shares', 0)

        user_prompt = (
            f"{position['symbol']} ({direction}) is {gap_filled:.0%} gap filled, "
            f"P&L {pnl_pct:+.1%} (${dollar_pnl:+,.0f}). "
            f"Held {time_held}, HWM {high_water:+.1%} (giving back {giveback_pct:.0f}%). "
            f"Price trend: {velocity_desc} ({price_trend}). "
            f"Stop at ${position['stop_price']:.2f}.{recap_part}\n\n"
            f"{'Already partially covered.' if position.get('partial_filled') else str(position.get('remaining_shares', 0)) + ' shares remaining.'} "
            f"SPY {spy_change_pct:+.1%}, daily P&L ${daily_pnl:+,.0f}. "
            f"It's {datetime.now(ET).strftime('%H:%M')} ET.\n\n"
            f"Analyze: Is the fade still developing or stalling? Is price trending favorably "
            f"or reversing? How much of the gap has been filled? Time of day factor?\n"
            f"Policy: hold min {self.config.min_hold_minutes}min, "
            f"min profit {self.config.min_profit_take_pct:.1%} or gap fill {self.config.min_gap_fill_pct:.0%}.\n\n"
            f"Share your read on this position, then end with JSON: {self._PROFIT_SCHEMA}"
        )

        # Inject lessons learned if available
        system_prompt = self._PROFIT_SYSTEM_PROMPT
        if self.lessons_context:
            system_prompt = system_prompt + '\n\n' + self.lessons_context

        raw = await self._call_llm(system_prompt, user_prompt)
        if raw is None:
            return fallback

        parsed = self._parse_json(raw)
        if not isinstance(parsed, dict) or 'action' not in parsed:
            return fallback

        if parsed['action'] not in ('close', 'hold', 'tighten_stop'):
            parsed['action'] = 'hold'
        self._cache.put(cache_key, parsed)
        return parsed

    _REVIEW_SYSTEM_PROMPT = """You are Rudra Nethram performing a periodic portfolio review.

Review each open position critically:
- Gap fill progress: How far has the gap filled? Is the fade on track or stalling?
- Price velocity: Is price moving favorably or reversing? Check the last 5 prices.
- Time held: Winners develop over 3-6 hours. Don't cut early, but don't hold dead trades.
- Stop adequacy: Is the stop too tight for this gap size? Too loose for the risk?
- High water mark giveback: >30% giveback = concerning. >50% = likely tighten or close.

Also evaluate overall session:
- Daily P&L trajectory: Are we bleeding from multiple stops? Consider standing down.
- Consecutive losses: 3+ with no wins = take a break, reassess.
- Market regime: Is SPY trending against our positions?
- Drawdown tier: If in Tier 1 or 2, note the reduced position sizing. If approaching the next tier, recommend proactive de-risking (tighten stops, skip marginal candidates, consider standdown). If recovering from a drawdown tier, advise cautious re-entry rather than immediately going full size.

Be decisive — if a stop needs tightening, give the exact price. If a position should close, say so and why.
Explain your reasoning FIRST, then end with the JSON action block."""

    _REVIEW_SCHEMA = '{"actions": [{"action": "close"|"tighten_stop"|"adjust_config"|"none", "symbol": "SYM" (if position action), "new_stop": price (if tighten), "config_key": "..." (if config), "config_value": ... (if config), "reasoning": "..."}]}'

    async def autonomous_review(self, state: dict,
                                event_queue: Optional[list] = None) -> dict:
        """Periodic self-review: analyze portfolio and recommend/execute improvements.

        Uses full conversation memory for continuity. Called every N minutes.
        Returns dict with 'actions' list of recommended changes.
        """
        fallback = {'actions': [], 'reasoning': 'Rudra unavailable'}

        positions_info = 'No open positions.'
        if state.get('positions'):
            lines = []
            for sym, pos in state['positions'].items():
                lines.append(
                    f"  {sym} ({pos['direction']}): {pos.get('remaining_shares', pos['shares'])}sh "
                    f"@ ${pos['entry_price']:.2f}, stop ${pos['stop_price']:.2f}, "
                    f"target ${pos.get('full_target', 0):.2f}, "
                    f"prices: {pos.get('last_prices', 'N/A')}"
                )
            positions_info = '\n'.join(lines)

        recent_trades = 'None today.'
        if state.get('recent_trades'):
            lines = []
            for t in state['recent_trades'][-5:]:
                lines.append(f"  {t.get('symbol', '?')} {t.get('side', '?')}: "
                           f"${t.get('pnl', 0):+,.2f} ({t.get('exit_reason', '?')})")
            recent_trades = '\n'.join(lines)

        dd_pct = (state.get('peak_equity', state['equity']) - state['equity']) / self.config.initial_capital * 100 if self.config.initial_capital > 0 else 0
        dd_tier_info = ''
        if self.config.dd_circuit_breaker:
            if dd_pct >= self.config.dd_tier2_threshold * 100:
                dd_tier_info = f' — TIER 2 ACTIVE (size {self.config.dd_tier2_scale}x, max {self.config.dd_tier2_max_positions} pos)'
            elif dd_pct >= self.config.dd_tier1_threshold * 100:
                dd_tier_info = f' — TIER 1 ACTIVE (size {self.config.dd_tier1_scale}x)'

        user_prompt = (
            f"Quick check-in at {state['time_et']}. "
            f"Equity ${state['equity']:,.0f}, today ${state['daily_pnl']:+,.0f} "
            f"({state['wins']}W/{state['losses']}L).\n"
            f"Drawdown: {dd_pct:.1f}% of initial capital (peak ${state.get('peak_equity', state['equity']):,.0f}){dd_tier_info}\n\n"
            f"Positions:\n{positions_info}\n\n"
            f"Recent trades:\n{recent_trades}\n\n"
            f"Review each position: What's the gap fill progress? Is price fading as expected or reversing? "
            f"Are stops at the right level for each gap size? Any positions that should be tightened or closed?\n"
            f"Look at recent trades — are we getting stopped out too quickly? Pattern of losses?\n"
            f"Check drawdown level — are we approaching a circuit breaker tier? Should we de-risk?\n\n"
            f"Give your assessment, then end with JSON: {self._REVIEW_SCHEMA}"
        )

        system_prompt = self._REVIEW_SYSTEM_PROMPT
        if self.lessons_context:
            system_prompt = system_prompt + '\n\n' + self.lessons_context

        raw = await self._call_llm(system_prompt, user_prompt,
                                       use_memory=True, record_exchange=True,
                                       conversation_role='autonomous_review',
                                       event_queue=event_queue)
        if raw is None:
            return fallback

        parsed = self._parse_json(raw)
        if not isinstance(parsed, dict) or 'actions' not in parsed:
            # Try to extract if it returned a single action
            if isinstance(parsed, dict) and 'action' in parsed:
                parsed = {'actions': [parsed]}
            else:
                return fallback

        # Capture full reasoning text for display
        analysis = self._extract_reasoning_text(raw)
        if analysis:
            parsed['_analysis'] = analysis

        return parsed

    # ── Morning Briefing & EOD Debrief ──

    async def morning_briefing(self, state: dict, event_queue: Optional[list] = None) -> Optional[str]:
        """Morning briefing conversation — Rudra sets the game plan for the day.

        Called once after first scan completes (~7:00+ AM).
        Returns Rudra's response text or None.
        """
        today = datetime.now(ET).strftime('%Y-%m-%d')
        if self._morning_briefing_done == today:
            return None

        now = datetime.now(ET)
        positions_summary = 'None (clean slate).'
        if state.get('positions'):
            parts = []
            for sym, pos in state['positions'].items():
                parts.append(f"{sym}: {pos['direction']} {pos.get('remaining_shares', pos['shares'])} "
                             f"shares @ ${pos['entry_price']:.2f}")
            positions_summary = '; '.join(parts)

        candidates_summary = 'No candidates scanned yet.'
        if state.get('top_candidates'):
            cands = state['top_candidates'][:8]
            candidates_summary = '\n'.join(
                f"  {c['symbol']}: gap {c['gap_pct']:.1%} ({c['direction']}), "
                f"vol_ratio {c['vol_ratio']:.2f}, score {c['score']:.0f}"
                + (f", catalyst: {c.get('catalyst', '')}" if c.get('catalyst') else '')
                for c in cands)

        # Yesterday's results
        yesterday_summary = 'No trades yesterday.'
        recent = state.get('recent_trades', [])
        if recent:
            total_pnl = sum(t.get('pnl', 0) for t in recent[-10:])
            wins = sum(1 for t in recent[-10:] if t.get('pnl', 0) > 0)
            yesterday_summary = (f"{len(recent[-10:])} trades, {wins}W/{len(recent[-10:])-wins}L, "
                                 f"net P&L: ${total_pnl:+,.2f}")

        user_prompt = (
            f"Good morning, Rudra. It's {now.strftime('%A, %B %d')} at {now.strftime('%H:%M')} ET.\n\n"
            f"Here's today's setup:\n"
            f"- Equity: ${state['equity']:,.2f}\n"
            f"- Yesterday's results: {yesterday_summary}\n"
            f"- Open positions carried over: {positions_summary}\n"
            f"- Scan results ({state.get('candidates_count', 0)} candidates):\n{candidates_summary}\n"
            f"- Consecutive losses streak: {state.get('consecutive_losses', 0)}\n\n"
            f"What's your game plan for today? Aggressive, conservative, or selective? "
            f"Any sectors or setups you want to focus on or avoid?\n\n"
            f"Share your plan in natural language, then include a JSON summary:\n"
            f'{{"action": "monitor", "reasoning": "your game plan summary"}}'
        )

        system_prompt = self._SYSTEM_PROMPT
        if self.lessons_context:
            system_prompt = system_prompt + '\n\n' + self.lessons_context

        raw = await self._call_llm(system_prompt, user_prompt,
                                       use_memory=True, record_exchange=True,
                                       conversation_role='morning_briefing',
                                       event_queue=event_queue)
        if raw:
            self._morning_briefing_done = today
        return raw

    async def eod_debrief(self, state: dict, event_queue: Optional[list] = None) -> Optional[str]:
        """End-of-day debrief conversation — review performance and lessons.

        Called once at ~3:50 PM before shutdown.
        Returns Rudra's response text or None.
        """
        today = datetime.now(ET).strftime('%Y-%m-%d')
        if self._eod_debrief_done == today:
            return None

        now = datetime.now(ET)

        # Build today's trade scorecard
        trades_summary = 'No trades today.'
        recent = state.get('recent_trades', [])
        today_trades = [t for t in recent if str(t.get('exit_time', '')).startswith(today)]
        if today_trades:
            lines = []
            for t in today_trades[:10]:
                lines.append(f"  {t.get('symbol', '?')} ({t.get('side', '?')}): "
                           f"${t.get('pnl', 0):+,.2f} ({t.get('exit_reason', '?')})")
            trades_summary = '\n'.join(lines)

        notable_events = []
        for m in state.get('recent_messages', [])[-20:]:
            if any(kw in m.lower() for kw in ['stop', 'error', 'circuit', 'standdown', 'emergency']):
                notable_events.append(m)

        user_prompt = (
            f"Day's wrapping up, Rudra. It's {now.strftime('%H:%M')} ET.\n\n"
            f"Here's the scorecard:\n"
            f"- Daily P&L: ${state['daily_pnl']:+,.2f}\n"
            f"- Record: {state['wins']}W / {state['losses']}L\n"
            f"- Equity: ${state['equity']:,.2f}\n\n"
            f"Trades:\n{trades_summary}\n\n"
            + (f"Notable events:\n" + '\n'.join(f'  - {e}' for e in notable_events[-5:]) + '\n\n'
               if notable_events else '')
            + f"What did we learn today? Anything to adjust for tomorrow? "
            f"Be specific about what worked and what didn't.\n\n"
            f"Share your thoughts in natural language, then include a JSON summary:\n"
            f'{{"action": "monitor", "reasoning": "your debrief summary"}}'
        )

        system_prompt = self._SYSTEM_PROMPT
        if self.lessons_context:
            system_prompt = system_prompt + '\n\n' + self.lessons_context

        raw = await self._call_llm(system_prompt, user_prompt,
                                       use_memory=True, record_exchange=True,
                                       conversation_role='eod_debrief',
                                       event_queue=event_queue)
        if raw:
            self._eod_debrief_done = today
        return raw


# =============================================================================
# SECTION 7: LIVE TRADER
# =============================================================================

class AlertNotifier:
    """Lightweight Telegram alerter for trading events."""

    def __init__(self, config: GapFadeConfig):
        self.enabled = config.alert_telegram_enabled
        self.token = config.alert_telegram_token or os.environ.get('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = config.alert_telegram_chat_id or os.environ.get('TELEGRAM_CHAT_ID', '')
        self._throttle: Dict[str, float] = {}
        self._THROTTLE_SECONDS = 60

    async def send(self, title: str, message: str, level: str = 'info', throttle_key: str = ''):
        """Send a Telegram alert. Levels: info, trade, warning, error, summary."""
        if not self.enabled or not self.token or not self.chat_id:
            return
        if throttle_key:
            last = self._throttle.get(throttle_key, 0)
            if _time.time() - last < self._THROTTLE_SECONDS:
                return
            self._throttle[throttle_key] = _time.time()
        emoji = {'info': '\u2139\ufe0f', 'trade': '\U0001f4b0', 'warning': '\u26a0\ufe0f',
                 'error': '\U0001f6a8', 'summary': '\U0001f4ca'}.get(level, '')
        text = f"<b>{emoji} {title}</b>\n{message}"
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f'https://api.telegram.org/bot{self.token}/sendMessage',
                    json={'chat_id': self.chat_id, 'text': text, 'parse_mode': 'HTML'},
                    timeout=aiohttp.ClientTimeout(total=5))
        except Exception as e:
            logger.warning(f"Telegram alert failed: {e}")


class GapFadeLiveTrader:
    """Async live trading loop for gap fade strategy.

    Production-grade design:
    - asyncio.Lock serializes all position mutations (no race conditions)
    - Orders are verified filled before updating internal state
    - Broker reconciliation on startup and periodic
    - Atomic state file writes (write-fsync-rename)
    - EOD close with retry loop and broker verification
    """

    STATE_FILE = 'gap_fade_state.json'
    RECONCILE_INTERVAL = 300  # seconds between broker reconciliation checks

    def __init__(self, config: GapFadeConfig = None):
        self.config = config or GapFadeConfig()
        self.engine = GapFadeEngine(self.config)
        self.scanner = GapScanner(self.config)
        self.streamer: Optional[AlpacaTickStreamer] = None
        self.status = 'stopped'        # stopped, scanning, trading, paused, waiting
        self.candidates: List[GapCandidate] = []
        self._task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self.last_scan_time = ''
        self.messages: List[dict] = []  # decision feed
        self._position_lock = asyncio.Lock()  # P0-2: serializes ALL position mutations
        self._last_reconcile = 0.0  # monotonic time of last broker reconciliation
        self._last_autonomous_review = 0.0  # monotonic time of last LLM review
        self._AUTONOMOUS_REVIEW_INTERVAL = 900  # seconds (15 min, event trigger can fire sooner)
        self._last_llm_decide = 0.0  # monotonic time of last decide_action call
        self._LLM_DECIDE_INTERVAL = 120  # seconds — don't re-ask Rudra unless state changed
        self._last_llm_state_hash = ''  # detect meaningful state changes
        self._last_loop_heartbeat = _time.monotonic()  # watchdog: trading loop health
        self._last_model_rebuild = ''  # ISO date of last model rebuild

        # SQLite database for persistent storage
        _db = get_price_db()

        # Event bus for LLM conversation (replaces flat _event_queue)
        self._event_bus = EventBus(db=_db)
        self._event_queue = self._event_bus  # backward compat alias
        self._urgent_llm_event: bool = False  # Tier 3 flag for immediate LLM call
        self._morning_briefing_done: str = ''  # date string, prevents re-triggering

        # Trading journal (append-only decision audit trail, SQLite-backed)
        self.journal = TradingJournal(db=_db)

        # Market event detector (zero-LLM, pure arithmetic)
        self.event_detector = MarketEventDetector()

        # Telegram alerts
        self.alerter = AlertNotifier(self.config)

        # Load persisted state (before LLM init so config overrides apply)
        self.llm_supervisor = None
        self._load_state()

        # Strategy plugin system
        self.strategy = None
        self.indicator_engine = None
        self._strategy_config = {}  # strategy-specific config from state
        self._load_strategy(self.config.active_strategy)

        # LLM Supervisor (after state load so llm_enabled from saved config takes effect)
        if self.config.llm_enabled:
            self.llm_supervisor = LLMSupervisor(self.config)
            # Restore conversation history from saved state (same-day only)
            saved_conv = getattr(self, '_saved_conversation', {})
            if saved_conv:
                self.llm_supervisor.memory = ConversationMemory.from_dict(saved_conv)
                if self.llm_supervisor.memory.history:
                    logger.info(f"Restored {len(self.llm_supervisor.memory.history)} conversation messages")
            # Apply strategy-specific LLM prompts
            if self.strategy:
                sys_prompt = self.strategy.get_llm_system_prompt()
                if sys_prompt:
                    self.llm_supervisor._SYSTEM_PROMPT = sys_prompt
                profit_prompt = self.strategy.get_llm_profit_prompt()
                if profit_prompt:
                    self.llm_supervisor._PROFIT_SYSTEM_PROMPT = profit_prompt
            logger.info(f"Rudra enabled: {self.config.llm_model} @ {self.config.llm_url}")
        else:
            logger.info("Rudra disabled (llm_enabled=false)")

    def _add_message(self, msg_type: str, text: str, data: dict = None):
        """Add a message to the decision feed."""
        msg = {
            'type': msg_type,
            'text': text,
            'time': datetime.now(ET).strftime('%H:%M:%S'),
            'data': data or {},
        }
        self.messages.append(msg)
        if len(self.messages) > 200:
            self.messages = self.messages[-100:]

    def _add_event(self, text: str, tier: int = 2):
        """Add an event to the LLM conversation queue.

        Tiers:
            1 = Silent: injected into memory as a brief note
            2 = Batched: queued for next scheduled LLM call
            3 = Triggered: immediate LLM conversation call (stop-outs, loss limits)
            4 = Human escalation: Telegram alert + urgent LLM + auto-standdown
        """
        self._event_bus.push_text(text, tier)
        # Tier 1: also inject directly into conversation memory
        if tier == 1 and self.llm_supervisor:
            self.llm_supervisor.memory.add_event(text)
        # Tier 3+: flag for immediate processing (checked by trading loop)
        if tier >= 3:
            self._urgent_llm_event = True

    def _load_strategy(self, strategy_id: str):
        """Load a strategy by ID, creating indicator engine if needed."""
        try:
            from gap_fade_strategies import GapFadeStrategyRegistry, TickIndicatorEngine
            if not GapFadeStrategyRegistry.has_strategy(strategy_id):
                logger.warning(f"Strategy '{strategy_id}' not found, falling back to classic_gap_fade")
                strategy_id = 'classic_gap_fade'
            self.strategy = GapFadeStrategyRegistry.create_strategy(
                strategy_id, config=self._strategy_config.get(strategy_id))
            self.config.active_strategy = strategy_id

            # Create indicator engine if strategy needs it
            required = self.strategy.get_required_indicators()
            if required:
                ema_period = self.strategy.config.get('ema_trailing_period', 9)
                or_minutes = self.strategy.config.get('opening_range_minutes', 5)
                ema_buffer = self.strategy.config.get('ema_trailing_buffer_pct', 0.001)
                self.indicator_engine = TickIndicatorEngine(
                    indicators=required,
                    ema_period=ema_period,
                    or_minutes=or_minutes,
                    ema_buffer_pct=ema_buffer,
                )
                logger.info(f"Indicator engine created: {required}")
            else:
                self.indicator_engine = None

            logger.info(f"Strategy loaded: {self.strategy.name} ({strategy_id})")
        except Exception as e:
            logger.error(f"Failed to load strategy '{strategy_id}': {e}")
            self.strategy = None
            self.indicator_engine = None

    def switch_strategy(self, strategy_id: str, strategy_config: dict = None):
        """Hot-swap the active strategy (no restart needed).

        Existing positions continue with engine defaults.
        """
        if strategy_config:
            self._strategy_config[strategy_id] = strategy_config
        self._load_strategy(strategy_id)

        # Update LLM prompts if supervisor is active
        if self.llm_supervisor and self.strategy:
            sys_prompt = self.strategy.get_llm_system_prompt()
            if sys_prompt:
                self.llm_supervisor._SYSTEM_PROMPT = sys_prompt
            profit_prompt = self.strategy.get_llm_profit_prompt()
            if profit_prompt:
                self.llm_supervisor._PROFIT_SYSTEM_PROMPT = profit_prompt

        self._save_state()
        self._add_message('strategy', f'Switched to: {self.strategy.name if self.strategy else strategy_id}')

    async def start(self):
        """Start the live trading loop."""
        if self.status in ('trading', 'scanning'):
            logger.info(f"Start called but already {self.status}, skipping")
            return
        self.status = 'scanning'
        # Only reset daily stats on a genuinely new day (not mid-day restart)
        today_str = datetime.now(ET).strftime('%Y-%m-%d')
        if self.engine.daily_stats.date != today_str:
            self.engine.reset_daily()
            self.event_detector.reset_daily()
        # P0-4: reconcile with broker on startup
        await self._reconcile_with_broker()
        # Start tick streamer for existing positions (e.g. after restart)
        if self.engine.positions and not self.streamer:
            syms = list(self.engine.positions.keys())
            self.streamer = AlpacaTickStreamer(syms, on_tick=self._on_tick)
            await self.streamer.start()
            logger.info(f"Tick streamer started for {len(syms)} existing positions: {syms}")
        self._task = asyncio.create_task(self._trading_loop())
        logger.info("Live trader started")
        self._add_message('system', 'Live trader started')
        await broadcast({'type': 'live_status', 'status': self.status})

    async def stop(self):
        """Stop the live trader and close all positions."""
        self.status = 'stopped'
        # Cancel all broker-side stop orders
        for sym, pos in list(self.engine.positions.items()):
            if pos.stop_order_id:
                try:
                    await asyncio.to_thread(alpaca_cancel_order, pos.stop_order_id)
                    pos.stop_order_id = ''
                except Exception:
                    pass
        if self.streamer:
            await self.streamer.stop()
            self.streamer = None
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self._add_message('system', 'Live trader stopped')
        self._save_state()
        await broadcast({'type': 'live_status', 'status': self.status})

    async def pause(self):
        self.status = 'paused'
        self._add_message('system', 'Live trader paused')
        await broadcast({'type': 'live_status', 'status': self.status})

    async def resume(self):
        if self.status == 'paused':
            self.status = 'trading'
            self._add_message('system', 'Live trader resumed')
            await broadcast({'type': 'live_status', 'status': self.status})

    # -------------------------------------------------------------------------
    # P0-4: Broker Reconciliation
    # -------------------------------------------------------------------------

    async def _reconcile_with_broker(self):
        """Compare internal positions with Alpaca broker positions.

        On startup and periodically during trading hours. Broker is source of
        truth — if we have a position internally that doesn't exist on broker,
        we remove it. If broker has a position we don't know about, we adopt it.
        """
        try:
            broker_positions = await asyncio.to_thread(alpaca_get_positions)
        except Exception as e:
            logger.warning(f"Reconciliation failed (could not reach broker): {e}")
            await self.alerter.send('Broker API Failure',
                f'Could not reach Alpaca broker: {e}',
                level='error', throttle_key='broker_api')
            return

        broker_map = {}  # symbol -> {qty, avg_entry_price, side}
        for bp in broker_positions:
            sym = bp.get('symbol', '')
            qty = abs(int(bp.get('qty', 0)))
            side = bp.get('side', '')
            avg_price = float(bp.get('avg_entry_price', 0))
            if sym and qty > 0:
                broker_map[sym] = {'qty': qty, 'avg_entry_price': avg_price, 'side': side}

        internal_syms = set(self.engine.positions.keys())
        broker_syms = set(broker_map.keys())

        # Positions we think we have but broker doesn't
        orphaned = internal_syms - broker_syms
        for sym in orphaned:
            pos = self.engine.positions[sym]
            logger.warning(f"RECONCILE: Internal position {sym} ({pos.remaining_shares} shares) "
                          f"NOT found on broker — removing from internal state")
            self._add_message('warning', f'RECONCILE: {sym} not on broker — removed internally')
            del self.engine.positions[sym]

        # Positions broker has that we don't know about
        unknown = broker_syms - internal_syms
        for sym in unknown:
            bp = broker_map[sym]
            direction = bp['side']  # 'long' or 'short'
            if direction not in ('long', 'short'):
                logger.warning(f"RECONCILE: Unknown side '{direction}' for {sym} — skipping")
                continue
            logger.warning(f"RECONCILE: Broker has {direction} {sym} ({bp['qty']} shares @ "
                          f"${bp['avg_entry_price']:.2f}) — adopting into internal state")
            self._add_message('warning',
                f'RECONCILE: Adopting broker {direction} {sym} ({bp["qty"]} shares)')
            if direction == 'short':
                stop_price = bp['avg_entry_price'] * (1 + self.config.stop_pct)
                half_target = bp['avg_entry_price'] * 0.99
                full_target = bp['avg_entry_price'] * 0.97
                prev_close = bp['avg_entry_price'] * 0.97
            else:
                stop_price = bp['avg_entry_price'] * (1 - self.config.stop_pct)
                half_target = bp['avg_entry_price'] * 1.01
                full_target = bp['avg_entry_price'] * 1.03
                prev_close = bp['avg_entry_price'] * 1.03
            pos = GapPosition(
                symbol=sym, shares=bp['qty'], entry_price=bp['avg_entry_price'],
                stop_price=stop_price,
                half_target=half_target,
                full_target=full_target,
                prev_close=prev_close,
                entry_time=datetime.now(ET).strftime('%Y-%m-%d %H:%M'),
                remaining_shares=bp['qty'],
                entry_fill_price=bp['avg_entry_price'],
                direction=direction,
            )
            self.engine.positions[sym] = pos

        # Quantity mismatches on shared positions
        shared = internal_syms & broker_syms
        for sym in shared:
            bp = broker_map[sym]
            pos = self.engine.positions[sym]
            if bp['qty'] != pos.remaining_shares:
                logger.warning(f"RECONCILE: {sym} qty mismatch — internal={pos.remaining_shares}, "
                              f"broker={bp['qty']}. Adopting broker qty.")
                self._add_message('warning',
                    f'RECONCILE: {sym} qty adjusted {pos.remaining_shares} → {bp["qty"]}')
                pos.remaining_shares = bp['qty']
                pos.shares = bp['qty']

        self._last_reconcile = _time.monotonic()
        if orphaned or unknown or any(
            broker_map.get(s, {}).get('qty', 0) != self.engine.positions.get(s, GapPosition(
                symbol='', shares=0, entry_price=0, stop_price=0,
                half_target=0, full_target=0, prev_close=0)).remaining_shares
            for s in shared
        ):
            self._save_state()

        logger.info(f"Reconciliation complete: {len(self.engine.positions)} positions, "
                   f"{len(orphaned)} orphaned, {len(unknown)} adopted")

    # -------------------------------------------------------------------------
    # Main Trading Loop
    # -------------------------------------------------------------------------

    def _build_llm_state(self, now: datetime) -> dict:
        """Build state dict for LLM supervisor context."""
        state = {
            'time_et': now.strftime('%H:%M:%S'),
            'day_of_week': now.strftime('%A'),
            'equity': self.engine.equity,
            'daily_pnl': self.engine.daily_stats.pnl,
            'daily_trades': self.engine.daily_stats.trades,
            'wins': self.engine.daily_stats.wins,
            'losses': self.engine.daily_stats.losses,
            'consecutive_losses': self.engine.daily_stats.consecutive_losses,
            'positions': {s: asdict(p) for s, p in self.engine.positions.items()},
            'candidates_count': len(self.candidates),
            'top_candidates': [asdict(c) for c in self.candidates[:10]],
            'last_scan_time': self.last_scan_time,
            'stopped_today': list(self.engine._stopped_today.keys()),
            'status': self.status,
            'recent_messages': [m['text'] for m in self.messages[-10:]],
            'recent_trades': [asdict(t) for t in self.engine.all_trade_log[-5:]],
            'peak_equity': self.engine.peak_equity,
            'config': {
                'stop_pct': self.config.stop_pct,
                'max_positions': self.config.max_positions,
                'gap_threshold': self.config.gap_threshold,
                'adaptive_stops': self.config.adaptive_stops,
                'reentry_enabled': self.config.reentry_enabled,
                'dd_circuit_breaker': self.config.dd_circuit_breaker,
                'dd_tier1_threshold': self.config.dd_tier1_threshold,
                'dd_tier2_threshold': self.config.dd_tier2_threshold,
                'dd_hard_stop': self.config.dd_hard_stop,
            },
        }
        # Enrich with journal summary and indicator snapshot
        state['journal_summary'] = self.journal.get_summary(minutes=20)
        if self.indicator_engine and self.engine.positions:
            indicators = {}
            for sym in self.engine.positions:
                td = self.indicator_engine.get_data(sym)
                if td:
                    indicators[sym] = {k: round(v, 2) if isinstance(v, float) else v
                                       for k, v in td.items()}
            if indicators:
                state['indicators'] = indicators
        return state

    def _build_lessons_learned(self) -> str:
        """Analyze trade history and build a concise lessons-learned summary for LLM context.

        Returns a 5-10 line string summarizing patterns from all_trade_log.
        Cached for 5 minutes via _lessons_cache.
        """
        now_mono = _time.monotonic()
        if hasattr(self, '_lessons_cache') and self._lessons_cache:
            cached_time, cached_text = self._lessons_cache
            if now_mono - cached_time < 300:  # 5 min cache
                return cached_text

        trades = self.engine.all_trade_log
        if len(trades) < 3:
            return ''

        lines = ['## LESSONS FROM TRADE HISTORY']

        # Win rate by entry hour
        hour_wins = {}
        hour_total = {}
        for t in trades:
            try:
                h = int(t.entry_time.split(' ')[1].split(':')[0])
            except (IndexError, ValueError):
                continue
            hour_total[h] = hour_total.get(h, 0) + 1
            if t.pnl > 0:
                hour_wins[h] = hour_wins.get(h, 0) + 1
        if hour_total:
            parts = []
            for h in sorted(hour_total.keys()):
                wr = hour_wins.get(h, 0) / hour_total[h] * 100
                parts.append(f'{h}:00={wr:.0f}%({hour_total[h]})')
            lines.append(f'- Win rate by hour: {", ".join(parts)}')

        # Avg holding time: winners vs losers
        winner_mins = [t.holding_minutes for t in trades if t.pnl > 0 and t.holding_minutes > 0]
        loser_mins = [t.holding_minutes for t in trades if t.pnl <= 0 and t.holding_minutes > 0]
        if winner_mins and loser_mins:
            avg_w = sum(winner_mins) / len(winner_mins)
            avg_l = sum(loser_mins) / len(loser_mins)
            lines.append(f'- Avg hold: winners {avg_w:.0f} min, losers {avg_l:.0f} min')

        # P&L by exit reason
        reason_pnl = {}
        reason_cnt = {}
        for t in trades:
            r = t.exit_reason
            reason_pnl[r] = reason_pnl.get(r, 0) + t.pnl
            reason_cnt[r] = reason_cnt.get(r, 0) + 1
        if reason_pnl:
            parts = []
            for r in sorted(reason_pnl.keys(), key=lambda x: reason_pnl[x]):
                parts.append(f'{r}=${reason_pnl[r]:+,.0f}({reason_cnt[r]})')
            lines.append(f'- P&L by exit: {", ".join(parts)}')

        # Churning symbols (traded 3+ times with net loss)
        sym_pnl = {}
        sym_cnt = {}
        for t in trades:
            sym_pnl[t.symbol] = sym_pnl.get(t.symbol, 0) + t.pnl
            sym_cnt[t.symbol] = sym_cnt.get(t.symbol, 0) + 1
        churners = [(s, sym_cnt[s], sym_pnl[s]) for s in sym_pnl
                     if sym_cnt[s] >= 3 and sym_pnl[s] < 0]
        if churners:
            churners.sort(key=lambda x: x[2])
            parts = [f'{s}({cnt}x, ${pnl:+,.0f})' for s, cnt, pnl in churners[:5]]
            lines.append(f'- Churning symbols (net loss): {", ".join(parts)}')

        # Overall stats
        total = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in trades)
        lines.append(f'- Overall: {total} trades, {wins/total*100:.0f}% win rate, ${total_pnl:+,.0f} total P&L')

        result = '\n'.join(lines)
        self._lessons_cache = (now_mono, result)
        return result

    async def _weekly_model_maintenance(self):
        """Rebuild LLM training data and model weekly (Friday after hours).

        Only runs if >= 20 new trades since last rebuild.
        """
        try:
            # Check if already rebuilt this week
            today_str = datetime.now(ET).strftime('%Y-%m-%d')
            if self._last_model_rebuild == today_str:
                return

            # Need enough trades to make rebuilding worthwhile
            total_trades = len(self.engine.all_trade_log)
            if total_trades < 20:
                logger.info(f"Model rebuild skipped: only {total_trades} trades (need 20+)")
                return

            logger.info(f"Starting weekly model maintenance ({total_trades} trades)")
            self._add_message('system', f'Weekly model rebuild starting ({total_trades} trades)')

            # Save state first
            self._save_state()

            # Generate training data
            import subprocess
            script_dir = os.path.dirname(os.path.abspath(__file__))
            gen_script = os.path.join(script_dir, 'generate_training_data.py')
            if os.path.exists(gen_script):
                result = await asyncio.to_thread(
                    subprocess.run,
                    ['python', gen_script],
                    cwd=script_dir,
                    capture_output=True, text=True, timeout=120)
                if result.returncode != 0:
                    logger.error(f"Training data generation failed: {result.stderr[:500]}")
                    await self.alerter.send('Model Rebuild Failed',
                        f'generate_training_data.py failed:\n{result.stderr[:200]}',
                        level='error')
                    return
                logger.info(f"Training data generated: {result.stdout[:200]}")

            # Rebuild Ollama model
            modelfile = os.path.join(script_dir, 'Modelfile.gapfade')
            if os.path.exists(modelfile):
                result = await asyncio.to_thread(
                    subprocess.run,
                    ['ollama', 'create', 'rudra', '-f', modelfile],
                    cwd=script_dir,
                    capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    logger.error(f"Ollama model rebuild failed: {result.stderr[:500]}")
                    await self.alerter.send('Model Rebuild Failed',
                        f'ollama create failed:\n{result.stderr[:200]}',
                        level='error')
                    return
                logger.info(f"Ollama model rebuilt: {result.stdout[:200]}")

            self._last_model_rebuild = today_str
            self._add_message('system', f'Weekly model rebuild complete ({total_trades} trades)')
            await self.alerter.send('Model Rebuilt',
                f'Training data regenerated and model rebuilt\nTrades: {total_trades}',
                level='info')
            logger.info("Weekly model maintenance complete")

        except Exception as e:
            logger.error(f"Weekly model maintenance failed: {e}\n{traceback.format_exc()}")
            await self.alerter.send('Model Maintenance Error',
                f'{type(e).__name__}: {e}', level='error')

    def _is_market_day(self, now: datetime) -> tuple:
        """Check if today is a market trading day.

        Returns (is_market_day: bool, next_open: datetime).
        """
        # Weekend check
        if now.weekday() >= 5:
            days_until_monday = 7 - now.weekday()
            next_open = (now + timedelta(days=days_until_monday)).replace(
                hour=7, minute=0, second=0, microsecond=0)
            return False, next_open

        # US market holidays (fixed + observed)
        year = now.year
        holidays = set()

        # Fixed-date holidays
        fixed = [
            (1, 1),   # New Year's Day
            (6, 19),  # Juneteenth
            (7, 4),   # Independence Day
            (12, 25), # Christmas
        ]
        for m, d in fixed:
            dt = date(year, m, d)
            if dt.weekday() == 5:    # Saturday → observe Friday
                holidays.add(date(year, m, d - 1))
            elif dt.weekday() == 6:  # Sunday → observe Monday
                holidays.add(date(year, m, d + 1))
            else:
                holidays.add(dt)

        # MLK Day: 3rd Monday of January
        holidays.add(self._nth_weekday(year, 1, 0, 3))
        # Presidents' Day: 3rd Monday of February
        holidays.add(self._nth_weekday(year, 2, 0, 3))
        # Memorial Day: last Monday of May
        holidays.add(self._last_weekday(year, 5, 0))
        # Labor Day: 1st Monday of September
        holidays.add(self._nth_weekday(year, 9, 0, 1))
        # Thanksgiving: 4th Thursday of November
        holidays.add(self._nth_weekday(year, 11, 3, 4))

        # Good Friday (Easter - 2 days)
        holidays.add(self._good_friday(year))

        today = now.date()
        if today in holidays:
            # Find next non-holiday weekday
            next_day = now + timedelta(days=1)
            while next_day.weekday() >= 5 or next_day.date() in holidays:
                next_day += timedelta(days=1)
            next_open = next_day.replace(hour=7, minute=0, second=0, microsecond=0)
            return False, next_open

        return True, now

    @staticmethod
    def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
        """Get the nth occurrence of a weekday in a month (weekday: 0=Mon, 3=Thu)."""
        first = date(year, month, 1)
        day_of_week = first.weekday()
        diff = (weekday - day_of_week) % 7
        return first + timedelta(days=diff + 7 * (n - 1))

    @staticmethod
    def _last_weekday(year: int, month: int, weekday: int) -> date:
        """Get the last occurrence of a weekday in a month."""
        if month == 12:
            last_day = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(year, month + 1, 1) - timedelta(days=1)
        diff = (last_day.weekday() - weekday) % 7
        return last_day - timedelta(days=diff)

    @staticmethod
    def _good_friday(year: int) -> date:
        """Compute Good Friday using anonymous Gregorian algorithm for Easter."""
        a = year % 19
        b, c = divmod(year, 100)
        d, e = divmod(b, 4)
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i, k = divmod(c, 4)
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month, day = divmod(h + l - 7 * m + 114, 31)
        easter = date(year, month, day + 1)
        return easter - timedelta(days=2)

    async def _trading_loop(self):
        """Main trading loop — LLM-driven when enabled, schedule-based fallback.

        When LLM supervisor is active, it decides every action (scan, enter,
        monitor, standdown). Safety nets (before 7AM sleep, EOD close, after-hours
        shutdown) always run mechanically regardless of LLM.
        """
        # Track what has been done today (reset on new date) — used by fallback schedule
        _done_date = ''
        _did_scan_7am = False
        _did_scan_925 = False
        _did_enter = False
        _did_eod = False

        try:
            while self.status != 'stopped':
              try:
                self._last_loop_heartbeat = _time.monotonic()
                now = datetime.now(ET)
                today = now.strftime('%Y-%m-%d')

                # Reset flags on new day
                if today != _done_date:
                    _done_date = today
                    _did_scan_7am = False
                    _did_scan_925 = False
                    _did_enter = False
                    _did_eod = False

                # ── SAFETY NETS (always active, LLM cannot override) ──

                # Before pre-market (before 7 AM) — sleep
                if now.hour < 7:
                    self.status = 'waiting'
                    await broadcast({'type': 'live_status', 'status': 'waiting'})
                    target = now.replace(hour=7, minute=0, second=0, microsecond=0)
                    sleep_sec = max(30, (target - now).total_seconds())
                    await asyncio.sleep(sleep_sec)
                    continue

                # Weekend / holiday check — sleep until next market day
                is_market, next_open = self._is_market_day(now)
                if not is_market:
                    self.status = 'waiting'
                    await broadcast({'type': 'live_status', 'status': 'waiting'})
                    self._add_message('info',
                        f'Market closed ({now.strftime("%A")}). '
                        f'Next open: {next_open.strftime("%a %b %d %H:%M")}')
                    sleep_sec = max(60, (next_open - now).total_seconds())
                    await asyncio.sleep(min(sleep_sec, 86400))  # cap at 24h
                    continue

                # EOD close at 3:50+ PM (once per day, mechanical safety net)
                if now.hour == 15 and now.minute >= 50 and not _did_eod:
                    # EOD Debrief — run before closing positions
                    if self.llm_supervisor and self.llm_supervisor.is_available():
                        state = self._build_llm_state(now)
                        debrief = await self.llm_supervisor.eod_debrief(
                            state, event_queue=self._event_bus)
                        if debrief:
                            self._add_message('conversation',
                                f'BOT: Day wrap-up request | RUDRA: {debrief[:300]}')
                            await broadcast({'type': 'conversation',
                                'speaker': 'rudra', 'text': debrief[:500]})
                    if self.engine.positions:
                        _did_eod = True
                        await self._eod_close()
                        await asyncio.sleep(60)
                        continue

                # After hours (4 PM+) — save state, sleep until next morning
                if now.hour >= 16:
                    if self.engine.positions:
                        logger.error("CRITICAL: After hours with open positions! Emergency close.")
                        self._add_message('error', 'EMERGENCY: Positions open after hours — force closing')
                        await self._eod_close()

                    # Send daily P&L summary before resetting
                    stats = self.engine.daily_stats
                    today_trades = [t for t in self.engine.all_trade_log
                                    if hasattr(t, 'exit_time') and t.exit_time
                                    and t.exit_time.startswith(today)]
                    wins = sum(1 for t in today_trades if t.pnl > 0)
                    losses = len(today_trades) - wins
                    trade_details = ', '.join(
                        f'{t.symbol} {"+" if t.pnl >= 0 else ""}${t.pnl:.0f}'
                        for t in today_trades[:10])
                    await self.alerter.send(
                        f'Daily Summary \u2014 {now.strftime("%a %b %d")}',
                        f'Equity: ${self.engine.equity:,.2f}\n'
                        f'Today: {"+" if stats.pnl >= 0 else ""}${stats.pnl:.2f} ({wins}W/{losses}L)\n'
                        f'Trades: {trade_details or "none"}',
                        level='summary')

                    # Weekly model maintenance (Friday after hours)
                    if now.weekday() == 4:
                        await self._weekly_model_maintenance()

                    self._save_state()
                    self.engine.reset_daily()
                    self.event_detector.reset_daily()
                    self.status = 'waiting'
                    await broadcast({'type': 'live_status', 'status': 'waiting'})
                    self._add_message('info', 'After hours — sleeping until 7:00 AM ET')
                    tomorrow_7am = (now + timedelta(days=1)).replace(
                        hour=7, minute=0, second=0, microsecond=0)
                    sleep_sec = max(60, (tomorrow_7am - now).total_seconds())
                    await asyncio.sleep(sleep_sec)
                    continue

                # ── LLM SUPERVISOR MODE ──
                if self.llm_supervisor and self.llm_supervisor.is_available():
                    # Update lessons learned from trade history
                    self.llm_supervisor.lessons_context = self._build_lessons_learned()

                    # Morning briefing (once per day, after first scan)
                    if (self._morning_briefing_done != today
                            and (self.candidates or now.hour >= 9)):
                        state = self._build_llm_state(now)
                        briefing = await self.llm_supervisor.morning_briefing(
                            state, event_queue=self._event_bus)
                        if briefing:
                            self._morning_briefing_done = today
                            self._add_message('conversation',
                                f'BOT: Morning briefing | RUDRA: {briefing[:300]}')
                            await broadcast({'type': 'conversation',
                                'speaker': 'rudra', 'text': briefing[:500]})

                    # Tier 4: Human escalation — Telegram alert + standdown
                    tier4_events = [e for e in self._event_bus._queue if e.tier >= 4]
                    if tier4_events:
                        for t4 in tier4_events:
                            await self.alerter.send(
                                f'HUMAN ESCALATION: {t4.event_type}',
                                f'{t4.description}\n'
                                f'Equity: ${self.engine.equity:,.2f}, '
                                f'Daily P&L: ${self.engine.daily_stats.pnl:+,.2f}',
                                level='error', throttle_key=f'tier4:{t4.event_type}')
                            self.journal.log('action', 'event_detector',
                                f'TIER 4 ESCALATION: {t4.description}',
                                data=t4.data)
                        self._add_message('error',
                            f'HUMAN ESCALATION: {tier4_events[0].description} — '
                            f'Telegram alert sent, entering standdown')

                    # Tier 3+ urgent event — immediate LLM consultation
                    if self._event_bus.has_urgent() or self._urgent_llm_event:
                        self._urgent_llm_event = False
                        # Peek at urgent events for prompt context
                        urgent_texts = [e.as_text() for e in self._event_bus._queue if e.tier >= 3][-3:]
                        if not urgent_texts:
                            urgent_texts = [e.as_text() for e in self._event_bus._queue][-3:]
                        state = self._build_llm_state(now)
                        urgent_prompt = (
                            f"Urgent situation. {' '.join(urgent_texts)}\n"
                            f"Current equity: ${state['equity']:,.2f}, "
                            f"daily P&L: ${state['daily_pnl']:+,.2f}, "
                            f"consecutive losses: {state['consecutive_losses']}.\n"
                            f"What's your read? Analyze the situation — is this a pattern of losses or isolated? "
                            f"Should we stand down, tighten up, or keep trading?\n\n"
                            f"Share your assessment, then end with JSON: {self.llm_supervisor._ACTION_SCHEMA}"
                        )
                        system_prompt = self.llm_supervisor._SYSTEM_PROMPT
                        if self.llm_supervisor.lessons_context:
                            system_prompt += '\n\n' + self.llm_supervisor.lessons_context
                        self.journal.log('observation', 'event_detector',
                            f'Urgent event triggered LLM consultation: {"; ".join(urgent_texts)}')
                        raw = await self.llm_supervisor._call_llm(
                            system_prompt, urgent_prompt,
                            use_memory=True, record_exchange=True,
                            conversation_role='urgent_event',
                            event_queue=self._event_bus)
                        if raw:
                            self._add_message('conversation',
                                f'BOT: Urgent event | RUDRA: {raw[:300]}')
                            await broadcast({'type': 'conversation',
                                'speaker': 'rudra', 'text': raw[:500]})
                            self.journal.log('reasoning', 'llm_rudra',
                                raw[:500], llm_call=True)
                            # Parse action if present
                            parsed = self.llm_supervisor._parse_json(raw)
                            if isinstance(parsed, dict) and parsed.get('action') == 'standdown':
                                standdown_min = parsed.get('standdown_minutes', 15)
                                self.status = 'standdown'
                                await broadcast({'type': 'live_status', 'status': 'standdown'})
                                self._add_message('llm', f'Standing down {standdown_min}m after urgent event')
                                self.journal.log('action', 'llm_rudra',
                                    f'Standing down {standdown_min}m after urgent event')
                                await asyncio.sleep(min(standdown_min * 60, 900))
                                self.status = 'scanning'
                                await broadcast({'type': 'live_status', 'status': 'scanning'})
                                continue

                    state = self._build_llm_state(now)

                    # ── THROTTLE: skip decide_action if nothing changed ──
                    _now_mono = _time.monotonic()
                    _state_hash = (f"{len(self.engine.positions)}|"
                                   f"{state.get('candidates_count', 0)}|"
                                   f"{state['wins']}|{state['losses']}|"
                                   f"{self.status}")
                    _state_changed = (_state_hash != self._last_llm_state_hash)
                    _time_elapsed = (_now_mono - self._last_llm_decide >= self._LLM_DECIDE_INTERVAL)
                    _has_events = bool(self._event_bus)

                    if _state_changed or _time_elapsed or _has_events:
                        # Ask Rudra — something meaningful changed or enough time passed
                        self._last_llm_decide = _now_mono
                        self._last_llm_state_hash = _state_hash

                        decision = await self.llm_supervisor.decide_action(
                            state, event_queue=self._event_bus)
                        action = decision.get('action', 'monitor')
                        reasoning = decision.get('reasoning', '')
                        # Use full analysis text if available, fall back to JSON reasoning
                        rudra_analysis = decision.get('_analysis', reasoning)
                        rudra_display = rudra_analysis if rudra_analysis else reasoning

                        # ── MECHANICAL GUARD: no entries before market open ──
                        _market_open = (now.hour > 9 or (now.hour == 9 and now.minute >= 30))
                        if action == 'enter' and not _market_open:
                            self._add_message('conversation',
                                f'BOT: What should we do? | RUDRA: Enter — but market not open yet '
                                f'({now.strftime("%H:%M")} ET), overriding to wait')
                            action = 'wait'
                            reasoning = 'Pre-market: waiting for 9:30 AM open'
                            rudra_display = reasoning

                        # Show bot<->Rudra exchange in activity feed
                        _bot_context = (f'{len(self.engine.positions)} positions, '
                                        f'{state.get("candidates_count", 0)} candidates'
                                        if self.engine.positions or state.get('candidates_count')
                                        else state['status'])
                        self._add_message('conversation',
                            f'BOT: {_bot_context} | RUDRA: [{action}] {rudra_display[:400]}')
                        await broadcast({'type': 'conversation',
                            'speaker': 'rudra',
                            'text': f'[{action}] {rudra_display[:500]}'})
                        # Journal: log LLM decision
                        _jtype = 'action' if action in ('enter', 'scan', 'standdown') else 'no_action'
                        self.journal.log(_jtype, 'llm_rudra',
                            f'[{action}] {rudra_display[:200]}', llm_call=True)
                    else:
                        # Nothing changed — silently continue with last action
                        action = 'monitor'

                    if action == 'scan':
                        await self._run_scan()
                        await asyncio.sleep(15)
                    elif action == 'enter':
                        llm_symbols = decision.get('enter_symbols')
                        size_mult = decision.get('position_size_mult', 1.0)
                        await self._enter_positions(llm_symbols=llm_symbols, size_mult=size_mult)
                        await asyncio.sleep(15)
                    elif action == 'standdown':
                        standdown_min = decision.get('standdown_minutes', 30)
                        self.status = 'standdown'
                        await broadcast({'type': 'live_status', 'status': 'standdown'})
                        self._add_message('llm', f'Standing down for {standdown_min} minutes')
                        await asyncio.sleep(min(standdown_min * 60, 1800))
                        self.status = 'scanning'
                        await broadcast({'type': 'live_status', 'status': 'scanning'})
                    elif action == 'wait':
                        new_status = 'trading' if self.engine.positions else 'scanning'
                        if self.status != new_status:
                            self.status = new_status
                            await broadcast({'type': 'live_status', 'status': new_status})
                        await asyncio.sleep(30)
                    else:  # monitor
                        if self.engine.positions:
                            if self.status != 'trading':
                                self.status = 'trading'
                                await broadcast({'type': 'live_status', 'status': 'trading'})
                            await self._check_positions()
                            if _time.monotonic() - self._last_reconcile > self.RECONCILE_INTERVAL:
                                await self._reconcile_with_broker()
                            # Autonomous review: periodic portfolio self-check
                            await self._execute_scheduled_reflection()
                        else:
                            if self.status != 'scanning':
                                self.status = 'scanning'
                                await broadcast({'type': 'live_status', 'status': 'scanning'})
                        await asyncio.sleep(15)
                    continue

                # ── FALLBACK: original schedule (LLM disabled or circuit open) ──
                if self.llm_supervisor and not self.llm_supervisor.is_available():
                    self._add_message('llm_fallback', 'Rudra unavailable (circuit breaker), using rules schedule')
                    await self.alerter.send('Rudra Circuit Breaker',
                        f'Rudra unavailable — falling back to rules schedule\n'
                        f'Failures: {self.llm_supervisor._failure_count}/{self.llm_supervisor._max_failures}',
                        level='warning', throttle_key='llm_circuit')

                # Pre-market scan (7:00+ AM, once per day)
                if 7 <= now.hour < 9 and not _did_scan_7am:
                    _did_scan_7am = True
                    await self._run_scan()
                    await asyncio.sleep(30)
                    continue

                # Re-scan at 9:25+ AM (once per day, refine candidates)
                if now.hour == 9 and now.minute >= 25 and not _did_scan_925:
                    _did_scan_925 = True
                    await self._run_scan()
                    # Fast-poll until entry window opens
                    _ew = self.strategy.get_entry_window() if self.strategy else None
                    _ew_min = _ew[1] if _ew else 31
                    _seconds_to_entry = max(0, (_ew_min - now.minute) * 60 - now.second)
                    if _seconds_to_entry > 0:
                        self._add_message('info', f'Scan done — waiting {_seconds_to_entry}s for 9:{_ew_min:02d} entry')
                        await asyncio.sleep(min(_seconds_to_entry, 5))
                    continue

                # Market open — enter positions (strategy controls entry window)
                _entry_window = self.strategy.get_entry_window() if self.strategy else None
                if _entry_window:
                    _entry_start_h, _entry_start_m, _cutoff_h, _cutoff_m = _entry_window
                else:
                    _entry_start_h, _entry_start_m = 9, 31
                    _cutoff_h = self.config.entry_cutoff_hour
                    _cutoff_m = self.config.entry_cutoff_min
                _before_cutoff = (now.hour < _cutoff_h or
                                  (now.hour == _cutoff_h and now.minute < _cutoff_m))
                _after_entry_start = (now.hour > _entry_start_h or
                                      (now.hour == _entry_start_h and now.minute >= _entry_start_m))
                if _after_entry_start and _before_cutoff and not _did_enter:
                    if self.status in ('scanning', 'waiting'):
                        # If no candidates (e.g. mid-day restart), scan first
                        if not self.candidates:
                            self._add_message('scan', 'No candidates cached — running fresh scan before entry')
                            await self._run_scan()
                        _did_enter = True
                        await self._enter_positions()
                        await asyncio.sleep(30)
                        continue

                # During trading hours — monitor every 15 seconds
                if 9 <= now.hour < 16 and self.status == 'trading':
                    await self._check_positions()
                    if _time.monotonic() - self._last_reconcile > self.RECONCILE_INTERVAL:
                        await self._reconcile_with_broker()
                    # Autonomous review (fallback schedule too)
                    if self.engine.positions:
                        await self._execute_scheduled_reflection()
                    await asyncio.sleep(15)
                    continue

                # Waiting for next event
                await asyncio.sleep(30)

              except asyncio.CancelledError:
                raise
              except Exception as e:
                logger.error(f"Trading loop iteration error: {e}\n{traceback.format_exc()}")
                self._add_message('error', f'Loop error (retrying): {e}')
                await asyncio.sleep(15)  # backoff before retrying

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Trading loop fatal error: {e}\n{traceback.format_exc()}")
            self._add_message('error', f'Trading loop fatal error: {e}')
            await self.alerter.send('Trading Loop Error',
                f'{type(e).__name__}: {e}', level='error')

    async def _run_scan(self):
        """Run the pre-market scanner (non-blocking)."""
        self._add_message('scan', 'Running pre-market scan...')
        await broadcast({'type': 'live_status', 'status': 'scanning'})

        self.candidates, universe_size = await asyncio.to_thread(self.scanner.scan_premarket)
        self.last_scan_time = datetime.now(ET).strftime('%H:%M:%S')

        # Catalyst detection — classify gaps before ranking
        if self.config.catalyst_enabled and self.candidates:
            try:
                await detect_catalysts(self.candidates, self.config)
                self._apply_catalyst_scores()
                self.candidates.sort(key=lambda c: c.score, reverse=True)
                cat_counts = defaultdict(int)
                for c in self.candidates:
                    cat_counts[c.catalyst or 'noise'] += 1
                self._add_message('catalyst', f'Catalyst scan: {dict(cat_counts)}')
            except Exception as e:
                logger.warning(f"Catalyst detection failed (non-fatal): {e}")

        # Strategy plugin: filter and re-score candidates
        if self.strategy and self.candidates:
            pre_count = len(self.candidates)
            filtered = []
            for c in self.candidates:
                ok, reason = self.strategy.filter_candidate(asdict(c))
                if ok:
                    # Strategy scoring override
                    new_score = self.strategy.score_candidate(asdict(c))
                    if new_score is not None:
                        c.score = new_score
                    filtered.append(c)
                else:
                    logger.debug(f"Strategy filtered {c.symbol}: {reason}")
            if len(filtered) < pre_count:
                self._add_message('strategy',
                    f'{self.strategy.name}: filtered {pre_count - len(filtered)} candidates '
                    f'({len(filtered)} remain)')
            self.candidates = filtered
            self.candidates.sort(key=lambda c: c.score, reverse=True)

        self._add_message('scan', f'Found {len(self.candidates)} candidates from {universe_size} symbols', {
            'candidates': [asdict(c) for c in self.candidates[:10]]
        })
        await broadcast({
            'type': 'scan_results',
            'candidates': [asdict(c) for c in self.candidates[:10]],
            'scan_time': self.last_scan_time,
            'universe_size': universe_size,
        })

        # LLM candidate evaluation (non-fatal, best-effort)
        if self.llm_supervisor and self.llm_supervisor.is_available() and self.candidates:
            try:
                regime_info = None
                try:
                    regime = await fetch_market_regime_live(self.config)
                    if regime:
                        regime_info = {'spy_gap_pct': regime.spy_gap_pct, 'vix_level': regime.vix_level}
                except Exception:
                    pass
                evals = await self.llm_supervisor.evaluate_candidates(
                    [asdict(c) for c in self.candidates[:15]], regime=regime_info)
                if evals:
                    skip_syms = {e['symbol'] for e in evals
                                 if isinstance(e, dict) and e.get('action') == 'skip'}
                    trade_syms = [e for e in evals
                                  if isinstance(e, dict) and e.get('action') == 'trade']
                    # Build conversation summary
                    trade_names = ', '.join(e['symbol'] for e in trade_syms[:5]) if trade_syms else 'none'
                    skip_names = ', '.join(list(skip_syms)[:5]) if skip_syms else 'none'
                    self._add_message('conversation',
                        f'BOT: {len(self.candidates)} candidates from scan | '
                        f'RUDRA: Trade {trade_names}; skip {skip_names}')
                    await broadcast({'type': 'conversation',
                        'speaker': 'rudra',
                        'text': f'Candidates: trade {trade_names}, skip {skip_names}'})
                    if skip_syms:
                        before = len(self.candidates)
                        self.candidates = [c for c in self.candidates if c.symbol not in skip_syms]
            except Exception as e:
                logger.warning(f"Rudra candidate evaluation failed (non-fatal): {e}")

    def _apply_catalyst_scores(self):
        """Adjust candidate scores based on catalyst classification.

        Earnings handling: if catalyst_skip_earnings=True, hard-skip (score=-999).
        Otherwise, apply a heavy penalty (catalyst_earnings_penalty, default 40).
        This lets high-scoring candidates survive even with an earnings tag,
        which is important because yfinance earnings data is often inaccurate.
        """
        filtered = []
        for c in self.candidates:
            if c.catalyst == 'earnings':
                if self.config.catalyst_skip_earnings:
                    c.score = -999
                    self._add_message('catalyst',
                        f'{c.symbol}: SKIP — earnings gap ({c.catalyst_detail})')
                    continue
                else:
                    penalty = self.config.catalyst_earnings_penalty
                    c.score -= penalty
                    self._add_message('catalyst',
                        f'{c.symbol}: -{penalty} penalty (earnings: {c.catalyst_detail})')
            elif c.catalyst == 'ma':
                c.score = -999
                self._add_message('catalyst',
                    f'{c.symbol}: SKIP — M&A gap ({c.catalyst_detail})')
                continue
            elif c.catalyst == 'fda':
                c.score -= 20
                self._add_message('catalyst',
                    f'{c.symbol}: -20 penalty (FDA: {c.catalyst_detail})')
            elif c.catalyst in ('upgrade', 'downgrade'):
                c.score -= self.config.catalyst_news_penalty
                self._add_message('catalyst',
                    f'{c.symbol}: -{self.config.catalyst_news_penalty} penalty '
                    f'({c.catalyst}: {c.catalyst_detail})')
            elif c.catalyst == 'offering':
                c.score += self.config.catalyst_noise_bonus
                self._add_message('catalyst',
                    f'{c.symbol}: +{self.config.catalyst_noise_bonus} bonus '
                    f'(offering = bearish catalyst)')
            else:
                # No catalyst = clean noise gap
                c.score += self.config.catalyst_noise_bonus
            filtered.append(c)
        self.candidates = filtered

    # -------------------------------------------------------------------------
    # Entry Flow — with fill verification
    # -------------------------------------------------------------------------

    async def _enter_positions(self, llm_symbols=None, size_mult=1.0):
        """Enter positions on top candidates with fill verification.

        Args:
            llm_symbols: if set, only enter these symbols (LLM-selected, in order)
            size_mult: scale position size by this factor (0.5-1.0, LLM confidence)
        """
        self.status = 'trading'
        await broadcast({'type': 'live_status', 'status': 'trading'})

        # Market regime filter (live)
        regime = await fetch_market_regime_live(self.config)
        if regime and regime.position_reduction == -999:
            self._add_message('regime', f'BLOCKED: {regime.note}')
            self.status = 'regime_blocked'
            return
        if regime and regime.position_reduction == -1:
            self._add_message('regime', f'HALVED: {regime.note}')

        # Apply LLM symbol filter if provided
        candidates_to_trade = self.candidates
        if llm_symbols:
            sym_set = set(llm_symbols)
            candidates_to_trade = [c for c in self.candidates if c.symbol in sym_set]
            sym_order = {s: i for i, s in enumerate(llm_symbols)}
            candidates_to_trade.sort(key=lambda c: sym_order.get(c.symbol, 999))
            self._add_message('llm', f'Rudra selected {len(candidates_to_trade)} symbols: {llm_symbols}')

        # Thin day logic: if fewer candidates than threshold, trade all of them
        eff_max = self.engine.effective_max_positions(len(candidates_to_trade))
        if regime and regime.position_reduction == -1:
            eff_max = max(1, eff_max // 2)
        self.engine._effective_max_positions = eff_max
        self._add_message('entry',
            f'{len(candidates_to_trade)} candidates, trading top {eff_max}'
            + (' (thin day — trading all)' if eff_max > self.config.max_positions else '')
            + (f' ({regime.note})' if regime and regime.note else '')
            + (f' (size x{size_mult:.1f})' if size_mult != 1.0 else ''))

        symbols_entered = []
        for candidate in candidates_to_trade[:eff_max]:
            async with self._position_lock:
                ok, reason = self.engine.should_enter(candidate)
                if not ok:
                    self._add_message('skip', f'Skipping {candidate.symbol}: {reason}')
                    if 'halted' in reason or 'daily loss' in reason or 'drawdown' in reason:
                        await self.alerter.send('Circuit Breaker',
                            f'Entries blocked: {reason}',
                            level='warning', throttle_key='circuit_breaker')
                    continue

                # Strategy plugin: should_enter_now gate
                if self.strategy:
                    tick_data = self.indicator_engine.get_data(candidate.symbol) if self.indicator_engine else None
                    strat_ok, strat_reason = self.strategy.should_enter_now(
                        asdict(candidate), candidate.premarket_price, tick_data, datetime.now(ET))
                    if not strat_ok:
                        self._add_message('strategy',
                            f'Strategy skip {candidate.symbol}: {strat_reason}')
                        continue

                entry_price = candidate.premarket_price
                entry_time = datetime.now(ET).strftime('%Y-%m-%d %H:%M')
                pos = self.engine.open_position(candidate, entry_price, entry_time)
                if pos is None:
                    continue

                # Tag position with strategy
                pos.strategy_id = self.config.active_strategy

                # LLM size adjustment
                if size_mult != 1.0 and pos.shares > 1:
                    pos.shares = max(1, int(pos.shares * size_mult))
                    pos.remaining_shares = pos.shares

            # Submit order and wait for fill (outside lock to not block other ops)
            direction = candidate.direction
            entry_side = 'buy' if direction == 'long' else 'sell'
            side_label = 'LONG' if direction == 'long' else 'SHORT'

            if self.config.limit_orders_only:
                if direction == 'long':
                    limit_px = round(entry_price * (1 + self.config.limit_offset_pct), 2)
                else:
                    limit_px = round(entry_price * (1 - self.config.limit_offset_pct), 2)
                order_type = 'limit'
                order_label = f'LIMIT @ ${limit_px:.2f}'
            else:
                limit_px = None
                order_type = 'market'
                order_label = 'MARKET'

            self._add_message('entry',
                f'Submitting {side_label} {pos.shares} {candidate.symbol} {order_label}...')

            fill = await alpaca_submit_and_confirm(
                candidate.symbol, pos.shares, entry_side,
                order_type=order_type, limit_price=limit_px,
                timeout_sec=15.0 if order_type == 'limit' else 10.0,
            )

            async with self._position_lock:
                if fill.is_filled:
                    # P0-3: Use actual fill price for position tracking
                    pos.entry_fill_price = fill.filled_avg_price
                    pos.entry_order_id = fill.order_id
                    # Recalculate stops/targets based on actual fill price
                    if fill.filled_avg_price > 0:
                        eff_stop_pct = compute_adaptive_stop_pct(self.config, candidate.gap_pct)
                        pos.entry_price = fill.filled_avg_price
                        if direction == 'long':
                            pos.stop_price = fill.filled_avg_price * (1 - eff_stop_pct)
                        else:
                            pos.stop_price = fill.filled_avg_price * (1 + eff_stop_pct)
                        pos.half_target = (fill.filled_avg_price + candidate.prev_close) / 2

                        # Strategy plugin: override stop and targets
                        if self.strategy:
                            tick_data = self.indicator_engine.get_data(candidate.symbol) if self.indicator_engine else None
                            strat_stop = self.strategy.compute_stop_price(
                                fill.filled_avg_price, asdict(candidate), tick_data)
                            if strat_stop is not None:
                                pos.stop_price = strat_stop
                            strat_targets = self.strategy.compute_targets(
                                fill.filled_avg_price, asdict(candidate), tick_data)
                            if strat_targets is not None:
                                pos.half_target, pos.full_target = strat_targets

                    # Place broker-side stop order immediately
                    stop_result = await asyncio.to_thread(
                        alpaca_place_stop_order, candidate.symbol,
                        fill.filled_qty, pos.stop_price, 0.003, direction
                    )
                    if 'error' not in stop_result:
                        pos.stop_order_id = stop_result.get('id', '')
                        self._add_message('entry',
                            f'Broker stop set: {candidate.symbol} @ ${pos.stop_price:.2f}')
                    else:
                        logger.warning(f"Broker stop failed for {candidate.symbol}: "
                                      f"{stop_result['error']} — using software stop")

                    symbols_entered.append(candidate.symbol)
                    self._add_message('entry',
                        f'FILLED {side_label} {fill.filled_qty} {candidate.symbol} '
                        f'@ ${fill.filled_avg_price:.2f} {order_label} '
                        f'(gap {candidate.gap_pct:.1%}, score {candidate.score:.0f})', {
                        'symbol': candidate.symbol,
                        'shares': fill.filled_qty,
                        'entry_price': fill.filled_avg_price,
                        'stop': pos.stop_price,
                        'target': pos.full_target,
                    })
                    await self.alerter.send('Entry Fill',
                        f'{side_label} {fill.filled_qty} {candidate.symbol} @ ${fill.filled_avg_price:.2f}\n'
                        f'Gap: {candidate.gap_pct:.1%} | Stop: ${pos.stop_price:.2f}',
                        level='trade')
                    # Inject entry event into LLM conversation (Tier 2 — batched)
                    self._add_event(
                        f'Entered {candidate.symbol} {direction} {fill.filled_qty} shares '
                        f'@ ${fill.filled_avg_price:.2f} (gap {candidate.gap_pct:.1%})',
                        tier=2)
                    # Journal: log entry action
                    self.journal.log('action', 'rules_engine',
                        f'ENTRY {side_label} {fill.filled_qty} {candidate.symbol} '
                        f'@ ${fill.filled_avg_price:.2f} (gap {candidate.gap_pct:.1%})',
                        symbol=candidate.symbol, data={
                            'direction': direction, 'shares': fill.filled_qty,
                            'fill_price': fill.filled_avg_price,
                            'gap_pct': candidate.gap_pct, 'stop': pos.stop_price})
                elif fill.status == 'partially_filled' and fill.filled_qty > 0:
                    # Partial fill — adjust position to actual filled quantity
                    pos.shares = fill.filled_qty
                    pos.remaining_shares = fill.filled_qty
                    pos.entry_fill_price = fill.filled_avg_price
                    pos.entry_order_id = fill.order_id
                    if fill.filled_avg_price > 0:
                        eff_stop_pct = compute_adaptive_stop_pct(self.config, candidate.gap_pct)
                        pos.entry_price = fill.filled_avg_price
                        if direction == 'long':
                            pos.stop_price = fill.filled_avg_price * (1 - eff_stop_pct)
                        else:
                            pos.stop_price = fill.filled_avg_price * (1 + eff_stop_pct)
                        pos.half_target = (fill.filled_avg_price + candidate.prev_close) / 2

                    # Place broker-side stop for partial fill too
                    stop_result = await asyncio.to_thread(
                        alpaca_place_stop_order, candidate.symbol,
                        fill.filled_qty, pos.stop_price, 0.003, direction
                    )
                    if 'error' not in stop_result:
                        pos.stop_order_id = stop_result.get('id', '')

                    symbols_entered.append(candidate.symbol)
                    self._add_message('warning',
                        f'PARTIAL FILL {fill.filled_qty}/{pos.shares} {candidate.symbol} '
                        f'@ ${fill.filled_avg_price:.2f}')
                elif fill.status == 'timeout' and order_type == 'limit':
                    # Limit order timed out — retry with market order
                    self._add_message('warning',
                        f'{candidate.symbol} limit timed out, retrying MARKET...')
                    fill = await alpaca_submit_and_confirm(
                        candidate.symbol, pos.shares, entry_side,
                        order_type='market', timeout_sec=10.0,
                    )
                    if fill.is_filled:
                        pos.entry_fill_price = fill.filled_avg_price
                        pos.entry_order_id = fill.order_id
                        if fill.filled_avg_price > 0:
                            eff_stop_pct = compute_adaptive_stop_pct(self.config, candidate.gap_pct)
                            pos.entry_price = fill.filled_avg_price
                            if direction == 'long':
                                pos.stop_price = fill.filled_avg_price * (1 - eff_stop_pct)
                            else:
                                pos.stop_price = fill.filled_avg_price * (1 + eff_stop_pct)
                            pos.half_target = (fill.filled_avg_price + candidate.prev_close) / 2
                        stop_result = await asyncio.to_thread(
                            alpaca_place_stop_order, candidate.symbol,
                            fill.filled_qty, pos.stop_price, 0.003, direction
                        )
                        if 'error' not in stop_result:
                            pos.stop_order_id = stop_result.get('id', '')
                            self._add_message('entry',
                                f'Broker stop set: {candidate.symbol} @ ${pos.stop_price:.2f}')
                        symbols_entered.append(candidate.symbol)
                        self._add_message('entry',
                            f'FILLED {side_label} {fill.filled_qty} {candidate.symbol} '
                            f'@ ${fill.filled_avg_price:.2f} MARKET (retry) '
                            f'(gap {candidate.gap_pct:.1%}, score {candidate.score:.0f})')
                    else:
                        self._add_message('error',
                            f'Entry MARKET retry also FAILED for {candidate.symbol}: {fill.error}')
                        if candidate.symbol in self.engine.positions:
                            del self.engine.positions[candidate.symbol]
                else:
                    # Order failed/rejected — remove position from internal state
                    self._add_message('error',
                        f'Entry order FAILED for {candidate.symbol}: {fill.error}')
                    if candidate.symbol in self.engine.positions:
                        del self.engine.positions[candidate.symbol]

        # Start or reuse tick streamer for entered symbols
        if symbols_entered:
            if self.streamer and self.streamer.connected:
                # Reuse existing connection — dynamically subscribe to new symbols
                await self.streamer.add_symbols(symbols_entered)
            else:
                # No active streamer — stop stale one if any, then create new
                if self.streamer:
                    await self.streamer.stop()
                self.streamer = AlpacaTickStreamer(symbols_entered, on_tick=self._on_tick)
                await self.streamer.start()

        await broadcast({
            'type': 'positions_update',
            'positions': {s: asdict(p) for s, p in self.engine.positions.items()},
        })
        self._save_state()

    # -------------------------------------------------------------------------
    # Exit Flow — evaluate → place order → confirm fill → update state
    # -------------------------------------------------------------------------

    async def _execute_exit(self, symbol: str, exit_signal: dict, price: float) -> Optional[TradeRecord]:
        """Execute a single exit: cancel broker stop, place cover order, wait for fill, confirm.

        Must be called with _position_lock already held.
        Returns the TradeRecord if successful, None if order failed.
        """
        pos = self.engine.positions.get(symbol)
        if pos is None or pos.closing:
            return None

        reason = exit_signal['reason']
        shares = exit_signal['shares']
        is_eod = reason == 'eod'
        is_stop = reason == 'stop'

        # Mark position as closing to prevent re-entry
        pos.closing = True

        # Cancel broker-side stop order before placing a new cover order
        # (except for stop exits where the broker stop may have already filled)
        if pos.stop_order_id and not is_stop:
            await asyncio.to_thread(alpaca_cancel_order, pos.stop_order_id)
            pos.stop_order_id = ''

        # Direction-aware exit side: sell to close longs, buy to cover shorts
        direction = pos.direction
        exit_side = 'sell' if direction == 'long' else 'buy'

        # Choose order type: EOD always market for guaranteed execution
        if is_eod:
            order_type = 'market'
            limit_px = None
        elif self.config.limit_orders_only:
            if direction == 'long':
                limit_px = round(price * (1 - self.config.limit_offset_pct), 2)
            else:
                limit_px = round(price * (1 + self.config.limit_offset_pct), 2)
            order_type = 'limit'
        else:
            order_type = 'market'
            limit_px = None

        fill = await alpaca_submit_and_confirm(
            symbol, shares, exit_side,
            order_type=order_type, limit_price=limit_px,
            timeout_sec=10.0 if order_type == 'limit' else 15.0,
        )

        # Market fallback: if limit cover timed out, retry with market order
        if fill.status == 'timeout' and order_type == 'limit':
            self._add_message('warning',
                f'{symbol} limit cover timed out, retrying MARKET...')
            await asyncio.sleep(1)  # brief pause for broker to clear the cancel hold
            fill = await alpaca_submit_and_confirm(
                symbol, shares, exit_side,
                order_type='market', timeout_sec=15.0,
            )

        if fill.is_filled:
            now = datetime.now(ET)
            trade = self.engine.confirm_exit(
                symbol, fill.filled_qty, fill.filled_avg_price, reason, now
            )
            if trade:
                self._add_message('exit',
                    f'{reason.upper()} {fill.filled_qty} {symbol} '
                    f'@ ${fill.filled_avg_price:.2f} P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.1%})')
                # Invalidate lessons cache so next LLM call includes this trade
                self._lessons_cache = None
                # Inject event into LLM conversation
                event_tier = 3 if is_stop and trade.pnl < 0 else 2
                self._add_event(
                    f'{symbol} {reason}: {fill.filled_qty} shares @ ${fill.filled_avg_price:.2f}, '
                    f'P&L ${trade.pnl:+,.2f} ({trade.pnl_pct:+.1%})',
                    tier=event_tier)
                pnl_sign = '+' if trade.pnl >= 0 else ''
                await self.alerter.send(f'Exit: {symbol}',
                    f'{reason.upper()} {fill.filled_qty} {symbol} @ ${fill.filled_avg_price:.2f}\n'
                    f'P&L: {pnl_sign}${trade.pnl:.2f} ({trade.pnl_pct:+.1%})',
                    level='trade')
                # Journal: log exit action
                self.journal.log('action', 'rules_engine',
                    f'EXIT {reason}: {fill.filled_qty} {symbol} @ ${fill.filled_avg_price:.2f}, '
                    f'P&L ${trade.pnl:+,.2f} ({trade.pnl_pct:+.1%})',
                    symbol=symbol, data={'pnl': trade.pnl, 'reason': reason,
                        'fill_price': fill.filled_avg_price, 'shares': fill.filled_qty})

            # If partial profit exit, replace broker stop at breakeven
            if reason == 'partial' and symbol in self.engine.positions:
                new_pos = self.engine.positions[symbol]
                new_stop_result = await alpaca_replace_stop_order(
                    '', symbol, new_pos.remaining_shares, new_pos.stop_price
                )
                if 'error' not in new_stop_result:
                    new_pos.stop_order_id = new_stop_result.get('id', '')
                    self._add_message('info',
                        f'Stop moved to breakeven: {symbol} @ ${new_pos.stop_price:.2f}')

            return trade

        elif fill.status == 'partially_filled' and fill.filled_qty > 0:
            now = datetime.now(ET)
            trade = self.engine.confirm_exit(
                symbol, fill.filled_qty, fill.filled_avg_price, reason, now
            )
            if trade:
                self._add_message('warning',
                    f'PARTIAL COVER {fill.filled_qty}/{shares} {symbol} '
                    f'@ ${fill.filled_avg_price:.2f}')
            return trade

        else:
            # Order failed — clear closing flag so we can retry
            if symbol in self.engine.positions:
                self.engine.positions[symbol].closing = False
            self._add_message('error',
                f'Cover order FAILED for {symbol}: {fill.error} — will retry')
            logger.error(f"Cover order failed: {symbol} {shares} shares: {fill.error}")
            return None

    async def _execute_scheduled_reflection(self):
        """Run periodic LLM reflection with enriched context. Replaces autonomous_review.

        Triggers on:
        - Time interval (900s / 15 min)
        - Event accumulation (3+ pending events)
        Both subject to LLM call budget.
        """
        if not self.llm_supervisor or not self.llm_supervisor.is_available():
            return
        if not self.engine.positions:
            return

        now_mono = _time.monotonic()
        _time_trigger = (now_mono - self._last_autonomous_review >= self._AUTONOMOUS_REVIEW_INTERVAL)
        _event_trigger = (self._event_bus.pending_count() >= 3)
        if not _time_trigger and not _event_trigger:
            return

        self._last_autonomous_review = now_mono

        now = datetime.now(ET)
        state = self._build_llm_state(now)

        # Enrich state with indicator snapshot
        indicator_snapshot = ''
        if self.indicator_engine and self.engine.positions:
            lines = []
            for sym in self.engine.positions:
                td = self.indicator_engine.get_data(sym)
                if td:
                    vwap = td.get('vwap', 0)
                    ema = td.get('ema', 0)
                    px = self.streamer.latest_prices.get(sym, 0) if self.streamer else 0
                    if vwap or ema:
                        lines.append(f'{sym}: price=${px:.2f} VWAP=${vwap:.2f} EMA=${ema:.2f}')
            if lines:
                indicator_snapshot = '\nIndicators:\n' + '\n'.join(lines)

        # Enrich with journal summary
        journal_summary = self.journal.get_summary(minutes=20)

        state['indicator_snapshot'] = indicator_snapshot
        state['journal_summary'] = journal_summary

        trigger_reason = 'event_accumulation' if _event_trigger else 'scheduled_interval'
        logger.info(f"Scheduled reflection starting ({len(self.engine.positions)} positions, "
                    f"trigger={trigger_reason})")
        self.journal.log('observation', 'scheduled_reflection',
            f'Reflection triggered ({trigger_reason}), {len(self.engine.positions)} positions')

        try:
            review = await self.llm_supervisor.autonomous_review(
                state, event_queue=self._event_bus)
        except Exception as e:
            logger.warning(f"Scheduled reflection failed: {e}")
            return

        actions = review.get('actions', [])

        # Show the review exchange in activity feed
        n_pos = len(self.engine.positions)
        review_analysis = review.get('_analysis', '')

        # Parse ReAct response for journal
        react = self.llm_supervisor._parse_react_response(review_analysis)
        if react['observe']:
            self.journal.log('observation', 'llm_rudra', react['observe'], llm_call=True)
        if react['think']:
            self.journal.log('reasoning', 'llm_rudra', react['think'], llm_call=True)

        if actions:
            action_summary = ', '.join(
                f"{a.get('action', '?')} {a.get('symbol', '')}" for a in actions[:3])
            rudra_text = review_analysis if review_analysis else action_summary
            self._add_message('conversation',
                f'BOT: Reflection ({n_pos} positions) | RUDRA: [{action_summary}] {rudra_text[:400]}')
            await broadcast({'type': 'conversation',
                'speaker': 'rudra', 'text': f'[{action_summary}] {rudra_text[:500]}'})
            self.journal.log('action', 'llm_rudra',
                f'Reflection actions: {action_summary}', llm_call=True)
        else:
            rudra_text = review_analysis if review_analysis else 'All looks good, no changes needed.'
            self._add_message('conversation',
                f'BOT: Reflection ({n_pos} positions) | RUDRA: {rudra_text[:400]}')
            self.journal.log('no_action', 'llm_rudra',
                f'Reflection: {rudra_text[:200]}', llm_call=True)
            logger.info("Scheduled reflection: no actions recommended")
            return

        executed = []
        for rec in actions:
            action = rec.get('action', 'none')
            symbol = rec.get('symbol', '').upper()
            reasoning = rec.get('reasoning', '')

            if action == 'close' and symbol:
                pos = self.engine.positions.get(symbol)
                if pos and not pos.closing:
                    # Get current price
                    price = 0.0
                    if self.streamer and self.streamer.latest_prices:
                        price = self.streamer.latest_prices.get(symbol, 0.0)
                    if price <= 0:
                        price = pos.entry_price
                    exit_signal = {
                        'reason': 'llm_profit',
                        'shares': pos.remaining_shares,
                        'is_full_close': True,
                        'trigger_price': price,
                    }
                    async with self._position_lock:
                        trade = await self._execute_exit(symbol, exit_signal, price)
                    if trade:
                        self._add_message('llm_review',
                            f'AUTO-CLOSE {symbol}: ${trade.pnl:+,.2f} — {reasoning}')
                        executed.append(f'close {symbol}')
                        await broadcast({'type': 'trade', 'trades': [asdict(trade)]})

            elif action == 'tighten_stop' and symbol:
                new_stop = rec.get('new_stop')
                if new_stop:
                    pos = self.engine.positions.get(symbol)
                    if pos:
                        old_stop = pos.stop_price
                        async with self._position_lock:
                            pos = self.engine.positions.get(symbol)
                            if pos:
                                pos.stop_price = float(new_stop)
                                # Update broker stop
                                if pos.stop_order_id:
                                    try:
                                        await asyncio.to_thread(alpaca_cancel_order, pos.stop_order_id)
                                        stop_result = await asyncio.to_thread(
                                            alpaca_place_stop_order, symbol,
                                            pos.remaining_shares, float(new_stop),
                                            0.003, pos.direction)
                                        if 'error' not in stop_result:
                                            pos.stop_order_id = stop_result.get('id', '')
                                    except Exception as e:
                                        logger.warning(f"Broker stop update failed: {e}")
                        self._add_message('llm_review',
                            f'AUTO-TIGHTEN {symbol} stop: ${old_stop:.2f} → ${new_stop:.2f} — {reasoning}')
                        executed.append(f'tighten {symbol}')

            elif action == 'adjust_config':
                key = rec.get('config_key')
                value = rec.get('config_value')
                if key and value is not None and hasattr(self.config, key):
                    old_val = getattr(self.config, key)
                    try:
                        setattr(self.config, key, type(old_val)(value))
                        self._add_message('llm_review',
                            f'AUTO-CONFIG {key}: {old_val} → {value} — {reasoning}')
                        executed.append(f'config {key}')
                    except (ValueError, TypeError):
                        pass

        if executed:
            self._save_state()
            logger.info(f"Autonomous review executed: {', '.join(executed)}")
            await broadcast({
                'type': 'positions_update',
                'positions': {s: asdict(p) for s, p in self.engine.positions.items()},
                'stats': asdict(self.engine.daily_stats),
                'equity': self.engine.equity,
            })

    async def _check_broker_stops(self):
        """Check if any broker-side stop orders have been filled.

        This is the primary stop-loss mechanism — Alpaca triggers the stop
        server-side with zero latency. We just need to detect the fill and
        update internal state.
        """
        for sym in list(self.engine.positions.keys()):
            pos = self.engine.positions.get(sym)
            if not pos or not pos.stop_order_id or pos.closing:
                continue

            order_data = await asyncio.to_thread(alpaca_get_order, pos.stop_order_id)
            if order_data is None:
                continue

            status = order_data.get('status', '')

            if status == 'filled':
                fill_qty = int(order_data.get('filled_qty', pos.remaining_shares))
                fill_price = float(order_data.get('filled_avg_price', pos.stop_price))

                logger.info(f"Broker stop FILLED: {sym} {fill_qty} shares @ ${fill_price:.2f}")
                now = datetime.now(ET)
                trade = self.engine.confirm_exit(sym, fill_qty, fill_price, 'stop', now)
                pos.stop_order_id = ''

                if trade:
                    self._add_message('exit',
                        f'STOP (broker) {fill_qty} {sym} '
                        f'@ ${fill_price:.2f} P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.1%})')
                    await broadcast({'type': 'trade', 'trades': [asdict(trade)]})
                    await broadcast({
                        'type': 'positions_update',
                        'positions': {s: asdict(p) for s, p in self.engine.positions.items()},
                    })
                    self._save_state()

    async def _on_tick(self, symbol: str, price: float, size: int = 0):
        """Tick callback from Alpaca stream — evaluate non-stop exits under lock.

        Stop losses are handled by broker-side stop orders (zero latency).
        This only evaluates: partial target, full target, time exit,
        and strategy-specific exits (EMA cross, trailing stop).
        Also broadcasts tracker_tick for the position tracker UI.
        """
        # Feed indicator engine (if active)
        if self.indicator_engine and size > 0:
            self.indicator_engine.on_tick(symbol, price, size)

        # Broadcast to position tracker (even when trading is paused)
        if symbol in _tracker_watchlist:
            await broadcast({
                'type': 'tracker_tick',
                'symbol': symbol,
                'price': price,
                'timestamp': _time.time(),
            })

        if self.status != 'trading':
            return

        async with self._position_lock:
            if symbol not in self.engine.positions:
                return

            pos = self.engine.positions[symbol]
            now = datetime.now(ET)
            pos.update_tracking(price, now.strftime('%H:%M:%S'))

            # Event detection (pure arithmetic, <1ms)
            _tick_data = self.indicator_engine.get_data(symbol) if self.indicator_engine else None
            tick_events = self.event_detector.detect_tick_events(
                symbol, price, size, asdict(pos), _tick_data)
            if tick_events:
                self._event_bus.push_many(tick_events)
                for te in tick_events:
                    self.journal.log('observation', 'event_detector',
                        te.description, symbol=te.symbol, data=te.data)

            # Strategy-specific trailing stop update
            if self.strategy:
                tick_data = self.indicator_engine.get_data(symbol) if self.indicator_engine else None
                new_stop = self.strategy.update_trailing_stop(
                    asdict(pos), price, tick_data, now)
                if new_stop is not None:
                    old_stop = pos.stop_price
                    pos.stop_price = new_stop
                    if abs(new_stop - old_stop) > 0.001:
                        self._add_message('strategy',
                            f'Trailing stop {symbol}: ${old_stop:.2f} -> ${new_stop:.2f}')
                        # Update broker stop if we have one
                        if pos.stop_order_id:
                            try:
                                await asyncio.to_thread(alpaca_cancel_order, pos.stop_order_id)
                                stop_result = await asyncio.to_thread(
                                    alpaca_place_stop_order, symbol,
                                    pos.remaining_shares, new_stop,
                                    0.003, pos.direction)
                                if 'error' not in stop_result:
                                    pos.stop_order_id = stop_result.get('id', '')
                            except Exception as e:
                                logger.warning(f"Broker trailing stop update failed for {symbol}: {e}")

            # Strategy-specific exit check (before engine exits)
            exit_signal = None
            if self.strategy:
                tick_data = self.indicator_engine.get_data(symbol) if self.indicator_engine else None
                strat_exit = self.strategy.evaluate_exit(
                    asdict(pos), price, price, tick_data, now)
                if strat_exit is not None:
                    exit_signal = {
                        'reason': strat_exit.reason,
                        'shares': strat_exit.shares or pos.remaining_shares,
                        'is_full_close': True,
                        'trigger_price': price,
                    }

            # Engine exit check (stop, partial, full target, time)
            if exit_signal is None:
                exit_signal = self.engine.evaluate_exit(symbol, price, price, now)
            if exit_signal is None:
                return

            # Skip stop signals — broker handles those
            if exit_signal['reason'] == 'stop' and pos.stop_order_id:
                return

            trade = await self._execute_exit(symbol, exit_signal, price)

        if trade:
            await broadcast({
                'type': 'trade',
                'trades': [asdict(trade)],
            })
            await broadcast({
                'type': 'positions_update',
                'positions': {s: asdict(p) for s, p in self.engine.positions.items()},
            })
            self._save_state()

    _spy_cache: Tuple[float, float] = (0.0, 0.0)  # (timestamp, change_pct)
    _SPY_CACHE_TTL = 60.0  # refresh SPY every 60 seconds
    _LLM_PROFIT_INTERVAL = 60.0  # seconds between LLM profit evals per position

    async def _get_spy_change(self) -> float:
        """Get SPY intraday change %. Cached for 60s."""
        now_mono = _time.monotonic()
        if now_mono - self._spy_cache[0] < self._SPY_CACHE_TTL:
            return self._spy_cache[1]
        try:
            snaps = await asyncio.to_thread(fetch_alpaca_snapshots, ['SPY'])
            spy = snaps.get('SPY', {})
            daily = spy.get('dailyBar', {})
            prev = spy.get('prevDailyBar', {})
            if daily.get('c') and prev.get('c'):
                change = (daily['c'] - prev['c']) / prev['c']
                self._spy_cache = (now_mono, change)
                return change
        except Exception:
            pass
        return self._spy_cache[1]

    async def _check_positions(self):
        """Periodic position check: poll broker stops + evaluate non-stop exits + LLM profit."""
        if not self.engine.positions:
            return

        # Check if any broker-side stop orders have filled
        await self._check_broker_stops()

        now = datetime.now(ET)
        now_str = now.strftime('%H:%M:%S')
        now_mono = _time.monotonic()
        trades_executed = []

        # Fetch SPY once per cycle (cached)
        spy_change = 0.0
        if self.llm_supervisor and self.llm_supervisor.is_available():
            spy_change = await self._get_spy_change()

        # Get prices: prefer streamer, fall back to REST snapshots
        prices: dict = {}
        if self.streamer and self.streamer.latest_prices:
            prices = dict(self.streamer.latest_prices)
        if not prices and self.engine.positions:
            # REST fallback when WebSocket is down
            try:
                syms = list(self.engine.positions.keys())
                snaps = await asyncio.to_thread(fetch_alpaca_snapshots, syms)
                for sym, snap in snaps.items():
                    lt = snap.get('latestTrade', {})
                    if lt.get('p'):
                        prices[sym] = lt['p']
                if prices:
                    logger.debug(f"Using REST snapshot prices for {len(prices)} positions")
            except Exception as e:
                logger.warning(f"REST snapshot fallback failed: {e}")

        if prices:
            for sym in list(self.engine.positions.keys()):
                price = prices.get(sym)
                if not price:
                    continue

                async with self._position_lock:
                    pos = self.engine.positions.get(sym)
                    if not pos:
                        continue

                    pos.update_tracking(price, now_str)

                    # Strategy plugin: trailing stop update
                    if self.strategy and self.indicator_engine:
                        tick_data = self.indicator_engine.get_data(sym)
                        new_stop = self.strategy.update_trailing_stop(
                            asdict(pos), price, tick_data, now)
                        if new_stop is not None:
                            old_stop = pos.stop_price
                            pos.stop_price = new_stop
                            if abs(new_stop - old_stop) > 0.001:
                                self._add_message('strategy',
                                    f'Trailing stop {sym}: ${old_stop:.2f} -> ${new_stop:.2f}')
                                if pos.stop_order_id:
                                    try:
                                        await asyncio.to_thread(alpaca_cancel_order, pos.stop_order_id)
                                        stop_result = await asyncio.to_thread(
                                            alpaca_place_stop_order, sym,
                                            pos.remaining_shares, new_stop,
                                            0.003, pos.direction)
                                        if 'error' not in stop_result:
                                            pos.stop_order_id = stop_result.get('id', '')
                                    except Exception as e:
                                        logger.warning(f"Broker trailing stop update failed for {sym}: {e}")

                    # Strategy plugin: evaluate exit (before engine)
                    exit_signal = None
                    if self.strategy and self.indicator_engine:
                        tick_data = self.indicator_engine.get_data(sym)
                        strat_exit = self.strategy.evaluate_exit(
                            asdict(pos), price, price, tick_data, now)
                        if strat_exit is not None:
                            exit_signal = {
                                'reason': strat_exit.reason,
                                'shares': strat_exit.shares or pos.remaining_shares,
                                'is_full_close': True,
                                'trigger_price': price,
                            }

                    # Engine exit check (stop, partial, full target, time)
                    if exit_signal is None:
                        exit_signal = self.engine.evaluate_exit(sym, price, price, now)

                    # No exit signal from rules — check LLM for profit-taking
                    if exit_signal is None:
                        if (self.llm_supervisor and self.llm_supervisor.is_available()
                                and not pos.closing
                                and now_mono - pos.last_llm_profit_check >= self._LLM_PROFIT_INTERVAL):
                            # Minimum hold time gate — let positions develop
                            _skip_llm_profit = False
                            try:
                                _entry_dt = datetime.strptime(pos.entry_time, '%Y-%m-%d %H:%M')
                                if now.tzinfo and not _entry_dt.tzinfo:
                                    _entry_dt = _entry_dt.replace(tzinfo=now.tzinfo)
                                _held_min = (now - _entry_dt).total_seconds() / 60
                                if _held_min < self.config.min_hold_minutes:
                                    _skip_llm_profit = True
                            except (ValueError, TypeError):
                                pass

                            # Minimum profit / gap fill gate
                            if not _skip_llm_profit:
                                if pos.direction == 'short':
                                    _pnl_pct = (pos.entry_price - price) / pos.entry_price if pos.entry_price > 0 else 0
                                else:
                                    _pnl_pct = (price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0
                                _gap_fill = pos.gap_fill_pct(price)
                                if _pnl_pct < self.config.min_profit_take_pct and _gap_fill < self.config.min_gap_fill_pct:
                                    _skip_llm_profit = True

                            # Only ask LLM if position is in profit and gates pass
                            if pos.direction == 'short':
                                in_profit = price < pos.entry_price
                            else:
                                in_profit = price > pos.entry_price
                            if in_profit and not _skip_llm_profit:
                                pos.last_llm_profit_check = now_mono
                                llm_profit = await self.llm_supervisor.evaluate_profit(
                                    asdict(pos), price, spy_change,
                                    self.engine.daily_stats.pnl)
                                action = llm_profit.get('action', 'hold')
                                reasoning = llm_profit.get('reasoning', '')
                                if action == 'close':
                                    pnl_pct = pos.gap_fill_pct(price)
                                    self._add_message('conversation',
                                        f'BOT: {sym} profit check ({pnl_pct:.0%} fill) | '
                                        f'RUDRA: Take profit — {reasoning}')
                                    exit_signal = {
                                        'action': 'close', 'reason': 'llm_profit',
                                        'shares': pos.remaining_shares,
                                    }
                                elif action == 'tighten_stop':
                                    new_stop = llm_profit.get('new_stop', pos.stop_price)
                                    if new_stop and new_stop != pos.stop_price:
                                        old_stop = pos.stop_price
                                        pos.stop_price = new_stop
                                        self._add_message('conversation',
                                            f'BOT: {sym} profit check | '
                                            f'RUDRA: Tighten stop ${old_stop:.2f} → ${new_stop:.2f} — {reasoning}')
                                        # Update broker stop if we have one
                                        if pos.stop_order_id:
                                            try:
                                                await asyncio.to_thread(alpaca_cancel_order, pos.stop_order_id)
                                                stop_result = await asyncio.to_thread(
                                                    alpaca_place_stop_order, sym,
                                                    pos.remaining_shares, new_stop,
                                                    0.003, pos.direction)
                                                if 'error' not in stop_result:
                                                    pos.stop_order_id = stop_result.get('id', '')
                                            except Exception as e:
                                                logger.warning(f"Broker stop update failed for {sym}: {e}")
                                # else: hold — do nothing

                    if exit_signal is None:
                        continue

                    # Skip stop signals — broker handles those
                    if exit_signal['reason'] == 'stop' and pos.stop_order_id:
                        continue

                    # LLM exit override (software stops only, not broker/EOD/time)
                    if (self.llm_supervisor and self.llm_supervisor.is_available()
                            and exit_signal['reason'] == 'stop'
                            and not pos.stop_order_id
                            and pos.llm_hold_overrides < self.config.llm_max_hold_overrides):
                        llm_exit = await self.llm_supervisor.evaluate_exit(
                            asdict(pos), price, exit_signal)
                        llm_action = llm_exit.get('action', 'close')
                        if llm_action == 'hold':
                            pos.llm_hold_overrides += 1
                            self._add_message('conversation',
                                f'BOT: {sym} hitting stop | '
                                f'RUDRA: Hold ({pos.llm_hold_overrides}/'
                                f'{self.config.llm_max_hold_overrides}) — '
                                f'{llm_exit.get("reasoning", "")}')
                            continue
                        elif llm_action == 'tighten':
                            pos.llm_hold_overrides += 1
                            new_stop = llm_exit.get('new_stop', pos.stop_price)
                            self._add_message('conversation',
                                f'BOT: {sym} hitting stop | '
                                f'RUDRA: Tighten ${pos.stop_price:.2f} → ${new_stop:.2f} '
                                f'({pos.llm_hold_overrides}/{self.config.llm_max_hold_overrides}) — '
                                f'{llm_exit.get("reasoning", "")}')
                            pos.stop_price = new_stop
                            continue

                    trade = await self._execute_exit(sym, exit_signal, price)

                if trade:
                    trades_executed.append(trade)

        if trades_executed:
            await broadcast({
                'type': 'trade',
                'trades': [asdict(t) for t in trades_executed],
            })
            self._save_state()

        # Portfolio-level event detection (every 15s cycle)
        pos_dicts = {s: asdict(p) for s, p in self.engine.positions.items()}
        portfolio_events = self.event_detector.detect_position_events(
            pos_dicts, self.engine.daily_stats, self.engine.equity, prices or {})
        if portfolio_events:
            self._event_bus.push_many(portfolio_events)
            for pe in portfolio_events:
                self.journal.log('observation', 'event_detector',
                    pe.description, symbol=pe.symbol, data=pe.data)

        # Check for re-entry opportunities on stopped-out symbols
        if self.config.reentry_enabled and self.engine._stopped_today:
            now = datetime.now(ET)
            for sym, rec in list(self.engine._stopped_today.items()):
                if sym in self.engine.positions:
                    continue  # already in position
                price = prices.get(sym) if prices else None
                if not price:
                    continue
                ok_re, reason_re = self.engine.can_reenter(sym, price, now)
                if not ok_re:
                    continue
                # Build a candidate for re-entry
                re_dir = rec.direction
                re_candidate = GapCandidate(
                    symbol=sym, gap_pct=rec.gap_pct, prev_close=rec.prev_close,
                    premarket_price=price, avg_vol_20d=rec.avg_vol_20d,
                    vol_ratio=1.0, shortable=True, easy_to_borrow=True,
                    direction=re_dir,
                )
                async with self._position_lock:
                    ok, reason = self.engine.should_enter(re_candidate)
                    if not ok:
                        continue
                    re_time = now.strftime('%Y-%m-%d %H:%M')
                    re_pos = self.engine.open_position(re_candidate, price, re_time)
                    if re_pos is None:
                        continue
                    # Use tighter re-entry stop
                    if re_dir == 'long':
                        re_pos.stop_price = price * (1 - self.config.reentry_stop_pct)
                    else:
                        re_pos.stop_price = price * (1 + self.config.reentry_stop_pct)
                    rec.reentry_count += 1

                # Submit re-entry order
                re_entry_side = 'buy' if re_dir == 'long' else 'sell'
                re_side_label = 'LONG' if re_dir == 'long' else 'SHORT'
                self._add_message('entry',
                    f'RE-ENTRY {re_side_label} {re_pos.shares} {sym} @ ${price:.2f} '
                    f'(stop ${re_pos.stop_price:.2f}, {self.config.reentry_stop_pct:.1%})')

                if self.config.limit_orders_only:
                    if re_dir == 'long':
                        re_limit_px = round(price * (1 + self.config.limit_offset_pct), 2)
                    else:
                        re_limit_px = round(price * (1 - self.config.limit_offset_pct), 2)
                else:
                    re_limit_px = None

                fill = await alpaca_submit_and_confirm(
                    sym, re_pos.shares, re_entry_side,
                    order_type='limit' if self.config.limit_orders_only else 'market',
                    limit_price=re_limit_px,
                    timeout_sec=15.0,
                )
                async with self._position_lock:
                    if fill.is_filled:
                        re_pos.entry_fill_price = fill.filled_avg_price
                        re_pos.entry_order_id = fill.order_id
                        if fill.filled_avg_price > 0:
                            re_pos.entry_price = fill.filled_avg_price
                            if re_dir == 'long':
                                re_pos.stop_price = fill.filled_avg_price * (1 - self.config.reentry_stop_pct)
                            else:
                                re_pos.stop_price = fill.filled_avg_price * (1 + self.config.reentry_stop_pct)
                        stop_result = await asyncio.to_thread(
                            alpaca_place_stop_order, sym, fill.filled_qty, re_pos.stop_price,
                            0.003, re_dir)
                        if 'error' not in stop_result:
                            re_pos.stop_order_id = stop_result.get('id', '')
                        self._add_message('entry',
                            f'RE-ENTRY FILLED {fill.filled_qty} {sym} @ ${fill.filled_avg_price:.2f}')
                    else:
                        self._add_message('error', f'RE-ENTRY FAILED {sym}: {fill.error}')
                        if sym in self.engine.positions:
                            del self.engine.positions[sym]

        await broadcast({
            'type': 'positions_update',
            'positions': {s: asdict(p) for s, p in self.engine.positions.items()},
            'stats': asdict(self.engine.daily_stats),
            'equity': self.engine.equity,
        })

    # -------------------------------------------------------------------------
    # P0-5: EOD Close — retry loop with broker verification
    # -------------------------------------------------------------------------

    async def _eod_close(self):
        """Force close all positions at end of day with retry and verification."""
        self._add_message('system', 'EOD — closing all positions (with verification)')
        logger.info("EOD close initiated")

        # Cancel all broker-side stop orders first
        for sym, pos in list(self.engine.positions.items()):
            if pos.stop_order_id:
                await asyncio.to_thread(alpaca_cancel_order, pos.stop_order_id)
                pos.stop_order_id = ''

        max_retries = 3
        for attempt in range(max_retries):
            if not self.engine.positions:
                break

            symbols_to_close = list(self.engine.positions.keys())
            for sym in symbols_to_close:
                async with self._position_lock:
                    pos = self.engine.positions.get(sym)
                    if pos is None or pos.closing:
                        continue

                    price = 0.0
                    if self.streamer and self.streamer.latest_prices:
                        price = self.streamer.latest_prices.get(sym, 0.0)
                    if price <= 0:
                        try:
                            snaps = await asyncio.to_thread(fetch_alpaca_snapshots, [sym])
                            price = snaps.get(sym, {}).get('latestTrade', {}).get('p', 0.0)
                        except Exception:
                            pass
                    if price <= 0:
                        price = pos.entry_price  # fallback

                    exit_signal = {'reason': 'eod', 'shares': pos.remaining_shares,
                                   'is_full_close': True, 'trigger_price': price}
                    trade = await self._execute_exit(sym, exit_signal, price)

                if trade:
                    await broadcast({'type': 'trade', 'trades': [asdict(trade)]})

            # Verify with broker that positions are actually closed
            await asyncio.sleep(2)  # give broker time to settle
            broker_positions = await asyncio.to_thread(alpaca_get_positions)
            # Check for any remaining positions (both long and short)
            tracked_syms = set(self.engine.positions.keys())
            broker_remaining = [p for p in broker_positions
                                if p.get('symbol') in tracked_syms]

            if not broker_remaining:
                logger.info("EOD close verified: no broker positions remaining")
                self._add_message('system', 'EOD close verified — all positions closed')
                break

            remaining = [p['symbol'] for p in broker_remaining]
            logger.warning(f"EOD close attempt {attempt + 1}/{max_retries}: "
                          f"broker still has positions: {remaining}")
            self._add_message('warning',
                f'EOD retry {attempt + 1}: {len(remaining)} positions still on broker: {remaining}')

            # Clear closing flags for retry
            async with self._position_lock:
                for sym in remaining:
                    if sym in self.engine.positions:
                        self.engine.positions[sym].closing = False

            if attempt < max_retries - 1:
                await asyncio.sleep(3)  # wait before retry

        # Final safety check
        broker_positions = await asyncio.to_thread(alpaca_get_positions)
        tracked_syms = set(self.engine.positions.keys())
        broker_remaining = [p for p in broker_positions
                            if p.get('symbol') in tracked_syms]
        if broker_remaining:
            remaining = [(p['symbol'], p.get('qty')) for p in broker_remaining]
            logger.error(f"CRITICAL: EOD close FAILED after {max_retries} retries! "
                        f"Remaining broker positions: {remaining}")
            self._add_message('error',
                f'CRITICAL: EOD close failed! Still have positions: {remaining}. '
                f'Manual intervention required!')
            await self.alerter.send('EOD CLOSE FAILED',
                f'Positions still open after {max_retries} retries!\n'
                f'Remaining: {remaining}\nManual intervention required!',
                level='error')

        if self.streamer:
            await self.streamer.stop()
            self.streamer = None

        # Clean up any remaining internal positions (broker is truth)
        async with self._position_lock:
            if not broker_remaining:
                for sym in list(self.engine.positions.keys()):
                    del self.engine.positions[sym]

        self._save_state()

    # -------------------------------------------------------------------------
    # State Persistence — atomic writes
    # -------------------------------------------------------------------------

    def _save_state(self):
        """Persist state to JSON with atomic write (write-fsync-rename)."""
        state = {
            'status': self.status,
            'equity': self.engine.equity,
            'peak_equity': self.engine.peak_equity,
            'positions': {s: asdict(p) for s, p in self.engine.positions.items()},
            'daily_stats': asdict(self.engine.daily_stats),
            'config': asdict(self.config),
            'trade_log': [asdict(t) for t in self.engine.all_trade_log[-500:]],
            'messages': self.messages[-50:],
            'last_model_rebuild': self._last_model_rebuild,
            'saved_at': datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S'),
            'active_strategy': self.config.active_strategy,
            'strategy_config': self._strategy_config,
            'conversation_history': self.llm_supervisor.memory.to_dict() if self.llm_supervisor else {},
            'event_bus_pending': self._event_bus.pending_count(),
            'journal_entries_today': len(self.journal._entries),
        }
        tmp_path = self.STATE_FILE + '.tmp'
        try:
            with open(tmp_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.STATE_FILE)  # atomic on POSIX
        except Exception as e:
            logger.error(f"State save failed: {e}")
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _load_state(self):
        """Load persisted state on startup."""
        if not os.path.exists(self.STATE_FILE):
            return
        try:
            with open(self.STATE_FILE, 'r') as f:
                state = json.load(f)
            self.engine.equity = state.get('equity', self.config.initial_capital)
            self.engine.peak_equity = state.get('peak_equity', self.engine.equity)
            self.messages = state.get('messages', [])

            # Restore positions
            for sym, pd_dict in state.get('positions', {}).items():
                try:
                    self.engine.positions[sym] = GapPosition(**{
                        k: v for k, v in pd_dict.items() if k in GapPosition.__dataclass_fields__
                    })
                except Exception as e:
                    logger.warning(f"Could not restore position {sym}: {e}")

            # Restore trade log — SQLite is source of truth, JSON is fallback
            if not self.engine.all_trade_log:
                loaded_from = 'none'
                try:
                    db = get_price_db()
                    db_trades = db.load_all_trades()
                    if db_trades:
                        for td in db_trades:
                            self.engine.all_trade_log.append(TradeRecord(**{
                                k: v for k, v in td.items() if k in TradeRecord.__dataclass_fields__
                            }))
                        loaded_from = 'sqlite'
                        logger.info(f"Restored {len(db_trades)} trades from SQLite")
                except Exception as e:
                    logger.warning(f"SQLite trade load failed, falling back to JSON: {e}")

                if loaded_from == 'none':
                    # Fall back to JSON
                    json_trades = state.get('trade_log', [])
                    for td in json_trades:
                        self.engine.all_trade_log.append(TradeRecord(**{
                            k: v for k, v in td.items() if k in TradeRecord.__dataclass_fields__
                        }))
                    # Auto-migrate JSON trades into SQLite for future restarts
                    if json_trades:
                        try:
                            db = get_price_db()
                            migrated = 0
                            for td in json_trades:
                                if db.insert_trade(td):
                                    migrated += 1
                            logger.info(f"Auto-migrated {migrated}/{len(json_trades)} trades from JSON to SQLite")
                        except Exception as e:
                            logger.warning(f"Trade migration to SQLite failed (non-fatal): {e}")

            # Restore daily stats (survive mid-day restarts)
            saved_daily = state.get('daily_stats', {})
            today_str = datetime.now(ET).strftime('%Y-%m-%d')
            if saved_daily.get('date') == today_str and saved_daily.get('trades', 0) > 0:
                # Same day with actual data — restore the saved counters
                for key, value in saved_daily.items():
                    if hasattr(self.engine.daily_stats, key) and key in DailyStats.__dataclass_fields__:
                        try:
                            field_type = type(getattr(self.engine.daily_stats, key))
                            setattr(self.engine.daily_stats, key, field_type(value))
                        except (TypeError, ValueError):
                            pass
                logger.info(f"Restored daily stats: {saved_daily.get('trades', 0)} trades, "
                           f"${saved_daily.get('pnl', 0):.2f} P&L")
            else:
                # Zero/stale stats or different day — recalculate from trade log
                self._recalculate_daily_stats(today_str)

            # Restore saved config (user-tuned parameters survive restarts)
            saved_config = state.get('config', {})
            restored_keys = []
            for key, value in saved_config.items():
                if hasattr(self.config, key) and key in GapFadeConfig.__dataclass_fields__:
                    try:
                        field_type = type(getattr(self.config, key))
                        setattr(self.config, key, field_type(value))
                        restored_keys.append(key)
                    except (TypeError, ValueError):
                        pass
            if restored_keys:
                # Sync engine config reference
                self.engine.config = self.config
                logger.info(f"Restored {len(restored_keys)} config parameters from state")

            # Restore model rebuild timestamp
            self._last_model_rebuild = state.get('last_model_rebuild', '')

            # Restore strategy config
            self._strategy_config = state.get('strategy_config', {})
            saved_strategy = state.get('active_strategy', '')
            if saved_strategy:
                self.config.active_strategy = saved_strategy

            # Restore conversation history (only if same trading day)
            self._saved_conversation = state.get('conversation_history', {})

            logger.info(f"Loaded state: equity=${self.engine.equity:.2f}, "
                       f"{len(self.engine.positions)} positions, "
                       f"{len(self.engine.all_trade_log)} historical trades")
        except Exception as e:
            logger.warning(f"State load failed: {e}")

    def _recalculate_daily_stats(self, today_str: str = ''):
        """Recalculate daily stats from trade_log for today.

        Fallback for when daily_stats weren't saved or are from a different day.
        """
        if not today_str:
            today_str = datetime.now(ET).strftime('%Y-%m-%d')
        stats = self.engine.daily_stats
        stats.date = today_str
        stats.trades = 0
        stats.wins = 0
        stats.losses = 0
        stats.pnl = 0.0
        stats.consecutive_losses = 0
        stats.halted = False
        stats.halt_reason = ''

        for t in self.engine.all_trade_log:
            if hasattr(t, 'exit_time') and t.exit_time and t.exit_time.startswith(today_str):
                stats.trades += 1
                stats.pnl += t.pnl
                if t.pnl > 0:
                    stats.wins += 1
                    stats.consecutive_losses = 0
                elif t.pnl < 0:
                    stats.losses += 1
                    stats.consecutive_losses += 1

        stats.peak_equity = self.engine.equity
        if stats.trades > 0:
            logger.info(f"Recalculated daily stats from trade log: {stats.trades} trades, "
                       f"{stats.wins}W/{stats.losses}L, ${stats.pnl:+.2f} P&L")

    def get_state(self) -> dict:
        """Get current state for API response."""
        llm_state = None
        if self.llm_supervisor:
            llm_state = self.llm_supervisor.get_status()
            llm_state['enabled'] = True
        else:
            llm_state = {'enabled': False}
        strategy_state = {
            'active': self.config.active_strategy,
            'name': self.strategy.name if self.strategy else 'none',
            'has_indicators': self.indicator_engine is not None,
        }
        return {
            'status': self.status,
            'equity': round(self.engine.equity, 2),
            'peak_equity': round(self.engine.peak_equity, 2),
            'positions': {s: asdict(p) for s, p in self.engine.positions.items()},
            'daily_stats': asdict(self.engine.daily_stats),
            'candidates': [asdict(c) for c in self.candidates[:10]],
            'last_scan_time': self.last_scan_time,
            'metrics': self.engine.get_metrics(),
            'messages': self.messages[-30:],
            'config': asdict(self.config),
            'today_trades': [asdict(t) for t in self.engine.trade_log],
            'llm': llm_state,
            'strategy': strategy_state,
        }


# =============================================================================
# SECTION 8: WEBSOCKET & BROADCAST
# =============================================================================

connected_websockets: List[WebSocket] = []


async def broadcast(msg: dict):
    """Send message to all connected WebSocket clients."""
    dead = []
    for ws in connected_websockets:
        try:
            await ws.send_json(msg)
        except Exception as e:
            logger.warning(f"WS broadcast failed ({msg.get('type','?')}): {e}")
            dead.append(ws)
    for ws in dead:
        connected_websockets.remove(ws)


# =============================================================================
# SECTION 9: FASTAPI ENDPOINTS
# =============================================================================

from contextlib import asynccontextmanager

async def _delayed_auto_start():
    """Auto-start the trading loop after a short delay (lets FastAPI fully initialize)."""
    await asyncio.sleep(5)
    try:
        if not live_trader.config.auto_start:
            return
        now = datetime.now(ET)
        # Only auto-start on weekdays within the trading window
        if now.weekday() >= 5:
            logger.info(f"Auto-start skipped: weekend ({now.strftime('%A')})")
            return
        # Allow auto-start between 6:55 AM and 3:45 PM ET
        start_ok = now.hour > 6 or (now.hour == 6 and now.minute >= 55)
        end_ok = now.hour < 15 or (now.hour == 15 and now.minute <= 45)
        if not (start_ok and end_ok):
            logger.info(f"Auto-start skipped: outside trading window ({now.strftime('%H:%M')} ET)")
            return
        if live_trader.status == 'trading':
            return
        logger.info("Auto-starting trading loop")
        live_trader._add_message('system', 'Auto-started trading loop on app boot')
        await live_trader.start()
        if hasattr(live_trader, 'alerter'):
            await live_trader.alerter.send('Bot Auto-Started',
                f'Trading loop started automatically at {now.strftime("%H:%M ET")}',
                level='info')
    except Exception as e:
        logger.error(f"Auto-start failed: {e}\n{traceback.format_exc()}")


async def _startup_reconcile():
    """Always reconcile with broker on startup, regardless of trading hours.

    The bot must know about any open broker positions at all times.
    Runs after a short delay to let the event loop settle.
    """
    await asyncio.sleep(2)
    try:
        await live_trader._reconcile_with_broker()
        if live_trader.engine.positions:
            syms = list(live_trader.engine.positions.keys())
            logger.info(f"Startup reconciliation: {len(syms)} positions: {syms}")
            await live_trader.alerter.send('Bot Restarted With Positions',
                f'Reconciled {len(syms)} open positions: {", ".join(syms)}',
                level='warning')
    except Exception as e:
        logger.error(f"Startup reconciliation failed: {e}")


@asynccontextmanager
async def lifespan(app):
    asyncio.create_task(_watchdog_heartbeat())
    # Always reconcile with broker on boot (catch orphaned positions)
    asyncio.create_task(_startup_reconcile())
    # Auto-start trading loop if configured
    if live_trader.config.auto_start:
        asyncio.create_task(_delayed_auto_start())
    logger.info("Gap Fade app started")
    yield

app = FastAPI(title="Gap Fade Strategy", version=APP_VERSION, lifespan=lifespan)
_app_start_time = _time.time()

# ---------------------------------------------------------------------------
# OAuth session support (authlib needs SessionMiddleware for OAuth state)
# ---------------------------------------------------------------------------
from starlette.middleware.sessions import SessionMiddleware
_load_env_file()
_session_secret = os.environ.get('AUTH_JWT_SECRET', os.urandom(32).hex())
app.add_middleware(SessionMiddleware, secret_key=_session_secret)

from auth import router as auth_router, auth_enabled, get_current_user
app.include_router(auth_router)

# ---------------------------------------------------------------------------
# P0-6: API Authentication Middleware
# ---------------------------------------------------------------------------
# Set GAP_FADE_API_KEY in .env to require X-API-Key header on mutating endpoints.
# Read-only endpoints (GET, WebSocket, dashboard) remain open.

_API_KEY = None  # loaded lazily

def _get_api_key() -> Optional[str]:
    global _API_KEY
    if _API_KEY is None:
        _load_env_file()
        _API_KEY = os.environ.get('GAP_FADE_API_KEY', '') or ''
    return _API_KEY if _API_KEY else None

# Endpoints that DON'T require auth (read-only)
# Only these paths are accessible without an API key
_AUTH_EXEMPT_PATHS = {'/', '/api/health', '/api/backtest/status', '/api/backtest/results', '/docs', '/openapi.json', '/ws'}

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Auth gate: exempt paths, then API key header, then JWT cookie."""
    path = request.url.path.rstrip('/')

    # Always allow exempt paths and auth endpoints
    if path in _AUTH_EXEMPT_PATHS or path.startswith('/api/auth'):
        return await call_next(request)

    # Check API key header (backward compat — always works if valid)
    api_key = _get_api_key()
    if api_key:
        provided = request.headers.get('X-API-Key', '')
        if provided == api_key:
            return await call_next(request)

    # If OAuth auth is enabled, require valid JWT cookie on /api/ paths
    if auth_enabled() and path.startswith('/api/'):
        user = get_current_user(request)
        if not user:
            return JSONResponse(
                status_code=401,
                content={'error': 'Unauthorized: please log in'}
            )

    # If API key is set but not provided (and no valid JWT), block mutating endpoints
    if api_key and path.startswith('/api/'):
        provided = request.headers.get('X-API-Key', '')
        if provided != api_key and not get_current_user(request):
            logger.warning(f"Auth failed for {request.method} {path} from {request.client.host}")
            return JSONResponse(
                status_code=403,
                content={'error': 'Forbidden: invalid or missing credentials'}
            )

    return await call_next(request)


# ---------------------------------------------------------------------------
# P0-8: Config Validation
# ---------------------------------------------------------------------------

CONFIG_VALID_RANGES = {
    'stop_pct':           (0.005, 0.10),    # 0.5% to 10%
    'risk_pct':           (0.001, 0.10),    # 0.1% to 10%
    'max_positions':      (1, 10),
    'initial_capital':    (1000, 10_000_000),
    'gap_threshold':      (0.01, 0.50),     # 1% to 50%
    'max_gap_pct':        (0.10, 1.0),      # 10% to 100%
    'vol_ratio_max':      (0.5, 20.0),
    'daily_loss_limit':   (0.005, 0.10),    # 0.5% to 10%
    'max_drawdown':       (0.01, 0.30),     # 1% to 30%
    'max_consec_losses':  (1, 20),
    'kelly_fraction':     (0.05, 1.0),
    'max_notional':       (1000, 1_000_000),
    'slippage_pct':       (0.0, 0.05),      # 0% to 5%
    'limit_offset_pct':   (0.0, 0.05),      # 0% to 5%
    'partial_cover_frac': (0.0, 1.0),
    'min_price':          (0.01, 1000),
    'min_avg_volume':     (0, 100_000_000),
    'catalyst_earnings_penalty': (0, 100),
    'catalyst_news_penalty':     (0, 100),
    'catalyst_noise_bonus':      (0, 50),
    'llm_timeout':               (1.0, 60.0),
    'llm_max_failures':          (1, 50),
    'llm_circuit_reset':         (10.0, 600.0),
    'llm_max_hold_overrides':    (0, 10),
    # Drawdown circuit breakers
    'dd_tier1_threshold':    (0.05, 0.50),
    'dd_tier1_scale':        (0.01, 1.0),
    'dd_tier2_threshold':    (0.10, 0.70),
    'dd_tier2_scale':        (0.01, 1.0),
    'dd_tier2_max_positions':(1, 5),
    'dd_hard_stop':          (0.0, 1.0),
}


def validate_config(updates: dict) -> Tuple[dict, List[str]]:
    """Validate config updates against safe ranges.

    Returns (validated_updates, errors). validated_updates only contains
    keys that passed validation.
    """
    validated = {}
    errors = []
    for key, value in updates.items():
        if not hasattr(GapFadeConfig, key):
            continue
        field_type = type(getattr(GapFadeConfig(), key))
        try:
            typed_value = field_type(value)
        except (ValueError, TypeError):
            errors.append(f"{key}: invalid type (expected {field_type.__name__})")
            continue

        if key in CONFIG_VALID_RANGES:
            lo, hi = CONFIG_VALID_RANGES[key]
            if not (lo <= typed_value <= hi):
                errors.append(f"{key}: {typed_value} out of range [{lo}, {hi}]")
                continue

        validated[key] = typed_value
    return validated, errors


# Singletons
live_trader = GapFadeLiveTrader()
backtester = GapFadeBacktester()


async def _watchdog_heartbeat():
    """Ping systemd watchdog every 30s so it knows we're alive.

    Validates that the trading loop is actually progressing during market hours.
    If the loop is stale (>90s since last iteration), skip the watchdog ping
    so systemd will restart us.
    """
    try:
        import socket
        addr = os.environ.get('NOTIFY_SOCKET')
        if not addr:
            return  # Not running under systemd watchdog
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        if addr[0] == '@':
            addr = '\0' + addr[1:]
        # Send READY on startup
        sock.sendto(b'READY=1', addr)
        logger.info("systemd watchdog: READY sent")
        while True:
            # Check if trading loop is alive (during market hours, when running)
            now = datetime.now(ET)
            if 7 <= now.hour < 17 and live_trader.status not in ('stopped', 'waiting'):
                age = _time.monotonic() - live_trader._last_loop_heartbeat
                if age > 90:
                    logger.error(f"Trading loop stale ({age:.0f}s) — NOT sending watchdog ping")
                    await live_trader.alerter.send('Watchdog: Loop Stale',
                        f'Trading loop has not iterated in {age:.0f}s — systemd will restart',
                        level='error')
                    await asyncio.sleep(30)
                    continue  # skip ping → systemd will restart us
            sock.sendto(b'WATCHDOG=1', addr)
            await asyncio.sleep(30)
    except Exception as e:
        logger.warning(f"Watchdog heartbeat error: {e}")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the embedded HTML dashboard with API key injected."""
    api_key = _get_api_key() or ''
    # Inject the key as a JS variable so the dashboard can authenticate POST requests
    html = DASHBOARD_HTML.replace(
        '/*__API_KEY_PLACEHOLDER__*/',
        f'const __API_KEY__ = "{api_key}";',
    ).replace('__APP_VERSION__', APP_VERSION)
    return HTMLResponse(html)


@app.get("/api/health")
async def health_check():
    """Health check for monitoring / watchdog."""
    now = datetime.now(ET)
    return {
        'status': 'ok',
        'timestamp': now.isoformat(),
        'trader_status': live_trader.status,
        'uptime_seconds': int((_time.time() - _app_start_time)),
        'positions': len(live_trader.engine.positions) if live_trader.engine else 0,
    }


@app.get("/api/state")
async def get_state():
    """Get current live trader state."""
    return live_trader.get_state()


@app.post("/api/scan")
async def run_scan():
    """Trigger a pre-market scan (non-blocking)."""
    scanner = GapScanner(live_trader.config)
    candidates, universe_size = await asyncio.to_thread(scanner.scan_premarket)
    live_trader.candidates = candidates
    live_trader.last_scan_time = datetime.now(ET).strftime('%H:%M:%S')

    # Catalyst detection
    if live_trader.config.catalyst_enabled and candidates:
        try:
            await detect_catalysts(candidates, live_trader.config)
            live_trader._apply_catalyst_scores()
            candidates = live_trader.candidates  # _apply_catalyst_scores filters in-place
            candidates.sort(key=lambda c: c.score, reverse=True)
        except Exception as e:
            logger.warning(f"Catalyst detection failed (non-fatal): {e}")

    # LLM candidate evaluation
    llm_evals = []
    if live_trader.llm_supervisor and live_trader.llm_supervisor.is_available() and candidates:
        try:
            llm_evals = await live_trader.llm_supervisor.evaluate_candidates(
                [asdict(c) for c in candidates[:15]])
            if llm_evals:
                skip_syms = {e['symbol'] for e in llm_evals
                             if isinstance(e, dict) and e.get('action') == 'skip'}
                if skip_syms:
                    live_trader.candidates = [c for c in live_trader.candidates
                                              if c.symbol not in skip_syms]
                    candidates = live_trader.candidates
        except Exception as e:
            logger.warning(f"Rudra candidate evaluation failed (non-fatal): {e}")

    return {
        'candidates': [asdict(c) for c in candidates[:15]],
        'scan_time': live_trader.last_scan_time,
        'count': len(candidates),
        'universe_size': universe_size,
        'llm_evals': llm_evals if llm_evals else None,
    }


@app.post("/api/start")
async def start_trading():
    """Start the live trading loop."""
    await live_trader.start()
    return {'status': live_trader.status}


@app.post("/api/enter")
async def enter_positions(body: dict = None):
    """Manually trigger entry on current candidates.

    Optional body: {"symbols": ["SYM1", "SYM2"]} to enter specific symbols.
    Without body, enters top candidates up to max_positions.
    """
    if live_trader.status == 'stopped':
        return {'error': 'Trader is stopped. Start it first.'}
    body = body or {}
    llm_symbols = body.get('symbols')
    size_mult = body.get('size_mult', 1.0)
    # Scan first if no candidates cached
    if not live_trader.candidates:
        await live_trader._run_scan()
    await live_trader._enter_positions(llm_symbols=llm_symbols, size_mult=size_mult)
    return {
        'status': live_trader.status,
        'positions': {s: asdict(p) for s, p in live_trader.engine.positions.items()},
    }


@app.post("/api/stop")
async def stop_trading():
    """Stop the live trader."""
    await live_trader.stop()
    return {'status': live_trader.status}


@app.post("/api/pause")
async def pause_trading():
    await live_trader.pause()
    return {'status': live_trader.status}


@app.post("/api/resume")
async def resume_trading():
    await live_trader.resume()
    return {'status': live_trader.status}


@app.post("/api/reset")
async def reset_trader():
    """Reset equity and trade log."""
    await live_trader.stop()
    config = live_trader.config
    live_trader.engine = GapFadeEngine(config)
    live_trader.candidates = []
    live_trader.messages = []
    return {'status': 'reset', 'equity': config.initial_capital}


@app.get("/api/strategies")
async def list_strategies():
    """List available strategies and active strategy."""
    try:
        from gap_fade_strategies import GapFadeStrategyRegistry
        strategies = GapFadeStrategyRegistry.list_strategies()
    except Exception as e:
        strategies = []
        logger.warning(f"Failed to list strategies: {e}")
    return {
        'strategies': strategies,
        'active': live_trader.config.active_strategy,
        'active_name': live_trader.strategy.name if live_trader.strategy else 'none',
        'has_indicators': live_trader.indicator_engine is not None,
        'strategy_config': live_trader._strategy_config,
    }


@app.post("/api/strategy/switch")
async def switch_strategy(body: dict):
    """Switch the active strategy.

    Body: {"strategy_id": "vwap_gap_fade", "config": {...optional overrides...}}
    """
    strategy_id = body.get('strategy_id', '')
    if not strategy_id:
        return {'error': 'Missing strategy_id'}
    try:
        from gap_fade_strategies import GapFadeStrategyRegistry
        if not GapFadeStrategyRegistry.has_strategy(strategy_id):
            available = [s['id'] for s in GapFadeStrategyRegistry.list_strategies()]
            return {'error': f"Unknown strategy '{strategy_id}'. Available: {available}"}
    except Exception as e:
        return {'error': f'Strategy system unavailable: {e}'}

    strat_config = body.get('config')
    live_trader.switch_strategy(strategy_id, strat_config)
    return {
        'active': live_trader.config.active_strategy,
        'name': live_trader.strategy.name if live_trader.strategy else 'unknown',
        'has_indicators': live_trader.indicator_engine is not None,
    }


@app.post("/api/strategy/config")
async def update_strategy_config(body: dict):
    """Update strategy-specific config parameters.

    Body: {"param_name": value, ...}
    """
    if not live_trader.strategy:
        return {'error': 'No strategy loaded'}
    strategy_id = live_trader.config.active_strategy
    live_trader.strategy.update_config(body)
    live_trader._strategy_config[strategy_id] = dict(live_trader.strategy.config)
    live_trader._save_state()
    return {
        'strategy': strategy_id,
        'config': live_trader.strategy.config,
    }


@app.post("/api/config")
async def update_config(body: dict):
    """Update strategy configuration with validation."""
    # P0-8: Validate all updates against safe ranges
    validated, errors = validate_config(body)
    if errors:
        logger.warning(f"Config validation errors: {errors}")

    config = live_trader.config
    applied = []
    for key, value in validated.items():
        old_val = getattr(config, key, None)
        setattr(config, key, value)
        if old_val != value:
            applied.append(f"{key}: {old_val} → {value}")

    if applied:
        logger.info(f"Config updated: {', '.join(applied)}")

    # Sync capital if changed while idle (no open positions)
    old_capital = live_trader.engine.config.initial_capital
    live_trader.engine.config = config
    if config.initial_capital != old_capital and not live_trader.engine.positions:
        live_trader.engine.equity = config.initial_capital
        live_trader.engine.peak_equity = config.initial_capital
    # Rebuild scanner when universe config changes
    live_trader.scanner = GapScanner(config)
    # Toggle LLM supervisor on/off at runtime, rebuild if config changed
    if config.llm_enabled:
        if live_trader.llm_supervisor is None:
            live_trader.llm_supervisor = LLMSupervisor(config)
            logger.info(f"Rudra enabled: {config.llm_model}")
        else:
            # Rebuild if url/model/timeout changed
            sup = live_trader.llm_supervisor
            if (sup.url != config.llm_url or sup.model != config.llm_model
                    or sup.timeout != config.llm_timeout
                    or sup.provider != config.llm_provider):
                live_trader.llm_supervisor = LLMSupervisor(config)
                logger.info(f"Rudra rebuilt: {config.llm_model} @ {config.llm_url}")
    elif live_trader.llm_supervisor is not None:
        live_trader.llm_supervisor = None
        logger.info("Rudra disabled")
    # Persist config changes immediately (survive restarts)
    live_trader._save_state()

    result = {'config': asdict(config)}
    if errors:
        result['validation_errors'] = errors
    return result


@app.get("/api/llm/status")
async def llm_status():
    """Get LLM supervisor status (circuit breaker, call stats)."""
    if live_trader.llm_supervisor is None:
        return {
            'enabled': False,
            'available': False,
            'model': live_trader.config.llm_model,
            'url': live_trader.config.llm_url,
        }
    status = live_trader.llm_supervisor.get_status()
    status['enabled'] = True
    return status


@app.post("/api/llm/test")
async def llm_test():
    """Test LLM connectivity by sending a simple ping prompt."""
    if not live_trader.config.llm_enabled:
        return {'error': 'Rudra not enabled. Set llm_enabled=true via /api/config first.'}
    if live_trader.llm_supervisor is None:
        live_trader.llm_supervisor = LLMSupervisor(live_trader.config)
    raw = await live_trader.llm_supervisor._call_llm(
        'You are a test assistant.', 'Respond with exactly: {"status": "ok"}')
    if raw is None:
        return {'status': 'error', 'message': 'Could not reach Ollama. Check url/model.',
                'supervisor': live_trader.llm_supervisor.get_status()}
    return {'status': 'ok', 'response': raw[:200],
            'supervisor': live_trader.llm_supervisor.get_status()}


@app.get("/api/llm/conversation")
async def llm_conversation():
    """Get current conversation history between bot and Rudra."""
    if live_trader.llm_supervisor is None:
        return {'history': [], 'summary': '', 'day': ''}
    mem = live_trader.llm_supervisor.memory
    return {
        'history': mem.history[-40:],  # last 40 messages
        'summary': mem.summary,
        'day': mem.day,
        'total_messages': len(mem.history),
    }


@app.get("/api/journal")
async def journal_api(date: str = '', n: int = 50, entry_type: str = ''):
    """Get trading journal entries.

    Query params:
        date: YYYY-MM-DD (default: today). Load from SQLite for historical.
        n: max entries to return (default 50)
        entry_type: filter by type (observation, reasoning, action, no_action, reflection)
    """
    if not date:
        date = datetime.now(ET).strftime('%Y-%m-%d')
    today = datetime.now(ET).strftime('%Y-%m-%d')

    if date == today:
        # Return from in-memory buffer (hot path for LLM context)
        entries = live_trader.journal.get_recent(n=n, entry_type=entry_type or None)
        return {
            'date': date,
            'entries': [asdict(e) for e in entries],
            'count': len(entries),
        }
    else:
        # Load from SQLite
        db = get_price_db()
        entries = db.query_journal(date=date, n=n, entry_type=entry_type)
        stats = db.get_journal_stats(date)
        return {
            'date': date,
            'entries': entries,
            'count': len(entries),
            'stats': stats,
        }


@app.get("/api/events")
async def events_api(date: str = ''):
    """Get event bus status and recent event log.

    Query params:
        date: YYYY-MM-DD (optional). If given, load from SQLite. Otherwise in-memory.
    """
    if date:
        # Historical: load from SQLite
        db = get_price_db()
        events = db.query_events(date=date, n=100)
        return {
            'pending_count': 0,
            'has_urgent': False,
            'recent_events': events,
        }
    # Current session: in-memory
    return {
        'pending_count': live_trader._event_bus.pending_count(),
        'has_urgent': live_trader._event_bus.has_urgent(),
        'recent_events': [
            {'type': e.event_type, 'tier': e.tier, 'symbol': e.symbol,
             'description': e.description, 'data': e.data}
            for e in list(live_trader._event_bus._event_log)[-30:]
        ],
    }


@app.get("/api/llm-calls")
async def llm_calls_api(date: str = '', n: int = 50):
    """Get LLM call history and stats.

    Query params:
        date: YYYY-MM-DD (default: today)
        n: max entries to return (default 50)
    """
    if not date:
        date = datetime.now(ET).strftime('%Y-%m-%d')
    db = get_price_db()
    calls = db.query_llm_calls(date=date, n=n)
    stats = db.get_llm_stats(date=date)
    return {'date': date, 'calls': calls, 'stats': stats}


_CHAT_SYSTEM_PROMPT = """You are Rudra, the AI trading supervisor for a gap fade bot. You have full access to the bot's live state and can EXECUTE actions by returning a JSON action block.

## YOUR CAPABILITIES
You can both advise AND act. When the user asks you to DO something, execute it.

## EXECUTABLE ACTIONS
To execute an action, include a JSON block at the END of your response in this exact format:
```action
{"action": "ACTION_NAME", ...params}
```

Available actions:

1. START TRADING: `{"action": "start"}`
2. STOP TRADING (closes all positions): `{"action": "stop"}`
3. PAUSE (keep positions, stop new trades): `{"action": "pause"}`
4. RESUME after pause: `{"action": "resume"}`
5. RUN SCAN for candidates: `{"action": "scan"}`
6. ENTER POSITIONS: `{"action": "enter"}` or `{"action": "enter", "symbols": ["AAPL","MSFT"], "size_mult": 0.5}`
7. UPDATE CONFIG: `{"action": "config", "params": {"max_positions": 5, "stop_pct": 0.02, ...}}`
   Settable config fields: gap_threshold, max_gap_pct, vol_ratio_max, stop_pct, risk_pct,
   kelly_fraction, max_positions, initial_capital, daily_loss_limit, max_consec_losses,
   max_drawdown, time_exit_hour, min_avg_volume, min_price, max_notional, slippage_pct,
   borrow_rate_annual, max_pct_adv, limit_offset_pct, limit_orders_only,
   adaptive_stops, stop_gap_fraction, stop_min_pct, stop_max_pct,
   regime_filter, regime_spy_gap_limit, regime_spy_block_pct, regime_vix_threshold,
   reentry_enabled, reentry_cooldown_minutes, reentry_max_per_symbol, reentry_stop_pct,
   trade_gap_downs, gap_down_threshold, gap_down_max_pct, gap_down_vol_ratio_max,
   dd_circuit_breaker, dd_tier1_threshold, dd_tier1_scale, dd_tier2_threshold,
   dd_tier2_scale, dd_tier2_max_positions, dd_hard_stop,
   llm_enabled, llm_model, llm_url, llm_timeout, catalyst_enabled,
   entry_cutoff_hour, entry_cutoff_min, min_hold_minutes, min_profit_take_pct, min_gap_fill_pct
8. RESET equity & trade log: `{"action": "reset"}`
9. RUN BACKTEST: `{"action": "backtest", "params": {"symbols": "TSLA,NVDA", "start_date": "2024-01-01", "end_date": "2024-12-31", "gap_threshold": 0.07}}`

## RULES
- Always explain WHAT you're doing and WHY before the action block
- For destructive actions (stop, reset), confirm with the user first unless they're explicit
- Use the live context below to give informed answers with real numbers
- Be concise. Use tables for position summaries. Use $ and % values.
- If the user asks a question (not an action), just answer — no action block needed
- You CAN close a single position: `{"action": "close_position", "symbol": "TSLA"}`
- You CAN adjust a single position's stop: `{"action": "adjust_stop", "symbol": "TSLA", "stop_price": 185.50}`
- The bot runs an autonomous review every 5 minutes — it will auto-tighten stops and close positions when appropriate"""

@app.post("/api/llm/chat")
async def llm_chat(body: dict):
    """Chat with the LLM trading supervisor. Injects live context, executes actions."""
    message = (body.get('message') or '').strip()
    if not message:
        return {'error': 'message is required'}
    if not live_trader.config.llm_enabled:
        return {'error': 'Rudra not enabled. Set llm_enabled=true via /api/config first.'}
    if live_trader.llm_supervisor is None:
        live_trader.llm_supervisor = LLMSupervisor(live_trader.config)

    # Build live context
    now = datetime.now(ET)
    positions_text = 'No open positions.'
    if live_trader.engine.positions:
        lines = []
        for sym, pos in live_trader.engine.positions.items():
            cur_price = 0.0
            if live_trader.streamer and live_trader.streamer.latest_prices:
                cur_price = live_trader.streamer.latest_prices.get(sym, 0.0)
            if cur_price <= 0:
                try:
                    snaps = await asyncio.to_thread(fetch_alpaca_snapshots, [sym])
                    cur_price = snaps.get(sym, {}).get('latestTrade', {}).get('p', 0.0)
                except Exception:
                    pass
            if cur_price > 0:
                if pos.direction == 'short':
                    pnl_pct = (pos.entry_price - cur_price) / pos.entry_price * 100
                    pnl_dollar = (pos.entry_price - cur_price) * pos.remaining_shares
                else:
                    pnl_pct = (cur_price - pos.entry_price) / pos.entry_price * 100
                    pnl_dollar = (cur_price - pos.entry_price) * pos.remaining_shares
                lines.append(
                    f"  {sym}: {pos.direction} {pos.remaining_shares} shares, "
                    f"entry ${pos.entry_price:.2f}, current ${cur_price:.2f}, "
                    f"stop ${pos.stop_price:.2f}, P&L {pnl_pct:+.1f}% (${pnl_dollar:+,.2f}), "
                    f"HWM {pos.high_water_pnl_pct*100:.1f}%, gap fill {pos.gap_fill_pct(cur_price):.0%}")
            else:
                lines.append(
                    f"  {sym}: {pos.direction} {pos.remaining_shares} shares, "
                    f"entry ${pos.entry_price:.2f}, stop ${pos.stop_price:.2f}")
        positions_text = '\n'.join(lines)

    candidates_text = 'No candidates scanned yet.'
    if live_trader.candidates:
        cands = live_trader.candidates[:8]
        candidates_text = '\n'.join(
            f"  {c.symbol}: gap {c.gap_pct:.1%}, vol_ratio {c.vol_ratio:.2f}, "
            f"score {c.score:.0f}, dir={c.direction}"
            for c in cands)

    recent_msgs = '\n'.join(
        f"  [{m['type']}] {m['text'][:120]}"
        for m in live_trader.messages[-12:])

    stats = live_trader.engine.daily_stats
    trades_today = live_trader.engine.trade_log[-10:] if live_trader.engine.trade_log else []
    trades_text = 'None today.'
    if trades_today:
        trades_text = '\n'.join(
            f"  {t.symbol}: {t.side} {t.shares}sh, "
            f"entry ${t.entry_price:.2f} -> exit ${t.exit_price:.2f}, "
            f"P&L ${t.pnl:+,.2f} ({t.pnl_pct:+.1%}), reason={t.exit_reason}"
            for t in trades_today)

    cfg = live_trader.config
    context = f"""LIVE TRADING CONTEXT (as of {now.strftime('%H:%M:%S ET, %A %B %d')}):

Bot status: {live_trader.status}
Equity: ${live_trader.engine.equity:,.2f}
Today P&L: ${stats.pnl:+,.2f} ({stats.wins}W/{stats.losses}L)
Consecutive losses: {stats.consecutive_losses}

Open positions ({len(live_trader.engine.positions)}/{cfg.max_positions}):
{positions_text}

Top candidates:
{candidates_text}

Completed trades today:
{trades_text}

Recent activity:
{recent_msgs}

Config: max_positions={cfg.max_positions}, stop_pct={cfg.stop_pct}, risk_pct={cfg.risk_pct}, \
reentry={cfg.reentry_enabled}, adaptive_stops={cfg.adaptive_stops}, regime_filter={cfg.regime_filter}, \
llm_enabled={cfg.llm_enabled}, llm_model={cfg.llm_model}, limit_orders={cfg.limit_orders_only}, \
gap_threshold={cfg.gap_threshold}, initial_capital={cfg.initial_capital}

Drawdown: {((live_trader.engine.peak_equity - live_trader.engine.equity) / cfg.initial_capital * 100) if cfg.initial_capital > 0 else 0:.1f}% of initial capital \
(peak=${live_trader.engine.peak_equity:,.0f}, current=${live_trader.engine.equity:,.0f})
DD Circuit Breaker: {"ON" if cfg.dd_circuit_breaker else "OFF"}{f", Tier 1 at {cfg.dd_tier1_threshold:.0%} (scale {cfg.dd_tier1_scale}), Tier 2 at {cfg.dd_tier2_threshold:.0%} (scale {cfg.dd_tier2_scale}, max {cfg.dd_tier2_max_positions} pos), Hard Stop at {cfg.dd_hard_stop:.0%}" if cfg.dd_circuit_breaker else ""}

USER MESSAGE: {message}"""

    chat_system = _CHAT_SYSTEM_PROMPT
    if live_trader.llm_supervisor.lessons_context:
        chat_system = chat_system + '\n\n' + live_trader.llm_supervisor.lessons_context

    # Use conversation memory so Rudra remembers the chat and intraday context
    raw = await live_trader.llm_supervisor._call_llm(
        chat_system, context,
        use_memory=True, record_exchange=True,
        conversation_role='user_chat',
        event_queue=live_trader._event_bus)
    if raw is None:
        return {'error': 'Rudra call failed. Check Ollama connectivity.'}

    # Parse and execute any action block from the response
    import re as _re
    action_result = None
    action_match = _re.search(r'```action\s*\n(\{.*?\})\s*\n```', raw, _re.DOTALL)
    if action_match:
        try:
            action = json.loads(action_match.group(1))
            act = action.get('action', '')
            if act == 'start':
                await live_trader.start()
                action_result = {'executed': 'start', 'status': live_trader.status}
            elif act == 'stop':
                await live_trader.stop()
                action_result = {'executed': 'stop', 'status': live_trader.status}
            elif act == 'pause':
                live_trader.status = 'paused'
                action_result = {'executed': 'pause', 'status': 'paused'}
            elif act == 'resume':
                if live_trader.status == 'paused':
                    live_trader.status = 'trading'
                action_result = {'executed': 'resume', 'status': live_trader.status}
            elif act == 'scan':
                candidates = await live_trader._run_scan()
                action_result = {'executed': 'scan', 'candidates': len(live_trader.candidates)}
            elif act == 'enter':
                syms = action.get('symbols')
                mult = action.get('size_mult', 1.0)
                await live_trader._enter_positions(llm_symbols=syms, size_mult=mult)
                action_result = {'executed': 'enter',
                                 'positions': list(live_trader.engine.positions.keys())}
            elif act == 'config':
                params = action.get('params', {})
                if params:
                    for k, v in params.items():
                        if hasattr(cfg, k):
                            setattr(cfg, k, type(getattr(cfg, k))(v))
                    live_trader.engine.config = cfg
                    live_trader.scanner = GapScanner(cfg)
                    live_trader._save_state()
                    action_result = {'executed': 'config', 'updated': list(params.keys())}
            elif act == 'reset':
                live_trader.engine.equity = cfg.initial_capital
                live_trader.engine.peak_equity = cfg.initial_capital
                live_trader.engine.trade_log.clear()
                live_trader._save_state()
                action_result = {'executed': 'reset', 'equity': cfg.initial_capital}
            elif act == 'close_position':
                sym = action.get('symbol', '').upper()
                if sym and sym in live_trader.engine.positions:
                    resp = await close_position(sym)
                    action_result = {'executed': 'close_position', 'symbol': sym, **resp}
                else:
                    action_result = {'executed': 'error', 'error': f'No position for {sym}'}
            elif act == 'adjust_stop':
                sym = action.get('symbol', '').upper()
                stop_px = action.get('stop_price')
                if sym and stop_px:
                    resp = await adjust_stop(sym, {'stop_price': stop_px})
                    action_result = {'executed': 'adjust_stop', 'symbol': sym, **resp}
                else:
                    action_result = {'executed': 'error', 'error': 'symbol and stop_price required'}
            elif act == 'backtest':
                params = action.get('params', {})
                action_result = {'executed': 'backtest', 'note': 'Use the Backtest tab to run backtests with full control.'}
        except Exception as e:
            action_result = {'executed': 'error', 'error': str(e)}

    # Strip the action block from displayed response
    display_text = _re.sub(r'\n?```action\s*\n\{.*?\}\s*\n```', '', raw, flags=_re.DOTALL).strip()

    # Add conversation to activity feed
    user_brief = message[:80] + ('...' if len(message) > 80 else '')
    rudra_brief = display_text[:200] + ('...' if len(display_text) > 200 else '')
    live_trader._add_message('conversation', f'BOT: {user_brief} | RUDRA: {rudra_brief}')
    await broadcast({'type': 'conversation', 'speaker': 'rudra', 'text': display_text[:500]})

    result = {'response': display_text}
    if action_result:
        result['action_result'] = action_result
    return result


@app.post("/api/positions/{symbol}/close")
async def close_position(symbol: str):
    """Close a specific position by symbol."""
    symbol = symbol.upper()
    pos = live_trader.engine.positions.get(symbol)
    if pos is None:
        return {'error': f'No open position for {symbol}'}
    if pos.closing:
        return {'error': f'{symbol} is already closing'}

    # Get current price
    price = 0.0
    if live_trader.streamer and live_trader.streamer.latest_prices:
        price = live_trader.streamer.latest_prices.get(symbol, 0.0)
    if price <= 0:
        try:
            snaps = await asyncio.to_thread(fetch_alpaca_snapshots, [symbol])
            price = snaps.get(symbol, {}).get('latestTrade', {}).get('p', 0.0)
        except Exception:
            pass
    if price <= 0:
        price = pos.entry_price  # fallback

    exit_signal = {
        'reason': 'manual',
        'shares': pos.remaining_shares,
        'is_full_close': True,
        'trigger_price': price,
    }

    async with live_trader._position_lock:
        trade = await live_trader._execute_exit(symbol, exit_signal, price)

    if trade:
        live_trader._add_message('manual', f'Closed {symbol}: P&L ${trade.pnl:+,.2f} ({trade.pnl_pct:+.1%})')
        live_trader._save_state()
        await broadcast({'type': 'trade', 'trades': [asdict(trade)]})
        return {'status': 'closed', 'symbol': symbol, 'pnl': trade.pnl, 'pnl_pct': trade.pnl_pct}
    else:
        return {'error': f'Failed to close {symbol} — order may not have filled'}


@app.post("/api/positions/{symbol}/stop")
async def adjust_stop(symbol: str, body: dict):
    """Adjust the stop price for a specific position."""
    symbol = symbol.upper()
    new_stop = body.get('stop_price')
    if new_stop is None:
        return {'error': 'stop_price is required'}
    try:
        new_stop = float(new_stop)
    except (ValueError, TypeError):
        return {'error': 'stop_price must be a number'}
    if new_stop <= 0:
        return {'error': 'stop_price must be positive'}

    pos = live_trader.engine.positions.get(symbol)
    if pos is None:
        return {'error': f'No open position for {symbol}'}

    old_stop = pos.stop_price

    async with live_trader._position_lock:
        pos = live_trader.engine.positions.get(symbol)
        if pos is None:
            return {'error': f'Position {symbol} gone'}
        pos.stop_price = new_stop

        # Update broker-side stop order if one exists
        if pos.stop_order_id:
            try:
                await asyncio.to_thread(alpaca_cancel_order, pos.stop_order_id)
                stop_result = await asyncio.to_thread(
                    alpaca_place_stop_order, symbol,
                    pos.remaining_shares, new_stop,
                    0.003, pos.direction)
                if 'error' not in stop_result:
                    pos.stop_order_id = stop_result.get('id', '')
                else:
                    logger.warning(f"Broker stop update failed for {symbol}: {stop_result}")
            except Exception as e:
                logger.warning(f"Broker stop update failed for {symbol}: {e}")

    live_trader._add_message('manual',
        f'{symbol} stop adjusted: ${old_stop:.2f} → ${new_stop:.2f}')
    live_trader._save_state()
    return {
        'status': 'updated',
        'symbol': symbol,
        'old_stop': old_stop,
        'new_stop': new_stop,
    }


@app.post("/api/backtest")
async def run_backtest(body: dict):
    """Run a historical backtest."""
    global backtester
    if backtester.status == 'running':
        return {'error': 'A backtest is already running. Cancel it first.'}
    symbol = body.get('symbol')
    symbols = body.get('symbols')
    start_date = body.get('start_date')
    end_date = body.get('end_date')

    # Validate dates
    for dkey in ['start_date', 'end_date']:
        val = body.get(dkey)
        if val:
            try:
                datetime.strptime(val, '%Y-%m-%d')
            except ValueError:
                return {'error': f'Invalid {dkey}: "{val}" — expected YYYY-MM-DD format'}
    if start_date and end_date and start_date > end_date:
        return {'error': f'start_date ({start_date}) must be before end_date ({end_date})'}

    # Validate numeric params
    validation_errors = []
    bt_numeric_keys = [
        'gap_threshold', 'max_gap_pct', 'vol_ratio_max', 'stop_pct', 'risk_pct',
        'kelly_fraction', 'max_positions', 'initial_capital',
        'daily_loss_limit', 'max_consec_losses', 'max_drawdown',
        'max_notional', 'slippage_pct', 'partial_cover_frac',
        'min_price', 'min_avg_volume',
    ]
    for key in bt_numeric_keys:
        if key in body and key in CONFIG_VALID_RANGES:
            try:
                val = float(body[key]) if not isinstance(body[key], bool) else body[key]
                lo, hi = CONFIG_VALID_RANGES[key]
                if not (lo <= val <= hi):
                    validation_errors.append(f"{key}: {val} out of range [{lo}, {hi}]")
            except (ValueError, TypeError):
                validation_errors.append(f"{key}: invalid value '{body[key]}'")
    if validation_errors:
        return {'error': 'Validation failed: ' + '; '.join(validation_errors)}

    # Build config from body
    config = GapFadeConfig()
    config_errors = []
    for key in ['gap_threshold', 'max_gap_pct', 'vol_ratio_max', 'stop_pct', 'risk_pct',
                'kelly_fraction', 'max_positions', 'initial_capital',
                'daily_loss_limit', 'max_consec_losses', 'max_drawdown',
                'time_exit_hour', 'min_avg_volume', 'min_price',
                'max_notional', 'slippage_pct', 'borrow_rate_annual',
                'max_pct_adv', 'adverse_fill', 'adverse_fill_pct',
                'partial_cover_frac', 'bounce_entry_pct',
                'limit_orders_only', 'limit_offset_pct',
                # Adaptive stops
                'adaptive_stops', 'stop_gap_fraction', 'stop_min_pct', 'stop_max_pct',
                # Market regime filter
                'regime_filter', 'regime_spy_gap_limit', 'regime_spy_block_pct', 'regime_vix_threshold',
                # Re-entry after stop-out
                'reentry_enabled', 'reentry_cooldown_minutes', 'reentry_max_per_symbol',
                'reentry_stop_pct', 'reentry_trigger_pct',
                # Gap-down fading
                'trade_gap_downs', 'gap_down_threshold', 'gap_down_max_pct', 'gap_down_vol_ratio_max',
                # Drawdown circuit breakers
                'dd_circuit_breaker', 'dd_tier1_threshold', 'dd_tier1_scale',
                'dd_tier2_threshold', 'dd_tier2_scale', 'dd_tier2_max_positions',
                'dd_hard_stop']:
        if key in body:
            field_type = type(getattr(config, key))
            try:
                setattr(config, key, field_type(body[key]))
            except (ValueError, TypeError):
                config_errors.append(f"{key}: cannot convert '{body[key]}' to {field_type.__name__}")
    if config_errors:
        return {'error': 'Config errors: ' + '; '.join(config_errors)}

    # Cross-field validation for drawdown circuit breakers
    if config.dd_circuit_breaker:
        if config.dd_tier1_threshold >= config.dd_tier2_threshold:
            return {'error': f'DD Tier 1 ({config.dd_tier1_threshold:.0%}) must be less than Tier 2 ({config.dd_tier2_threshold:.0%})'}
        if config.dd_hard_stop > 0 and config.dd_hard_stop <= config.dd_tier2_threshold:
            return {'error': f'DD Hard Stop ({config.dd_hard_stop:.0%}) must be greater than Tier 2 ({config.dd_tier2_threshold:.0%})'}

    if symbols and isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(',')]

    # Resolve universe if no explicit symbols provided
    bt_universe = body.get('universe', '')
    if not symbols and not symbol and bt_universe:
        if bt_universe == 'alpaca':
            symbols = fetch_alpaca_assets(min_price=config.min_price)
        elif bt_universe == 'study':
            symbols = UNIVERSE
        # 'custom' handled via symbols param directly

    async def progress_cb(pct, msg):
        await broadcast({'type': 'backtest_progress', 'progress': round(pct, 1), 'message': msg})

    bt_strategy_id = body.get('strategy_id', '')
    use_1min = body.get('use_1min', False)
    if _USE_VBT_BACKTESTER and not use_1min:
        backtester = _VbtBacktester(config, strategy_id=bt_strategy_id)
    else:
        backtester = GapFadeBacktester(config, strategy_id=bt_strategy_id)

    async def _run_bt():
        try:
            logger.info(f"[BT-TASK] Starting backtest run (vbt={_USE_VBT_BACKTESTER}, type={type(backtester).__name__})")
            result = await backtester.run(
                symbol=symbol, symbols=symbols,
                start_date=start_date, end_date=end_date,
                config=config, progress_callback=progress_cb,
                use_1min=use_1min,
                strategy_id=bt_strategy_id,
                strategy_config=body.get('strategy_config'),
            )
            logger.info(f"[BT-TASK] Backtest run() returned: {result.get('total_trades', '?')} trades, status={backtester.status}")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Backtest task crashed: {e}\n{tb}")
            backtester.status = 'error'
            backtester.result = {'error': str(e), 'traceback': tb}
            await broadcast({'type': 'backtest_complete', 'error': str(e), 'traceback': tb})

    asyncio.create_task(_run_bt())
    return {'status': 'started'}


@app.get("/api/backtest/status")
async def backtest_status():
    return {'progress': backtester.progress, 'status': backtester.status}


@app.get("/api/backtest/results")
async def backtest_results():
    """Return backtest results once complete."""
    if backtester.status == 'done' and backtester.result is not None:
        return backtester.result
    return {'status': backtester.status, 'progress': backtester.progress}


@app.post("/api/backtest/cancel")
async def cancel_backtest():
    backtester.cancel()
    return {'status': 'cancelled'}


# ---------------------------------------------------------------------------
# Backtrader visual backtester (1-20 symbols, chart generation)
# ---------------------------------------------------------------------------
bt_runner_status: dict = {'status': 'idle'}


@app.post("/api/backtest/bt")
async def run_bt_backtest(request: Request):
    """Start a backtrader backtest for chart generation (max 20 symbols)."""
    global bt_runner_status
    try:
        from bt_backtest import run_backtrader_backtest, HAS_BACKTRADER
    except ImportError:
        return JSONResponse({'status': 'error', 'error': 'bt_backtest module not found'}, 500)

    if not HAS_BACKTRADER:
        return JSONResponse({'status': 'error', 'error': 'backtrader not installed. Run: pip install backtrader'}, 500)

    if bt_runner_status.get('status') == 'running':
        return JSONResponse({'status': 'error', 'error': 'Backtrader backtest already running'}, 409)

    body = await request.json()
    symbols_raw = body.get('symbols', '')
    if isinstance(symbols_raw, str):
        symbols = [s.strip().upper() for s in symbols_raw.split(',') if s.strip()]
    else:
        symbols = [s.strip().upper() for s in symbols_raw if s.strip()]

    if not symbols:
        return JSONResponse({'status': 'error', 'error': 'No symbols provided'}, 400)
    if len(symbols) > 20:
        return JSONResponse({'status': 'error', 'error': f'Too many symbols ({len(symbols)}). Max 20.'}, 400)

    start_date = body.get('start_date', '')
    end_date = body.get('end_date', '')
    if not start_date or not end_date:
        return JSONResponse({'status': 'error', 'error': 'start_date and end_date required'}, 400)

    # Build config from defaults + overrides
    config = {}
    cfg = live_trader.engine.config
    for key in ('gap_threshold', 'max_gap_pct', 'vol_ratio_max', 'stop_pct',
                'adaptive_stops', 'stop_gap_fraction', 'stop_min_pct', 'stop_max_pct',
                'partial_cover_frac', 'trade_gap_downs', 'gap_down_threshold',
                'risk_pct', 'max_positions', 'slippage_pct', 'initial_capital'):
        config[key] = getattr(cfg, key, None)
    # Apply body overrides
    for k, v in body.get('config', {}).items():
        config[k] = v

    db_path = get_price_db().DB_PATH
    charts_dir = os.path.join(os.path.dirname(__file__) or '.', 'bt_charts')

    bt_runner_status = {'status': 'running', 'symbols': symbols, 'start': start_date, 'end': end_date}

    async def _run():
        global bt_runner_status
        try:
            result = await asyncio.to_thread(
                run_backtrader_backtest, symbols, start_date, end_date, config, db_path, charts_dir
            )
            if result['status'] == 'ok':
                bt_runner_status = {
                    'status': 'ok',
                    'metrics': result['metrics'],
                    'charts': result['charts'],
                    'trades': result['trades'],
                    'symbols_loaded': result['symbols_loaded'],
                    'errors': result.get('errors', []),
                }
            else:
                bt_runner_status = {'status': 'error', 'error': result.get('error', 'Unknown error')}
        except Exception as exc:
            logger.exception('Backtrader backtest failed')
            bt_runner_status = {'status': 'error', 'error': str(exc)}

    asyncio.create_task(_run())
    return {'status': 'started'}


@app.get("/api/backtest/bt/status")
async def bt_backtest_status():
    """Return backtrader backtest status, metrics, and chart paths."""
    return bt_runner_status


@app.get("/api/backtest/bt/charts/{filename}")
async def bt_chart_file(filename: str):
    """Serve a backtrader chart PNG."""
    if not re.match(r'^[a-zA-Z0-9_.-]+\.png$', filename):
        return JSONResponse({'error': 'Invalid filename'}, 400)
    charts_dir = os.path.join(os.path.dirname(__file__) or '.', 'bt_charts')
    path = os.path.join(charts_dir, filename)
    if not os.path.isfile(path):
        return JSONResponse({'error': 'Chart not found'}, 404)
    return FileResponse(path, media_type='image/png')


@app.get("/api/metrics")
async def get_metrics():
    return live_trader.engine.get_metrics()


@app.get("/api/trades")
async def get_trades():
    try:
        total_trades = get_price_db().count_trades()
    except Exception:
        total_trades = len(live_trader.engine.all_trade_log)
    return {
        'trades': [asdict(t) for t in live_trader.engine.all_trade_log[-200:]],
        'today': [asdict(t) for t in live_trader.engine.trade_log],
        'total_trades': total_trades,
    }


@app.get("/api/trades/history")
async def get_trades_history(
    start: str = None, end: str = None, symbol: str = None,
    limit: int = 100, offset: int = 0
):
    """Paginated full trade history from SQLite."""
    limit = max(1, min(limit, 1000))
    offset = max(0, offset)
    db = get_price_db()
    return db.query_trades(start_date=start, end_date=end,
                           symbol=symbol, limit=limit, offset=offset)


@app.get("/api/account")
async def get_account():
    """Get Alpaca paper account info."""
    acct = alpaca_get_account()
    if acct:
        return {
            'equity': acct.get('equity'),
            'buying_power': acct.get('buying_power'),
            'cash': acct.get('cash'),
            'portfolio_value': acct.get('portfolio_value'),
        }
    return {'error': 'Could not fetch account'}


# ── Price Database Endpoints ──────────────────────────────────────

@app.get("/api/db/stats")
async def db_stats():
    """Get price database statistics."""
    db = get_price_db()
    return db.get_stats()


@app.post("/api/db/build")
async def db_build(body: dict):
    """Bulk-load daily bars from Alpaca into local DB.

    Body: {universe: 'alpaca'|'study', start_date: 'YYYY-MM-DD'}
    Streams progress via WebSocket (db_build_progress messages).
    """
    universe = body.get('universe', 'study')
    start_date = body.get('start_date', (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d'))
    # Validate start_date
    if start_date:
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
        except ValueError:
            return {'error': f'Invalid start_date: "{start_date}" — expected YYYY-MM-DD format'}
    end_date = datetime.now().strftime('%Y-%m-%d')

    if universe == 'alpaca':
        symbols = fetch_alpaca_assets(min_price=1.0)
    elif universe == 'study':
        # UNIVERSE is defined later; import at call time
        symbols = UNIVERSE
    else:
        return {'error': f'Unknown universe: {universe}'}

    if not symbols:
        return {'error': 'No symbols resolved for universe'}

    db = get_price_db()
    total = len(symbols)
    batch_size = 100
    loaded = 0

    await broadcast({'type': 'db_build_progress', 'progress': 0,
                     'message': f'Building DB: 0/{total} symbols...'})

    for i in range(0, total, batch_size):
        batch = symbols[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        try:
            dfs = fetch_alpaca_bars_multi(batch, start_date, end_date, '1Day', 'iex')
            if dfs:
                db.upsert_bars_batch(dfs)
                loaded += len(dfs)
        except Exception as e:
            logger.warning(f"DB build batch {batch_num} error: {e}")

        pct = min(99, (i + len(batch)) / total * 100)
        await broadcast({'type': 'db_build_progress', 'progress': round(pct, 1),
                         'message': f'Building DB: {loaded}/{total} symbols (batch {batch_num}/{total_batches})...'})

    await broadcast({'type': 'db_build_progress', 'progress': 100,
                     'message': f'Done: {loaded} symbols loaded'})

    return db.get_stats()


@app.post("/api/db/update")
async def db_update():
    """Incremental update — fetch only new bars since last stored date per symbol.

    Typically 1-2 new bars per symbol, completes in ~30-60s.
    """
    db = get_price_db()
    last_dates = db.get_last_dates_batch()
    if not last_dates:
        return {'error': 'DB is empty — use Build first'}

    symbols = list(last_dates.keys())
    today = datetime.now().strftime('%Y-%m-%d')
    total = len(symbols)
    batch_size = 100
    updated = 0

    await broadcast({'type': 'db_build_progress', 'progress': 0,
                     'message': f'Updating: 0/{total} symbols...'})

    for i in range(0, total, batch_size):
        batch = symbols[i:i + batch_size]
        batch_num = i // batch_size + 1

        # Determine the earliest "last date + 1 day" for this batch
        start_dates = []
        for sym in batch:
            ld = last_dates.get(sym, '2020-01-01')
            next_day = (datetime.strptime(ld, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            start_dates.append(next_day)
        batch_start = min(start_dates)

        if batch_start > today:
            # All symbols in batch are up to date
            updated += len(batch)
        else:
            try:
                dfs = fetch_alpaca_bars_multi(batch, batch_start, today, '1Day', 'iex')
                if dfs:
                    db.upsert_bars_batch(dfs)
                    updated += len(dfs)
            except Exception as e:
                logger.warning(f"DB update batch {batch_num} error: {e}")

        pct = min(99, (i + len(batch)) / total * 100)
        await broadcast({'type': 'db_build_progress', 'progress': round(pct, 1),
                         'message': f'Updating: {min(i + len(batch), total)}/{total} symbols...'})

    await broadcast({'type': 'db_build_progress', 'progress': 100,
                     'message': f'Update complete: {updated} symbols refreshed'})

    return db.get_stats()


@app.post("/api/db/polygon-update")
async def db_polygon_update(body: dict = {}):
    """Update DB using Polygon/Massive grouped daily — one API call per day.

    Body params:
        days: number of trading days to backfill (default 1 = yesterday only)
    """
    api_key = _get_polygon_key()
    if not api_key:
        return {'error': 'POLYGON_API_KEY or MASSIVE_API_KEY not set in .env'}

    db = get_price_db()
    days = body.get('days', 1)
    today = datetime.now(ET)
    total_inserted = 0
    days_fetched = 0

    await broadcast({'type': 'db_build_progress', 'progress': 0,
                     'message': f'Polygon: fetching {days} day(s) of data...'})

    # Walk backwards from yesterday
    current = today - timedelta(days=1)
    attempts = 0
    while days_fetched < days and attempts < days + 10:
        attempts += 1
        # Skip weekends
        if current.weekday() >= 5:
            current -= timedelta(days=1)
            continue

        ds = current.strftime('%Y-%m-%d')
        try:
            bars = fetch_polygon_grouped_daily(ds)
            if bars:
                db.upsert_bars_dicts(bars)
                total_inserted += len(bars)
                days_fetched += 1

                pct = min(99, days_fetched / days * 100)
                await broadcast({'type': 'db_build_progress', 'progress': round(pct, 1),
                                 'message': f'Polygon: {ds} — {len(bars)} tickers ({days_fetched}/{days} days)'})
            else:
                logger.info(f"Polygon: no data for {ds} (holiday?), skipping")
        except Exception as e:
            logger.warning(f"Polygon fetch error for {ds}: {e}")
            await broadcast({'type': 'db_build_progress', 'progress': -1,
                             'message': f'Polygon error: {e}'})
            if 'invalid' in str(e).lower() or '403' in str(e):
                return {'error': str(e)}

        current -= timedelta(days=1)

        # Rate limit: 5 req/min on free tier
        if days_fetched < days:
            await asyncio.sleep(13)

    await broadcast({'type': 'db_build_progress', 'progress': 100,
                     'message': f'Polygon update complete: {total_inserted} rows across {days_fetched} days'})

    stats = db.get_stats()
    stats['polygon_inserted'] = total_inserted
    stats['polygon_days'] = days_fetched
    return stats


# =============================================================================
# SECTION 9b: POSITION TRACKER API
# =============================================================================

_tracker_watchlist: set = set()


@app.get("/api/tracker/bars")
async def get_tracker_bars(symbol: str, timeframe: str = '1Min', limit: int = 390):
    """Fetch historical OHLCV bars from Alpaca for a symbol/timeframe."""
    allowed_tf = {'1Min', '5Min', '15Min', '1Hour', '1Day'}
    if timeframe not in allowed_tf:
        return {'error': f'Invalid timeframe. Use: {allowed_tf}'}
    if limit < 1 or limit > 2000:
        limit = min(max(limit, 1), 2000)

    today = datetime.now()
    if timeframe == '1Day':
        days_back = int(limit * 1.5) + 10
    elif timeframe == '1Hour':
        days_back = int(limit / 6.5) + 5
    else:
        days_back = 5

    start_date = (today - timedelta(days=days_back)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    df = await asyncio.to_thread(
        fetch_alpaca_bars, symbol, start_date, end_date, timeframe, 'iex'
    )
    if df is None or df.empty:
        return {'symbol': symbol, 'timeframe': timeframe, 'bars': []}

    bars = []
    for idx, row in df.tail(limit).iterrows():
        ts = int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(pd.Timestamp(idx).timestamp())
        bars.append({
            't': ts,
            'o': round(float(row['open']), 4),
            'h': round(float(row['high']), 4),
            'l': round(float(row['low']), 4),
            'c': round(float(row['close']), 4),
            'v': int(row['volume']),
        })

    return {'symbol': symbol, 'timeframe': timeframe, 'bars': bars}


@app.get("/api/tracker/positions")
async def get_tracker_positions():
    """Get all Alpaca broker positions + account info."""
    positions = await asyncio.to_thread(alpaca_get_positions)
    account = await asyncio.to_thread(alpaca_get_account)
    return {
        'positions': positions or [],
        'equity': float(account.get('equity', 0)) if account else 0,
        'buying_power': float(account.get('buying_power', 0)) if account else 0,
    }


@app.post("/api/tracker/order")
async def place_tracker_order(request: Request):
    """Place a market order via Alpaca (Buy/Short from tracker UI)."""
    body = await request.json()
    symbol = (body.get('symbol') or '').upper().strip()
    side = body.get('side', '')
    qty = int(body.get('qty', 0))

    if not symbol:
        return {'error': 'Symbol required'}
    if side not in ('buy', 'sell'):
        return {'error': 'Side must be buy or sell'}
    if qty <= 0:
        return {'error': 'Qty must be positive'}

    result = await alpaca_submit_and_confirm(symbol, qty, side)
    return {
        'status': result.status,
        'filled_qty': result.filled_qty,
        'filled_avg_price': result.filled_avg_price,
        'error': result.error,
        'order_id': result.order_id,
    }


@app.post("/api/tracker/watchlist")
async def update_tracker_watchlist(request: Request):
    """Update symbols being tracked for real-time ticks.

    Reuses the live trader's existing AlpacaTickStreamer connection
    (Alpaca IEX allows only ONE WebSocket per API key). Dynamically
    subscribes tracker symbols on the shared connection.
    """
    global _tracker_watchlist
    body = await request.json()
    symbols = [s.upper().strip() for s in body.get('symbols', []) if s.strip()]
    _tracker_watchlist = set(symbols)

    streamer = getattr(live_trader, 'streamer', None)
    if streamer:
        # Reuse existing streamer — add new symbols dynamically
        new_syms = _tracker_watchlist - set(streamer.symbols)
        if new_syms:
            await streamer.add_symbols(list(new_syms))
    elif _tracker_watchlist:
        # No streamer exists yet — create one (only if trading loop hasn't started)
        cfg = _get_alpaca_config()
        if cfg:
            streamer = AlpacaTickStreamer(list(_tracker_watchlist), on_tick=_tracker_on_tick)
            live_trader.streamer = streamer
            await streamer.start()

    return {'watchlist': sorted(_tracker_watchlist), 'streaming': bool(streamer and streamer.connected)}


async def _tracker_on_tick(symbol: str, price: float, size: int = 0):
    """Tick handler that broadcasts tracker ticks and forwards to live trader."""
    engine = getattr(live_trader, 'engine', None)
    if engine and symbol in getattr(engine, 'positions', {}):
        orig_handler = getattr(live_trader, '_on_tick_impl', None)
        if orig_handler:
            await orig_handler(symbol, price, size)

    if symbol in _tracker_watchlist:
        await broadcast({
            'type': 'tracker_tick',
            'symbol': symbol,
            'price': price,
            'timestamp': _time.time(),
        })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live updates (checks JWT cookie when auth is enabled)."""
    if auth_enabled():
        user = get_current_user(websocket)
        api_key = _get_api_key()
        key_in_query = websocket.query_params.get('apiKey', '')
        if not user and not (api_key and key_in_query == api_key):
            await websocket.close(code=4401, reason="Unauthorized")
            return
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == 'ping':
                await websocket.send_text('pong')
    except WebSocketDisconnect:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)


# =============================================================================
# SECTION 10: DASHBOARD HTML
# =============================================================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gap Fade Terminal</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {
    --bg: #0A0E17;
    --surface: rgba(15, 23, 42, 0.7);
    --border: rgba(0, 212, 255, 0.15);
    --border-hover: rgba(0, 212, 255, 0.35);
    --text: #e2e8f0;
    --muted: #64748b;
    --positive: #00D4FF;
    --negative: #FF3B5C;
    --green: #00D4FF;
    --red: #FF3B5C;
    --blue: #00D4FF;
    --yellow: #eab308;
    --purple: #a855f7;
    --orange: #f97316;
    --cyan: #00D4FF;
    --grid: 8px;
    --accent-green: #22c55e;
    --sidebar-bg: rgba(8, 11, 20, 0.95);
    --sidebar-hover: rgba(34, 197, 94, 0.08);
    --sidebar-active: rgba(34, 197, 94, 0.15);
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 13px;
    line-height: 1.5;
    height: 100vh;
    overflow: hidden;
    -webkit-font-smoothing: antialiased;
  }

  /* ── Layout Grid ── */
  #app {
    display: grid;
    grid-template-columns: 180px 1fr 300px;
    grid-template-rows: 48px auto 1fr 28px;
    grid-template-areas:
      "sidebar header    rightpanel"
      "sidebar metrics   rightpanel"
      "sidebar content   rightpanel"
      "statusbar statusbar statusbar";
    height: 100vh;
  }

  /* ── Sidebar ── */
  .sidebar {
    grid-area: sidebar;
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .sidebar-logo {
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }
  .sidebar-logo .brand-text { font-weight: 700; font-size: 16px; color: #fff; }
  .sidebar-logo .brand-sub { font-weight: 400; font-size: 11px; color: var(--muted); margin-left: 4px; }
  .sidebar-actions { padding: 10px 12px 4px; }
  .sidebar-actions button { width: 100%; padding: 8px 0; font-size: 12px; }
  .sidebar-nav { flex: 1; padding: 8px 0; overflow-y: auto; }
  .sidebar-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    cursor: pointer;
    color: var(--muted);
    font-size: 12px;
    font-weight: 500;
    border-left: 3px solid transparent;
    transition: all 0.15s;
    user-select: none;
  }
  .sidebar-item:hover { background: var(--sidebar-hover); color: var(--text); }
  .sidebar-item.active {
    background: var(--sidebar-active);
    color: var(--accent-green);
    border-left-color: var(--accent-green);
  }
  .sidebar-item svg { width: 16px; height: 16px; flex-shrink: 0; }
  .sidebar-sep { height: 1px; background: var(--border); margin: 4px 16px; }
  .sidebar-footer {
    padding: 12px 16px;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 10px;
  }

  /* ── Top Header ── */
  .top-bar {
    grid-area: header;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
    border-bottom: 1px solid var(--border);
    background: rgba(10, 14, 23, 0.95);
    backdrop-filter: blur(12px);
  }
  .top-bar .page-title { font-weight: 600; font-size: 15px; color: #fff; }
  .top-bar .header-controls { display: flex; align-items: center; gap: 6px; }
  .top-bar .right { display: flex; align-items: center; gap: 16px; }
  .conn-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--negative);
    transition: background 0.2s, box-shadow 0.2s;
  }
  .conn-dot.connected {
    background: var(--positive);
    box-shadow: 0 0 8px rgba(0, 212, 255, 0.6);
    animation: pulse-dot 2s ease-in-out infinite;
  }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.85; box-shadow: 0 0 12px rgba(0, 212, 255, 0.4); }
  }
  #clock { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--muted); }
  .status-badge {
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .status-stopped { background: rgba(255,255,255,0.08); color: var(--muted); }
  .status-waiting { background: rgba(0, 212, 255, 0.12); color: var(--positive); }
  .status-scanning { background: rgba(0, 212, 255, 0.2); color: var(--positive); }
  .status-trading { background: rgba(0, 212, 255, 0.2); color: var(--positive); }
  .status-paused { background: rgba(234, 179, 8, 0.2); color: var(--yellow); }
  .status-halted { background: rgba(255, 59, 92, 0.2); color: var(--negative); }
  .status-standdown { background: rgba(168, 85, 247, 0.2); color: #c084fc; }
  .llm-badge {
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.5px;
    font-family: 'JetBrains Mono', monospace;
  }
  .llm-badge.llm-on { background: rgba(168, 85, 247, 0.2); color: #c084fc; }
  .llm-badge.llm-circuit { background: rgba(234, 179, 8, 0.2); color: #fde68a; }
  .llm-badge.llm-off { background: rgba(255,255,255,0.06); color: var(--muted); }

  .strategy-badge {
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.5px;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.2);
  }

  /* ── Chat ── */
  .chat-msg { padding: 10px 14px; border-radius: 10px; font-size: 13px; line-height: 1.6; max-width: 85%; white-space: pre-wrap; word-wrap: break-word; }
  .chat-user { background: rgba(168, 85, 247, 0.15); color: var(--text); align-self: flex-end; border-bottom-right-radius: 2px; }
  .chat-assistant { background: var(--surface); color: var(--text); align-self: flex-start; border: 1px solid var(--border); border-bottom-left-radius: 2px; }
  .chat-system { background: rgba(6,182,212,0.08); color: var(--muted); align-self: center; text-align: center; font-size: 12px; border-radius: 6px; }
  .chat-error { background: rgba(239,68,68,0.12); color: #fca5a5; align-self: center; text-align: center; font-size: 12px; }
  .chat-thinking { background: var(--surface); color: var(--muted); align-self: flex-start; font-style: italic; animation: pulse 1.5s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity: 0.5; } 50% { opacity: 1; } }
  .chat-suggestion { padding: 6px 12px; background: rgba(168,85,247,0.1); border: 1px solid rgba(168,85,247,0.25); border-radius: 16px; color: #c084fc; font-size: 12px; cursor: pointer; transition: all 0.15s; }
  .chat-suggestion:hover { background: rgba(168,85,247,0.2); border-color: rgba(168,85,247,0.4); }

  /* ── Metrics Bar ── */
  .metrics-bar {
    grid-area: metrics;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border-bottom: 1px solid var(--border);
    background: rgba(10, 14, 23, 0.8);
    overflow-x: auto;
  }
  .metric-card {
    display: flex;
    flex-direction: column;
    padding: 4px 12px;
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid var(--border);
    border-radius: 6px;
    min-width: 90px;
    white-space: nowrap;
  }
  .metric-card .label { font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .metric-card .value { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; margin-top: 1px; }
  .metric-card .value.green { color: var(--positive); }
  .metric-card .value.red { color: var(--negative); }

  /* ── Main Content ── */
  .main-content {
    grid-area: content;
    overflow-y: auto;
    padding: 12px 16px;
  }
  .page { display: none; }
  .page.active { display: block; }

  /* ── Right Panel ── */
  .right-panel {
    grid-area: rightpanel;
    background: var(--sidebar-bg);
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
  }
  .right-panel-resize {
    position: absolute;
    left: -3px;
    top: 0;
    width: 6px;
    height: 100%;
    cursor: col-resize;
    z-index: 10;
  }
  .right-panel-resize:hover,
  .right-panel-resize.dragging { background: var(--accent-green); opacity: 0.4; }
  .right-panel-tabs {
    display: flex;
    border-bottom: 1px solid var(--border);
  }
  .right-panel-tab {
    flex: 1;
    padding: 10px 0;
    text-align: center;
    cursor: pointer;
    color: var(--muted);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 2px solid transparent;
    transition: all 0.15s;
  }
  .right-panel-tab:hover { color: var(--text); }
  .right-panel-tab.active { color: var(--accent-green); border-bottom-color: var(--accent-green); }
  .right-tab-content { display: none; flex: 1; overflow: auto; padding: 8px; font-size: 11px; }
  .right-tab-content.active { display: flex; flex-direction: column; }

  /* ── Cards ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s;
  }
  .card:hover { border-color: var(--border-hover); }
  .card h2 {
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: var(--grid);
  }
  .stat {
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid var(--border);
    padding: var(--grid) 10px;
    border-radius: 6px;
  }
  .stat .label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .stat .value { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; margin-top: 2px; }
  .stat .value.green { color: var(--positive); }
  .stat .value.red { color: var(--negative); }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  th {
    text-align: left;
    padding: 8px 10px;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  td {
    padding: 8px 10px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-variant-numeric: tabular-nums;
  }
  tbody tr:nth-child(even) { background: rgba(255,255,255,0.02); }
  tr:hover { background: rgba(255,255,255,0.06); }

  .controls {
    display: flex;
    gap: var(--grid);
    flex-wrap: wrap;
  }
  button {
    padding: 6px 14px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: rgba(0, 0, 0, 0.2);
    color: var(--text);
    cursor: pointer;
    font-size: 12px;
    font-family: inherit;
    transition: all 0.2s;
  }
  button:hover { background: rgba(255,255,255,0.08); border-color: var(--border-hover); }
  button:focus-visible { outline: 2px solid var(--positive); outline-offset: 2px; }
  button.primary { background: rgba(0, 212, 255, 0.15); border-color: rgba(0, 212, 255, 0.4); color: var(--positive); }
  button.primary:hover { background: rgba(0, 212, 255, 0.25); }
  button.danger { background: rgba(255, 59, 92, 0.15); border-color: rgba(255, 59, 92, 0.4); color: var(--negative); }
  button.danger:hover { background: rgba(255, 59, 92, 0.25); }
  button.success { background: rgba(34, 197, 94, 0.15); border-color: rgba(34, 197, 94, 0.4); color: var(--accent-green); }
  button.success:hover { background: rgba(34, 197, 94, 0.25); }

  .toast {
    position: fixed;
    top: 16px;
    right: 16px;
    z-index: 9999;
    padding: 12px 18px;
    border-radius: 8px;
    font-size: 13px;
    font-family: inherit;
    max-width: 420px;
    word-wrap: break-word;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    animation: toastIn 0.3s ease-out;
    cursor: pointer;
    border: 1px solid;
  }
  .toast-error { background: rgba(255, 59, 92, 0.92); color: #fff; border-color: rgba(255, 59, 92, 0.8); }
  .toast-warning { background: rgba(234, 179, 8, 0.92); color: #000; border-color: rgba(234, 179, 8, 0.8); }
  .toast-success { background: rgba(0, 160, 200, 0.92); color: #fff; border-color: rgba(0, 212, 255, 0.8); }
  @keyframes toastIn { from { opacity: 0; transform: translateX(24px); } to { opacity: 1; transform: translateX(0); } }

  .feed {
    max-height: 100%;
    overflow-y: auto;
    font-size: 12px;
    flex: 1;
  }
  .feed-item {
    padding: 8px 10px;
    border-left: 3px solid var(--border);
    margin-bottom: 4px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 0 6px 6px 0;
    transition: background 0.2s;
  }
  .feed-item:hover { background: rgba(255,255,255,0.04); }
  .feed-item.entry { border-left-color: var(--negative); }
  .feed-item.exit { border-left-color: var(--positive); }
  .feed-item.scan { border-left-color: var(--positive); }
  .feed-item.system { border-left-color: var(--purple); }
  .feed-item.error { border-left-color: var(--negative); }
  .feed-item.warning { border-left-color: #eab308; }
  .feed-item.catalyst { border-left-color: #f97316; }
  .feed-item.regime { border-left-color: #f97316; }
  .feed-item.llm { border-left-color: #a855f7; background: rgba(168, 85, 247, 0.06); }
  .feed-item.llm_fallback { border-left-color: #eab308; background: rgba(234, 179, 8, 0.06); }
  .feed-item .time { color: var(--muted); font-size: 10px; margin-right: 8px; font-family: 'JetBrains Mono', monospace; }

  .circuit-breakers {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .cb-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    border-radius: 6px;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid var(--border);
    font-size: 10px;
  }
  .cb-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }
  .cb-dot.ok { background: var(--positive); }
  .cb-dot.warn { background: var(--yellow); }
  .cb-dot.halt { background: var(--negative); }

  .pos-close-btn {
    background: rgba(255,59,48,0.15);
    border: 1px solid rgba(255,59,48,0.3);
    color: var(--negative);
    cursor: pointer;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 11px;
    font-weight: 600;
    transition: all 0.15s;
  }
  .pos-close-btn:hover {
    background: rgba(255,59,48,0.35);
    border-color: var(--negative);
  }

  input, select {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px 10px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    width: 80px;
  }
  input:focus, select:focus { outline: none; border-color: var(--positive); }
  label { font-size: 11px; color: var(--muted); margin-right: 4px; }

  .config-grid { display: flex; flex-direction: column; gap: 16px; }
  .cfg-section {
    border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
    background: rgba(255,255,255,0.02);
  }
  .cfg-section-head {
    padding: 10px 14px; font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.8px; color: var(--muted); background: rgba(255,255,255,0.03);
    border-bottom: 1px solid var(--border);
  }
  .cfg-section-body {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 2px; padding: 10px 12px;
  }
  .config-item { display: flex; flex-direction: column; gap: 3px; padding: 4px 0; }
  .config-item label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.3px; font-weight: 600; white-space: nowrap; }
  .config-item input[type="number"],
  .config-item input[type="text"] {
    background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 4px;
    padding: 6px 8px; font-size: 13px; font-family: 'JetBrains Mono', monospace; width: 100%;
    box-sizing: border-box; transition: border-color 0.15s;
  }
  .config-item input:focus { border-color: var(--accent-green); outline: none; }

  .feature-section { border: 1px solid var(--border); border-radius: 8px; border-left: 3px solid var(--muted); transition: border-color 0.2s; overflow: hidden; }
  .feature-section.enabled { border-left-color: var(--accent-green); }
  .feature-header { display: flex; align-items: center; gap: 10px; padding: 12px 14px; cursor: pointer; user-select: none; }
  .feature-header input[type="checkbox"] { width: 16px; height: 16px; accent-color: var(--accent-green); cursor: pointer; flex-shrink: 0; }
  .feature-title { font-weight: 700; font-size: 13px; color: var(--text); white-space: nowrap; }
  .feature-desc { font-size: 11px; color: var(--muted); line-height: 1.4; }
  .feature-params { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 2px; padding: 4px 14px 12px 14px; border-top: 1px solid var(--border); background: rgba(0,0,0,0.15); }

  .pnl-pos { color: var(--positive); }
  .pnl-neg { color: var(--negative); }

  .progress-bar {
    width: 100%;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 8px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: var(--positive);
    transition: width 0.3s ease;
    border-radius: 2px;
  }

  #backtest-panel { display: none; }
  #backtest-panel.active { display: block; }

  .equity-chart {
    width: 100%;
    height: 200px;
    position: relative;
  }
  .equity-chart canvas {
    width: 100%;
    height: 100%;
  }

  /* ── Status Bar ── */
  .status-bar {
    grid-area: statusbar;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
    border-top: 1px solid var(--border);
    background: rgba(10, 14, 23, 0.9);
    font-size: 11px;
    color: var(--muted);
  }
  .status-bar .left { display: flex; gap: 24px; }
  .status-bar .right { display: flex; gap: 16px; }
  .version-link { cursor: pointer; text-decoration: underline; text-decoration-style: dotted; text-underline-offset: 3px; }
  .version-link:hover { color: var(--cyan); }

  /* ── Release Notes Modal ── */
  .rn-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.7); z-index:9999; justify-content:center; align-items:center; }
  .rn-overlay.open { display:flex; }
  .rn-modal { background:var(--surface); border:1px solid var(--border); border-radius:8px; width:540px; max-height:80vh; display:flex; flex-direction:column; }
  .rn-header { display:flex; justify-content:space-between; align-items:center; padding:16px 20px; border-bottom:1px solid var(--border); }
  .rn-header h3 { margin:0; font-size:16px; color:var(--cyan); }
  .rn-close { background:none; border:none; color:var(--muted); font-size:20px; cursor:pointer; padding:0 4px; }
  .rn-close:hover { color:var(--text); }
  .rn-body { padding:20px; overflow-y:auto; font-size:13px; line-height:1.7; }
  .rn-body h4 { color:var(--cyan); margin:16px 0 8px 0; font-size:14px; }
  .rn-body h4:first-child { margin-top:0; }
  .rn-body ul { margin:4px 0 12px 0; padding-left:20px; }
  .rn-body li { margin:2px 0; color:var(--text); }
  .rn-body .rn-tag { display:inline-block; background:rgba(0,212,255,0.15); color:var(--cyan); padding:1px 8px; border-radius:4px; font-size:11px; font-family:var(--mono); margin-left:6px; }

  /* ── P&L granularity buttons ── */
  .pnl-gran-btn {
    padding: 3px 10px;
    font-size: 11px;
    border-radius: 4px;
    background: rgba(0,0,0,0.2);
    border: 1px solid var(--border);
    color: var(--muted);
    cursor: pointer;
    transition: all 0.15s;
  }
  .pnl-gran-btn:hover { color: var(--text); border-color: var(--border-hover); }
  .pnl-gran-btn.active { background: rgba(0,212,255,0.15); border-color: rgba(0,212,255,0.4); color: var(--positive); }

  /* ── Right panel table compact ── */
  .right-panel table { font-size: 11px; }
  .right-panel th { padding: 6px 6px; font-size: 9px; }
  .right-panel td { padding: 6px 6px; }

  /* ── Responsive ── */
  @media (max-width: 1100px) {
    #app {
      grid-template-columns: 1fr;
      grid-template-rows: 48px auto 1fr 28px;
      grid-template-areas:
        "header"
        "metrics"
        "content"
        "statusbar";
    }
    .sidebar { display: none; }
    .right-panel { display: none; }
    .main-content { padding: 8px; }
  }

  /* ── Position Tracker ── */
  .tracker-controls {
    display: flex; align-items: center; gap: 8px; margin-bottom: 12px;
  }
  .tracker-controls input {
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    padding: 6px 12px; border-radius: 6px; font-size: 12px; font-family: 'JetBrains Mono', monospace;
    outline: none; width: 180px;
  }
  .tracker-controls input:focus { border-color: var(--positive); }
  .tracker-controls .add-btn {
    padding: 6px 16px; background: rgba(0,212,255,0.12); color: var(--positive);
    border: 1px solid rgba(0,212,255,0.3); border-radius: 6px; font-size: 12px;
    font-weight: 600; cursor: pointer;
  }
  .tracker-controls .add-btn:hover { background: rgba(0,212,255,0.2); }
  .tracker-positions-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 12px; margin-bottom: 12px;
  }
  .tracker-positions-card h3 {
    font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 8px;
  }
  .tracker-positions-card table { width: 100%; border-collapse: collapse; font-size: 11px; }
  .tracker-positions-card th {
    text-align: left; color: var(--muted); font-weight: 500; padding: 4px 8px;
    border-bottom: 1px solid var(--border); font-size: 10px; text-transform: uppercase;
  }
  .tracker-positions-card td {
    padding: 5px 8px; border-bottom: 1px solid rgba(255,255,255,0.03);
    font-family: 'JetBrains Mono', monospace;
  }
  #trackerChartGrid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
  }
  @media (max-width: 1200px) { #trackerChartGrid { grid-template-columns: repeat(2, 1fr); } }
  @media (max-width: 800px) { #trackerChartGrid { grid-template-columns: 1fr; } }
  .tracker-chart-panel {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    overflow: hidden; transition: border-color 0.2s;
  }
  .tracker-chart-panel:hover { border-color: var(--border-hover); }
  .tracker-chart-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 10px; border-bottom: 1px solid var(--border); background: rgba(0,0,0,0.2);
  }
  .tracker-symbol {
    font-weight: 700; font-size: 14px; color: #fff; font-family: 'JetBrains Mono', monospace;
  }
  .tracker-price {
    font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600; margin-left: 8px;
  }
  .tracker-ohlc {
    font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--muted);
    flex: 1; text-align: center; white-space: nowrap; overflow: hidden;
  }
  .tracker-actions { display: flex; gap: 4px; }
  .tracker-buy {
    padding: 3px 10px; font-size: 10px; font-weight: 600;
    background: rgba(34,197,94,0.15); color: var(--accent-green);
    border: 1px solid rgba(34,197,94,0.3); border-radius: 4px; cursor: pointer;
  }
  .tracker-buy:hover { background: rgba(34,197,94,0.25); }
  .tracker-short {
    padding: 3px 10px; font-size: 10px; font-weight: 600;
    background: rgba(255,59,92,0.15); color: var(--red);
    border: 1px solid rgba(255,59,92,0.3); border-radius: 4px; cursor: pointer;
  }
  .tracker-short:hover { background: rgba(255,59,92,0.25); }
  .tracker-remove {
    padding: 3px 6px; background: none; border: 1px solid var(--border);
    color: var(--muted); border-radius: 4px; cursor: pointer; font-size: 12px; line-height: 1;
  }
  .tracker-remove:hover { color: var(--red); border-color: var(--red); }
  .tracker-tf-bar {
    display: flex; gap: 2px; padding: 4px 10px; background: rgba(0,0,0,0.15);
  }
  .tf-btn {
    padding: 2px 8px; font-size: 10px; font-weight: 600; background: none;
    border: 1px solid transparent; color: var(--muted); border-radius: 3px; cursor: pointer;
  }
  .tf-btn:hover { color: var(--text); }
  .tf-btn.active {
    background: rgba(0,212,255,0.12); color: var(--positive); border-color: rgba(0,212,255,0.3);
  }
</style>
</head>
<body>

<!-- ── Auth Login Overlay ── -->
<div id="authOverlay" style="display:none;position:fixed;inset:0;z-index:9999;background:#0A0E17;display:none;align-items:center;justify-content:center;">
  <div style="width:100%;max-width:360px;padding:32px;border-radius:16px;border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.03);backdrop-filter:blur(20px);box-shadow:0 0 80px rgba(0,212,255,0.06);text-align:center;">
    <h1 style="font-size:24px;font-weight:700;color:#fff;margin-bottom:8px;">Gap Fade Terminal</h1>
    <span style="display:inline-block;padding:2px 10px;border-radius:4px;font-size:12px;font-weight:600;background:rgba(0,212,255,0.2);color:#00D4FF;border:1px solid rgba(0,212,255,0.3);">Secure</span>
    <p style="color:#64748b;font-size:14px;margin:16px 0 24px;">Sign in to access your dashboard</p>
    <div style="display:flex;flex-direction:column;gap:12px;">
      <a href="/api/auth/login/google" style="display:flex;align-items:center;justify-content:center;gap:10px;width:100%;padding:10px;border-radius:8px;font-size:14px;font-weight:500;background:#fff;color:#1f2937;text-decoration:none;">
        <svg viewBox="0 0 24 24" width="20" height="20" fill="none"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4"/><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18A10.96 10.96 0 0 0 1 12c0 1.77.42 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"/><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/></svg>
        Continue with Google
      </a>
      <a href="/api/auth/login/github" style="display:flex;align-items:center;justify-content:center;gap:10px;width:100%;padding:10px;border-radius:8px;font-size:14px;font-weight:500;background:#24292e;color:#fff;text-decoration:none;">
        <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844a9.59 9.59 0 0 1 2.504.337c1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2z"/></svg>
        Continue with GitHub
      </a>
      <a href="/api/auth/login/discord" style="display:flex;align-items:center;justify-content:center;gap:10px;width:100%;padding:10px;border-radius:8px;font-size:14px;font-weight:500;background:#5865F2;color:#fff;text-decoration:none;">
        <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.947 2.418-2.157 2.418z"/></svg>
        Continue with Discord
      </a>
    </div>
  </div>
</div>

<!-- ── Access Denied Overlay ── -->
<div id="accessDeniedOverlay" style="display:none;position:fixed;inset:0;z-index:9999;background:#0A0E17;align-items:center;justify-content:center;">
  <div style="width:100%;max-width:360px;padding:32px;border-radius:16px;border:1px solid rgba(255,59,92,0.3);background:rgba(255,59,92,0.04);backdrop-filter:blur(20px);box-shadow:0 0 60px rgba(255,59,92,0.08);text-align:center;">
    <svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="#FF3B5C" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:16px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
    <h1 style="font-size:20px;font-weight:700;color:#fff;margin-bottom:8px;">Access Denied</h1>
    <p style="color:#64748b;font-size:14px;margin-bottom:24px;">Your account is not authorized. Contact the administrator.</p>
    <a href="/api/auth/login/google" style="font-size:14px;font-weight:500;color:#00D4FF;text-decoration:none;">Try a different account</a>
  </div>
</div>

<div id="app" style="display:none;">

<!-- ── Sidebar ── -->
<aside class="sidebar">
  <div class="sidebar-logo">
    <span class="brand-text">Gap Fade</span><span class="brand-sub">Terminal</span>
  </div>
  <div class="sidebar-actions">
    <button class="success" onclick="startTrading()">Start Trading</button>
  </div>
  <nav class="sidebar-nav">
    <div class="sidebar-item active" data-page="dashboard" onclick="showPage('dashboard')">
      <svg viewBox="0 0 16 16" fill="currentColor"><rect x="1" y="1" width="6" height="6" rx="1"/><rect x="9" y="1" width="6" height="6" rx="1"/><rect x="1" y="9" width="6" height="6" rx="1"/><rect x="9" y="9" width="6" height="6" rx="1"/></svg>
      Dashboard
    </div>
    <div class="sidebar-item" data-page="candidates" onclick="showPage('candidates')">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="7" cy="7" r="4.5"/><line x1="10.2" y1="10.2" x2="14" y2="14"/></svg>
      Scanner
    </div>
    <div class="sidebar-item" data-page="trades" onclick="showPage('trades')">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="2" width="12" height="12" rx="1.5"/><line x1="2" y1="6" x2="14" y2="6"/><line x1="6" y1="6" x2="6" y2="14"/></svg>
      Trade Log
    </div>
    <div class="sidebar-item" data-page="backtest" onclick="showPage('backtest')">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="2,12 5,6 9,9 14,3"/><polyline points="10,3 14,3 14,7"/></svg>
      Backtest
    </div>
    <div class="sidebar-sep"></div>
    <div class="sidebar-item" data-page="config" onclick="showPage('config')">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8" cy="8" r="2.5"/><path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M3.4 12.6l1.4-1.4M11.2 4.8l1.4-1.4"/></svg>
      Config
    </div>
    <div class="sidebar-item" data-page="database" onclick="showPage('database')">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><ellipse cx="8" cy="4" rx="5.5" ry="2.5"/><path d="M2.5 4v8c0 1.4 2.5 2.5 5.5 2.5s5.5-1.1 5.5-2.5V4"/><path d="M2.5 8c0 1.4 2.5 2.5 5.5 2.5s5.5-1.1 5.5-2.5"/></svg>
      Database
    </div>
    <div class="sidebar-item" data-page="tracker" onclick="showPage('tracker')">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="1,12 4,5 7,8 10,3 13,7 15,4"/><line x1="1" y1="14" x2="15" y2="14"/></svg>
      Tracker
    </div>
    <div class="sidebar-item" data-page="chat" onclick="showPage('chat')">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 3h12v8H5l-3 3V3z" rx="1.5"/><line x1="5" y1="6" x2="11" y2="6"/><line x1="5" y1="9" x2="9" y2="9"/></svg>
      Rudra Chat
    </div>
    <div class="sidebar-item" data-page="guide" onclick="showPage('guide')">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2.5" y="1.5" width="11" height="13" rx="1.5"/><line x1="5" y1="5" x2="11" y2="5"/><line x1="5" y1="8" x2="11" y2="8"/><line x1="5" y1="11" x2="9" y2="11"/></svg>
      Guide
    </div>
  </nav>
  <div class="sidebar-footer">
    <div class="conn-dot" id="connectionDot"></div>
    <span id="connLabel" style="color:var(--muted);">Connecting...</span>
  </div>
</aside>

<!-- ── Top Header ── -->
<header class="top-bar">
  <div style="display:flex;align-items:center;gap:12px;">
    <span class="page-title" id="pageTitle">Dashboard</span>
    <span id="statusBadge" class="status-badge status-stopped">STOPPED</span>
    <span id="llmBadge" class="llm-badge" style="display:none;" title="Rudra">Rudra</span>
    <span id="strategyBadge" class="strategy-badge" title="Active strategy">Classic</span>
  </div>
  <div class="header-controls">
    <select id="strategySelect" onchange="switchStrategy(this.value)" title="Strategy"
      style="background:#1a1f2e;color:#e2e8f0;border:1px solid rgba(255,255,255,0.1);border-radius:6px;padding:4px 8px;font-size:12px;cursor:pointer;max-width:160px;">
      <option value="classic_gap_fade">Classic Gap Fade</option>
    </select>
    <button class="primary" onclick="runScan()">Scan</button>
    <button onclick="pauseTrading()">Pause</button>
    <button onclick="resumeTrading()">Resume</button>
    <button class="danger" onclick="stopTrading()">Stop</button>
    <button onclick="resetTrader()">Reset</button>
  </div>
  <div class="right" style="display:flex;align-items:center;gap:12px;">
    <span id="clock"></span>
    <div id="userBadge" style="display:none;align-items:center;gap:8px;">
      <img id="userAvatar" src="" alt="" style="width:24px;height:24px;border-radius:50%;border:1px solid rgba(255,255,255,0.2);display:none;" referrerpolicy="no-referrer">
      <span id="userInitial" style="display:none;width:24px;height:24px;border-radius:50%;background:rgba(0,212,255,0.2);border:1px solid rgba(0,212,255,0.3);font-size:11px;font-weight:600;color:#00D4FF;line-height:24px;text-align:center;"></span>
      <span id="userName" style="font-size:12px;color:#94a3b8;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"></span>
      <button onclick="doLogout()" title="Sign out" style="background:none;border:none;cursor:pointer;padding:4px;border-radius:4px;color:#64748b;display:flex;align-items:center;" onmouseover="this.style.color='#FF3B5C'" onmouseout="this.style.color='#64748b'">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
      </button>
    </div>
  </div>
</header>

<!-- ── Metrics Row ── -->
<div class="metrics-bar">
  <div class="metric-card">
    <div class="label">Equity</div>
    <div class="value" id="statEquity">$100,000</div>
  </div>
  <div class="metric-card">
    <div class="label">Today P&L</div>
    <div class="value" id="statTodayPnl">$0.00</div>
  </div>
  <div class="metric-card">
    <div class="label">Total P&L</div>
    <div class="value" id="statTotalPnl">$0.00</div>
  </div>
  <div class="metric-card">
    <div class="label">Win Rate</div>
    <div class="value" id="statWinRate">0%</div>
  </div>
  <div class="metric-card">
    <div class="label">Trades</div>
    <div class="value" id="statTrades">0</div>
  </div>
  <div class="metric-card">
    <div class="label">Profit Factor</div>
    <div class="value" id="statPF">0</div>
  </div>
  <div class="metric-card">
    <div class="label">Max DD</div>
    <div class="value" id="statMaxDD">0%</div>
  </div>
  <div class="metric-card">
    <div class="label">Avg Hold</div>
    <div class="value" id="statAvgHold">0m</div>
  </div>
  <div style="margin-left:auto;"></div>
  <div class="circuit-breakers" id="circuitBreakers">
    <div class="cb-indicator"><div class="cb-dot ok" id="cbDaily"></div><span id="cbDailyVal">$0</span></div>
    <div class="cb-indicator"><div class="cb-dot ok" id="cbConsec"></div><span id="cbConsecVal">0</span></div>
    <div class="cb-indicator"><div class="cb-dot ok" id="cbDD"></div><span id="cbDDVal">0%</span></div>
  </div>
</div>

<!-- ── Main Content Area ── -->
<main class="main-content">

  <!-- Dashboard Page (default) -->
  <div class="page active" id="page-dashboard">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;">
      <div class="card">
        <h2>Account Overview</h2>
        <div style="display:flex;align-items:baseline;gap:20px;">
          <div>
            <div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px;">Equity</div>
            <div style="font-size:28px;font-family:'JetBrains Mono',monospace;font-weight:700;color:var(--positive);" id="dashEquity">$100,000</div>
          </div>
          <div>
            <div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px;">Today P&L</div>
            <div style="font-size:20px;font-family:'JetBrains Mono',monospace;font-weight:600;" id="dashTodayPnl">$0.00</div>
          </div>
        </div>
      </div>
      <div class="card">
        <h2>Session Stats</h2>
        <div class="stats-grid" style="grid-template-columns:repeat(2,1fr);">
          <div class="stat"><div class="label">Win Rate</div><div class="value" id="dashWinRate">0%</div></div>
          <div class="stat"><div class="label">Trades</div><div class="value" id="dashTrades">0</div></div>
          <div class="stat"><div class="label">Profit Factor</div><div class="value" id="dashPF">0</div></div>
          <div class="stat"><div class="label">Max DD</div><div class="value" id="dashMaxDD">0%</div></div>
        </div>
      </div>
    </div>
    <div class="card">
      <h2>Equity Curve</h2>
      <div class="equity-chart" style="position:relative;"><canvas id="equityChart"></canvas>
        <div id="eqTooltip" style="display:none;position:absolute;pointer-events:none;background:rgba(20,20,30,0.95);border:1px solid rgba(255,255,255,0.2);border-radius:4px;padding:4px 8px;font-size:11px;font-family:'JetBrains Mono',monospace;color:#fff;white-space:nowrap;z-index:10;"></div>
        <div id="eqCrosshair" style="display:none;position:absolute;top:0;width:1px;height:100%;background:rgba(255,255,255,0.25);pointer-events:none;z-index:9;"></div>
      </div>
      <div style="color:var(--muted);text-align:center;padding:16px;font-size:12px;">Equity curve updates as trades execute</div>
    </div>
  </div>

  <!-- Scanner Page -->
  <div class="page" id="page-candidates">
    <h2 style="margin-bottom:10px;">Gap Candidates <span style="color:var(--muted);font-size:11px;" id="scanTime"></span> <span style="color:var(--muted);font-size:11px;" id="universeInfo"></span></h2>
    <table>
      <thead><tr>
        <th>Symbol</th><th>Dir</th><th>Gap %</th><th>Prev Close</th><th>Current</th><th>Vol Ratio</th>
        <th>Avg Vol</th><th>Shortable</th><th>ETB</th><th>Catalyst</th><th>Score</th>
      </tr></thead>
      <tbody id="candidatesTable"></tbody>
    </table>
    <div id="noCandidates" style="color:var(--muted);padding:20px;text-align:center;">No candidates — run a scan</div>
  </div>

  <!-- Trade Log Page -->
  <div class="page" id="page-trades">
    <h2 style="margin-bottom:10px;">Trade History <span id="tradeHistCount" style="font-size:13px;color:var(--muted);font-weight:400;"></span></h2>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;align-items:flex-end;">
      <div class="config-item"><label>Start</label><input type="date" id="thStart" style="width:130px;"></div>
      <div class="config-item"><label>End</label><input type="date" id="thEnd" style="width:130px;"></div>
      <div class="config-item"><label>Symbol</label><input type="text" id="thSymbol" placeholder="e.g. TSLA" style="width:80px;text-transform:uppercase;"></div>
      <button onclick="tradeHistSearch(0)" style="padding:4px 14px;background:var(--green);color:#000;border:none;border-radius:4px;cursor:pointer;font-weight:600;font-size:12px;">Search</button>
      <button onclick="tradeHistReset()" style="padding:4px 10px;background:var(--border);color:var(--text);border:none;border-radius:4px;cursor:pointer;font-size:12px;">Reset</button>
    </div>
    <table>
      <thead><tr>
        <th>Time</th><th>Symbol</th><th>Side</th><th>Shares</th><th>Entry</th><th>Exit</th>
        <th>P&L</th><th>P&L %</th><th>Reason</th><th>Hold</th>
      </tr></thead>
      <tbody id="tradesTable"></tbody>
    </table>
    <div id="tradeHistPager" style="display:flex;gap:8px;align-items:center;margin-top:8px;font-size:12px;"></div>
  </div>

  <!-- Backtest Page -->
  <div class="page" id="page-backtest">
    <h2 style="margin-bottom:10px;">Historical Backtest</h2>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;">
      <div class="config-item">
        <label>Symbol(s):</label>
        <input id="btSymbol" value="" placeholder="TSLA,NVDA... (blank=universe)" style="width:200px;">
      </div>
      <div class="config-item">
        <label>Universe:</label>
        <select id="btUniverse" style="background:var(--bg);color:var(--text);border:1px solid var(--border);padding:4px 6px;border-radius:4px;font-size:12px;">
          <option value="study">Study (167)</option>
          <option value="alpaca" selected>All Tradeable (Alpaca)</option>
        </select>
      </div>
      <div class="config-item">
        <label>Strategy:</label>
        <select id="btStrategy" style="background:var(--bg);color:var(--text);border:1px solid var(--border);padding:4px 6px;border-radius:4px;font-size:12px;">
          <option value="">Engine Default</option>
        </select>
      </div>
      <div class="config-item">
        <label>Start:</label>
        <input type="date" id="btStart" style="width:130px;">
      </div>
      <div class="config-item">
        <label>End:</label>
        <input type="date" id="btEnd" style="width:130px;">
      </div>
      <div class="config-item">
        <label>Gap %:</label>
        <input type="number" id="btGap" value="7" step="1" min="1" max="50" style="width:60px;" title="Min gap % to consider">
      </div>
      <div class="config-item">
        <label>Max Gap %:</label>
        <input type="number" id="btMaxGap" value="50" step="1" min="10" max="100" style="width:60px;" title="Filter out mega-gaps above this % (M&A, biotech catalysts)">
      </div>
      <div class="config-item">
        <label>Vol Max:</label>
        <input type="number" id="btVol" value="3.0" step="0.1" min="0.5" max="20" style="width:60px;" title="Max volume ratio vs 20d avg">
      </div>
      <div class="config-item">
        <label>Stop %:</label>
        <input type="number" id="btStop" value="1.5" step="0.5" min="0.5" max="10" style="width:60px;" title="Stop loss above entry">
      </div>
      <div class="config-item">
        <label>Max Pos:</label>
        <input type="number" id="btMaxPos" value="3" step="1" min="1" max="10" style="width:50px;" title="Max simultaneous positions">
      </div>
      <div class="config-item">
        <label>Slip %:</label>
        <input type="number" id="btSlip" value="0.05" step="0.01" min="0" max="5" style="width:60px;" title="Slippage per side (0.05=limit orders, 0.15=market orders)">
      </div>
      <div class="config-item">
        <label>Cover frac:</label>
        <input type="number" id="btCoverFrac" value="0.33" step="0.01" min="0" max="1" style="width:60px;" title="Fraction to cover at partial target (0=none, 0.33=1/3, 0.5=half)">
      </div>
      <label style="display:flex;align-items:center;gap:4px;cursor:pointer;">
        <input type="checkbox" id="bt1min"> 1-min bars (slow, detailed)
      </label>
    </div>
    <!-- Feature toggles -->
    <div style="margin-top:8px;display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;">
      <!-- Adaptive Stops -->
      <div style="padding:8px 10px;background:rgba(168,85,247,0.06);border:1px solid rgba(168,85,247,0.15);border-radius:6px;">
        <label style="display:flex;align-items:center;gap:4px;cursor:pointer;font-size:12px;font-weight:600;color:var(--purple);">
          <input type="checkbox" id="btAdaptiveStops" onchange="document.getElementById('btAdaptiveStopsOpts').style.display=this.checked?'flex':'none'"> Adaptive Stops
        </label>
        <div id="btAdaptiveStopsOpts" style="display:none;gap:6px;margin-top:6px;flex-wrap:wrap;font-size:11px;">
          <div class="config-item"><label>Gap Frac:</label><input type="number" id="btStopGapFrac" value="0.15" step="0.01" min="0.01" max="1" style="width:55px;" title="stop = gap% x this"></div>
          <div class="config-item"><label>Min %:</label><input type="number" id="btStopMin" value="1.0" step="0.1" min="0.1" max="10" style="width:50px;" title="Minimum stop %"></div>
          <div class="config-item"><label>Max %:</label><input type="number" id="btStopMax" value="5.0" step="0.5" min="0.5" max="20" style="width:50px;" title="Maximum stop %"></div>
        </div>
      </div>
      <!-- Market Regime Filter -->
      <div style="padding:8px 10px;background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.15);border-radius:6px;">
        <label style="display:flex;align-items:center;gap:4px;cursor:pointer;font-size:12px;font-weight:600;color:var(--cyan);">
          <input type="checkbox" id="btRegimeFilter" onchange="document.getElementById('btRegimeOpts').style.display=this.checked?'flex':'none'"> Market Regime Filter
        </label>
        <div id="btRegimeOpts" style="display:none;gap:6px;margin-top:6px;flex-wrap:wrap;font-size:11px;">
          <div class="config-item"><label>SPY Gap Limit %:</label><input type="number" id="btRegimeSpyGap" value="1.0" step="0.1" min="0.1" max="10" style="width:50px;" title="SPY gap > this -> halve positions"></div>
          <div class="config-item"><label>SPY Block %:</label><input type="number" id="btRegimeSpyBlock" value="1.5" step="0.1" min="0.1" max="10" style="width:50px;" title="SPY gap > this -> block all entries"></div>
          <div class="config-item"><label>VIX Thresh:</label><input type="number" id="btRegimeVix" value="25" step="1" min="10" max="80" style="width:50px;" title="VIX > this -> halve positions"></div>
        </div>
      </div>
      <!-- Re-entry After Stop -->
      <div style="padding:8px 10px;background:rgba(234,179,8,0.06);border:1px solid rgba(234,179,8,0.15);border-radius:6px;">
        <label style="display:flex;align-items:center;gap:4px;cursor:pointer;font-size:12px;font-weight:600;color:var(--yellow);">
          <input type="checkbox" id="btReentry" onchange="document.getElementById('btReentryOpts').style.display=this.checked?'flex':'none'"> Re-entry After Stop
        </label>
        <div id="btReentryOpts" style="display:none;gap:6px;margin-top:6px;flex-wrap:wrap;font-size:11px;">
          <div class="config-item"><label>Cooldown min:</label><input type="number" id="btReentryCooldown" value="30" step="5" min="1" max="240" style="width:50px;" title="Minutes to wait after stop-out"></div>
          <div class="config-item"><label>Max re-entries:</label><input type="number" id="btReentryMax" value="1" step="1" min="1" max="5" style="width:40px;" title="Max re-entries per symbol per day"></div>
          <div class="config-item"><label>Re-entry stop %:</label><input type="number" id="btReentryStop" value="1.0" step="0.1" min="0.1" max="10" style="width:50px;" title="Tighter stop on re-entry"></div>
          <div class="config-item"><label>Trigger drop %:</label><input type="number" id="btReentryTrigger" value="0.0" step="0.1" min="0" max="10" style="width:50px;" title="Price must drop this % below original entry"></div>
        </div>
      </div>
      <!-- Gap-Down Fading -->
      <div style="padding:8px 10px;background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.15);border-radius:6px;">
        <label style="display:flex;align-items:center;gap:4px;cursor:pointer;font-size:12px;font-weight:600;color:var(--green);">
          <input type="checkbox" id="btGapDowns" onchange="document.getElementById('btGapDownOpts').style.display=this.checked?'flex':'none'"> Gap-Down Fading (Longs)
        </label>
        <div id="btGapDownOpts" style="display:none;gap:6px;margin-top:6px;flex-wrap:wrap;font-size:11px;">
          <div class="config-item"><label>Gap Down %:</label><input type="number" id="btGapDownThresh" value="5" step="1" min="1" max="50" style="width:50px;" title="Min gap-down % to consider"></div>
          <div class="config-item"><label>Max Gap Down %:</label><input type="number" id="btGapDownMax" value="50" step="5" min="10" max="100" style="width:55px;" title="Max gap-down %"></div>
          <div class="config-item"><label>Vol Max:</label><input type="number" id="btGapDownVol" value="3.0" step="0.1" min="0.5" max="20" style="width:50px;" title="Max vol ratio for gap-downs"></div>
        </div>
      </div>
      <!-- Drawdown Circuit Breaker -->
      <div style="padding:8px 10px;background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.15);border-radius:6px;">
        <label style="display:flex;align-items:center;gap:4px;cursor:pointer;font-size:12px;font-weight:600;color:var(--red);">
          <input type="checkbox" id="btDDCircuitBreaker" onchange="document.getElementById('btDDOpts').style.display=this.checked?'flex':'none'"> Drawdown Circuit Breaker
        </label>
        <div id="btDDOpts" style="display:none;gap:6px;margin-top:6px;flex-wrap:wrap;font-size:11px;">
          <div class="config-item"><label>Tier 1 DD%:</label><input type="number" id="btDDTier1" value="15" step="1" min="5" max="50" style="width:50px;" title="DD% to reduce position size"></div>
          <div class="config-item"><label>Tier 1 Scale:</label><input type="number" id="btDDScale1" value="0.50" step="0.05" min="0.01" max="1" style="width:55px;" title="Position size multiplier at Tier 1"></div>
          <div class="config-item"><label>Tier 2 DD%:</label><input type="number" id="btDDTier2" value="25" step="1" min="10" max="70" style="width:50px;" title="DD% for minimal trading"></div>
          <div class="config-item"><label>Tier 2 Scale:</label><input type="number" id="btDDScale2" value="0.25" step="0.05" min="0.01" max="1" style="width:55px;" title="Position size multiplier at Tier 2"></div>
          <div class="config-item"><label>Tier 2 Max Pos:</label><input type="number" id="btDDMaxPos" value="1" step="1" min="1" max="5" style="width:40px;" title="Max concurrent positions in Tier 2"></div>
          <div class="config-item"><label>Hard Stop DD%:</label><input type="number" id="btDDHardStop" value="0" step="5" min="0" max="100" style="width:50px;" title="Permanent halt (0=disabled)"></div>
        </div>
      </div>
    </div>
    <div style="margin-top:8px;display:flex;gap:6px;">
      <button class="primary" onclick="runBacktest()">Run Backtest</button>
      <button onclick="cancelBacktest()">Cancel</button>
    </div>
    <div id="btProgress" style="display:none;">
      <div style="color:var(--muted);font-size:12px;" id="btProgressMsg">Running...</div>
      <div class="progress-bar"><div class="progress-fill" id="btProgressBar" style="width:0%"></div></div>
    </div>
    <div id="btLog" style="max-height:300px;overflow-y:auto;font-size:11px;margin-top:8px;
         background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:8px;display:none;"></div>
    <div id="btResults" style="margin-top:12px;"></div>

    <!-- Backtrader Visual Charts Section -->
    <div style="margin-top:20px;padding-top:16px;border-top:1px solid var(--border);">
      <h3 style="margin-bottom:10px;font-size:15px;">Backtrader Charts <span style="font-size:11px;color:var(--muted);font-weight:normal;">(1-20 symbols, visual analysis)</span></h3>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:8px;">
        <div>
          <label style="font-size:11px;color:var(--muted);">Symbols (comma-separated)</label>
          <input id="btChartSymbols" value="AAPL, TSLA, NVDA" style="width:100%;padding:6px;background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--text);font-size:12px;">
        </div>
        <div>
          <label style="font-size:11px;color:var(--muted);">Start Date</label>
          <input type="date" id="btChartStart" style="width:100%;padding:6px;background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--text);font-size:12px;">
        </div>
        <div>
          <label style="font-size:11px;color:var(--muted);">End Date</label>
          <input type="date" id="btChartEnd" style="width:100%;padding:6px;background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--text);font-size:12px;">
        </div>
      </div>
      <div style="display:flex;gap:8px;align-items:center;">
        <button class="primary" onclick="runBtBacktest()" id="btChartRunBtn">Run Backtrader</button>
        <span id="btChartStatus" style="font-size:12px;color:var(--muted);"></span>
      </div>

      <!-- Metrics cards -->
      <div id="btChartMetrics" style="display:none;margin-top:12px;">
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;">
          <div class="metric-card" style="text-align:center;padding:10px;">
            <div style="font-size:11px;color:var(--muted);">Total Return</div>
            <div id="btmReturn" style="font-size:18px;font-weight:700;">—</div>
          </div>
          <div class="metric-card" style="text-align:center;padding:10px;">
            <div style="font-size:11px;color:var(--muted);">Sharpe Ratio</div>
            <div id="btmSharpe" style="font-size:18px;font-weight:700;">—</div>
          </div>
          <div class="metric-card" style="text-align:center;padding:10px;">
            <div style="font-size:11px;color:var(--muted);">Max Drawdown</div>
            <div id="btmDD" style="font-size:18px;font-weight:700;">—</div>
          </div>
          <div class="metric-card" style="text-align:center;padding:10px;">
            <div style="font-size:11px;color:var(--muted);">Win Rate</div>
            <div id="btmWinRate" style="font-size:18px;font-weight:700;">—</div>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:8px;">
          <div class="metric-card" style="text-align:center;padding:10px;">
            <div style="font-size:11px;color:var(--muted);">Total Trades</div>
            <div id="btmTrades" style="font-size:18px;font-weight:700;">—</div>
          </div>
          <div class="metric-card" style="text-align:center;padding:10px;">
            <div style="font-size:11px;color:var(--muted);">Profit Factor</div>
            <div id="btmPF" style="font-size:18px;font-weight:700;">—</div>
          </div>
          <div class="metric-card" style="text-align:center;padding:10px;">
            <div style="font-size:11px;color:var(--muted);">Net P&L</div>
            <div id="btmPnL" style="font-size:18px;font-weight:700;">—</div>
          </div>
          <div class="metric-card" style="text-align:center;padding:10px;">
            <div style="font-size:11px;color:var(--muted);">Final Equity</div>
            <div id="btmEquity" style="font-size:18px;font-weight:700;">—</div>
          </div>
        </div>
      </div>

      <!-- Charts display -->
      <div id="btChartImages" style="display:none;margin-top:12px;"></div>

      <!-- Trade table -->
      <div id="btChartTrades" style="display:none;margin-top:12px;"></div>
    </div>
  </div>

  <!-- Config Page -->
  <div class="page" id="page-config">
    <h2 style="margin-bottom:10px;">Strategy Configuration</h2>
    <div style="margin-bottom:16px;padding:12px;background:rgba(168,85,247,0.06);border:1px solid rgba(168,85,247,0.2);border-radius:8px;">
      <div style="font-size:12px;color:var(--purple);font-weight:600;margin-bottom:8px;text-transform:uppercase;">Scan Universe</div>
      <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
        <label style="display:flex;align-items:center;gap:4px;cursor:pointer;">
          <input type="radio" name="scanUniverse" value="alpaca" id="univAlpaca"> All Tradeable (Alpaca ~4000+)
        </label>
        <label style="display:flex;align-items:center;gap:4px;cursor:pointer;">
          <input type="radio" name="scanUniverse" value="study" id="univStudy" checked> Study Universe (167)
        </label>
        <label style="display:flex;align-items:center;gap:4px;cursor:pointer;">
          <input type="radio" name="scanUniverse" value="custom" id="univCustom"> Custom
        </label>
      </div>
      <div id="customSymbolsRow" style="margin-top:8px;display:none;">
        <input id="customSymbolsInput" placeholder="TSLA,AAPL,NVDA,..." style="width:100%;background:var(--bg);color:var(--text);border:1px solid var(--border);padding:6px 8px;border-radius:4px;font-size:12px;">
      </div>
    </div>
    <div class="config-grid" id="configGrid"></div>
    <div style="margin-top:12px;">
      <button class="primary" onclick="saveConfig()">Save Config</button>
    </div>
  </div>

  <!-- Database Page -->
  <div class="page" id="page-database">
    <h2 style="margin-bottom:10px;">Price Database (Local Cache)</h2>
    <div style="margin-bottom:16px;padding:12px;background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.2);border-radius:8px;">
      <div style="color:var(--muted);font-size:11px;margin-bottom:8px;" id="dbStatus">Loading...</div>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        <label>Universe:</label>
        <select id="dbUniverse" style="width:auto;">
          <option value="study">Study (167)</option>
          <option value="alpaca">All Tradeable (Alpaca)</option>
        </select>
        <label>Start:</label>
        <input type="date" id="dbStartDate" style="width:130px;">
        <button class="primary" onclick="buildDB()">Build DB</button>
        <button onclick="updateDB()">Update (Alpaca)</button>
        <button onclick="polygonUpdate(1)" style="background:var(--purple);color:white;">Polygon Update</button>
        <label style="margin-left:8px;font-size:11px;">Days:</label>
        <input type="number" id="polygonDays" value="1" min="1" max="30" style="width:50px;">
      </div>
      <div id="dbProgress" style="display:none;margin-top:8px;">
        <div style="color:var(--muted);font-size:11px;" id="dbProgressMsg">Building...</div>
        <div class="progress-bar"><div class="progress-fill" id="dbProgressBar" style="width:0%;background:var(--cyan);"></div></div>
      </div>
    </div>
  </div>

  <!-- Guide Page -->
  <div class="page" id="page-chat">
    <div style="display:flex;flex-direction:column;height:calc(100vh - 180px);max-width:800px;">
      <div id="chatMessages" style="flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:10px;border:1px solid var(--border);border-radius:8px;background:rgba(0,0,0,0.2);margin-bottom:12px;">
        <div class="chat-msg chat-system">Ask Rudra anything about your positions, strategy, or market conditions. It has full access to live trading state.</div>
      </div>
      <div style="display:flex;gap:8px;">
        <input type="text" id="chatInput" placeholder="Ask about positions, strategy, market..."
          style="flex:1;padding:10px 14px;background:var(--surface);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:14px;font-family:'Inter',sans-serif;outline:none;"
          onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChat();}"/>
        <button onclick="sendChat()" id="chatSendBtn"
          style="padding:10px 20px;background:var(--purple);color:#fff;border:none;border-radius:8px;font-weight:600;cursor:pointer;font-size:14px;white-space:nowrap;">
          Send
        </button>
      </div>
      <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;" id="chatSuggestions">
        <button class="chat-suggestion" onclick="sendPreset(this)">How are my positions doing?</button>
        <button class="chat-suggestion" onclick="sendPreset(this)">Should I close anything?</button>
        <button class="chat-suggestion" onclick="sendPreset(this)">What's the best candidate right now?</button>
        <button class="chat-suggestion" onclick="sendPreset(this)">Summarize today's trading</button>
      </div>
    </div>
  </div>

  <div class="page" id="page-guide">
    <h2 style="margin-bottom:10px;">How the Bot Works</h2>

    <!-- Section 1: Strategy Overview -->
    <div style="margin-bottom:20px;padding:16px;background:rgba(168,85,247,0.06);border:1px solid rgba(168,85,247,0.2);border-radius:8px;">
      <h3 style="color:var(--purple);margin:0 0 10px 0;font-size:14px;">What Is Gap Fading?</h3>
      <p style="color:var(--text);line-height:1.7;margin:0 0 10px 0;font-size:13px;">
        When a stock opens significantly higher than yesterday's close, that jump is called a <strong style="color:var(--purple);">gap up</strong>.
        Most of the time, the price drifts back down toward yesterday's close during the trading day &mdash; this is called <strong style="color:var(--purple);">fading the gap</strong>.
      </p>
      <p style="color:var(--text);line-height:1.7;margin:0 0 10px 0;font-size:13px;">
        This bot finds stocks that gapped up on <em>below-average volume</em> (a sign the move lacks conviction) and shorts them,
        betting the price will fall back. It closes all positions before market close &mdash; no overnight risk.
      </p>
      <div style="display:inline-block;padding:8px 14px;background:rgba(168,85,247,0.12);border-radius:6px;margin-top:4px;">
        <span style="color:var(--purple);font-weight:600;font-size:13px;">Statistical Edge:</span>
        <span style="color:var(--text);font-size:13px;"> 71% of low-volume gap-ups fade &mdash; historical study of 12,000+ events across 450 tickers.</span>
      </div>
    </div>

    <!-- Section 2: Daily Schedule Timeline -->
    <div style="margin-bottom:20px;padding:16px;background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.2);border-radius:8px;">
      <h3 style="color:var(--cyan);margin:0 0 14px 0;font-size:14px;">Daily Schedule</h3>
      <div id="guideTimeline">
        <div class="tl-step" data-phase="premarket" style="display:flex;align-items:flex-start;margin-bottom:14px;position:relative;padding-left:28px;">
          <div style="position:absolute;left:0;top:2px;width:14px;height:14px;border-radius:50%;background:var(--blue);border:2px solid rgba(59,130,246,0.4);"></div>
          <div style="border-left:2px solid var(--border);position:absolute;left:6px;top:18px;height:calc(100% - 4px);"></div>
          <div>
            <span style="color:var(--blue);font-weight:600;font-size:13px;">7:00 AM ET &mdash; Pre-Market Scan</span>
            <p style="color:var(--muted);margin:3px 0 0 0;font-size:12px;line-height:1.5;">Bot wakes up and scans for stocks that gapped up overnight. Filters by gap size, volume ratio, and market cap.</p>
          </div>
        </div>
        <div class="tl-step" data-phase="refresh" style="display:flex;align-items:flex-start;margin-bottom:14px;position:relative;padding-left:28px;">
          <div style="position:absolute;left:0;top:2px;width:14px;height:14px;border-radius:50%;background:var(--blue);border:2px solid rgba(59,130,246,0.4);"></div>
          <div style="border-left:2px solid var(--border);position:absolute;left:6px;top:18px;height:calc(100% - 4px);"></div>
          <div>
            <span style="color:var(--blue);font-weight:600;font-size:13px;">9:25 AM ET &mdash; Final Scan Refresh</span>
            <p style="color:var(--muted);margin:3px 0 0 0;font-size:12px;line-height:1.5;">Re-checks candidates with the latest pre-market data. Drops any that no longer qualify.</p>
          </div>
        </div>
        <div class="tl-step" data-phase="entry" style="display:flex;align-items:flex-start;margin-bottom:14px;position:relative;padding-left:28px;">
          <div style="position:absolute;left:0;top:2px;width:14px;height:14px;border-radius:50%;background:var(--red);border:2px solid rgba(239,68,68,0.4);"></div>
          <div style="border-left:2px solid var(--border);position:absolute;left:6px;top:18px;height:calc(100% - 4px);"></div>
          <div>
            <span style="color:var(--red);font-weight:600;font-size:13px;">9:31 AM ET &mdash; Enter Positions</span>
            <p style="color:var(--muted);margin:3px 0 0 0;font-size:12px;line-height:1.5;">One minute after open, the bot shorts qualified gap-ups. Waits one minute to avoid the chaotic opening auction.</p>
          </div>
        </div>
        <div class="tl-step" data-phase="monitor" style="display:flex;align-items:flex-start;margin-bottom:14px;position:relative;padding-left:28px;">
          <div style="position:absolute;left:0;top:2px;width:14px;height:14px;border-radius:50%;background:var(--yellow);border:2px solid rgba(234,179,8,0.4);"></div>
          <div style="border-left:2px solid var(--border);position:absolute;left:6px;top:18px;height:calc(100% - 4px);"></div>
          <div>
            <span style="color:var(--yellow);font-weight:600;font-size:13px;">9:31 AM &ndash; 3:55 PM ET &mdash; Monitor &amp; Manage</span>
            <p style="color:var(--muted);margin:3px 0 0 0;font-size:12px;line-height:1.5;">Watches positions, enforces stop-losses, and checks circuit breakers. The dashboard updates live during this window.</p>
          </div>
        </div>
        <div class="tl-step" data-phase="close" style="display:flex;align-items:flex-start;margin-bottom:14px;position:relative;padding-left:28px;">
          <div style="position:absolute;left:0;top:2px;width:14px;height:14px;border-radius:50%;background:var(--green);border:2px solid rgba(34,197,94,0.4);"></div>
          <div style="border-left:2px solid var(--border);position:absolute;left:6px;top:18px;height:calc(100% - 4px);"></div>
          <div>
            <span style="color:var(--green);font-weight:600;font-size:13px;">3:55 PM ET &mdash; Close All Positions</span>
            <p style="color:var(--muted);margin:3px 0 0 0;font-size:12px;line-height:1.5;">All positions are closed before market close. No overnight exposure &mdash; every day starts flat.</p>
          </div>
        </div>
        <div class="tl-step" data-phase="sleep" style="display:flex;align-items:flex-start;position:relative;padding-left:28px;">
          <div style="position:absolute;left:0;top:2px;width:14px;height:14px;border-radius:50%;background:var(--purple);border:2px solid rgba(168,85,247,0.4);"></div>
          <div>
            <span style="color:var(--purple);font-weight:600;font-size:13px;">After 4:00 PM ET &mdash; Save &amp; Sleep</span>
            <p style="color:var(--muted);margin:3px 0 0 0;font-size:12px;line-height:1.5;">Logs results, updates statistics, and sleeps until the next trading day.</p>
          </div>
        </div>
      </div>
      <div id="guideNextEvent" style="margin-top:14px;padding:8px 12px;background:rgba(6,182,212,0.10);border-radius:6px;font-size:12px;color:var(--cyan);display:none;"></div>
    </div>

    <!-- Section 3: Safety Features -->
    <div style="margin-bottom:20px;padding:16px;background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.2);border-radius:8px;">
      <h3 style="color:var(--green);margin:0 0 12px 0;font-size:14px;">Safety Features (Circuit Breakers)</h3>
      <p style="color:var(--muted);margin:0 0 12px 0;font-size:12px;">The bot has three automatic safety switches that pause or halt trading to protect your account:</p>
      <div style="display:flex;flex-direction:column;gap:10px;">
        <div style="display:flex;align-items:flex-start;gap:10px;">
          <div style="min-width:32px;height:32px;display:flex;align-items:center;justify-content:center;background:rgba(239,68,68,0.12);border-radius:6px;font-size:16px;">&#128721;</div>
          <div>
            <span style="color:var(--text);font-weight:600;font-size:13px;">Daily Loss Limit</span>
            <p style="color:var(--muted);margin:2px 0 0 0;font-size:12px;line-height:1.5;">If total losses for the day exceed the configured limit, trading stops for the rest of the day. Prevents one bad day from wiping out weeks of gains.</p>
          </div>
        </div>
        <div style="display:flex;align-items:flex-start;gap:10px;">
          <div style="min-width:32px;height:32px;display:flex;align-items:center;justify-content:center;background:rgba(234,179,8,0.12);border-radius:6px;font-size:16px;">&#9208;</div>
          <div>
            <span style="color:var(--text);font-weight:600;font-size:13px;">Consecutive Loss Pause</span>
            <p style="color:var(--muted);margin:2px 0 0 0;font-size:12px;line-height:1.5;">After several losses in a row, the bot pauses to avoid revenge trading. It waits before resuming to let conditions change.</p>
          </div>
        </div>
        <div style="display:flex;align-items:flex-start;gap:10px;">
          <div style="min-width:32px;height:32px;display:flex;align-items:center;justify-content:center;background:rgba(239,68,68,0.12);border-radius:6px;font-size:16px;">&#9940;</div>
          <div>
            <span style="color:var(--text);font-weight:600;font-size:13px;">Max Drawdown Halt</span>
            <p style="color:var(--muted);margin:2px 0 0 0;font-size:12px;line-height:1.5;">If the account drops below a percentage threshold from its peak, all trading halts until you manually review and restart. This is the final safety net.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Section 4: Quick Start -->
    <div style="margin-bottom:20px;padding:16px;background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.2);border-radius:8px;">
      <h3 style="color:var(--blue);margin:0 0 12px 0;font-size:14px;">Quick Start &mdash; 3 Steps</h3>
      <div style="display:flex;flex-direction:column;gap:12px;">
        <div style="display:flex;align-items:flex-start;gap:12px;">
          <div style="min-width:28px;height:28px;display:flex;align-items:center;justify-content:center;background:var(--blue);border-radius:50%;color:#fff;font-weight:700;font-size:13px;">1</div>
          <div>
            <span style="color:var(--text);font-weight:600;font-size:13px;">Configure API Keys</span>
            <p style="color:var(--muted);margin:2px 0 0 0;font-size:12px;line-height:1.5;">Go to the <strong>Config</strong> page and enter your Alpaca API credentials. The bot needs these to place real trades. Use a paper-trading key first to practice.</p>
          </div>
        </div>
        <div style="display:flex;align-items:flex-start;gap:12px;">
          <div style="min-width:28px;height:28px;display:flex;align-items:center;justify-content:center;background:var(--blue);border-radius:50%;color:#fff;font-weight:700;font-size:13px;">2</div>
          <div>
            <span style="color:var(--text);font-weight:600;font-size:13px;">Click Start Trading</span>
            <p style="color:var(--muted);margin:2px 0 0 0;font-size:12px;line-height:1.5;">Hit the green Start button in the sidebar. The bot will wait for the right time and begin scanning automatically.</p>
          </div>
        </div>
        <div style="display:flex;align-items:flex-start;gap:12px;">
          <div style="min-width:28px;height:28px;display:flex;align-items:center;justify-content:center;background:var(--blue);border-radius:50%;color:#fff;font-weight:700;font-size:13px;">3</div>
          <div>
            <span style="color:var(--text);font-weight:600;font-size:13px;">Watch the Dashboard</span>
            <p style="color:var(--muted);margin:2px 0 0 0;font-size:12px;line-height:1.5;">The right panel shows positions and activity. The <strong>Scanner</strong> page shows today's candidates. Everything updates automatically via WebSocket.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Section 5: Status Badge Legend -->
    <div style="margin-bottom:10px;padding:16px;background:rgba(148,163,184,0.06);border:1px solid rgba(148,163,184,0.15);border-radius:8px;">
      <h3 style="color:var(--muted);margin:0 0 12px 0;font-size:14px;">Status Badge Legend</h3>
      <div style="display:grid;grid-template-columns:repeat(auto-fill, minmax(280px, 1fr));gap:10px;">
        <div style="display:flex;align-items:center;gap:10px;padding:8px 10px;background:var(--card);border:1px solid var(--border);border-radius:6px;">
          <span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;background:rgba(148,163,184,0.15);color:var(--muted);">STOPPED</span>
          <span style="color:var(--muted);font-size:12px;">Bot is idle. Click Start to begin.</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;padding:8px 10px;background:var(--card);border:1px solid var(--border);border-radius:6px;">
          <span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;background:rgba(6,182,212,0.15);color:var(--cyan);">SCANNING</span>
          <span style="color:var(--muted);font-size:12px;">Looking for gap-up candidates.</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;padding:8px 10px;background:var(--card);border:1px solid var(--border);border-radius:6px;">
          <span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;background:rgba(34,197,94,0.15);color:var(--green);">TRADING</span>
          <span style="color:var(--muted);font-size:12px;">Actively managing positions.</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;padding:8px 10px;background:var(--card);border:1px solid var(--border);border-radius:6px;">
          <span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;background:rgba(234,179,8,0.15);color:var(--yellow);">PAUSED</span>
          <span style="color:var(--muted);font-size:12px;">Temporarily paused by a circuit breaker.</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;padding:8px 10px;background:var(--card);border:1px solid var(--border);border-radius:6px;">
          <span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;background:rgba(239,68,68,0.15);color:var(--red);">HALTED</span>
          <span style="color:var(--muted);font-size:12px;">Max drawdown hit. Manual review needed.</span>
        </div>
      </div>
    </div>

  </div>

  <!-- ── Position Tracker Page ── -->
  <div class="page" id="page-tracker">
    <div class="tracker-controls">
      <input id="trackerSymbolInput" placeholder="Add symbol (e.g. AAPL)" onkeydown="if(event.key==='Enter')addWatchSymbol()">
      <button class="add-btn" onclick="addWatchSymbol()">+ Add</button>
      <span style="color:var(--muted);font-size:11px;margin-left:8px;" id="trackerWatchCount">0 / 6 symbols</span>
    </div>

    <div class="tracker-positions-card">
      <h3>Broker Positions</h3>
      <div style="overflow-x:auto;">
        <table>
          <thead><tr>
            <th>Symbol</th><th>Qty</th><th>Mkt Value</th><th>Mark</th>
            <th>Avg Price</th><th>Last</th><th>1D P&amp;L</th>
          </tr></thead>
          <tbody id="trackerPositionsBody"></tbody>
        </table>
      </div>
      <div id="trackerNoPositions" style="color:var(--muted);padding:12px;text-align:center;font-size:12px;">
        No broker positions
      </div>
    </div>

    <div id="trackerChartGrid"></div>
  </div>

</main>

<!-- ── Right Panel ── -->
<aside class="right-panel" id="rightPanel">
  <div class="right-panel-resize" id="rightPanelResize"></div>
  <div class="right-panel-tabs">
    <div class="right-panel-tab active" data-rtab="positions" onclick="showRightTab('positions')">Positions</div>
    <div class="right-panel-tab" data-rtab="feed" onclick="showRightTab('feed')">Feed</div>
    <div class="right-panel-tab" data-rtab="journal" onclick="showRightTab('journal');loadJournal()">Journal</div>
  </div>
  <div class="right-tab-content active" id="rightPositions">
    <div style="overflow-x:auto;">
      <table>
        <thead><tr>
          <th>Symbol</th><th>Shares</th><th>Entry</th><th>Current</th><th>Stop</th><th>Target</th><th>P&L</th><th></th>
        </tr></thead>
        <tbody id="positionsTable"></tbody>
      </table>
    </div>
    <div id="noPositions" style="color:var(--muted);padding:20px;text-align:center;font-size:12px;">No active positions</div>
  </div>
  <div class="right-tab-content" id="rightFeed">
    <div class="feed" id="feedContainer"></div>
  </div>
  <div class="right-tab-content" id="rightJournal">
    <div style="display:flex;gap:4px;margin-bottom:8px;flex-wrap:wrap;">
      <button onclick="loadJournal()" style="font-size:10px;padding:3px 8px;cursor:pointer;background:var(--surface);border:1px solid var(--border);color:var(--text);border-radius:4px;">Refresh</button>
      <select id="journalFilter" onchange="loadJournal()" style="font-size:10px;padding:3px 6px;background:var(--surface);border:1px solid var(--border);color:var(--text);border-radius:4px;">
        <option value="">All types</option>
        <option value="observation">Observations</option>
        <option value="reasoning">Reasoning</option>
        <option value="action">Actions</option>
        <option value="no_action">No-Actions</option>
        <option value="reflection">Reflections</option>
      </select>
      <span id="journalBudget" style="font-size:10px;color:var(--muted);margin-left:auto;"></span>
    </div>
    <div id="journalEntries" style="flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:4px;"></div>
  </div>
</aside>

<!-- ── Status Bar ── -->
<footer class="status-bar">
  <div class="left">
    <span>API: &mdash; ms</span>
    <span id="statusLastTrade">Last: &mdash;</span>
  </div>
  <div class="right">
    <span id="statusConn" style="color:var(--negative);">&#9675; Offline</span>
    <span class="version-link" onclick="document.getElementById('releaseNotesModal').classList.add('open')">Gap Fade __APP_VERSION__</span>
  </div>
</footer>

<!-- ── Release Notes Modal ── -->
<div id="releaseNotesModal" class="rn-overlay" onclick="if(event.target===this)this.classList.remove('open')">
  <div class="rn-modal">
    <div class="rn-header">
      <h3>Release Notes</h3>
      <button class="rn-close" onclick="document.getElementById('releaseNotesModal').classList.remove('open')">&times;</button>
    </div>
    <div class="rn-body">
      <h4>v10.0 <span class="rn-tag">current</span></h4>
      <ul>
        <li>Drawdown circuit breakers &mdash; graduated Tier 1/Tier 2/Hard Stop response</li>
        <li>Position size reduction during drawdowns (backtest + live)</li>
        <li>Circuit breaker stats in backtest results (days in tier, trades skipped)</li>
        <li>New UI toggle and config fields for DD thresholds and scales</li>
        <li>DD metric aligned: circuit breaker and dashboard Max DD use same formula</li>
      </ul>
      <h4>v9.0</h4>
      <ul>
        <li>Vectorbt-powered backtesting module with bulk SQLite data loading</li>
        <li>Vectorized gap scanning with pandas &mdash; 10x faster than row-by-row</li>
        <li>SimulationState engine with Kelly sizing, adaptive stops, regime filter</li>
        <li>Strategy plugin support in backtester (classic, VWAP, confluence, Minervini)</li>
        <li>MetricsAdapter for dashboard-compatible result format</li>
        <li>Full test suite (43 tests)</li>
      </ul>
      <h4>v8.0</h4>
      <ul>
        <li>Multi-provider LLM support (Ollama, OpenAI-compatible: GPT, Groq, Together)</li>
        <li>Remove LLM budget rate-limiting for faster autonomous decisions</li>
        <li>Provider-specific API key and endpoint configuration</li>
      </ul>
      <h4>v7.0</h4>
      <ul>
        <li>Upgrade Rudra LLM to gpt-oss:20b with full persona</li>
        <li>Gap-down fading strategy (long entries on gap-downs)</li>
        <li>Direction-aware position sizing, stops, and targets</li>
      </ul>
      <h4>v6.0</h4>
      <ul>
        <li>Mobile OAuth flow with dynamic redirect URL detection</li>
        <li>Responsive auth UI for mobile devices</li>
      </ul>
      <h4>v5.0</h4>
      <ul>
        <li>Fix LLM supervisor not initializing from saved config</li>
        <li>Status badge now updates for all LLM actions (wait, monitor)</li>
        <li>Trading loop resilience &mdash; single errors no longer kill the loop</li>
        <li>Double-start guard prevents duplicate trading loops</li>
        <li>Relaxed standdown rules &mdash; consecutive losses alone don't halt trading</li>
        <li>Version number displayed in UI from git tags</li>
      </ul>
      <h4>v4.0</h4>
      <ul>
        <li>Real-time position tracker with candlestick charts</li>
        <li>Watchlist with live price streaming</li>
        <li>Per-position controls and autonomous LLM review loop</li>
        <li>LLM learning from trades: metadata tracking, dynamic lessons</li>
        <li>Full zero-human-intervention automation</li>
      </ul>
      <h4>v3.0</h4>
      <ul>
        <li>LLM chat interface for interactive trading conversations</li>
        <li>Ask Rudra questions, trigger actions via natural language</li>
      </ul>
      <h4>v2.0</h4>
      <ul>
        <li>LLM supervisor (Rudra) for autonomous trading decisions</li>
        <li>Scan, enter, monitor, standdown &mdash; all LLM-driven</li>
        <li>Circuit breaker with fallback to rules-based schedule</li>
      </ul>
      <h4>v1.0</h4>
      <ul>
        <li>Gap fade strategy dashboard with embedded UI</li>
        <li>OAuth authentication (Google/GitHub/Discord)</li>
        <li>Alpaca paper trading integration</li>
        <li>SQLite price database with historical backtesting</li>
        <li>Catalyst detection (earnings, FDA, M&amp;A filtering)</li>
      </ul>
    </div>
  </div>
</div>

</div><!-- #app -->

<script>
// ── Auth (injected by server) ────────────────────────────────────
/*__API_KEY_PLACEHOLDER__*/

// ── OAuth Auth Gate ──────────────────────────────────────────────
let _authUser = null;
let _authEnabled = false;

async function checkAuth() {
  try {
    const r = await fetch('/api/auth/me', { credentials: 'include' });
    const data = await r.json();
    _authEnabled = data.auth_enabled || false;
    if (!_authEnabled) {
      // Auth not configured — open access
      document.getElementById('app').style.display = '';
      return;
    }
    if (window.location.hash === '#/access-denied') {
      document.getElementById('accessDeniedOverlay').style.display = 'flex';
      return;
    }
    if (data.user) {
      _authUser = data.user;
      document.getElementById('app').style.display = '';
      showUserBadge(data.user);
    } else {
      document.getElementById('authOverlay').style.display = 'flex';
    }
  } catch(e) {
    // Auth endpoint unreachable — open access
    document.getElementById('app').style.display = '';
  }
}

function showUserBadge(user) {
  const badge = document.getElementById('userBadge');
  badge.style.display = 'flex';
  const nameEl = document.getElementById('userName');
  nameEl.textContent = user.name || user.email || '';
  if (user.picture) {
    const img = document.getElementById('userAvatar');
    img.src = user.picture;
    img.style.display = 'block';
  } else {
    const init = document.getElementById('userInitial');
    init.textContent = (user.name || user.email || '?')[0].toUpperCase();
    init.style.display = 'block';
  }
}

function doLogout() {
  // Navigate to server logout — server clears httpOnly cookie and redirects back
  window.location.href = '/api/auth/logout';
}

// ── Toast notifications ──────────────────────────────────────────
let _toastCount = 0;
function showToast(msg, type='error') {
  const t = document.createElement('div');
  t.className = 'toast toast-' + type;
  t.textContent = msg;
  t.style.top = (16 + _toastCount * 56) + 'px';
  _toastCount++;
  t.onclick = () => { t.remove(); _toastCount = Math.max(0, _toastCount - 1); };
  document.body.appendChild(t);
  setTimeout(() => { if (t.parentNode) { t.remove(); _toastCount = Math.max(0, _toastCount - 1); } }, 5000);
}

// ── State ────────────────────────────────────────────────────────
let ws = null;
let state = {};
let activePage = 'dashboard';

// ── Page Navigation ──────────────────────────────────────────────
const PAGE_TITLES = {
  dashboard: 'Dashboard',
  candidates: 'Scanner',
  trades: 'Trade Log',
  backtest: 'Backtest',
  config: 'Config',
  database: 'Database',
  tracker: 'Position Tracker',
  chat: 'Rudra Chat',
  guide: 'Guide'
};

function showPage(name) {
  document.querySelectorAll('.sidebar-item').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.page').forEach(el => el.classList.remove('active'));
  const page = document.getElementById('page-' + name);
  if (page) page.classList.add('active');
  const navItem = document.querySelector(`.sidebar-item[data-page="${name}"]`);
  if (navItem) navItem.classList.add('active');
  const title = document.getElementById('pageTitle');
  if (title) title.textContent = PAGE_TITLES[name] || name;
  activePage = name;
}

// Backward compat: showTab maps to showPage
function showTab(name) {
  const map = { live: 'dashboard', candidates: 'candidates', trades: 'trades', backtest: 'backtest', config: 'config', chat: 'chat', guide: 'guide' };
  showPage(map[name] || name);
}

function showRightTab(name) {
  document.querySelectorAll('.right-panel-tab').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.right-tab-content').forEach(el => el.classList.remove('active'));
  const tab = document.querySelector(`.right-panel-tab[data-rtab="${name}"]`);
  if (tab) tab.classList.add('active');
  const contentMap = {positions: 'rightPositions', feed: 'rightFeed', journal: 'rightJournal'};
  const content = document.getElementById(contentMap[name] || 'rightFeed');
  if (content) content.classList.add('active');
}

// ── Journal Tab ───────────────────────────────────────────────────
async function loadJournal() {
  try {
    const filter = document.getElementById('journalFilter')?.value || '';
    const qs = filter ? `?entry_type=${filter}&n=50` : '?n=50';
    const [jRes, eRes] = await Promise.all([
      fetch('/api/journal' + qs),
      fetch('/api/events'),
    ]);
    const journal = await jRes.json();
    const events = await eRes.json();
    const container = document.getElementById('journalEntries');
    if (!container) return;
    const budget = events.budget || {};
    const budgetEl = document.getElementById('journalBudget');
    if (budgetEl) {
      budgetEl.textContent = `LLM: ${budget.calls_last_5min||0}/${budget.max_per_5min||8} (5m) | Events: ${events.pending_count||0} pending`;
    }
    if (!journal.entries || journal.entries.length === 0) {
      container.innerHTML = '<div style="color:var(--muted);padding:20px;text-align:center;">No journal entries yet</div>';
      return;
    }
    const typeColors = {observation:'#64748b',reasoning:'#a855f7',action:'#22c55e',no_action:'#eab308',reflection:'#00D4FF'};
    container.innerHTML = journal.entries.map(e => {
      const time = (e.timestamp||'').split('T')[1] || e.timestamp || '';
      const color = typeColors[e.entry_type] || '#64748b';
      const sym = e.symbol ? `<span style="color:var(--text);font-weight:600;">[${e.symbol}]</span> ` : '';
      const llm = e.llm_call ? ' <span style="color:var(--purple);font-size:9px;">LLM</span>' : '';
      return `<div style="padding:4px 6px;border-left:2px solid ${color};background:rgba(0,0,0,0.15);border-radius:0 4px 4px 0;font-size:10px;line-height:1.4;">
        <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
          <span style="color:${color};font-weight:600;text-transform:uppercase;font-size:9px;">${e.entry_type}${llm}</span>
          <span style="color:var(--muted);font-size:9px;">${time}</span>
        </div>
        <div style="color:var(--text);">${sym}${(e.content||'').substring(0,200)}</div>
      </div>`;
    }).reverse().join('');
  } catch(err) { console.error('Journal load error:', err); }
}

// ── Right Panel Resize ───────────────────────────────────────────
(function() {
  const handle = document.getElementById('rightPanelResize');
  const app = document.getElementById('app');
  if (!handle || !app) return;
  let dragging = false, startX = 0, startW = 0;
  handle.addEventListener('mousedown', e => {
    e.preventDefault();
    dragging = true;
    startX = e.clientX;
    startW = document.getElementById('rightPanel').offsetWidth;
    handle.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    const w = Math.max(200, Math.min(window.innerWidth - 400, startW + (startX - e.clientX)));
    app.style.gridTemplateColumns = `180px 1fr ${w}px`;
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  });
})();

// ── WebSocket ────────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen = () => {
    const dot = document.getElementById('connectionDot');
    const lbl = document.getElementById('connLabel');
    const statusConn = document.getElementById('statusConn');
    if (dot) { dot.classList.add('connected'); }
    if (lbl) lbl.textContent = 'Live';
    if (statusConn) { statusConn.textContent = '\u25cf Connected'; statusConn.style.color = 'var(--positive)'; }
  };
  ws.onclose = () => {
    const dot = document.getElementById('connectionDot');
    const lbl = document.getElementById('connLabel');
    const statusConn = document.getElementById('statusConn');
    if (dot) dot.classList.remove('connected');
    if (lbl) lbl.textContent = 'Offline';
    if (statusConn) { statusConn.textContent = '\u25cb Disconnected'; statusConn.style.color = 'var(--negative)'; }
    setTimeout(connectWS, 2000);
  };
  ws.onmessage = (e) => {
    if (e.data === 'pong') return;
    try {
      const msg = JSON.parse(e.data);
      handleMessage(msg);
    } catch(err) { console.error('WS parse error:', err, e.data?.substring(0, 200)); }
  };
}
// Keep alive (single interval, not inside connectWS to avoid stacking)
setInterval(() => { if (ws && ws.readyState === 1) ws.send('ping'); }, 15000);

function handleMessage(msg) {
  if (msg.type === 'live_status') {
    updateStatus(msg.status);
  } else if (msg.type === 'positions_update') {
    if (msg.positions) renderPositions(msg.positions);
    if (msg.stats) updateCircuitBreakers(msg.stats);
    if (msg.equity) updateEquity(msg.equity);
  } else if (msg.type === 'scan_results') {
    renderCandidates(msg.candidates || []);
    if (msg.scan_time) document.getElementById('scanTime').textContent = `Last scan: ${msg.scan_time}`;
    if (msg.universe_size) document.getElementById('universeInfo').textContent = `(${msg.universe_size.toLocaleString()} symbols scanned)`;
  } else if (msg.type === 'trade') {
    fetchState();
  } else if (msg.type === 'backtest_progress') {
    document.getElementById('btProgress').style.display = 'block';
    document.getElementById('btProgressMsg').textContent = msg.message || 'Running...';
    document.getElementById('btProgressBar').style.width = msg.progress + '%';
  } else if (msg.type === 'backtest_complete') {
    if (window._btResultPoll) { clearTimeout(window._btResultPoll); window._btResultPoll = null; }
    document.getElementById('btProgress').style.display = 'none';
    if (msg.error) {
      showToast('Backtest failed: ' + msg.error, 'error');
      document.getElementById('btProgressMsg').textContent = 'Error: ' + msg.error;
      document.getElementById('btProgress').style.display = 'block';
    } else {
      try { renderBacktestResults(msg.result); } catch(e) {
        console.error('renderBacktestResults failed:', e);
        showToast('Error rendering results: ' + e.message, 'error');
      }
    }
  } else if (msg.type === 'bt_log') {
    appendBtLog(msg.entry);
  } else if (msg.type === 'db_build_progress') {
    document.getElementById('dbProgress').style.display = 'block';
    document.getElementById('dbProgressMsg').textContent = msg.message || 'Working...';
    document.getElementById('dbProgressBar').style.width = msg.progress + '%';
    if (msg.progress >= 100) {
      setTimeout(() => { document.getElementById('dbProgress').style.display = 'none'; fetchDbStats(); }, 2000);
    }
  } else if (msg.type === 'tracker_tick') {
    handleTrackerTick(msg.symbol, msg.price, msg.timestamp);
  } else if (msg.type === 'conversation') {
    // Real-time conversation message from bot<->Rudra dialogue
    const feed = document.getElementById('feedContainer');
    if (feed) {
      const div = document.createElement('div');
      div.className = 'feed-item conversation';
      div.style.borderLeft = '2px solid #34d399';
      div.style.paddingLeft = '8px';
      const now = new Date().toLocaleTimeString('en-US', {hour12:false, hour:'2-digit', minute:'2-digit', second:'2-digit'});
      div.innerHTML = `<span class="time">${now}</span><span style="color:#c084fc;font-weight:600;">RUDRA:</span> ${(msg.text||'').substring(0,300)}`;
      feed.insertBefore(div, feed.firstChild);
    }
  }
}

// ── API calls ────────────────────────────────────────────────────
async function api(path, method='GET', body=null) {
  const headers = { 'Content-Type': 'application/json' };
  if (typeof __API_KEY__ !== 'undefined' && __API_KEY__) {
    headers['X-API-Key'] = __API_KEY__;
  }
  const opts = { method, headers };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch('/api/' + path, opts);
  return r.json();
}

async function fetchState() {
  state = await api('state');
  renderState(state);
}

async function runScan() {
  document.getElementById('universeInfo').textContent = 'Scanning...';
  const r = await api('scan', 'POST');
  renderCandidates(r.candidates || []);
  if (r.scan_time) document.getElementById('scanTime').textContent = `Last scan: ${r.scan_time}`;
  if (r.universe_size) document.getElementById('universeInfo').textContent = `(${r.universe_size.toLocaleString()} symbols scanned)`;
}

async function startTrading() { await api('start', 'POST'); fetchState(); }
async function stopTrading() { await api('stop', 'POST'); fetchState(); }
async function pauseTrading() { await api('pause', 'POST'); fetchState(); }
async function resumeTrading() { await api('resume', 'POST'); fetchState(); }
async function resetTrader() {
  if (!confirm('Reset equity and all trade history?')) return;
  await api('reset', 'POST');
  fetchState();
}

// ── Strategy Plugin System ──
async function loadStrategies() {
  try {
    const r = await api('strategies');
    const sel = document.getElementById('strategySelect');
    if (!sel || !r.strategies) return;
    sel.innerHTML = '';
    for (const s of r.strategies) {
      const opt = document.createElement('option');
      opt.value = s.id;
      opt.textContent = s.name;
      if (s.id === r.active) opt.selected = true;
      sel.appendChild(opt);
    }
    const badge = document.getElementById('strategyBadge');
    if (badge) badge.textContent = r.active_name || 'Classic';
    // Also populate backtest strategy dropdown
    const btSel = document.getElementById('btStrategy');
    if (btSel) {
      btSel.innerHTML = '<option value="">Engine Default</option>';
      for (const s of r.strategies) {
        const opt = document.createElement('option');
        opt.value = s.id;
        opt.textContent = s.name;
        btSel.appendChild(opt);
      }
    }
  } catch(e) { console.warn('Strategy load failed:', e); }
}

async function switchStrategy(strategyId) {
  if (!strategyId) return;
  try {
    const r = await api('strategy/switch', 'POST', {strategy_id: strategyId});
    if (r.error) { showToast(r.error); return; }
    const badge = document.getElementById('strategyBadge');
    if (badge) badge.textContent = r.name || strategyId;
    showToast('Strategy: ' + (r.name || strategyId));
    fetchState();
  } catch(e) { showToast('Strategy switch failed: ' + e.message); }
}

async function runBacktest() {
  // ── Validate dates ──
  const btStartVal = document.getElementById('btStart').value;
  const btEndVal = document.getElementById('btEnd').value;
  const dateRe = /^\\d{4}-\\d{2}-\\d{2}$/;
  if (btStartVal && !dateRe.test(btStartVal)) { showToast('Start date must be YYYY-MM-DD format'); return; }
  if (btEndVal && !dateRe.test(btEndVal)) { showToast('End date must be YYYY-MM-DD format'); return; }
  if (btStartVal && isNaN(new Date(btStartVal).getTime())) { showToast('Start date is not a valid date'); return; }
  if (btEndVal && isNaN(new Date(btEndVal).getTime())) { showToast('End date is not a valid date'); return; }
  if (btStartVal && btEndVal && btStartVal > btEndVal) { showToast('Start date must be before end date'); return; }

  // ── Validate numerics with ranges ──
  const numFields = [
    ['btGap',       'Gap %',          1, 50],
    ['btMaxGap',    'Max Gap %',     10, 100],
    ['btVol',       'Vol Max',      0.5, 20],
    ['btStop',      'Stop %',       0.5, 10],
    ['btMaxPos',    'Max Positions',   1, 10],
    ['btSlip',      'Slippage %',     0, 5],
    ['btCoverFrac', 'Cover Fraction',  0, 1],
  ];
  for (const [id, label, lo, hi] of numFields) {
    const v = parseFloat(document.getElementById(id).value);
    if (isNaN(v)) { showToast(label + ' is not a valid number'); return; }
    if (v < lo || v > hi) { showToast(label + ': ' + v + ' out of range [' + lo + ', ' + hi + ']'); return; }
  }

  const syms = document.getElementById('btSymbol').value.trim();
  const body = {
    start_date: btStartVal,
    end_date: btEndVal,
    gap_threshold: parseFloat(document.getElementById('btGap').value) / 100,
    max_gap_pct: parseFloat(document.getElementById('btMaxGap').value) / 100,
    vol_ratio_max: parseFloat(document.getElementById('btVol').value),
    stop_pct: parseFloat(document.getElementById('btStop').value) / 100,
    max_positions: parseInt(document.getElementById('btMaxPos').value),
    slippage_pct: parseFloat(document.getElementById('btSlip').value) / 100,
    partial_cover_frac: parseFloat(document.getElementById('btCoverFrac').value),
  };
  if (syms) {
    body.symbols = syms;
  } else {
    body.universe = document.getElementById('btUniverse').value;
  }
  if (document.getElementById('bt1min').checked) body.use_1min = true;
  const btStratEl = document.getElementById('btStrategy');
  if (btStratEl && btStratEl.value) body.strategy_id = btStratEl.value;

  // Feature toggles
  if (document.getElementById('btAdaptiveStops').checked) {
    body.adaptive_stops = true;
    const gapFrac = parseFloat(document.getElementById('btStopGapFrac').value);
    const stopMin = parseFloat(document.getElementById('btStopMin').value);
    const stopMax = parseFloat(document.getElementById('btStopMax').value);
    if (isNaN(gapFrac) || isNaN(stopMin) || isNaN(stopMax)) { showToast('Adaptive stops: invalid number'); return; }
    if (stopMin >= stopMax) { showToast('Adaptive stops: Min % must be less than Max %'); return; }
    body.stop_gap_fraction = gapFrac;
    body.stop_min_pct = stopMin / 100;
    body.stop_max_pct = stopMax / 100;
  } else {
    body.adaptive_stops = false;
  }
  if (document.getElementById('btRegimeFilter').checked) {
    body.regime_filter = true;
    const spyGap = parseFloat(document.getElementById('btRegimeSpyGap').value);
    const spyBlock = parseFloat(document.getElementById('btRegimeSpyBlock').value);
    const vixThresh = parseFloat(document.getElementById('btRegimeVix').value);
    if (isNaN(spyGap) || isNaN(spyBlock) || isNaN(vixThresh)) { showToast('Regime filter: invalid number'); return; }
    if (spyGap >= spyBlock) { showToast('Regime filter: SPY Gap Limit must be less than SPY Block %'); return; }
    body.regime_spy_gap_limit = spyGap / 100;
    body.regime_spy_block_pct = spyBlock / 100;
    body.regime_vix_threshold = vixThresh;
  } else {
    body.regime_filter = false;
  }
  if (document.getElementById('btReentry').checked) {
    body.reentry_enabled = true;
    const cooldown = parseInt(document.getElementById('btReentryCooldown').value);
    const maxRe = parseInt(document.getElementById('btReentryMax').value);
    const reStop = parseFloat(document.getElementById('btReentryStop').value);
    const reTrig = parseFloat(document.getElementById('btReentryTrigger').value);
    if (isNaN(cooldown) || isNaN(maxRe) || isNaN(reStop) || isNaN(reTrig)) { showToast('Re-entry: invalid number'); return; }
    body.reentry_cooldown_minutes = cooldown;
    body.reentry_max_per_symbol = maxRe;
    body.reentry_stop_pct = reStop / 100;
    body.reentry_trigger_pct = reTrig / 100;
  } else {
    body.reentry_enabled = false;
  }
  if (document.getElementById('btGapDowns').checked) {
    body.trade_gap_downs = true;
    const gdThresh = parseFloat(document.getElementById('btGapDownThresh').value);
    const gdMax = parseFloat(document.getElementById('btGapDownMax').value);
    const gdVol = parseFloat(document.getElementById('btGapDownVol').value);
    if (isNaN(gdThresh) || isNaN(gdMax) || isNaN(gdVol)) { showToast('Gap-down: invalid number'); return; }
    body.gap_down_threshold = gdThresh / 100;
    body.gap_down_max_pct = gdMax / 100;
    body.gap_down_vol_ratio_max = gdVol;
  } else {
    body.trade_gap_downs = false;
  }
  if (document.getElementById('btDDCircuitBreaker').checked) {
    body.dd_circuit_breaker = true;
    const t1 = parseFloat(document.getElementById('btDDTier1').value);
    const s1 = parseFloat(document.getElementById('btDDScale1').value);
    const t2 = parseFloat(document.getElementById('btDDTier2').value);
    const s2 = parseFloat(document.getElementById('btDDScale2').value);
    const mp = parseInt(document.getElementById('btDDMaxPos').value);
    const hs = parseFloat(document.getElementById('btDDHardStop').value);
    if (isNaN(t1) || isNaN(s1) || isNaN(t2) || isNaN(s2) || isNaN(mp) || isNaN(hs)) { showToast('DD Circuit Breaker: invalid number'); return; }
    if (t1 >= t2) { showToast('DD Circuit Breaker: Tier 1 DD% must be less than Tier 2 DD%'); return; }
    if (hs > 0 && hs <= t2) { showToast('DD Circuit Breaker: Hard Stop must be greater than Tier 2 DD%'); return; }
    body.dd_tier1_threshold = t1 / 100;
    body.dd_tier1_scale = s1;
    body.dd_tier2_threshold = t2 / 100;
    body.dd_tier2_scale = s2;
    body.dd_tier2_max_positions = mp;
    body.dd_hard_stop = hs / 100;
  }

  document.getElementById('btProgress').style.display = 'block';
  document.getElementById('btProgressMsg').textContent = 'Starting backtest...';
  document.getElementById('btProgressBar').style.width = '0%';
  document.getElementById('btResults').innerHTML = '';
  document.getElementById('btLog').innerHTML = '';
  document.getElementById('btLog').style.display = 'block';

  try {
    const r = await api('backtest', 'POST', body);
    if (r.error) {
      document.getElementById('btProgressMsg').textContent = 'Error: ' + r.error;
      return;
    }
    // Results arrive via WebSocket 'backtest_complete', but poll as fallback
    if (window._btResultPoll) clearTimeout(window._btResultPoll);
    window._btResultPollCount = 0;
    window._btResultPoll = setTimeout(async function pollResult() {
      try {
        const sr = await api('backtest/status', 'GET');
        if (sr && sr.status === 'done') {
          const rr = await api('backtest/results', 'GET');
          if (rr && rr.total_trades !== undefined) {
            window._btResultPoll = null;
            document.getElementById('btProgress').style.display = 'none';
            try { renderBacktestResults(rr); } catch(e) {
              console.error('renderBacktestResults failed:', e);
              showToast('Error rendering: ' + e.message, 'error');
            }
            return;
          }
        } else if (sr && sr.status === 'error') {
          window._btResultPoll = null;
          const rr = await api('backtest/results', 'GET');
          showToast('Backtest error: ' + (rr.error || 'unknown'), 'error');
          document.getElementById('btProgressMsg').textContent = 'Error: ' + (rr.error || 'unknown');
          return;
        }
      } catch(e) {}
      window._btResultPollCount = (window._btResultPollCount || 0) + 1;
      if (window._btResultPollCount < 120) {
        window._btResultPoll = setTimeout(pollResult, 2000);
      } else {
        window._btResultPoll = null;
        document.getElementById('btProgressMsg').textContent = 'Timed out waiting for results';
      }
    }, 3000);
  } catch(e) {
    document.getElementById('btProgressMsg').textContent = 'Error: ' + e.message;
  }
}

async function cancelBacktest() { await api('backtest/cancel', 'POST'); }

// ── Backtrader Charts ──
let btChartPoll = null;
function initBtChartDates() {
  const end = new Date();
  const start = new Date();
  start.setFullYear(end.getFullYear() - 1);
  const fmt = d => d.toISOString().slice(0,10);
  const elS = document.getElementById('btChartStart');
  const elE = document.getElementById('btChartEnd');
  if (elS && !elS.value) elS.value = fmt(start);
  if (elE && !elE.value) elE.value = fmt(end);
}
setTimeout(initBtChartDates, 500);

async function runBtBacktest() {
  const symbols = document.getElementById('btChartSymbols').value;
  const startDate = document.getElementById('btChartStart').value;
  const endDate = document.getElementById('btChartEnd').value;
  if (!symbols.trim()) { showToast('Enter at least one symbol'); return; }
  if (!startDate || !endDate) { showToast('Set start and end dates'); return; }

  document.getElementById('btChartRunBtn').disabled = true;
  document.getElementById('btChartStatus').textContent = 'Starting...';
  document.getElementById('btChartMetrics').style.display = 'none';
  document.getElementById('btChartImages').style.display = 'none';
  document.getElementById('btChartTrades').style.display = 'none';

  try {
    await fetch('/api/backtest/bt', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({symbols, start_date: startDate, end_date: endDate, config: {}})
    });
    if (btChartPoll) clearInterval(btChartPoll);
    btChartPoll = setInterval(pollBtStatus, 2000);
    document.getElementById('btChartStatus').textContent = 'Running...';
  } catch(e) {
    showToast('Failed to start backtrader: ' + e.message);
    document.getElementById('btChartRunBtn').disabled = false;
    document.getElementById('btChartStatus').textContent = '';
  }
}

async function pollBtStatus() {
  try {
    const r = await fetch('/api/backtest/bt/status');
    const d = await r.json();
    if (d.status === 'running') {
      document.getElementById('btChartStatus').textContent = 'Running backtrader...';
      return;
    }
    clearInterval(btChartPoll);
    btChartPoll = null;
    document.getElementById('btChartRunBtn').disabled = false;

    if (d.status === 'ok') {
      document.getElementById('btChartStatus').textContent = 'Done!';
      showBtResults(d);
    } else if (d.status === 'error') {
      document.getElementById('btChartStatus').textContent = 'Error: ' + (d.error || 'unknown');
      showToast('Backtrader error: ' + (d.error || 'unknown'));
    } else {
      document.getElementById('btChartStatus').textContent = '';
    }
  } catch(e) {
    clearInterval(btChartPoll);
    btChartPoll = null;
    document.getElementById('btChartRunBtn').disabled = false;
    document.getElementById('btChartStatus').textContent = 'Poll error';
  }
}

function showBtResults(d) {
  const m = d.metrics || {};
  const retColor = m.total_return_pct >= 0 ? 'var(--green)' : 'var(--red)';
  const pnlColor = m.net_pnl >= 0 ? 'var(--green)' : 'var(--red)';
  document.getElementById('btmReturn').innerHTML = `<span style="color:${retColor}">${m.total_return_pct||0}%</span>`;
  document.getElementById('btmSharpe').textContent = m.sharpe_ratio || '—';
  document.getElementById('btmDD').innerHTML = `<span style="color:var(--red)">-${m.max_drawdown_pct||0}%</span>`;
  document.getElementById('btmWinRate').textContent = (m.win_rate||0) + '%';
  document.getElementById('btmTrades').textContent = m.total_trades || 0;
  document.getElementById('btmPF').textContent = m.profit_factor || '—';
  document.getElementById('btmPnL').innerHTML = `<span style="color:${pnlColor}">$${(m.net_pnl||0).toLocaleString()}</span>`;
  document.getElementById('btmEquity').textContent = '$' + (m.final_equity||0).toLocaleString();
  document.getElementById('btChartMetrics').style.display = 'block';

  // Charts
  const charts = d.charts || {};
  const imgDiv = document.getElementById('btChartImages');
  imgDiv.innerHTML = '';
  const ts = Date.now();
  const chartOrder = ['equity_curve', 'drawdown'];
  // Add equity + drawdown first, then trade charts
  for (const key of chartOrder) {
    if (charts[key]) {
      imgDiv.innerHTML += `<img src="/api/backtest/bt/charts/${charts[key]}?t=${ts}" style="width:100%;border-radius:8px;margin-bottom:8px;">`;
    }
  }
  // Per-symbol trade charts
  const tradeCharts = Object.entries(charts).filter(([k]) => k.startsWith('trades_'));
  if (tradeCharts.length) {
    imgDiv.innerHTML += '<div style="font-size:13px;font-weight:600;margin:8px 0 4px;">Per-Symbol Trade Charts</div>';
    imgDiv.innerHTML += '<div style="max-height:600px;overflow-y:auto;">';
    for (const [k, v] of tradeCharts) {
      imgDiv.innerHTML += `<img src="/api/backtest/bt/charts/${v}?t=${ts}" style="width:100%;border-radius:8px;margin-bottom:8px;">`;
    }
    imgDiv.innerHTML += '</div>';
  }
  imgDiv.style.display = 'block';

  // Trade table
  const trades = d.trades || [];
  if (trades.length) {
    let html = '<div style="font-size:13px;font-weight:600;margin-bottom:4px;">Trades (' + trades.length + ')</div>';
    html += '<div style="max-height:300px;overflow-y:auto;"><table style="width:100%;font-size:11px;border-collapse:collapse;">';
    html += '<tr style="border-bottom:1px solid var(--border);"><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>Entry$</th><th>Exit$</th><th>P&L%</th><th>Reason</th></tr>';
    for (const t of trades) {
      const c = t.pnl_pct >= 0 ? 'var(--green)' : 'var(--red)';
      html += `<tr style="border-bottom:1px solid var(--border);">
        <td>${t.symbol}</td><td>${t.side}</td><td>${t.entry_date}</td><td>${t.exit_date}</td>
        <td>$${t.entry_price}</td><td>$${t.exit_price}</td>
        <td style="color:${c}">${t.pnl_pct>0?'+':''}${t.pnl_pct}%</td><td>${t.exit_reason}</td></tr>`;
    }
    html += '</table></div>';
    document.getElementById('btChartTrades').innerHTML = html;
    document.getElementById('btChartTrades').style.display = 'block';
  }
}

async function fetchDbStats() {
  try {
    const stats = await api('db/stats');
    const el = document.getElementById('dbStatus');
    if (stats.symbol_count > 0) {
      el.innerHTML = `<span style="color:var(--text)">${stats.symbol_count.toLocaleString()}</span> symbols | ` +
        `${stats.min_date} to ${stats.max_date} | ` +
        `<span style="color:var(--text)">${stats.size_mb} MB</span> ` +
        `<span style="color:var(--muted)">(${stats.total_rows.toLocaleString()} rows)</span>`;
    } else {
      el.textContent = 'Empty \u2014 click Build DB to populate';
    }
  } catch(e) {
    document.getElementById('dbStatus').textContent = 'Error loading stats';
  }
}

async function buildDB() {
  const universe = document.getElementById('dbUniverse').value;
  const startDate = document.getElementById('dbStartDate').value;
  if (startDate) {
    const dateRe = /^\\d{4}-\\d{2}-\\d{2}$/;
    if (!dateRe.test(startDate) || isNaN(new Date(startDate).getTime())) {
      showToast('DB start date must be a valid YYYY-MM-DD date'); return;
    }
  }
  const body = { universe };
  if (startDate) body.start_date = startDate;
  document.getElementById('dbProgress').style.display = 'block';
  document.getElementById('dbProgressMsg').textContent = 'Starting build...';
  document.getElementById('dbProgressBar').style.width = '0%';
  try {
    const r = await api('db/build', 'POST', body);
    if (r.error) { showToast('DB build error: ' + r.error); }
    fetchDbStats();
  } catch(e) {
    document.getElementById('dbProgressMsg').textContent = 'Error: ' + e.message;
  }
}

async function updateDB() {
  document.getElementById('dbProgress').style.display = 'block';
  document.getElementById('dbProgressMsg').textContent = 'Starting Alpaca update...';
  document.getElementById('dbProgressBar').style.width = '0%';
  try {
    await api('db/update', 'POST', {});
    fetchDbStats();
  } catch(e) {
    document.getElementById('dbProgressMsg').textContent = 'Error: ' + e.message;
  }
}

async function polygonUpdate(days) {
  days = days || parseInt(document.getElementById('polygonDays').value) || 1;
  document.getElementById('dbProgress').style.display = 'block';
  document.getElementById('dbProgressMsg').textContent = 'Polygon: fetching ' + days + ' day(s)...';
  document.getElementById('dbProgressBar').style.width = '0%';
  try {
    const r = await api('db/polygon-update', 'POST', {days});
    if (r.error) {
      document.getElementById('dbProgressMsg').textContent = 'Error: ' + r.error;
    } else {
      document.getElementById('dbProgressMsg').textContent =
        `Polygon done: ${r.polygon_inserted} rows, ${r.polygon_days} days`;
    }
    fetchDbStats();
  } catch(e) {
    document.getElementById('dbProgressMsg').textContent = 'Error: ' + e.message;
  }
}

async function saveConfig() {
  const inputs = document.querySelectorAll('#configGrid input');
  const body = {};
  let hasErr = false;
  inputs.forEach(inp => {
    const key = inp.dataset.key;
    if (!key) return;
    if (inp.type === 'checkbox') {
      body[key] = inp.checked;
    } else if (inp.type === 'number') {
      const v = parseFloat(inp.value);
      if (isNaN(v)) { showToast('Config "' + key + '" is not a valid number'); hasErr = true; return; }
      body[key] = v;
    } else {
      body[key] = inp.value;
    }
  });
  if (hasErr) return;
  // Include scan universe settings
  const univRadio = document.querySelector('input[name="scanUniverse"]:checked');
  if (univRadio) body.scan_universe = univRadio.value;
  body.custom_symbols = document.getElementById('customSymbolsInput').value.trim();
  const r = await api('config', 'POST', body);
  if (r.validation_errors && r.validation_errors.length) {
    r.validation_errors.forEach(e => showToast(e, 'warning'));
  } else {
    showToast('Config saved', 'success');
  }
  fetchState();
}

async function testLlm() {
  const el = document.getElementById('llmTestResult');
  el.textContent = 'Testing...';
  el.style.color = 'var(--muted)';
  try {
    const r = await api('llm/test', 'POST');
    if (r.status === 'ok') {
      el.textContent = 'Connected — ' + (r.supervisor?.model || 'ok');
      el.style.color = '#c084fc';
    } else {
      el.textContent = r.message || 'Connection failed';
      el.style.color = 'var(--negative)';
    }
  } catch(e) {
    el.textContent = 'Error: ' + e.message;
    el.style.color = 'var(--negative)';
  }
}

// ── Rudra Chat ───────────────────────────────────────────────────
async function sendChat() {
  const input = document.getElementById('chatInput');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';

  const container = document.getElementById('chatMessages');
  // Hide suggestions after first message
  const sug = document.getElementById('chatSuggestions');
  if (sug) sug.style.display = 'none';

  // Add user message
  const userDiv = document.createElement('div');
  userDiv.className = 'chat-msg chat-user';
  userDiv.textContent = msg;
  container.appendChild(userDiv);

  // Add thinking indicator
  const thinkDiv = document.createElement('div');
  thinkDiv.className = 'chat-msg chat-thinking';
  thinkDiv.textContent = 'Thinking...';
  container.appendChild(thinkDiv);
  container.scrollTop = container.scrollHeight;

  // Disable send
  const btn = document.getElementById('chatSendBtn');
  btn.disabled = true;
  btn.style.opacity = '0.5';

  try {
    const r = await api('llm/chat', 'POST', { message: msg });
    thinkDiv.remove();
    const respDiv = document.createElement('div');
    if (r.error) {
      respDiv.className = 'chat-msg chat-error';
      respDiv.textContent = r.error;
    } else {
      respDiv.className = 'chat-msg chat-assistant';
      respDiv.textContent = r.response || 'No response';
    }
    container.appendChild(respDiv);
    // Show action result badge if an action was executed
    if (r.action_result) {
      const actDiv = document.createElement('div');
      actDiv.className = 'chat-msg chat-system';
      const ar = r.action_result;
      if (ar.error) {
        actDiv.style.background = 'rgba(239,68,68,0.12)';
        actDiv.textContent = 'Action failed: ' + ar.error;
      } else {
        actDiv.style.background = 'rgba(16,185,129,0.12)';
        actDiv.textContent = 'Executed: ' + ar.executed + (ar.status ? ' (status: '+ar.status+')' : '')
          + (ar.candidates !== undefined ? ' — '+ar.candidates+' candidates found' : '')
          + (ar.positions ? ' — positions: '+ar.positions.join(', ') : '')
          + (ar.updated ? ' — updated: '+ar.updated.join(', ') : '');
      }
      container.appendChild(actDiv);
      // Refresh state after action
      fetchState();
    }
  } catch (e) {
    thinkDiv.remove();
    const errDiv = document.createElement('div');
    errDiv.className = 'chat-msg chat-error';
    errDiv.textContent = 'Error: ' + e.message;
    container.appendChild(errDiv);
  }

  btn.disabled = false;
  btn.style.opacity = '1';
  container.scrollTop = container.scrollHeight;
}

function sendPreset(el) {
  document.getElementById('chatInput').value = el.textContent;
  sendChat();
}

// ── Rendering ────────────────────────────────────────────────────
function renderState(s) {
  updateStatus(s.status);
  updateEquity(s.equity);
  if (s.llm) updateLlmBadge(s.llm);
  if (s.strategy) {
    const badge = document.getElementById('strategyBadge');
    if (badge) badge.textContent = s.strategy.name || 'Classic';
    const sel = document.getElementById('strategySelect');
    if (sel && sel.value !== s.strategy.active) sel.value = s.strategy.active;
  }

  const m = s.metrics || {};
  const ds = s.daily_stats || {};

  const todayPnl = ds.pnl || 0;
  const el = (id, val) => { const e = document.getElementById(id); if(e) e.textContent = val; };
  const cls = (id, c) => { const e = document.getElementById(id); if(e) { e.className = 'value ' + c; } };

  el('statEquity', '$' + (s.equity||0).toLocaleString(undefined,{minimumFractionDigits:0}));
  el('statTodayPnl', pnlFmt(todayPnl));
  cls('statTodayPnl', todayPnl >= 0 ? 'green' : 'red');

  // Dashboard page elements
  el('dashEquity', '$' + (s.equity||0).toLocaleString(undefined,{minimumFractionDigits:0}));
  el('dashTodayPnl', pnlFmt(todayPnl));
  const dashPnlEl = document.getElementById('dashTodayPnl');
  if (dashPnlEl) dashPnlEl.style.color = todayPnl >= 0 ? 'var(--positive)' : 'var(--negative)';
  el('dashWinRate', ((m.win_rate||0)*100).toFixed(1) + '%');
  el('dashTrades', m.total_trades || 0);
  el('dashPF', (m.profit_factor||0).toFixed(2));
  el('dashMaxDD', (m.max_drawdown_pct||0).toFixed(1) + '%');

  el('statTotalPnl', pnlFmt(m.total_pnl||0));
  cls('statTotalPnl', (m.total_pnl||0) >= 0 ? 'green' : 'red');
  el('statWinRate', ((m.win_rate||0)*100).toFixed(1) + '%');
  el('statTrades', m.total_trades || 0);
  el('statPF', (m.profit_factor||0).toFixed(2));
  el('statMaxDD', (m.max_drawdown_pct||0).toFixed(1) + '%');
  el('statAvgHold', (m.avg_holding_min||0).toFixed(0));

  updateCircuitBreakers(ds);
  renderPositions(s.positions || {});
  renderCandidates(s.candidates || []);
  renderMessages(s.messages || []);
  renderTrades(s.today_trades || [], (s.metrics||{}).total_trades||0);
  renderEquityCurve((s.config || {}).initial_capital || 25000);
  const cfgGrid = document.getElementById('configGrid');
  if (!cfgGrid || !cfgGrid.contains(document.activeElement)) renderConfig(s.config || {});
}

function updateStatus(status) {
  const badge = document.getElementById('statusBadge');
  badge.textContent = status.toUpperCase();
  badge.className = 'status-badge status-' + status;
}

function updateLlmBadge(llm) {
  const badge = document.getElementById('llmBadge');
  if (!llm || !llm.enabled) {
    badge.style.display = 'none';
    return;
  }
  badge.style.display = 'inline-block';
  if (llm.circuit_open) {
    badge.className = 'llm-badge llm-circuit';
    badge.textContent = 'RUDRA CIRCUIT';
    badge.title = 'Rudra circuit breaker open — ' + Math.round(llm.circuit_resets_in) + 's until retry';
  } else if (llm.available) {
    badge.className = 'llm-badge llm-on';
    badge.textContent = 'Rudra';
    badge.title = 'Rudra active: ' + llm.model + ' (' + llm.total_calls + ' calls)';
  } else {
    badge.className = 'llm-badge llm-off';
    badge.textContent = 'RUDRA OFF';
    badge.title = 'Rudra unavailable';
  }
}

function updateEquity(eq) {
  document.getElementById('statEquity').textContent = '$' + (eq||0).toLocaleString(undefined,{minimumFractionDigits:0});
  const dashEq = document.getElementById('dashEquity');
  if (dashEq) dashEq.textContent = '$' + (eq||0).toLocaleString(undefined,{minimumFractionDigits:0});
}

function updateCircuitBreakers(ds) {
  const dailyPnl = ds.pnl || 0;
  const consec = ds.consecutive_losses || 0;
  const halted = ds.halted || false;

  document.getElementById('cbDailyVal').textContent = pnlFmt(dailyPnl);
  document.getElementById('cbDailyVal').style.color = dailyPnl >= 0 ? 'var(--green)' : 'var(--red)';
  document.getElementById('cbDaily').className = 'cb-dot ' + (dailyPnl < -1000 ? 'halt' : (dailyPnl < 0 ? 'warn' : 'ok'));

  document.getElementById('cbConsecVal').textContent = consec;
  document.getElementById('cbConsec').className = 'cb-dot ' + (consec >= 3 ? 'halt' : (consec >= 2 ? 'warn' : 'ok'));

  document.getElementById('cbDD').className = 'cb-dot ' + (halted ? 'halt' : 'ok');
  document.getElementById('cbDDVal').textContent = halted ? (ds.halt_reason||'HALTED') : 'OK';
}

function renderPositions(positions) {
  const tbody = document.getElementById('positionsTable');
  const noPos = document.getElementById('noPositions');
  const keys = Object.keys(positions);

  if (keys.length === 0) {
    tbody.innerHTML = '';
    noPos.style.display = 'block';
    return;
  }
  noPos.style.display = 'none';

  tbody.innerHTML = keys.map(sym => {
    const p = positions[sym];
    const dir = p.direction || 'short';
    // Current price from last_prices ring buffer
    let curPrice = 0;
    if (p.last_prices) {
      const parts = p.last_prices.split(',').filter(Boolean);
      if (parts.length) curPrice = parseFloat(parts[parts.length - 1]);
    }
    const hasCur = curPrice > 0;
    // P&L calculation
    let pnl = 0, pnlPct = 0;
    if (hasCur) {
      pnl = dir === 'short'
        ? (p.entry_price - curPrice) * p.remaining_shares
        : (curPrice - p.entry_price) * p.remaining_shares;
      pnlPct = dir === 'short'
        ? (p.entry_price - curPrice) / p.entry_price * 100
        : (curPrice - p.entry_price) / p.entry_price * 100;
    }
    const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
    const dirColor = dir === 'long' ? 'var(--green)' : 'var(--red)';
    return `<tr>
      <td style="font-weight:600;color:${dirColor};">${sym} <span style="font-size:10px;opacity:0.7;">${dir.toUpperCase()}</span></td>
      <td>${p.remaining_shares}</td>
      <td>$${p.entry_price.toFixed(2)}</td>
      <td>${hasCur ? '$'+curPrice.toFixed(2) : '\u2014'}</td>
      <td style="color:var(--red);cursor:pointer;" title="Click to adjust stop" onclick="promptStopAdjust('${sym}', ${p.stop_price})">$${p.stop_price.toFixed(2)} ✏</td>
      <td style="color:var(--green);">$${p.full_target.toFixed(2)}</td>
      <td style="color:${pnlColor};font-weight:600;">${hasCur ? (pnl>=0?'+':'')+pnl.toFixed(2)+' ('+pnlPct.toFixed(1)+'%)' : '\u2014'}</td>
      <td><button class="pos-close-btn" onclick="closePosition('${sym}')" title="Close position">✕</button></td>
    </tr>`;
  }).join('');
}

async function closePosition(sym) {
  if (!confirm('Close ' + sym + ' at market?')) return;
  try {
    const r = await fetch('/api/positions/' + sym + '/close', {method:'POST'});
    const d = await r.json();
    if (d.error) { alert('Error: ' + d.error); return; }
    addActivityMsg('manual', 'Closed ' + sym + ': $' + (d.pnl||0).toFixed(2));
    refreshState();
  } catch(e) { alert('Failed: ' + e.message); }
}

async function promptStopAdjust(sym, curStop) {
  const newStop = prompt('New stop price for ' + sym + ' (current: $' + curStop.toFixed(2) + '):', curStop.toFixed(2));
  if (!newStop) return;
  const px = parseFloat(newStop);
  if (isNaN(px) || px <= 0) { alert('Invalid price'); return; }
  try {
    const r = await fetch('/api/positions/' + sym + '/stop', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({stop_price: px})
    });
    const d = await r.json();
    if (d.error) { alert('Error: ' + d.error); return; }
    addActivityMsg('manual', sym + ' stop: $' + curStop.toFixed(2) + ' → $' + px.toFixed(2));
    refreshState();
  } catch(e) { alert('Failed: ' + e.message); }
}

function addActivityMsg(type, text) {
  const feed = document.getElementById('feedContainer');
  if (!feed) return;
  const now = new Date().toLocaleTimeString('en-US', {hour12:false, hour:'2-digit', minute:'2-digit', second:'2-digit'});
  const div = document.createElement('div');
  div.className = 'feed-item ' + type;
  div.innerHTML = '<span class="time">' + now + '</span>' + text;
  feed.prepend(div);
}

function refreshState() { fetchState(); }

function catalystBadge(cat, detail) {
  const labels = {earnings:'EARN',fda:'FDA',ma:'M&A',offering:'OFFER',upgrade:'UPG',downgrade:'DNG'};
  const colors = {
    earnings:'var(--red)',fda:'var(--red)',ma:'var(--red)',
    offering:'var(--green)',upgrade:'var(--orange)',downgrade:'var(--orange)',
  };
  if (!cat) return '<span style="color:var(--green);font-weight:600;" title="No catalyst \u2014 clean noise gap">NOISE</span>';
  const label = labels[cat] || cat.toUpperCase();
  const color = colors[cat] || 'var(--muted)';
  const tip = detail ? detail.replace(/"/g, '&quot;') : '';
  return `<span style="color:${color};font-weight:600;cursor:help;" title="${tip}">${label}</span>`;
}

function renderCandidates(candidates) {
  const tbody = document.getElementById('candidatesTable');
  const noC = document.getElementById('noCandidates');

  if (!candidates || candidates.length === 0) {
    tbody.innerHTML = '';
    noC.style.display = 'block';
    return;
  }
  noC.style.display = 'none';

  tbody.innerHTML = candidates.map(c => {
    const dir = c.direction || 'short';
    const dirColor = dir === 'long' ? 'var(--green)' : 'var(--red)';
    const dirLabel = dir === 'long' ? 'LONG' : 'SHORT';
    return `<tr>
    <td style="font-weight:600;">${c.symbol}</td>
    <td style="color:${dirColor};font-weight:600;">${dirLabel}</td>
    <td style="color:var(--orange);">${(c.gap_pct*100).toFixed(1)}%</td>
    <td>$${c.prev_close.toFixed(2)}</td>
    <td>$${c.premarket_price.toFixed(2)}</td>
    <td style="color:${c.vol_ratio < 0.5 ? 'var(--green)' : (c.vol_ratio < 1 ? 'var(--cyan)' : 'var(--red)')};">
      ${c.vol_ratio.toFixed(2)}x</td>
    <td>${(c.avg_vol_20d/1000).toFixed(0)}K</td>
    <td>${c.shortable ? '\u2713' : '\u2717'}</td>
    <td>${c.easy_to_borrow ? '\u2713' : '\u2014'}</td>
    <td>${catalystBadge(c.catalyst || '', c.catalyst_detail || '')}</td>
    <td>${c.score.toFixed(1)}</td>
  </tr>`;
  }).join('');
}

function renderMessages(messages) {
  const feed = document.getElementById('feedContainer');
  const prefix = (type) => {
    if (type === 'llm') return '<span style="color:#c084fc;font-weight:600;margin-right:4px;">RUDRA</span>';
    if (type === 'llm_review') return '<span style="color:#06b6d4;font-weight:600;margin-right:4px;">AUTO</span>';
    if (type === 'llm_fallback') return '<span style="color:#fde68a;font-weight:600;margin-right:4px;">RUDRA</span>';
    if (type === 'manual') return '<span style="color:#f59e0b;font-weight:600;margin-right:4px;">MANUAL</span>';
    if (type === 'conversation') return '<span style="color:#34d399;font-weight:600;margin-right:4px;">CONV</span>';
    return '';
  };
  const formatMsg = (m) => {
    if (m.type === 'conversation') {
      // Split BOT/RUDRA conversation for styled display
      const parts = m.text.split('RUDRA:');
      if (parts.length === 2) {
        const botPart = parts[0].replace('BOT:', '').trim();
        const rudraPart = parts[1].trim();
        return `<div class="feed-item conversation" style="border-left:2px solid #34d399;padding-left:8px;">
          <span class="time">${m.time}</span>
          <span style="color:#60a5fa;font-weight:600;">BOT:</span> ${botPart}<br/>
          <span style="color:#c084fc;font-weight:600;">RUDRA:</span> ${rudraPart}
        </div>`;
      }
    }
    return `<div class="feed-item ${m.type}">
      <span class="time">${m.time}</span>${prefix(m.type)}${m.text}
    </div>`;
  };
  feed.innerHTML = messages.slice().reverse().map(formatMsg).join('');
}

// ── Equity Curve (canvas, fetched from SQLite) ──
let _eqLoaded = false, _eqState = null;
function renderEquityCurve(initialCapital) {
  if (_eqLoaded) return;
  _eqLoaded = true;
  fetch('/api/trades/history?limit=1000&offset=0').then(r => r.json()).then(d => {
    const trades = (d.trades || []).slice().reverse(); // oldest first
    if (!trades.length) return;
    const canvas = document.getElementById('equityChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    const pad = { top: 16, right: 12, bottom: 24, left: 55 };

    // Build equity points: [label, equity, symbol, pnl]
    const cap = initialCapital || 25000;
    let eq = cap;
    const pts = [[trades[0].entry_time.slice(0,10), cap, '', 0]];
    trades.forEach(t => { eq += t.pnl; pts.push([t.exit_time.slice(0,10), eq, t.symbol, t.pnl]); });

    const vals = pts.map(p => p[1]);
    const minY = Math.min(...vals) * 0.998;
    const maxY = Math.max(...vals) * 1.002;
    const rangeY = maxY - minY || 1;
    const xStep = (W - pad.left - pad.right) / Math.max(pts.length - 1, 1);
    const toX = i => pad.left + i * xStep;
    const toY = v => pad.top + (1 - (v - minY) / rangeY) * (H - pad.top - pad.bottom);

    // Store state for mouseover
    _eqState = { pts, toX, toY, pad, W, H, cap };

    // Background
    ctx.fillStyle = 'rgba(0,0,0,0.2)';
    ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    const nGrid = 4;
    for (let i = 0; i <= nGrid; i++) {
      const y = pad.top + i * (H - pad.top - pad.bottom) / nGrid;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
      const val = maxY - i * rangeY / nGrid;
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText('$' + val.toFixed(0), pad.left - 4, y + 3);
    }

    // Baseline at initial capital
    const baseY = toY(cap);
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(pad.left, baseY); ctx.lineTo(W - pad.right, baseY); ctx.stroke();
    ctx.setLineDash([]);

    // Fill gradient
    const grad = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
    const finalEq = pts[pts.length-1][1];
    if (finalEq >= cap) {
      grad.addColorStop(0, 'rgba(34,197,94,0.25)');
      grad.addColorStop(1, 'rgba(34,197,94,0.02)');
    } else {
      grad.addColorStop(0, 'rgba(239,68,68,0.02)');
      grad.addColorStop(1, 'rgba(239,68,68,0.25)');
    }
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(pts[0][1]));
    pts.forEach((p, i) => ctx.lineTo(toX(i), toY(p[1])));
    ctx.lineTo(toX(pts.length-1), H - pad.bottom);
    ctx.lineTo(toX(0), H - pad.bottom);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.strokeStyle = finalEq >= cap ? '#22c55e' : '#ef4444';
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    pts.forEach((p, i) => { i === 0 ? ctx.moveTo(toX(i), toY(p[1])) : ctx.lineTo(toX(i), toY(p[1])); });
    ctx.stroke();

    // X-axis labels (sparse)
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    const labelEvery = Math.max(1, Math.floor(pts.length / 6));
    pts.forEach((p, i) => {
      if (i % labelEvery === 0 || i === pts.length - 1) {
        ctx.fillText(p[0].slice(5), toX(i), H - 4); // MM-DD
      }
    });

    // Final equity label
    ctx.fillStyle = finalEq >= cap ? '#22c55e' : '#ef4444';
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('$' + finalEq.toFixed(0), toX(pts.length-1) + 4, toY(finalEq) + 3);

    // Mouse interaction
    const tooltip = document.getElementById('eqTooltip');
    const crosshair = document.getElementById('eqCrosshair');
    const parent = canvas.parentElement;
    canvas.addEventListener('mousemove', e => {
      if (!_eqState) return;
      const br = canvas.getBoundingClientRect();
      const mx = e.clientX - br.left;
      // Find nearest point
      let best = 0, bestDist = Infinity;
      for (let i = 0; i < _eqState.pts.length; i++) {
        const dist = Math.abs(_eqState.toX(i) - mx);
        if (dist < bestDist) { bestDist = dist; best = i; }
      }
      const p = _eqState.pts[best];
      const px = _eqState.toX(best);
      const py = _eqState.toY(p[1]);
      const pnlFromStart = p[1] - _eqState.cap;
      const sign = pnlFromStart >= 0 ? '+' : '';
      const clr = pnlFromStart >= 0 ? '#22c55e' : '#ef4444';
      let html = '<div style="color:var(--muted)">' + p[0] + '</div>';
      html += '<div style="font-weight:600;">$' + p[1].toFixed(2) + '</div>';
      html += '<div style="color:' + clr + '">' + sign + '$' + pnlFromStart.toFixed(2) + '</div>';
      if (p[2]) html += '<div style="color:var(--muted)">' + p[2] + ' ' + (p[3]>=0?'+':'') + '$' + p[3].toFixed(2) + '</div>';
      tooltip.innerHTML = html;
      tooltip.style.display = 'block';
      // px/py are in canvas CSS coordinates (same as parent-relative)
      const tipLeft = (px + 12 + 140 > _eqState.W) ? px - 140 : px + 12;
      tooltip.style.left = tipLeft + 'px';
      tooltip.style.top = Math.max(0, py - 10) + 'px';
      crosshair.style.display = 'block';
      crosshair.style.left = px + 'px';
    });
    canvas.addEventListener('mouseleave', () => {
      tooltip.style.display = 'none';
      crosshair.style.display = 'none';
    });
  }).catch(e => console.error('Equity curve fetch error:', e));
}

// ── Trade History (SQLite-backed, paginated) ──
let _thOffset = 0, _thLimit = 50, _thLoaded = false;
function tradeHistSearch(offset) {
  _thOffset = offset || 0;
  const start = document.getElementById('thStart').value || '';
  const end = document.getElementById('thEnd').value || '';
  const sym = document.getElementById('thSymbol').value.trim().toUpperCase() || '';
  let url = '/api/trades/history?limit=' + _thLimit + '&offset=' + _thOffset;
  if (start) url += '&start=' + start;
  if (end) url += '&end=' + end + 'T23:59:59';
  if (sym) url += '&symbol=' + sym;
  fetch(url).then(r => r.json()).then(d => {
    _thLoaded = true;
    const tbody = document.getElementById('tradesTable');
    const trades = d.trades || [];
    const total = d.total || 0;
    document.getElementById('tradeHistCount').textContent = '(' + total + ' trades)';
    tbody.innerHTML = trades.map(t => {
      const side = (t.side || 'short').toUpperCase();
      const sideColor = side === 'LONG' ? 'var(--green)' : 'var(--red)';
      return `<tr>
        <td>${t.exit_time || ''}</td>
        <td style="font-weight:600;">${t.symbol}</td>
        <td style="color:${sideColor};">${side}</td>
        <td>${t.shares}</td>
        <td>$${t.entry_price.toFixed(2)}</td>
        <td>$${t.exit_price.toFixed(2)}</td>
        <td class="${t.pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">${pnlFmt(t.pnl)}</td>
        <td class="${t.pnl_pct >= 0 ? 'pnl-pos' : 'pnl-neg'}">${(t.pnl_pct*100).toFixed(2)}%</td>
        <td>${t.exit_reason}</td>
        <td>${t.holding_minutes || 0}m</td>
      </tr>`;
    }).join('') || '<tr><td colspan="10" style="text-align:center;color:var(--muted);">No trades found</td></tr>';
    // Pager
    const pager = document.getElementById('tradeHistPager');
    const page = Math.floor(_thOffset / _thLimit) + 1;
    const pages = Math.ceil(total / _thLimit);
    let html = '';
    if (_thOffset > 0)
      html += '<button onclick="tradeHistSearch(' + (_thOffset - _thLimit) + ')" style="padding:3px 10px;background:var(--border);color:var(--text);border:none;border-radius:4px;cursor:pointer;">&laquo; Prev</button>';
    html += '<span style="color:var(--muted);">Page ' + page + ' / ' + pages + '</span>';
    if (_thOffset + _thLimit < total)
      html += '<button onclick="tradeHistSearch(' + (_thOffset + _thLimit) + ')" style="padding:3px 10px;background:var(--border);color:var(--text);border:none;border-radius:4px;cursor:pointer;">Next &raquo;</button>';
    pager.innerHTML = html;
  }).catch(e => console.error('Trade history fetch error:', e));
}
function tradeHistReset() {
  document.getElementById('thStart').value = '';
  document.getElementById('thEnd').value = '';
  document.getElementById('thSymbol').value = '';
  tradeHistSearch(0);
}
function renderTrades(todayTrades, totalCount) {
  // Update last-trade status badge from today's trades
  const trades = todayTrades || [];
  const lastEl = document.getElementById('statusLastTrade');
  if (lastEl && trades.length > 0) {
    const t = trades[trades.length - 1];
    lastEl.textContent = 'Last: ' + (t.symbol || '') + ' ' + (t.side || '').toUpperCase() + ' ' + (t.shares || 0) + ' @ $' + (t.exit_price || 0).toFixed(2);
  } else if (lastEl) lastEl.textContent = 'Last: \u2014';
  // Auto-load full history on first render
  if (!_thLoaded) tradeHistSearch(0);
}

function renderConfig(config) {
  const grid = document.getElementById('configGrid');
  const v = (k, def) => config[k] !== undefined ? config[k] : def;
  const inp = (key, label, def) => {
    const val = v(key, def);
    const t = typeof val === 'string' ? 'text' : 'number';
    return `<div class="config-item"><label>${label}</label>
      <input type="${t}" data-key="${key}" value="${val}" ${t==='number'?'step="any"':''}></div>`;
  };

  // Grouped sections
  const sections = [
    { title: 'Gap Detection', fields: [
      ['gap_threshold', 'Gap Threshold', 0.07], ['max_gap_pct', 'Max Gap %', 0.5],
      ['vol_ratio_max', 'Vol Ratio Max', 3], ['min_avg_volume', 'Min Avg Volume', 5000],
      ['min_price', 'Min Price', 10],
    ]},
    { title: 'Position Sizing', fields: [
      ['max_positions', 'Max Positions', 5], ['initial_capital', 'Initial Capital', 25000],
      ['risk_pct', 'Risk Per Trade %', 0.02], ['kelly_fraction', 'Kelly Fraction', 0.25],
      ['max_notional', 'Max Notional $', 50000],
    ]},
    { title: 'Risk Management', fields: [
      ['stop_pct', 'Fixed Stop Loss %', 0.015], ['daily_loss_limit', 'Daily Loss Limit', 0.02],
      ['max_consec_losses', 'Max Consec Losses', 3], ['max_drawdown', 'Max Drawdown', 0.05],
      ['time_exit_hour', 'Time Exit Hour', 15],
    ]},
    { title: 'Order Execution', fields: [
      ['slippage_pct', 'Slippage %', 0.0015], ['borrow_rate_annual', 'Borrow Rate', 0.02],
      ['max_pct_adv', 'Max % ADV', 0.02], ['limit_offset_pct', 'Limit Offset %', 0.001],
    ]},
  ];

  grid.innerHTML = sections.map(s => `
    <div class="cfg-section">
      <div class="cfg-section-head">${s.title}</div>
      <div class="cfg-section-body">
        ${s.title === 'Order Execution' ? `
          <div class="config-item">
            <label>Limit Orders Only</label>
            <div style="display:flex;align-items:center;gap:8px;padding:4px 0;">
              <input type="checkbox" id="cfgLimitOrders" data-key="limit_orders_only"
                ${config.limit_orders_only ? 'checked' : ''}
                style="width:16px;height:16px;accent-color:var(--accent-green);">
              <span style="font-size:11px;color:var(--muted);">Better fills, less slippage</span>
            </div>
          </div>` : ''}
        ${s.title === 'Risk Management' && config.adaptive_stops ? `
          <div style="grid-column:1/-1;font-size:10px;color:#f59e0b;padding:2px 6px;background:rgba(245,158,11,0.08);border-radius:4px;margin-bottom:4px;">
            Adaptive Stops ON — Fixed Stop Loss is overridden. Actual stop = gap% \u00d7 ${(config.stop_gap_fraction||0.25).toFixed(2)}, clamped to [${((config.stop_min_pct||0.015)*100).toFixed(1)}%, ${((config.stop_max_pct||0.05)*100).toFixed(1)}%]
          </div>` : ''}
        ${s.fields.map(f => inp(...f)).join('')}
      </div>
    </div>
  `).join('');

  // Feature toggle sections
  const features = [
    { key: 'adaptive_stops', title: 'Adaptive Stops', desc: 'Scale stop with gap size (overrides Fixed Stop Loss %). Stop = gap% \u00d7 fraction, clamped to [min, max]',
      params: [['stop_gap_fraction', 'Gap Fraction', 0.25], ['stop_min_pct', 'Min Stop %', 0.015], ['stop_max_pct', 'Max Stop %', 0.05]] },
    { key: 'regime_filter', title: 'Market Regime Filter', desc: 'Reduce or block entries when SPY gaps up or VIX is elevated',
      params: [['regime_spy_gap_limit', 'SPY Gap Limit', 0.01], ['regime_spy_block_pct', 'SPY Block %', 0.015], ['regime_vix_threshold', 'VIX Threshold', 25.0]] },
    { key: 'reentry_enabled', title: 'Re-entry After Stop-out', desc: 'Re-enter a stopped position if price reverses favorably',
      params: [['reentry_cooldown_minutes', 'Cooldown (min)', 30], ['reentry_max_per_symbol', 'Max Per Symbol', 1], ['reentry_stop_pct', 'Stop %', 0.01], ['reentry_trigger_pct', 'Trigger %', 0.0]] },
    { key: 'trade_gap_downs', title: 'Gap-Down Fading (Longs)', desc: 'Buy gap-downs and fade back toward previous close',
      params: [['gap_down_threshold', 'Gap Down %', 0.05], ['gap_down_max_pct', 'Max Gap Down %', 0.50], ['gap_down_vol_ratio_max', 'Vol Ratio Max', 3.0]] },
    { key: 'dd_circuit_breaker', title: 'Drawdown Circuit Breaker', desc: 'Graduated position size reduction during drawdowns — Tier 1 reduces size, Tier 2 caps to 1 position, Hard Stop halts trading',
      params: [['dd_tier1_threshold', 'Tier 1 DD%', 0.15], ['dd_tier1_scale', 'Tier 1 Scale', 0.50], ['dd_tier2_threshold', 'Tier 2 DD%', 0.25], ['dd_tier2_scale', 'Tier 2 Scale', 0.25], ['dd_tier2_max_positions', 'Tier 2 Max Pos', 1], ['dd_hard_stop', 'Hard Stop DD%', 0.0]] },
    { key: 'llm_enabled', title: 'Rudra (LLM Supervisor)', desc: 'Autonomous decisions via local Ollama model — scan timing, candidate selection, profit-taking, exit overrides',
      params: [['llm_url', 'Ollama URL', 'http://localhost:11434'], ['llm_model', 'Model', 'gpt-oss:20b'], ['llm_timeout', 'Timeout (sec)', 30], ['llm_max_failures', 'Circuit Breaker Failures', 5], ['llm_circuit_reset', 'Circuit Reset (sec)', 120], ['llm_max_hold_overrides', 'Max Hold Overrides', 2]] },
  ];

  // Section header for feature toggles
  grid.insertAdjacentHTML('beforeend', `
    <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:var(--muted);padding:4px 0;">Feature Toggles</div>
  `);

  features.forEach(f => {
    const on = !!config[f.key];
    const secId = 'params-' + f.key;
    const html = `
      <div class="feature-section${on ? ' enabled' : ''}" id="sec-${f.key}">
        <div class="feature-header" onclick="const cb=this.querySelector('input');cb.checked=!cb.checked;cb.dispatchEvent(new Event('change'));">
          <input type="checkbox" data-key="${f.key}" ${on ? 'checked' : ''}
            onclick="event.stopPropagation();"
            onchange="const sec=document.getElementById('sec-${f.key}');const p=document.getElementById('${secId}');p.style.display=this.checked?'grid':'none';sec.classList.toggle('enabled',this.checked);">
          <span class="feature-title">${f.title}</span>
          <span class="feature-desc">${f.desc}</span>
        </div>
        <div class="feature-params" id="${secId}" style="display:${on ? 'grid' : 'none'}">
          ${f.params.map(([pk, pl, def]) => inp(pk, pl, def)).join('')}
        </div>
      </div>`;
    grid.insertAdjacentHTML('beforeend', html);
  });

  // LLM test button
  const llmParams = document.getElementById('params-llm_enabled');
  if (llmParams) {
    llmParams.insertAdjacentHTML('beforeend', `
      <div class="config-item" style="grid-column:1/-1;">
        <label>&nbsp;</label>
        <div style="display:flex;align-items:center;gap:10px;">
          <button onclick="testLlm()" style="padding:7px 18px;border-radius:6px;background:rgba(168,85,247,0.15);color:#c084fc;border:1px solid rgba(168,85,247,0.25);cursor:pointer;font-size:12px;font-weight:600;transition:all 0.15s;"
            onmouseenter="this.style.background='rgba(168,85,247,0.25)'" onmouseleave="this.style.background='rgba(168,85,247,0.15)'">
            Test Connection
          </button>
          <span id="llmTestResult" style="font-size:11px;color:var(--muted);"></span>
        </div>
      </div>
    `);
  }

  // Scan universe radio buttons
  const univ = config.scan_universe || 'study';
  const radioMap = { alpaca: 'univAlpaca', study: 'univStudy', custom: 'univCustom' };
  const radioEl = document.getElementById(radioMap[univ] || 'univStudy');
  if (radioEl) radioEl.checked = true;
  document.getElementById('customSymbolsInput').value = config.custom_symbols || '';
  document.getElementById('customSymbolsRow').style.display = univ === 'custom' ? 'block' : 'none';
}

// ── P&L Bar Chart ────────────────────────────────────────────────
function setPnlGranularity(gran, btn) {
  document.querySelectorAll('.pnl-gran-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  renderPnlBarChart(gran);
}

function _dateToBucket(dateStr, gran) {
  // dateStr is 'YYYY-MM-DD' (from daily_summary.date) or 'YYYY-MM-DD HH:MM' (from trade)
  const ds = dateStr.slice(0, 10);
  if (gran === 'day') return ds;
  const d = new Date(ds + 'T12:00:00');
  if (gran === 'year') return d.getFullYear().toString();
  if (gran === 'month') return ds.slice(0, 7); // YYYY-MM
  // week: find Monday of that week
  const tmp = new Date(d);
  const dow = (tmp.getDay() + 6) % 7; // 0=Mon, 6=Sun
  tmp.setDate(tmp.getDate() - dow);
  return tmp.toISOString().slice(0, 10);
}

function renderPnlBarChart(gran) {
  const canvas = document.getElementById('pnlBarChart');
  const wrap = document.getElementById('pnlChartWrap');
  const tooltip = document.getElementById('pnlTooltip');
  const countEl = document.getElementById('pnlBarCount');
  if (!canvas || !wrap) return;
  const ctx = canvas.getContext('2d');

  // Build daily P&L map from daily_summary (preferred) or all_trades
  const dailyPnl = {};
  if (window._btDailySummary && window._btDailySummary.length > 0) {
    window._btDailySummary.forEach(d => {
      if (d.date && d.pnl !== undefined) {
        const key = d.date.slice(0, 10);
        dailyPnl[key] = (dailyPnl[key] || 0) + d.pnl;
      }
    });
  } else if (window._btAllTrades && window._btAllTrades.length > 0) {
    window._btAllTrades.forEach(t => {
      const ds = (t.exit_time || t.entry_time || '').slice(0, 10);
      if (ds) dailyPnl[ds] = (dailyPnl[ds] || 0) + (t.pnl || 0);
    });
  }

  // Re-bucket into chosen granularity
  const buckets = {};
  Object.keys(dailyPnl).forEach(day => {
    const key = _dateToBucket(day, gran);
    buckets[key] = (buckets[key] || 0) + dailyPnl[day];
  });

  const keys = Object.keys(buckets).sort();
  const vals = keys.map(k => buckets[k]);
  if (countEl) countEl.textContent = keys.length + ' ' + gran + (keys.length !== 1 ? 's' : '');
  if (keys.length === 0) { ctx.clearRect(0,0,canvas.width,canvas.height); return; }

  const maxAbs = Math.max(Math.abs(Math.min(...vals)), Math.abs(Math.max(...vals)), 1);

  // Sizing: ensure minimum bar width so every label is visible
  const dpr = window.devicePixelRatio || 1;
  const visibleW = wrap.clientWidth;
  const h = wrap.clientHeight;
  const padL = 58, padR = 14, padT = 12, padB = 52;
  const minBarStep = 22; // minimum px per bar (bar + gap)
  const neededW = Math.max(visibleW, padL + padR + keys.length * minBarStep);

  canvas.width = neededW * dpr;
  canvas.height = h * dpr;
  canvas.style.width = neededW + 'px';
  canvas.style.height = h + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const chartW = neededW - padL - padR;
  const chartH = h - padT - padB;
  const step = chartW / keys.length;
  const barW = Math.max(4, step * 0.7);
  const barGap = step - barW;
  const zeroY = padT + chartH / 2;

  ctx.clearRect(0, 0, neededW, h);

  // Store bar rects for hit-testing
  window._pnlBars = [];

  // Grid lines + Y axis labels
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  const ticks = 4;
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'right';
  for (let i = 0; i <= ticks; i++) {
    const frac = i / ticks;
    const y = padT + frac * chartH;
    const val = maxAbs - frac * 2 * maxAbs;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(neededW - padR, y); ctx.stroke();
    ctx.fillStyle = '#64748b';
    ctx.fillText('$' + val.toFixed(0), padL - 6, y + 4);
  }

  // Zero line
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(padL, zeroY); ctx.lineTo(neededW - padR, zeroY); ctx.stroke();

  // Bars + labels
  keys.forEach((key, i) => {
    const v = vals[i];
    const x = padL + barGap / 2 + i * step;
    const barH = (Math.abs(v) / maxAbs) * (chartH / 2);
    const y = v >= 0 ? zeroY - barH : zeroY;
    const colorFade = v >= 0 ? 'rgba(0,212,255,0.7)' : 'rgba(255,59,92,0.7)';

    // Store rect for tooltip hit-test
    window._pnlBars.push({ x, y: v >= 0 ? y : zeroY, w: barW, h: barH, key, val: v });

    ctx.fillStyle = colorFade;
    ctx.beginPath();
    const r = Math.min(3, barW / 2);
    if (v >= 0) {
      ctx.moveTo(x, zeroY);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.lineTo(x + barW - r, y);
      ctx.quadraticCurveTo(x + barW, y, x + barW, y + r);
      ctx.lineTo(x + barW, zeroY);
    } else {
      ctx.moveTo(x, zeroY);
      ctx.lineTo(x, zeroY + barH - r);
      ctx.quadraticCurveTo(x, zeroY + barH, x + r, zeroY + barH);
      ctx.lineTo(x + barW - r, zeroY + barH);
      ctx.quadraticCurveTo(x + barW, zeroY + barH, x + barW, zeroY + barH - r);
      ctx.lineTo(x + barW, zeroY);
    }
    ctx.fill();

    // X label — every bar gets a label, rotated 45deg
    ctx.save();
    ctx.fillStyle = '#64748b';
    ctx.font = '9px JetBrains Mono, monospace';
    const lx = x + barW / 2;
    const ly = h - padB + 10;
    ctx.translate(lx, ly);
    ctx.rotate(-Math.PI / 4);
    ctx.textAlign = 'right';
    ctx.fillText(key, 0, 0);
    ctx.restore();
  });

  // ── Tooltip on hover ──
  // Remove old listeners to avoid stacking
  canvas._pnlMove && canvas.removeEventListener('mousemove', canvas._pnlMove);
  canvas._pnlLeave && canvas.removeEventListener('mouseleave', canvas._pnlLeave);

  canvas._pnlMove = function(e) {
    if (!tooltip || !window._pnlBars) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / dpr / rect.width;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * (canvas.height / dpr / rect.height);
    let hit = null;
    for (const bar of window._pnlBars) {
      if (mx >= bar.x && mx <= bar.x + bar.w && my >= Math.min(bar.y, zeroY) && my <= Math.max(bar.y + bar.h, zeroY)) {
        hit = bar; break;
      }
    }
    if (!hit) {
      // Also detect by x-column regardless of y (easier to aim)
      for (const bar of window._pnlBars) {
        if (mx >= bar.x - 2 && mx <= bar.x + bar.w + 2) { hit = bar; break; }
      }
    }
    if (hit) {
      const sign = hit.val >= 0 ? '+' : '-';
      const color = hit.val >= 0 ? '#00D4FF' : '#FF3B5C';
      tooltip.innerHTML = '<span style="color:var(--muted);">' + hit.key + '</span> &nbsp; <span style="color:' + color + ';font-weight:600;">' + sign + '$' + Math.abs(hit.val).toFixed(2) + '</span>';
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX + 12) + 'px';
      tooltip.style.top = (e.clientY - 10) + 'px';
      canvas.style.cursor = 'crosshair';
    } else {
      tooltip.style.display = 'none';
      canvas.style.cursor = '';
    }
  };
  canvas._pnlLeave = function() {
    if (tooltip) tooltip.style.display = 'none';
    canvas.style.cursor = '';
  };
  canvas.addEventListener('mousemove', canvas._pnlMove);
  canvas.addEventListener('mouseleave', canvas._pnlLeave);

  // Auto-scroll to the right so latest bars are visible
  wrap.scrollLeft = wrap.scrollWidth;
}

function renderBacktestResults(r) {
  if (r.error) {
    document.getElementById('btResults').innerHTML = `<div style="color:var(--red);">${r.error}</div>`;
    return;
  }

  const warnings = r.warnings || [];
  const realism = r.realism || {};
  const html = `
    ${r.strategy ? `<div style="margin-bottom:8px;padding:6px 12px;background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.15);border-radius:6px;font-size:12px;color:#60a5fa;">
      Strategy: <strong>${r.strategy.name}</strong>
    </div>` : ''}
    ${warnings.length > 0 ? `<div style="margin-bottom:10px;padding:10px 14px;background:rgba(234,179,8,0.10);border:1px solid rgba(234,179,8,0.3);border-radius:6px;font-size:12px;color:var(--yellow);">
      <strong>Warnings:</strong><br>${warnings.map(w => '&bull; ' + w).join('<br>')}
    </div>` : ''}
    <div style="margin-bottom:10px;padding:8px 12px;background:rgba(168,85,247,0.08);border-radius:6px;font-size:12px;color:var(--muted);">
      Funnel: <span style="color:var(--text)">${r.raw_gaps||'?'}</span> raw gaps
      &rarr; <span style="color:var(--text)">${r.gap_days_found||0}</span> vol-filtered
      ${r.bars_missing ? `&rarr; <span style="color:var(--orange)">${r.bars_missing} no 1m data</span>` : ''}
      &rarr; <span style="color:var(--green)">${r.gap_days_traded||0}</span> traded
      &nbsp;|&nbsp; Slippage: ${((realism.slippage_pct||0)*100).toFixed(2)}%
      &nbsp;|&nbsp; Borrow: ${((realism.borrow_rate_annual||0)*100).toFixed(1)}%/yr
      &nbsp;|&nbsp; ADV cap: ${((realism.max_pct_adv||0)*100).toFixed(0)}%
      &nbsp;|&nbsp; Adverse fill: ${realism.adverse_fill ? ((realism.adverse_fill_pct||1)*100).toFixed(0)+'%' : 'OFF'}
    </div>
    ${r.circuit_breaker ? `<div style="margin-bottom:10px;padding:8px 12px;background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.15);border-radius:6px;font-size:12px;color:var(--muted);">
      <strong style="color:var(--red);">Circuit Breaker:</strong>
      &nbsp; Tier 1 (${(r.circuit_breaker.tier1_threshold*100).toFixed(0)}% DD): <span style="color:var(--yellow)">${r.circuit_breaker.days_in_tier1}</span> days
      &nbsp;|&nbsp; Tier 2 (${(r.circuit_breaker.tier2_threshold*100).toFixed(0)}% DD): <span style="color:var(--orange)">${r.circuit_breaker.days_in_tier2}</span> days
      &nbsp;|&nbsp; Hard Stop: <span style="color:${r.circuit_breaker.hard_stopped?'var(--red)':'var(--green)'}">${r.circuit_breaker.hard_stopped ? 'TRIGGERED ' + r.circuit_breaker.hard_stop_date : (r.circuit_breaker.hard_stop > 0 ? 'Not triggered' : 'Disabled')}</span>
      ${r.circuit_breaker.trades_skipped_hard_stop > 0 ? '&nbsp;|&nbsp; Skipped: <span style="color:var(--red)">' + r.circuit_breaker.trades_skipped_hard_stop + '</span> trades' : ''}
      &nbsp;|&nbsp; Final DD: <span style="color:var(--red)">${r.circuit_breaker.final_drawdown.toFixed(1)}%</span>
    </div>` : ''}
    <div class="stats-grid" style="margin-bottom:12px;">
      <div class="stat"><div class="label">Trades</div><div class="value">${r.total_trades||0}</div></div>
      <div class="stat"><div class="label">Win Rate</div><div class="value">${((r.win_rate||0)*100).toFixed(1)}%</div></div>
      <div class="stat"><div class="label">Total P&L</div><div class="value ${(r.total_pnl||0)>=0?'green':'red'}">$${(r.total_pnl||0).toFixed(0)}</div></div>
      <div class="stat"><div class="label">Return</div><div class="value ${(r.return_pct||0)>=0?'green':'red'}">${(r.return_pct||0).toFixed(1)}%</div></div>
      <div class="stat"><div class="label">Profit Factor</div><div class="value">${(r.profit_factor||0).toFixed(2)}</div></div>
      <div class="stat"><div class="label">Max DD</div><div class="value red">${(r.max_drawdown_pct||0).toFixed(1)}%</div></div>
      <div class="stat"><div class="label">Avg Win</div><div class="value green">$${(r.avg_win||0).toFixed(0)}</div></div>
      <div class="stat"><div class="label">Avg Loss</div><div class="value red">$${(r.avg_loss||0).toFixed(0)}</div></div>
      <div class="stat"><div class="label">Sharpe</div><div class="value">${(r.sharpe||0).toFixed(2)}</div></div>
      <div class="stat"><div class="label">Gap Days</div><div class="value">${r.gap_days_found||0}</div></div>
      <div class="stat"><div class="label">Traded</div><div class="value">${r.gap_days_traded||0}</div></div>
    </div>
    ${(r.daily_summary && r.daily_summary.length > 0) || (r.all_trades && r.all_trades.length > 0) ? `
      <div style="margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
          <h2 style="margin:0;">P&L Distribution</h2>
          <span id="pnlBarCount" style="font-size:11px;color:var(--muted);"></span>
          <div style="display:flex;gap:2px;margin-left:auto;">
            <button class="pnl-gran-btn active" onclick="setPnlGranularity('day',this)">Day</button>
            <button class="pnl-gran-btn" onclick="setPnlGranularity('week',this)">Week</button>
            <button class="pnl-gran-btn" onclick="setPnlGranularity('month',this)">Month</button>
            <button class="pnl-gran-btn" onclick="setPnlGranularity('year',this)">Year</button>
          </div>
        </div>
        <div id="pnlChartWrap" style="position:relative;height:220px;background:rgba(0,0,0,0.2);border:1px solid var(--border);border-radius:6px;overflow-x:auto;overflow-y:hidden;">
          <canvas id="pnlBarChart" style="display:block;"></canvas>
          <div id="pnlTooltip" style="display:none;position:fixed;pointer-events:none;z-index:100;padding:6px 10px;background:rgba(10,14,23,0.95);border:1px solid var(--border-hover);border-radius:6px;font-size:11px;font-family:'JetBrains Mono',monospace;white-space:nowrap;box-shadow:0 4px 16px rgba(0,0,0,0.5);"></div>
        </div>
      </div>
    ` : ''}
    ${r.trades && r.trades.length > 0 ? '<div id="btTradesSection"></div>' : ''}
  `;
  document.getElementById('btResults').innerHTML = html;

  // Render P&L bar chart — use daily_summary (complete, not truncated) or fall back to all_trades
  if (r.daily_summary && r.daily_summary.length > 0) {
    window._btDailySummary = r.daily_summary;
    window._btAllTrades = r.all_trades || r.trades || [];
    renderPnlBarChart('day');
  } else if (r.all_trades && r.all_trades.length > 0) {
    window._btDailySummary = null;
    window._btAllTrades = r.all_trades;
    renderPnlBarChart('day');
  }

  // Render backtest trades table separately (avoids nested template literal issues)
  if (r.trades && r.trades.length > 0) {
    const section = document.getElementById('btTradesSection');
    if (section) {
      const rows = r.trades.slice(-50).reverse().map(t => {
        const s = (t.side||'short').toUpperCase();
        const sc = s === 'LONG' ? 'var(--green)' : 'var(--red)';
        return '<tr>' +
          '<td>' + (t.entry_time||'') + '</td>' +
          '<td>' + (t.exit_time||'') + '</td>' +
          '<td style="font-weight:600;">' + t.symbol + '</td>' +
          '<td style="color:' + sc + ';font-weight:600;">' + s + '</td>' +
          '<td>' + (t.shares||0) + '</td>' +
          '<td>$' + (t.entry_price||0).toFixed(2) + '</td>' +
          '<td>$' + (t.exit_price||0).toFixed(2) + '</td>' +
          '<td class="' + ((t.pnl||0)>=0?'pnl-pos':'pnl-neg') + '">' + pnlFmt(t.pnl||0) + '</td>' +
          '<td class="' + ((t.pnl_pct||0)>=0?'pnl-pos':'pnl-neg') + '">' + ((t.pnl_pct||0)*100).toFixed(2) + '%</td>' +
          '<td>' + (t.exit_reason||'') + '</td>' +
          '<td>' + (t.holding_minutes ? t.holding_minutes + 'm' : '\u2014') + '</td>' +
          '</tr>';
      }).join('');
      section.innerHTML = '<h2 style="margin-top:12px;">Recent Trades</h2>' +
        '<table><thead><tr><th>Entry Time</th><th>Exit Time</th><th>Symbol</th><th>Side</th>' +
        '<th>Shares</th><th>Entry</th><th>Exit</th><th>P&L</th><th>P&L %</th>' +
        '<th>Reason</th><th>Hold</th></tr></thead><tbody>' + rows + '</tbody></table>';
    }
  }
}

// ── Guide Timeline ──────────────────────────────────────────────
function updateGuideTimeline() {
  const now = new Date();
  const et = new Date(now.toLocaleString('en-US', {timeZone: 'America/New_York'}));
  const mins = et.getHours() * 60 + et.getMinutes();
  const day = et.getDay();
  const isWeekend = day === 0 || day === 6;

  const phases = [
    {name: 'premarket', start: 420, label: 'Pre-Market Scan'},
    {name: 'refresh',   start: 565, label: 'Final Scan Refresh'},
    {name: 'entry',     start: 571, label: 'Enter Positions'},
    {name: 'monitor',   start: 572, label: 'Monitor & Manage'},
    {name: 'close',     start: 955, label: 'Close All Positions'},
    {name: 'sleep',     start: 960, label: 'Save & Sleep'}
  ];

  let current = null;
  let next = null;
  if (!isWeekend) {
    for (let i = phases.length - 1; i >= 0; i--) {
      if (mins >= phases[i].start) { current = phases[i]; next = phases[i + 1] || null; break; }
    }
    if (!current && mins < 420) next = phases[0];
  }

  document.querySelectorAll('#guideTimeline .tl-step').forEach(el => {
    const phase = el.dataset.phase;
    const dot = el.querySelector('div[style*="border-radius:50%"]');
    if (current && phase === current.name) {
      el.style.background = 'rgba(255,255,255,0.03)';
      el.style.borderRadius = '6px';
      el.style.padding = '8px 8px 8px 28px';
      el.style.marginLeft = '-8px';
      if (dot) dot.style.boxShadow = '0 0 8px currentColor';
    } else {
      el.style.background = '';
      el.style.borderRadius = '';
      el.style.padding = '';
      el.style.paddingLeft = '28px';
      el.style.marginLeft = '';
      if (dot) dot.style.boxShadow = '';
    }
  });

  const evtEl = document.getElementById('guideNextEvent');
  if (isWeekend) {
    evtEl.style.display = 'block';
    evtEl.textContent = 'Market is closed \u2014 trading resumes Monday at 7:00 AM ET';
  } else if (next) {
    evtEl.style.display = 'block';
    const h = Math.floor(next.start / 60), m = next.start % 60;
    const ampm = h >= 12 ? 'PM' : 'AM';
    const h12 = h > 12 ? h - 12 : h;
    evtEl.textContent = 'Next: ' + next.label + ' at ' + h12 + ':' + String(m).padStart(2,'0') + ' ' + ampm + ' ET';
  } else if (current && current.name === 'sleep') {
    evtEl.style.display = 'block';
    evtEl.textContent = 'Trading day complete \u2014 next session tomorrow at 7:00 AM ET';
  } else {
    evtEl.style.display = 'block';
    evtEl.textContent = 'Waiting for market \u2014 pre-market scan starts at 7:00 AM ET';
  }
}

// ── Helpers ──────────────────────────────────────────────────────
function pnlFmt(v) {
  const sign = v >= 0 ? '+' : '-';
  return sign + '$' + Math.abs(v).toFixed(2).replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');
}

const LOG_COLORS = {
  info: 'var(--blue)', scan: 'var(--cyan)', entry: 'var(--red)',
  exit: 'var(--green)', stop: 'var(--orange)', skip: 'var(--muted)',
  warn: 'var(--yellow)', summary: 'var(--purple)', error: 'var(--red)',
};

function appendBtLog(entry) {
  const el = document.getElementById('btLog');
  if (!el) return;
  const color = LOG_COLORS[entry.level] || 'var(--text)';
  const tag = entry.level.toUpperCase().padEnd(7);
  const line = document.createElement('div');
  line.style.cssText = 'margin-bottom:2px;white-space:pre-wrap;word-break:break-all;';
  line.innerHTML = `<span style="color:${color};font-weight:600;">[${tag}]</span> ${escHtml(entry.msg)}`;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent = now.toLocaleTimeString('en-US', { timeZone: 'America/New_York' }) + ' ET';
}

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await checkAuth();
  // If auth is enabled but user not logged in, stop here — don't poll APIs
  if (_authEnabled && !_authUser) {
    setInterval(updateClock, 1000);
    updateClock();
    return;
  }

  // Set default backtest dates
  const end = new Date();
  const start = new Date(end);
  start.setFullYear(start.getFullYear() - 2);
  document.getElementById('btEnd').value = end.toISOString().slice(0,10);
  document.getElementById('btStart').value = start.toISOString().slice(0,10);

  // Toggle custom symbols input visibility
  document.querySelectorAll('input[name="scanUniverse"]').forEach(r => {
    r.addEventListener('change', () => {
      document.getElementById('customSymbolsRow').style.display = r.value === 'custom' ? 'block' : 'none';
    });
  });

  // Set default DB start date (5 years ago)
  const dbStart = new Date(end);
  dbStart.setFullYear(dbStart.getFullYear() - 5);
  document.getElementById('dbStartDate').value = dbStart.toISOString().slice(0,10);

  connectWS();
  fetchState();
  fetchDbStats();
  loadStrategies();
  setInterval(updateClock, 1000);
  setInterval(fetchState, 10000);
  setInterval(updateGuideTimeline, 30000);
  updateClock();
  updateGuideTimeline();
  showPage('dashboard');

  // ── Position Tracker Init ──
  trackerInit();
});

// ==========================================================================
// POSITION TRACKER
// ==========================================================================

const trackerCharts = {};
let trackerWatchlist = [];
const trackerTickBuffers = {};
const TRACKER_MAX = 6;

function trackerInit() {
  try {
    trackerWatchlist = JSON.parse(localStorage.getItem('gf_tracker_wl') || '[]');
  } catch(e) { trackerWatchlist = []; }
  document.getElementById('trackerWatchCount').textContent = trackerWatchlist.length + ' / ' + TRACKER_MAX + ' symbols';
  if (trackerWatchlist.length > 0) {
    trackerWatchlist.forEach(sym => createTrackerPanel(sym));
    setTimeout(() => {
      trackerWatchlist.forEach((sym, i) => setTimeout(() => loadTrackerBars(sym, '1Min'), i * 150));
      updateTrackerWatchlistServer();
    }, 500);
  }
  fetchTrackerPositions();
  setInterval(fetchTrackerPositions, 10000);
}

function saveTrackerWatchlist() {
  localStorage.setItem('gf_tracker_wl', JSON.stringify(trackerWatchlist));
  document.getElementById('trackerWatchCount').textContent = trackerWatchlist.length + ' / ' + TRACKER_MAX + ' symbols';
}

async function addWatchSymbol() {
  const inp = document.getElementById('trackerSymbolInput');
  const sym = inp.value.trim().toUpperCase();
  if (!sym || trackerWatchlist.includes(sym)) { inp.value = ''; return; }
  if (trackerWatchlist.length >= TRACKER_MAX) { showToast('Maximum ' + TRACKER_MAX + ' symbols', 'error'); return; }
  trackerWatchlist.push(sym);
  inp.value = '';
  saveTrackerWatchlist();
  createTrackerPanel(sym);
  await loadTrackerBars(sym, '1Min');
  updateTrackerWatchlistServer();
}

function addWatchSymbolDirect(sym) {
  sym = sym.toUpperCase();
  if (trackerWatchlist.includes(sym)) return;
  if (trackerWatchlist.length >= TRACKER_MAX) return;
  trackerWatchlist.push(sym);
  saveTrackerWatchlist();
  createTrackerPanel(sym);
  loadTrackerBars(sym, '1Min');
  updateTrackerWatchlistServer();
}

function removeWatchSymbol(sym) {
  trackerWatchlist = trackerWatchlist.filter(s => s !== sym);
  saveTrackerWatchlist();
  const tc = trackerCharts[sym];
  if (tc) {
    try { tc.chart.remove(); } catch(e) {}
    try { tc.macdChart.remove(); } catch(e) {}
    delete trackerCharts[sym];
  }
  const panel = document.getElementById('tp-' + sym);
  if (panel) panel.remove();
  delete trackerTickBuffers[sym];
  updateTrackerWatchlistServer();
}

async function updateTrackerWatchlistServer() {
  try { await api('tracker/watchlist', 'POST', { symbols: trackerWatchlist }); } catch(e) {}
}

function createTrackerPanel(sym) {
  const grid = document.getElementById('trackerChartGrid');
  const panel = document.createElement('div');
  panel.className = 'tracker-chart-panel';
  panel.id = 'tp-' + sym;
  panel.innerHTML = `
    <div class="tracker-chart-header">
      <span class="tracker-symbol">${sym}</span>
      <span class="tracker-price" id="tp-price-${sym}"></span>
      <span class="tracker-ohlc" id="tp-ohlc-${sym}">O: -- H: -- L: -- C: --</span>
      <div class="tracker-actions">
        <button class="tracker-buy" onclick="trackerBuy('${sym}')">Buy</button>
        <button class="tracker-short" onclick="trackerShort('${sym}')">Short</button>
        <button class="tracker-remove" onclick="removeWatchSymbol('${sym}')">&times;</button>
      </div>
    </div>
    <div class="tracker-tf-bar">
      <button class="tf-btn active" onclick="changeTrackerTF('${sym}','1Min',this)">1m</button>
      <button class="tf-btn" onclick="changeTrackerTF('${sym}','5Min',this)">5m</button>
      <button class="tf-btn" onclick="changeTrackerTF('${sym}','15Min',this)">15m</button>
      <button class="tf-btn" onclick="changeTrackerTF('${sym}','1Hour',this)">1H</button>
      <button class="tf-btn" onclick="changeTrackerTF('${sym}','1Day',this)">1D</button>
    </div>
    <div id="tp-chart-${sym}" style="height:200px;"></div>
    <div id="tp-macd-${sym}" style="height:60px;border-top:1px solid var(--border);"></div>
  `;
  grid.appendChild(panel);
  requestAnimationFrame(() => initTrackerCharts(sym));
}

function initTrackerCharts(sym) {
  const chartEl = document.getElementById('tp-chart-' + sym);
  const macdEl = document.getElementById('tp-macd-' + sym);
  if (!chartEl || !macdEl || typeof LightweightCharts === 'undefined') return;

  const chart = LightweightCharts.createChart(chartEl, {
    width: chartEl.clientWidth, height: 200,
    layout: { background: { type: 'solid', color: 'transparent' }, textColor: '#64748b', fontSize: 10 },
    grid: { vertLines: { color: 'rgba(0,212,255,0.06)' }, horzLines: { color: 'rgba(0,212,255,0.06)' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: 'rgba(0,212,255,0.15)', scaleMargins: { top: 0.05, bottom: 0.25 } },
    timeScale: { borderColor: 'rgba(0,212,255,0.15)', timeVisible: true, secondsVisible: false },
  });

  const candleSeries = chart.addCandlestickSeries({
    upColor: '#00D4FF', downColor: '#FF3B5C',
    borderUpColor: '#00D4FF', borderDownColor: '#FF3B5C',
    wickUpColor: '#00D4FF', wickDownColor: '#FF3B5C',
  });

  const volumeSeries = chart.addHistogramSeries({
    priceFormat: { type: 'volume' }, priceScaleId: 'vol',
  });
  chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });

  const macdChart = LightweightCharts.createChart(macdEl, {
    width: macdEl.clientWidth, height: 60,
    layout: { background: { type: 'solid', color: 'transparent' }, textColor: '#64748b', fontSize: 9 },
    grid: { vertLines: { color: 'rgba(0,212,255,0.04)' }, horzLines: { color: 'rgba(0,212,255,0.04)' } },
    rightPriceScale: { borderColor: 'rgba(0,212,255,0.1)' },
    timeScale: { visible: false },
  });
  const macdLine = macdChart.addLineSeries({ color: '#00D4FF', lineWidth: 1.5 });
  const signalLine = macdChart.addLineSeries({ color: '#FF3B5C', lineWidth: 1 });
  const histSeries = macdChart.addHistogramSeries({});

  chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
    if (range) macdChart.timeScale().setVisibleLogicalRange(range);
  });

  chart.subscribeCrosshairMove(param => {
    const data = param.seriesData?.get(candleSeries);
    if (data && data.open !== undefined) {
      document.getElementById('tp-ohlc-' + sym).textContent =
        'O:' + data.open.toFixed(2) + ' H:' + data.high.toFixed(2) + ' L:' + data.low.toFixed(2) + ' C:' + data.close.toFixed(2);
    }
  });

  trackerCharts[sym] = { chart, candleSeries, volumeSeries, macdChart, macdLine, signalLine, histSeries, timeframe: '1Min', bars: [] };

  new ResizeObserver(() => {
    chart.applyOptions({ width: chartEl.clientWidth });
    macdChart.applyOptions({ width: macdEl.clientWidth });
  }).observe(chartEl);
}

async function loadTrackerBars(sym, timeframe) {
  const tc = trackerCharts[sym];
  if (!tc) return;
  tc.timeframe = timeframe;
  const limits = { '1Min': 390, '5Min': 390, '15Min': 200, '1Hour': 200, '1Day': 365 };
  const limit = limits[timeframe] || 390;

  let data;
  try { data = await api('tracker/bars?symbol=' + sym + '&timeframe=' + timeframe + '&limit=' + limit); }
  catch(e) { showToast('Failed to load bars for ' + sym, 'error'); return; }

  const bars = data.bars || [];
  if (!bars.length) { showToast('No data for ' + sym, 'error'); return; }
  tc.bars = bars;

  tc.candleSeries.setData(bars.map(b => ({ time: b.t, open: b.o, high: b.h, low: b.l, close: b.c })));
  tc.volumeSeries.setData(bars.map(b => ({
    time: b.t, value: b.v,
    color: b.c >= b.o ? 'rgba(0,212,255,0.3)' : 'rgba(255,59,92,0.3)'
  })));

  const last = bars[bars.length - 1];
  const prev = bars.length > 1 ? bars[bars.length - 2] : last;
  const chg = last.c - prev.c;
  const pctChg = prev.c ? ((chg / prev.c) * 100).toFixed(2) : '0.00';
  const priceEl = document.getElementById('tp-price-' + sym);
  if (priceEl) {
    priceEl.textContent = '$' + last.c.toFixed(2) + ' ' + (chg >= 0 ? '+' : '') + pctChg + '%';
    priceEl.style.color = chg >= 0 ? 'var(--positive)' : 'var(--negative)';
  }

  const closes = bars.map(b => b.c);
  const macd = computeTrackerMACD(closes);
  const times = bars.map(b => b.t);
  tc.macdLine.setData(macd.macd.map((v,i) => ({ time: times[i], value: v })).filter(d => d.value !== null));
  tc.signalLine.setData(macd.signal.map((v,i) => ({ time: times[i], value: v })).filter(d => d.value !== null));
  tc.histSeries.setData(macd.histogram.map((v,i) => ({
    time: times[i], value: v,
    color: v >= 0 ? 'rgba(0,212,255,0.5)' : 'rgba(255,59,92,0.5)'
  })).filter(d => d.value !== null));
}

function computeTrackerMACD(closes, fast=12, slow=26, sig=9) {
  function ema(data, period) {
    const k = 2 / (period + 1);
    const r = new Array(data.length).fill(null);
    let prev = data[0];
    r[0] = prev;
    for (let i = 1; i < data.length; i++) {
      prev = data[i] * k + prev * (1 - k);
      r[i] = i >= period - 1 ? prev : null;
    }
    return r;
  }
  const emaF = ema(closes, fast), emaS = ema(closes, slow);
  const macdArr = closes.map((_, i) => (emaF[i] !== null && emaS[i] !== null) ? emaF[i] - emaS[i] : null);
  const validM = macdArr.filter(v => v !== null);
  const sigEma = ema(validM, sig);
  const signalArr = new Array(closes.length).fill(null);
  let si = 0;
  for (let i = 0; i < closes.length; i++) { if (macdArr[i] !== null) signalArr[i] = sigEma[si++]; }
  const hist = closes.map((_, i) => (macdArr[i] !== null && signalArr[i] !== null) ? macdArr[i] - signalArr[i] : null);
  return { macd: macdArr, signal: signalArr, histogram: hist };
}

function changeTrackerTF(sym, tf, btn) {
  const panel = document.getElementById('tp-' + sym);
  if (panel) panel.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  loadTrackerBars(sym, tf);
}

function handleTrackerTick(sym, price, timestamp) {
  const tc = trackerCharts[sym];
  if (!tc) return;

  const priceEl = document.getElementById('tp-price-' + sym);
  if (priceEl) {
    const prevClose = tc.bars.length > 0 ? tc.bars[tc.bars.length - 1].c : price;
    const chg = price - prevClose;
    const pct = prevClose ? ((chg / prevClose) * 100).toFixed(2) : '0.00';
    priceEl.textContent = '$' + price.toFixed(2) + ' ' + (chg >= 0 ? '+' : '') + pct + '%';
    priceEl.style.color = chg >= 0 ? 'var(--positive)' : 'var(--negative)';
  }

  const tf = tc.timeframe;
  const intervalSec = { '1Min': 60, '5Min': 300, '15Min': 900, '1Hour': 3600, '1Day': 86400 };
  const interval = intervalSec[tf] || 60;
  const candleTime = Math.floor(timestamp / interval) * interval;

  const buf = trackerTickBuffers[sym];
  if (!buf || buf.t !== candleTime) {
    trackerTickBuffers[sym] = { t: candleTime, o: price, h: price, l: price, c: price, v: 0 };
  } else {
    buf.h = Math.max(buf.h, price);
    buf.l = Math.min(buf.l, price);
    buf.c = price;
  }

  const candle = trackerTickBuffers[sym];
  tc.candleSeries.update({ time: candle.t, open: candle.o, high: candle.h, low: candle.l, close: candle.c });

  document.getElementById('tp-ohlc-' + sym).textContent =
    'O:' + candle.o.toFixed(2) + ' H:' + candle.h.toFixed(2) + ' L:' + candle.l.toFixed(2) + ' C:' + candle.c.toFixed(2);
}

async function fetchTrackerPositions() {
  try {
    const data = await api('tracker/positions');
    renderTrackerPositions(data.positions || []);
  } catch(e) {}
}

function renderTrackerPositions(positions) {
  const tbody = document.getElementById('trackerPositionsBody');
  const noPos = document.getElementById('trackerNoPositions');
  if (!positions.length) {
    tbody.innerHTML = '';
    noPos.style.display = 'block';
    return;
  }
  noPos.style.display = 'none';
  tbody.innerHTML = positions.map(p => {
    const pl = parseFloat(p.unrealized_pl || 0);
    const plPct = parseFloat(p.unrealized_plpc || 0) * 100;
    const color = pl >= 0 ? 'var(--positive)' : 'var(--negative)';
    return '<tr>' +
      '<td style="font-weight:600;color:var(--positive);cursor:pointer;" onclick="addWatchSymbolDirect(\\x27' + p.symbol + '\\x27)">' + p.symbol + '</td>' +
      '<td>' + p.qty + '</td>' +
      '<td>$' + parseFloat(p.market_value).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}) + '</td>' +
      '<td>$' + parseFloat(p.current_price).toFixed(2) + '</td>' +
      '<td>$' + parseFloat(p.avg_entry_price).toFixed(2) + '</td>' +
      '<td>$' + parseFloat(p.current_price).toFixed(2) + '</td>' +
      '<td style="color:' + color + ';font-weight:600;">$' + pl.toFixed(2) + ' (' + plPct.toFixed(2) + '%)</td>' +
    '</tr>';
  }).join('');
}

async function trackerBuy(sym) {
  const qty = prompt('Buy how many shares of ' + sym + '?', '10');
  if (!qty) return;
  const q = parseInt(qty);
  if (isNaN(q) || q <= 0) { alert('Invalid quantity'); return; }
  showToast('Submitting buy order...', 'info');
  const r = await api('tracker/order', 'POST', { symbol: sym, side: 'buy', qty: q });
  if (r.error) { showToast('Order failed: ' + r.error, 'error'); return; }
  showToast('Bought ' + (r.filled_qty || q) + ' ' + sym + ' @ $' + (r.filled_avg_price ? r.filled_avg_price.toFixed(2) : '?'), 'success');
  fetchTrackerPositions();
}

async function trackerShort(sym) {
  const qty = prompt('Short how many shares of ' + sym + '?', '10');
  if (!qty) return;
  const q = parseInt(qty);
  if (isNaN(q) || q <= 0) { alert('Invalid quantity'); return; }
  showToast('Submitting short order...', 'info');
  const r = await api('tracker/order', 'POST', { symbol: sym, side: 'sell', qty: q });
  if (r.error) { showToast('Order failed: ' + r.error, 'error'); return; }
  showToast('Shorted ' + (r.filled_qty || q) + ' ' + sym + ' @ $' + (r.filled_avg_price ? r.filled_avg_price.toFixed(2) : '?'), 'success');
  fetchTrackerPositions();
}

</script>
</body>
</html>"""


# =============================================================================
# SECTION 11: MAIN
# =============================================================================

if __name__ == '__main__':
    _load_env_file()
    app_port = int(os.environ.get('GAP_FADE_PORT', '8002'))

    print("=" * 60)
    print(f"  Gap Fade Strategy Dashboard ({APP_VERSION})")
    print(f"  http://localhost:{app_port}")
    print("=" * 60)
    print()
    print("  Strategy: Short gap-ups on below-average volume")
    print("  Edge:     71% fade rate, +2.5% avg P&L (vol < 1x)")
    print("  Data:     3,754 events, 156 stocks, 5 years (Alpaca SIP)")
    print()

    # Check Alpaca credentials
    cfg = _get_alpaca_config()
    if cfg:
        print("  Alpaca:   Configured (paper account)")
    else:
        print("  Alpaca:   NOT configured — set ALPACA_API_KEY + ALPACA_SECRET_KEY")
        print("            (Backtesting still works, live trading requires credentials)")

    # Check Polygon/Massive credentials
    poly_key = _get_polygon_key()
    if poly_key:
        print("  Polygon:  Configured (grouped daily for bulk DB updates)")
    else:
        print("  Polygon:  NOT configured — set POLYGON_API_KEY in .env (optional)")

    # Auth status
    api_key = _get_api_key()
    if api_key:
        print(f"  Auth:     ENABLED (GAP_FADE_API_KEY set, {len(api_key)} chars)")
    else:
        print("  Auth:     DISABLED — set GAP_FADE_API_KEY in .env to protect POST endpoints")

    # Rudra (LLM) status
    if live_trader.config.llm_enabled:
        print(f"  Rudra:    ENABLED ({live_trader.config.llm_model} @ {live_trader.config.llm_url})")
    else:
        print("  Rudra:    DISABLED — set llm_enabled=true via /api/config to activate")

    # P0-6: Bind to localhost by default. Set GAP_FADE_BIND_ALL=1 to listen on all interfaces.
    bind_host = "0.0.0.0" if os.environ.get('GAP_FADE_BIND_ALL', '') == '1' else "127.0.0.1"
    print(f"  Bind:     {bind_host}:{app_port}" +
          (" (use GAP_FADE_BIND_ALL=1 for all interfaces)" if bind_host == "127.0.0.1" else ""))
    print()

    uvicorn.run(app, host=bind_host, port=app_port, log_level="info")
