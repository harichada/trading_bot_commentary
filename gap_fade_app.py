#!/usr/bin/env python3
"""
Gap Fade Trading Strategy — Standalone FastAPI App (port 8002)

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
import sqlite3
import time as _time
import traceback
from collections import defaultdict
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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

logger = logging.getLogger('GapFadeApp')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

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


def alpaca_place_order(symbol: str, qty: int, side: str, order_type: str = 'market',
                       limit_price: float = None, stop_price: float = None,
                       time_in_force: str = 'day') -> dict:
    """Place an order on Alpaca paper account.
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


def alpaca_get_positions() -> List[dict]:
    """Get current positions on Alpaca paper account for crash recovery."""
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
        self.symbols = symbols
        self.on_tick = on_tick
        self.latest_prices: Dict[str, float] = {}
        self.connected = False
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_push: Dict[str, float] = {}

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
        self.connected = False

    async def _run(self, cfg: dict):
        while True:
            try:
                self._session = aiohttp.ClientSession()
                async with self._session.ws_connect(self.WS_URL) as ws:
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
                                    self.latest_prices[sym] = price
                                    if self.on_tick:
                                        now = _time.monotonic()
                                        last = self._last_push.get(sym, 0.0)
                                        if now - last >= self.THROTTLE_SEC:
                                            self._last_push[sym] = now
                                            await self.on_tick(sym, price)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Alpaca stream error: {e}, reconnecting in 3s...")
            finally:
                self.connected = False
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
        self._conn.commit()

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
                      min_avg_volume: int = 5000, vol_window: int = 20) -> List[dict]:
        """Find gap-up days using pure SQL with window functions.

        ~50x faster than loading individual DataFrames.  Uses LAG() for
        prev_close and prev_volume (no look-ahead), and AVG() window for
        rolling average volume.

        Returns list of gap dicts ready for backtester simulation.
        """
        # Fetch extra days before start for rolling avg volume warmup
        adj_start = (datetime.strptime(start, '%Y-%m-%d') - timedelta(days=60)).strftime('%Y-%m-%d')

        all_gaps = []
        chunk_size = 500
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            placeholders = ','.join('?' * len(chunk))

            # Window functions: LAG for prev close/volume, AVG for rolling avg vol
            # All computed on ordered partitions — no look-ahead
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
              AND (open - prev_close) / prev_close >= ?
              {f'AND (open - prev_close) / prev_close <= ?' if max_gap_pct else ''}
              AND open >= ?
              AND avg_vol >= ?
            ORDER BY date, symbol
            """
            params = chunk + [adj_start, end, start, gap_threshold]
            if max_gap_pct:
                params.append(max_gap_pct)
            params.extend([min_price, min_avg_volume])
            cur = self._conn.execute(sql, params)

            for row in cur:
                sym, date_str, bar_open, bar_high, bar_low, bar_close, volume, \
                    prev_close, prev_volume, avg_vol = row
                gap_pct = (bar_open - prev_close) / prev_close
                # vol_ratio from PREVIOUS day's volume (no look-ahead)
                vol_ratio = prev_volume / avg_vol if avg_vol and avg_vol > 0 else 1.0

                all_gaps.append({
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
                })

        return all_gaps


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
    min_price: float = 3.0             # minimum stock price

    # Position sizing
    initial_capital: float = 25_000
    risk_pct: float = 0.02             # risk 2% of equity per trade
    kelly_fraction: float = 0.25       # quarter Kelly
    max_positions: int = 3             # max concurrent shorts (3 = more capital per trade)

    # Stops and targets (optimized: 2.5% stop, 1/3 partial cover)
    stop_pct: float = 0.015            # 1.5% stop loss above entry (walk-forward optimal)
    partial_target_pct: float = 0.50   # cover fraction when price drops to midpoint
    partial_cover_frac: float = 0.33   # fraction of position to cover at partial target (0.33 = 1/3)
    bounce_entry_pct: float = 0.0      # wait for bounce above open before shorting (0 = disabled)
    # Full target = prev_close (full gap fill)

    # Time exits
    time_exit_hour: int = 15           # close remaining by 3:00 PM ET (study edge = full day)
    time_exit_min: int = 0
    eod_exit_hour: int = 15            # force close ALL by 3:50 PM ET
    eod_exit_min: int = 50

    # Circuit breakers
    daily_loss_limit: float = 0.02     # halt if daily P&L <= -2%
    max_consec_losses: int = 3         # pause after N consecutive losses
    max_drawdown: float = 0.05         # halt if drawdown >= 5%

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
    catalyst_skip_earnings: bool = True    # skip earnings gaps entirely
    catalyst_news_penalty: float = 15.0    # score penalty for news-driven gaps
    catalyst_noise_bonus: float = 5.0      # score bonus for clean noise gaps


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


@dataclass
class GapPosition:
    """Active short position."""
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

    def __post_init__(self):
        if self.remaining_shares == 0:
            self.remaining_shares = self.shares


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
    """Check if any symbols had earnings in the last 2 days using yfinance.

    Returns {symbol: 'earnings_YYYY-MM-DD'} for symbols with recent earnings.
    """
    result: Dict[str, str] = {}
    cutoff = datetime.now(timezone.utc) - timedelta(days=2)

    def _check_one(sym: str) -> Optional[Tuple[str, str]]:
        try:
            ticker = yf.Ticker(sym)
            dates = ticker.earnings_dates
            if dates is not None and len(dates) > 0:
                for dt in dates.index:
                    dt_aware = dt.to_pydatetime()
                    if dt_aware.tzinfo is None:
                        dt_aware = dt_aware.replace(tzinfo=timezone.utc)
                    if dt_aware >= cutoff:
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
                if gap_pct < self.config.gap_threshold:
                    continue
                if self.config.max_gap_pct > 0 and gap_pct > self.config.max_gap_pct:
                    continue

                if current_price < self.config.min_price:
                    continue

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

        # Filter by volume ratio
        candidates = [c for c in raw_candidates if c.vol_ratio <= self.config.vol_ratio_max]

        # Check shortability for top candidates
        candidates.sort(key=lambda c: c.gap_pct, reverse=True)
        for c in candidates[:20]:
            shortable, etb = alpaca_check_shortable(c.symbol)
            c.shortable = shortable
            c.easy_to_borrow = etb
            _time.sleep(0.15)

        candidates = [c for c in candidates if c.shortable]

        # Score: bigger gap = higher priority (from data: larger gaps fade more)
        for c in candidates:
            c.score = c.gap_pct * 100
            if c.easy_to_borrow:
                c.score += 5
            if c.vol_ratio < 0.5:
                c.score += 10  # very low volume = stronger signal

        candidates.sort(key=lambda c: c.score, reverse=True)
        logger.info(f"Scanner found {len(candidates)} gap-up candidates from {universe_size} symbols")
        return candidates, universe_size

    def scan_historical(self, symbol: str, start_date: str, end_date: str) -> List[dict]:
        """Find historical gap-up days for backtesting (single symbol)."""
        # Fetch extra days for rolling avg volume
        adj_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=45)).strftime('%Y-%m-%d')
        df = fetch_alpaca_bars(symbol, adj_start, end_date, '1Day', 'iex')
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        return self._find_gaps_in_df(df, symbol, after_date=start_dt,
                                     gap_threshold=self.config.gap_threshold,
                                     max_gap_pct=self.config.max_gap_pct if self.config.max_gap_pct > 0 else None,
                                     min_price=self.config.min_price,
                                     min_avg_volume=self.config.min_avg_volume)

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
        all_gaps = db.scan_gaps_sql(
            symbols, start_date, end_date,
            gap_threshold=self.config.gap_threshold,
            max_gap_pct=self.config.max_gap_pct if self.config.max_gap_pct > 0 else None,
            min_price=self.config.min_price,
            min_avg_volume=self.config.min_avg_volume,
        )
        elapsed = _time.time() - t0
        logger.info(f"SQL gap scan: {len(all_gaps)} gaps in {elapsed:.1f}s")
        return all_gaps

    @staticmethod
    def _find_gaps_in_df(df: Optional[pd.DataFrame], symbol: str,
                         after_date: datetime = None, min_bars: int = 5,
                         gap_threshold: float = None, max_gap_pct: float = None,
                         min_price: float = None,
                         min_avg_volume: int = None) -> List[dict]:
        """Extract gap-up days from a DataFrame of daily bars.

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
            if gap_threshold is not None and gap < gap_threshold:
                continue
            if max_gap_pct is not None and gap > max_gap_pct:
                continue
            if min_price is not None and opens[i] < min_price:
                continue

            # Use avg volume computed up to the PREVIOUS day (no look-ahead)
            prev_avg_vol = avg_vol[i-1] if i >= 1 and not np.isnan(avg_vol[i-1]) else (
                avg_vol[i] if not np.isnan(avg_vol[i]) else volumes[i-1])
            if min_avg_volume is not None and prev_avg_vol < min_avg_volume:
                continue

            # vol_ratio uses PREVIOUS day's volume (known at pre-market)
            # NOT the gap day's volume (that's look-ahead bias)
            vol_ratio = volumes[i-1] / prev_avg_vol if prev_avg_vol > 0 else 1.0

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
            })

        return gaps


# =============================================================================
# SECTION 5: STRATEGY ENGINE
# =============================================================================

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
        self.trade_log: List[TradeRecord] = []
        self.all_trade_log: List[TradeRecord] = []   # persists across resets

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

        # Per-position cap: divide equity by max_positions so all slots fit without leverage
        open_count = len(self.positions)
        slots = max(1, self.config.max_positions - open_count)
        per_slot_equity = self.equity / max(1, self.config.max_positions)
        max_shares = int(per_slot_equity / entry_price) if entry_price > 0 else 0
        shares = min(shares, max_shares)

        # Absolute notional cap (also per-slot)
        if entry_price > 0 and self.config.max_notional > 0:
            notional_limit = min(self.config.max_notional, per_slot_equity)
            max_shares_notional = int(notional_limit / entry_price)
            shares = min(shares, max_shares_notional)

        return max(0, shares)

    def should_enter(self, candidate: GapCandidate) -> Tuple[bool, str]:
        """Check if we should enter a new short. Returns (ok, reason)."""
        # Catalyst-driven gaps: never short into earnings or M&A
        if candidate.catalyst in ('earnings', 'ma'):
            return False, f"catalyst-driven gap ({candidate.catalyst})"

        # Already in this symbol?
        if candidate.symbol in self.positions:
            return False, "already in position"

        # Max positions (applies in both modes)
        if len(self.positions) >= self.config.max_positions:
            return False, f"max positions ({self.config.max_positions}) reached"

        # Volume ratio check (applies in both modes)
        if candidate.vol_ratio > self.config.vol_ratio_max:
            return False, f"vol ratio {candidate.vol_ratio:.1f} > {self.config.vol_ratio_max}"

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

            dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
            if dd >= self.config.max_drawdown:
                self.daily_stats.halted = True
                self.daily_stats.halt_reason = f"max drawdown ({dd:.1%})"
                return False, self.daily_stats.halt_reason

        return True, "ok"

    def open_position(self, candidate: GapCandidate, entry_price: float,
                      entry_time: str) -> Optional[GapPosition]:
        """Create a new short position. Applies slippage in backtest mode."""
        # Slippage: short sell gets worse (lower) fill price
        slip = self.config.slippage_pct if self.backtest_mode else 0
        fill_price = entry_price * (1 - slip)

        stop_price = fill_price * (1 + self.config.stop_pct)
        half_target = (fill_price + candidate.prev_close) / 2
        full_target = candidate.prev_close

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
            holding_min = int((current_time - entry_dt).total_seconds() / 60)
        except (ValueError, TypeError):
            holding_min = 0

        # 1. Stop loss — use HIGH for honest stop check
        if high >= pos.stop_price:
            exit_price = pos.stop_price * (1 + slip)
            pnl = (pos.entry_price - exit_price) * pos.remaining_shares
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
            trades.append(TradeRecord(
                symbol=symbol, entry_price=pos.entry_price, exit_price=exit_price,
                shares=pos.remaining_shares, pnl=pnl, pnl_pct=pnl_pct,
                entry_time=pos.entry_time, exit_time=now_str,
                exit_reason='stop', holding_minutes=holding_min,
            ))
            self._record_trade(trades[-1])
            del self.positions[symbol]
            return trades

        # 2. Partial profit — cover fraction when price hits midpoint target
        if not pos.partial_filled and price <= pos.half_target:
            cover_shares = max(1, int(pos.remaining_shares * self.config.partial_cover_frac))
            if cover_shares > 0:
                fill_price = price * (1 + slip)
                pnl = (pos.entry_price - fill_price) * cover_shares
                pnl_pct = (pos.entry_price - fill_price) / pos.entry_price
                trades.append(TradeRecord(
                    symbol=symbol, entry_price=pos.entry_price, exit_price=fill_price,
                    shares=cover_shares, pnl=pnl, pnl_pct=pnl_pct,
                    entry_time=pos.entry_time, exit_time=now_str,
                    exit_reason='partial', holding_minutes=holding_min,
                ))
                self._record_trade(trades[-1])
                pos.remaining_shares -= cover_shares
                pos.partial_filled = True
                # Move stop to breakeven after partial fill
                pos.stop_price = pos.entry_price

        # 3. Full target — cover remaining when price reaches prev_close
        if price <= pos.full_target and pos.remaining_shares > 0:
            fill_price = price * (1 + slip)
            pnl = (pos.entry_price - fill_price) * pos.remaining_shares
            pnl_pct = (pos.entry_price - fill_price) / pos.entry_price
            trades.append(TradeRecord(
                symbol=symbol, entry_price=pos.entry_price, exit_price=fill_price,
                shares=pos.remaining_shares, pnl=pnl, pnl_pct=pnl_pct,
                entry_time=pos.entry_time, exit_time=now_str,
                exit_reason='full_target', holding_minutes=holding_min,
            ))
            self._record_trade(trades[-1])
            del self.positions[symbol]
            return trades

        # 4. Time exit
        et_hour = current_time.hour if current_time.tzinfo else current_time.hour
        et_min = current_time.minute
        if (et_hour > self.config.time_exit_hour or
            (et_hour == self.config.time_exit_hour and et_min >= self.config.time_exit_min)):
            if pos.remaining_shares > 0:
                fill_price = price * (1 + slip)
                pnl = (pos.entry_price - fill_price) * pos.remaining_shares
                pnl_pct = (pos.entry_price - fill_price) / pos.entry_price
                trades.append(TradeRecord(
                    symbol=symbol, entry_price=pos.entry_price, exit_price=fill_price,
                    shares=pos.remaining_shares, pnl=pnl, pnl_pct=pnl_pct,
                    entry_time=pos.entry_time, exit_time=now_str,
                    exit_reason='time_exit', holding_minutes=holding_min,
                ))
                self._record_trade(trades[-1])
                del self.positions[symbol]
                return trades

        # 5. EOD exit
        if (et_hour > self.config.eod_exit_hour or
            (et_hour == self.config.eod_exit_hour and et_min >= self.config.eod_exit_min)):
            if pos.remaining_shares > 0:
                fill_price = price * (1 + slip)
                pnl = (pos.entry_price - fill_price) * pos.remaining_shares
                pnl_pct = (pos.entry_price - fill_price) / pos.entry_price
                trades.append(TradeRecord(
                    symbol=symbol, entry_price=pos.entry_price, exit_price=fill_price,
                    shares=pos.remaining_shares, pnl=pnl, pnl_pct=pnl_pct,
                    entry_time=pos.entry_time, exit_time=now_str,
                    exit_reason='eod', holding_minutes=holding_min,
                ))
                self._record_trade(trades[-1])
                del self.positions[symbol]
                return trades

        return trades

    def force_close_all(self, prices: Dict[str, float], reason: str = 'eod') -> List[TradeRecord]:
        """Force close all positions at current prices."""
        trades = []
        now_str = datetime.now(ET).strftime('%Y-%m-%d %H:%M')
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            price = prices.get(sym, pos.entry_price)
            pnl = (pos.entry_price - price) * pos.remaining_shares
            pnl_pct = (pos.entry_price - price) / pos.entry_price if pos.entry_price > 0 else 0
            trades.append(TradeRecord(
                symbol=sym, entry_price=pos.entry_price, exit_price=price,
                shares=pos.remaining_shares, pnl=pnl, pnl_pct=pnl_pct,
                entry_time=pos.entry_time, exit_time=now_str,
                exit_reason=reason,
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

    def reset_daily(self):
        """Reset daily stats for new trading day."""
        self.daily_stats = DailyStats(
            date=datetime.now(ET).strftime('%Y-%m-%d'),
            peak_equity=self.equity,
        )
        self.trade_log = []

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
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')

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

class GapFadeBacktester:
    """Backtest gap fade strategy using 1-minute Alpaca bars."""

    def __init__(self, config: GapFadeConfig = None):
        self.config = config or GapFadeConfig()
        self.progress = 0.0
        self.status = 'idle'
        self._cancel = False
        self.result = None

    async def run(self, symbol: str = None, symbols: List[str] = None,
                  start_date: str = None, end_date: str = None,
                  config: GapFadeConfig = None,
                  progress_callback=None, **kwargs) -> dict:
        """Run backtest. use_1min=True for detailed 1-min bar simulation (slow)."""
        if config:
            self.config = config

        self.status = 'running'
        self._cancel = False
        self.progress = 0
        self.bt_log: List[dict] = []  # detailed decision log

        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=self.config.backtest_years * 365)).strftime('%Y-%m-%d')

        syms = symbols or ([symbol] if symbol else UNIVERSE)

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

            # Filter by vol_ratio
            all_gap_days = [g for g in all_gap_days if g['vol_ratio'] <= self.config.vol_ratio_max]
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
                # Filter by vol_ratio
                gaps = [g for g in gaps if g['vol_ratio'] <= self.config.vol_ratio_max]
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

        # Sort each day's gaps by gap_pct descending (biggest fade opportunity first)
        for d in gaps_by_date:
            gaps_by_date[d].sort(key=lambda g: g['gap_pct'], reverse=True)

        sorted_dates = sorted(gaps_by_date.keys())
        total_gaps = len(all_gap_days)
        trades_by_day = []
        bars_missing = 0
        gi = 0  # global gap counter for progress

        for date_str in sorted_dates:
            if self._cancel:
                break

            day_gaps = gaps_by_date[date_str]
            # Enforce max_positions per day — only trade top N gaps
            max_pos = self.config.max_positions
            day_gaps = day_gaps[:max_pos]

            # Reset daily stats for each new calendar day
            engine.daily_stats = DailyStats(date=date_str, peak_equity=engine.equity)

            for gap in day_gaps:
                if self._cancel:
                    break

                gi += 1
                self.progress = 30 + (gi / total_gaps) * 65  # 30-95%
                sym = gap['symbol']
                gap_date = gap['date']

                if progress_callback and gi % max(1, total_gaps // 40) == 0:
                    await progress_callback(self.progress, f"Simulating {sym} {gap_date} ({gi+1}/{total_gaps})")

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

        self.progress = 100
        self.status = 'done'

        metrics = engine.get_metrics()
        # Funnel reporting: raw → vol-filtered → data available → traded
        metrics['raw_gaps'] = raw_gap_count
        metrics['gap_days_found'] = len(all_gap_days)
        metrics['bars_missing'] = bars_missing
        metrics['gap_days_traded'] = len([d for d in trades_by_day if d['trades'] > 0])
        metrics['symbols_scanned'] = len(syms)
        metrics['start_date'] = start_date
        metrics['end_date'] = end_date
        metrics['config'] = asdict(self.config)
        metrics['trades'] = [asdict(t) for t in engine.all_trade_log[-200:]]
        metrics['daily_summary'] = trades_by_day[-100:]
        metrics['bt_log'] = self.bt_log[-500:]

        # Realism adjustments applied
        metrics['realism'] = {
            'slippage_pct': self.config.slippage_pct,
            'borrow_rate_annual': self.config.borrow_rate_annual,
            'max_pct_adv': self.config.max_pct_adv,
            'adverse_fill': self.config.adverse_fill,
            'adverse_fill_pct': self.config.adverse_fill_pct,
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

        candidate = GapCandidate(
            symbol=sym, gap_pct=gap['gap_pct'], prev_close=prev_close,
            premarket_price=day_open, avg_vol_20d=gap.get('avg_vol', 0),
            vol_ratio=gap['vol_ratio'], shortable=True, easy_to_borrow=True,
        )

        # Bounce entry: if configured, wait for a small bounce above open before shorting
        # This gives a better entry price but may miss trades that fade immediately
        bounce = self.config.bounce_entry_pct
        if bounce > 0 and day_high >= day_open * (1 + bounce):
            # Bounce happened — enter at bounce level (minus slippage)
            entry_price = day_open * (1 + bounce) * (1 - slip)
        elif bounce > 0:
            # No bounce reached — skip this trade
            log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — no bounce to {bounce:.1%} above open'})
            return day_trades, log
        else:
            # Entry with slippage: short sell gets worse (lower) fill
            entry_price = day_open * (1 - slip)
        entry_time_str = f'{gap["date"]} 09:31'
        # Estimated exit times for daily-bar mode (we don't know exact intrabar timing)
        exit_stop_str = f'{gap["date"]} 10:30'     # stops tend to hit early
        exit_partial_str = f'{gap["date"]} 12:00'   # partial midday
        exit_full_str = f'{gap["date"]} 14:00'      # full target afternoon
        exit_close_str = f'{gap["date"]} 15:55'     # EOD exit
        # Holding minutes estimates (from 09:31)
        hold_stop = 59       # ~1h
        hold_partial = 149   # ~2.5h
        hold_full = 269      # ~4.5h
        hold_close = 384     # ~6.5h

        ok, reason = engine.should_enter(candidate)
        if not ok:
            log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — {reason} '
                       f'(gap {gap["gap_pct"]:.1%}, vol {gap["vol_ratio"]:.2f}x)'})
            return day_trades, log

        stop_price = entry_price * (1 + self.config.stop_pct)
        half_target = (entry_price + prev_close) / 2
        full_target = prev_close

        risk_per_share = stop_price - entry_price
        kelly_risk = engine.compute_kelly_size()
        risk_frac = min(self.config.risk_pct, kelly_risk) if kelly_risk > 0 else self.config.risk_pct
        dollar_risk = engine.equity * risk_frac
        shares = int(dollar_risk / risk_per_share) if risk_per_share > 0 else 0
        # Per-position cap: divide equity evenly across max_positions slots
        per_slot_equity = engine.equity / max(1, self.config.max_positions)
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

        # Borrow fee: annualized rate prorated to 1 day
        notional = shares * entry_price
        borrow_cost = notional * (self.config.borrow_rate_annual / 252)

        stop_dist = (stop_price - entry_price) / entry_price
        target_dist = (entry_price - full_target) / entry_price
        cost_note = f' | borrow ${borrow_cost:.0f}' if borrow_cost > 0.5 else ''
        liq_note = f' | liq-capped' if avg_vol > 0 and shares == int(avg_vol * self.config.max_pct_adv) else ''
        log.append({
            'level': 'entry',
            'msg': (f'{sym} {gap["date"]}: SHORT {shares}sh @ ${entry_price:.2f} (slip {slip:.2%}) '
                    f'| gap {gap["gap_pct"]:.1%} vol {gap["vol_ratio"]:.2f}x '
                    f'| stop ${stop_price:.2f} (+{stop_dist:.1%}) '
                    f'| half ${half_target:.2f} full ${full_target:.2f} (-{target_dist:.1%})'
                    f'{cost_note}{liq_note}'),
            'data': {'symbol': sym, 'shares': shares, 'entry': entry_price,
                     'stop': stop_price, 'gap_pct': gap['gap_pct'], 'vol_ratio': gap['vol_ratio']}
        })

        remaining = shares
        total_pnl = 0.0

        # 1. Stop check — high breaches stop (conservative: check first)
        stopped = day_high >= stop_price
        # 2. Partial target — low reaches midpoint
        partial_hit = day_low <= half_target
        # 3. Full target — close at or below prev_close
        full_hit = day_close <= full_target

        if stopped and partial_hit and self.config.adverse_fill:
            # AMBIGUOUS BAR: both stop and target reachable from daily OHLCV.
            # Use close as heuristic: close < open → stock dropped (target likely first),
            # close > open → stock rallied (stop likely first).
            # Fallback: random with adverse_fill_pct probability of stop-first.
            import random
            if day_close < day_open:
                # Bearish close → more likely target hit first, then bounced to stop
                assume_stop = random.random() < (self.config.adverse_fill_pct * 0.5)
            elif day_close > entry_price:
                # Close above entry → stock held up, likely stop hit first
                assume_stop = random.random() < min(1.0, self.config.adverse_fill_pct * 1.5)
            else:
                # Neutral → use base probability
                assume_stop = random.random() < self.config.adverse_fill_pct

            if assume_stop:
                # Adverse: stop hit first → full loss
                exit_price = stop_price * (1 + slip)
                pnl = (entry_price - exit_price) * remaining - borrow_cost
                pnl_pct = pnl / (entry_price * remaining) if remaining > 0 else 0
                day_trades.append(TradeRecord(
                    symbol=sym, entry_price=entry_price, exit_price=exit_price,
                    shares=remaining, pnl=pnl, pnl_pct=pnl_pct,
                    entry_time=entry_time_str, exit_time=exit_stop_str,
                    exit_reason='stop_adverse', holding_minutes=hold_stop,
                ))
                engine._record_trade(day_trades[-1])
                total_pnl += pnl
                remaining = 0
                log.append({
                    'level': 'stop',
                    'msg': (f'{sym} {gap["date"]}: STOP (adverse) {shares}sh @ ${exit_price:.2f} '
                            f'| P&L ${pnl:.0f} ({pnl_pct:+.1%}) '
                            f'| ambiguous bar — resolved as stop first'),
                    'data': {'pnl': pnl, 'reason': 'stop_adverse'}
                })
            else:
                # Favorable: target hit first → treat as partial_hit path
                # (fall through to the partial_hit branch below)
                stopped = False

        elif stopped:
            # Pure stop-out: entire position lost at stop (+slippage)
            exit_price = stop_price * (1 + slip)
            pnl = (entry_price - exit_price) * remaining - borrow_cost
            pnl_pct = pnl / (entry_price * remaining) if remaining > 0 else 0
            day_trades.append(TradeRecord(
                symbol=sym, entry_price=entry_price, exit_price=exit_price,
                shares=remaining, pnl=pnl, pnl_pct=pnl_pct,
                entry_time=entry_time_str, exit_time=exit_stop_str,
                exit_reason='stop', holding_minutes=hold_stop,
            ))
            engine._record_trade(day_trades[-1])
            total_pnl += pnl
            remaining = 0
            log.append({
                'level': 'stop',
                'msg': (f'{sym} {gap["date"]}: STOP {shares}sh @ ${exit_price:.2f} '
                        f'| P&L ${pnl:.0f} ({pnl_pct:+.1%}) | high ${day_high:.2f}'),
                'data': {'pnl': pnl, 'reason': 'stop'}
            })

        elif partial_hit:
            # Partial fill, no stop hit
            cover_shares = max(1, int(shares * self.config.partial_cover_frac))
            if cover_shares > 0:
                exit_p = half_target * (1 + slip)
                pnl1 = (entry_price - exit_p) * cover_shares - (borrow_cost * cover_shares / shares)
                pnl_pct1 = pnl1 / (entry_price * cover_shares) if cover_shares > 0 else 0
                day_trades.append(TradeRecord(
                    symbol=sym, entry_price=entry_price, exit_price=exit_p,
                    shares=cover_shares, pnl=pnl1, pnl_pct=pnl_pct1,
                    entry_time=entry_time_str, exit_time=exit_partial_str,
                    exit_reason='partial', holding_minutes=hold_partial,
                ))
                engine._record_trade(day_trades[-1])
                total_pnl += pnl1
                remaining -= cover_shares
                log.append({
                    'level': 'exit',
                    'msg': (f'{sym} {gap["date"]}: PARTIAL {cover_shares}sh @ ${exit_p:.2f} '
                            f'| P&L ${pnl1:+.0f} ({pnl_pct1:+.1%})'),
                    'data': {'pnl': pnl1, 'reason': 'partial'}
                })

            # Check if remaining hits full target
            remaining_borrow = borrow_cost * remaining / shares if shares > 0 else 0
            if remaining > 0 and full_hit:
                exit_p = full_target * (1 + slip)
                pnl2 = (entry_price - exit_p) * remaining - remaining_borrow
                pnl_pct2 = pnl2 / (entry_price * remaining) if remaining > 0 else 0
                day_trades.append(TradeRecord(
                    symbol=sym, entry_price=entry_price, exit_price=exit_p,
                    shares=remaining, pnl=pnl2, pnl_pct=pnl_pct2,
                    entry_time=entry_time_str, exit_time=exit_full_str,
                    exit_reason='full_target', holding_minutes=hold_full,
                ))
                engine._record_trade(day_trades[-1])
                total_pnl += pnl2
                remaining = 0
                log.append({
                    'level': 'exit',
                    'msg': (f'{sym} {gap["date"]}: FULL TARGET {day_trades[-1].shares}sh @ ${exit_p:.2f} '
                            f'| P&L ${pnl2:+.0f} ({pnl_pct2:+.1%})'),
                    'data': {'pnl': pnl2, 'reason': 'full_target'}
                })
            elif remaining > 0:
                # Exit at close (time exit)
                exit_p = day_close * (1 + slip)
                pnl2 = (entry_price - exit_p) * remaining - remaining_borrow
                pnl_pct2 = pnl2 / (entry_price * remaining) if remaining > 0 else 0
                day_trades.append(TradeRecord(
                    symbol=sym, entry_price=entry_price, exit_price=exit_p,
                    shares=remaining, pnl=pnl2, pnl_pct=pnl_pct2,
                    entry_time=entry_time_str, exit_time=exit_close_str,
                    exit_reason='time_exit', holding_minutes=hold_close,
                ))
                engine._record_trade(day_trades[-1])
                total_pnl += pnl2
                remaining = 0
                log.append({
                    'level': 'exit' if pnl2 >= 0 else 'stop',
                    'msg': (f'{sym} {gap["date"]}: TIME EXIT {day_trades[-1].shares}sh @ ${exit_p:.2f} '
                            f'| P&L ${pnl2:+.0f} ({pnl_pct2:+.1%})'),
                    'data': {'pnl': pnl2, 'reason': 'time_exit'}
                })

        else:
            # No targets hit, no stop — exit at close
            exit_p = day_close * (1 + slip)
            pnl = (entry_price - exit_p) * remaining - borrow_cost
            pnl_pct = pnl / (entry_price * remaining) if remaining > 0 else 0
            day_trades.append(TradeRecord(
                symbol=sym, entry_price=entry_price, exit_price=exit_p,
                shares=remaining, pnl=pnl, pnl_pct=pnl_pct,
                entry_time=entry_time_str, exit_time=exit_close_str,
                exit_reason='time_exit', holding_minutes=hold_close,
            ))
            engine._record_trade(day_trades[-1])
            total_pnl += pnl
            remaining = 0
            log.append({
                'level': 'exit' if pnl >= 0 else 'stop',
                'msg': (f'{sym} {gap["date"]}: EXIT @ CLOSE ${exit_p:.2f} '
                        f'| P&L ${pnl:+.0f} ({pnl_pct:+.1%}) | H ${day_high:.2f} L ${day_low:.2f}'),
                'data': {'pnl': pnl, 'reason': 'time_exit'}
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
        """Find first bar at/after 9:31 ET."""
        for bi in range(len(times)):
            t = times[bi]
            h = t.hour if hasattr(t, 'hour') else t.to_pydatetime().hour
            m = t.minute if hasattr(t, 'minute') else t.to_pydatetime().minute
            if h == 9 and m >= 31:
                return bi
            elif h > 9:
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

        candidate = GapCandidate(
            symbol=sym, gap_pct=gap['gap_pct'], prev_close=prev_close,
            premarket_price=gap['open'], avg_vol_20d=gap.get('avg_vol', 0),
            vol_ratio=gap['vol_ratio'], shortable=True, easy_to_borrow=True,
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

        stop_dist = (pos.stop_price - entry_price) / entry_price
        target_dist = (entry_price - pos.full_target) / entry_price
        log.append({
            'level': 'entry',
            'msg': (f'{sym} {entry_time_str}: SHORT {pos.shares} shares @ ${entry_price:.2f} '
                    f'| gap {gap["gap_pct"]:.1%} vol {gap["vol_ratio"]:.2f}x '
                    f'| stop ${pos.stop_price:.2f} (+{stop_dist:.1%}) '
                    f'| half ${pos.half_target:.2f} full ${pos.full_target:.2f} (-{target_dist:.1%}) '
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
# SECTION 7: LIVE TRADER
# =============================================================================

class GapFadeLiveTrader:
    """Async live paper trading loop for gap fade strategy."""

    STATE_FILE = 'gap_fade_state.json'

    def __init__(self, config: GapFadeConfig = None):
        self.config = config or GapFadeConfig()
        self.engine = GapFadeEngine(self.config)
        self.scanner = GapScanner(self.config)
        self.streamer: Optional[AlpacaTickStreamer] = None
        self.status = 'stopped'        # stopped, scanning, trading, paused
        self.candidates: List[GapCandidate] = []
        self._task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self.last_scan_time = ''
        self.messages: List[dict] = []  # decision feed

        # Load persisted state
        self._load_state()

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

    async def start(self):
        """Start the live trading loop."""
        if self.status == 'trading':
            return
        self.status = 'scanning'
        self.engine.reset_daily()
        self._task = asyncio.create_task(self._trading_loop())
        logger.info("Live trader started")
        self._add_message('system', 'Live trader started')
        await broadcast({'type': 'live_status', 'status': self.status})

    async def stop(self):
        """Stop the live trader and close all positions."""
        self.status = 'stopped'
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

    async def _trading_loop(self):
        """Main trading loop — runs on schedule."""
        try:
            while self.status != 'stopped':
                now = datetime.now(ET)

                # Pre-market scan (7:00 AM)
                if now.hour == 7 and now.minute == 0:
                    await self._run_scan()
                    await asyncio.sleep(60)
                    continue

                # Re-scan at 9:25 AM
                if now.hour == 9 and now.minute == 25:
                    await self._run_scan()
                    await asyncio.sleep(60)
                    continue

                # Market open — enter positions at 9:31 AM
                if now.hour == 9 and now.minute == 31 and self.status == 'scanning':
                    await self._enter_positions()
                    await asyncio.sleep(60)
                    continue

                # During trading hours — monitor every 15 seconds
                if 9 <= now.hour < 16 and self.status == 'trading':
                    await self._check_positions()
                    await asyncio.sleep(15)
                    continue

                # EOD close at 3:55 PM
                if now.hour == 15 and now.minute >= 55 and self.engine.positions:
                    await self._eod_close()
                    await asyncio.sleep(300)
                    continue

                # After hours — save state and sleep until pre-market
                if now.hour >= 16:
                    self._save_state()
                    self.engine.reset_daily()
                    self.status = 'waiting'
                    await broadcast({'type': 'live_status', 'status': 'waiting'})
                    self._add_message('info', 'After hours — sleeping until 7:00 AM ET')
                    await asyncio.sleep(3600)
                    continue

                # Before pre-market — sleep
                if now.hour < 7:
                    self.status = 'waiting'
                    await broadcast({'type': 'live_status', 'status': 'waiting'})
                    await asyncio.sleep(600)
                    continue

                await asyncio.sleep(30)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Trading loop error: {e}\n{traceback.format_exc()}")
            self._add_message('error', f'Trading loop error: {e}')

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

        self._add_message('scan', f'Found {len(self.candidates)} candidates from {universe_size} symbols', {
            'candidates': [asdict(c) for c in self.candidates[:10]]
        })
        await broadcast({
            'type': 'scan_results',
            'candidates': [asdict(c) for c in self.candidates[:10]],
            'scan_time': self.last_scan_time,
            'universe_size': universe_size,
        })

    def _apply_catalyst_scores(self):
        """Adjust candidate scores based on catalyst classification."""
        filtered = []
        for c in self.candidates:
            if c.catalyst == 'earnings' and self.config.catalyst_skip_earnings:
                c.score = -999
                self._add_message('catalyst',
                    f'{c.symbol}: SKIP — earnings gap ({c.catalyst_detail})')
                continue
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

    async def _enter_positions(self):
        """Enter short positions on top candidates."""
        self.status = 'trading'
        await broadcast({'type': 'live_status', 'status': 'trading'})

        symbols_entered = []
        for candidate in self.candidates[:self.config.max_positions]:
            ok, reason = self.engine.should_enter(candidate)
            if not ok:
                self._add_message('skip', f'Skipping {candidate.symbol}: {reason}')
                continue

            entry_price = candidate.premarket_price
            entry_time = datetime.now(ET).strftime('%Y-%m-%d %H:%M')
            pos = self.engine.open_position(candidate, entry_price, entry_time)
            if pos is None:
                continue

            # Place actual order on Alpaca
            if self.config.limit_orders_only:
                # Limit sell: slightly below current price to get filled quickly
                limit_px = round(entry_price * (1 - self.config.limit_offset_pct), 2)
                result = alpaca_place_order(candidate.symbol, pos.shares, 'sell',
                                           order_type='limit', limit_price=limit_px)
                order_label = f'LIMIT @ ${limit_px:.2f}'
            else:
                result = alpaca_place_order(candidate.symbol, pos.shares, 'sell')
                order_label = 'MARKET'
            if 'error' in result:
                self._add_message('error', f'Order failed for {candidate.symbol}: {result["error"]}')
                del self.engine.positions[candidate.symbol]
                continue

            self._add_message('entry', f'SHORT {pos.shares} {candidate.symbol} {order_label} '
                             f'(gap {candidate.gap_pct:.1%}, score {candidate.score:.0f})')
            symbols_entered.append(candidate.symbol)
            self._add_message('entry', f'SHORT {pos.shares} {candidate.symbol} @ ${entry_price:.2f} '
                            f'(gap {candidate.gap_pct:.1%}, vol {candidate.vol_ratio:.1f}x)', {
                'symbol': candidate.symbol,
                'shares': pos.shares,
                'entry_price': entry_price,
                'stop': pos.stop_price,
                'target': pos.full_target,
            })

        # Start tick streamer for entered symbols
        if symbols_entered:
            self.streamer = AlpacaTickStreamer(symbols_entered, on_tick=self._on_tick)
            await self.streamer.start()

        await broadcast({
            'type': 'positions_update',
            'positions': {s: asdict(p) for s, p in self.engine.positions.items()},
        })
        self._save_state()

    async def _on_tick(self, symbol: str, price: float):
        """Tick callback from Alpaca stream."""
        if self.status != 'trading':
            return
        if symbol not in self.engine.positions:
            return

        now = datetime.now(ET)
        closed = self.engine.check_exits(symbol, price, price, now)
        for trade in closed:
            # Place cover order
            if self.config.limit_orders_only:
                # Limit buy: slightly above current price for quick fill
                limit_px = round(price * (1 + self.config.limit_offset_pct), 2)
                alpaca_place_order(symbol, trade.shares, 'buy',
                                  order_type='limit', limit_price=limit_px)
            else:
                alpaca_place_order(symbol, trade.shares, 'buy')
            self._add_message('exit', f'{trade.exit_reason.upper()} {trade.shares} {symbol} '
                            f'@ ${trade.exit_price:.2f} P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.1%})')

        if closed:
            await broadcast({
                'type': 'trade',
                'trades': [asdict(t) for t in closed],
            })
            await broadcast({
                'type': 'positions_update',
                'positions': {s: asdict(p) for s, p in self.engine.positions.items()},
            })
            self._save_state()

    async def _check_positions(self):
        """Periodic position check (fallback to polling if streamer lags)."""
        if not self.engine.positions:
            return

        now = datetime.now(ET)

        # Use streamer prices if available
        any_closed = False
        if self.streamer and self.streamer.latest_prices:
            for sym in list(self.engine.positions.keys()):
                price = self.streamer.latest_prices.get(sym)
                if price:
                    closed = self.engine.check_exits(sym, price, price, now)
                    for trade in closed:
                        if self.config.limit_orders_only:
                            limit_px = round(price * (1 + self.config.limit_offset_pct), 2)
                            alpaca_place_order(sym, trade.shares, 'buy',
                                              order_type='limit', limit_price=limit_px)
                        else:
                            alpaca_place_order(sym, trade.shares, 'buy')
                        self._add_message('exit', f'{trade.exit_reason.upper()} {sym} '
                                        f'@ ${trade.exit_price:.2f} P&L: ${trade.pnl:.2f}')
                    if closed:
                        any_closed = True
                        await broadcast({
                            'type': 'trade',
                            'trades': [asdict(t) for t in closed],
                        })

        if any_closed:
            self._save_state()

        await broadcast({
            'type': 'positions_update',
            'positions': {s: asdict(p) for s, p in self.engine.positions.items()},
            'stats': asdict(self.engine.daily_stats),
            'equity': self.engine.equity,
        })

    async def _eod_close(self):
        """Force close all positions at end of day."""
        self._add_message('system', 'EOD — closing all positions')

        # Get latest prices
        prices = {}
        if self.streamer:
            prices = dict(self.streamer.latest_prices)

        trades = self.engine.force_close_all(prices, 'eod')
        for trade in trades:
            # EOD: always use market orders to guarantee close before bell
            alpaca_place_order(trade.symbol, trade.shares, 'buy')
            self._add_message('exit', f'EOD close {trade.symbol} @ ${trade.exit_price:.2f} '
                            f'P&L: ${trade.pnl:.2f}')

        if self.streamer:
            await self.streamer.stop()
            self.streamer = None

        await broadcast({
            'type': 'trade',
            'trades': [asdict(t) for t in trades],
        })
        self._save_state()

    def _save_state(self):
        """Persist state to JSON."""
        state = {
            'status': self.status,
            'equity': self.engine.equity,
            'peak_equity': self.engine.peak_equity,
            'positions': {s: asdict(p) for s, p in self.engine.positions.items()},
            'daily_stats': asdict(self.engine.daily_stats),
            'config': asdict(self.config),
            'trade_log': [asdict(t) for t in self.engine.all_trade_log[-500:]],
            'messages': self.messages[-50:],
            'saved_at': datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S'),
        }
        try:
            with open(self.STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"State save failed: {e}")

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

            # Restore positions (crash recovery)
            for sym, pd_dict in state.get('positions', {}).items():
                try:
                    self.engine.positions[sym] = GapPosition(**{
                        k: v for k, v in pd_dict.items() if k in GapPosition.__dataclass_fields__
                    })
                except Exception as e:
                    logger.warning(f"Could not restore position {sym}: {e}")

            # Restore trade log (only if empty to prevent duplication on restarts)
            if not self.engine.all_trade_log:
                for td in state.get('trade_log', []):
                    self.engine.all_trade_log.append(TradeRecord(**{
                        k: v for k, v in td.items() if k in TradeRecord.__dataclass_fields__
                    }))

            logger.info(f"Loaded state: equity=${self.engine.equity:.2f}, "
                       f"{len(self.engine.all_trade_log)} historical trades")
        except Exception as e:
            logger.warning(f"State load failed: {e}")

    def get_state(self) -> dict:
        """Get current state for API response."""
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
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_websockets.remove(ws)


# =============================================================================
# SECTION 9: FASTAPI ENDPOINTS
# =============================================================================

app = FastAPI(title="Gap Fade Strategy", version="1.0")
_app_start_time = _time.time()

# Singletons
live_trader = GapFadeLiveTrader()
backtester = GapFadeBacktester()


async def _watchdog_heartbeat():
    """Ping systemd watchdog every 30s so it knows we're alive."""
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
            sock.sendto(b'WATCHDOG=1', addr)
            await asyncio.sleep(30)
    except Exception as e:
        logger.warning(f"Watchdog heartbeat error: {e}")


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(_watchdog_heartbeat())
    logger.info("Gap Fade app started")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the embedded HTML dashboard."""
    return HTMLResponse(DASHBOARD_HTML)


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

    return {
        'candidates': [asdict(c) for c in candidates[:15]],
        'scan_time': live_trader.last_scan_time,
        'count': len(candidates),
        'universe_size': universe_size,
    }


@app.post("/api/start")
async def start_trading():
    """Start the live trading loop."""
    await live_trader.start()
    return {'status': live_trader.status}


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


@app.post("/api/config")
async def update_config(body: dict):
    """Update strategy configuration."""
    config = live_trader.config
    for key, value in body.items():
        if hasattr(config, key):
            field_type = type(getattr(config, key))
            try:
                setattr(config, key, field_type(value))
            except (ValueError, TypeError):
                pass
    # Sync capital if changed while idle (no open positions)
    old_capital = live_trader.engine.config.initial_capital
    live_trader.engine.config = config
    if config.initial_capital != old_capital and not live_trader.engine.positions:
        live_trader.engine.equity = config.initial_capital
        live_trader.engine.peak_equity = config.initial_capital
    # Rebuild scanner when universe config changes
    live_trader.scanner = GapScanner(config)
    return {'config': asdict(config)}


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

    # Build config from body
    config = GapFadeConfig()
    for key in ['gap_threshold', 'max_gap_pct', 'vol_ratio_max', 'stop_pct', 'risk_pct',
                'kelly_fraction', 'max_positions', 'initial_capital',
                'daily_loss_limit', 'max_consec_losses', 'max_drawdown',
                'time_exit_hour', 'min_avg_volume', 'min_price',
                'max_notional', 'slippage_pct', 'borrow_rate_annual',
                'max_pct_adv', 'adverse_fill', 'adverse_fill_pct',
                'partial_cover_frac', 'bounce_entry_pct',
                'limit_orders_only', 'limit_offset_pct']:
        if key in body:
            field_type = type(getattr(config, key))
            try:
                setattr(config, key, field_type(body[key]))
            except (ValueError, TypeError):
                pass

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

    backtester = GapFadeBacktester(config)
    use_1min = body.get('use_1min', False)
    asyncio.create_task(backtester.run(
        symbol=symbol, symbols=symbols,
        start_date=start_date, end_date=end_date,
        config=config, progress_callback=progress_cb,
        use_1min=use_1min,
    ))
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


@app.get("/api/metrics")
async def get_metrics():
    return live_trader.engine.get_metrics()


@app.get("/api/trades")
async def get_trades():
    return {
        'trades': [asdict(t) for t in live_trader.engine.all_trade_log[-200:]],
        'today': [asdict(t) for t in live_trader.engine.trade_log],
    }


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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live updates."""
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
<title>Gap Fade Strategy</title>
<style>
  :root {
    --bg: #0a0e17;
    --card: #111827;
    --border: #1e293b;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --green: #22c55e;
    --red: #ef4444;
    --blue: #3b82f6;
    --yellow: #eab308;
    --purple: #a855f7;
    --orange: #f97316;
    --cyan: #06b6d4;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
    background: var(--bg);
    color: var(--text);
    font-size: 13px;
    line-height: 1.5;
  }
  .header {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid var(--purple);
  }
  .header h1 { font-size: 20px; color: #c4b5fd; }
  .header .subtitle { color: var(--muted); font-size: 12px; }
  .status-badge {
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
  }
  .status-stopped { background: #374151; color: var(--muted); }
  .status-waiting { background: #1e293b; color: #94a3b8; }
  .status-scanning { background: #1e3a5f; color: var(--blue); }
  .status-trading { background: #14532d; color: var(--green); }
  .status-paused { background: #422006; color: var(--yellow); }
  .status-halted { background: #450a0a; color: var(--red); }

  .main { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 12px; }
  .full-width { grid-column: 1 / -1; }

  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
  }
  .card h2 {
    font-size: 13px;
    color: var(--purple);
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 8px;
  }
  .stat {
    background: rgba(255,255,255,0.03);
    padding: 8px 10px;
    border-radius: 6px;
  }
  .stat .label { font-size: 10px; color: var(--muted); text-transform: uppercase; }
  .stat .value { font-size: 16px; font-weight: 600; margin-top: 2px; }
  .stat .value.green { color: var(--green); }
  .stat .value.red { color: var(--red); }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  th {
    text-align: left;
    padding: 6px 8px;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    font-size: 10px;
    text-transform: uppercase;
  }
  td {
    padding: 6px 8px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  tr:hover { background: rgba(255,255,255,0.03); }

  .controls {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  button {
    padding: 6px 16px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--card);
    color: var(--text);
    cursor: pointer;
    font-size: 12px;
    font-family: inherit;
    transition: all 0.15s;
  }
  button:hover { background: rgba(255,255,255,0.08); }
  button.primary { background: var(--purple); border-color: var(--purple); color: #fff; }
  button.primary:hover { background: #9333ea; }
  button.danger { background: var(--red); border-color: var(--red); color: #fff; }
  button.danger:hover { background: #dc2626; }
  button.success { background: var(--green); border-color: var(--green); color: #fff; }
  button.success:hover { background: #16a34a; }

  .feed {
    max-height: 350px;
    overflow-y: auto;
    font-size: 12px;
  }
  .feed-item {
    padding: 6px 8px;
    border-left: 3px solid var(--border);
    margin-bottom: 4px;
    background: rgba(255,255,255,0.02);
    border-radius: 0 4px 4px 0;
  }
  .feed-item.entry { border-left-color: var(--red); }
  .feed-item.exit { border-left-color: var(--green); }
  .feed-item.scan { border-left-color: var(--blue); }
  .feed-item.system { border-left-color: var(--purple); }
  .feed-item.error { border-left-color: var(--orange); }
  .feed-item .time { color: var(--muted); font-size: 10px; margin-right: 8px; }

  .circuit-breakers {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
  }
  .cb-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 4px;
    background: rgba(255,255,255,0.03);
    font-size: 11px;
  }
  .cb-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }
  .cb-dot.ok { background: var(--green); }
  .cb-dot.warn { background: var(--yellow); }
  .cb-dot.halt { background: var(--red); }

  input, select {
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 4px 8px;
    border-radius: 4px;
    font-family: inherit;
    font-size: 12px;
    width: 80px;
  }
  label { font-size: 11px; color: var(--muted); margin-right: 4px; }

  .config-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 8px;
  }
  .config-item { display: flex; align-items: center; gap: 6px; }

  .pnl-pos { color: var(--green); }
  .pnl-neg { color: var(--red); }

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
    background: var(--purple);
    transition: width 0.3s;
    border-radius: 2px;
  }

  #backtest-panel { display: none; }
  #backtest-panel.active { display: block; }

  .tab-bar {
    display: flex;
    gap: 0;
    border-bottom: 2px solid var(--border);
    margin-bottom: 12px;
  }
  .tab {
    padding: 8px 20px;
    cursor: pointer;
    color: var(--muted);
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: all 0.15s;
  }
  .tab:hover { color: var(--text); }
  .tab.active {
    color: var(--purple);
    border-bottom-color: var(--purple);
  }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  .equity-chart {
    width: 100%;
    height: 200px;
    position: relative;
  }
  .equity-chart canvas {
    width: 100%;
    height: 100%;
  }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>Gap Fade Strategy</h1>
    <div class="subtitle">Short gap-ups on below-average volume | Study: 71% fade rate, +2.5% avg P&L</div>
  </div>
  <div style="display:flex;align-items:center;gap:12px;">
    <span id="statusBadge" class="status-badge status-stopped">STOPPED</span>
    <span id="clock" style="color:var(--muted);font-size:12px;"></span>
  </div>
</div>

<div class="main">

  <!-- Controls -->
  <div class="card full-width">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
      <div class="controls">
        <button class="primary" onclick="runScan()">Scan Now</button>
        <button class="success" onclick="startTrading()">Start Trading</button>
        <button onclick="pauseTrading()">Pause</button>
        <button onclick="resumeTrading()">Resume</button>
        <button class="danger" onclick="stopTrading()">Stop</button>
        <button onclick="resetTrader()">Reset</button>
        <span style="border-left:1px solid var(--border);margin:0 8px;"></span>
        <button onclick="showTab('backtest')" style="background:#1e1b4b;">Run Backtest</button>
      </div>
      <div class="circuit-breakers" id="circuitBreakers">
        <div class="cb-indicator"><div class="cb-dot ok" id="cbDaily"></div>Daily P&L: <span id="cbDailyVal">$0</span></div>
        <div class="cb-indicator"><div class="cb-dot ok" id="cbConsec"></div>Consec Losses: <span id="cbConsecVal">0</span></div>
        <div class="cb-indicator"><div class="cb-dot ok" id="cbDD"></div>Drawdown: <span id="cbDDVal">0%</span></div>
      </div>
    </div>
  </div>

  <!-- Stats -->
  <div class="card full-width">
    <div class="stats-grid">
      <div class="stat"><div class="label">Equity</div><div class="value" id="statEquity">$100,000</div></div>
      <div class="stat"><div class="label">Today P&L</div><div class="value" id="statTodayPnl">$0.00</div></div>
      <div class="stat"><div class="label">Total P&L</div><div class="value" id="statTotalPnl">$0.00</div></div>
      <div class="stat"><div class="label">Win Rate</div><div class="value" id="statWinRate">0%</div></div>
      <div class="stat"><div class="label">Trades</div><div class="value" id="statTrades">0</div></div>
      <div class="stat"><div class="label">Profit Factor</div><div class="value" id="statPF">0</div></div>
      <div class="stat"><div class="label">Max Drawdown</div><div class="value" id="statMaxDD">0%</div></div>
      <div class="stat"><div class="label">Avg Hold (min)</div><div class="value" id="statAvgHold">0</div></div>
    </div>
  </div>

  <!-- Tabs: Live / Backtest -->
  <div class="card full-width">
    <div class="tab-bar">
      <div class="tab active" data-tab="live" onclick="showTab('live')">Live Trading</div>
      <div class="tab" data-tab="candidates" onclick="showTab('candidates')">Scanner</div>
      <div class="tab" data-tab="trades" onclick="showTab('trades')">Trade Log</div>
      <div class="tab" data-tab="backtest" onclick="showTab('backtest')">Backtest</div>
      <div class="tab" data-tab="config" onclick="showTab('config')">Config</div>
      <div class="tab" data-tab="guide" onclick="showTab('guide')">Guide</div>
    </div>

    <!-- Live tab -->
    <div class="tab-content active" id="tab-live">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
        <!-- Positions -->
        <div>
          <h2>Active Positions</h2>
          <table>
            <thead><tr>
              <th>Symbol</th><th>Shares</th><th>Entry</th><th>Current</th><th>Stop</th><th>Target</th><th>P&L</th>
            </tr></thead>
            <tbody id="positionsTable"></tbody>
          </table>
          <div id="noPositions" style="color:var(--muted);padding:20px;text-align:center;">No active positions</div>
        </div>
        <!-- Decision Feed -->
        <div>
          <h2>Decision Feed</h2>
          <div class="feed" id="feedContainer"></div>
        </div>
      </div>
    </div>

    <!-- Candidates tab -->
    <div class="tab-content" id="tab-candidates">
      <h2>Gap-Up Candidates <span style="color:var(--muted);font-size:11px;" id="scanTime"></span> <span style="color:var(--muted);font-size:11px;" id="universeInfo"></span></h2>
      <table>
        <thead><tr>
          <th>Symbol</th><th>Gap %</th><th>Prev Close</th><th>Current</th><th>Vol Ratio</th>
          <th>Avg Vol</th><th>Shortable</th><th>ETB</th><th>Catalyst</th><th>Score</th>
        </tr></thead>
        <tbody id="candidatesTable"></tbody>
      </table>
      <div id="noCandidates" style="color:var(--muted);padding:20px;text-align:center;">No candidates — run a scan</div>
    </div>

    <!-- Trade log tab -->
    <div class="tab-content" id="tab-trades">
      <h2>Trade History</h2>
      <table>
        <thead><tr>
          <th>Time</th><th>Symbol</th><th>Side</th><th>Shares</th><th>Entry</th><th>Exit</th>
          <th>P&L</th><th>P&L %</th><th>Reason</th><th>Hold</th>
        </tr></thead>
        <tbody id="tradesTable"></tbody>
      </table>
    </div>

    <!-- Backtest tab -->
    <div class="tab-content" id="tab-backtest">
      <h2>Historical Backtest</h2>
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
          <label>Start:</label>
          <input type="date" id="btStart" style="width:130px;">
        </div>
        <div class="config-item">
          <label>End:</label>
          <input type="date" id="btEnd" style="width:130px;">
        </div>
        <div class="config-item">
          <label>Gap %:</label>
          <input type="number" id="btGap" value="7" step="1" style="width:60px;" title="Min gap % to consider">
        </div>
        <div class="config-item">
          <label>Max Gap %:</label>
          <input type="number" id="btMaxGap" value="50" step="1" style="width:60px;" title="Filter out mega-gaps above this % (M&A, biotech catalysts)">
        </div>
        <div class="config-item">
          <label>Vol Max:</label>
          <input type="number" id="btVol" value="3.0" step="0.1" style="width:60px;" title="Max volume ratio vs 20d avg">
        </div>
        <div class="config-item">
          <label>Stop %:</label>
          <input type="number" id="btStop" value="1.5" step="0.5" style="width:60px;" title="Stop loss above entry">
        </div>
        <div class="config-item">
          <label>Max Pos:</label>
          <input type="number" id="btMaxPos" value="3" step="1" style="width:50px;" title="Max simultaneous positions">
        </div>
        <div class="config-item">
          <label>Slip %:</label>
          <input type="number" id="btSlip" value="0.05" step="0.01" style="width:60px;" title="Slippage per side (0.05=limit orders, 0.15=market orders)">
        </div>
        <div class="config-item">
          <label>Cover frac:</label>
          <input type="number" id="btCoverFrac" value="0.33" step="0.01" style="width:60px;" title="Fraction to cover at partial target (0=none, 0.33=1/3, 0.5=half)">
        </div>
        <label style="display:flex;align-items:center;gap:4px;cursor:pointer;">
          <input type="checkbox" id="bt1min"> 1-min bars (slow, detailed)
        </label>
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
    </div>

    <!-- Config tab -->
    <div class="tab-content" id="tab-config">
      <h2>Strategy Configuration</h2>
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
      <!-- Price Database Section -->
      <div style="margin-bottom:16px;padding:12px;background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.2);border-radius:8px;">
        <div style="font-size:12px;color:var(--cyan);font-weight:600;margin-bottom:8px;text-transform:uppercase;">Price Database (Local Cache)</div>
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
      <div class="config-grid" id="configGrid"></div>
      <div style="margin-top:12px;">
        <button class="primary" onclick="saveConfig()">Save Config</button>
      </div>
    </div>

    <!-- Guide tab -->
    <div class="tab-content" id="tab-guide">
      <h2>How the Bot Works</h2>

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
              <p style="color:var(--muted);margin:2px 0 0 0;font-size:12px;line-height:1.5;">Go to the <strong>Config</strong> tab and enter your Alpaca API credentials. The bot needs these to place real trades. Use a paper-trading key first to practice.</p>
            </div>
          </div>
          <div style="display:flex;align-items:flex-start;gap:12px;">
            <div style="min-width:28px;height:28px;display:flex;align-items:center;justify-content:center;background:var(--blue);border-radius:50%;color:#fff;font-weight:700;font-size:13px;">2</div>
            <div>
              <span style="color:var(--text);font-weight:600;font-size:13px;">Click Start Trading</span>
              <p style="color:var(--muted);margin:2px 0 0 0;font-size:12px;line-height:1.5;">On the <strong>Live Trading</strong> tab, hit the green Start button. The bot will wait for the right time and begin scanning automatically.</p>
            </div>
          </div>
          <div style="display:flex;align-items:flex-start;gap:12px;">
            <div style="min-width:28px;height:28px;display:flex;align-items:center;justify-content:center;background:var(--blue);border-radius:50%;color:#fff;font-weight:700;font-size:13px;">3</div>
            <div>
              <span style="color:var(--text);font-weight:600;font-size:13px;">Watch the Dashboard</span>
              <p style="color:var(--muted);margin:2px 0 0 0;font-size:12px;line-height:1.5;">The Live Trading tab shows real-time status, positions, and activity logs. The <strong>Scanner</strong> tab shows today's candidates. Everything updates automatically.</p>
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
  </div>

</div>

<script>
// ── State ────────────────────────────────────────────────────────
let ws = null;
let state = {};
let activeTab = 'live';

// ── WebSocket ────────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen = () => console.log('WS connected');
  ws.onclose = () => { setTimeout(connectWS, 2000); };
  ws.onmessage = (e) => {
    if (e.data === 'pong') return;
    try {
      const msg = JSON.parse(e.data);
      handleMessage(msg);
    } catch(err) {}
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
    document.getElementById('btProgress').style.display = 'none';
    renderBacktestResults(msg.result);
  } else if (msg.type === 'bt_log') {
    appendBtLog(msg.entry);
  } else if (msg.type === 'db_build_progress') {
    document.getElementById('dbProgress').style.display = 'block';
    document.getElementById('dbProgressMsg').textContent = msg.message || 'Working...';
    document.getElementById('dbProgressBar').style.width = msg.progress + '%';
    if (msg.progress >= 100) {
      setTimeout(() => { document.getElementById('dbProgress').style.display = 'none'; fetchDbStats(); }, 2000);
    }
  }
}

// ── API calls ────────────────────────────────────────────────────
async function api(path, method='GET', body=null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
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

async function runBacktest() {
  const syms = document.getElementById('btSymbol').value.trim();
  const body = {
    start_date: document.getElementById('btStart').value,
    end_date: document.getElementById('btEnd').value,
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
    }
    // Results will arrive via WebSocket 'backtest_complete' message
  } catch(e) {
    document.getElementById('btProgressMsg').textContent = 'Error: ' + e.message;
  }
}

async function cancelBacktest() { await api('backtest/cancel', 'POST'); }

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
      el.textContent = 'Empty — click Build DB to populate';
    }
  } catch(e) {
    document.getElementById('dbStatus').textContent = 'Error loading stats';
  }
}

async function buildDB() {
  const universe = document.getElementById('dbUniverse').value;
  const startDate = document.getElementById('dbStartDate').value;
  const body = { universe };
  if (startDate) body.start_date = startDate;
  document.getElementById('dbProgress').style.display = 'block';
  document.getElementById('dbProgressMsg').textContent = 'Starting build...';
  document.getElementById('dbProgressBar').style.width = '0%';
  try {
    await api('db/build', 'POST', body);
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
  inputs.forEach(inp => {
    const key = inp.dataset.key;
    if (!key) return;
    if (inp.type === 'checkbox') {
      body[key] = inp.checked;
    } else {
      body[key] = inp.type === 'number' ? parseFloat(inp.value) : inp.value;
    }
  });
  // Include scan universe settings
  const univRadio = document.querySelector('input[name="scanUniverse"]:checked');
  if (univRadio) body.scan_universe = univRadio.value;
  body.custom_symbols = document.getElementById('customSymbolsInput').value.trim();
  await api('config', 'POST', body);
  fetchState();
}

// ── Rendering ────────────────────────────────────────────────────
function renderState(s) {
  updateStatus(s.status);
  updateEquity(s.equity);

  const m = s.metrics || {};
  const ds = s.daily_stats || {};

  const todayPnl = ds.pnl || 0;
  const el = (id, val) => { const e = document.getElementById(id); if(e) e.textContent = val; };
  const cls = (id, c) => { const e = document.getElementById(id); if(e) { e.className = 'value ' + c; } };

  el('statEquity', '$' + (s.equity||0).toLocaleString(undefined,{minimumFractionDigits:0}));
  el('statTodayPnl', pnlFmt(todayPnl));
  cls('statTodayPnl', todayPnl >= 0 ? 'green' : 'red');
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
  renderConfig(s.config || {});
}

function updateStatus(status) {
  const badge = document.getElementById('statusBadge');
  badge.textContent = status.toUpperCase();
  badge.className = 'status-badge status-' + status;
}

function updateEquity(eq) {
  document.getElementById('statEquity').textContent = '$' + (eq||0).toLocaleString(undefined,{minimumFractionDigits:0});
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
    const unrealized = ((p.entry_price - (p.entry_price * 0.99)) * p.remaining_shares); // placeholder
    return `<tr>
      <td style="font-weight:600;color:var(--red);">${sym}</td>
      <td>${p.remaining_shares}</td>
      <td>$${p.entry_price.toFixed(2)}</td>
      <td>—</td>
      <td style="color:var(--red);">$${p.stop_price.toFixed(2)}</td>
      <td style="color:var(--green);">$${p.full_target.toFixed(2)}</td>
      <td>—</td>
    </tr>`;
  }).join('');
}

function catalystBadge(cat, detail) {
  const labels = {earnings:'EARN',fda:'FDA',ma:'M&A',offering:'OFFER',upgrade:'UPG',downgrade:'DNG'};
  const colors = {
    earnings:'var(--red)',fda:'var(--red)',ma:'var(--red)',
    offering:'var(--green)',upgrade:'var(--orange)',downgrade:'var(--orange)',
  };
  if (!cat) return '<span style="color:var(--green);font-weight:600;" title="No catalyst — clean noise gap">NOISE</span>';
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

  tbody.innerHTML = candidates.map(c => `<tr>
    <td style="font-weight:600;">${c.symbol}</td>
    <td style="color:var(--orange);">${(c.gap_pct*100).toFixed(1)}%</td>
    <td>$${c.prev_close.toFixed(2)}</td>
    <td>$${c.premarket_price.toFixed(2)}</td>
    <td style="color:${c.vol_ratio < 0.5 ? 'var(--green)' : (c.vol_ratio < 1 ? 'var(--cyan)' : 'var(--red)')};">
      ${c.vol_ratio.toFixed(2)}x</td>
    <td>${(c.avg_vol_20d/1000).toFixed(0)}K</td>
    <td>${c.shortable ? '✓' : '✗'}</td>
    <td>${c.easy_to_borrow ? '✓' : '—'}</td>
    <td>${catalystBadge(c.catalyst || '', c.catalyst_detail || '')}</td>
    <td>${c.score.toFixed(1)}</td>
  </tr>`).join('');
}

function renderMessages(messages) {
  const feed = document.getElementById('feedContainer');
  feed.innerHTML = messages.slice().reverse().map(m => `
    <div class="feed-item ${m.type}">
      <span class="time">${m.time}</span>${m.text}
    </div>
  `).join('');
}

function renderTrades(todayTrades, totalCount) {
  const tbody = document.getElementById('tradesTable');
  const trades = todayTrades || [];
  tbody.innerHTML = trades.slice().reverse().map(t => `<tr>
    <td>${t.exit_time || ''}</td>
    <td style="font-weight:600;">${t.symbol}</td>
    <td style="color:var(--red);">SHORT</td>
    <td>${t.shares}</td>
    <td>$${t.entry_price.toFixed(2)}</td>
    <td>$${t.exit_price.toFixed(2)}</td>
    <td class="${t.pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">${pnlFmt(t.pnl)}</td>
    <td class="${t.pnl_pct >= 0 ? 'pnl-pos' : 'pnl-neg'}">${(t.pnl_pct*100).toFixed(2)}%</td>
    <td>${t.exit_reason}</td>
    <td>${t.holding_minutes || 0}m</td>
  </tr>`).join('');
}

function renderConfig(config) {
  const grid = document.getElementById('configGrid');
  const fields = [
    ['gap_threshold', 'Gap Threshold', 'number'],
    ['max_gap_pct', 'Max Gap % (0=no cap)', 'number'],
    ['vol_ratio_max', 'Vol Ratio Max', 'number'],
    ['stop_pct', 'Stop %', 'number'],
    ['risk_pct', 'Risk %', 'number'],
    ['kelly_fraction', 'Kelly Fraction', 'number'],
    ['max_positions', 'Max Positions', 'number'],
    ['initial_capital', 'Initial Capital', 'number'],
    ['daily_loss_limit', 'Daily Loss Limit', 'number'],
    ['max_consec_losses', 'Max Consec Losses', 'number'],
    ['max_drawdown', 'Max Drawdown', 'number'],
    ['time_exit_hour', 'Time Exit Hour', 'number'],
    ['min_avg_volume', 'Min Avg Volume', 'number'],
    ['min_price', 'Min Price', 'number'],
    ['max_notional', 'Max Notional $', 'number'],
    ['slippage_pct', 'Slippage % (per side)', 'number'],
    ['borrow_rate_annual', 'Borrow Rate (annual)', 'number'],
    ['max_pct_adv', 'Max % of ADV', 'number'],
    ['limit_offset_pct', 'Limit Offset %', 'number'],
  ];
  grid.innerHTML = `
    <div class="config-item" style="grid-column: span 2; display:flex; align-items:center; gap:8px;">
      <label style="font-weight:600;">Limit Orders Only:</label>
      <input type="checkbox" id="cfgLimitOrders" data-key="limit_orders_only"
             ${config.limit_orders_only ? 'checked' : ''}
             style="width:18px;height:18px;">
      <span style="font-size:11px;color:var(--muted);">Less slippage, better fills. Disable for guaranteed execution.</span>
    </div>
  ` + fields.map(([key, label, type]) => `
    <div class="config-item">
      <label>${label}:</label>
      <input type="${type}" data-key="${key}" value="${config[key] !== undefined ? config[key] : ''}" step="any">
    </div>
  `).join('');

  // Set scan universe radio buttons from config
  const univ = config.scan_universe || 'study';
  const radioMap = { alpaca: 'univAlpaca', study: 'univStudy', custom: 'univCustom' };
  const radioEl = document.getElementById(radioMap[univ] || 'univStudy');
  if (radioEl) radioEl.checked = true;
  document.getElementById('customSymbolsInput').value = config.custom_symbols || '';
  document.getElementById('customSymbolsRow').style.display = univ === 'custom' ? 'block' : 'none';
}

function renderBacktestResults(r) {
  if (r.error) {
    document.getElementById('btResults').innerHTML = `<div style="color:var(--red);">${r.error}</div>`;
    return;
  }

  const warnings = r.warnings || [];
  const realism = r.realism || {};
  const html = `
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
    ${r.trades && r.trades.length > 0 ? `
    <h2 style="margin-top:12px;">Recent Trades</h2>
    <table>
      <thead><tr><th>Entry Time</th><th>Exit Time</th><th>Symbol</th><th>Shares</th><th>Entry</th><th>Exit</th><th>P&L</th><th>P&L %</th><th>Reason</th><th>Hold</th></tr></thead>
      <tbody>${r.trades.slice(-50).reverse().map(t => `<tr>
        <td>${t.entry_time||''}</td>
        <td>${t.exit_time||''}</td>
        <td style="font-weight:600;">${t.symbol}</td>
        <td>${t.shares||0}</td>
        <td>$${(t.entry_price||0).toFixed(2)}</td>
        <td>$${(t.exit_price||0).toFixed(2)}</td>
        <td class="${(t.pnl||0)>=0?'pnl-pos':'pnl-neg'}">${pnlFmt(t.pnl||0)}</td>
        <td class="${(t.pnl_pct||0)>=0?'pnl-pos':'pnl-neg'}">${((t.pnl_pct||0)*100).toFixed(2)}%</td>
        <td>${t.exit_reason||''}</td>
        <td>${t.holding_minutes ? t.holding_minutes + 'm' : '—'}</td>
      </tr>`).join('')}</tbody>
    </table>` : ''}
  `;
  document.getElementById('btResults').innerHTML = html;
}

// ── Tabs ─────────────────────────────────────────────────────────
function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  // Highlight matching tab button by data-tab attribute
  const tabBtn = document.querySelector(`.tab[data-tab="${name}"]`);
  if (tabBtn) tabBtn.classList.add('active');
  activeTab = name;
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
    evtEl.textContent = 'Market is closed — trading resumes Monday at 7:00 AM ET';
  } else if (next) {
    evtEl.style.display = 'block';
    const h = Math.floor(next.start / 60), m = next.start % 60;
    const ampm = h >= 12 ? 'PM' : 'AM';
    const h12 = h > 12 ? h - 12 : h;
    evtEl.textContent = 'Next: ' + next.label + ' at ' + h12 + ':' + String(m).padStart(2,'0') + ' ' + ampm + ' ET';
  } else if (current && current.name === 'sleep') {
    evtEl.style.display = 'block';
    evtEl.textContent = 'Trading day complete — next session tomorrow at 7:00 AM ET';
  } else {
    evtEl.style.display = 'block';
    evtEl.textContent = 'Waiting for market — pre-market scan starts at 7:00 AM ET';
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
document.addEventListener('DOMContentLoaded', () => {
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
  setInterval(updateClock, 1000);
  setInterval(fetchState, 10000);
  setInterval(updateGuideTimeline, 30000);
  updateClock();
  updateGuideTimeline();
});
</script>
</body>
</html>"""


# =============================================================================
# SECTION 11: MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Gap Fade Strategy Dashboard")
    print("  http://localhost:8002")
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
    print()

    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
