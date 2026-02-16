#!/usr/bin/env python3
"""
Claude AI Backtest Dashboard
Standalone FastAPI server on port 8001 for running backtests
with the ClaudeStrategy and visualizing results.
"""

import asyncio
import json
import logging
import os
import traceback
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import requests as _requests
except ImportError:
    _requests = None

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from schwab.auth import easy_client
    from schwab.client import Client as SchwabClient
    from schwab.streaming import StreamClient as SchwabStreamClient
    SCHWAB_AVAILABLE = True
except ImportError:
    SCHWAB_AVAILABLE = False
    SchwabStreamClient = None

from dataclasses import dataclass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from claude_strategy import ClaudeStrategy
from strategy_system import StrategyConfig, StrategySignal, SignalType

# TradingBrain for autonomous learning
try:
    from trading_bot_commentary_updated import TradingBrain
    BRAIN_AVAILABLE = True
except Exception:
    BRAIN_AVAILABLE = False

# Telegram alerts from observability module
try:
    from observability import TelegramChannel, Alert, AlertSeverity
    TELEGRAM_IMPORTS_OK = True
except Exception:
    TELEGRAM_IMPORTS_OK = False

import aiohttp as _aiohttp  # used by Telegram + news fetcher

# News/sentiment dependencies
try:
    import feedparser as _feedparser
    FEEDPARSER_OK = True
except ImportError:
    _feedparser = None
    FEEDPARSER_OK = False

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as _VADER
    _vader = _VADER()
    VADER_OK = True
except Exception:
    _vader = None
    VADER_OK = False

logger = logging.getLogger('ClaudeBacktestApp')
logging.basicConfig(level=logging.DEBUG)

# =============================================================================
# SCHWAB DATA PROVIDER
# =============================================================================

# Singleton Schwab client — initialized once, reused
_schwab_client = None

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
        logger.info(f"Loaded credentials from {env_path}")
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
    }

def _alpaca_symbol(symbol: str) -> Optional[str]:
    """Map internal symbol to Alpaca format. Returns None for unsupported (crypto, forex)."""
    if symbol.endswith('-USD') or '/' in symbol:
        return None  # crypto/forex: skip
    return symbol  # equities pass through

def _alpaca_place_order(symbol: str, qty: int, side: str) -> dict:
    """Place a market order on Alpaca paper account. Returns order dict or error."""
    config = _get_alpaca_config()
    if config is None:
        return {'error': 'Alpaca not configured'}
    headers = {
        'APCA-API-KEY-ID': config['api_key'],
        'APCA-API-SECRET-KEY': config['secret_key'],
        'Content-Type': 'application/json',
    }
    payload = {
        'symbol': symbol,
        'qty': str(qty),
        'side': side,       # 'buy' or 'sell'
        'type': 'market',
        'time_in_force': 'day',
    }
    try:
        resp = _requests.post(
            f"{config['base_url']}/v2/orders",
            headers=headers, json=payload, timeout=10
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            logger.info(f"Alpaca order placed: {side} {qty} {symbol} -> {data.get('id','?')}")
            return {'id': data.get('id'), 'status': data.get('status'), 'symbol': symbol,
                    'qty': qty, 'side': side}
        else:
            err = resp.text[:200]
            logger.warning(f"Alpaca order failed ({resp.status_code}): {err}")
            return {'error': f"HTTP {resp.status_code}: {err}"}
    except Exception as e:
        logger.warning(f"Alpaca order error: {e}")
        return {'error': str(e)}


def get_schwab_client():
    """Get or create Schwab client singleton."""
    global _schwab_client
    if _schwab_client is not None:
        return _schwab_client

    if not SCHWAB_AVAILABLE:
        logger.warning("schwab-py not installed. Install with: pip install schwab-py")
        return None

    # Try loading from .env file if env vars are missing
    _load_env_file()

    api_key = os.environ.get('SCHWAB_API_KEY', '')
    app_secret = (os.environ.get('SCHWAB_SECRET', '')
                  or os.environ.get('SCHWAB_APP_SECRET', ''))
    token_path = 'token_1.json'

    if not api_key or not app_secret:
        logger.warning("Schwab credentials missing. Either:\n"
                       "  1. export SCHWAB_API_KEY=... && export SCHWAB_SECRET=...\n"
                       "  2. Create a .env file (run setup_wizard.py)\n"
                       "Falling back to yfinance.")
        return None

    if not os.path.exists(token_path):
        logger.warning(f"Token file {token_path} not found — run setup_wizard.py first")
        return None

    # Try client_from_token_file first (reuses existing valid token)
    try:
        from schwab import auth
        _schwab_client = auth.client_from_token_file(
            token_path=token_path,
            api_key=api_key,
            app_secret=app_secret,
        )
        logger.info("Schwab client initialized (from token file)")
        return _schwab_client
    except Exception as e:
        logger.info(f"Token file auth failed ({e}), trying easy_client...")

    # Fall back to easy_client (can refresh expired token)
    try:
        _schwab_client = easy_client(
            token_path=token_path,
            api_key=api_key,
            app_secret=app_secret,
            callback_url='https://127.0.0.1:8182'
        )
        logger.info("Schwab client initialized (via easy_client)")
        return _schwab_client
    except Exception as e:
        logger.error(f"Failed to initialize Schwab client: {e}")
        return None


# Map UI interval strings to Schwab API enums
SCHWAB_INTERVAL_MAP = {
    '1m':  (SchwabClient.PriceHistory.FrequencyType.MINUTE, SchwabClient.PriceHistory.Frequency.EVERY_MINUTE) if SCHWAB_AVAILABLE else None,
    '5m':  (SchwabClient.PriceHistory.FrequencyType.MINUTE, SchwabClient.PriceHistory.Frequency.EVERY_FIVE_MINUTES) if SCHWAB_AVAILABLE else None,
    '10m': (SchwabClient.PriceHistory.FrequencyType.MINUTE, SchwabClient.PriceHistory.Frequency.EVERY_TEN_MINUTES) if SCHWAB_AVAILABLE else None,
    '15m': (SchwabClient.PriceHistory.FrequencyType.MINUTE, SchwabClient.PriceHistory.Frequency.EVERY_FIFTEEN_MINUTES) if SCHWAB_AVAILABLE else None,
    '30m': (SchwabClient.PriceHistory.FrequencyType.MINUTE, SchwabClient.PriceHistory.Frequency.EVERY_THIRTY_MINUTES) if SCHWAB_AVAILABLE else None,
    '1d':  (SchwabClient.PriceHistory.FrequencyType.DAILY, SchwabClient.PriceHistory.Frequency.EVERY_MINUTE) if SCHWAB_AVAILABLE else None,  # frequency ignored for daily
}


def fetch_schwab_data(symbol: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
    """Fetch historical OHLCV data from Schwab API.

    Returns DataFrame with lowercase columns (open, high, low, close, volume)
    and datetime index, or None on failure.
    """
    client = get_schwab_client()
    if client is None:
        return None

    mapping = SCHWAB_INTERVAL_MAP.get(interval)
    if mapping is None:
        logger.warning(f"Interval '{interval}' not supported by Schwab API")
        return None

    freq_type, frequency = mapping

    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)  # inclusive end

        if interval == '1d':
            response = client.get_price_history(
                symbol,
                period_type=SchwabClient.PriceHistory.PeriodType.YEAR,
                frequency_type=SchwabClient.PriceHistory.FrequencyType.DAILY,
                start_datetime=start_dt,
                end_datetime=end_dt,
                need_extended_hours_data=False,
            )
        else:
            response = client.get_price_history(
                symbol,
                period_type=SchwabClient.PriceHistory.PeriodType.DAY,
                frequency_type=freq_type,
                frequency=frequency,
                start_datetime=start_dt,
                end_datetime=end_dt,
                need_extended_hours_data=False,
            )

        if response.status_code != 200:
            logger.error(f"Schwab API returned status {response.status_code}: {response.text[:200]}")
            return None

        data = response.json()
        candles = data.get('candles', [])
        if not candles:
            logger.warning(f"Schwab returned 0 candles for {symbol} {interval}")
            return None

        df = pd.DataFrame(candles)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.columns = [c.lower() for c in df.columns]
        df.index.name = symbol

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
        df = df.dropna()

        logger.info(f"Schwab: fetched {len(df)} bars for {symbol} ({interval})")
        return df

    except Exception as e:
        logger.error(f"Schwab data fetch failed: {e}")
        return None


def fetch_yfinance_data(symbol: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
    """Fetch data from yfinance as fallback."""
    if yf is None:
        return None

    # Crypto symbols: force daily bars (yfinance intraday crypto is unreliable)
    if symbol.endswith('-USD') and interval not in ('1d', '1wk', '1mo'):
        logger.warning(f"yfinance: crypto {symbol} intraday not supported, forcing 1d")
        interval = '1d'

    # Forex symbols: convert EUR/USD → EURUSD=X for yfinance
    yf_symbol = symbol
    if '/' in symbol:
        yf_symbol = symbol.replace('/', '') + '=X'

    try:
        ticker = yf.Ticker(yf_symbol)

        # yfinance intraday limits
        if interval in ('1m', '2m', '5m', '15m', '30m', '60m', '90m'):
            max_days = {'1m': 7, '2m': 60, '5m': 60, '15m': 60, '30m': 60, '60m': 730, '90m': 730}
            requested_start = datetime.strptime(start_date, '%Y-%m-%d')
            requested_end = datetime.strptime(end_date, '%Y-%m-%d')
            days_requested = (requested_end - requested_start).days
            limit = max_days.get(interval, 60)

            if days_requested > limit:
                logger.warning(f"yfinance: {interval} data limited to {limit} days, clamping from {days_requested}d")
                start_date = (requested_end - timedelta(days=limit)).strftime('%Y-%m-%d')

        df = ticker.history(start=start_date, end=end_date, interval=interval)
        if df.empty:
            return None

        df.columns = [c.lower() for c in df.columns]
        # Strip timezone so index is tz-naive like Schwab/Coinbase providers
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = symbol
        logger.info(f"yfinance: fetched {len(df)} bars for {symbol} ({interval})")
        return df

    except Exception as e:
        logger.error(f"yfinance data fetch failed: {e}")
        return None

# =============================================================================
# COINBASE CRYPTO DATA PROVIDER (public API, no auth needed)
# =============================================================================

COINBASE_INTERVAL_MAP = {
    '1m': 'ONE_MINUTE',
    '5m': 'FIVE_MINUTE',
    '15m': 'FIFTEEN_MINUTE',
    '1h': 'ONE_HOUR',
    '1d': 'ONE_DAY',
}


def fetch_coinbase_data(symbol: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
    """Fetch crypto OHLCV data from Coinbase public API.

    Supports BTC-USD, ETH-USD, SOL-USD etc. No auth required.
    Max 300 candles per request — paginates automatically for longer ranges.
    """
    if _requests is None:
        logger.warning("requests library not installed, cannot use Coinbase API")
        return None

    granularity = COINBASE_INTERVAL_MAP.get(interval)
    if granularity is None:
        logger.debug(f"Coinbase: interval '{interval}' not supported, skipping")
        return None

    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)

        # Interval durations in seconds for pagination chunking
        interval_seconds = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '1d': 86400}
        secs = interval_seconds[interval]
        max_candles = 300
        chunk_seconds = max_candles * secs

        all_candles = []
        chunk_start = int(start_dt.timestamp())
        final_end = int(end_dt.timestamp())

        while chunk_start < final_end:
            chunk_end = min(chunk_start + chunk_seconds, final_end)

            url = f'https://api.coinbase.com/api/v3/brokerage/market/products/{symbol}/candles'
            params = {
                'start': str(chunk_start),
                'end': str(chunk_end),
                'granularity': granularity,
            }

            resp = _requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Coinbase API returned {resp.status_code}: {resp.text[:200]}")
                break

            data = resp.json()
            candles = data.get('candles', [])
            if not candles:
                chunk_start = chunk_end
                continue

            all_candles.extend(candles)
            chunk_start = chunk_end

        if not all_candles:
            logger.warning(f"Coinbase: 0 candles for {symbol} {interval}")
            return None

        df = pd.DataFrame(all_candles)
        # Coinbase returns: start (unix), low, high, open, close, volume (all as strings)
        df['datetime'] = pd.to_datetime(df['start'].astype(int), unit='s')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].dropna()
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        df.index.name = symbol

        logger.info(f"Coinbase: fetched {len(df)} bars for {symbol} ({interval})")
        return df

    except Exception as e:
        logger.error(f"Coinbase data fetch failed: {e}")
        return None


# =============================================================================
# ALPACA EQUITY DATA PROVIDER (API key auth)
# =============================================================================

ALPACA_INTERVAL_MAP = {
    '1m': '1Min', '5m': '5Min', '10m': '10Min', '15m': '15Min',
    '30m': '30Min', '1h': '1Hour', '1d': '1Day',
}


def fetch_alpaca_bars(symbol: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
    """Fetch historical bars from Alpaca Data API v2."""
    cfg = _get_alpaca_config()
    if cfg is None:
        return None

    timeframe = ALPACA_INTERVAL_MAP.get(interval)
    if timeframe is None:
        logger.warning(f"Interval '{interval}' not supported by Alpaca")
        return None

    headers = {
        'APCA-API-KEY-ID': cfg['api_key'],
        'APCA-API-SECRET-KEY': cfg['secret_key'],
    }

    # Build RFC-3339 timestamps
    start_rfc = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%dT00:00:00Z')
    end_rfc = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')

    all_bars = []
    page_token = None
    url = f'https://data.alpaca.markets/v2/stocks/{symbol}/bars'

    try:
        while True:
            params = {
                'timeframe': timeframe,
                'start': start_rfc,
                'end': end_rfc,
                'limit': 10000,
                'feed': 'iex',
            }
            if page_token:
                params['page_token'] = page_token

            resp = _requests.get(url, headers=headers, params=params, timeout=15)
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
            logger.warning(f"Alpaca returned 0 bars for {symbol} ({interval})")
            return None

        df = pd.DataFrame(all_bars)
        df['datetime'] = pd.to_datetime(df['t'])
        df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].dropna()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        # Strip timezone so index is tz-naive like Schwab/Coinbase providers
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df.index.name = symbol

        logger.debug(f"Alpaca: fetched {len(df)} bars for {symbol} ({interval})")
        return df

    except Exception as e:
        logger.error(f"Alpaca data fetch failed: {e}")
        return None


class AlpacaTickStreamer:
    """Real-time trade stream via Alpaca WebSocket (wss://stream.data.alpaca.markets/v2/iex).

    Supports multiple symbols on a single connection. Fires per-symbol callbacks
    throttled to at most ~4 broadcasts/sec per symbol.
    """

    WS_URL = 'wss://stream.data.alpaca.markets/v2/iex'
    THROTTLE_SEC = 0.25  # max 4 chart pushes per second per symbol

    def __init__(self, symbols, on_tick=None):
        # Accept single string or list of symbols
        if isinstance(symbols, str):
            symbols = [symbols]
        self.symbols = symbols
        self.on_tick = on_tick          # async callback(symbol: str, price: float)
        self.latest_prices: Dict[str, float] = {}
        self.connected = False
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[_aiohttp.ClientSession] = None
        self._last_push: Dict[str, float] = {}  # per-symbol throttle timestamps

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
        """Connect, authenticate, subscribe, and stream trades with auto-reconnect."""
        import time as _time
        while True:
            try:
                self._session = _aiohttp.ClientSession()
                async with self._session.ws_connect(self.WS_URL) as ws:
                    # Server sends welcome on connect
                    await ws.receive_json()

                    # Authenticate
                    await ws.send_json({
                        'action': 'auth',
                        'key': cfg['api_key'],
                        'secret': cfg['secret_key'],
                    })
                    auth_resp = await ws.receive_json()
                    if not any(m.get('msg') == 'authenticated' for m in auth_resp):
                        # Connection limit exceeded → back off longer to let stale conns expire
                        is_limit = any(m.get('code') == 406 for m in auth_resp)
                        wait = 60 if is_limit else 5
                        logger.error(f"Alpaca stream auth failed: {auth_resp} — retry in {wait}s")
                        await self._session.close()
                        await asyncio.sleep(wait)
                        continue

                    # Subscribe to trades for ALL symbols
                    await ws.send_json({
                        'action': 'subscribe',
                        'trades': self.symbols,
                    })
                    sub_resp = await ws.receive_json()
                    logger.info(f"Alpaca stream: subscribed to {self.symbols} trades — {sub_resp}")
                    self.connected = True

                    # Read trade messages
                    async for msg in ws:
                        if msg.type == _aiohttp.WSMsgType.TEXT:
                            for item in json.loads(msg.data):
                                if item.get('T') == 't':
                                    sym = item.get('S', '')
                                    price = float(item['p'])
                                    self.latest_prices[sym] = price
                                    # Fire callback (throttled per symbol)
                                    if self.on_tick:
                                        now = _time.monotonic()
                                        last = self._last_push.get(sym, 0.0)
                                        if now - last >= self.THROTTLE_SEC:
                                            self._last_push[sym] = now
                                            await self.on_tick(sym, price)
                        elif msg.type in (_aiohttp.WSMsgType.CLOSED, _aiohttp.WSMsgType.ERROR):
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
# SCHWAB REAL-TIME TICK STREAM (Level 1 equity quotes via StreamClient)
# =============================================================================

class SchwabTickStreamer:
    """Real-time Level 1 equity quotes via Schwab StreamClient.
    Same interface as AlpacaTickStreamer."""

    THROTTLE_SEC = 0.25

    def __init__(self, symbols, on_tick=None):
        if isinstance(symbols, str):
            symbols = [symbols]
        self.symbols = symbols
        self.on_tick = on_tick
        self.latest_prices: Dict[str, float] = {}
        self.connected = False
        self._task: Optional[asyncio.Task] = None
        self._stream_client = None
        self._last_push: Dict[str, float] = {}

    async def start(self):
        client = get_schwab_client()
        if client is None:
            logger.warning("SchwabTickStreamer: no Schwab client available")
            return
        self._task = asyncio.create_task(self._run(client))

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._stream_client:
            try:
                await self._stream_client.logout()
            except Exception:
                pass
            self._stream_client = None
        self.connected = False

    async def _run(self, client):
        import time as _time
        while True:
            try:
                self._stream_client = SchwabStreamClient(client)
                await self._stream_client.login()

                def _handler(msg):
                    for item in msg.get('content', []):
                        sym = item.get('key', '')
                        price = item.get('LAST_PRICE')
                        if sym and price is not None:
                            price = float(price)
                            self.latest_prices[sym] = price
                            if self.on_tick:
                                now = _time.monotonic()
                                last = self._last_push.get(sym, 0.0)
                                if now - last >= self.THROTTLE_SEC:
                                    self._last_push[sym] = now
                                    asyncio.ensure_future(self.on_tick(sym, price))

                self._stream_client.add_level_one_equity_handler(_handler)

                Fields = SchwabStreamClient.LevelOneEquityFields
                await self._stream_client.level_one_equity_subs(
                    self.symbols, fields=[Fields.SYMBOL, Fields.LAST_PRICE])

                self.connected = True
                logger.info(f"Schwab stream: subscribed to {self.symbols}")

                while True:
                    await self._stream_client.handle_message()

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Schwab stream error: {e}, reconnecting in 5s...")
            finally:
                self.connected = False
                if self._stream_client:
                    try:
                        await self._stream_client.logout()
                    except Exception:
                        pass
                    self._stream_client = None
            await asyncio.sleep(5)


# =============================================================================
# ALPACA REAL-TIME NEWS STREAM (header auth, VADER sentiment)
# =============================================================================

class AlpacaNewsStreamer:
    """Stream real-time news from Alpaca and score sentiment with VADER."""

    WS_URL = 'wss://stream.data.alpaca.markets/v1beta1/news'

    def __init__(self, symbols: list, on_news=None):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.on_news = on_news
        self.connected = False
        self._task: Optional[asyncio.Task] = None
        self._session = None
        self._seen_ids: set = set()

    async def start(self):
        cfg = _get_alpaca_config()
        if cfg is None:
            return
        self._task = asyncio.ensure_future(self._run(cfg))

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self._session and not self._session.closed:
            await self._session.close()
        self.connected = False

    async def _run(self, cfg: dict):
        """Connect with header auth, subscribe, and stream news with auto-reconnect."""
        while True:
            try:
                self._session = _aiohttp.ClientSession()
                headers = {
                    'APCA-API-KEY-ID': cfg['api_key'],
                    'APCA-API-SECRET-KEY': cfg['secret_key'],
                }
                async with self._session.ws_connect(self.WS_URL, headers=headers) as ws:
                    # Server sends connected message, then authenticated (header auth)
                    resp = await ws.receive_json()   # [{"T":"success","msg":"connected"}]
                    resp2 = await ws.receive_json()  # [{"T":"success","msg":"authenticated"}]
                    if not any(m.get('msg') == 'authenticated' for m in resp2):
                        is_limit = any(m.get('code') == 406 for m in resp2)
                        wait = 60 if is_limit else 5
                        logger.error(f"Alpaca news auth failed: {resp2} — retry in {wait}s")
                        await self._session.close()
                        await asyncio.sleep(wait)
                        continue

                    # Subscribe to news for traded symbols
                    await ws.send_json({'action': 'subscribe', 'news': self.symbols})
                    sub_resp = await ws.receive_json()
                    logger.info(f"Alpaca news stream: subscribed to {self.symbols} — {sub_resp}")
                    self.connected = True

                    # Read news messages
                    async for msg in ws:
                        if msg.type == _aiohttp.WSMsgType.TEXT:
                            for item in json.loads(msg.data):
                                if item.get('T') == 'n':
                                    await self._handle_news(item)
                        elif msg.type in (_aiohttp.WSMsgType.CLOSED, _aiohttp.WSMsgType.ERROR):
                            break

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Alpaca news stream error: {e}, reconnecting in 5s...")
            finally:
                self.connected = False
                if self._session and not self._session.closed:
                    await self._session.close()
            await asyncio.sleep(5)

    async def _handle_news(self, item: dict):
        """Dedup, score sentiment, and fire callback."""
        news_id = item.get('id')
        if news_id in self._seen_ids:
            return
        self._seen_ids.add(news_id)
        # Cap dedup set to prevent unbounded growth
        if len(self._seen_ids) > 1000:
            self._seen_ids = set(list(self._seen_ids)[-500:])

        headline = item.get('headline', '')
        sentiment_score = 0.0
        sentiment_label = 'neutral'
        if _vader and headline:
            compound = _vader.polarity_scores(headline)['compound']
            sentiment_score = round(compound * 100, 1)
            if sentiment_score > 15:
                sentiment_label = 'bullish'
            elif sentiment_score < -15:
                sentiment_label = 'bearish'

        if self.on_news:
            await self.on_news({
                'id': news_id,
                'headline': headline,
                'summary': item.get('summary', ''),
                'symbols': item.get('symbols', []),
                'source': item.get('source', ''),
                'created_at': item.get('created_at', ''),
                'url': item.get('url', ''),
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
            })


# =============================================================================
# OANDA FOREX DATA PROVIDER (Bearer token auth)
# =============================================================================

OANDA_INTERVAL_MAP = {
    '1m': 'M1',
    '5m': 'M5',
    '10m': 'M10',
    '15m': 'M15',
    '30m': 'M30',
    '1h': 'H1',
    '1d': 'D',
}


def _get_oanda_config() -> Optional[dict]:
    """Get OANDA config from environment. Returns None if not configured."""
    _load_env_file()
    token = os.environ.get('OANDA_TOKEN', '')
    if not token:
        return None
    env = os.environ.get('OANDA_ENVIRONMENT', 'practice')
    base_url = ('https://api-fxtrade.oanda.com' if env == 'live'
                else 'https://api-fxpractice.oanda.com')
    return {'token': token, 'base_url': base_url}


def fetch_oanda_data(symbol: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
    """Fetch forex OHLCV data from OANDA REST v20 API.

    Symbol format: 'EUR/USD' → converted to 'EUR_USD' for API.
    Requires OANDA_TOKEN env var.
    """
    if _requests is None:
        logger.warning("requests library not installed, cannot use OANDA API")
        return None

    config = _get_oanda_config()
    if config is None:
        logger.debug("OANDA: not configured (set OANDA_TOKEN)")
        return None

    granularity = OANDA_INTERVAL_MAP.get(interval)
    if granularity is None:
        logger.debug(f"OANDA: interval '{interval}' not supported")
        return None

    # Convert symbol format: EUR/USD → EUR_USD
    instrument = symbol.replace('/', '_')

    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        # OANDA rejects 'to' times in the future — cap at current UTC time
        from datetime import timezone
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        if end_dt > now_utc:
            end_dt = now_utc

        url = f"{config['base_url']}/v3/instruments/{instrument}/candles"
        headers = {
            'Authorization': f"Bearer {config['token']}",
            'Content-Type': 'application/json',
        }
        params = {
            'granularity': granularity,
            'from': start_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'to': end_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'price': 'M',  # mid prices
        }

        resp = _requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"OANDA API returned {resp.status_code}: {resp.text[:200]}")
            return None

        data = resp.json()
        candles = data.get('candles', [])
        if not candles:
            logger.warning(f"OANDA: 0 candles for {symbol} {interval}")
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
            logger.warning(f"OANDA: no complete candles for {symbol} {interval}")
            return None

        df = pd.DataFrame(rows)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
        df = df.dropna()
        df.index.name = symbol

        logger.info(f"OANDA: fetched {len(df)} bars for {symbol} ({interval})")
        return df

    except Exception as e:
        logger.error(f"OANDA data fetch failed: {e}")
        return None


# =============================================================================
# PRECOMPUTED ARRAYS & MODULE-LEVEL quick_backtest
# =============================================================================

@dataclass
class PrecomputedArrays:
    """Holds all numpy arrays extracted from a DataFrame for fast backtesting."""
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    open: np.ndarray
    rsi: np.ndarray
    stoch_k: np.ndarray
    stoch_d: np.ndarray
    macd: np.ndarray
    macd_signal: np.ndarray
    macd_hist: np.ndarray
    bb_pct: np.ndarray
    adx: np.ndarray
    di_plus: np.ndarray
    di_minus: np.ndarray
    atr: np.ndarray
    atr_pctrank: np.ndarray   # rolling percentile rank of ATR (0-1)
    ema9: np.ndarray
    ema21: np.ndarray
    ema50: np.ndarray
    ema21_slope: np.ndarray
    vwap: np.ndarray
    vol_ratio: np.ndarray
    nearest_r: np.ndarray     # nearest resistance level above current price (0 = none)
    nearest_s: np.ndarray     # nearest support level below current price (0 = none)
    volume: np.ndarray
    # Candlestick patterns (1.0 = detected, 0.0 = not)
    engulf_bull: np.ndarray
    engulf_bear: np.ndarray
    hammer: np.ndarray
    shooting_star: np.ndarray
    # Divergence (1.0 = detected)
    div_bull: np.ndarray       # bullish RSI/MACD divergence
    div_bear: np.ndarray       # bearish RSI/MACD divergence
    # Volatility compression
    bb_squeeze: np.ndarray     # BB width in bottom 25th percentile (1.0/0.0)
    # Volume flow
    obv_slope: np.ndarray      # normalized OBV slope (+ = accumulation, - = distribution)
    # Extended candlestick patterns (1.0 = detected, 0.0 = not)
    doji_dragon: np.ndarray        # Dragonfly Doji (bullish)
    doji_grave: np.ndarray         # Gravestone Doji (bearish)
    morning_star: np.ndarray       # Morning Star / Morning Doji Star (bullish, 3-bar)
    evening_star: np.ndarray       # Evening Star / Evening Doji Star (bearish, 3-bar)
    three_white: np.ndarray        # Three White Soldiers (bullish, 3-bar)
    three_black: np.ndarray        # Three Black Crows (bearish, 3-bar)
    piercing: np.ndarray           # Piercing Line (bullish, 2-bar)
    dark_cloud: np.ndarray         # Dark Cloud Cover (bearish, 2-bar)
    harami_bull: np.ndarray        # Bullish Harami (2-bar)
    harami_bear: np.ndarray        # Bearish Harami (2-bar)
    marubozu_bull: np.ndarray      # Bullish Marubozu (single bar)
    marubozu_bear: np.ndarray      # Bearish Marubozu (single bar)
    hanging_man: np.ndarray        # Hanging Man (bearish, hammer shape in uptrend)
    spinning_top: np.ndarray       # Spinning Top (indecision, small body + long wicks)
    # Chart patterns (1.0 = detected, 0.0 = not)
    pat_double_top: np.ndarray
    pat_double_bottom: np.ndarray
    pat_head_shoulders: np.ndarray    # bearish H&S
    pat_inv_hs: np.ndarray            # inverse (bullish) H&S
    pat_asc_triangle: np.ndarray
    pat_desc_triangle: np.ndarray
    pat_bull_flag: np.ndarray
    pat_cup_handle: np.ndarray
    pat_falling_wedge: np.ndarray
    pat_rising_wedge: np.ndarray
    # Extended chart patterns
    pat_inv_cup_handle: np.ndarray    # Inverse Cup & Handle (bearish)
    pat_bear_flag: np.ndarray         # Bear Flag (bearish continuation)
    pat_pennant_bull: np.ndarray      # Bullish Pennant (bullish continuation)
    pat_pennant_bear: np.ndarray      # Bearish Pennant (bearish continuation)
    pat_rounding_bottom: np.ndarray   # Rounding Bottom (bullish)
    pat_rounding_top: np.ndarray      # Rounding Top (bearish)
    pat_rectangle_bull: np.ndarray    # Rectangle Bullish (breakout up)
    pat_rectangle_bear: np.ndarray    # Rectangle Bearish (breakdown)
    pat_broadening_bottom: np.ndarray # Broadening Bottom / Megaphone (bullish)
    pat_broadening_top: np.ndarray    # Broadening Top / Megaphone (bearish)
    pat_triple_bottom: np.ndarray     # Triple Bottom (bullish)
    pat_triple_top: np.ndarray        # Triple Top (bearish)
    pat_sym_tri_bull: np.ndarray      # Symmetrical Triangle Bullish
    pat_sym_tri_bear: np.ndarray      # Symmetrical Triangle Bearish
    days: Optional[np.ndarray]  # None if not day-trading
    bar_hours: Optional[np.ndarray]  # hour as float (9.5 = 9:30 AM ET), None if daily
    n: int
    lookback: int
    force_eod_close: bool


def _nan(v):
    """Replace NaN with 0."""
    return 0.0 if np.isnan(v) else v


def precompute_arrays(df: pd.DataFrame, trading_style: str, interval: str, lookback: int = 50) -> PrecomputedArrays:
    """Pre-compute indicators and extract numpy arrays from a DataFrame."""
    BacktestRunner._precompute_indicators(df)
    n = len(df)
    intraday = interval not in ('1d', '1wk', '1mo')
    # Only force EOD close in day-trading mode with intraday intervals
    force_eod_close = (trading_style == 'day') and intraday

    days = None
    bar_hours = None
    if intraday:
        days = np.array([df.index[i].date() for i in range(n)], dtype=object)
        # Store bar time as fractional hour in UTC
        # Schwab data comes as timezone-naive UTC (14:30 = 9:30 AM ET)
        bar_hours = np.array([
            df.index[i].hour + df.index[i].minute / 60.0
            if hasattr(df.index[i], 'hour') else 17.0
            for i in range(n)
        ], dtype=np.float64)

    def _col(name, fallback=0.0):
        """Safe column extract — returns zeros if indicator column is missing."""
        if name in df.columns:
            return df[name].fillna(fallback).values
        return np.full(n, fallback)

    return PrecomputedArrays(
        close=df['close'].values,
        high=df['high'].values,
        low=df['low'].values,
        open=df['open'].values if 'open' in df.columns else df['close'].values,
        rsi=_col('_rsi', 50.0),
        stoch_k=_col('_stoch_k', 50.0),
        stoch_d=_col('_stoch_d', 50.0),
        macd=_col('_macd'),
        macd_signal=_col('_macd_signal'),
        macd_hist=_col('_macd_hist'),
        bb_pct=_col('_bb_pct', 0.5),
        adx=_col('_adx'),
        di_plus=_col('_di_plus'),
        di_minus=_col('_di_minus'),
        atr=_col('_atr'),
        atr_pctrank=pd.Series(_col('_atr')).rolling(100, min_periods=20).rank(pct=True).fillna(0.5).values,
        ema9=_col('_ema9'),
        ema21=_col('_ema21'),
        ema50=_col('_ema50'),
        ema21_slope=_col('_ema21_slope'),
        vwap=_col('_vwap'),
        vol_ratio=_col('_vol_ratio', 1.0),
        nearest_r=_col('_nearest_r', 0.0),
        nearest_s=_col('_nearest_s', 0.0),
        volume=df['volume'].values if 'volume' in df.columns else np.ones(n),
        engulf_bull=_col('_engulf_bull', 0.0),
        engulf_bear=_col('_engulf_bear', 0.0),
        hammer=_col('_hammer', 0.0),
        shooting_star=_col('_shooting_star', 0.0),
        doji_dragon=_col('_doji_dragon', 0.0),
        doji_grave=_col('_doji_grave', 0.0),
        morning_star=_col('_morning_star', 0.0),
        evening_star=_col('_evening_star', 0.0),
        three_white=_col('_three_white', 0.0),
        three_black=_col('_three_black', 0.0),
        piercing=_col('_piercing', 0.0),
        dark_cloud=_col('_dark_cloud', 0.0),
        harami_bull=_col('_harami_bull', 0.0),
        harami_bear=_col('_harami_bear', 0.0),
        marubozu_bull=_col('_marubozu_bull', 0.0),
        marubozu_bear=_col('_marubozu_bear', 0.0),
        hanging_man=_col('_hanging_man', 0.0),
        spinning_top=_col('_spinning_top', 0.0),
        div_bull=_col('_div_bull', 0.0),
        div_bear=_col('_div_bear', 0.0),
        bb_squeeze=_col('_bb_squeeze', 0.0),
        obv_slope=_col('_obv_slope', 0.0),
        pat_double_top=_col('_pat_double_top', 0.0),
        pat_double_bottom=_col('_pat_double_bottom', 0.0),
        pat_head_shoulders=_col('_pat_head_shoulders', 0.0),
        pat_inv_hs=_col('_pat_inv_hs', 0.0),
        pat_asc_triangle=_col('_pat_asc_triangle', 0.0),
        pat_desc_triangle=_col('_pat_desc_triangle', 0.0),
        pat_bull_flag=_col('_pat_bull_flag', 0.0),
        pat_cup_handle=_col('_pat_cup_handle', 0.0),
        pat_falling_wedge=_col('_pat_falling_wedge', 0.0),
        pat_rising_wedge=_col('_pat_rising_wedge', 0.0),
        pat_inv_cup_handle=_col('_pat_inv_cup_handle', 0.0),
        pat_bear_flag=_col('_pat_bear_flag', 0.0),
        pat_pennant_bull=_col('_pat_pennant_bull', 0.0),
        pat_pennant_bear=_col('_pat_pennant_bear', 0.0),
        pat_rounding_bottom=_col('_pat_rounding_bottom', 0.0),
        pat_rounding_top=_col('_pat_rounding_top', 0.0),
        pat_rectangle_bull=_col('_pat_rectangle_bull', 0.0),
        pat_rectangle_bear=_col('_pat_rectangle_bear', 0.0),
        pat_broadening_bottom=_col('_pat_broadening_bottom', 0.0),
        pat_broadening_top=_col('_pat_broadening_top', 0.0),
        pat_triple_bottom=_col('_pat_triple_bottom', 0.0),
        pat_triple_top=_col('_pat_triple_top', 0.0),
        pat_sym_tri_bull=_col('_pat_sym_tri_bull', 0.0),
        pat_sym_tri_bear=_col('_pat_sym_tri_bear', 0.0),
        days=days,
        bar_hours=bar_hours,
        n=n,
        lookback=lookback,
        force_eod_close=force_eod_close,
    )


# =============================================================================
# MARKET CONTEXT (SPY trend + VIX regime)
# =============================================================================

@dataclass
class MarketContext:
    """Market-wide context arrays aligned to traded symbol's bar index."""
    spy_trend: np.ndarray    # per-bar: +1 bullish, 0 neutral, -1 bearish
    vix_level: np.ndarray    # per-bar: raw VIX close value
    n: int


def fetch_market_context(target_df: pd.DataFrame, interval: str) -> Optional[MarketContext]:
    """Fetch SPY + VIX data and align to target_df's index.

    Returns None if data unavailable (crypto/forex, no yfinance, etc.)."""
    if yf is None:
        return None
    n = len(target_df)
    start = str(target_df.index[0])[:10]
    end_dt = pd.Timestamp(str(target_df.index[-1])) + timedelta(days=1)
    end = end_dt.strftime('%Y-%m-%d')

    # Fetch SPY at same interval
    spy_df = fetch_yfinance_data('SPY', start, end, interval)
    if spy_df is None or len(spy_df) < 30:
        return None

    # Compute SPY trend from triple EMA alignment
    spy_c = spy_df['close'].values
    spy_ema9 = pd.Series(spy_c).ewm(span=9, adjust=False).mean().values
    spy_ema21 = pd.Series(spy_c).ewm(span=21, adjust=False).mean().values
    spy_ema50 = pd.Series(spy_c).ewm(span=50, adjust=False).mean().values
    spy_trend_raw = np.where(
        (spy_ema9 > spy_ema21) & (spy_ema21 > spy_ema50), 1,
        np.where((spy_ema9 < spy_ema21) & (spy_ema21 < spy_ema50), -1, 0)
    ).astype(float)
    spy_series = pd.Series(spy_trend_raw, index=spy_df.index)

    # Fetch VIX (daily only — yfinance doesn't have intraday VIX)
    vix_df = fetch_yfinance_data('^VIX', start, end, '1d')
    if vix_df is not None and not vix_df.empty:
        vix_series = pd.Series(vix_df['close'].values, index=vix_df.index)
    else:
        vix_series = pd.Series(dtype=float)

    # Align to target index via forward-fill
    spy_aligned = spy_series.reindex(target_df.index, method='ffill').fillna(0).values
    if not vix_series.empty:
        vix_aligned = vix_series.reindex(target_df.index, method='ffill').fillna(20).values
    else:
        vix_aligned = np.full(n, 20.0)

    logger.info(f"MarketContext: SPY {len(spy_df)} bars, VIX {len(vix_df) if vix_df is not None else 0} bars → aligned to {n}")
    return MarketContext(spy_trend=spy_aligned, vix_level=vix_aligned, n=n)


def _effective_stop_mult(params: dict, arrays: PrecomputedArrays, i: int) -> float:
    """Compute the effective ATR stop multiplier, adapting for ATR regime + hard cap."""
    base = params.get('atr_stop_mult', 3.0)
    if not params.get('adaptive_risk'):
        return base
    # Scale inversely with ATR percentile: high ATR → tighter stop
    pctrank = arrays.atr_pctrank[i]
    scale = max(0.6, min(1.3, 1.0 + 0.5 * (0.5 - pctrank)))
    effective = base * scale
    # Hard cap: stop distance cannot exceed max_stop_pct of price
    cp = arrays.close[i]
    atr = _nan(arrays.atr[i])
    if atr <= 0:
        atr = cp * 0.01
    max_pct = params.get('max_stop_pct', 0.04)
    max_dist = cp * max_pct
    if atr * effective > max_dist:
        effective = max_dist / atr if atr > 0 else effective
    return effective


@dataclass
class PositionState:
    """Mutable position state for evaluate_rules_signal()."""
    position: int = 0        # +shares = long, -shares = short, 0 = flat
    entry_price: float = 0.0
    r_value: float = 0.0
    trailing_stop: float = 0.0
    highest_high: float = 0.0
    lowest_low: float = 0.0
    bars_since_exit: int = 999  # high default so first entry isn't blocked
    last_score: int = 0         # winning score of last signal (for strength calc)
    last_direction: int = 0     # 1=last was long, -1=last was short (for whipsaw detection)
    trades_today: int = 0       # number of entries today (for max-per-day limit)
    last_trade_day: object = None  # date of last trade (resets counter on new day)


@dataclass
class SymbolState:
    """Per-symbol mutable state for multi-ticker paper trading."""
    symbol: str = ''
    pos: PositionState = None
    df: Optional[pd.DataFrame] = None
    arrays: Optional['PrecomputedArrays'] = None
    last_bar_time: Optional[str] = None
    bars_processed: int = 0
    bh_shares: float = 0.0          # buy-and-hold shares for this symbol
    forming: Optional[dict] = None  # tick-streamed forming candle

    def __post_init__(self):
        if self.pos is None:
            self.pos = PositionState()


def _market_structure_bias(arrays: PrecomputedArrays, i: int, lookback: int = 100) -> tuple:
    """Analyze multi-bar market structure at bar *i*.

    Looks at recent swing highs/lows to classify the broader structure —
    the same kind of analysis a human does when glancing at a chart
    (e.g. "higher lows forming", "W-bottom at support").

    Returns (bias, struct_type) where:
        bias: -2 (strong bearish) to +2 (strong bullish)
        struct_type: string label like 'uptrend', 'w_bottom', 'distribution', etc.
    """
    if i < 50:
        return 0, 'insufficient_data'

    h = arrays.high
    l = arrays.low
    c = arrays.close
    atr_v = arrays.atr[i]
    if np.isnan(atr_v) or atr_v <= 0:
        atr_v = c[i] * 0.01
    tol = atr_v * 0.5  # tolerance for "same level"

    # --- Find recent swing highs and lows (5-bar window) ---
    start = max(4, i - lookback)
    recent_sh = []  # [(bar_idx, price)]
    recent_sl = []
    for ii in range(start, i + 1):
        if ii < 4:
            continue
        wh = h[ii - 4:ii + 1]
        wl = l[ii - 4:ii + 1]
        if len(wh) == 5 and wh[2] == wh.max() and wh[2] != wh[0] and wh[2] != wh[4]:
            recent_sh.append((ii - 2, float(h[ii - 2])))
        if len(wl) == 5 and wl[2] == wl.min() and wl[2] != wl[0] and wl[2] != wl[4]:
            recent_sl.append((ii - 2, float(l[ii - 2])))

    if len(recent_sh) < 2 or len(recent_sl) < 2:
        return 0, 'insufficient_swings'

    # --- Classify swing sequences ---
    sh_prices = [p for _, p in recent_sh[-4:]]
    sl_prices = [p for _, p in recent_sl[-4:]]

    hh = sum(1 for j in range(1, len(sh_prices)) if sh_prices[j] > sh_prices[j - 1] + tol)
    lh = sum(1 for j in range(1, len(sh_prices)) if sh_prices[j] < sh_prices[j - 1] - tol)
    hl = sum(1 for j in range(1, len(sl_prices)) if sl_prices[j] > sl_prices[j - 1] + tol)
    ll = sum(1 for j in range(1, len(sl_prices)) if sl_prices[j] < sl_prices[j - 1] - tol)

    highs_asc = hh > lh
    highs_desc = lh > hh
    lows_asc = hl > ll
    lows_desc = ll > hl

    bias = 0
    struct_type = 'range'

    if highs_asc and lows_asc:
        struct_type = 'uptrend'
        bias = 2
    elif highs_desc and lows_desc:
        struct_type = 'downtrend'
        bias = -2
    elif highs_desc and lows_asc:
        struct_type = 'triangle'
        bias = 0
    elif highs_asc and lows_desc:
        struct_type = 'broadening'
        bias = 0
    elif lows_asc and not highs_desc:
        struct_type = 'accumulation'
        bias = 1
    elif highs_desc and not lows_asc:
        struct_type = 'distribution'
        bias = -1

    # --- W-bottom detection (double bottom with neckline) ---
    cp = c[i]
    if len(recent_sl) >= 2:
        low1_bar, low1_px = recent_sl[-2]
        low2_bar, low2_px = recent_sl[-1]
        if abs(low1_px - low2_px) < tol and (low2_bar - low1_bar) >= 10:
            mid_highs = [p for b, p in recent_sh if low1_bar < b < low2_bar]
            if mid_highs:
                neckline = max(mid_highs)
                if cp > neckline:
                    struct_type = 'w_bottom_confirmed'
                    bias = max(bias, 2)
                elif cp > min(low1_px, low2_px):
                    struct_type = 'w_bottom_forming'
                    bias = max(bias, 1)

    # --- M-top detection (double top with neckline) ---
    if len(recent_sh) >= 2:
        high1_bar, high1_px = recent_sh[-2]
        high2_bar, high2_px = recent_sh[-1]
        if abs(high1_px - high2_px) < tol and (high2_bar - high1_bar) >= 10:
            mid_lows = [p for b, p in recent_sl if high1_bar < b < high2_bar]
            if mid_lows:
                neckline = min(mid_lows)
                if cp < neckline:
                    struct_type = 'm_top_confirmed'
                    bias = min(bias, -2)
                elif cp < max(high1_px, high2_px):
                    struct_type = 'm_top_forming'
                    bias = min(bias, -1)

    return bias, struct_type


def evaluate_rules_signal(params: dict, arrays: PrecomputedArrays,
                          i: int, pos: PositionState,
                          mkt: Optional[MarketContext] = None) -> int:
    """Evaluate the v3 rules engine at bar index *i* and return signal.

    Returns 0=HOLD, 1=BUY, 2=SELL.  Mutates *pos* trailing-stop state
    (highest_high / lowest_low / trailing_stop) in place.
    """
    atr_stop_m = _effective_stop_mult(params, arrays, i)
    atr_target_m = params.get('atr_target_mult', 3.0)
    min_trig = params.get('min_triggers', 2)
    pb_zone = params.get('pullback_zone_atr', 1.0)
    min_adx = params.get('min_trend_adx', 20)
    min_vr = params.get('min_vol_ratio', 0.5)

    cp = arrays.close[i]
    hi = arrays.high[i]
    lo = arrays.low[i]
    op = arrays.open[i]

    atr = _nan(arrays.atr[i])
    if atr <= 0:
        atr = cp * 0.01

    ema9 = _nan(arrays.ema9[i])
    ema21 = _nan(arrays.ema21[i])
    ema50 = _nan(arrays.ema50[i])
    slope = _nan(arrays.ema21_slope[i])
    adx_v = _nan(arrays.adx[i])
    dip = _nan(arrays.di_plus[i])
    dim = _nan(arrays.di_minus[i])
    rsi = _nan(arrays.rsi[i])
    rsi_prev = _nan(arrays.rsi[i - 1]) if i > 0 else rsi
    sk = _nan(arrays.stoch_k[i])
    sd = _nan(arrays.stoch_d[i])
    sk_prev = _nan(arrays.stoch_k[i - 1]) if i > 0 else sk
    sd_prev = _nan(arrays.stoch_d[i - 1]) if i > 0 else sd
    macd_v = _nan(arrays.macd[i])
    macd_s = _nan(arrays.macd_signal[i])
    macd_h = _nan(arrays.macd_hist[i])
    macd_h_prev = _nan(arrays.macd_hist[i - 1]) if i > 0 else macd_h
    macd_v_prev = _nan(arrays.macd[i - 1]) if i > 0 else macd_v
    macd_s_prev = _nan(arrays.macd_signal[i - 1]) if i > 0 else macd_s
    vol_r = _nan(arrays.vol_ratio[i])

    # Trend
    bull_trend = ema9 > ema21 and slope > 0.02
    bear_trend = ema9 < ema21 and slope < -0.02
    strong_bull = bull_trend and ema21 > ema50 and adx_v > min_adx and dip > dim
    strong_bear = bear_trend and ema21 < ema50 and adx_v > min_adx and dim > dip

    body = abs(cp - op)
    candle_bull = cp > op and body > atr * 0.15
    candle_bear = cp < op and body > atr * 0.15

    # Momentum helpers
    macd_bull_cross = macd_v > macd_s and macd_v_prev <= macd_s_prev
    macd_bear_cross = macd_v < macd_s and macd_v_prev >= macd_s_prev
    macd_accel_up = macd_h > macd_h_prev and macd_h < 0
    macd_accel_down = macd_h < macd_h_prev and macd_h > 0
    macd_falling = macd_h < 0 and macd_h < macd_h_prev  # histogram getting more negative
    macd_rising = macd_h > 0 and macd_h > macd_h_prev   # histogram getting more positive
    stoch_bull_x = sk > sd and sk_prev <= sd_prev
    stoch_bear_x = sk < sd and sk_prev >= sd_prev
    rsi_turn_up = rsi > rsi_prev and rsi_prev < 45
    rsi_turn_down = rsi < rsi_prev and rsi_prev > 55
    vol_ok = vol_r >= min_vr
    vol_surge = vol_r > 1.5
    adx_prev = _nan(arrays.adx[i - 1]) if i > 0 else adx_v
    adx_rising = adx_v > adx_prev

    # Near EMA21?
    dist_ema21 = abs(cp - ema21) / atr if atr > 0 else 999
    near_ema21 = dist_ema21 < pb_zone

    # S/R proximity
    nr = _nan(arrays.nearest_r[i])
    ns = _nan(arrays.nearest_s[i])
    near_resistance = nr > 0 and (nr - cp) / atr < 1.0
    near_support = ns > 0 and (cp - ns) / atr < 1.0
    at_support = ns > 0 and (cp - ns) / atr < 0.3
    at_resistance = nr > 0 and (nr - cp) / atr < 0.3
    broke_above_r = nr > 0 and cp > nr
    broke_below_s = ns > 0 and cp < ns

    # --- Candlestick patterns (original 4 + 14 new) ---
    engulf_bull = arrays.engulf_bull[i] > 0.5
    engulf_bear = arrays.engulf_bear[i] > 0.5
    is_hammer = arrays.hammer[i] > 0.5
    is_shooting_star = arrays.shooting_star[i] > 0.5
    is_doji_dragon = arrays.doji_dragon[i] > 0.5
    is_doji_grave = arrays.doji_grave[i] > 0.5
    is_morning_star = arrays.morning_star[i] > 0.5
    is_evening_star = arrays.evening_star[i] > 0.5
    is_three_white = arrays.three_white[i] > 0.5
    is_three_black = arrays.three_black[i] > 0.5
    is_piercing = arrays.piercing[i] > 0.5
    is_dark_cloud = arrays.dark_cloud[i] > 0.5
    is_harami_bull = arrays.harami_bull[i] > 0.5
    is_harami_bear = arrays.harami_bear[i] > 0.5
    is_marubozu_bull = arrays.marubozu_bull[i] > 0.5
    is_marubozu_bear = arrays.marubozu_bear[i] > 0.5
    is_hanging_man = arrays.hanging_man[i] > 0.5
    # Strong reversal: multi-bar patterns count more
    bull_reversal_candle = (engulf_bull or is_hammer or is_doji_dragon
        or is_morning_star or is_three_white or is_piercing or is_harami_bull)
    bear_reversal_candle = (engulf_bear or is_shooting_star or is_doji_grave
        or is_evening_star or is_three_black or is_dark_cloud or is_harami_bear
        or is_hanging_man)
    # Strong conviction candles (marubozu = extreme directional commitment)
    strong_bull_candle = is_marubozu_bull or is_three_white
    strong_bear_candle = is_marubozu_bear or is_three_black

    # --- Divergence ---
    div_bull = arrays.div_bull[i] > 0.5
    div_bear = arrays.div_bear[i] > 0.5

    # --- Bollinger squeeze ---
    bb_in_squeeze = arrays.bb_squeeze[i] > 0.5
    bb_was_squeeze = arrays.bb_squeeze[i - 1] > 0.5 if i > 0 else False
    bb_squeeze_fire = bb_was_squeeze and not bb_in_squeeze

    # --- OBV ---
    obv_s = arrays.obv_slope[i]
    obv_bullish = obv_s > 0.1
    obv_bearish = obv_s < -0.1

    # --- Chart patterns (10 original + 14 new) ---
    any_bull_pattern = (arrays.pat_double_bottom[i] > 0.5 or arrays.pat_inv_hs[i] > 0.5
        or arrays.pat_asc_triangle[i] > 0.5 or arrays.pat_bull_flag[i] > 0.5
        or arrays.pat_cup_handle[i] > 0.5 or arrays.pat_falling_wedge[i] > 0.5
        or arrays.pat_triple_bottom[i] > 0.5 or arrays.pat_pennant_bull[i] > 0.5
        or arrays.pat_rounding_bottom[i] > 0.5 or arrays.pat_rectangle_bull[i] > 0.5
        or arrays.pat_broadening_bottom[i] > 0.5 or arrays.pat_sym_tri_bull[i] > 0.5)
    any_bear_pattern = (arrays.pat_double_top[i] > 0.5 or arrays.pat_head_shoulders[i] > 0.5
        or arrays.pat_desc_triangle[i] > 0.5 or arrays.pat_rising_wedge[i] > 0.5
        or arrays.pat_inv_cup_handle[i] > 0.5 or arrays.pat_bear_flag[i] > 0.5
        or arrays.pat_pennant_bear[i] > 0.5 or arrays.pat_rounding_top[i] > 0.5
        or arrays.pat_rectangle_bear[i] > 0.5 or arrays.pat_broadening_top[i] > 0.5
        or arrays.pat_triple_top[i] > 0.5 or arrays.pat_sym_tri_bear[i] > 0.5)

    sig = 0  # 0=HOLD, 1=BUY, 2=SELL

    # Configurable exit parameters (defaults chosen for 5m bars)
    breakeven_r = params.get('breakeven_r', 0.75)
    trail_start_r = params.get('trail_start_r', 1.0)
    trail_atr_mult = params.get('trail_atr_mult', 2.0)

    if pos.position > 0:
        # === Long position management ===
        if hi > pos.highest_high:
            pos.highest_high = hi
        current_r = (cp - pos.entry_price) / pos.r_value if pos.r_value > 0 else 0
        if current_r >= breakeven_r:
            be = pos.entry_price + atr * 0.1
            if be > pos.trailing_stop:
                pos.trailing_stop = be
        if current_r >= trail_start_r:
            trail = pos.highest_high - atr * trail_atr_mult
            if trail > pos.trailing_stop:
                pos.trailing_stop = trail
        if lo <= pos.trailing_stop:
            sig = 2
        elif current_r >= atr_target_m:
            sig = 2
        elif current_r > 0.5 and strong_bear:
            sig = 2  # only exit early on genuine trend reversal
        if sig == 2:
            pos.last_direction = 1  # was long

    elif pos.position < 0:
        # === Short position management ===
        if lo < pos.lowest_low:
            pos.lowest_low = lo
        current_r = (pos.entry_price - cp) / pos.r_value if pos.r_value > 0 else 0
        if current_r >= breakeven_r:
            be = pos.entry_price - atr * 0.1
            if be < pos.trailing_stop:
                pos.trailing_stop = be
        if current_r >= trail_start_r:
            trail = pos.lowest_low + atr * trail_atr_mult
            if trail < pos.trailing_stop:
                pos.trailing_stop = trail
        if hi >= pos.trailing_stop:
            sig = 1
        elif current_r >= atr_target_m:
            sig = 1
        elif current_r > 0.5 and strong_bull:
            sig = 1  # only exit early on genuine trend reversal
        if sig == 1:
            pos.last_direction = -1  # was short

    else:
        # === Entry detection (simplified v3 setup logic) ===
        pos.bars_since_exit += 1
        reentry_cd = params.get('reentry_cooldown_bars', 5)
        if pos.bars_since_exit < reentry_cd:
            return 0  # still in cooldown after last exit

        long_score = 0
        short_score = 0

        # Pullback in uptrend
        if (bull_trend or strong_bull) and near_ema21 and cp > ema21 and candle_bull:
            long_score += 1
            if rsi_turn_up: long_score += 1
            if stoch_bull_x or (sk < 50 and sk > sd): long_score += 1
            if macd_accel_up or macd_bull_cross: long_score += 1
            if vol_surge: long_score += 1

        # MACD cross in trend — require volume surge, not just vol_ok
        if macd_bull_cross and bull_trend and candle_bull:
            long_score = max(long_score, 1)
            if vol_surge: long_score = max(long_score, 2)

        # Momentum breakdown long — strong uptrend accelerating, no near-EMA required
        if strong_bull and candle_bull and macd_rising and adx_rising and adx_v >= 30:
            long_score = max(long_score, 2)
            if vol_surge: long_score = max(long_score, 3)

        # BREAKOUT — price closes above resistance with above-avg volume
        if broke_above_r and candle_bull and vol_r > 1.2:
            long_score = max(long_score, 2)
            if adx_rising: long_score = max(long_score, 3)
            if macd_h > 0: long_score = max(long_score, long_score + 1)
            if bb_squeeze_fire: long_score = max(long_score, 3)

        # BOUNCE — oversold at support with reversal candle (not in downtrend)
        # Tightened: require short-term trend to NOT be bearish (was: not strong_bear)
        if at_support and rsi < 35 and candle_bull and not bear_trend:
            long_score = max(long_score, 1)
            if bull_reversal_candle: long_score = max(long_score, 2)
            if stoch_bull_x or (sk < 30 and sk > sd): long_score = max(long_score, 2)
            if macd_accel_up or macd_bull_cross: long_score = max(long_score, long_score + 1)

        # Pullback in downtrend
        if (bear_trend or strong_bear) and near_ema21 and cp < ema21 and candle_bear:
            short_score += 1
            if rsi_turn_down: short_score += 1
            if stoch_bear_x or (sk > 50 and sk < sd): short_score += 1
            if macd_accel_down or macd_bear_cross: short_score += 1
            if vol_surge: short_score += 1

        # MACD cross in trend — require volume surge
        if macd_bear_cross and bear_trend and candle_bear:
            short_score = max(short_score, 1)
            if vol_surge: short_score = max(short_score, 2)

        # Momentum breakdown short — strong downtrend accelerating, no near-EMA required
        if strong_bear and candle_bear and macd_falling and adx_rising and adx_v >= 30:
            short_score = max(short_score, 2)
            if vol_surge: short_score = max(short_score, 3)

        # BREAKOUT short — price closes below support with above-avg volume
        if broke_below_s and candle_bear and vol_r > 1.2:
            short_score = max(short_score, 2)
            if adx_rising: short_score = max(short_score, 3)
            if macd_h < 0: short_score = max(short_score, short_score + 1)
            if bb_squeeze_fire: short_score = max(short_score, 3)

        # BOUNCE short — overbought at resistance with reversal (not in uptrend)
        # Tightened: require short-term trend to NOT be bullish (was: not strong_bull)
        if at_resistance and rsi > 65 and candle_bear and not bull_trend:
            short_score = max(short_score, 1)
            if bear_reversal_candle: short_score = max(short_score, 2)
            if stoch_bear_x or (sk > 70 and sk < sd): short_score = max(short_score, 2)
            if macd_accel_down or macd_bear_cross: short_score = max(short_score, short_score + 1)

        # DIVERGENCE long — bullish RSI/MACD divergence
        if div_bull and candle_bull:
            long_score = max(long_score, 1)
            if at_support: long_score = max(long_score, 2)
            if bull_reversal_candle: long_score = max(long_score, long_score + 1)
            if obv_bullish: long_score = max(long_score, long_score + 1)

        # DIVERGENCE short — bearish RSI/MACD divergence
        if div_bear and candle_bear:
            short_score = max(short_score, 1)
            if at_resistance: short_score = max(short_score, 2)
            if bear_reversal_candle: short_score = max(short_score, short_score + 1)
            if obv_bearish: short_score = max(short_score, short_score + 1)

        # SQUEEZE BREAKOUT long — BB squeeze release with bullish move
        if bb_squeeze_fire and candle_bull and cp > ema9:
            long_score = max(long_score, 1)
            if vol_surge: long_score = max(long_score, 2)
            if adx_rising: long_score = max(long_score, long_score + 1)

        # SQUEEZE BREAKOUT short — BB squeeze release with bearish move
        if bb_squeeze_fire and candle_bear and cp < ema9:
            short_score = max(short_score, 1)
            if vol_surge: short_score = max(short_score, 2)
            if adx_rising: short_score = max(short_score, short_score + 1)

        # Volume flow confirmation (OBV)
        if long_score > 0 and obv_bullish:
            long_score += 1
        if long_score > 0 and obv_bearish and not broke_above_r:
            long_score = max(0, long_score - 1)
        if short_score > 0 and obv_bearish:
            short_score += 1
        if short_score > 0 and obv_bullish and not broke_below_s:
            short_score = max(0, short_score - 1)

        # --- MARKET STRUCTURE ANALYSIS (big-picture context) ---
        struct_bias, struct_type = _market_structure_bias(arrays, i)

        # Chart patterns WITH structural confirmation → legitimate independent entries
        # (Structure answers "is the big picture supporting this pattern?")
        if any_bull_pattern and candle_bull:
            if struct_bias >= 1:
                # Bullish structure + bullish pattern = high conviction entry
                long_score = max(long_score, 2)
                if vol_surge: long_score += 1
            elif long_score > 0:
                # No structural support — just boost existing setup
                if vol_surge: long_score += 1
                if bull_reversal_candle: long_score += 1

        if any_bear_pattern and candle_bear:
            if struct_bias <= -1:
                # Bearish structure + bearish pattern = high conviction entry
                short_score = max(short_score, 2)
                if vol_surge: short_score += 1
            elif short_score > 0:
                if vol_surge: short_score += 1
                if bear_reversal_candle: short_score += 1

        # CONVICTION CANDLE — only boost existing setups
        if strong_bull_candle and candle_bull and (bull_trend or obv_bullish) and long_score > 0:
            long_score += 1
        if strong_bear_candle and candle_bear and (bear_trend or obv_bearish) and short_score > 0:
            short_score += 1

        # Structural bias scoring — reward trades aligned with structure,
        # penalize trades fighting the structure
        if struct_bias >= 2:    # strong bullish (uptrend, w-bottom confirmed)
            if long_score > 0: long_score += 1
            if short_score > 0 and not strong_bear: short_score = max(0, short_score - 2)
        elif struct_bias <= -2: # strong bearish (downtrend, m-top confirmed)
            if short_score > 0: short_score += 1
            if long_score > 0 and not strong_bull: long_score = max(0, long_score - 2)
        elif struct_bias == 1:  # mild bullish (accumulation, w-bottom forming)
            if long_score > 0: long_score += 1
            if short_score > 0 and not strong_bear: short_score = max(0, short_score - 1)
        elif struct_bias == -1: # mild bearish (distribution, m-top forming)
            if short_score > 0: short_score += 1
            if long_score > 0 and not strong_bull: long_score = max(0, long_score - 1)

        # --- DIVERGENCE CONFLICT ---
        # Never short into a confirmed bullish divergence (and vice versa).
        # Divergence means the underlying momentum disagrees with price.
        if div_bull and short_score > 0 and not broke_below_s:
            short_score = max(0, short_score - 2)
        if div_bear and long_score > 0 and not broke_above_r:
            long_score = max(0, long_score - 2)

        # --- EXTREME RSI GUARD ---
        # Tiered: hard block at extreme levels, penalty at warning levels.
        rsi_ext_lo = params.get('rsi_extreme_lo', 25)
        rsi_ext_hi = params.get('rsi_extreme_hi', 75)
        if rsi < rsi_ext_lo and short_score > 0 and not broke_below_s:
            if rsi < 15:
                short_score = 0  # extreme exhaustion — hard block
            else:
                short_score = max(0, short_score - 2)
        if rsi > rsi_ext_hi and long_score > 0 and not broke_above_r:
            if rsi > 85:
                long_score = 0  # extreme exhaustion — hard block
            else:
                long_score = max(0, long_score - 2)

        # --- OVEREXTENSION GUARD ---
        # Price stretched too far from EMA21 = easy money already made, wait for pullback.
        stretch_limit = params.get('max_stretch_atr', 2.5)
        if dist_ema21 > stretch_limit:
            if cp > ema21 and long_score > 0 and not broke_above_r:
                long_score = max(0, long_score - 1)
            if cp < ema21 and short_score > 0 and not broke_below_s:
                short_score = max(0, short_score - 1)

        # S/R blocking — penalize entries heading into a wall
        # Don't go long into resistance (unless breakout)
        if near_resistance and not broke_above_r and long_score > 0:
            long_score = max(0, long_score - 1)
        # Don't go short into support (unless breakdown)
        if near_support and not broke_below_s and short_score > 0:
            short_score = max(0, short_score - 1)

        # Choppy market penalty — low ADX means no directional trend
        adx_floor = params.get('min_entry_adx', 20)
        if adx_v < 15:
            # Dead zone — ADX this low means price action is noise
            long_score = max(0, long_score - 2)
            short_score = max(0, short_score - 2)
        elif adx_v < adx_floor:
            long_score = max(0, long_score - 1)
            short_score = max(0, short_score - 1)

        # --- Counter-trend penalty ---
        # Entering against the medium-term trend (EMA21 vs EMA50) needs extra conviction.
        # Deduct from score so counter-trend trades need more triggers to qualify.
        _ct_penalty = params.get('counter_trend_penalty', 1)
        if _ct_penalty > 0:
            if long_score > 0 and ema21 < ema50:
                long_score = max(0, long_score - _ct_penalty)
            if short_score > 0 and ema21 > ema50:
                short_score = max(0, short_score - _ct_penalty)

        # --- Market context filter (adaptive risk) ---
        if mkt is not None and params.get('adaptive_risk') and i < mkt.n:
            spy_t = mkt.spy_trend[i]
            vix_v = mkt.vix_level[i]
            # Block counter-trend entries (unless strong setup)
            if spy_t > 0 and short_score > 0 and not strong_bear:
                short_score = 0
            if spy_t < 0 and long_score > 0 and not strong_bull:
                long_score = 0
            # Extreme VIX → require extra trigger
            if vix_v > 30:
                min_trig = max(min_trig, min_trig + 1)

        # --- Opening volatility cooldown ---
        # Skip new entries in the first N minutes after market open (9:30 AM ET = 14:30 UTC)
        # to avoid opening whipsaw. Only applies to intraday data with bar_hours.
        skip_open_mins = params.get('skip_open_minutes', 20)
        if skip_open_mins > 0 and arrays.bar_hours is not None:
            bh = arrays.bar_hours[i]
            market_open_utc = 14.5  # 14:30 UTC = 9:30 AM ET
            cooldown_end_utc = market_open_utc + skip_open_mins / 60.0  # e.g. 15.0 for 30 mins
            if market_open_utc <= bh < cooldown_end_utc:
                long_score = 0
                short_score = 0

        # --- Last 15 minutes guard (avoid close-of-day chop) ---
        # 3:45-4:00 PM ET = 19:45-20:00 UTC
        if arrays.bar_hours is not None:
            bh = arrays.bar_hours[i]
            if 19.75 <= bh < 20.0:
                long_score = 0
                short_score = 0

        # --- Max trades per day limit ---
        # Prevent whipsaw cascading: cap entries at N per day
        max_tpd = params.get('max_trades_per_day', 2)
        if arrays.days is not None:
            bar_day = arrays.days[i]
            if pos.last_trade_day is not None and bar_day != pos.last_trade_day:
                pos.trades_today = 0  # reset on new day
            if pos.trades_today >= max_tpd:
                long_score = 0
                short_score = 0

        if long_score >= min_trig and vol_ok:
            sig = 1
            pos.last_score = long_score
            # Track daily trade count
            if arrays.days is not None:
                pos.trades_today += 1
                pos.last_trade_day = arrays.days[i]
        elif short_score >= min_trig and vol_ok:
            sig = 2
            pos.last_score = short_score
            if arrays.days is not None:
                pos.trades_today += 1
                pos.last_trade_day = arrays.days[i]
        else:
            pos.last_score = max(long_score, short_score)

    return sig


def describe_signal_reasoning(arrays: PrecomputedArrays, i: int, params: dict) -> str:
    """Build a human-readable reasoning string for the signal at bar *i*."""
    cp = arrays.close[i]
    atr = _nan(arrays.atr[i])
    ema9 = _nan(arrays.ema9[i])
    ema21 = _nan(arrays.ema21[i])
    ema50 = _nan(arrays.ema50[i])
    slope = _nan(arrays.ema21_slope[i])
    adx_v = _nan(arrays.adx[i])
    rsi = _nan(arrays.rsi[i])
    macd_h = _nan(arrays.macd_hist[i])
    vol_r = _nan(arrays.vol_ratio[i])
    sk = _nan(arrays.stoch_k[i])

    parts = []
    bull_trend = ema9 > ema21 and slope > 0.02
    bear_trend = ema9 < ema21 and slope < -0.02
    if bull_trend:
        parts.append(f"Uptrend (EMA9>{ema9:.1f} > EMA21>{ema21:.1f}, slope {slope:.2f}%)")
    elif bear_trend:
        parts.append(f"Downtrend (EMA9>{ema9:.1f} < EMA21>{ema21:.1f}, slope {slope:.2f}%)")
    else:
        parts.append(f"No trend (slope {slope:.2f}%)")

    if ema21 > ema50:
        parts.append(f"EMA21>{ema50:.1f}=EMA50 (mid-term UP)")
    elif ema21 < ema50:
        parts.append(f"EMA21<{ema50:.1f}=EMA50 (mid-term DOWN)")
    parts.append(f"RSI {rsi:.1f}")
    parts.append(f"MACD hist {macd_h:.3f}")
    parts.append(f"ADX {adx_v:.1f}")
    parts.append(f"Stoch K {sk:.1f}")
    parts.append(f"Vol ratio {vol_r:.2f}")
    parts.append(f"ATR {atr:.4f}")
    parts.append(f"Price ${cp:.2f}")

    nr = _nan(arrays.nearest_r[i])
    ns = _nan(arrays.nearest_s[i])
    if nr > 0:
        parts.append(f"R ${nr:.2f}")
    if ns > 0:
        parts.append(f"S ${ns:.2f}")

    # Candlestick patterns (18 total)
    if arrays.engulf_bull[i] > 0.5: parts.append("ENGULF BULL")
    if arrays.engulf_bear[i] > 0.5: parts.append("ENGULF BEAR")
    if arrays.hammer[i] > 0.5: parts.append("HAMMER")
    if arrays.shooting_star[i] > 0.5: parts.append("SHOOTING STAR")
    if arrays.doji_dragon[i] > 0.5: parts.append("DRAGONFLY DOJI")
    if arrays.doji_grave[i] > 0.5: parts.append("GRAVESTONE DOJI")
    if arrays.morning_star[i] > 0.5: parts.append("MORNING STAR")
    if arrays.evening_star[i] > 0.5: parts.append("EVENING STAR")
    if arrays.three_white[i] > 0.5: parts.append("THREE WHITE SOLDIERS")
    if arrays.three_black[i] > 0.5: parts.append("THREE BLACK CROWS")
    if arrays.piercing[i] > 0.5: parts.append("PIERCING LINE")
    if arrays.dark_cloud[i] > 0.5: parts.append("DARK CLOUD COVER")
    if arrays.harami_bull[i] > 0.5: parts.append("BULL HARAMI")
    if arrays.harami_bear[i] > 0.5: parts.append("BEAR HARAMI")
    if arrays.marubozu_bull[i] > 0.5: parts.append("BULL MARUBOZU")
    if arrays.marubozu_bear[i] > 0.5: parts.append("BEAR MARUBOZU")
    if arrays.hanging_man[i] > 0.5: parts.append("HANGING MAN")
    if arrays.spinning_top[i] > 0.5: parts.append("SPINNING TOP")
    # Divergence & squeeze
    if arrays.div_bull[i] > 0.5: parts.append("BULL DIVERGENCE")
    if arrays.div_bear[i] > 0.5: parts.append("BEAR DIVERGENCE")
    if arrays.bb_squeeze[i] > 0.5: parts.append("BB SQUEEZE")
    obv_s = arrays.obv_slope[i]
    if obv_s > 0.1: parts.append(f"OBV +{obv_s:.2f}")
    elif obv_s < -0.1: parts.append(f"OBV {obv_s:.2f}")

    # Chart patterns (24 total)
    if arrays.pat_double_top[i] > 0.5: parts.append("DOUBLE TOP")
    if arrays.pat_double_bottom[i] > 0.5: parts.append("DOUBLE BOTTOM")
    if arrays.pat_head_shoulders[i] > 0.5: parts.append("HEAD & SHOULDERS")
    if arrays.pat_inv_hs[i] > 0.5: parts.append("INV HEAD & SHOULDERS")
    if arrays.pat_asc_triangle[i] > 0.5: parts.append("ASC TRIANGLE")
    if arrays.pat_desc_triangle[i] > 0.5: parts.append("DESC TRIANGLE")
    if arrays.pat_bull_flag[i] > 0.5: parts.append("BULL FLAG")
    if arrays.pat_cup_handle[i] > 0.5: parts.append("CUP & HANDLE")
    if arrays.pat_falling_wedge[i] > 0.5: parts.append("FALLING WEDGE")
    if arrays.pat_rising_wedge[i] > 0.5: parts.append("RISING WEDGE")
    if arrays.pat_inv_cup_handle[i] > 0.5: parts.append("INV CUP & HANDLE")
    if arrays.pat_bear_flag[i] > 0.5: parts.append("BEAR FLAG")
    if arrays.pat_pennant_bull[i] > 0.5: parts.append("BULL PENNANT")
    if arrays.pat_pennant_bear[i] > 0.5: parts.append("BEAR PENNANT")
    if arrays.pat_rounding_bottom[i] > 0.5: parts.append("ROUNDING BOTTOM")
    if arrays.pat_rounding_top[i] > 0.5: parts.append("ROUNDING TOP")
    if arrays.pat_rectangle_bull[i] > 0.5: parts.append("RECT BREAKOUT")
    if arrays.pat_rectangle_bear[i] > 0.5: parts.append("RECT BREAKDOWN")
    if arrays.pat_broadening_bottom[i] > 0.5: parts.append("BROADENING BOTTOM")
    if arrays.pat_broadening_top[i] > 0.5: parts.append("BROADENING TOP")
    if arrays.pat_triple_bottom[i] > 0.5: parts.append("TRIPLE BOTTOM")
    if arrays.pat_triple_top[i] > 0.5: parts.append("TRIPLE TOP")
    if arrays.pat_sym_tri_bull[i] > 0.5: parts.append("SYM TRI BULL")
    if arrays.pat_sym_tri_bear[i] > 0.5: parts.append("SYM TRI BEAR")

    # Market structure context
    struct_bias, struct_type = _market_structure_bias(arrays, i)
    struct_labels = {
        'uptrend': 'UPTREND (HH+HL)', 'downtrend': 'DOWNTREND (LH+LL)',
        'accumulation': 'ACCUMULATION (higher lows)', 'distribution': 'DISTRIBUTION (lower highs)',
        'w_bottom_confirmed': 'W-BOTTOM CONFIRMED', 'w_bottom_forming': 'W-BOTTOM FORMING',
        'm_top_confirmed': 'M-TOP CONFIRMED', 'm_top_forming': 'M-TOP FORMING',
        'triangle': 'TRIANGLE (converging)', 'broadening': 'BROADENING',
        'range': 'RANGE-BOUND',
    }
    label = struct_labels.get(struct_type, struct_type.upper())
    bias_arrow = {2: '++', 1: '+', 0: '~', -1: '-', -2: '--'}.get(struct_bias, '~')
    parts.append(f"STRUCT: {label} [{bias_arrow}]")

    # Guard flags — show what would block/penalize entries
    guards = []
    if rsi < 25: guards.append('RSI_EXTREME_LO')
    if rsi > 75: guards.append('RSI_EXTREME_HI')
    dist_ema21 = abs(cp - ema21) / atr if atr > 0 else 0
    if dist_ema21 > 2.5: guards.append(f'STRETCHED_{dist_ema21:.1f}ATR')
    if arrays.div_bull[i] > 0.5 and ema9 < ema21: guards.append('DIV_BULL_VS_BEAR_TREND')
    if arrays.div_bear[i] > 0.5 and ema9 > ema21: guards.append('DIV_BEAR_VS_BULL_TREND')
    if arrays.bar_hours is not None:
        bh = arrays.bar_hours[i]
        skip_mins = params.get('skip_open_minutes', 20)
        if 14.5 <= bh < 14.5 + skip_mins / 60.0:
            guards.append(f'OPENING_{skip_mins}MIN')
        if 19.75 <= bh < 20.0:
            guards.append('LAST_15MIN')
    if guards:
        parts.append(f"GUARD: {','.join(guards)}")

    return " | ".join(parts)


# Pattern classification for chart highlighting
BULLISH_PATTERNS = {
    'engulf_bull': 'Engulf', 'hammer': 'Hammer', 'doji_dragon': 'DrgnDoji',
    'morning_star': 'MornStar', 'three_white': '3White', 'piercing': 'Pierce',
    'harami_bull': 'Harami', 'marubozu_bull': 'Marubozu', 'div_bull': 'BullDiv',
    'pat_double_bottom': 'DblBot', 'pat_inv_hs': 'InvHS', 'pat_asc_triangle': 'AscTri',
    'pat_bull_flag': 'BullFlg', 'pat_cup_handle': 'Cup&H', 'pat_falling_wedge': 'FallWdg',
    'pat_triple_bottom': 'TplBot', 'pat_pennant_bull': 'BullPen', 'pat_rounding_bottom': 'RndBot',
    'pat_rectangle_bull': 'RectBull', 'pat_broadening_bottom': 'BrdBot',
    'pat_sym_tri_bull': 'SymTriB',
}
BEARISH_PATTERNS = {
    'engulf_bear': 'Engulf', 'shooting_star': 'ShootStar', 'doji_grave': 'GrvDoji',
    'evening_star': 'EvnStar', 'three_black': '3Black', 'dark_cloud': 'DkCloud',
    'harami_bear': 'Harami', 'marubozu_bear': 'Marubozu', 'hanging_man': 'HangMan',
    'div_bear': 'BearDiv',
    'pat_double_top': 'DblTop', 'pat_head_shoulders': 'H&S', 'pat_desc_triangle': 'DescTri',
    'pat_rising_wedge': 'RiseWdg', 'pat_inv_cup_handle': 'InvC&H', 'pat_bear_flag': 'BearFlg',
    'pat_pennant_bear': 'BearPen', 'pat_rounding_top': 'RndTop',
    'pat_rectangle_bear': 'RectBear', 'pat_broadening_top': 'BrdTop',
    'pat_triple_top': 'TplTop', 'pat_sym_tri_bear': 'SymTriBr',
}
NEUTRAL_PATTERNS = {
    'spinning_top': 'SpinTop', 'bb_squeeze': 'Squeeze',
}


def _get_patterns_at_bar(arrays: PrecomputedArrays, i: int) -> list:
    """Return detected patterns at bar i for chart highlighting.

    Returns list of {'name': 'DblBot', 'type': 'bullish'|'bearish'|'neutral'}.
    """
    result = []
    for field, abbr in BULLISH_PATTERNS.items():
        arr = getattr(arrays, field, None)
        if arr is not None and i < len(arr) and arr[i] > 0.5:
            result.append({'name': abbr, 'type': 'bullish'})
    for field, abbr in BEARISH_PATTERNS.items():
        arr = getattr(arrays, field, None)
        if arr is not None and i < len(arr) and arr[i] > 0.5:
            result.append({'name': abbr, 'type': 'bearish'})
    for field, abbr in NEUTRAL_PATTERNS.items():
        arr = getattr(arrays, field, None)
        if arr is not None and i < len(arr) and arr[i] > 0.5:
            result.append({'name': abbr, 'type': 'neutral'})
    return result


def _snapshot_active_patterns(arrays: PrecomputedArrays, i: int) -> list:
    """Return list of active pattern/indicator names at bar *i*.

    Used by walk-forward trade logging to attribute which patterns contributed
    to each trade entry.  O(1) per call — just reads pre-computed arrays.
    """
    active = []
    # Candlestick patterns (18)
    if arrays.engulf_bull[i] > 0.5: active.append('ENGULF_BULL')
    if arrays.engulf_bear[i] > 0.5: active.append('ENGULF_BEAR')
    if arrays.hammer[i] > 0.5: active.append('HAMMER')
    if arrays.shooting_star[i] > 0.5: active.append('SHOOTING_STAR')
    if arrays.doji_dragon[i] > 0.5: active.append('DOJI_DRAGON')
    if arrays.doji_grave[i] > 0.5: active.append('DOJI_GRAVE')
    if arrays.morning_star[i] > 0.5: active.append('MORNING_STAR')
    if arrays.evening_star[i] > 0.5: active.append('EVENING_STAR')
    if arrays.three_white[i] > 0.5: active.append('THREE_WHITE')
    if arrays.three_black[i] > 0.5: active.append('THREE_BLACK')
    if arrays.piercing[i] > 0.5: active.append('PIERCING')
    if arrays.dark_cloud[i] > 0.5: active.append('DARK_CLOUD')
    if arrays.harami_bull[i] > 0.5: active.append('HARAMI_BULL')
    if arrays.harami_bear[i] > 0.5: active.append('HARAMI_BEAR')
    if arrays.marubozu_bull[i] > 0.5: active.append('MARUBOZU_BULL')
    if arrays.marubozu_bear[i] > 0.5: active.append('MARUBOZU_BEAR')
    if arrays.hanging_man[i] > 0.5: active.append('HANGING_MAN')
    if arrays.spinning_top[i] > 0.5: active.append('SPINNING_TOP')
    # Divergence
    if arrays.div_bull[i] > 0.5: active.append('DIV_BULL')
    if arrays.div_bear[i] > 0.5: active.append('DIV_BEAR')
    # Squeeze
    if arrays.bb_squeeze[i] > 0.5: active.append('BB_SQUEEZE')
    if i > 0 and arrays.bb_squeeze[i - 1] > 0.5 and arrays.bb_squeeze[i] <= 0.5:
        active.append('SQUEEZE_FIRE')
    # OBV
    obv = arrays.obv_slope[i]
    if obv > 0.1: active.append('OBV_BULL')
    elif obv < -0.1: active.append('OBV_BEAR')
    # Chart patterns (24)
    if arrays.pat_double_top[i] > 0.5: active.append('DOUBLE_TOP')
    if arrays.pat_double_bottom[i] > 0.5: active.append('DOUBLE_BOTTOM')
    if arrays.pat_head_shoulders[i] > 0.5: active.append('HEAD_SHOULDERS')
    if arrays.pat_inv_hs[i] > 0.5: active.append('INV_HS')
    if arrays.pat_asc_triangle[i] > 0.5: active.append('ASC_TRIANGLE')
    if arrays.pat_desc_triangle[i] > 0.5: active.append('DESC_TRIANGLE')
    if arrays.pat_bull_flag[i] > 0.5: active.append('BULL_FLAG')
    if arrays.pat_cup_handle[i] > 0.5: active.append('CUP_HANDLE')
    if arrays.pat_falling_wedge[i] > 0.5: active.append('FALLING_WEDGE')
    if arrays.pat_rising_wedge[i] > 0.5: active.append('RISING_WEDGE')
    if arrays.pat_inv_cup_handle[i] > 0.5: active.append('INV_CUP_HANDLE')
    if arrays.pat_bear_flag[i] > 0.5: active.append('BEAR_FLAG')
    if arrays.pat_pennant_bull[i] > 0.5: active.append('PENNANT_BULL')
    if arrays.pat_pennant_bear[i] > 0.5: active.append('PENNANT_BEAR')
    if arrays.pat_rounding_bottom[i] > 0.5: active.append('ROUNDING_BOTTOM')
    if arrays.pat_rounding_top[i] > 0.5: active.append('ROUNDING_TOP')
    if arrays.pat_rectangle_bull[i] > 0.5: active.append('RECT_BULL')
    if arrays.pat_rectangle_bear[i] > 0.5: active.append('RECT_BEAR')
    if arrays.pat_broadening_bottom[i] > 0.5: active.append('BROADENING_BOTTOM')
    if arrays.pat_broadening_top[i] > 0.5: active.append('BROADENING_TOP')
    if arrays.pat_triple_bottom[i] > 0.5: active.append('TRIPLE_BOTTOM')
    if arrays.pat_triple_top[i] > 0.5: active.append('TRIPLE_TOP')
    if arrays.pat_sym_tri_bull[i] > 0.5: active.append('SYM_TRI_BULL')
    if arrays.pat_sym_tri_bear[i] > 0.5: active.append('SYM_TRI_BEAR')
    # Trend context
    ema9 = _nan(arrays.ema9[i])
    ema21 = _nan(arrays.ema21[i])
    ema50 = _nan(arrays.ema50[i])
    slope = _nan(arrays.ema21_slope[i])
    if ema9 > ema21 > ema50 and slope > 0.05: active.append('STRONG_BULL')
    elif ema9 > ema21 and slope > 0.02: active.append('BULL_TREND')
    elif ema9 < ema21 < ema50 and slope < -0.05: active.append('STRONG_BEAR')
    elif ema9 < ema21 and slope < -0.02: active.append('BEAR_TREND')
    # Volume
    if _nan(arrays.vol_ratio[i]) > 1.5: active.append('VOL_SURGE')
    return active


def quick_backtest(params: dict, arrays: PrecomputedArrays,
                   capital: float = 100000, commission: float = 0.001,
                   slippage_factor: float = 0.0,
                   mkt: Optional[MarketContext] = None,
                   trade_log: Optional[list] = None,
                   start_bar: Optional[int] = None,
                   end_bar: Optional[int] = None) -> dict:
    """Run a fast v3 backtest — setup-based entries, ATR trailing stops, pure numpy.

    Module-level function that operates on PrecomputedArrays (no closure captures).
    Delegates signal evaluation to evaluate_rules_signal().

    Optional walk-forward parameters:
      trade_log: if provided (list), append detailed trade dicts for pattern attribution
      start_bar / end_bar: restrict evaluation to a bar range (for fold windowing)

    Adaptive risk features (enabled via params['adaptive_risk'] = True):
      - Vol-normalized position sizing (risk fixed % of equity per trade)
      - Stop execution at stop price, not bar close
      - Drawdown throttle (halve risk at 5% DD, skip at 10%)
      - Consecutive-loss cooldown (skip after N losses in a row)
      - Market context filtering (SPY trend + VIX regime)
    """
    n = arrays.n
    lookback = arrays.lookback
    arr_close = arrays.close
    arr_high = arrays.high
    arr_low = arrays.low
    arr_open = arrays.open
    arr_atr = arrays.atr
    arr_days = arrays.days
    commission_pct = commission

    # Adaptive risk params (defaults preserve legacy behavior when off)
    adaptive = params.get('adaptive_risk', False)
    risk_per_trade = params.get('risk_per_trade', 0.02) if adaptive else 0.0
    dd_half = params.get('drawdown_half_risk', 0.05) if adaptive else 1.0
    dd_skip = params.get('drawdown_skip', 0.10) if adaptive else 1.0
    cooldown_n = params.get('cooldown_losses', 3) if adaptive else 999
    cooldown_decay_bars = params.get('reentry_cooldown_bars', 5) * 6 if adaptive else 9999
    # ^ After this many flat bars in cooldown, reduce consec_losses by 1 (prevents permanent lockout)

    cash = capital
    pos = PositionState()
    trades = []
    peak_equity = capital
    max_dd = 0.0
    current_day = None
    consec_losses = 0
    _flat_bars_since_cooldown = 0  # counts bars spent flat while in cooldown
    _logging = trade_log is not None
    _entry_bar = 0       # bar index where current position was entered
    _entry_patterns = []  # patterns active at entry

    loop_start = max(start_bar or lookback, lookback)
    loop_end = min(end_bar or n, n)

    for i in range(loop_start, loop_end):
        cp = arr_close[i]
        hi = arr_high[i]
        lo = arr_low[i]
        op = arr_open[i]

        # Day trading: force close on new day (only when force_eod_close is set)
        if arrays.force_eod_close and arr_days is not None:
            bar_day = arr_days[i]
            if current_day is not None and bar_day != current_day and pos.position != 0:
                pp = arr_close[i - 1]
                slip = slippage_factor * _nan(arr_atr[i - 1]) if slippage_factor > 0 else 0.0
                if pos.position > 0:
                    sell_px = pp - slip
                    pnl = pos.position * sell_px * (1 - commission_pct) - pos.position * pos.entry_price * (1 + commission_pct)
                    cash += pos.position * sell_px * (1 - commission_pct)
                    trades.append(pnl)
                    consec_losses = consec_losses + 1 if pnl < 0 else 0
                    if _logging:
                        trade_log.append({
                            'entry_bar': _entry_bar, 'exit_bar': i, 'direction': 'long',
                            'entry_price': pos.entry_price, 'exit_price': sell_px,
                            'shares': pos.position, 'pnl': round(pnl, 2),
                            'return_pct': round((sell_px / pos.entry_price - 1) * 100, 3) if pos.entry_price > 0 else 0,
                            'patterns_at_entry': _entry_patterns, 'exit_reason': 'eod_close',
                        })
                else:
                    s = abs(pos.position)
                    cover_px = pp + slip
                    pnl = s * pos.entry_price * (1 - commission_pct) - s * cover_px * (1 + commission_pct)
                    cash -= s * cover_px * (1 + commission_pct)
                    trades.append(pnl)
                    consec_losses = consec_losses + 1 if pnl < 0 else 0
                    if _logging:
                        trade_log.append({
                            'entry_bar': _entry_bar, 'exit_bar': i, 'direction': 'short',
                            'entry_price': pos.entry_price, 'exit_price': cover_px,
                            'shares': s, 'pnl': round(pnl, 2),
                            'return_pct': round((pos.entry_price / cover_px - 1) * 100, 3) if cover_px > 0 else 0,
                            'patterns_at_entry': _entry_patterns, 'exit_reason': 'eod_close',
                        })
                pos.position = 0
                pos.entry_price = 0.0
                pos.bars_since_exit = 0
            current_day = bar_day

        atr = _nan(arr_atr[i])
        if atr <= 0:
            atr = cp * 0.01

        # --- Overnight gap protection (universal, works regardless of adaptive_risk) ---
        _max_gap_pct = params.get('max_gap_pct', 0.0)
        if _max_gap_pct > 0 and pos.position != 0 and i > lookback:
            _prev_close = arr_close[i - 1]
            if _prev_close > 0:
                _gap_pct = (op - _prev_close) / _prev_close
                _gap_exit = False
                if pos.position > 0 and _gap_pct < -_max_gap_pct:
                    _gap_exit = True
                elif pos.position < 0 and _gap_pct > _max_gap_pct:
                    _gap_exit = True
                if _gap_exit:
                    slip = slippage_factor * atr if slippage_factor > 0 else 0.0
                    if pos.position > 0:
                        sell_px = op - slip
                        proceeds = pos.position * sell_px * (1 - commission_pct)
                        pnl = proceeds - pos.position * pos.entry_price * (1 + commission_pct)
                        cash += proceeds
                        trades.append(pnl)
                        consec_losses = consec_losses + 1 if pnl < 0 else 0
                        if _logging:
                            trade_log.append({
                                'entry_bar': _entry_bar, 'exit_bar': i, 'direction': 'long',
                                'entry_price': pos.entry_price, 'exit_price': sell_px,
                                'shares': pos.position, 'pnl': round(pnl, 2),
                                'return_pct': round((sell_px / pos.entry_price - 1) * 100, 3) if pos.entry_price > 0 else 0,
                                'patterns_at_entry': _entry_patterns, 'exit_reason': 'gap_protection',
                            })
                    else:
                        s = abs(pos.position)
                        cover_px = op + slip
                        cost = s * cover_px * (1 + commission_pct)
                        pnl = s * pos.entry_price * (1 - commission_pct) - cost
                        cash -= cost
                        trades.append(pnl)
                        consec_losses = consec_losses + 1 if pnl < 0 else 0
                        if _logging:
                            trade_log.append({
                                'entry_bar': _entry_bar, 'exit_bar': i, 'direction': 'short',
                                'entry_price': pos.entry_price, 'exit_price': cover_px,
                                'shares': s, 'pnl': round(pnl, 2),
                                'return_pct': round((pos.entry_price / cover_px - 1) * 100, 3) if cover_px > 0 else 0,
                                'patterns_at_entry': _entry_patterns, 'exit_reason': 'gap_protection',
                            })
                    pos.position = 0; pos.entry_price = 0.0; pos.bars_since_exit = 0
                    # Update equity tracking after forced exit
                    eq = cash
                    if eq > peak_equity: peak_equity = eq
                    continue

        # --- Max trade loss cap (universal, works regardless of adaptive_risk) ---
        _max_loss_pct = params.get('max_trade_loss_pct', 0.0)
        if _max_loss_pct > 0 and pos.position != 0:
            if pos.position > 0:
                _eq_now = cash + pos.position * cp
                _unr_pnl = (cp - pos.entry_price) * pos.position
            else:
                _eq_now = cash - abs(pos.position) * cp
                _unr_pnl = (pos.entry_price - cp) * abs(pos.position)
            _entry_eq = _eq_now - _unr_pnl if (_eq_now - _unr_pnl) > 0 else peak_equity
            if _unr_pnl < 0 and abs(_unr_pnl) > _entry_eq * _max_loss_pct:
                slip = slippage_factor * atr if slippage_factor > 0 else 0.0
                if pos.position > 0:
                    sell_px = cp - slip
                    proceeds = pos.position * sell_px * (1 - commission_pct)
                    pnl = proceeds - pos.position * pos.entry_price * (1 + commission_pct)
                    cash += proceeds
                    trades.append(pnl)
                    consec_losses = consec_losses + 1 if pnl < 0 else 0
                    if _logging:
                        trade_log.append({
                            'entry_bar': _entry_bar, 'exit_bar': i, 'direction': 'long',
                            'entry_price': pos.entry_price, 'exit_price': sell_px,
                            'shares': pos.position, 'pnl': round(pnl, 2),
                            'return_pct': round((sell_px / pos.entry_price - 1) * 100, 3) if pos.entry_price > 0 else 0,
                            'patterns_at_entry': _entry_patterns, 'exit_reason': 'max_loss_cap',
                        })
                else:
                    s = abs(pos.position)
                    cover_px = cp + slip
                    cost = s * cover_px * (1 + commission_pct)
                    pnl = s * pos.entry_price * (1 - commission_pct) - cost
                    cash -= cost
                    trades.append(pnl)
                    consec_losses = consec_losses + 1 if pnl < 0 else 0
                    if _logging:
                        trade_log.append({
                            'entry_bar': _entry_bar, 'exit_bar': i, 'direction': 'short',
                            'entry_price': pos.entry_price, 'exit_price': cover_px,
                            'shares': s, 'pnl': round(pnl, 2),
                            'return_pct': round((pos.entry_price / cover_px - 1) * 100, 3) if cover_px > 0 else 0,
                            'patterns_at_entry': _entry_patterns, 'exit_reason': 'max_loss_cap',
                        })
                pos.position = 0; pos.entry_price = 0.0; pos.bars_since_exit = 0
                eq = cash
                if eq > peak_equity: peak_equity = eq
                continue

        # --- Stop-price execution (before signal eval) ---
        # When stop triggered, fill at stop price, not bar close.
        # Check intrabar stop hits BEFORE evaluate_rules_signal runs.
        stop_fill_price = 0.0
        if adaptive and pos.position > 0 and lo <= pos.trailing_stop:
            # Long stop hit: fill at max(stop, open) — gap-down fills at open
            stop_fill_price = max(pos.trailing_stop, op)
        elif adaptive and pos.position < 0 and hi >= pos.trailing_stop:
            # Short stop hit: fill at min(stop, open) — gap-up fills at open
            stop_fill_price = min(pos.trailing_stop, op)

        sig = evaluate_rules_signal(params, arrays, i, pos, mkt=mkt)

        # --- Drawdown throttle ---
        if pos.position > 0:
            eq = cash + pos.position * cp
        elif pos.position < 0:
            eq = cash - abs(pos.position) * cp
        else:
            eq = cash
        current_dd = (peak_equity - eq) / peak_equity if peak_equity > 0 else 0

        # === Execute trades ===
        eff_stop_m = _effective_stop_mult(params, arrays, i)
        slip = slippage_factor * atr if slippage_factor > 0 else 0.0

        # Cooldown decay: if flat and in cooldown, count bars and gradually allow re-entry
        if adaptive and pos.position == 0 and consec_losses >= cooldown_n:
            _flat_bars_since_cooldown += 1
            if _flat_bars_since_cooldown >= cooldown_decay_bars:
                consec_losses = max(0, consec_losses - 1)
                _flat_bars_since_cooldown = 0
        elif pos.position != 0:
            _flat_bars_since_cooldown = 0

        # Skip entries if in cooldown or deep drawdown
        skip_entry = adaptive and (consec_losses >= cooldown_n or current_dd >= dd_skip)

        if sig == 1 and pos.position == 0 and not skip_entry:
            buy_px = cp + slip
            if adaptive and risk_per_trade > 0:
                # Vol-normalized sizing: risk a fixed % of equity per trade
                eff_risk = risk_per_trade
                if current_dd >= dd_half:
                    eff_risk *= 0.5  # halve risk in drawdown
                stop_dist = atr * eff_stop_m
                shares = int((eq * eff_risk) / stop_dist) if stop_dist > 0 else 0
                max_shares = int((cash * 0.95) / buy_px) if buy_px > 0 else 0
                shares = min(shares, max_shares)
            else:
                shares = int((cash * 0.95) / buy_px)
            if shares > 0:
                cash -= shares * buy_px * (1 + commission_pct)
                pos.position = shares
                pos.entry_price = buy_px
                pos.r_value = atr * eff_stop_m
                pos.trailing_stop = buy_px - pos.r_value
                pos.highest_high = hi
                if _logging:
                    _entry_bar = i
                    _entry_patterns = _snapshot_active_patterns(arrays, i)

        elif sig == 2 and pos.position == 0 and not skip_entry:
            short_px = cp - slip
            if adaptive and risk_per_trade > 0:
                eff_risk = risk_per_trade
                if current_dd >= dd_half:
                    eff_risk *= 0.5
                stop_dist = atr * eff_stop_m
                shares = int((eq * eff_risk) / stop_dist) if stop_dist > 0 else 0
                max_shares = int((cash * 0.95) / short_px) if short_px > 0 else 0
                shares = min(shares, max_shares)
            else:
                shares = int((cash * 0.95) / short_px)
            if shares > 0:
                cash += shares * short_px * (1 - commission_pct)
                pos.position = -shares
                pos.entry_price = short_px
                pos.r_value = atr * eff_stop_m
                pos.trailing_stop = short_px + pos.r_value
                pos.lowest_low = lo
                if _logging:
                    _entry_bar = i
                    _entry_patterns = _snapshot_active_patterns(arrays, i)

        elif sig == 2 and pos.position > 0:
            # Sell long — use stop price if stop triggered, else close
            if stop_fill_price > 0:
                sell_px = stop_fill_price
            else:
                sell_px = cp - slip
            proceeds = pos.position * sell_px * (1 - commission_pct)
            pnl = proceeds - pos.position * pos.entry_price * (1 + commission_pct)
            trades.append(pnl)
            cash += proceeds
            consec_losses = consec_losses + 1 if pnl < 0 else 0
            if _logging:
                trade_log.append({
                    'entry_bar': _entry_bar, 'exit_bar': i, 'direction': 'long',
                    'entry_price': pos.entry_price, 'exit_price': sell_px,
                    'shares': pos.position, 'pnl': round(pnl, 2),
                    'return_pct': round((sell_px / pos.entry_price - 1) * 100, 3) if pos.entry_price > 0 else 0,
                    'patterns_at_entry': _entry_patterns,
                    'exit_reason': 'stop' if stop_fill_price > 0 else 'signal',
                })
            pos.position = 0; pos.entry_price = 0; pos.bars_since_exit = 0

        elif sig == 1 and pos.position < 0:
            # Cover short — use stop price if stop triggered, else close
            s = abs(pos.position)
            if stop_fill_price > 0:
                cover_px = stop_fill_price
            else:
                cover_px = cp + slip
            cost = s * cover_px * (1 + commission_pct)
            pnl = s * pos.entry_price * (1 - commission_pct) - cost
            trades.append(pnl)
            cash -= cost
            consec_losses = consec_losses + 1 if pnl < 0 else 0
            if _logging:
                trade_log.append({
                    'entry_bar': _entry_bar, 'exit_bar': i, 'direction': 'short',
                    'entry_price': pos.entry_price, 'exit_price': cover_px,
                    'shares': s, 'pnl': round(pnl, 2),
                    'return_pct': round((pos.entry_price / cover_px - 1) * 100, 3) if cover_px > 0 else 0,
                    'patterns_at_entry': _entry_patterns,
                    'exit_reason': 'stop' if stop_fill_price > 0 else 'signal',
                })
            pos.position = 0; pos.entry_price = 0; pos.bars_since_exit = 0

        # Track equity and drawdown
        if pos.position > 0: eq = cash + pos.position * cp
        elif pos.position < 0: eq = cash - abs(pos.position) * cp
        else: eq = cash
        if eq > peak_equity: peak_equity = eq
        dd = (peak_equity - eq) / peak_equity if peak_equity > 0 else 0
        if dd > max_dd: max_dd = dd

    # Close remaining
    if pos.position != 0:
        cp = arr_close[-1]
        final_atr = _nan(arr_atr[-1])
        slip = slippage_factor * final_atr if slippage_factor > 0 else 0.0
        if pos.position > 0:
            sell_px = cp - slip
            pnl = pos.position * sell_px * (1 - commission_pct) - pos.position * pos.entry_price * (1 + commission_pct)
            trades.append(pnl)
            cash += pos.position * sell_px * (1 - commission_pct)
            if _logging:
                trade_log.append({
                    'entry_bar': _entry_bar, 'exit_bar': n - 1, 'direction': 'long',
                    'entry_price': pos.entry_price, 'exit_price': sell_px,
                    'shares': pos.position, 'pnl': round(pnl, 2),
                    'return_pct': round((sell_px / pos.entry_price - 1) * 100, 3) if pos.entry_price > 0 else 0,
                    'patterns_at_entry': _entry_patterns, 'exit_reason': 'end_of_data',
                })
        else:
            s = abs(pos.position)
            cover_px = cp + slip
            pnl = s * pos.entry_price * (1 - commission_pct) - s * cover_px * (1 + commission_pct)
            trades.append(pnl)
            cash -= s * cover_px * (1 + commission_pct)
            if _logging:
                trade_log.append({
                    'entry_bar': _entry_bar, 'exit_bar': n - 1, 'direction': 'short',
                    'entry_price': pos.entry_price, 'exit_price': cover_px,
                    'shares': s, 'pnl': round(pnl, 2),
                    'return_pct': round((pos.entry_price / cover_px - 1) * 100, 3) if cover_px > 0 else 0,
                    'patterns_at_entry': _entry_patterns, 'exit_reason': 'end_of_data',
                })

    final = cash
    if not trades:
        return {'return_pct': 0, 'total_pnl': 0, 'num_trades': 0, 'win_rate': 0,
                'max_drawdown': 0, 'profit_factor': 0, 'params': params}

    ret = (final / capital - 1) * 100
    wins = sum(1 for t in trades if t > 0)
    gross_p = sum(t for t in trades if t > 0)
    gross_l = abs(sum(t for t in trades if t < 0))
    worst = min(trades) if trades else 0

    return {
        'return_pct': round(ret, 2),
        'total_pnl': round(sum(trades), 2),
        'num_trades': len(trades),
        'win_rate': round(wins / len(trades) * 100, 1),
        'max_drawdown': round(max_dd * 100, 2),
        'profit_factor': round(gross_p / gross_l if gross_l > 0 else (999 if gross_p > 0 else 0), 2),
        'worst_trade': round(worst, 2),
        'params': params,
    }


app = FastAPI(title="Claude AI Backtest Dashboard")


@app.on_event("shutdown")
async def _shutdown_cleanup():
    """Clean up WebSocket connections on server shutdown."""
    try:
        # Stop paper trader's tick streamer to free Alpaca connection slot
        from claude_backtest_app import paper_trader as _pt
        if _pt._streamer:
            await _pt._streamer.stop()
            _pt._streamer = None
        _pt.stop()
    except Exception:
        pass


# =============================================================================
# BACKTEST STATE
# =============================================================================

backtest_state: Dict[str, Any] = {
    'status': 'idle',  # idle | running | complete | error
    'progress': 0,
    'message': '',
    'results': None,
    'run_id': None,
}

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
# BACKTEST RUNNER
# =============================================================================

class BacktestRunner:
    """Runs a backtest using ClaudeStrategy on historical data."""

    def __init__(self, config: dict):
        # Multi-symbol support: parse comma-separated symbols
        raw_sym = config.get('symbol', 'SPY')
        if isinstance(raw_sym, str):
            self.symbols = [s.strip().upper() for s in raw_sym.split(',') if s.strip()]
        elif isinstance(raw_sym, list):
            self.symbols = [s.strip().upper() for s in raw_sym if s.strip()]
        else:
            self.symbols = ['SPY']
        if not self.symbols:
            self.symbols = ['SPY']
        self.symbol = self.symbols[0]  # backward compat — used by single-symbol path
        # Default date range depends on interval
        default_interval = config.get('interval', '5m')
        default_days = {'1m': 10, '5m': 60, '10m': 60, '15m': 60, '30m': 120, '1d': 365}.get(default_interval, 60)
        self.start_date = config.get('start_date', (datetime.now() - timedelta(days=default_days)).strftime('%Y-%m-%d'))
        self.end_date = config.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        self.initial_capital = config.get('initial_capital', 100000)
        self.interval = config.get('interval', '5m')  # 1m, 2m, 5m, 15m, 30m, 60m, 1d
        self.commission_pct = config.get('commission_pct', 0.1) / 100  # Input as %, store as decimal
        self.mode = config.get('mode', 'fallback')  # 'fallback' or 'live_api'
        self.call_frequency = max(1, int(config.get('call_frequency', 1)))  # Call Claude every N bars
        self.trading_style = config.get('trading_style', 'swing')  # 'day' or 'swing'

        # Factor toggles
        factors = config.get('factors', {})
        self.factors = {
            'price_action': factors.get('price_action', True),
            'technical_indicators': factors.get('technical_indicators', True),
            'news_sentiment': factors.get('news_sentiment', False),
            'market_regime': factors.get('market_regime', True),
        }

        # Strategy setup
        strategy_config = StrategyConfig(
            name='Claude AI Backtest',
            enabled=True,
            weight=1.5,
            parameters={
                'factors': self.factors,
                'model': config.get('model', 'claude-3-5-haiku-latest'),
                'max_calls_per_hour': config.get('max_calls_per_hour', 5000),
                'cache_ttl_seconds': 0,
                'backtesting_mode': self.mode,
                'rules_params': config.get('rules_params', {}),
                'llm_provider': config.get('llm_provider', 'anthropic'),
                'ollama_url': config.get('ollama_url', 'http://localhost:11434'),
                'ollama_model': config.get('ollama_model', 'qwen3-coder:30b'),
            }
        )
        self.strategy = ClaudeStrategy(strategy_config)
        self.strategy.set_backtesting(True, self.mode)

        # Log LLM provider status
        llm_provider = config.get('llm_provider', 'anthropic')
        api_available = self.strategy.api_client.is_available
        if self.mode == 'live_api':
            if llm_provider == 'ollama':
                ollama_model = config.get('ollama_model', 'qwen3-coder:30b')
                logger.info(f"Live API mode: Using local Ollama ({ollama_model})")
            elif api_available:
                key = self.strategy.api_client.api_key
                logger.info(f"Live API mode: API key found ({key[:8]}...{key[-4:]})")
            else:
                logger.warning("Live API mode selected but ANTHROPIC_API_KEY is NOT SET! "
                               "Set it in the SAME terminal: export ANTHROPIC_API_KEY=sk-ant-...")

    async def _fetch_single_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data for one symbol — routes to appropriate provider."""
        df = None
        source = 'none'

        if '/' in symbol:
            # Forex: OANDA → yfinance
            df = fetch_oanda_data(symbol, self.start_date, self.end_date, self.interval)
            source = 'OANDA'
            if df is None:
                await broadcast({'type': 'progress', 'progress': 7,
                                 'message': f'OANDA unavailable for {symbol}, trying yfinance...'})
                df = fetch_yfinance_data(symbol, self.start_date, self.end_date, self.interval)
                source = 'yfinance'
        elif symbol.endswith('-USD'):
            # Crypto: Coinbase → yfinance
            df = fetch_coinbase_data(symbol, self.start_date, self.end_date, self.interval)
            source = 'Coinbase'
            if df is None:
                await broadcast({'type': 'progress', 'progress': 7,
                                 'message': f'Coinbase unavailable for {symbol}, trying yfinance...'})
                df = fetch_yfinance_data(symbol, self.start_date, self.end_date, self.interval)
                source = 'yfinance'
        else:
            # Equities: Schwab → Alpaca → yfinance
            df = fetch_schwab_data(symbol, self.start_date, self.end_date, self.interval)
            source = 'Schwab'
            if df is None:
                df = fetch_alpaca_bars(symbol, self.start_date, self.end_date, self.interval)
                source = 'Alpaca'
            if df is None:
                await broadcast({'type': 'progress', 'progress': 7,
                                 'message': f'Schwab/Alpaca unavailable for {symbol}, trying yfinance...'})
                df = fetch_yfinance_data(symbol, self.start_date, self.end_date, self.interval)
                source = 'yfinance'

        if df is None or df.empty:
            raise RuntimeError(f"No data for {symbol} ({self.interval}). "
                              f"Check data source credentials or yfinance availability.")

        await broadcast({'type': 'progress', 'progress': 10,
                         'message': f'{source}: {len(df)} {self.interval} bars for {symbol}'})
        return df

    async def fetch_data(self) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data — backward-compat wrapper for single symbol."""
        await broadcast({'type': 'progress', 'progress': 5,
                         'message': f'Fetching {self.interval} data for {self.symbol}...'})
        return await self._fetch_single_symbol(self.symbol)

    def _resolve_rules_params(self, symbol: str) -> dict:
        """Get effective rules_params for a symbol — supports per-symbol overrides.

        If rules_params contains a 'per_symbol' dict with params for this symbol,
        merge them over the base params.  Otherwise return the base params as-is.
        """
        base = self.strategy.parameters.get('rules_params', {})
        per_sym = base.get('per_symbol', {})
        if symbol in per_sym:
            merged = {k: v for k, v in base.items() if k != 'per_symbol'}
            merged.update(per_sym[symbol])
            return merged
        return {k: v for k, v in base.items() if k != 'per_symbol'}

    async def _run_single(self, symbol: str, capital_override: float = None) -> dict:
        """Run backtest for a single symbol. Returns results dict (no 'complete' broadcast)."""

        initial_capital = capital_override if capital_override is not None else self.initial_capital

        # Log config clearly
        api_key = self.strategy.api_client.api_key
        print(f"\n{'='*60}")
        print(f"  BACKTEST: {symbol} | {self.interval} | Mode: {self.mode} | Style: {self.trading_style}")
        print(f"  API Key: {'SET (' + api_key[:8] + '...)' if api_key else 'NOT SET'}")
        print(f"  Factors: {[k for k,v in self.factors.items() if v]}")
        print(f"  Call Frequency: every {self.call_frequency} bar(s)")
        print(f"{'='*60}\n")

        if self.mode == 'live_api' and not api_key:
            logger.error("LIVE API MODE BUT NO API KEY! Set ANTHROPIC_API_KEY in THIS terminal.")

        # Fetch data
        await broadcast({'type': 'progress', 'progress': 5,
                         'message': f'Fetching {self.interval} data for {symbol}...'})
        df = await self._fetch_single_symbol(symbol)
        if df is None or len(df) < 10:
            return {'error': f'Insufficient data for {symbol}'}

        # Simulation state
        cash = initial_capital
        position = 0  # Number of shares (positive=long, negative=short, zero=flat)
        entry_price = 0.0
        trades: List[dict] = []
        equity_curve: List[dict] = []
        buy_hold_curve: List[dict] = []
        signals_log: List[dict] = []

        total_bars = len(df)
        lookback = self.strategy.get_required_lookback()

        # Buy-and-hold baseline
        bh_shares = initial_capital / df['close'].iloc[0]
        bh_entry = df['close'].iloc[0]

        bars_to_process = total_bars - lookback
        if bars_to_process <= 0:
            msg = f"Not enough data: {total_bars} bars but need {lookback} for lookback. Use a longer date range."
            logger.warning(msg)
            await broadcast({'type': 'log', 'bar': 0, 'date': '-', 'price': 0,
                             'signal': 'ERROR', 'strength': 0, 'regime': msg,
                             'regime_confidence': '', 'source': ''})
            await broadcast({'type': 'progress', 'progress': 95, 'message': msg})

        api_calls_needed = max(1, bars_to_process // self.call_frequency)
        freq_msg = f" (calling every {self.call_frequency} bars = ~{api_calls_needed} API calls)" if self.call_frequency > 1 else ""
        logger.info(f"Backtest: {total_bars} bars loaded, lookback={lookback}, processing {bars_to_process} bars{freq_msg}")

        # Warn in dashboard if live_api mode but no key
        if self.mode == 'live_api' and not self.strategy.api_client.is_available:
            warn_msg = "WARNING: Live API mode but no API key! Set ANTHROPIC_API_KEY in the terminal running this app."
            await broadcast({'type': 'log', 'bar': 0, 'date': '-', 'price': 0,
                             'signal': 'ERROR', 'strength': 0, 'regime': warn_msg,
                             'regime_confidence': '', 'source': ''})
        freq_ui_msg = f', calling every {self.call_frequency} bars (~{api_calls_needed} calls)' if self.call_frequency > 1 else ''
        await broadcast({'type': 'progress', 'progress': 15,
                         'message': f'Processing {bars_to_process} bars{freq_ui_msg}...'})

        # Force-close positions at end of day only in day-trading mode with intraday intervals
        force_eod_close = self.trading_style == 'day' and self.interval not in ('1d', '1wk', '1mo')
        current_trading_day = None
        entry_day = None  # Track which day the position was opened

        # Throttle broadcasts for large datasets — target ~2000 chart points max
        chart_every = max(1, bars_to_process // 2000)
        # Throttle decision/stats broadcasts similarly (cap DOM elements)
        decision_every = max(1, bars_to_process // 500)
        # Real sleep every N chart broadcasts to flush WebSocket data to the browser.
        # Without this, rules mode completes in <1s and the browser sees nothing until done.
        # Target ~60 flushes total → smooth streaming without excessive overhead.
        flush_every = max(1, (bars_to_process // chart_every) // 60)

        # Pre-compute all indicators once on the full DataFrame — O(n) instead of O(n^2)
        is_rules_mode = self.mode == 'rules'
        _pa = None      # PrecomputedArrays for rules mode
        _pos = None     # PositionState for rules mode
        _rules_p = {}   # rules_params for rules mode
        if is_rules_mode:
            await broadcast({'type': 'progress', 'progress': 12,
                             'message': f'Pre-computing indicators on {total_bars} bars...'})
            # Use the modern evaluate_rules_signal() engine (same as optimizer/paper trader)
            # instead of the legacy _rules_analyze() in claude_strategy.py
            _pa = precompute_arrays(df, self.trading_style, self.interval, lookback)
            _pos = PositionState()
            _rules_p = self._resolve_rules_params(symbol)
            # Adaptive risk tracking (match quick_backtest behavior)
            _adaptive = _rules_p.get('adaptive_risk', False)
            _cooldown_n = _rules_p.get('cooldown_losses', 3) if _adaptive else 999
            _cooldown_decay_bars = _rules_p.get('reentry_cooldown_bars', 5) * 6 if _adaptive else 9999
            _dd_half = _rules_p.get('drawdown_half_risk', 0.05) if _adaptive else 1.0
            _dd_skip = _rules_p.get('drawdown_skip', 0.10) if _adaptive else 1.0
            _consec_losses = 0
            _flat_bars_since_cooldown = 0
            _peak_equity = initial_capital
            await broadcast({'type': 'progress', 'progress': 15,
                             'message': f'Processing {bars_to_process} bars{freq_ui_msg}...'})

        chart_broadcast_count = 0
        for i in range(lookback, total_bars):
            current_price = float(df['close'].iloc[i])
            bar_date = str(df.index[i])

            # Detect trading day from bar timestamp
            try:
                bar_dt = df.index[i]
                bar_day = bar_dt.date() if hasattr(bar_dt, 'date') else str(bar_dt)[:10]
            except Exception:
                bar_day = bar_date[:10]

            # Day trading: force close position if new day started and we have an overnight position
            if force_eod_close and current_trading_day is not None and bar_day != current_trading_day and position != 0:
                close_price = float(df['close'].iloc[i - 1]) if i > lookback else current_price
                if position > 0:  # Close long
                    proceeds = position * close_price * (1 - self.commission_pct)
                    pnl = proceeds - (position * entry_price * (1 + self.commission_pct))
                    cash += proceeds
                    trades.append({
                        'date': f'{current_trading_day} EOD',
                        'symbol': symbol,
                        'type': 'EOD_CLOSE',
                        'price': close_price,
                        'shares': position,
                        'proceeds': proceeds,
                        'pnl': pnl,
                        'return_pct': ((close_price / entry_price) - 1) * 100,
                        'strength': 0,
                        'reasoning': 'Day trading: forced close at end of day',
                        'risk_notes': 'Position closed to avoid overnight risk',
                        'factors_used': [],
                        'raw_response': {},
                        'indicators': {},
                    })
                    await broadcast({'type': 'trade', 'trade': trades[-1]})
                elif position < 0:  # Cover short
                    shares = abs(position)
                    cost = shares * close_price * (1 + self.commission_pct)
                    entry_proceeds = shares * entry_price * (1 - self.commission_pct)
                    pnl = entry_proceeds - cost
                    cash -= cost
                    trades.append({
                        'date': f'{current_trading_day} EOD',
                        'symbol': symbol,
                        'type': 'EOD_COVER',
                        'price': close_price,
                        'shares': shares,
                        'cost': cost,
                        'pnl': pnl,
                        'return_pct': ((entry_price / close_price) - 1) * 100,
                        'strength': 0,
                        'reasoning': 'Day trading: forced cover at end of day',
                        'risk_notes': 'Short covered to avoid overnight risk',
                        'factors_used': [],
                        'raw_response': {},
                        'indicators': {},
                    })
                    await broadcast({'type': 'trade', 'trade': trades[-1]})
                position = 0
                entry_price = 0
                entry_day = None
                # Sync PositionState after EOD close (rules mode)
                if is_rules_mode and _pos is not None and _pos.position != 0:
                    _pos.position = 0; _pos.entry_price = 0.0; _pos.bars_since_exit = 0

            current_trading_day = bar_day

            # --- Overnight gap protection (universal) ---
            _max_gap_pct = _rules_p.get('max_gap_pct', 0.0) if is_rules_mode else self.strategy.parameters.get('rules_params', {}).get('max_gap_pct', 0.0)
            if _max_gap_pct > 0 and position != 0 and i > lookback:
                _prev_close = float(df['close'].iloc[i - 1])
                _open_px = float(df['open'].iloc[i])
                if _prev_close > 0:
                    _gap_pct = (_open_px - _prev_close) / _prev_close
                    _gap_exit = False
                    if position > 0 and _gap_pct < -_max_gap_pct:
                        _gap_exit = True
                    elif position < 0 and _gap_pct > _max_gap_pct:
                        _gap_exit = True
                    if _gap_exit:
                        if position > 0:
                            proceeds = position * _open_px * (1 - self.commission_pct)
                            pnl = proceeds - (position * entry_price * (1 + self.commission_pct))
                            cash += proceeds
                            trades.append({
                                'date': bar_date, 'symbol': symbol, 'type': 'SELL',
                                'price': _open_px, 'shares': position, 'proceeds': proceeds,
                                'pnl': pnl, 'return_pct': ((_open_px / entry_price) - 1) * 100 if entry_price > 0 else 0,
                                'strength': 0, 'reasoning': f'Gap protection: {_gap_pct*100:+.1f}% gap exceeded {_max_gap_pct*100:.1f}% threshold',
                                'risk_notes': 'Forced exit at open price due to adverse overnight gap',
                                'factors_used': [], 'raw_response': {}, 'indicators': {},
                            })
                        else:
                            shares = abs(position)
                            cost = shares * _open_px * (1 + self.commission_pct)
                            entry_proceeds = shares * entry_price * (1 - self.commission_pct)
                            pnl = entry_proceeds - cost
                            cash -= cost
                            trades.append({
                                'date': bar_date, 'symbol': symbol, 'type': 'COVER',
                                'price': _open_px, 'shares': shares, 'cost': cost,
                                'pnl': pnl, 'return_pct': ((entry_price / _open_px) - 1) * 100 if _open_px > 0 else 0,
                                'strength': 0, 'reasoning': f'Gap protection: {_gap_pct*100:+.1f}% gap exceeded {_max_gap_pct*100:.1f}% threshold',
                                'risk_notes': 'Forced exit at open price due to adverse overnight gap',
                                'factors_used': [], 'raw_response': {}, 'indicators': {},
                            })
                        await broadcast({'type': 'trade', 'trade': trades[-1]})
                        position = 0; entry_price = 0; entry_day = None
                        if is_rules_mode and _pos is not None:
                            _pos.position = 0; _pos.entry_price = 0.0; _pos.bars_since_exit = 0
                        continue

            # --- Max trade loss cap (universal) ---
            _max_loss_pct = _rules_p.get('max_trade_loss_pct', 0.0) if is_rules_mode else self.strategy.parameters.get('rules_params', {}).get('max_trade_loss_pct', 0.0)
            if _max_loss_pct > 0 and position != 0:
                if position > 0:
                    _unr_pnl = (current_price - entry_price) * position
                    _eq_now = cash + position * current_price
                else:
                    _unr_pnl = (entry_price - current_price) * abs(position)
                    _eq_now = cash - abs(position) * current_price
                _entry_eq = _eq_now - _unr_pnl if (_eq_now - _unr_pnl) > 0 else initial_capital
                if _unr_pnl < 0 and abs(_unr_pnl) > _entry_eq * _max_loss_pct:
                    if position > 0:
                        proceeds = position * current_price * (1 - self.commission_pct)
                        pnl = proceeds - (position * entry_price * (1 + self.commission_pct))
                        cash += proceeds
                        trades.append({
                            'date': bar_date, 'symbol': symbol, 'type': 'SELL',
                            'price': current_price, 'shares': position, 'proceeds': proceeds,
                            'pnl': pnl, 'return_pct': ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0,
                            'strength': 0, 'reasoning': f'Max loss cap: unrealized loss ${abs(_unr_pnl):.0f} exceeded {_max_loss_pct*100:.1f}% of equity',
                            'risk_notes': 'Forced exit — trade loss exceeded maximum allowed per-trade loss',
                            'factors_used': [], 'raw_response': {}, 'indicators': {},
                        })
                    else:
                        shares = abs(position)
                        cost = shares * current_price * (1 + self.commission_pct)
                        entry_proceeds = shares * entry_price * (1 - self.commission_pct)
                        pnl = entry_proceeds - cost
                        cash -= cost
                        trades.append({
                            'date': bar_date, 'symbol': symbol, 'type': 'COVER',
                            'price': current_price, 'shares': shares, 'cost': cost,
                            'pnl': pnl, 'return_pct': ((entry_price / current_price) - 1) * 100 if current_price > 0 else 0,
                            'strength': 0, 'reasoning': f'Max loss cap: unrealized loss ${abs(_unr_pnl):.0f} exceeded {_max_loss_pct*100:.1f}% of equity',
                            'risk_notes': 'Forced exit — trade loss exceeded maximum allowed per-trade loss',
                            'factors_used': [], 'raw_response': {}, 'indicators': {},
                        })
                    await broadcast({'type': 'trade', 'trade': trades[-1]})
                    position = 0; entry_price = 0; entry_day = None
                    if is_rules_mode and _pos is not None:
                        _pos.position = 0; _pos.entry_price = 0.0; _pos.bars_since_exit = 0
                    continue

            # Get indicators for this bar (pre-computed for rules mode, calculated otherwise)
            if is_rules_mode:
                indicators = self._get_indicators_at(df, i)
            else:
                indicators = self._calc_indicators(df.iloc[:i + 1])

            # Build current positions dict (works for both long and short)
            current_positions = {}
            if position != 0:
                if position > 0:  # Long position
                    unrealized_pnl = (current_price - entry_price) * position
                else:  # Short position (position < 0)
                    unrealized_pnl = (entry_price - current_price) * abs(position)
                current_positions[symbol] = {
                    'entry_price': entry_price,
                    'size': abs(position),
                    'side': 'long' if position > 0 else 'short',
                    'pnl': unrealized_pnl,
                }

            # Decide whether to call Claude on this bar
            bar_index = i - lookback  # 0-based within processing range
            is_decision_bar = (self.call_frequency <= 1 or
                               bar_index % self.call_frequency == 0 or
                               i == total_bars - 1)  # Always call on last bar

            if is_decision_bar:
                if is_rules_mode and _pa is not None:
                    # Modern rules engine — same engine as optimizer & paper trader
                    try:
                        sig_int = evaluate_rules_signal(_rules_p, _pa, i, _pos)
                        reasoning = describe_signal_reasoning(_pa, i, _rules_p)
                        if sig_int == 1:
                            sig_type = SignalType.BUY
                            _base_trig = _rules_p.get('min_triggers', 2)
                            _excess = max(0, _pos.last_score - _base_trig)
                            strength = round(min(0.95, 0.5 + 0.15 * _excess), 2)
                        elif sig_int == 2:
                            sig_type = SignalType.SELL
                            _base_trig = _rules_p.get('min_triggers', 2)
                            _excess = max(0, _pos.last_score - _base_trig)
                            strength = round(min(0.95, 0.5 + 0.15 * _excess), 2)
                        else:
                            sig_type = SignalType.HOLD
                            strength = 0.3
                        signal = StrategySignal(
                            symbol=symbol,
                            signal_type=sig_type,
                            strength=strength,
                            strategy_name='rules_engine_v4',
                            timestamp=datetime.now(),
                            entry_price=current_price if sig_int in (1, 2) else None,
                            metadata={
                                'reasoning': reasoning,
                                'risk_notes': reasoning,
                                'source': 'rules_engine_v4',
                                'backtest_mode': 'rules',
                                'factors_used': ['price_action', 'technical_indicators', 'market_regime'],
                                'raw_response': {'signal': sig_type.value, 'strength': strength},
                            }
                        )
                        signals = [signal] if sig_int != 0 else [signal]
                    except Exception as e:
                        logger.error(f"Rules engine error at bar {i}: {e}")
                        signal = None
                        signals = []
                else:
                    # Live API or fallback mode — use ClaudeStrategy
                    try:
                        bar_data = df.iloc[:i + 1].copy()
                        signals = await self.strategy.analyze_async(bar_data, current_positions)
                        if self.mode == 'live_api':
                            await asyncio.sleep(0.3)
                    except Exception as e:
                        logger.error(f"Strategy error at bar {i}: {e}")
                        signals = []
                    signal = signals[0] if signals else None
            else:
                # Skip this bar - no API call, just carry forward
                signal = None
                signals = []

            signal_entry = {
                'bar': i,
                'date': bar_date,
                'price': current_price,
                'signal': signal.signal_type.value if signal else 'NONE',
                'strength': signal.strength if signal else 0,
                'reasoning': signal.metadata.get('reasoning', '') if signal else '',
                'risk_notes': signal.metadata.get('risk_notes', '') if signal else '',
                'factors_used': signal.metadata.get('factors_used', []) if signal else [],
                'source': signal.metadata.get('source', '') if signal else '',
                'prompt_sent': signal.metadata.get('prompt_sent', '') if signal else '',
                'raw_response': signal.metadata.get('raw_response', {}) if signal else {},
                'indicators': indicators,
            }
            signals_log.append(signal_entry)

            # Broadcast per-bar log with indicators
            ind_str = (f"RSI:{indicators.get('rsi','?')} "
                       f"MACD:{indicators.get('macd_hist','?')} "
                       f"BB:{indicators.get('bb_pct','?')} "
                       f"ADX:{indicators.get('adx','?')}")

            if is_decision_bar:
                sig_type = signal.signal_type.value if signal else 'HOLD'
                is_actionable = sig_type not in ('HOLD', 'NONE')
                # Throttle decision cards: always for actionable signals, otherwise every Nth bar
                should_broadcast_decision = is_actionable or bar_index % decision_every == 0

                if should_broadcast_decision:
                    reasoning = signal.metadata.get('reasoning', '') if signal else ''
                    risk_notes = signal.metadata.get('risk_notes', '') if signal else ''
                    source = signal.metadata.get('source', '') if signal else ''
                    strength = round(signal.strength, 2) if signal else 0
                    regime_info = signal.metadata.get('regime', '') if signal else ''
                    regime_conf = signal.metadata.get('regime_confidence', '') if signal else ''
                    pos_info = ''
                    if position > 0:
                        unrealized = round((current_price - entry_price) / entry_price * 100, 2)
                        pos_info = f'LONG {position} shares @ ${entry_price:.2f} ({unrealized:+.2f}%)'
                    elif position < 0:
                        unrealized = round((entry_price - current_price) / entry_price * 100, 2)
                        pos_info = f'SHORT {abs(position)} shares @ ${entry_price:.2f} ({unrealized:+.2f}%)'
                    else:
                        pos_info = 'FLAT'

                    # Build clear action description from signal + position
                    is_buy_sig = sig_type == 'BUY'
                    is_sell_sig = sig_type in ('SELL', 'CLOSE_LONG', 'CLOSE_SHORT')
                    if sig_type == 'HOLD':
                        if position > 0:
                            action = f'HOLDING LONG {position} shares'
                        elif position < 0:
                            action = f'HOLDING SHORT {abs(position)} shares'
                        else:
                            action = 'STAYING FLAT — no trade'
                    elif is_buy_sig and position == 0:
                        action = 'OPENING LONG position'
                    elif is_buy_sig and position < 0:
                        action = f'COVERING SHORT — closing {abs(position)} shares for {"profit" if entry_price > current_price else "loss"}'
                    elif is_buy_sig and position > 0:
                        action = f'HOLDING LONG — already long {position} shares'
                    elif is_sell_sig and position == 0:
                        action = 'OPENING SHORT position'
                    elif is_sell_sig and position > 0:
                        action = f'CLOSING LONG — selling {position} shares for {"profit" if current_price > entry_price else "loss"}'
                    elif is_sell_sig and position < 0:
                        action = f'HOLDING SHORT — already short {abs(position)} shares'
                    else:
                        action = sig_type

                    await broadcast({
                        'type': 'decision',
                        'bar': i - lookback + 1,
                        'total_bars': bars_to_process,
                        'date': bar_date,
                        'price': round(current_price, 2),
                        'signal': sig_type,
                        'strength': strength,
                        'reasoning': reasoning,
                        'risk_notes': risk_notes,
                        'regime': regime_info,
                        'regime_confidence': round(regime_conf, 2) if isinstance(regime_conf, (int, float)) else '',
                        'source': source,
                        'indicators': ind_str,
                        'position': pos_info,
                        'action': action,
                    })
            elif i % max(1, (total_bars - lookback) // 20) == 0:
                # Heartbeat for skipped bars (just price tracking)
                await broadcast({
                    'type': 'log',
                    'bar': i - lookback + 1,
                    'date': bar_date.split(' ')[0] if ' ' in bar_date else bar_date[:10],
                    'price': round(current_price, 2),
                    'signal': 'SKIP',
                    'strength': 0,
                    'indicators': ind_str,
                })

            # Execute trades with short selling support
            is_buy = signal and signal.signal_type == SignalType.BUY
            is_sell = signal and signal.signal_type in (SignalType.SELL, SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT)

            # Adaptive position sizing helper
            _rp = self.strategy.parameters.get('rules_params', {})
            _bt_adaptive = _rp.get('adaptive_risk', False)
            def _size_shares(price, eq_override=None):
                if not _bt_adaptive or not _rp.get('risk_per_trade'):
                    return int((cash * 0.95) / price) if price > 0 else 0
                _atr_val = indicators.get('atr', current_price * 0.01)
                if isinstance(_atr_val, str):
                    try: _atr_val = float(_atr_val)
                    except: _atr_val = current_price * 0.01
                _esm = _effective_stop_mult(_rules_p, _pa, i) if (is_rules_mode and _pa is not None) else _rp.get('atr_stop_mult', 2.0)
                _max_pct = _rp.get('max_stop_pct', 0.04)
                _stop_dist = _atr_val * _esm
                _max_dist = price * _max_pct
                if _stop_dist > _max_dist:
                    _stop_dist = _max_dist
                if _stop_dist <= 0:
                    return 0
                _eq = eq_override if eq_override is not None else cash + (position * current_price if position > 0 else (-(abs(position) * current_price) if position < 0 else 0))
                # Drawdown-adjusted risk (rules mode)
                _eff_risk = _rp.get('risk_per_trade', 0.02)
                if is_rules_mode and _adaptive and _peak_equity > 0:
                    _cur_dd = (_peak_equity - _eq) / _peak_equity
                    if _cur_dd >= _dd_half:
                        _eff_risk *= 0.5
                _risk_amt = _eq * _eff_risk
                sized = int(_risk_amt / _stop_dist)
                max_s = int((cash * 0.95) / price) if price > 0 else 0
                return min(sized, max_s)

            # Cooldown decay: if flat and in cooldown, count bars and allow re-entry
            if is_rules_mode and _adaptive and position == 0 and _consec_losses >= _cooldown_n:
                _flat_bars_since_cooldown += 1
                if _flat_bars_since_cooldown >= _cooldown_decay_bars:
                    _consec_losses = max(0, _consec_losses - 1)
                    _flat_bars_since_cooldown = 0
            elif is_rules_mode and position != 0:
                _flat_bars_since_cooldown = 0

            # Rules mode: skip entries on cooldown / deep drawdown
            _skip_entry = False
            if is_rules_mode and _adaptive:
                _eq_now = cash + (position * current_price if position > 0 else (-(abs(position) * current_price) if position < 0 else 0))
                _cur_dd_pct = (_peak_equity - _eq_now) / _peak_equity if _peak_equity > 0 else 0
                _skip_entry = _consec_losses >= _cooldown_n or _cur_dd_pct >= _dd_skip

            # Case 1: BUY when flat → Enter LONG
            if is_buy and position == 0 and not _skip_entry:
                shares = _size_shares(current_price)
                if shares > 0:
                    cost = shares * current_price * (1 + self.commission_pct)
                    cash -= cost
                    position = shares  # Positive = long
                    entry_price = current_price
                    entry_day = bar_day
                    trades.append({
                        'date': bar_date,
                        'symbol': symbol,
                        'type': 'BUY',
                        'price': current_price,
                        'shares': shares,
                        'cost': cost,
                        'pnl': 0,
                        'strength': signal.strength,
                        'reasoning': signal.metadata.get('reasoning', ''),
                        'risk_notes': signal.metadata.get('risk_notes', ''),
                        'factors_used': signal.metadata.get('factors_used', []),
                        'raw_response': signal.metadata.get('raw_response', {}),
                        'indicators': indicators,
                    })
                    await broadcast({'type': 'trade', 'trade': trades[-1]})

            # Case 2: SELL when flat → Enter SHORT
            elif is_sell and position == 0 and not _skip_entry:
                shares = _size_shares(current_price)
                if shares > 0:
                    proceeds = shares * current_price * (1 - self.commission_pct)
                    cash += proceeds
                    position = -shares  # Negative = short
                    entry_price = current_price
                    entry_day = bar_day
                    trades.append({
                        'date': bar_date,
                        'symbol': symbol,
                        'type': 'SHORT',
                        'price': current_price,
                        'shares': shares,
                        'proceeds': proceeds,
                        'pnl': 0,
                        'strength': signal.strength,
                        'reasoning': signal.metadata.get('reasoning', ''),
                        'risk_notes': signal.metadata.get('risk_notes', ''),
                        'factors_used': signal.metadata.get('factors_used', []),
                        'raw_response': signal.metadata.get('raw_response', {}),
                        'indicators': indicators,
                    })
                    await broadcast({'type': 'trade', 'trade': trades[-1]})

            # Case 3: SELL when long → Exit LONG
            elif is_sell and position > 0:
                proceeds = position * current_price * (1 - self.commission_pct)
                pnl = proceeds - (position * entry_price * (1 + self.commission_pct))
                cash += proceeds
                trades.append({
                    'date': bar_date,
                    'symbol': symbol,
                    'type': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'proceeds': proceeds,
                    'pnl': pnl,
                    'return_pct': ((current_price / entry_price) - 1) * 100,
                    'strength': signal.strength,
                    'reasoning': signal.metadata.get('reasoning', ''),
                    'risk_notes': signal.metadata.get('risk_notes', ''),
                    'factors_used': signal.metadata.get('factors_used', []),
                    'raw_response': signal.metadata.get('raw_response', {}),
                })
                position = 0
                entry_price = 0
                entry_day = None
                await broadcast({'type': 'trade', 'trade': trades[-1]})

            # Case 4: BUY when short → Cover SHORT
            elif is_buy and position < 0:
                shares = abs(position)
                cost = shares * current_price * (1 + self.commission_pct)
                entry_proceeds = shares * entry_price * (1 - self.commission_pct)
                pnl = entry_proceeds - cost
                cash -= cost
                trades.append({
                    'date': bar_date,
                    'symbol': symbol,
                    'type': 'COVER',
                    'price': current_price,
                    'shares': shares,
                    'cost': cost,
                    'pnl': pnl,
                    'return_pct': ((entry_price / current_price) - 1) * 100,  # Reversed for shorts
                    'strength': signal.strength,
                    'reasoning': signal.metadata.get('reasoning', ''),
                    'risk_notes': signal.metadata.get('risk_notes', ''),
                    'factors_used': signal.metadata.get('factors_used', []),
                    'raw_response': signal.metadata.get('raw_response', {}),
                })
                position = 0
                entry_price = 0
                entry_day = None
                await broadcast({'type': 'trade', 'trade': trades[-1]})

            # Sync PositionState with BacktestRunner after trade execution (rules mode)
            if is_rules_mode and _pos is not None:
                if position != 0 and _pos.position == 0:
                    # Entry just happened — initialize PositionState
                    _atr_v = _nan(_pa.atr[i]) if _pa is not None else current_price * 0.01
                    if _atr_v <= 0: _atr_v = current_price * 0.01
                    _esm = _effective_stop_mult(_rules_p, _pa, i)
                    _pos.position = position
                    _pos.entry_price = current_price
                    _pos.r_value = _atr_v * _esm
                    if position > 0:
                        _pos.trailing_stop = current_price - _pos.r_value
                        _pos.highest_high = float(df['high'].iloc[i])
                    else:
                        _pos.trailing_stop = current_price + _pos.r_value
                        _pos.lowest_low = float(df['low'].iloc[i])
                elif position == 0 and _pos.position != 0:
                    # Exit just happened — update cooldown tracking + reset PositionState
                    last_pnl = trades[-1].get('pnl', 0) if trades else 0
                    _consec_losses = _consec_losses + 1 if last_pnl < 0 else 0
                    _pos.position = 0
                    _pos.entry_price = 0.0
                    _pos.bars_since_exit = 0

            # Track peak equity for drawdown throttle (rules mode)
            if is_rules_mode and _adaptive:
                _eq_track = cash + (position * current_price if position > 0 else (-(abs(position) * current_price) if position < 0 else 0))
                if _eq_track > _peak_equity:
                    _peak_equity = _eq_track

            # Record equity (handle both long and short positions)
            if position > 0:  # Long position
                portfolio_value = cash + (position * current_price)
                unrealized_pnl_val = (current_price - entry_price) * position
            elif position < 0:  # Short position
                # Cash already includes short sale proceeds, so subtract cost to cover
                shares_abs = abs(position)
                portfolio_value = cash - (shares_abs * current_price)
                unrealized_pnl_val = (entry_price - current_price) * shares_abs
            else:  # Flat
                portfolio_value = cash
                unrealized_pnl_val = 0
            equity_curve.append({
                'date': bar_date,
                'value': portfolio_value,
            })
            buy_hold_curve.append({
                'date': bar_date,
                'value': bh_shares * current_price,
            })

            # Broadcast bar data for live chart (throttled: ~2000 points max)
            if bar_index % chart_every == 0 or i == total_bars - 1:
                bar_msg = {
                    'type': 'bar',
                    'symbol': symbol,
                    'date': bar_date,
                    'open': float(df['open'].iloc[i]),
                    'high': float(df['high'].iloc[i]),
                    'low': float(df['low'].iloc[i]),
                    'close': current_price,
                    'volume': float(df['volume'].iloc[i]) if 'volume' in df.columns else 0,
                    'equity': portfolio_value,
                    'buy_hold': bh_shares * current_price,
                }
                # Attach detected patterns and market structure (rules mode only)
                if is_rules_mode and _pa is not None:
                    pats = _get_patterns_at_bar(_pa, i)
                    if pats:
                        bar_msg['patterns'] = pats
                    sb, st = _market_structure_bias(_pa, i)
                    if st != 'insufficient_data':
                        bar_msg['structure'] = {'bias': sb, 'type': st}
                await broadcast(bar_msg)
                chart_broadcast_count += 1
                # Real sleep periodically to flush WebSocket data to browser.
                # sleep(0) yields the event loop; sleep(0.005) actually lets uvicorn
                # push buffered frames over the wire so the UI updates in real-time.
                if chart_broadcast_count % flush_every == 0:
                    await asyncio.sleep(0.005)
                else:
                    await asyncio.sleep(0)

            # Track peak and drawdown
            peak_value = max(e['value'] for e in equity_curve)
            drawdown_pct = ((portfolio_value - peak_value) / peak_value * 100) if peak_value > 0 else 0

            # Calculate running win rate from closed trades
            closed_trades = [t for t in trades if 'pnl' in t and t.get('pnl', 0) != 0]
            winning = sum(1 for t in closed_trades if t['pnl'] > 0)
            win_rate = (winning / len(closed_trades) * 100) if closed_trades else 0

            # Broadcast live stats (throttled to ~200 updates)
            stats_every = max(1, bars_to_process // 200)
            if bar_index % stats_every == 0 or i == total_bars - 1:
                # Position info for stats bar
                if position > 0:
                    pos_side = 'long'
                    pos_text = f'LONG {position}'
                elif position < 0:
                    pos_side = 'short'
                    pos_text = f'SHORT {abs(position)}'
                else:
                    pos_side = 'flat'
                    pos_text = 'FLAT'

                total_pnl = portfolio_value - initial_capital
                total_return = (total_pnl / initial_capital) * 100

                await broadcast({
                    'type': 'stats',
                    'capital': round(portfolio_value, 2),
                    'pnl': round(total_pnl, 2),
                    'return_pct': round(total_return, 2),
                    'position_side': pos_side,
                    'position_text': pos_text,
                    'unrealized_pnl': round(unrealized_pnl_val, 2),
                    'num_trades': len(closed_trades),
                    'win_rate': round(win_rate, 1),
                    'peak': round(peak_value, 2),
                    'drawdown_pct': round(drawdown_pct, 2),
                    'bar': i - lookback + 1,
                    'total_bars': bars_to_process,
                    'price': round(current_price, 2),
                })

            # Progress update
            progress = 15 + int((i - lookback) / (total_bars - lookback) * 80)
            if i % max(1, (total_bars - lookback) // 20) == 0:
                await broadcast({
                    'type': 'progress',
                    'progress': progress,
                    'message': f'Processing bar {i - lookback + 1}/{total_bars - lookback}...',
                })

            # Note: event loop yielding is handled in the chart broadcast section above

        # Close any remaining position (handle both long and short)
        if position > 0:  # Close long position
            current_price = float(df['close'].iloc[-1])
            proceeds = position * current_price * (1 - self.commission_pct)
            pnl = proceeds - (position * entry_price * (1 + self.commission_pct))
            cash += proceeds
            trades.append({
                'date': str(df.index[-1]),
                'symbol': symbol,
                'type': 'CLOSE',
                'price': current_price,
                'shares': position,
                'proceeds': proceeds,
                'pnl': pnl,
                'return_pct': ((current_price / entry_price) - 1) * 100,
                'strength': 0,
                'reasoning': 'End of backtest - closing long position',
                'risk_notes': '',
                'factors_used': [],
                'raw_response': {},
                'indicators': {},
            })
            position = 0
        elif position < 0:  # Cover short position
            current_price = float(df['close'].iloc[-1])
            shares = abs(position)
            cost = shares * current_price * (1 + self.commission_pct)
            entry_proceeds = shares * entry_price * (1 - self.commission_pct)
            pnl = entry_proceeds - cost
            cash -= cost
            trades.append({
                'date': str(df.index[-1]),
                'symbol': symbol,
                'type': 'CLOSE_SHORT',
                'price': current_price,
                'shares': shares,
                'cost': cost,
                'pnl': pnl,
                'return_pct': ((entry_price / current_price) - 1) * 100,
                'strength': 0,
                'reasoning': 'End of backtest - covering short position',
                'risk_notes': '',
                'factors_used': [],
                'raw_response': {},
                'indicators': {},
            })
            position = 0

        # Signal summary
        signal_counts = {}
        for s in signals_log:
            sig_type = s['signal']
            signal_counts[sig_type] = signal_counts.get(sig_type, 0) + 1

        summary_msg = f"Processed {len(signals_log)} bars. Signals: " + ", ".join(
            f"{k}={v}" for k, v in sorted(signal_counts.items())
        ) + f". Trades: {len(trades)}."
        logger.info(summary_msg)
        await broadcast({'type': 'log', 'bar': 'SUMMARY', 'date': '-', 'price': 0,
                         'signal': 'SUMMARY', 'strength': 0,
                         'regime': summary_msg, 'regime_confidence': '', 'source': ''})

        await broadcast({'type': 'progress', 'progress': 95, 'message': f'Calculating metrics... ({summary_msg})'})

        # Calculate metrics — use last equity curve value (accounts for open positions)
        final_value = equity_curve[-1]['value'] if equity_curve else cash
        metrics = self._calculate_metrics(equity_curve, trades, initial_capital, final_value, bh_entry, float(df['close'].iloc[-1]))

        results = {
            'symbol': symbol,
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_curve,
            'buy_hold_curve': buy_hold_curve,
            'signals_log': signals_log,
            'config': {
                'symbol': symbol,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'interval': self.interval,
                'initial_capital': initial_capital,
                'mode': self.mode,
                'factors': self.factors,
            },
            'api_stats': self.strategy.get_api_stats(),
        }

        return results

    @staticmethod
    def _aggregate_equity_curves(per_symbol: dict, capital_per: float) -> list:
        """Merge per-symbol equity curves into a combined portfolio curve.

        Args:
            per_symbol: {symbol: [{'date': ..., 'value': ...}, ...]}
            capital_per: starting capital per symbol (for forward-fill)

        Returns: combined equity curve sorted by date
        """
        from collections import defaultdict
        # Collect all dates → sum of values (forward-fill gaps)
        date_values = defaultdict(float)
        sym_last = {}
        all_dates = set()
        for sym, curve in per_symbol.items():
            for pt in curve:
                all_dates.add(pt['date'])
        all_dates = sorted(all_dates)

        for sym, curve in per_symbol.items():
            sym_map = {pt['date']: pt['value'] for pt in curve}
            last_val = capital_per
            for d in all_dates:
                if d in sym_map:
                    last_val = sym_map[d]
                date_values[d] += last_val

        return [{'date': d, 'value': date_values[d]} for d in all_dates]

    async def run(self) -> dict:
        """Run backtest — orchestrates single or multi-symbol execution."""
        global backtest_state

        # Single symbol: fast path (backward compatible)
        if len(self.symbols) == 1:
            result = await self._run_single(self.symbols[0])
            if 'error' in result:
                return result
            await broadcast({'type': 'progress', 'progress': 100, 'message': 'Complete!'})
            await broadcast({'type': 'complete', 'metrics': result['metrics']})
            self._save_backtest_report(result)
            return result

        # Multi-symbol: sequential execution with equal capital allocation
        n_syms = len(self.symbols)
        capital_per = self.initial_capital / n_syms
        all_trades = []
        per_sym_equity = {}
        per_sym_bh = {}
        per_sym_metrics = {}
        per_sym_signals = []
        errors = []

        for idx, sym in enumerate(self.symbols):
            pct_base = int(5 + (idx / n_syms) * 85)
            await broadcast({'type': 'progress', 'progress': pct_base,
                             'message': f'Running {sym} ({idx+1}/{n_syms})...'})
            try:
                result = await self._run_single(sym, capital_override=capital_per)
                if 'error' in result:
                    errors.append(f"{sym}: {result['error']}")
                    continue
                all_trades.extend(result['trades'])
                per_sym_equity[sym] = result['equity_curve']
                per_sym_bh[sym] = result['buy_hold_curve']
                per_sym_metrics[sym] = result['metrics']
                per_sym_signals.extend(result.get('signals_log', []))
            except Exception as e:
                logger.error(f"Multi-symbol backtest error for {sym}: {e}")
                errors.append(f"{sym}: {e}")

        if not per_sym_equity:
            return {'error': f'All symbols failed: {"; ".join(errors)}'}

        # Sort trades chronologically across symbols
        all_trades.sort(key=lambda t: t.get('date', ''))

        # Aggregate equity curves
        combined_eq = self._aggregate_equity_curves(per_sym_equity, capital_per)
        combined_bh = self._aggregate_equity_curves(per_sym_bh, capital_per)

        # Aggregate metrics
        final_value = combined_eq[-1]['value'] if combined_eq else self.initial_capital
        bh_final = combined_bh[-1]['value'] if combined_bh else self.initial_capital
        bh_initial = self.initial_capital  # all symbols start at full capital in B&H
        combined_metrics = self._calculate_metrics(
            combined_eq, all_trades, self.initial_capital, final_value,
            bh_initial, bh_final
        )

        results = {
            'metrics': combined_metrics,
            'trades': all_trades,
            'equity_curve': combined_eq,
            'buy_hold_curve': combined_bh,
            'signals_log': per_sym_signals,
            'per_symbol_metrics': per_sym_metrics,
            'config': {
                'symbol': ', '.join(self.symbols),
                'symbols': self.symbols,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'interval': self.interval,
                'initial_capital': self.initial_capital,
                'capital_per_symbol': capital_per,
                'mode': self.mode,
                'factors': self.factors,
            },
            'api_stats': self.strategy.get_api_stats(),
        }

        if errors:
            results['warnings'] = errors

        await broadcast({'type': 'progress', 'progress': 100, 'message': 'Complete!'})
        await broadcast({'type': 'complete', 'metrics': combined_metrics,
                         'per_symbol_metrics': per_sym_metrics})

        self._save_backtest_report(results)
        return results

    def _save_backtest_report(self, results: dict) -> None:
        """Persist backtest results (trades, metrics, config) to backtest_reports/."""
        try:
            reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_reports')
            os.makedirs(reports_dir, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            sym_tag = '_'.join(self.symbols[:5])
            filepath = os.path.join(reports_dir, f'bt_{ts}_{sym_tag}_{self.interval}.json')

            # Build a serializable report with full trade detail
            report = {
                'timestamp': datetime.now().isoformat(),
                'config': results.get('config', {}),
                'metrics': results.get('metrics', {}),
                'per_symbol_metrics': results.get('per_symbol_metrics', {}),
                'trades': results.get('trades', []),
                'trade_count': len(results.get('trades', [])),
                'equity_curve_summary': {
                    'start': results['equity_curve'][0] if results.get('equity_curve') else None,
                    'end': results['equity_curve'][-1] if results.get('equity_curve') else None,
                    'points': len(results.get('equity_curve', [])),
                },
            }

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Backtest report saved: {filepath} ({report['trade_count']} trades)")
        except Exception as e:
            logger.error(f"Failed to save backtest report: {e}")

    @staticmethod
    def _precompute_indicators(df: pd.DataFrame) -> None:
        """Pre-compute all indicators on the full DataFrame at once.

        Adds indicator columns directly to df. All indicators are backward-looking
        (no look-ahead bias), so computing on the full series is mathematically
        identical to computing incrementally per-bar, but O(n) instead of O(n^2).
        """
        import ta as ta_lib
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume'] if 'volume' in df.columns else None
        n_bars = len(df)
        failed = []

        def _safe(name, fn):
            """Compute indicator, log and skip on failure."""
            try:
                fn()
            except Exception as e:
                failed.append(name)
                logger.debug(f"Indicator {name} failed ({n_bars} bars): {e}")

        _safe('rsi', lambda: df.__setitem__('_rsi', ta_lib.momentum.RSIIndicator(close, window=14).rsi()))

        def _stoch():
            stoch = ta_lib.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
            df['_stoch_k'] = stoch.stoch()
            df['_stoch_d'] = stoch.stoch_signal()
        _safe('stochastic', _stoch)

        def _macd():
            macd_ind = ta_lib.trend.MACD(close)
            df['_macd'] = macd_ind.macd()
            df['_macd_signal'] = macd_ind.macd_signal()
            df['_macd_hist'] = macd_ind.macd_diff()
        _safe('macd', _macd)

        def _bb():
            bb = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
            df['_bb_pct'] = bb.bollinger_pband()
            df['_bb_upper'] = bb.bollinger_hband()
            df['_bb_lower'] = bb.bollinger_lband()
        _safe('bollinger', _bb)

        def _adx():
            adx_ind = ta_lib.trend.ADXIndicator(high, low, close, window=14)
            df['_adx'] = adx_ind.adx()
            df['_di_plus'] = adx_ind.adx_pos()
            df['_di_minus'] = adx_ind.adx_neg()
        _safe('adx', _adx)

        _safe('atr', lambda: df.__setitem__('_atr', ta_lib.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()))

        def _emas():
            df['_ema9'] = ta_lib.trend.EMAIndicator(close, window=9).ema_indicator()
            df['_ema21'] = ta_lib.trend.EMAIndicator(close, window=21).ema_indicator()
            df['_ema50'] = ta_lib.trend.EMAIndicator(close, window=50).ema_indicator()
            df['_ema21_slope'] = (df['_ema21'] - df['_ema21'].shift(4)) / df['_ema21'].shift(4) * 100
        _safe('emas', _emas)

        # Detect if volume data is meaningful (forex often has zero volume)
        has_volume = volume is not None and (volume > 0).any()

        # VWAP (cumulative) — skip if volume is missing or all zeros
        if has_volume:
            typical = (high + low + close) / 3
            cumvol = volume.cumsum()
            df['_vwap'] = (typical * volume).cumsum() / cumvol.replace(0, np.nan)
        else:
            df['_vwap'] = close  # fallback: use close as VWAP proxy

        # Volume ratio (20-bar rolling average)
        if has_volume:
            avg_vol = volume.rolling(20, min_periods=1).mean()
            df['_vol_ratio'] = (volume / avg_vol.replace(0, 1.0))
        else:
            df['_vol_ratio'] = 1.0

        # Support / Resistance via swing high/low detection (5-bar window)
        def _sr():
            h = high.values
            l = low.values
            c = close.values
            n_b = len(df)
            swing_highs = np.full(n_b, np.nan)
            swing_lows = np.full(n_b, np.nan)

            for ii in range(4, n_b):
                wh = h[ii-4:ii+1]
                wl = l[ii-4:ii+1]
                if wh[2] == wh.max():
                    swing_highs[ii] = h[ii-2]
                if wl[2] == wl.min():
                    swing_lows[ii] = l[ii-2]

            # Track last 10 swing points → find nearest R above / S below
            last_sh, last_sl = [], []
            nr = np.full(n_b, np.nan)
            ns = np.full(n_b, np.nan)
            for ii in range(n_b):
                if not np.isnan(swing_highs[ii]):
                    last_sh.append(swing_highs[ii])
                    if len(last_sh) > 10: last_sh.pop(0)
                if not np.isnan(swing_lows[ii]):
                    last_sl.append(swing_lows[ii])
                    if len(last_sl) > 10: last_sl.pop(0)
                above = [p for p in last_sh if p > c[ii]]
                if above: nr[ii] = min(above)
                below = [p for p in last_sl if p < c[ii]]
                if below: ns[ii] = max(below)
            df['_nearest_r'] = nr
            df['_nearest_s'] = ns
        _safe('support_resistance', _sr)

        # --- Candlestick patterns (vectorized) ---
        def _candlestick_patterns():
            o = df['open'].values if 'open' in df.columns else close.values
            c = close.values
            h = high.values
            l = low.values

            body = np.abs(c - o)
            bar_range = h - l
            safe_range = np.where(bar_range > 0, bar_range, 1e-10)
            upper_wick = h - np.maximum(c, o)
            lower_wick = np.minimum(c, o) - l
            body_pct = body / safe_range

            # Previous bar values
            prev_c = np.roll(c, 1); prev_c[0] = c[0]
            prev_o = np.roll(o, 1); prev_o[0] = o[0]
            prev_body_hi = np.maximum(prev_o, prev_c)
            prev_body_lo = np.minimum(prev_o, prev_c)
            curr_body_hi = np.maximum(o, c)
            curr_body_lo = np.minimum(o, c)

            prev_bear = prev_c < prev_o
            prev_bull = prev_c > prev_o

            # Bullish Engulfing
            eb = (prev_bear & (c > o) & (curr_body_hi > prev_body_hi) & (curr_body_lo < prev_body_lo)).astype(float)
            eb[0] = 0.0

            # Bearish Engulfing
            eB = (prev_bull & (c < o) & (curr_body_hi > prev_body_hi) & (curr_body_lo < prev_body_lo)).astype(float)
            eB[0] = 0.0

            # Hammer: long lower wick, small upper wick
            hm = ((body_pct < 0.4) & (lower_wick >= 2.0 * body) & (upper_wick <= 0.3 * safe_range) & (body > 0.05 * safe_range)).astype(float)

            # Shooting Star: long upper wick, small lower wick
            ss = ((body_pct < 0.4) & (upper_wick >= 2.0 * body) & (lower_wick <= 0.3 * safe_range) & (body > 0.05 * safe_range)).astype(float)

            # --- Extended candlestick patterns (14 new) ---
            # 2-bar look-back values
            prev_h = np.roll(h, 1); prev_h[0] = h[0]
            prev_l = np.roll(l, 1); prev_l[0] = l[0]
            prev_range = prev_h - prev_l
            prev_mid = (prev_o + prev_c) / 2.0
            prev_body = np.abs(prev_c - prev_o)

            # 3-bar look-back values
            p2_c = np.roll(c, 2); p2_c[:2] = c[:2]
            p2_o = np.roll(o, 2); p2_o[:2] = o[:2]
            p2_h = np.roll(h, 2); p2_h[:2] = h[:2]
            p2_l = np.roll(l, 2); p2_l[:2] = l[:2]
            p2_body = np.abs(p2_c - p2_o)
            p2_body_hi = np.maximum(p2_o, p2_c)
            p2_body_lo = np.minimum(p2_o, p2_c)
            p2_mid = (p2_o + p2_c) / 2.0

            # Prev-bar body hi/lo (already computed as prev_body_hi/lo above)
            # Average body size (20-bar rolling)
            avg_body = pd.Series(body).rolling(20, min_periods=5).mean().fillna(body.mean()).values
            safe_avg_body = np.where(avg_body > 1e-10, avg_body, 1e-10)

            # Context: simple 5-bar trend for Hanging Man
            c_5back = np.roll(c, 5); c_5back[:5] = c[:5]
            in_uptrend_5 = c > c_5back  # price higher than 5 bars ago

            # 1. Dragonfly Doji (bullish): body near high, long lower shadow
            doji_dragon = ((body_pct < 0.10) & (upper_wick < 0.10 * safe_range)
                           & (lower_wick > 0.60 * safe_range) & (bar_range > 0)).astype(float)

            # 2. Gravestone Doji (bearish): body near low, long upper shadow
            doji_grave = ((body_pct < 0.10) & (lower_wick < 0.10 * safe_range)
                          & (upper_wick > 0.60 * safe_range) & (bar_range > 0)).astype(float)

            # 3. Morning Star (bullish 3-bar): big bear, small body, big bull closing > bar[-2] midpoint
            p2_bear = p2_c < p2_o
            p2_big = p2_body > safe_avg_body * 0.6
            p1_small = prev_body < p2_body * 0.4
            cur_bull = c > o
            cur_big = body > safe_avg_body * 0.6
            cur_above_mid = c > p2_mid
            ms = (p2_bear & p2_big & p1_small & cur_bull & cur_big & cur_above_mid).astype(float)
            ms[:2] = 0.0

            # 4. Evening Star (bearish 3-bar): big bull, small body, big bear closing < bar[-2] midpoint
            p2_bull = p2_c > p2_o
            cur_bear = c < o
            cur_below_mid = c < p2_mid
            es = (p2_bull & p2_big & p1_small & cur_bear & (body > safe_avg_body * 0.6) & cur_below_mid).astype(float)
            es[:2] = 0.0

            # 5. Three White Soldiers (bullish 3-bar): 3 consecutive bullish, each closing higher
            b3_bull = cur_bull & (prev_c > prev_o) & (p2_c > p2_o)
            b3_rising = (c > prev_c) & (prev_c > p2_c)
            b3_opens_in = (o >= prev_body_lo) & (o <= prev_body_hi) & (prev_o >= p2_body_lo) & (prev_o <= p2_body_hi)
            b3_decent = (body > safe_avg_body * 0.4) & (prev_body > safe_avg_body * 0.4) & (p2_body > safe_avg_body * 0.4)
            tw = (b3_bull & b3_rising & b3_opens_in & b3_decent).astype(float)
            tw[:2] = 0.0

            # 6. Three Black Crows (bearish 3-bar): 3 consecutive bearish, each closing lower
            b3_bear = cur_bear & (prev_c < prev_o) & (p2_c < p2_o)
            b3_falling = (c < prev_c) & (prev_c < p2_c)
            b3_opens_in_b = (o <= prev_body_hi) & (o >= prev_body_lo) & (prev_o <= p2_body_hi) & (prev_o >= p2_body_lo)
            b3_decent_b = (body > safe_avg_body * 0.4) & (prev_body > safe_avg_body * 0.4) & (p2_body > safe_avg_body * 0.4)
            tb = (b3_bear & b3_falling & b3_opens_in_b & b3_decent_b).astype(float)
            tb[:2] = 0.0

            # 7. Piercing Line (bullish 2-bar): bearish bar, then opens below low, closes above midpoint
            prev_bear_f = prev_c < prev_o
            opens_below = o < prev_l
            closes_above_mid = c > prev_mid
            closes_below_open = c < prev_o
            pl = (prev_bear_f & cur_bull & opens_below & closes_above_mid & closes_below_open & (body > safe_avg_body * 0.3)).astype(float)
            pl[0] = 0.0

            # 8. Dark Cloud Cover (bearish 2-bar): bullish bar, then opens above high, closes below midpoint
            prev_bull_f = prev_c > prev_o
            opens_above = o > prev_h
            closes_below_mid = c < prev_mid
            closes_above_close = c > prev_c
            dc = (prev_bull_f & cur_bear & opens_above & closes_below_mid & closes_above_close & (body > safe_avg_body * 0.3)).astype(float)
            dc[0] = 0.0

            # 9. Bullish Harami (2-bar): large bearish, then small bullish within body
            hb = (prev_bear_f & cur_bull & (prev_body > safe_avg_body * 0.8)
                  & (curr_body_hi < prev_body_hi) & (curr_body_lo > prev_body_lo)
                  & (body < prev_body * 0.5)).astype(float)
            hb[0] = 0.0

            # 10. Bearish Harami (2-bar): large bullish, then small bearish within body
            hB = (prev_bull_f & cur_bear & (prev_body > safe_avg_body * 0.8)
                  & (curr_body_hi < prev_body_hi) & (curr_body_lo > prev_body_lo)
                  & (body < prev_body * 0.5)).astype(float)
            hB[0] = 0.0

            # 11. Bullish Marubozu: full-body bullish candle, minimal wicks
            mb = (cur_bull & (body > 0.90 * safe_range) & (upper_wick < 0.05 * safe_range)
                  & (lower_wick < 0.05 * safe_range) & (bar_range > 0)).astype(float)

            # 12. Bearish Marubozu: full-body bearish candle, minimal wicks
            mB = (cur_bear & (body > 0.90 * safe_range) & (upper_wick < 0.05 * safe_range)
                  & (lower_wick < 0.05 * safe_range) & (bar_range > 0)).astype(float)

            # 13. Hanging Man: hammer shape BUT preceded by uptrend (bearish signal)
            hg = (hm.astype(bool) & in_uptrend_5).astype(float)

            # 14. Spinning Top: small body, long wicks both sides
            st = ((body_pct < 0.30) & (upper_wick > 0.25 * safe_range)
                  & (lower_wick > 0.25 * safe_range) & (bar_range > 0)).astype(float)

            df['_engulf_bull'] = eb
            df['_engulf_bear'] = eB
            df['_hammer'] = hm
            df['_shooting_star'] = ss
            df['_doji_dragon'] = doji_dragon
            df['_doji_grave'] = doji_grave
            df['_morning_star'] = ms
            df['_evening_star'] = es
            df['_three_white'] = tw
            df['_three_black'] = tb
            df['_piercing'] = pl
            df['_dark_cloud'] = dc
            df['_harami_bull'] = hb
            df['_harami_bear'] = hB
            df['_marubozu_bull'] = mb
            df['_marubozu_bear'] = mB
            df['_hanging_man'] = hg
            df['_spinning_top'] = st
        _safe('candlestick_patterns', _candlestick_patterns)

        # --- RSI + MACD Divergence (swing-based) ---
        def _divergence():
            c = close.values
            h = high.values
            l = low.values
            n_b = len(df)
            rsi_arr = df['_rsi'].values if '_rsi' in df.columns else np.full(n_b, 50.0)
            mh_arr = df['_macd_hist'].values if '_macd_hist' in df.columns else np.zeros(n_b)

            div_b = np.zeros(n_b)
            div_B = np.zeros(n_b)

            # 5-bar swing detection (same algo as S/R)
            swing_hi = np.full(n_b, np.nan)
            swing_lo = np.full(n_b, np.nan)
            for ii in range(4, n_b):
                wh = h[ii-4:ii+1]
                wl = l[ii-4:ii+1]
                if wh[2] == wh.max(): swing_hi[ii] = h[ii-2]
                if wl[2] == wl.min(): swing_lo[ii] = l[ii-2]

            recent_lows = []   # [(bar_idx, price, rsi, macd_h)]
            recent_highs = []

            for ii in range(n_b):
                if not np.isnan(swing_lo[ii]):
                    sb = max(ii - 2, 0)
                    r = rsi_arr[sb] if not np.isnan(rsi_arr[sb]) else 50.0
                    m = mh_arr[sb] if not np.isnan(mh_arr[sb]) else 0.0
                    recent_lows.append((sb, l[sb], r, m))
                    if len(recent_lows) > 5: recent_lows.pop(0)

                if not np.isnan(swing_hi[ii]):
                    sb = max(ii - 2, 0)
                    r = rsi_arr[sb] if not np.isnan(rsi_arr[sb]) else 50.0
                    m = mh_arr[sb] if not np.isnan(mh_arr[sb]) else 0.0
                    recent_highs.append((sb, h[sb], r, m))
                    if len(recent_highs) > 5: recent_highs.pop(0)

                # Bullish: price lower low, RSI or MACD higher low
                if len(recent_lows) >= 2:
                    cur, prev = recent_lows[-1], recent_lows[-2]
                    if cur[1] < prev[1] and ii - cur[0] <= 8:
                        if cur[2] > prev[2] + 2.0 or cur[3] > prev[3] + 0.001:
                            div_b[ii] = 1.0

                # Bearish: price higher high, RSI or MACD lower high
                if len(recent_highs) >= 2:
                    cur, prev = recent_highs[-1], recent_highs[-2]
                    if cur[1] > prev[1] and ii - cur[0] <= 8:
                        if cur[2] < prev[2] - 2.0 or cur[3] < prev[3] - 0.001:
                            div_B[ii] = 1.0

            df['_div_bull'] = div_b
            df['_div_bear'] = div_B
        _safe('divergence', _divergence)

        # --- Bollinger Band Squeeze ---
        def _bb_squeeze():
            if '_bb_upper' not in df.columns or '_bb_lower' not in df.columns:
                df['_bb_squeeze'] = 0.0
                return
            bb_u = df['_bb_upper'].values
            bb_l = df['_bb_lower'].values
            mid = (bb_u + bb_l) / 2.0
            safe_mid = np.where(np.abs(mid) > 1e-10, mid, 1e-10)
            width = (bb_u - bb_l) / safe_mid
            pctrank = pd.Series(width).fillna(1e6).rolling(120, min_periods=20).rank(pct=True).fillna(1.0).values
            df['_bb_squeeze'] = (pctrank <= 0.25).astype(float)
        _safe('bb_squeeze', _bb_squeeze)

        # --- OBV Slope (volume flow) ---
        def _obv_slope():
            if not has_volume:
                df['_obv_slope'] = 0.0
                return
            c = close.values
            v = volume.values if isinstance(volume, pd.Series) else volume
            price_chg = np.diff(c, prepend=c[0])
            signed_vol = np.where(price_chg > 0, v, np.where(price_chg < 0, -v, 0.0))
            obv = np.cumsum(signed_vol).astype(float)
            obv_ema = pd.Series(obv).ewm(span=10, adjust=False).mean().values
            obv_diff = np.diff(obv_ema, prepend=obv_ema[0])
            avg_vol = pd.Series(v.astype(float)).rolling(20, min_periods=1).mean().values
            safe_avg = np.where(avg_vol > 0, avg_vol, 1.0)
            result = obv_diff / safe_avg
            df['_obv_slope'] = np.where(np.isfinite(result), result, 0.0)
        _safe('obv_slope', _obv_slope)

        # --- Classic chart patterns (multi-bar structural) ---
        def _chart_patterns():
            """Detect classic chart patterns from swing point geometry."""
            c = close.values
            h = high.values
            l = low.values
            n_b = len(df)
            atr_arr = df['_atr'].values if '_atr' in df.columns else np.full(n_b, c.mean() * 0.01)
            pattern_tol = 0.5  # "same level" = within 0.5 * ATR

            # Pre-allocate 24 result arrays (10 existing + 14 new)
            pat_double_top = np.zeros(n_b)
            pat_double_bottom = np.zeros(n_b)
            pat_hs = np.zeros(n_b)
            pat_inv_hs = np.zeros(n_b)
            pat_asc_tri = np.zeros(n_b)
            pat_desc_tri = np.zeros(n_b)
            pat_bull_flag = np.zeros(n_b)
            pat_cup_handle = np.zeros(n_b)
            pat_falling_wedge = np.zeros(n_b)
            pat_rising_wedge = np.zeros(n_b)
            # 14 new chart patterns
            pat_inv_cup_handle = np.zeros(n_b)
            pat_bear_flag = np.zeros(n_b)
            pat_pennant_bull = np.zeros(n_b)
            pat_pennant_bear = np.zeros(n_b)
            pat_rounding_bottom = np.zeros(n_b)
            pat_rounding_top = np.zeros(n_b)
            pat_rectangle_bull = np.zeros(n_b)
            pat_rectangle_bear = np.zeros(n_b)
            pat_broadening_bottom = np.zeros(n_b)
            pat_broadening_top = np.zeros(n_b)
            pat_triple_bottom = np.zeros(n_b)
            pat_triple_top = np.zeros(n_b)
            pat_sym_tri_bull = np.zeros(n_b)
            pat_sym_tri_bear = np.zeros(n_b)

            recent_highs = []  # [(bar_idx, price)]
            recent_lows = []

            def _same_level(p1, p2, tol_v):
                return abs(p1 - p2) <= tol_v

            for ii in range(4, n_b):
                # 5-bar swing detection (same algorithm as S/R)
                wh = h[ii-4:ii+1]
                wl = l[ii-4:ii+1]
                if wh[2] == wh.max():
                    recent_highs.append((ii - 2, h[ii - 2]))
                    if len(recent_highs) > 10: recent_highs.pop(0)
                if wl[2] == wl.min():
                    recent_lows.append((ii - 2, l[ii - 2]))
                    if len(recent_lows) > 10: recent_lows.pop(0)

                atr_v = atr_arr[ii]
                if np.isnan(atr_v) or atr_v <= 0:
                    atr_v = c[ii] * 0.01
                tol = pattern_tol * atr_v

                # === DOUBLE TOP (bearish) ===
                if len(recent_highs) >= 2:
                    h2_idx, h2_px = recent_highs[-1]
                    h1_idx, h1_px = recent_highs[-2]
                    span = h2_idx - h1_idx
                    if 15 <= span <= 80 and _same_level(h1_px, h2_px, tol):
                        troughs = [px for idx, px in recent_lows if h1_idx < idx < h2_idx]
                        if troughs:
                            neckline = min(troughs)
                            if c[ii] < neckline:
                                pat_double_top[ii] = 1.0

                # === DOUBLE BOTTOM (bullish) ===
                if len(recent_lows) >= 2:
                    l2_idx, l2_px = recent_lows[-1]
                    l1_idx, l1_px = recent_lows[-2]
                    span = l2_idx - l1_idx
                    if 15 <= span <= 80 and _same_level(l1_px, l2_px, tol):
                        peaks = [px for idx, px in recent_highs if l1_idx < idx < l2_idx]
                        if peaks:
                            neckline = max(peaks)
                            if c[ii] > neckline:
                                pat_double_bottom[ii] = 1.0

                # === HEAD & SHOULDERS (bearish) ===
                if len(recent_highs) >= 3:
                    rs_idx, rs_px = recent_highs[-1]   # right shoulder
                    hd_idx, hd_px = recent_highs[-2]   # head
                    ls_idx, ls_px = recent_highs[-3]   # left shoulder
                    total_span = rs_idx - ls_idx
                    if (25 <= total_span <= 100
                        and hd_px > ls_px + tol * 0.5
                        and hd_px > rs_px + tol * 0.5
                        and _same_level(ls_px, rs_px, tol)):
                        t1 = [px for idx, px in recent_lows if ls_idx < idx < hd_idx]
                        t2 = [px for idx, px in recent_lows if hd_idx < idx < rs_idx]
                        if t1 and t2:
                            neckline = max(t1[-1], t2[-1])
                            if c[ii] < neckline:
                                pat_hs[ii] = 1.0

                # === INVERSE HEAD & SHOULDERS (bullish) ===
                if len(recent_lows) >= 3:
                    rs_idx, rs_px = recent_lows[-1]    # right shoulder
                    hd_idx, hd_px = recent_lows[-2]    # head (lowest)
                    ls_idx, ls_px = recent_lows[-3]    # left shoulder
                    total_span = rs_idx - ls_idx
                    if (25 <= total_span <= 100
                        and hd_px < ls_px - tol * 0.5
                        and hd_px < rs_px - tol * 0.5
                        and _same_level(ls_px, rs_px, tol)):
                        p1 = [px for idx, px in recent_highs if ls_idx < idx < hd_idx]
                        p2 = [px for idx, px in recent_highs if hd_idx < idx < rs_idx]
                        if p1 and p2:
                            neckline = min(p1[-1], p2[-1])
                            if c[ii] > neckline:
                                pat_inv_hs[ii] = 1.0

                # === ASCENDING TRIANGLE (bullish) ===
                if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                    rh = recent_highs[-3:] if len(recent_highs) >= 3 else recent_highs[-2:]
                    tri_span = rh[-1][0] - rh[0][0]
                    if 20 <= tri_span <= 80:
                        avg_h = sum(p for _, p in rh) / len(rh)
                        flat_r = all(_same_level(p, avg_h, tol) for _, p in rh)
                        rl = [sl for sl in recent_lows if sl[0] >= rh[0][0]]
                        if flat_r and len(rl) >= 2:
                            rising_s = all(rl[j+1][1] > rl[j][1] - tol * 0.3
                                           for j in range(len(rl) - 1))
                            if rising_s:
                                pat_asc_tri[ii] = 1.0

                # === DESCENDING TRIANGLE (bearish) ===
                if len(recent_lows) >= 2 and len(recent_highs) >= 2:
                    rl = recent_lows[-3:] if len(recent_lows) >= 3 else recent_lows[-2:]
                    tri_span = rl[-1][0] - rl[0][0]
                    if 20 <= tri_span <= 80:
                        avg_l = sum(p for _, p in rl) / len(rl)
                        flat_s = all(_same_level(p, avg_l, tol) for _, p in rl)
                        rh = [sh for sh in recent_highs if sh[0] >= rl[0][0]]
                        if flat_s and len(rh) >= 2:
                            falling_r = all(rh[j+1][1] < rh[j][1] + tol * 0.3
                                            for j in range(len(rh) - 1))
                            if falling_r:
                                pat_desc_tri[ii] = 1.0

                # === BULL FLAG (continuation) ===
                pole_lookback = min(60, ii)
                flag_lookback = min(20, ii)
                if pole_lookback >= 15 and flag_lookback >= 5:
                    flag_start = ii - flag_lookback
                    pole_start = ii - pole_lookback
                    pole_rise = h[flag_start] - l[pole_start:flag_start+1].min()
                    pole_range_atr = pole_rise / max(atr_v, 1e-10)
                    if pole_range_atr >= 3.0:
                        flag_highs = h[flag_start:ii+1]
                        flag_lows = l[flag_start:ii+1]
                        flag_range = flag_highs.max() - flag_lows.min()
                        flag_range_atr = flag_range / max(atr_v, 1e-10)
                        flag_retrace = (flag_highs.max() - c[ii]) / pole_rise if pole_rise > 0 else 1.0
                        if flag_range_atr < 2.5 and flag_retrace < 0.5:
                            flag_slope = (c[ii] - c[flag_start]) / max(flag_lookback, 1)
                            if flag_slope <= 0 or flag_slope / max(c[ii], 1e-10) < 0.001:
                                pat_bull_flag[ii] = 1.0

                # === CUP & HANDLE (bullish) ===
                if len(recent_lows) >= 5:
                    lows = recent_lows[-7:] if len(recent_lows) >= 7 else recent_lows[-5:]
                    cup_span = lows[-1][0] - lows[0][0]
                    if 30 <= cup_span <= 120:
                        prices = [p for _, p in lows]
                        min_idx = prices.index(min(prices))
                        n_pts = len(prices)
                        if n_pts * 0.2 <= min_idx <= n_pts * 0.8:
                            left_lip = prices[0]
                            right_lip = prices[-1]
                            if _same_level(left_lip, right_lip, tol * 1.5):
                                cup_depth = left_lip - min(prices)
                                if cup_depth >= atr_v:
                                    handle_pullback = max(prices[-2:]) - c[ii]
                                    if 0 < handle_pullback < cup_depth * 0.5:
                                        pat_cup_handle[ii] = 1.0

                # === FALLING WEDGE (bullish) ===
                if len(recent_highs) >= 3 and len(recent_lows) >= 3:
                    rh = recent_highs[-4:] if len(recent_highs) >= 4 else recent_highs[-3:]
                    rl = recent_lows[-4:] if len(recent_lows) >= 4 else recent_lows[-3:]
                    w_span = max(rh[-1][0], rl[-1][0]) - min(rh[0][0], rl[0][0])
                    if 20 <= w_span <= 80:
                        h_declining = all(rh[j+1][1] < rh[j][1] + tol * 0.2
                                          for j in range(len(rh) - 1))
                        l_declining = all(rl[j+1][1] < rl[j][1] + tol * 0.2
                                          for j in range(len(rl) - 1))
                        if h_declining and l_declining:
                            h_span = rh[-1][0] - rh[0][0]
                            l_span = rl[-1][0] - rl[0][0]
                            h_slope = (rh[-1][1] - rh[0][1]) / max(h_span, 1)
                            l_slope = (rl[-1][1] - rl[0][1]) / max(l_span, 1)
                            if l_slope < h_slope < 0:
                                initial_range = rh[0][1] - rl[0][1]
                                current_range = rh[-1][1] - rl[-1][1]
                                if initial_range > 0 and current_range > 0 and current_range < initial_range * 0.8:
                                    pat_falling_wedge[ii] = 1.0

                # === RISING WEDGE (bearish) ===
                if len(recent_highs) >= 3 and len(recent_lows) >= 3:
                    rh = recent_highs[-4:] if len(recent_highs) >= 4 else recent_highs[-3:]
                    rl = recent_lows[-4:] if len(recent_lows) >= 4 else recent_lows[-3:]
                    w_span = max(rh[-1][0], rl[-1][0]) - min(rh[0][0], rl[0][0])
                    if 20 <= w_span <= 80:
                        h_rising = all(rh[j+1][1] > rh[j][1] - tol * 0.2
                                       for j in range(len(rh) - 1))
                        l_rising = all(rl[j+1][1] > rl[j][1] - tol * 0.2
                                       for j in range(len(rl) - 1))
                        if h_rising and l_rising:
                            h_span = rh[-1][0] - rh[0][0]
                            l_span = rl[-1][0] - rl[0][0]
                            h_slope = (rh[-1][1] - rh[0][1]) / max(h_span, 1)
                            l_slope = (rl[-1][1] - rl[0][1]) / max(l_span, 1)
                            if 0 < h_slope < l_slope:
                                initial_range = rh[0][1] - rl[0][1]
                                current_range = rh[-1][1] - rl[-1][1]
                                if initial_range > 0 and current_range > 0 and current_range < initial_range * 0.8:
                                    pat_rising_wedge[ii] = 1.0

                # === TRIPLE TOP (bearish) ===
                if len(recent_highs) >= 3:
                    h3_idx, h3_px = recent_highs[-1]
                    h2_idx, h2_px = recent_highs[-2]
                    h1_idx, h1_px = recent_highs[-3]
                    span = h3_idx - h1_idx
                    if (20 <= span <= 100
                        and _same_level(h1_px, h2_px, tol)
                        and _same_level(h2_px, h3_px, tol)):
                        troughs = [px for idx, px in recent_lows
                                   if h1_idx < idx < h3_idx]
                        if len(troughs) >= 2:
                            neckline = min(troughs)
                            if c[ii] < neckline:
                                pat_triple_top[ii] = 1.0

                # === TRIPLE BOTTOM (bullish) ===
                if len(recent_lows) >= 3:
                    l3_idx, l3_px = recent_lows[-1]
                    l2_idx, l2_px = recent_lows[-2]
                    l1_idx, l1_px = recent_lows[-3]
                    span = l3_idx - l1_idx
                    if (20 <= span <= 100
                        and _same_level(l1_px, l2_px, tol)
                        and _same_level(l2_px, l3_px, tol)):
                        peaks = [px for idx, px in recent_highs
                                 if l1_idx < idx < l3_idx]
                        if len(peaks) >= 2:
                            neckline = max(peaks)
                            if c[ii] > neckline:
                                pat_triple_bottom[ii] = 1.0

                # === INVERSE CUP & HANDLE (bearish) ===
                if len(recent_highs) >= 5:
                    highs = recent_highs[-7:] if len(recent_highs) >= 7 else recent_highs[-5:]
                    cup_span = highs[-1][0] - highs[0][0]
                    if 30 <= cup_span <= 120:
                        prices_h = [p for _, p in highs]
                        max_idx_h = prices_h.index(max(prices_h))
                        n_pts = len(prices_h)
                        if n_pts * 0.2 <= max_idx_h <= n_pts * 0.8:
                            left_lip = prices_h[0]
                            right_lip = prices_h[-1]
                            if _same_level(left_lip, right_lip, tol * 1.5):
                                cup_height = max(prices_h) - left_lip
                                if cup_height >= atr_v:
                                    handle_rally = c[ii] - min(prices_h[-2:])
                                    if 0 < handle_rally < cup_height * 0.5:
                                        pat_inv_cup_handle[ii] = 1.0

                # === BEAR FLAG (bearish continuation) ===
                pole_lb = min(60, ii)
                flag_lb = min(20, ii)
                if pole_lb >= 15 and flag_lb >= 5:
                    fs = ii - flag_lb
                    ps = ii - pole_lb
                    pole_drop = h[ps:fs+1].max() - l[fs]
                    pole_atr = pole_drop / max(atr_v, 1e-10)
                    if pole_atr >= 3.0:
                        flag_h = h[fs:ii+1]
                        flag_l = l[fs:ii+1]
                        f_range = flag_h.max() - flag_l.min()
                        f_range_atr = f_range / max(atr_v, 1e-10)
                        f_retrace = (c[ii] - flag_l.min()) / pole_drop if pole_drop > 0 else 1.0
                        if f_range_atr < 2.5 and f_retrace < 0.5:
                            f_slope = (c[ii] - c[fs]) / max(flag_lb, 1)
                            if f_slope >= 0 or abs(f_slope) / max(c[ii], 1e-10) < 0.001:
                                pat_bear_flag[ii] = 1.0

                # === BULLISH PENNANT (continuation after up-move) ===
                if pole_lb >= 15 and flag_lb >= 5 and len(recent_highs) >= 2 and len(recent_lows) >= 2:
                    fs = ii - flag_lb
                    ps = ii - pole_lb
                    pole_rise = h[fs] - l[ps:fs+1].min()
                    if pole_rise / max(atr_v, 1e-10) >= 3.0:
                        fh = [(idx, px) for idx, px in recent_highs if idx >= fs]
                        fl = [(idx, px) for idx, px in recent_lows if idx >= fs]
                        if len(fh) >= 2 and len(fl) >= 2:
                            h_falling = fh[-1][1] < fh[0][1] - tol * 0.2
                            l_rising = fl[-1][1] > fl[0][1] + tol * 0.2
                            if h_falling and l_rising:
                                # Convergence check: pennant range < 60% of initial range
                                init_r = fh[0][1] - fl[0][1]
                                cur_r = fh[-1][1] - fl[-1][1]
                                if init_r > 0 and cur_r > 0 and cur_r < init_r * 0.6:
                                    pat_pennant_bull[ii] = 1.0

                # === BEARISH PENNANT (continuation after down-move) ===
                if pole_lb >= 15 and flag_lb >= 5 and len(recent_highs) >= 2 and len(recent_lows) >= 2:
                    fs = ii - flag_lb
                    ps = ii - pole_lb
                    pole_drop = h[ps:fs+1].max() - l[fs]
                    if pole_drop / max(atr_v, 1e-10) >= 3.0:
                        fh = [(idx, px) for idx, px in recent_highs if idx >= fs]
                        fl = [(idx, px) for idx, px in recent_lows if idx >= fs]
                        if len(fh) >= 2 and len(fl) >= 2:
                            h_falling = fh[-1][1] < fh[0][1] - tol * 0.2
                            l_rising = fl[-1][1] > fl[0][1] + tol * 0.2
                            if h_falling and l_rising:
                                init_r = fh[0][1] - fl[0][1]
                                cur_r = fh[-1][1] - fl[-1][1]
                                if init_r > 0 and cur_r > 0 and cur_r < init_r * 0.6:
                                    pat_pennant_bear[ii] = 1.0

                # === ROUNDING BOTTOM (bullish) ===
                if len(recent_lows) >= 5:
                    lows_r = recent_lows[-8:] if len(recent_lows) >= 8 else recent_lows[-5:]
                    r_span = lows_r[-1][0] - lows_r[0][0]
                    if 30 <= r_span <= 120:
                        prices_r = [p for _, p in lows_r]
                        min_idx_r = prices_r.index(min(prices_r))
                        n_pts_r = len(prices_r)
                        if n_pts_r * 0.25 <= min_idx_r <= n_pts_r * 0.75:
                            depth = prices_r[0] - min(prices_r)
                            if depth >= atr_v * 1.5:
                                # Left side must be declining, right side rising (U-shape)
                                left_dec = all(prices_r[j] >= prices_r[j+1] - tol * 0.3
                                               for j in range(min_idx_r))
                                right_inc = all(prices_r[j] <= prices_r[j+1] + tol * 0.3
                                                for j in range(min_idx_r, n_pts_r - 1))
                                if left_dec and right_inc:
                                    if prices_r[-1] > min(prices_r) + depth * 0.7:
                                        pat_rounding_bottom[ii] = 1.0

                # === ROUNDING TOP (bearish) ===
                if len(recent_highs) >= 5:
                    highs_r = recent_highs[-8:] if len(recent_highs) >= 8 else recent_highs[-5:]
                    r_span = highs_r[-1][0] - highs_r[0][0]
                    if 30 <= r_span <= 120:
                        prices_rt = [p for _, p in highs_r]
                        max_idx_r = prices_rt.index(max(prices_rt))
                        n_pts_r = len(prices_rt)
                        if n_pts_r * 0.25 <= max_idx_r <= n_pts_r * 0.75:
                            height = max(prices_rt) - prices_rt[0]
                            if height >= atr_v * 1.5:
                                # Left side rising, right side declining (inverted U)
                                left_inc = all(prices_rt[j] <= prices_rt[j+1] + tol * 0.3
                                               for j in range(max_idx_r))
                                right_dec = all(prices_rt[j] >= prices_rt[j+1] - tol * 0.3
                                                for j in range(max_idx_r, n_pts_r - 1))
                                if left_inc and right_dec:
                                    if prices_rt[-1] < max(prices_rt) - height * 0.7:
                                        pat_rounding_top[ii] = 1.0

                # === RECTANGLE BULLISH (breakout above flat range) ===
                if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                    rh_r = recent_highs[-4:] if len(recent_highs) >= 4 else recent_highs[-2:]
                    rl_r = recent_lows[-4:] if len(recent_lows) >= 4 else recent_lows[-2:]
                    rect_span = max(rh_r[-1][0], rl_r[-1][0]) - min(rh_r[0][0], rl_r[0][0])
                    if 20 <= rect_span <= 80 and len(rh_r) >= 2 and len(rl_r) >= 2:
                        avg_rh = sum(p for _, p in rh_r) / len(rh_r)
                        avg_rl = sum(p for _, p in rl_r) / len(rl_r)
                        flat_top = all(_same_level(p, avg_rh, tol) for _, p in rh_r)
                        flat_bot = all(_same_level(p, avg_rl, tol) for _, p in rl_r)
                        if flat_top and flat_bot and avg_rh > avg_rl + atr_v * 0.5:
                            if c[ii] > avg_rh:
                                pat_rectangle_bull[ii] = 1.0

                # === RECTANGLE BEARISH (breakdown below flat range) ===
                if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                    rh_r = recent_highs[-4:] if len(recent_highs) >= 4 else recent_highs[-2:]
                    rl_r = recent_lows[-4:] if len(recent_lows) >= 4 else recent_lows[-2:]
                    rect_span = max(rh_r[-1][0], rl_r[-1][0]) - min(rh_r[0][0], rl_r[0][0])
                    if 20 <= rect_span <= 80 and len(rh_r) >= 2 and len(rl_r) >= 2:
                        avg_rh = sum(p for _, p in rh_r) / len(rh_r)
                        avg_rl = sum(p for _, p in rl_r) / len(rl_r)
                        flat_top = all(_same_level(p, avg_rh, tol) for _, p in rh_r)
                        flat_bot = all(_same_level(p, avg_rl, tol) for _, p in rl_r)
                        if flat_top and flat_bot and avg_rh > avg_rl + atr_v * 0.5:
                            if c[ii] < avg_rl:
                                pat_rectangle_bear[ii] = 1.0

                # === BROADENING BOTTOM / MEGAPHONE (bullish) ===
                if len(recent_highs) >= 3 and len(recent_lows) >= 3:
                    rh_b = recent_highs[-4:] if len(recent_highs) >= 4 else recent_highs[-3:]
                    rl_b = recent_lows[-4:] if len(recent_lows) >= 4 else recent_lows[-3:]
                    b_span = max(rh_b[-1][0], rl_b[-1][0]) - min(rh_b[0][0], rl_b[0][0])
                    if 20 <= b_span <= 100:
                        hh = all(rh_b[j+1][1] > rh_b[j][1] - tol * 0.2
                                 for j in range(len(rh_b) - 1))
                        ll = all(rl_b[j+1][1] < rl_b[j][1] + tol * 0.2
                                 for j in range(len(rl_b) - 1))
                        if hh and ll:
                            if c[ii] > rh_b[-1][1]:
                                pat_broadening_bottom[ii] = 1.0

                # === BROADENING TOP / MEGAPHONE (bearish) ===
                if len(recent_highs) >= 3 and len(recent_lows) >= 3:
                    rh_b = recent_highs[-4:] if len(recent_highs) >= 4 else recent_highs[-3:]
                    rl_b = recent_lows[-4:] if len(recent_lows) >= 4 else recent_lows[-3:]
                    b_span = max(rh_b[-1][0], rl_b[-1][0]) - min(rh_b[0][0], rl_b[0][0])
                    if 20 <= b_span <= 100:
                        hh = all(rh_b[j+1][1] > rh_b[j][1] - tol * 0.2
                                 for j in range(len(rh_b) - 1))
                        ll = all(rl_b[j+1][1] < rl_b[j][1] + tol * 0.2
                                 for j in range(len(rl_b) - 1))
                        if hh and ll:
                            if c[ii] < rl_b[-1][1]:
                                pat_broadening_top[ii] = 1.0

                # === SYMMETRICAL TRIANGLE BULLISH ===
                if len(recent_highs) >= 3 and len(recent_lows) >= 3:
                    rh_s = recent_highs[-4:] if len(recent_highs) >= 4 else recent_highs[-3:]
                    rl_s = recent_lows[-4:] if len(recent_lows) >= 4 else recent_lows[-3:]
                    s_span = max(rh_s[-1][0], rl_s[-1][0]) - min(rh_s[0][0], rl_s[0][0])
                    if 20 <= s_span <= 80:
                        h_dec = all(rh_s[j+1][1] < rh_s[j][1] + tol * 0.2
                                    for j in range(len(rh_s) - 1))
                        l_inc = all(rl_s[j+1][1] > rl_s[j][1] - tol * 0.2
                                    for j in range(len(rl_s) - 1))
                        if h_dec and l_inc:
                            init_range = rh_s[0][1] - rl_s[0][1]
                            cur_range = rh_s[-1][1] - rl_s[-1][1]
                            if init_range > 0 and cur_range > 0 and cur_range < init_range * 0.7:
                                if c[ii] > rh_s[-1][1]:
                                    pat_sym_tri_bull[ii] = 1.0

                # === SYMMETRICAL TRIANGLE BEARISH ===
                if len(recent_highs) >= 3 and len(recent_lows) >= 3:
                    rh_s = recent_highs[-4:] if len(recent_highs) >= 4 else recent_highs[-3:]
                    rl_s = recent_lows[-4:] if len(recent_lows) >= 4 else recent_lows[-3:]
                    s_span = max(rh_s[-1][0], rl_s[-1][0]) - min(rh_s[0][0], rl_s[0][0])
                    if 20 <= s_span <= 80:
                        h_dec = all(rh_s[j+1][1] < rh_s[j][1] + tol * 0.2
                                    for j in range(len(rh_s) - 1))
                        l_inc = all(rl_s[j+1][1] > rl_s[j][1] - tol * 0.2
                                    for j in range(len(rl_s) - 1))
                        if h_dec and l_inc:
                            init_range = rh_s[0][1] - rl_s[0][1]
                            cur_range = rh_s[-1][1] - rl_s[-1][1]
                            if init_range > 0 and cur_range > 0 and cur_range < init_range * 0.7:
                                if c[ii] < rl_s[-1][1]:
                                    pat_sym_tri_bear[ii] = 1.0

            df['_pat_double_top'] = pat_double_top
            df['_pat_double_bottom'] = pat_double_bottom
            df['_pat_head_shoulders'] = pat_hs
            df['_pat_inv_hs'] = pat_inv_hs
            df['_pat_asc_triangle'] = pat_asc_tri
            df['_pat_desc_triangle'] = pat_desc_tri
            df['_pat_bull_flag'] = pat_bull_flag
            df['_pat_cup_handle'] = pat_cup_handle
            df['_pat_falling_wedge'] = pat_falling_wedge
            df['_pat_rising_wedge'] = pat_rising_wedge
            df['_pat_inv_cup_handle'] = pat_inv_cup_handle
            df['_pat_bear_flag'] = pat_bear_flag
            df['_pat_pennant_bull'] = pat_pennant_bull
            df['_pat_pennant_bear'] = pat_pennant_bear
            df['_pat_rounding_bottom'] = pat_rounding_bottom
            df['_pat_rounding_top'] = pat_rounding_top
            df['_pat_rectangle_bull'] = pat_rectangle_bull
            df['_pat_rectangle_bear'] = pat_rectangle_bear
            df['_pat_broadening_bottom'] = pat_broadening_bottom
            df['_pat_broadening_top'] = pat_broadening_top
            df['_pat_triple_bottom'] = pat_triple_bottom
            df['_pat_triple_top'] = pat_triple_top
            df['_pat_sym_tri_bull'] = pat_sym_tri_bull
            df['_pat_sym_tri_bear'] = pat_sym_tri_bear
        _safe('chart_patterns', _chart_patterns)

        df['_indicators_precomputed'] = True
        if failed:
            logger.warning(f"Some indicators failed ({n_bars} bars): {', '.join(failed)}")
        else:
            logger.info(f"Pre-computed {len(df)} bars of indicators in one pass")

    @staticmethod
    def _get_indicators_at(df: pd.DataFrame, i: int) -> dict:
        """Fast lookup of pre-computed indicators at bar index i."""
        try:
            row = df.iloc[i]
            rsi = row.get('_rsi', float('nan'))
            ema9 = row.get('_ema9', float('nan'))
            ema21 = row.get('_ema21', float('nan'))
            macd_val = row.get('_macd', float('nan'))
            macd_signal = row.get('_macd_signal', float('nan'))
            macd_hist = row.get('_macd_hist', float('nan'))
            bb_pct = row.get('_bb_pct', float('nan'))
            bb_upper = row.get('_bb_upper', float('nan'))
            bb_lower = row.get('_bb_lower', float('nan'))
            adx = row.get('_adx', float('nan'))
            atr = row.get('_atr', float('nan'))

            def r(v, d=2):
                return round(v, d) if pd.notna(v) else 0

            return {
                'rsi': r(rsi, 1),
                'macd': r(macd_val, 3),
                'macd_signal': r(macd_signal, 3),
                'macd_hist': r(macd_hist, 3),
                'bb_pct': r(bb_pct * 100 if pd.notna(bb_pct) else 0, 0),
                'bb_upper': r(bb_upper),
                'bb_lower': r(bb_lower),
                'adx': r(adx, 1),
                'atr': r(atr, 3),
                'ema9': r(ema9),
                'ema21': r(ema21),
                'ema_trend': 'BULL' if (pd.notna(ema9) and pd.notna(ema21) and ema9 > ema21) else 'BEAR',
            }
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def _calc_indicators(df: pd.DataFrame) -> dict:
        """Calculate key indicators for the current bar (fallback for non-precomputed data)."""
        try:
            import ta as ta_lib
            close = df['close']
            high = df['high']
            low = df['low']

            rsi = ta_lib.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

            macd_ind = ta_lib.trend.MACD(close)
            macd_val = macd_ind.macd().iloc[-1]
            macd_signal = macd_ind.macd_signal().iloc[-1]
            macd_hist = macd_ind.macd_diff().iloc[-1]

            bb = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
            bb_pct = bb.bollinger_pband().iloc[-1]
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]

            adx = ta_lib.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]
            atr = ta_lib.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]

            ema9 = ta_lib.trend.EMAIndicator(close, window=9).ema_indicator().iloc[-1]
            ema21 = ta_lib.trend.EMAIndicator(close, window=21).ema_indicator().iloc[-1]

            return {
                'rsi': round(rsi, 1),
                'macd': round(macd_val, 3),
                'macd_signal': round(macd_signal, 3),
                'macd_hist': round(macd_hist, 3),
                'bb_pct': round(bb_pct * 100, 0),
                'bb_upper': round(bb_upper, 2),
                'bb_lower': round(bb_lower, 2),
                'adx': round(adx, 1),
                'atr': round(atr, 3),
                'ema9': round(ema9, 2),
                'ema21': round(ema21, 2),
                'ema_trend': 'BULL' if ema9 > ema21 else 'BEAR',
            }
        except Exception as e:
            return {'error': str(e)}

    def _calculate_metrics(self, equity_curve: list, trades: list,
                           initial_capital: float, final_value: float,
                           bh_entry: float, bh_exit: float) -> dict:
        """Calculate performance metrics."""
        total_return = ((final_value / initial_capital) - 1) * 100
        bh_return = ((bh_exit / bh_entry) - 1) * 100

        # Equity values
        values = [e['value'] for e in equity_curve]
        if not values:
            return {'total_return': 0, 'total_trades': 0}

        # Per-bar returns
        returns = []
        for i in range(1, len(values)):
            r = (values[i] / values[i - 1]) - 1 if values[i - 1] != 0 else 0
            returns.append(r)

        returns_arr = np.array(returns) if returns else np.array([0])

        # Sharpe ratio (annualized based on interval)
        # Bars per year: daily=252, 60m=1764, 30m=3528, 15m=7056, 5m=19656
        bars_per_year = {'1d': 252, '30m': 3528, '15m': 7056, '10m': 9828, '5m': 19656, '1m': 98280}.get(self.interval, 252)
        if len(returns_arr) > 1 and returns_arr.std() > 0:
            sharpe = (returns_arr.mean() / returns_arr.std()) * np.sqrt(bars_per_year)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = initial_capital
        max_dd = 0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd

        # Win rate
        closed_trades = [t for t in trades if t.get('pnl', 0) != 0]
        winning = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losing = [t for t in closed_trades if t.get('pnl', 0) < 0]
        win_rate = (len(winning) / len(closed_trades) * 100) if closed_trades else 0

        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning)
        gross_loss = abs(sum(t['pnl'] for t in losing))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Average win/loss
        avg_win = (gross_profit / len(winning)) if winning else 0
        avg_loss = (gross_loss / len(losing)) if losing else 0

        return {
            'total_return': round(total_return, 2),
            'buy_hold_return': round(bh_return, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_dd * 100, 2),
            'win_rate': round(win_rate, 1),
            'total_trades': len(closed_trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'Inf',
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'initial_capital': initial_capital,
            'final_value': round(final_value, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
        }


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the backtest dashboard HTML."""
    return DASHBOARD_HTML


@app.get("/api/ollama/models")
async def get_ollama_models():
    """Proxy endpoint to fetch available Ollama models (avoids CORS)."""
    try:
        async with _aiohttp.ClientSession() as session:
            async with session.get(
                'http://localhost:11434/api/tags',
                timeout=_aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [
                        {'name': m['name'], 'size_gb': round(m.get('size', 0) / 1e9, 1)}
                        for m in data.get('models', [])
                    ]
                    return {'models': models}
        return {'models': [], 'error': 'Ollama not running'}
    except Exception as e:
        logger.error(f"Ollama models fetch error: {e}")
        return {'models': [], 'error': f'Ollama not reachable: {str(e)}'}


@app.post("/api/backtest/run")
async def run_backtest(config: dict = {}):
    """Start a backtest."""
    global backtest_state

    if backtest_state['status'] == 'running':
        return {'error': 'A backtest is already running'}

    run_id = str(uuid.uuid4())[:8]
    backtest_state = {
        'status': 'running',
        'progress': 0,
        'message': 'Starting...',
        'results': None,
        'run_id': run_id,
    }

    async def _run():
        global backtest_state
        try:
            runner = BacktestRunner(config)
            results = await runner.run()
            backtest_state['status'] = 'complete'
            backtest_state['results'] = results
            backtest_state['progress'] = 100
            backtest_state['message'] = 'Complete'
        except Exception as e:
            logger.error(f"Backtest error: {traceback.format_exc()}")
            backtest_state['status'] = 'error'
            backtest_state['message'] = str(e)
            await broadcast({'type': 'error', 'message': str(e)})

    asyncio.create_task(_run())
    return {'run_id': run_id, 'status': 'started'}


@app.get("/api/backtest/status")
async def backtest_status():
    """Get backtest progress."""
    return {
        'status': backtest_state['status'],
        'progress': backtest_state['progress'],
        'message': backtest_state['message'],
        'run_id': backtest_state['run_id'],
    }


@app.get("/api/backtest/results")
async def backtest_results():
    """Get completed backtest results."""
    if backtest_state['status'] != 'complete':
        return {'error': 'No completed backtest', 'status': backtest_state['status']}
    return backtest_state['results']


@app.get("/api/symbols")
async def list_symbols():
    """List popular symbols across equities, crypto, and forex."""
    return {
        'symbols': [
            {'symbol': 'SPY', 'name': 'S&P 500 ETF'},
            {'symbol': 'QQQ', 'name': 'Nasdaq 100 ETF'},
            {'symbol': 'NVDA', 'name': 'NVIDIA'},
            {'symbol': 'AAPL', 'name': 'Apple'},
            {'symbol': 'MSFT', 'name': 'Microsoft'},
            {'symbol': 'PLTR', 'name': 'Palantir'},
            {'symbol': 'TSLA', 'name': 'Tesla'},
            {'symbol': 'AMD', 'name': 'AMD'},
            {'symbol': 'AMZN', 'name': 'Amazon'},
            {'symbol': 'META', 'name': 'Meta Platforms'},
            {'symbol': 'GOOG', 'name': 'Alphabet'},
            {'symbol': 'IWM', 'name': 'Russell 2000 ETF'},
            {'symbol': 'BTC-USD', 'name': 'Bitcoin'},
            {'symbol': 'ETH-USD', 'name': 'Ethereum'},
            {'symbol': 'SOL-USD', 'name': 'Solana'},
            {'symbol': 'EUR/USD', 'name': 'Euro/US Dollar'},
            {'symbol': 'GBP/USD', 'name': 'British Pound/US Dollar'},
            {'symbol': 'USD/JPY', 'name': 'US Dollar/Japanese Yen'},
            {'symbol': 'USD/CHF', 'name': 'US Dollar/Swiss Franc'},
            {'symbol': 'AUD/USD', 'name': 'Australian Dollar/US Dollar'},
            {'symbol': 'NZD/USD', 'name': 'New Zealand Dollar/US Dollar'},
            {'symbol': 'USD/CAD', 'name': 'US Dollar/Canadian Dollar'},
            {'symbol': 'EUR/GBP', 'name': 'Euro/British Pound'},
        ]
    }


@app.get("/api/key-status")
async def api_key_status():
    """Check if ANTHROPIC_API_KEY is set."""
    key = os.environ.get('ANTHROPIC_API_KEY', '')
    return {
        'set': bool(key),
        'masked': f"{key[:8]}...{key[-4:]}" if len(key) > 12 else ('***' if key else ''),
    }


@app.get("/api/data-sources")
async def data_sources():
    """Check which data sources are available."""
    schwab_ok = get_schwab_client() is not None
    yfinance_ok = yf is not None
    coinbase_ok = _requests is not None  # public API, always available if requests installed
    oanda_ok = _get_oanda_config() is not None
    alpaca_ok = _get_alpaca_config() is not None
    return {
        'schwab': schwab_ok,
        'yfinance': yfinance_ok,
        'coinbase': coinbase_ok,
        'oanda': oanda_ok,
        'alpaca': alpaca_ok,
        'primary': 'schwab' if schwab_ok else ('yfinance' if yfinance_ok else 'none'),
    }


def build_param_grid():
    """Build parameter grid for v3 rules engine.

    Includes entry, exit, and time-of-day params.
    Total combos = 4*4*2*2*2*2*2 = 512.
    """
    grid = []
    for stop_m in [1.5, 2.0, 2.5, 3.0]:
        for target_m in [2.0, 3.0, 4.0, 5.0]:
            for triggers in [2, 3]:
                for pb_zone in [0.75, 1.0]:
                    for be_r in [0.5, 0.75]:
                        for trail_atr in [1.5, 2.0]:
                            for skip_open in [0, 20]:
                                grid.append({
                                    'atr_stop_mult': stop_m,
                                    'atr_target_mult': target_m,
                                    'min_triggers': triggers,
                                    'pullback_zone_atr': pb_zone,
                                    'breakeven_r': be_r,
                                    'trail_atr_mult': trail_atr,
                                    'trail_start_r': 1.0,
                                    'min_trend_adx': 20,
                                    'min_vol_ratio': 0.8,
                                    'skip_open_minutes': skip_open,
                                    'max_trades_per_day': 2,
                                })
    return grid


def run_grid_search(arrays: PrecomputedArrays, param_grid: list,
                    capital: float = 100000, commission: float = 0.001,
                    min_trades: int = 3, slippage_factor: float = 0.0,
                    mkt: Optional[MarketContext] = None) -> list:
    """Run all param combos and return sorted results (CPU-bound, call in thread)."""
    results = []
    for params in param_grid:
        try:
            r = quick_backtest(params, arrays, capital, commission, slippage_factor, mkt=mkt)
            if r['num_trades'] >= min_trades:
                results.append(r)
        except Exception:
            pass
    results.sort(key=lambda r: r['return_pct'] - r['max_drawdown'] * 0.5, reverse=True)
    return results


# =============================================================================
# WALK-FORWARD VALIDATION ENGINE
# =============================================================================

# All patterns/indicators tracked by the scorecard.
ALL_TRACKED_PATTERNS = [
    # Candlestick patterns (18)
    'ENGULF_BULL', 'ENGULF_BEAR', 'HAMMER', 'SHOOTING_STAR',
    'DOJI_DRAGON', 'DOJI_GRAVE', 'MORNING_STAR', 'EVENING_STAR',
    'THREE_WHITE', 'THREE_BLACK', 'PIERCING', 'DARK_CLOUD',
    'HARAMI_BULL', 'HARAMI_BEAR', 'MARUBOZU_BULL', 'MARUBOZU_BEAR',
    'HANGING_MAN', 'SPINNING_TOP',
    # Indicators
    'DIV_BULL', 'DIV_BEAR', 'BB_SQUEEZE', 'SQUEEZE_FIRE',
    'OBV_BULL', 'OBV_BEAR',
    # Chart patterns (24)
    'DOUBLE_TOP', 'DOUBLE_BOTTOM', 'HEAD_SHOULDERS', 'INV_HS',
    'ASC_TRIANGLE', 'DESC_TRIANGLE', 'BULL_FLAG', 'CUP_HANDLE',
    'FALLING_WEDGE', 'RISING_WEDGE',
    'INV_CUP_HANDLE', 'BEAR_FLAG', 'PENNANT_BULL', 'PENNANT_BEAR',
    'ROUNDING_BOTTOM', 'ROUNDING_TOP', 'RECT_BULL', 'RECT_BEAR',
    'BROADENING_BOTTOM', 'BROADENING_TOP', 'TRIPLE_BOTTOM', 'TRIPLE_TOP',
    'SYM_TRI_BULL', 'SYM_TRI_BEAR',
    # Context
    'STRONG_BULL', 'BULL_TREND', 'STRONG_BEAR', 'BEAR_TREND', 'VOL_SURGE',
]


def _walk_forward_defaults(interval: str) -> dict:
    """Return sensible train/test/step sizes (in bars) for a given interval."""
    # Tuned so each interval produces 3-6 folds on typical data lengths.
    defaults = {
        '1m':  {'train': 600,  'test': 200,  'step': 100},
        '5m':  {'train': 900,  'test': 300,  'step': 150},
        '15m': {'train': 600,  'test': 200,  'step': 100},
        '30m': {'train': 400,  'test': 150,  'step': 75},
        '1h':  {'train': 300,  'test': 100,  'step': 50},
        '1d':  {'train': 150,  'test': 50,   'step': 25},
    }
    return defaults.get(interval, defaults['5m'])


def run_walk_forward(arrays: PrecomputedArrays, interval: str,
                     base_params: dict, capital: float = 100000,
                     commission: float = 0.001, slippage: float = 0.0,
                     mkt: Optional[MarketContext] = None,
                     train_bars: Optional[int] = None,
                     test_bars: Optional[int] = None,
                     step_bars: Optional[int] = None) -> dict:
    """Rolling walk-forward validation with pattern-level attribution.

    For each fold:
      1. Optimize parameters on the TRAINING window (grid search).
      2. Run a DETAILED backtest on the TEST window (out-of-sample) with the
         best parameters, logging every trade and its active patterns.
      3. Record in-sample vs. out-of-sample performance to detect overfitting.

    After all folds, aggregate OOS results and compute a pattern scorecard
    that shows which patterns contribute to winning vs. losing trades.

    Returns a structured dict ready for JSON serialization or console display.
    """
    n = arrays.n
    lookback = arrays.lookback
    dfl = _walk_forward_defaults(interval)
    tw = train_bars or dfl['train']
    tsw = test_bars or dfl['test']
    sw = step_bars or dfl['step']

    # Validate: need at least one full fold
    if n < lookback + tw + tsw:
        return {
            'error': f'Insufficient data: {n} bars < {lookback + tw + tsw} needed for 1 fold',
            'folds': [], 'oos_metrics': {}, 'pattern_scorecard': [], 'recommendations': [],
        }

    # Generate fold boundaries
    folds_meta = []
    fold_start = lookback
    fold_num = 0
    while fold_start + tw + tsw <= n:
        train_s = fold_start
        train_e = fold_start + tw
        test_s = train_e
        test_e = min(train_e + tsw, n)
        folds_meta.append((fold_num, train_s, train_e, test_s, test_e))
        fold_start += sw
        fold_num += 1

    if not folds_meta:
        return {
            'error': 'Could not construct any folds with the given window sizes.',
            'folds': [], 'oos_metrics': {}, 'pattern_scorecard': [], 'recommendations': [],
        }

    # Build parameter grid (inject any base params like adaptive risk)
    grid = build_param_grid()
    for p in grid:
        for k, v in base_params.items():
            if k not in ('atr_stop_mult', 'atr_target_mult', 'min_triggers', 'pullback_zone_atr'):
                p[k] = v

    # Run each fold
    fold_results = []
    all_oos_trade_logs = []   # concatenated OOS trade logs
    all_oos_pnls = []         # concatenated OOS PnL floats

    for fold_num, train_s, train_e, test_s, test_e in folds_meta:
        # --- Phase 1: Optimize on training window ---
        is_results = []
        for params in grid:
            try:
                r = quick_backtest(params, arrays, capital, commission, slippage,
                                   mkt=mkt, start_bar=train_s, end_bar=train_e)
                if r['num_trades'] >= 3:
                    is_results.append(r)
            except Exception:
                pass
        is_results.sort(key=lambda r: r['return_pct'] - r['max_drawdown'] * 0.5, reverse=True)

        if not is_results:
            fold_results.append({
                'fold': fold_num, 'train_bars': f'{train_s}-{train_e}',
                'test_bars': f'{test_s}-{test_e}',
                'status': 'no_valid_params', 'is_return': 0, 'oos_return': 0,
                'oos_trades': 0, 'best_params': {},
            })
            continue

        best = is_results[0]
        best_params = best['params']

        # --- Phase 2: Out-of-sample test with trade logging ---
        oos_trade_log = []
        oos_result = quick_backtest(best_params, arrays, capital, commission, slippage,
                                    mkt=mkt, trade_log=oos_trade_log,
                                    start_bar=test_s, end_bar=test_e)

        is_ret = best['return_pct']
        oos_ret = oos_result['return_pct']
        overfit_ratio = oos_ret / is_ret if is_ret != 0 else 0.0

        fold_results.append({
            'fold': fold_num,
            'train_bars': f'{train_s}-{train_e}',
            'test_bars': f'{test_s}-{test_e}',
            'status': 'ok',
            'best_params': {k: v for k, v in best_params.items()
                           if k in ('atr_stop_mult', 'atr_target_mult',
                                    'min_triggers', 'pullback_zone_atr')},
            'is_return': is_ret,
            'is_trades': best['num_trades'],
            'is_win_rate': best['win_rate'],
            'is_profit_factor': best['profit_factor'],
            'is_max_dd': best['max_drawdown'],
            'oos_return': oos_ret,
            'oos_trades': oos_result['num_trades'],
            'oos_win_rate': oos_result['win_rate'],
            'oos_profit_factor': oos_result['profit_factor'],
            'oos_max_dd': oos_result['max_drawdown'],
            'overfit_ratio': round(overfit_ratio, 3),
        })

        all_oos_trade_logs.extend(oos_trade_log)
        all_oos_pnls.extend([t['pnl'] for t in oos_trade_log])

    # =========================================================================
    # AGGREGATE OOS METRICS
    # =========================================================================
    total_oos_trades = len(all_oos_pnls)
    oos_metrics = {}
    if total_oos_trades > 0:
        wins = [p for p in all_oos_pnls if p > 0]
        losses = [p for p in all_oos_pnls if p < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        total_pnl = sum(all_oos_pnls)
        oos_ret = total_pnl / capital * 100

        # Compute max drawdown on concatenated OOS equity curve
        eq = capital
        peak = capital
        max_dd_oos = 0.0
        for pnl in all_oos_pnls:
            eq += pnl
            if eq > peak: peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd_oos: max_dd_oos = dd

        # Sharpe approximation (annualized, from per-trade returns)
        if total_oos_trades >= 2:
            ret_arr = np.array(all_oos_pnls) / capital
            mean_r = ret_arr.mean()
            std_r = ret_arr.std()
            # Estimate actual trades/year from observed frequency
            total_oos_bars = sum(
                fm[4] - fm[3]
                for fm in folds_meta if fm[0] < len(fold_results)
                and fold_results[fm[0]].get('status') == 'ok'
            ) if folds_meta else tsw * len(fold_results)
            bars_per_year = {'1m': 98280, '5m': 19656, '15m': 6552,
                             '30m': 3276, '1h': 1638, '1d': 252}
            bpy = bars_per_year.get(interval, 6552)
            # Scale actual trade count to annual rate
            tpy = (total_oos_trades / max(total_oos_bars, 1)) * bpy
            sharpe = (mean_r / std_r * np.sqrt(max(tpy, 1))) if std_r > 0 else 0.0
        else:
            sharpe = 0.0

        oos_metrics = {
            'total_return_pct': round(oos_ret, 2),
            'total_pnl': round(total_pnl, 2),
            'num_trades': total_oos_trades,
            'win_rate': round(len(wins) / total_oos_trades * 100, 1),
            'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else (
                999.0 if gross_profit > 0 else 0.0),
            'max_drawdown_pct': round(max_dd_oos * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'avg_win': round(sum(wins) / len(wins), 2) if wins else 0,
            'avg_loss': round(sum(losses) / len(losses), 2) if losses else 0,
            'largest_win': round(max(wins), 2) if wins else 0,
            'largest_loss': round(min(losses), 2) if losses else 0,
            'avg_overfit_ratio': round(
                np.mean([f['overfit_ratio'] for f in fold_results if f.get('status') == 'ok']), 3
            ) if any(f.get('status') == 'ok' for f in fold_results) else 0,
        }
    else:
        oos_metrics = {
            'total_return_pct': 0, 'total_pnl': 0, 'num_trades': 0,
            'win_rate': 0, 'profit_factor': 0, 'max_drawdown_pct': 0,
            'sharpe_ratio': 0, 'avg_win': 0, 'avg_loss': 0,
            'largest_win': 0, 'largest_loss': 0, 'avg_overfit_ratio': 0,
        }

    # =========================================================================
    # PATTERN SCORECARD
    # =========================================================================
    # For each tracked pattern, compute stats across ALL OOS trades.
    pattern_stats = {}
    for pat in ALL_TRACKED_PATTERNS:
        pattern_stats[pat] = {'wins': 0, 'losses': 0, 'win_pnl': 0.0, 'loss_pnl': 0.0}

    for trade in all_oos_trade_logs:
        active = trade.get('patterns_at_entry', [])
        pnl = trade['pnl']
        is_win = pnl > 0
        for pat in active:
            if pat in pattern_stats:
                if is_win:
                    pattern_stats[pat]['wins'] += 1
                    pattern_stats[pat]['win_pnl'] += pnl
                else:
                    pattern_stats[pat]['losses'] += 1
                    pattern_stats[pat]['loss_pnl'] += pnl

    scorecard = []
    for pat in ALL_TRACKED_PATTERNS:
        s = pattern_stats[pat]
        total = s['wins'] + s['losses']
        if total == 0:
            continue
        win_rate = s['wins'] / total * 100
        avg_win = s['win_pnl'] / s['wins'] if s['wins'] > 0 else 0
        avg_loss = s['loss_pnl'] / s['losses'] if s['losses'] > 0 else 0  # negative number
        expectancy = (avg_win * s['wins'] + avg_loss * s['losses']) / total
        total_pnl = s['win_pnl'] + s['loss_pnl']

        scorecard.append({
            'pattern': pat,
            'trades': total,
            'wins': s['wins'],
            'losses': s['losses'],
            'win_rate': round(win_rate, 1),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'expectancy': round(expectancy, 2),
            'total_pnl': round(total_pnl, 2),
            'significant': total >= 10,  # minimum sample size for reliability
        })
    scorecard.sort(key=lambda x: x['expectancy'], reverse=True)

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    recommendations = []

    # Overfit warning
    avg_or = oos_metrics.get('avg_overfit_ratio', 0)
    if avg_or < 0:
        recommendations.append(
            f'OVERFIT WARNING: Average OOS/IS ratio is {avg_or:.2f} (negative). '
            'Parameters found in training are losing money out-of-sample. '
            'Consider increasing min_triggers or reducing the number of setups.')
    elif 0 < avg_or < 0.3:
        recommendations.append(
            f'OVERFIT LIKELY: OOS/IS ratio is {avg_or:.2f} (<0.3). '
            'Most of the in-sample edge is not surviving out-of-sample.')
    elif avg_or >= 0.5:
        recommendations.append(
            f'ROBUST: OOS/IS ratio is {avg_or:.2f} (>=0.5). '
            'The strategy retains a meaningful portion of its in-sample edge.')

    # Pattern recommendations
    profitable_patterns = [s for s in scorecard if s['significant'] and s['expectancy'] > 0]
    losing_patterns = [s for s in scorecard if s['significant'] and s['expectancy'] < 0]

    if profitable_patterns:
        top = profitable_patterns[:3]
        names = ', '.join(t['pattern'] for t in top)
        recommendations.append(
            f'TOP PATTERNS (by expectancy): {names}. '
            'These patterns had positive expected value per trade on out-of-sample data.')

    if losing_patterns:
        names = ', '.join(t['pattern'] for t in losing_patterns[:3])
        recommendations.append(
            f'UNDERPERFORMING PATTERNS: {names}. '
            'Consider disabling these — they had negative expectancy on OOS data.')

    # Insufficient data
    low_sample = [s for s in scorecard if not s['significant'] and s['trades'] >= 3]
    if low_sample:
        names = ', '.join(s['pattern'] for s in low_sample[:3])
        recommendations.append(
            f'INSUFFICIENT DATA for: {names}. '
            f'Need 10+ trades per pattern for reliable statistics (currently 3-9).')

    if total_oos_trades == 0:
        recommendations.append(
            'ZERO OOS TRADES across all folds. The strategy may be too restrictive, '
            'or the data is too short. Try reducing min_triggers or using a longer date range.')

    # Overall assessment
    if total_oos_trades >= 10:
        wr = oos_metrics['win_rate']
        pf = oos_metrics['profit_factor']
        if pf >= 1.5 and wr >= 45:
            recommendations.append(
                f'OVERALL: Strategy looks viable (PF={pf}, WR={wr}%). '
                'Consider paper trading with the most recent fold\'s parameters.')
        elif pf >= 1.0:
            recommendations.append(
                f'OVERALL: Marginal edge (PF={pf}, WR={wr}%). '
                'The strategy breaks even or slightly profits. Needs refinement.')
        else:
            recommendations.append(
                f'OVERALL: Strategy is losing money (PF={pf}, WR={wr}%). '
                'Do not paper trade. Re-evaluate pattern weights and stop distances.')

    return {
        'config': {
            'total_bars': n,
            'train_window': tw,
            'test_window': tsw,
            'step_size': sw,
            'num_folds': len(folds_meta),
            'interval': interval,
        },
        'folds': fold_results,
        'oos_metrics': oos_metrics,
        'pattern_scorecard': scorecard,
        'oos_trade_log': all_oos_trade_logs,
        'recommendations': recommendations,
    }


@app.post("/api/optimize")
async def run_optimizer(config: dict = {}):
    """Run parameter optimization for the rules engine.
    Tests multiple parameter combinations and returns the best ones."""

    symbol = config.get('symbol', 'PLTR')
    interval = config.get('interval', '15m')
    initial_capital = config.get('initial_capital', 100000)
    commission_pct = config.get('commission_pct', 0.1) / 100
    trading_style = config.get('trading_style', 'swing')
    slippage_factor = config.get('slippage_factor', 0.0)

    # Fetch data — route by asset class
    default_days = {'1m': 10, '5m': 30, '10m': 45, '15m': 45, '30m': 60, '1d': 365}.get(interval, 60)
    start = config.get('start_date', (datetime.now() - timedelta(days=default_days)).strftime('%Y-%m-%d'))
    end = config.get('end_date', datetime.now().strftime('%Y-%m-%d'))

    df = None
    if '/' in symbol:
        df = fetch_oanda_data(symbol, start, end, interval)
        if df is None:
            df = fetch_yfinance_data(symbol, start, end, interval)
    elif symbol.endswith('-USD'):
        df = fetch_coinbase_data(symbol, start, end, interval)
        if df is None:
            df = fetch_yfinance_data(symbol, start, end, interval)
    else:
        df = fetch_schwab_data(symbol, start, end, interval)
        if df is None:
            df = fetch_alpaca_bars(symbol, start, end, interval)
        if df is None:
            df = fetch_yfinance_data(symbol, start, end, interval)
    if df is None or df.empty:
        return {'error': f'No data for {symbol} — check data source credentials or yfinance'}

    adaptive_risk = config.get('adaptive_risk', False)
    param_grid = build_param_grid()
    # Inject adaptive risk flag into every param combo
    if adaptive_risk:
        for p in param_grid:
            p['adaptive_risk'] = True
            p['risk_per_trade'] = config.get('risk_per_trade', 0.02)
            p['max_stop_pct'] = config.get('max_stop_pct', 0.04)
            p['drawdown_half_risk'] = config.get('drawdown_half_risk', 0.05)
            p['drawdown_skip'] = config.get('drawdown_skip', 0.10)
            p['cooldown_losses'] = config.get('cooldown_losses', 3)
            p['reentry_cooldown_bars'] = config.get('reentry_cooldown_bars', 5)
            p['min_entry_adx'] = config.get('min_entry_adx', 20)

    # Pre-compute indicators and extract numpy arrays
    arrays = precompute_arrays(df, trading_style, interval)

    # Fetch market context (SPY + VIX) for adaptive risk
    mkt = None
    is_equity = '/' not in symbol and not symbol.endswith('-USD')
    if adaptive_risk and is_equity:
        await broadcast({'type': 'progress', 'progress': 8, 'message': 'Fetching SPY + VIX market context...'})
        mkt = fetch_market_context(df, interval)

    # Run all parameter combinations — use thread pool to avoid blocking event loop
    await broadcast({'type': 'progress', 'progress': 10, 'message': f'Optimizing {len(param_grid)} combinations on {len(df)} bars...'})

    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(None, run_grid_search, arrays, param_grid, initial_capital, commission_pct, 3, slippage_factor, mkt)

    # Poll until done, sending progress updates
    total = len(param_grid)
    while not future.done():
        await asyncio.sleep(0.5)
        await broadcast({'type': 'progress', 'progress': 50, 'message': f'Optimizing {total} combinations...'})

    results = await asyncio.wrap_future(future)

    await broadcast({'type': 'progress', 'progress': 100, 'message': f'Done! Tested {len(param_grid)} combinations, {len(results)} valid.'})

    return {
        'total_tested': len(param_grid),
        'valid_results': len(results),
        'best': results[:10],
        'symbol': symbol,
        'interval': interval,
        'bars': len(df),
    }


@app.post("/api/optimize/multi")
async def run_multi_optimizer(config: dict = {}):
    """Optimize parameters separately for each symbol in a comma-separated list.

    Returns per-symbol best params that can be passed as rules_params.per_symbol.
    """
    raw_sym = config.get('symbol', config.get('symbols', 'SPY'))
    symbols = [s.strip().upper() for s in raw_sym.split(',') if s.strip()]
    if not symbols:
        return {'error': 'No symbols provided'}

    interval = config.get('interval', '5m')
    initial_capital = config.get('initial_capital', 100000)
    commission_pct = config.get('commission_pct', 0.1) / 100
    trading_style = config.get('trading_style', 'swing')
    slippage_factor = config.get('slippage_factor', 0.0)
    adaptive_risk = config.get('adaptive_risk', False)

    default_days = {'1m': 10, '5m': 30, '10m': 45, '15m': 45, '30m': 60, '1d': 365}.get(interval, 60)
    start = config.get('start_date', (datetime.now() - timedelta(days=default_days)).strftime('%Y-%m-%d'))
    end = config.get('end_date', datetime.now().strftime('%Y-%m-%d'))

    per_symbol_results = {}
    per_symbol_best = {}
    n_syms = len(symbols)

    for idx, sym in enumerate(symbols):
        pct_base = int(5 + (idx / n_syms) * 90)
        await broadcast({'type': 'progress', 'progress': pct_base,
                         'message': f'Optimizing {sym} ({idx+1}/{n_syms})...'})
        try:
            # Fetch data
            df = None
            if '/' in sym:
                df = fetch_oanda_data(sym, start, end, interval)
                if df is None:
                    df = fetch_yfinance_data(sym, start, end, interval)
            elif sym.endswith('-USD'):
                df = fetch_coinbase_data(sym, start, end, interval)
                if df is None:
                    df = fetch_yfinance_data(sym, start, end, interval)
            else:
                df = fetch_schwab_data(sym, start, end, interval)
                if df is None:
                    df = fetch_alpaca_bars(sym, start, end, interval)
                if df is None:
                    df = fetch_yfinance_data(sym, start, end, interval)

            if df is None or df.empty:
                per_symbol_results[sym] = {'error': f'No data for {sym}'}
                continue

            # Build param grid
            param_grid = build_param_grid()
            if adaptive_risk:
                for p in param_grid:
                    p['adaptive_risk'] = True
                    p['risk_per_trade'] = config.get('risk_per_trade', 0.02)
                    p['max_stop_pct'] = config.get('max_stop_pct', 0.04)
                    p['drawdown_half_risk'] = config.get('drawdown_half_risk', 0.05)
                    p['drawdown_skip'] = config.get('drawdown_skip', 0.10)
                    p['cooldown_losses'] = config.get('cooldown_losses', 3)
                    p['reentry_cooldown_bars'] = config.get('reentry_cooldown_bars', 5)
                    p['min_entry_adx'] = config.get('min_entry_adx', 20)

            arrays = precompute_arrays(df, trading_style, interval)

            # Market context for equities
            mkt = None
            is_equity = '/' not in sym and not sym.endswith('-USD')
            if adaptive_risk and is_equity:
                mkt = fetch_market_context(df, interval)

            # Run grid search in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, run_grid_search, arrays, param_grid,
                initial_capital / n_syms, commission_pct, 3, slippage_factor, mkt
            )

            if results:
                best = results[0]
                per_symbol_best[sym] = best['params']
                per_symbol_results[sym] = {
                    'best_params': best['params'],
                    'return_pct': best['return_pct'],
                    'win_rate': best['win_rate'],
                    'profit_factor': best['profit_factor'],
                    'max_drawdown': best['max_drawdown'],
                    'num_trades': best['num_trades'],
                    'top3': [{'params': r['params'], 'return_pct': r['return_pct'],
                              'win_rate': r['win_rate'], 'profit_factor': r['profit_factor']}
                             for r in results[:3]],
                    'bars': len(df),
                    'valid_combos': len(results),
                }
            else:
                per_symbol_results[sym] = {'error': 'No valid results (all combos had <3 trades)'}

        except Exception as e:
            logger.error(f"Multi-optimizer error for {sym}: {e}")
            per_symbol_results[sym] = {'error': str(e)}

    await broadcast({'type': 'progress', 'progress': 100,
                     'message': f'Done! Optimized {len(per_symbol_best)}/{n_syms} symbols.'})

    return {
        'symbols': symbols,
        'per_symbol': per_symbol_results,
        'per_symbol_params': per_symbol_best,
        'interval': interval,
    }


@app.post("/api/walkforward")
async def run_walkforward_analysis(config: dict = {}):
    """Run rolling walk-forward validation with pattern attribution.

    Accepts same config as /api/optimize, plus optional:
      train_bars, test_bars, step_bars  (override auto-sizing)
    Returns: folds, OOS metrics, pattern scorecard, recommendations.
    """
    symbol = config.get('symbol', 'SPY')
    interval = config.get('interval', '5m')
    initial_capital = config.get('initial_capital', 100000)
    commission_pct = config.get('commission_pct', 0.1) / 100
    trading_style = config.get('trading_style', 'swing')
    slippage = config.get('slippage_factor', 0.1)

    # Base params (adaptive risk, etc.) — forwarded to grid but not grid-searched
    base_params = {}
    if config.get('adaptive_risk'):
        base_params['adaptive_risk'] = True
        base_params['risk_per_trade'] = config.get('risk_per_trade', 0.02)
        base_params['max_stop_pct'] = config.get('max_stop_pct', 0.04)
        base_params['drawdown_half_risk'] = config.get('drawdown_half_risk', 0.05)
        base_params['drawdown_skip'] = config.get('drawdown_skip', 0.10)
        base_params['cooldown_losses'] = config.get('cooldown_losses', 3)

    # Fetch data (same routing as optimizer)
    start = config.get('start_date')
    end = config.get('end_date')
    if not start or not end:
        from datetime import datetime as _dt, timedelta as _td
        end = _dt.now().strftime('%Y-%m-%d')
        # Use longer defaults for walk-forward (need more data)
        lookback_days = {'1m': 7, '5m': 30, '15m': 60, '30m': 90, '1h': 180, '1d': 365}
        start = (_dt.now() - _td(days=lookback_days.get(interval, 30))).strftime('%Y-%m-%d')

    df = None
    if '/' in symbol:
        df = fetch_yfinance_data(symbol.replace('/', ''), start, end, interval)
    elif symbol.endswith('-USD'):
        df = fetch_yfinance_data(symbol, start, end, interval)
    else:
        df = fetch_schwab_data(symbol, start, end, interval)
        if df is None:
            df = fetch_alpaca_bars(symbol, start, end, interval)
        if df is None:
            df = fetch_yfinance_data(symbol, start, end, interval)

    if df is None or len(df) < 100:
        return {'error': f'Insufficient data for {symbol} {interval}: got {len(df) if df is not None else 0} bars'}

    logger.info(f"Walk-forward: {symbol} {interval}, {len(df)} bars")

    loop = asyncio.get_event_loop()

    # Precompute indicators
    arrays = await loop.run_in_executor(
        None, precompute_arrays, df, trading_style, interval, 50)

    # Fetch market context if equities + adaptive
    mkt = None
    if base_params.get('adaptive_risk') and '/' not in symbol and not symbol.endswith('-USD'):
        try:
            mkt = await loop.run_in_executor(None, fetch_market_context, df, interval)
        except Exception:
            pass

    # Run walk-forward (CPU-bound)
    report = await loop.run_in_executor(
        None, run_walk_forward, arrays, interval, base_params,
        initial_capital, commission_pct, slippage, mkt,
        config.get('train_bars'), config.get('test_bars'), config.get('step_bars'))

    report['symbol'] = symbol
    report['interval'] = interval
    report['total_bars'] = len(df)

    # Log summary
    oos = report.get('oos_metrics', {})
    logger.info(
        f"Walk-forward complete: {symbol} {interval} | "
        f"{report['config']['num_folds']} folds | "
        f"OOS return: {oos.get('total_return_pct', 0):+.2f}% | "
        f"OOS trades: {oos.get('num_trades', 0)} | "
        f"OOS PF: {oos.get('profit_factor', 0):.2f} | "
        f"Patterns tracked: {len(report.get('pattern_scorecard', []))}"
    )

    # Strip the full trade log from the response (can be large); keep summary
    if 'oos_trade_log' in report:
        report['oos_trade_log_count'] = len(report['oos_trade_log'])
        report['oos_trade_log_sample'] = report['oos_trade_log'][:20]  # first 20 for inspection
        del report['oos_trade_log']

    return report


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live backtest updates."""
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle ping/pong
            if data == 'ping':
                await websocket.send_text('pong')
    except WebSocketDisconnect:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)


# =============================================================================
# TELEGRAM ALERT HELPERS
# =============================================================================

_telegram_channel = None


def _init_telegram():
    """Initialize singleton TelegramChannel from env vars."""
    global _telegram_channel
    if _telegram_channel is not None:
        return _telegram_channel
    if not TELEGRAM_IMPORTS_OK:
        return None
    _load_env_file()
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
    if not bot_token or not chat_id:
        return None
    try:
        _telegram_channel = TelegramChannel(bot_token=bot_token, chat_id=chat_id)
        logger.info("Telegram alert channel initialized")
        return _telegram_channel
    except Exception as e:
        logger.warning(f"Failed to init Telegram: {e}")
        return None


async def send_telegram_alert(title: str, message: str, severity=None):
    """Send an alert via Telegram (fire-and-forget)."""
    channel = _init_telegram()
    if channel is None:
        return
    if severity is None:
        severity = AlertSeverity.INFO
    try:
        alert = Alert(
            id=str(uuid.uuid4())[:8],
            severity=severity,
            title=title,
            message=message,
            source='auto_backtest',
            timestamp=datetime.now(),
        )
        await channel.send(alert)
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


# =============================================================================
# AUTONOMOUS BACKTEST ENGINE
# =============================================================================

AUTO_SYMBOLS = [
    'SPY', 'QQQ', 'NVDA', 'AAPL', 'MSFT', 'PLTR', 'TSLA', 'AMD', 'AMZN', 'META', 'GOOG', 'IWM',
    'BTC-USD', 'ETH-USD', 'SOL-USD',
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'NZD/USD', 'USD/CAD', 'EUR/GBP',
]
AUTO_INTERVALS = ['15m', '1d']
AUTO_STATE_FILE = os.path.join(os.path.dirname(__file__) or '.', 'auto_backtest_state.json')


class AutonomousBacktestEngine:
    """Continuously optimizes parameters across symbols/intervals, learns from results."""

    def __init__(self):
        self.status = 'stopped'  # stopped | running | paused
        self.task: Optional[asyncio.Task] = None
        self.current_job = ''
        self.current_phase = ''
        self.total_cycles = 0
        self.total_optimizations = 0
        self.symbol_results: Dict[str, Dict[str, Any]] = {}
        self.global_best: List[dict] = []
        self.enabled_symbols: set = set(AUTO_SYMBOLS)  # user-configurable subset
        self.brain = None
        self._data_cache: Dict[str, tuple] = {}  # key -> (df, timestamp)
        self._load_state()
        if BRAIN_AVAILABLE:
            try:
                self.brain = TradingBrain()
            except Exception as e:
                logger.warning(f"Could not init TradingBrain: {e}")

    def _load_state(self):
        """Load persisted state from JSON."""
        try:
            if os.path.exists(AUTO_STATE_FILE):
                with open(AUTO_STATE_FILE, 'r') as f:
                    state = json.load(f)
                self.total_cycles = state.get('total_cycles', 0)
                self.total_optimizations = state.get('total_optimizations', 0)
                self.symbol_results = state.get('symbol_results', {})
                self.global_best = state.get('global_best', [])
                saved_symbols = state.get('enabled_symbols')
                if saved_symbols is not None:
                    self.enabled_symbols = set(saved_symbols)
                logger.info(f"Loaded auto-backtest state: {self.total_cycles} cycles, {self.total_optimizations} opts, {len(self.enabled_symbols)} symbols")
        except Exception as e:
            logger.warning(f"Could not load auto state: {e}")

    def _save_state(self):
        """Persist state to JSON."""
        try:
            state = {
                'engine_status': self.status,
                'total_cycles': self.total_cycles,
                'total_optimizations': self.total_optimizations,
                'symbol_results': self.symbol_results,
                'global_best': self.global_best[:50],
                'enabled_symbols': list(self.enabled_symbols),
            }
            with open(AUTO_STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save auto state: {e}")

    def start(self):
        """Start the autonomous loop."""
        if self.status == 'running':
            return
        self.status = 'running'
        self.task = asyncio.create_task(self._run_loop())
        logger.info("Autonomous backtest engine started")

    def stop(self):
        """Stop the autonomous loop."""
        self.status = 'stopped'
        if self.task and not self.task.done():
            self.task.cancel()
        self.task = None
        self.current_job = ''
        self.current_phase = ''
        self._save_state()
        logger.info("Autonomous backtest engine stopped")

    def pause(self):
        """Toggle pause/resume."""
        if self.status == 'running':
            self.status = 'paused'
            logger.info("Autonomous backtest engine paused")
        elif self.status == 'paused':
            self.status = 'running'
            logger.info("Autonomous backtest engine resumed")

    async def _run_loop(self):
        """Main async loop: build queue -> optimize each job -> learn -> save -> sleep."""
        try:
            while self.status in ('running', 'paused'):
                if self.status == 'paused':
                    await asyncio.sleep(2)
                    continue

                queue = self._build_job_queue()
                for symbol, interval in queue:
                    if self.status != 'running':
                        break

                    # Yield to manual backtests
                    if backtest_state.get('status') == 'running':
                        await broadcast({'type': 'auto_status', 'status': 'waiting',
                                         'message': 'Waiting for manual backtest...'})
                        while backtest_state.get('status') == 'running' and self.status == 'running':
                            await asyncio.sleep(2)
                        if self.status != 'running':
                            break

                    self.current_job = f'{symbol} {interval}'
                    await broadcast({'type': 'auto_status', 'status': 'running',
                                     'message': f'Optimizing {symbol} {interval}...',
                                     'current_job': self.current_job,
                                     'cycle': self.total_cycles + 1,
                                     'total_opts': self.total_optimizations})

                    try:
                        await asyncio.wait_for(
                            self._optimize_symbol(symbol, interval),
                            timeout=300  # 5 min timeout per job
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Auto optimization timed out for {symbol} {interval}")
                    except Exception as e:
                        logger.error(f"Auto optimization error for {symbol} {interval}: {e}")

                    await asyncio.sleep(5)  # 5s between jobs

                self.total_cycles += 1
                self._save_state()

                await broadcast({'type': 'auto_status', 'status': 'running',
                                 'message': f'Cycle {self.total_cycles} complete. Sleeping...',
                                 'cycle': self.total_cycles,
                                 'total_opts': self.total_optimizations})
                await broadcast({'type': 'auto_leaderboard',
                                 'leaderboard': self.global_best[:20]})

                # Telegram cycle summary
                top = self.global_best[0] if self.global_best else None
                top_str = f"Top: {top['symbol']} {top['interval']} {top['return_pct']:+.1f}%" if top else "No results yet"
                await send_telegram_alert(
                    title=f"Auto cycle {self.total_cycles} done",
                    message=f"Optimizations: {self.total_optimizations}. {top_str}",
                )

                # Sleep between cycles
                for _ in range(12):  # 60s total, check every 5s
                    if self.status != 'running':
                        break
                    await asyncio.sleep(5)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Autonomous loop crashed: {e}")
            self.status = 'stopped'
        finally:
            self.current_job = ''
            self.current_phase = ''

    def _build_job_queue(self) -> List[tuple]:
        """Prioritize symbols x intervals by staleness + performance + novelty."""
        jobs = []
        now = datetime.now()
        symbols = [s for s in AUTO_SYMBOLS if s in self.enabled_symbols]
        for symbol in symbols:
            # Asset-class-specific intervals
            if symbol.endswith('-USD'):
                intervals = ['5m', '15m', '1d']  # Coinbase supports intraday
            elif '/' in symbol:
                intervals = ['15m', '1d']  # Forex via OANDA
            else:
                intervals = AUTO_INTERVALS  # Equities
            for interval in intervals:
                sr = self.symbol_results.get(symbol, {}).get(interval, {})
                last_test = sr.get('last_test', '')
                test_count = sr.get('test_count', 0)
                best_return = sr.get('best_return', 0)

                # Staleness (40%): hours since last test / 24, capped at 1.0
                if last_test:
                    try:
                        hours_since = (now - datetime.fromisoformat(last_test)).total_seconds() / 3600
                    except Exception:
                        hours_since = 24
                else:
                    hours_since = 24
                staleness = min(hours_since / 24.0, 1.0)

                # Performance (30%): best return / 20, capped at 1.0
                performance = min(abs(best_return) / 20.0, 1.0)

                # Novelty (30%): 1 / (test_count + 1) — less-tested get priority
                novelty = 1.0 / (test_count + 1)

                priority = 0.4 * staleness + 0.3 * performance + 0.3 * novelty
                jobs.append((symbol, interval, priority))

        jobs.sort(key=lambda x: x[2], reverse=True)
        return [(s, i) for s, i, _ in jobs]

    def _fetch_data_cached(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch data with 1hr in-memory cache. Routes by asset class."""
        cache_key = f'{symbol}_{interval}'
        now = datetime.now()

        if cache_key in self._data_cache:
            df, ts = self._data_cache[cache_key]
            if (now - ts).total_seconds() < 3600:
                return df.copy()

        default_days = {'1m': 10, '5m': 30, '10m': 45, '15m': 45, '30m': 60, '1d': 365}.get(interval, 60)
        start = (now - timedelta(days=default_days)).strftime('%Y-%m-%d')
        end = now.strftime('%Y-%m-%d')

        df = None
        if '/' in symbol:
            # Forex: OANDA → yfinance
            df = fetch_oanda_data(symbol, start, end, interval)
            if df is None:
                df = fetch_yfinance_data(symbol, start, end, interval)
        elif symbol.endswith('-USD'):
            # Crypto: Coinbase → yfinance
            df = fetch_coinbase_data(symbol, start, end, interval)
            if df is None:
                df = fetch_yfinance_data(symbol, start, end, interval)
        else:
            # Equities: Schwab → Alpaca → yfinance
            df = fetch_schwab_data(symbol, start, end, interval)
            if df is None:
                df = fetch_alpaca_bars(symbol, start, end, interval)
            if df is None:
                df = fetch_yfinance_data(symbol, start, end, interval)

        if df is not None and not df.empty:
            self._data_cache[cache_key] = (df, now)
            return df.copy()
        return None

    async def _optimize_symbol(self, symbol: str, interval: str):
        """3-phase optimization: grid(128) -> refine(~81) -> OOS validation."""
        loop = asyncio.get_event_loop()

        # Fetch data
        self.current_phase = 'fetching'
        df = await loop.run_in_executor(None, self._fetch_data_cached, symbol, interval)
        if df is None or len(df) < 60:
            logger.warning(f"Auto: insufficient data for {symbol} {interval}")
            return

        trading_style = 'day' if interval in ('1m', '5m') else 'swing'

        # Phase 1: Grid search (128 combos)
        self.current_phase = 'grid_search'
        arrays = await loop.run_in_executor(None, precompute_arrays, df, trading_style, interval)
        param_grid = build_param_grid()
        grid_results = await loop.run_in_executor(None, run_grid_search, arrays, param_grid, 100000, 0.001, 3, 0.1)

        if not grid_results:
            logger.info(f"Auto: no valid results for {symbol} {interval}")
            self.total_optimizations += 1
            return

        # Phase 2: Refine top 3 (±30% finer steps, ~81 combos)
        self.current_phase = 'refining'
        refine_grid = self._refine_param_grid(grid_results[:3])
        if refine_grid:
            refine_results = await loop.run_in_executor(None, run_grid_search, arrays, refine_grid, 100000, 0.001, 3, 0.1)
            all_results = grid_results + refine_results
            all_results.sort(key=lambda r: r['return_pct'] - r['max_drawdown'] * 0.5, reverse=True)
        else:
            all_results = grid_results

        best = all_results[0]

        # Phase 3: Out-of-sample validation (60/40 split)
        self.current_phase = 'validation'
        oos_return = None
        overfit_flag = False
        split_idx = int(len(df) * 0.6)
        if split_idx > 60:
            oos_df = df.iloc[split_idx:].copy()
            if len(oos_df) > 30:
                oos_arrays = await loop.run_in_executor(None, precompute_arrays, oos_df, trading_style, interval)
                oos_result = await loop.run_in_executor(None, quick_backtest, best['params'], oos_arrays, 100000, 0.001, 0.1)
                oos_return = oos_result.get('return_pct', 0)
                # Flag overfit if OOS return is < 30% of in-sample or negative when IS is positive
                if best['return_pct'] > 0 and oos_return < best['return_pct'] * 0.3:
                    overfit_flag = True

        self.total_optimizations += 1

        # Feed brain
        if self.brain and BRAIN_AVAILABLE:
            self._feed_brain(symbol, interval, best, oos_return, overfit_flag)

        # Update state
        self._update_state(symbol, interval, best, oos_return, overfit_flag)

        # Fetch sentiment for the symbol
        try:
            sentiment = await fetch_news_sentiment(symbol)
            sentiment_score = sentiment.get('overall_score', 0)
            sentiment_label = sentiment.get('overall_label', 'N/A')
        except Exception:
            sentiment_score = 0
            sentiment_label = 'N/A'

        # Broadcast result
        result_entry = {
            'symbol': symbol,
            'interval': interval,
            'return_pct': best['return_pct'],
            'win_rate': best['win_rate'],
            'max_drawdown': best['max_drawdown'],
            'profit_factor': best['profit_factor'],
            'num_trades': best['num_trades'],
            'oos_return': oos_return,
            'overfit': overfit_flag,
            'params': best['params'],
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
        }
        await broadcast({'type': 'auto_result', 'result': result_entry})
        await broadcast({'type': 'auto_leaderboard', 'leaderboard': self.global_best[:20]})

        # Telegram alert for strong results
        if best['return_pct'] > 5 and not overfit_flag:
            oos_str = f", OOS: {oos_return:+.1f}%" if oos_return is not None else ""
            await send_telegram_alert(
                title=f"Strong result: {symbol} {interval}",
                message=f"Return: {best['return_pct']:+.1f}%, WR: {best['win_rate']}%, DD: {best['max_drawdown']}%, PF: {best['profit_factor']}{oos_str}",
            )

    def _refine_param_grid(self, top_results: list) -> list:
        """Generate ±30% finer-step combos around top performers."""
        refined = []
        seen = set()
        for r in top_results:
            p = r['params']
            base_stop = p.get('atr_stop_mult', 2.0)
            base_target = p.get('atr_target_mult', 3.0)
            base_pb = p.get('pullback_zone_atr', 1.0)

            for stop_delta in [-0.3, 0, 0.3]:
                for target_delta in [-0.5, 0, 0.5]:
                    for pb_delta in [-0.15, 0, 0.15]:
                        new_stop = round(max(0.5, base_stop + stop_delta), 2)
                        new_target = round(max(1.0, base_target + target_delta), 2)
                        new_pb = round(max(0.2, base_pb + pb_delta), 2)
                        key = (new_stop, new_target, p.get('min_triggers', 2), new_pb)
                        if key not in seen:
                            seen.add(key)
                            refined.append({
                                'atr_stop_mult': new_stop,
                                'atr_target_mult': new_target,
                                'min_triggers': p.get('min_triggers', 2),
                                'pullback_zone_atr': new_pb,
                                'min_trend_adx': p.get('min_trend_adx', 20),
                                'min_vol_ratio': p.get('min_vol_ratio', 0.8),
                            })
        return refined

    def _feed_brain(self, symbol: str, interval: str, best: dict, oos_return, overfit: bool):
        """Feed optimization results to TradingBrain as learned patterns."""
        try:
            p = best['params']
            pattern_name = (f"auto_rules_{interval}_"
                            f"stop{p.get('atr_stop_mult','?')}_"
                            f"tgt{p.get('atr_target_mult','?')}_"
                            f"trig{p.get('min_triggers','?')}")
            self.brain.remember_trade(
                symbol=symbol,
                pattern=pattern_name,
                outcome='profitable' if best['return_pct'] > 0 else 'loss',
                pnl_percent=best['return_pct'],
                context={
                    'action': 'optimization',
                    'interval': interval,
                    'win_rate': best['win_rate'],
                    'max_drawdown': best['max_drawdown'],
                    'profit_factor': best['profit_factor'],
                    'num_trades': best['num_trades'],
                    'oos_return': oos_return,
                    'overfit': overfit,
                    'params': best['params'],
                }
            )
        except Exception as e:
            logger.warning(f"Brain learning failed: {e}")

    def _update_state(self, symbol: str, interval: str, best: dict, oos_return, overfit: bool):
        """Update best params, history, and global leaderboard."""
        now_str = datetime.now().isoformat()

        if symbol not in self.symbol_results:
            self.symbol_results[symbol] = {}
        if interval not in self.symbol_results[symbol]:
            self.symbol_results[symbol][interval] = {
                'best_params': {},
                'best_return': -999,
                'test_count': 0,
                'last_test': '',
                'history': [],
            }

        sr = self.symbol_results[symbol][interval]
        sr['test_count'] += 1
        sr['last_test'] = now_str

        # Update best if this result beats the current best
        if best['return_pct'] > sr.get('best_return', -999):
            sr['best_params'] = best['params']
            sr['best_return'] = best['return_pct']

        # Append to history (cap at 20)
        sr['history'].append({
            'date': now_str,
            'return_pct': best['return_pct'],
            'win_rate': best['win_rate'],
            'max_drawdown': best['max_drawdown'],
            'profit_factor': best['profit_factor'],
            'num_trades': best['num_trades'],
            'oos_return': oos_return,
            'overfit': overfit,
            'params': best['params'],
        })
        if len(sr['history']) > 20:
            sr['history'] = sr['history'][-20:]

        # Update global leaderboard
        entry = {
            'symbol': symbol,
            'interval': interval,
            'return_pct': best['return_pct'],
            'win_rate': best['win_rate'],
            'max_drawdown': best['max_drawdown'],
            'profit_factor': best['profit_factor'],
            'num_trades': best['num_trades'],
            'oos_return': oos_return,
            'overfit': overfit,
            'params': best['params'],
            'date': now_str,
            'test_count': sr['test_count'],
        }
        # Remove old entry for same symbol+interval
        self.global_best = [e for e in self.global_best
                            if not (e['symbol'] == symbol and e['interval'] == interval)]
        self.global_best.append(entry)
        self.global_best.sort(key=lambda e: e['return_pct'] - e['max_drawdown'] * 0.5, reverse=True)
        self.global_best = self.global_best[:50]

    def get_state(self) -> dict:
        """Return current engine state for API."""
        return {
            'status': self.status,
            'current_job': self.current_job,
            'current_phase': self.current_phase,
            'total_cycles': self.total_cycles,
            'total_optimizations': self.total_optimizations,
            'symbol_results': self.symbol_results,
            'global_best': self.global_best[:20],
            'enabled_symbols': list(self.enabled_symbols),
            'all_symbols': AUTO_SYMBOLS,
        }


# Singleton autonomous engine
auto_engine = AutonomousBacktestEngine()


# =============================================================================
# LIVE PAPER TRADING ENGINE
# =============================================================================

PAPER_STATE_FILE = os.path.join(os.path.dirname(__file__) or '.', 'paper_trade_state.json')

# Poll frequency (seconds) by interval
PAPER_POLL_FREQ = {
    '1m': 15, '5m': 30, '15m': 60, '30m': 90, '1h': 120, '1d': 300,
}

INTERVAL_DELTA = {
    '1m': timedelta(minutes=1), '5m': timedelta(minutes=5),
    '10m': timedelta(minutes=10), '15m': timedelta(minutes=15),
    '30m': timedelta(minutes=30), '1h': timedelta(hours=1),
    '1d': timedelta(days=1),
}


def _fetch_recent_bars(symbol: str, interval: str, count: int = 100) -> Optional[pd.DataFrame]:
    """Fetch *count* recent bars for live paper trading.  Routes by asset class."""
    now = datetime.now()
    # Over-fetch to ensure we have enough bars after indicator lookback
    extra_days = {
        '1m': max(2, count // 390 + 2),
        '5m': max(5, count // 78 + 2),
        '15m': max(10, count // 26 + 2),
        '30m': max(15, count // 13 + 2),
        '1h': max(30, count // 7 + 2),
        '1d': max(200, count + 100),
    }.get(interval, 30)
    start = (now - timedelta(days=extra_days)).strftime('%Y-%m-%d')
    end = now.strftime('%Y-%m-%d')

    df = None
    if '/' in symbol:
        df = fetch_oanda_data(symbol, start, end, interval)
        if df is None:
            df = fetch_yfinance_data(symbol, start, end, interval)
    elif symbol.endswith('-USD'):
        df = fetch_coinbase_data(symbol, start, end, interval)
        if df is None:
            df = fetch_yfinance_data(symbol, start, end, interval)
    else:
        df = fetch_schwab_data(symbol, start, end, interval)
        if df is None:
            df = fetch_alpaca_bars(symbol, start, end, interval)
        if df is None:
            df = fetch_yfinance_data(symbol, start, end, interval)
    return df


def _is_market_open(symbol: str) -> bool:
    """Simple check: crypto always open, forex/equities Mon-Fri only."""
    if symbol.endswith('-USD'):
        return True  # crypto 24/7
    now = datetime.now()
    weekday = now.weekday()  # 0=Mon, 6=Sun
    if weekday >= 5:
        return False  # Sat/Sun
    if '/' not in symbol:
        # Equities: rough 9:30-16:00 ET check (use UTC-5 approximation)
        from datetime import timezone, timedelta as _td
        try:
            et_now = datetime.now(timezone(timedelta(hours=-5)))
            hour, minute = et_now.hour, et_now.minute
            t = hour * 60 + minute
            if t < 9 * 60 + 30 or t > 16 * 60:
                return False
        except Exception:
            pass
    return True


class PaperTrader:
    """Live paper trading engine — polls for new bars, evaluates rules, streams to dashboard.

    Supports multiple symbols simultaneously with a shared cash pool.
    """

    def __init__(self):
        self.status = 'stopped'  # stopped | running | paused | market_closed
        self.task: Optional[asyncio.Task] = None
        self.symbols: List[str] = []
        self.active_chart_symbol: str = ''
        self.interval = '5m'
        self.initial_capital = 100000.0
        self.commission_pct = 0.001
        self.rules_params: dict = {}
        self.symbol_states: Dict[str, SymbolState] = {}
        self.cash = 100000.0
        self.trades: List[dict] = []
        self.equity_curve: List[dict] = []
        self.total_bars_processed = 0
        self._streamer = None  # SchwabTickStreamer or AlpacaTickStreamer
        self._news_streamer: Optional[AlpacaNewsStreamer] = None
        self._mkt_context: Optional[MarketContext] = None
        self._load_state()

    # Backwards-compat property: returns active chart symbol
    @property
    def symbol(self) -> str:
        return self.active_chart_symbol

    @symbol.setter
    def symbol(self, val: str):
        self.active_chart_symbol = val
        if val and val not in self.symbols:
            self.symbols = [val]

    def _per_symbol_budget(self) -> float:
        """Max budget per symbol for position sizing — divides cash evenly."""
        n = max(len(self.symbols), 1)
        return self.cash * 0.95 / n

    # --- State persistence ---
    def _load_state(self):
        try:
            if os.path.exists(PAPER_STATE_FILE):
                with open(PAPER_STATE_FILE, 'r') as f:
                    s = json.load(f)
                version = s.get('version', 1)
                self.interval = s.get('interval', '5m')
                self.initial_capital = s.get('initial_capital', 100000.0)
                self.commission_pct = s.get('commission_pct', 0.001)
                self.rules_params = s.get('rules_params', {})
                self.cash = s.get('cash', self.initial_capital)
                self.trades = s.get('trades', [])
                self.equity_curve = s.get('equity_curve', [])
                self.total_bars_processed = s.get('bars_processed', s.get('total_bars_processed', 0))

                if version >= 2:
                    # Multi-symbol v2 format
                    self.symbols = s.get('symbols', [])
                    self.active_chart_symbol = s.get('active_chart_symbol', self.symbols[0] if self.symbols else '')
                    for sym, ss_data in s.get('symbol_states', {}).items():
                        self.symbol_states[sym] = SymbolState(
                            symbol=sym,
                            pos=PositionState(
                                position=ss_data.get('position', 0),
                                entry_price=ss_data.get('entry_price', 0.0),
                                r_value=ss_data.get('r_value', 0.0),
                                trailing_stop=ss_data.get('trailing_stop', 0.0),
                                highest_high=ss_data.get('highest_high', 0.0),
                                lowest_low=ss_data.get('lowest_low', 0.0),
                                bars_since_exit=ss_data.get('bars_since_exit', 999),
                            ),
                            last_bar_time=ss_data.get('last_bar_time'),
                            bars_processed=ss_data.get('bars_processed', 0),
                        )
                else:
                    # Legacy v1 single-symbol format
                    sym = s.get('symbol', '')
                    if sym:
                        self.symbols = [sym]
                        self.active_chart_symbol = sym
                        self.symbol_states[sym] = SymbolState(
                            symbol=sym,
                            pos=PositionState(
                                position=s.get('position', 0),
                                entry_price=s.get('entry_price', 0.0),
                                r_value=s.get('r_value', 0.0),
                                trailing_stop=s.get('trailing_stop', 0.0),
                                highest_high=s.get('highest_high', 0.0),
                                lowest_low=s.get('lowest_low', 0.0),
                                bars_since_exit=s.get('bars_since_exit', 999),
                            ),
                            last_bar_time=s.get('last_bar_time'),
                            bars_processed=s.get('bars_processed', 0),
                        )

                # If engine was running when server stopped, mark as stopped
                saved_status = s.get('status', 'stopped')
                if saved_status in ('running', 'paused'):
                    self.status = 'stopped'  # will be resumed by start()
                sym_str = ','.join(self.symbols) if self.symbols else '(none)'
                logger.info(f"Paper trader state loaded: {sym_str} {self.interval}, "
                            f"{self.total_bars_processed} bars")
        except Exception as e:
            logger.warning(f"Could not load paper trade state: {e}")

    def _save_state(self):
        try:
            ss_data = {}
            for sym, ss in self.symbol_states.items():
                ss_data[sym] = {
                    'position': ss.pos.position,
                    'entry_price': ss.pos.entry_price,
                    'r_value': ss.pos.r_value,
                    'trailing_stop': ss.pos.trailing_stop,
                    'highest_high': ss.pos.highest_high,
                    'lowest_low': ss.pos.lowest_low,
                    'bars_since_exit': ss.pos.bars_since_exit,
                    'last_bar_time': ss.last_bar_time,
                    'bars_processed': ss.bars_processed,
                }
            state = {
                'version': 2,
                'status': self.status,
                'symbols': self.symbols,
                'active_chart_symbol': self.active_chart_symbol,
                'interval': self.interval,
                'initial_capital': self.initial_capital,
                'commission_pct': self.commission_pct,
                'rules_params': self.rules_params,
                'cash': self.cash,
                'symbol_states': ss_data,
                'trades': self.trades[-200:],  # cap
                'equity_curve': self.equity_curve[-500:],
                'total_bars_processed': self.total_bars_processed,
            }
            with open(PAPER_STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save paper state: {e}")

    # --- Control ---
    def start(self, config: dict):
        if self.status == 'running':
            return
        # Accept 'symbols' (list) or 'symbol' (string, backwards compat)
        raw_symbols = config.get('symbols') or [config.get('symbol', 'BTC-USD')]
        if isinstance(raw_symbols, str):
            raw_symbols = [s.strip() for s in raw_symbols.split(',') if s.strip()]
        self.symbols = [s.upper() for s in raw_symbols]
        self.active_chart_symbol = self.symbols[0] if self.symbols else ''
        self.interval = config.get('interval', '5m')
        self.initial_capital = config.get('initial_capital', 100000.0)
        self.commission_pct = config.get('commission_pct', 0.1) / 100
        self.rules_params = config.get('rules_params', {})
        # Reset if symbol list or interval changed from saved state
        saved_syms = []
        try:
            if os.path.exists(PAPER_STATE_FILE):
                with open(PAPER_STATE_FILE, 'r') as f:
                    saved = json.load(f)
                    saved_syms = saved.get('symbols', [saved.get('symbol', '')])
                    saved_int = saved.get('interval', '')
                if set(saved_syms) != set(self.symbols) or saved_int != self.interval:
                    self._reset_account()
        except Exception:
            pass
        if not saved_syms:
            self._reset_account()
        # Ensure SymbolState exists for each symbol
        for sym in self.symbols:
            if sym not in self.symbol_states:
                self.symbol_states[sym] = SymbolState(symbol=sym)
        self.status = 'running'
        self.task = asyncio.create_task(self._run_loop())
        logger.info(f"Paper trader started: {','.join(self.symbols)} {self.interval}")

    def stop(self):
        self.status = 'stopped'
        # Close tick/news streamers first to free Alpaca connection slots
        if self._streamer:
            asyncio.ensure_future(self._streamer.stop())
            self._streamer = None
        if self._news_streamer:
            asyncio.ensure_future(self._news_streamer.stop())
            self._news_streamer = None
        if self.task and not self.task.done():
            self.task.cancel()
        self.task = None
        self._save_state()
        logger.info("Paper trader stopped")

    def pause(self):
        if self.status == 'running':
            self.status = 'paused'
        elif self.status == 'paused':
            self.status = 'running'

    def reset(self):
        self.stop()
        self._reset_account()
        self._save_state()

    def _reset_account(self):
        self.cash = self.initial_capital
        self.symbol_states = {sym: SymbolState(symbol=sym) for sym in self.symbols}
        self.trades = []
        self.equity_curve = []
        self.total_bars_processed = 0

    # --- Execute trade on shared cash pool ---
    def _execute_trade(self, sym: str, sig: int, ss: SymbolState,
                       cp: float, atr: float, bar_date: str, row) -> Optional[dict]:
        """Execute a trade for the given symbol. Returns trade dict or None."""
        trade_msg = None
        pos = ss.pos
        budget = self._per_symbol_budget()
        params = self.rules_params or {}
        adaptive = params.get('adaptive_risk', False)
        eff_stop_m = _effective_stop_mult(params, ss.arrays, len(ss.df) - 1) if ss.arrays else params.get('atr_stop_mult', 2.0)

        def _adaptive_shares(price):
            """Vol-normalized sizing: risk fixed % of equity per trade."""
            if not adaptive or not params.get('risk_per_trade'):
                return int(budget / price) if price > 0 else 0
            eq = self._current_equity_all()
            risk_amt = eq * params.get('risk_per_trade', 0.02)
            stop_dist = atr * eff_stop_m
            if stop_dist <= 0:
                return 0
            sized = int(risk_amt / stop_dist)
            max_shares = int(budget / price) if price > 0 else 0
            return min(sized, max_shares)

        if sig == 1 and pos.position == 0:
            # Open long
            shares = _adaptive_shares(cp)
            if shares > 0:
                cost = shares * cp * (1 + self.commission_pct)
                if cost <= self.cash:
                    self.cash -= cost
                    pos.position = shares
                    pos.entry_price = cp
                    pos.r_value = atr * eff_stop_m
                    pos.trailing_stop = cp - pos.r_value
                    pos.highest_high = float(row['high'])
                    trade_msg = {'date': bar_date, 'type': 'BUY', 'price': cp,
                                 'shares': shares, 'cost': cost, 'pnl': 0, 'symbol': sym}
                    self.trades.append(trade_msg)

        elif sig == 2 and pos.position == 0:
            # Open short
            shares = _adaptive_shares(cp)
            if shares > 0:
                proceeds = shares * cp * (1 - self.commission_pct)
                self.cash += proceeds
                pos.position = -shares
                pos.entry_price = cp
                pos.r_value = atr * eff_stop_m
                pos.trailing_stop = cp + pos.r_value
                pos.lowest_low = float(row['low'])
                trade_msg = {'date': bar_date, 'type': 'SHORT', 'price': cp,
                             'shares': shares, 'proceeds': proceeds, 'pnl': 0, 'symbol': sym}
                self.trades.append(trade_msg)

        elif sig == 2 and pos.position > 0:
            # Close long
            proceeds = pos.position * cp * (1 - self.commission_pct)
            pnl = proceeds - (pos.position * pos.entry_price * (1 + self.commission_pct))
            self.cash += proceeds
            trade_msg = {'date': bar_date, 'type': 'SELL', 'price': cp,
                         'shares': pos.position, 'proceeds': proceeds, 'pnl': round(pnl, 2),
                         'return_pct': round(((cp / pos.entry_price) - 1) * 100, 2), 'symbol': sym}
            self.trades.append(trade_msg)
            pos.position = 0
            pos.entry_price = 0
            pos.bars_since_exit = 0

        elif sig == 1 and pos.position < 0:
            # Cover short
            s = abs(pos.position)
            cost = s * cp * (1 + self.commission_pct)
            entry_proceeds = s * pos.entry_price * (1 - self.commission_pct)
            pnl = entry_proceeds - cost
            self.cash -= cost
            trade_msg = {'date': bar_date, 'type': 'COVER', 'price': cp,
                         'shares': s, 'cost': cost, 'pnl': round(pnl, 2),
                         'return_pct': round(((pos.entry_price / cp) - 1) * 100, 2), 'symbol': sym}
            self.trades.append(trade_msg)
            pos.position = 0
            pos.entry_price = 0
            pos.bars_since_exit = 0

        return trade_msg

    # --- Process a single new bar for a symbol ---
    async def _process_bar(self, sym: str, new_bars, bar_idx: int, trading_style: str):
        """Process one new bar for the given symbol. Broadcasts bar/trade/decision."""
        ss = self.symbol_states[sym]
        row = new_bars.iloc[bar_idx]
        bar_date = str(new_bars.index[bar_idx])

        # Append to rolling DataFrame (cap at 250 rows), skip duplicates
        bar_ts = new_bars.index[bar_idx]
        if ss.df is not None and bar_ts in ss.df.index:
            return  # already processed this bar
        new_row = pd.DataFrame({
            'open': [float(row['open'])],
            'high': [float(row['high'])],
            'low': [float(row['low'])],
            'close': [float(row['close'])],
            'volume': [float(row.get('volume', 0))],
        }, index=[bar_ts])
        new_row.index.name = sym
        ss.df = pd.concat([ss.df, new_row])
        if len(ss.df) > 250:
            ss.df = ss.df.iloc[-250:]

        # Recompute indicators on full rolling window
        BacktestRunner._precompute_indicators(ss.df)
        ss.arrays = precompute_arrays(ss.df, trading_style, self.interval, lookback=50)

        i = len(ss.df) - 1
        if i < ss.arrays.lookback:
            return

        cp = float(row['close'])
        atr = _nan(ss.arrays.atr[i])
        if atr <= 0:
            atr = cp * 0.01

        # Evaluate signal — use per-symbol params if available
        _rp = self.rules_params or {}
        _per = _rp.get('per_symbol', {})
        if sym in _per:
            _eff_rp = {k: v for k, v in _rp.items() if k != 'per_symbol'}
            _eff_rp.update(_per[sym])
        else:
            _eff_rp = {k: v for k, v in _rp.items() if k != 'per_symbol'}

        # --- Overnight gap protection (universal) ---
        _max_gap_pct = _eff_rp.get('max_gap_pct', 0.0)
        if _max_gap_pct > 0 and ss.pos.position != 0 and ss.df is not None and len(ss.df) >= 2:
            _prev_close = float(ss.df['close'].iloc[-2])
            _bar_open = float(row['open'])
            if _prev_close > 0:
                _gap_pct = (_bar_open - _prev_close) / _prev_close
                _gap_exit = False
                if ss.pos.position > 0 and _gap_pct < -_max_gap_pct:
                    _gap_exit = True
                elif ss.pos.position < 0 and _gap_pct > _max_gap_pct:
                    _gap_exit = True
                if _gap_exit:
                    force_sig = 2 if ss.pos.position > 0 else 1
                    trade_msg = self._execute_trade(sym, force_sig, ss, _bar_open, atr, bar_date, row)
                    if trade_msg:
                        trade_msg['reasoning'] = f'Gap protection: {_gap_pct*100:+.1f}% gap exceeded {_max_gap_pct*100:.1f}% threshold'
                        trade_msg['exit_reason'] = 'gap_protection'
                        await broadcast({'type': 'trade', 'trade': trade_msg})
                    await broadcast({
                        'type': 'decision', 'symbol': sym, 'bar': 0, 'total_bars': 'LIVE',
                        'date': bar_date, 'price': round(_bar_open, 2), 'signal': 'GAP_EXIT',
                        'strength': 0, 'reasoning': f'Forced exit: {_gap_pct*100:+.1f}% adverse gap at open',
                        'risk_notes': '', 'regime': '', 'regime_confidence': '',
                        'source': 'gap_protection', 'indicators': '', 'position': 'FLAT',
                        'action': trade_msg['type'] if trade_msg else 'GAP_EXIT',
                    })
                    return  # skip further processing this bar

        # --- Max trade loss cap (universal) ---
        _max_loss_pct = _eff_rp.get('max_trade_loss_pct', 0.0)
        if _max_loss_pct > 0 and ss.pos.position != 0:
            if ss.pos.position > 0:
                _unr_pnl = (cp - ss.pos.entry_price) * ss.pos.position
            else:
                _unr_pnl = (ss.pos.entry_price - cp) * abs(ss.pos.position)
            _eq_now = self._current_equity_all()
            _entry_eq = _eq_now - _unr_pnl if (_eq_now - _unr_pnl) > 0 else self.initial_capital
            if _unr_pnl < 0 and abs(_unr_pnl) > _entry_eq * _max_loss_pct:
                force_sig = 2 if ss.pos.position > 0 else 1
                trade_msg = self._execute_trade(sym, force_sig, ss, cp, atr, bar_date, row)
                if trade_msg:
                    trade_msg['reasoning'] = f'Max loss cap: unrealized loss ${abs(_unr_pnl):.0f} exceeded {_max_loss_pct*100:.1f}% of equity'
                    trade_msg['exit_reason'] = 'max_loss_cap'
                    await broadcast({'type': 'trade', 'trade': trade_msg})
                await broadcast({
                    'type': 'decision', 'symbol': sym, 'bar': 0, 'total_bars': 'LIVE',
                    'date': bar_date, 'price': round(cp, 2), 'signal': 'MAX_LOSS_EXIT',
                    'strength': 0, 'reasoning': f'Forced exit: loss ${abs(_unr_pnl):.0f} exceeded {_max_loss_pct*100:.1f}% cap',
                    'risk_notes': '', 'regime': '', 'regime_confidence': '',
                    'source': 'max_loss_cap', 'indicators': '', 'position': 'FLAT',
                    'action': trade_msg['type'] if trade_msg else 'MAX_LOSS_EXIT',
                })
                return  # skip further processing this bar

        sig = evaluate_rules_signal(_eff_rp, ss.arrays, i, ss.pos,
                                    mkt=self._mkt_context)

        # --- News sentiment entry filter (paper trading only) ---
        _news_threshold = _eff_rp.get('news_sentiment_threshold', 0)
        _news_reason = ''
        if _news_threshold > 0 and sig != 0 and ss.pos.position == 0:
            try:
                _sentiment = await fetch_news_sentiment(sym)
                _score = _sentiment.get('overall_score', 0)
                _label = _sentiment.get('overall_label', 'Neutral')
                if sig == 1 and _score < -_news_threshold:
                    _news_reason = f'News BLOCKED BUY: sentiment={_score:.0f} ({_label}), threshold=-{_news_threshold}'
                    sig = 0
                elif sig == 2 and _score > _news_threshold:
                    _news_reason = f'News BLOCKED SHORT: sentiment=+{_score:.0f} ({_label}), threshold=+{_news_threshold}'
                    sig = 0
                else:
                    _news_reason = f'News OK: {_score:.0f} ({_label})'
            except Exception as e:
                logger.debug(f"News sentiment fetch failed for {sym}: {e}")
                _news_reason = f'News unavailable'

        # Execute trade on shared cash pool
        trade_msg = self._execute_trade(sym, sig, ss, cp, atr, bar_date, row)

        # Record aggregate equity
        eq = self._current_equity_all()
        self.equity_curve.append({'date': bar_date, 'value': round(eq, 2)})

        # Aggregate buy-hold across all symbols
        total_bh = self._total_buy_hold()

        # Broadcast bar for ALL symbols (frontend routes to correct chart)
        await broadcast({
            'type': 'bar',
            'symbol': sym,
            'date': bar_date,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': cp,
            'volume': float(row.get('volume', 0)),
            'equity': eq,
            'buy_hold': total_bh,
        })

        # Route to Alpaca paper account (equities only)
        if trade_msg:
            alp_sym = _alpaca_symbol(sym)
            if alp_sym is not None:
                alp_side = 'buy' if trade_msg['type'] in ('BUY', 'COVER') else 'sell'
                alp_result = _alpaca_place_order(alp_sym, trade_msg['shares'], alp_side)
                trade_msg['alpaca_order'] = alp_result

        # Broadcast trade
        if trade_msg:
            await broadcast({'type': 'trade', 'trade': trade_msg})

        # Broadcast decision
        sig_name = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(sig, 'HOLD')
        reasoning = describe_signal_reasoning(ss.arrays, i, self.rules_params or {})
        pos_text = 'FLAT'
        if ss.pos.position > 0:
            pos_text = f'LONG {ss.pos.position} @ ${ss.pos.entry_price:.2f}'
        elif ss.pos.position < 0:
            pos_text = f'SHORT {abs(ss.pos.position)} @ ${ss.pos.entry_price:.2f}'

        await broadcast({
            'type': 'decision',
            'symbol': sym,
            'bar': ss.bars_processed + bar_idx + 1,
            'total_bars': 'LIVE',
            'date': bar_date,
            'price': round(cp, 2),
            'signal': sig_name,
            'strength': round(min(0.95, 0.5 + 0.15 * max(0, ss.pos.last_score - _rp.get('min_triggers', 2))), 2) if sig != 0 else 0,
            'reasoning': reasoning,
            'risk_notes': _news_reason,
            'regime': '',
            'regime_confidence': '',
            'source': 'rules_engine',
            'indicators': f"RSI:{_nan(ss.arrays.rsi[i]):.1f} MACD:{_nan(ss.arrays.macd_hist[i]):.3f} ADX:{_nan(ss.arrays.adx[i]):.1f} ATR:{atr:.4f}",
            'position': pos_text,
            'action': trade_msg['type'] if trade_msg else ('HOLDING ' + pos_text if ss.pos.position != 0 else 'STAYING FLAT'),
        })

        ss.last_bar_time = bar_date

    # --- Tick streamer management ---
    async def _start_tick_streamer(self, symbols=None):
        """Start tick streamer — prefers Schwab, falls back to Alpaca."""
        if self._streamer:
            await self._streamer.stop()
            self._streamer = None

        # Filter to equity symbols only (no crypto/forex)
        if symbols is None:
            symbols = list(self.symbol_states.keys())
        equity_syms = [s for s in symbols
                       if not s.endswith('-USD') and '/' not in s
                       and s in self.symbol_states]
        if not equity_syms:
            return

        async def _on_tick(sym: str, price: float):
            ss = self.symbol_states.get(sym)
            if not ss:
                return
            if ss.forming is None:
                ss.forming = {'open': price, 'high': price,
                              'low': price, 'close': price}
            else:
                ss.forming['high'] = max(ss.forming['high'], price)
                ss.forming['low'] = min(ss.forming['low'], price)
                ss.forming['close'] = price
            delta = INTERVAL_DELTA.get(self.interval, timedelta(minutes=1))
            forming_time = str(pd.Timestamp(ss.last_bar_time) + delta) if ss.last_bar_time else ''
            eq = self._current_equity_all()
            total_bh = self._total_buy_hold()
            await broadcast({
                'type': 'bar',
                'symbol': sym,
                'date': forming_time,
                'open': ss.forming['open'], 'high': ss.forming['high'],
                'low': ss.forming['low'], 'close': ss.forming['close'],
                'volume': 0,
                'equity': eq, 'buy_hold': total_bh,
            })
            await self._broadcast_aggregate_stats()

        # Prefer Schwab streaming
        if SCHWAB_AVAILABLE and get_schwab_client() is not None:
            self._streamer = SchwabTickStreamer(equity_syms, on_tick=_on_tick)
            await self._streamer.start()
            logger.info(f"Tick streamer: Schwab for {equity_syms}")
            return

        # Fall back to Alpaca
        if _get_alpaca_config() is not None:
            self._streamer = AlpacaTickStreamer(equity_syms, on_tick=_on_tick)
            await self._streamer.start()
            logger.info(f"Tick streamer: Alpaca for {equity_syms}")
        else:
            logger.warning("No tick streamer: neither Schwab nor Alpaca configured")

    # --- News streamer management ---
    async def _start_news_streamer(self, symbols=None):
        """Start news streamer for equity symbols (stops existing one first)."""
        if self._news_streamer:
            await self._news_streamer.stop()
            self._news_streamer = None

        if _get_alpaca_config() is None:
            return

        syms = symbols or list(self.symbol_states.keys())
        equity_syms = [s for s in syms
                       if not s.endswith('-USD') and '/' not in s]
        if not equity_syms:
            return

        async def _on_news(news_item: dict):
            await broadcast({'type': 'news', **news_item})

        self._news_streamer = AlpacaNewsStreamer(equity_syms, on_news=_on_news)
        await self._news_streamer.start()

    # --- Chart symbol switching (legacy, kept for API compat) ---
    async def switch_chart_symbol(self, sym: str):
        """No-op in grid mode — all symbols are displayed simultaneously."""
        if sym in self.symbol_states:
            self.active_chart_symbol = sym

    # --- Main loop ---
    async def _run_loop(self):
        streamer = None
        try:
            sym_str = ', '.join(self.symbols)
            await broadcast({'type': 'paper_status', 'status': 'running',
                             'message': f'Fetching seed data for {sym_str} {self.interval}...'})

            loop = asyncio.get_event_loop()
            trading_style = 'day' if self.interval in ('1m', '5m') else 'swing'

            # Phase 1: Fetch seed bars for ALL symbols concurrently
            import concurrent.futures
            fetch_tasks = {
                sym: loop.run_in_executor(None, _fetch_recent_bars, sym, self.interval, 200)
                for sym in self.symbols
            }
            fetch_results = {}
            for sym, coro in fetch_tasks.items():
                fetch_results[sym] = await coro

            # Initialize each symbol
            active_symbols = []
            for sym in self.symbols:
                df = fetch_results.get(sym)
                if df is None or len(df) < 60:
                    await broadcast({'type': 'paper_status', 'status': 'running',
                                     'message': f'Warning: insufficient data for {sym}, skipping'})
                    continue
                ss = self.symbol_states[sym]
                ss.df = df
                BacktestRunner._precompute_indicators(ss.df)
                ss.arrays = precompute_arrays(ss.df, trading_style, self.interval, lookback=50)
                ss.bh_shares = (self.initial_capital / len(self.symbols)) / float(ss.df['close'].iloc[0])
                ss.last_bar_time = str(ss.df.index[-1])
                ss.bars_processed = len(ss.df)
                active_symbols.append(sym)

            if not active_symbols:
                await broadcast({'type': 'paper_status', 'status': 'error',
                                 'message': 'No symbols had sufficient data'})
                self.status = 'stopped'
                return

            # Ensure active_chart_symbol is valid
            if self.active_chart_symbol not in active_symbols:
                self.active_chart_symbol = active_symbols[0]

            # Broadcast seed bars for ALL symbols (frontend routes to correct charts)
            for sym in active_symbols:
                ss_seed = self.symbol_states[sym]
                for idx in range(len(ss_seed.df)):
                    row = ss_seed.df.iloc[idx]
                    bar_date = str(ss_seed.df.index[idx])
                    cp = float(row['close'])
                    eq = self._current_equity_all()
                    total_bh = self._total_buy_hold()
                    await broadcast({
                        'type': 'bar',
                        'symbol': sym,
                        'date': bar_date,
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': cp,
                        'volume': float(row['volume']) if 'volume' in ss_seed.df.columns else 0,
                        'equity': eq,
                        'buy_hold': total_bh,
                    })
                    if idx % 50 == 0:
                        await asyncio.sleep(0)
                await broadcast({'type': 'seed_complete', 'symbol': sym})

            self.total_bars_processed = max(ss.bars_processed for ss in self.symbol_states.values())

            # Send initial stats
            await self._broadcast_aggregate_stats()

            # Fetch market context (SPY + VIX) for adaptive risk
            if self.rules_params.get('adaptive_risk'):
                equity_syms = [s for s in active_symbols
                               if not s.endswith('-USD') and '/' not in s]
                if equity_syms:
                    first_ss = next((ss for ss in self.symbol_states.values() if ss.df is not None), None)
                    if first_ss and first_ss.df is not None:
                        self._mkt_context = fetch_market_context(first_ss.df, self.interval)
                        if self._mkt_context:
                            logger.info(f"Paper trader: loaded market context ({self._mkt_context.n} bars)")

            # Phase 2: Tick stream + bar-check loop
            bar_check_freq = PAPER_POLL_FREQ.get(self.interval, 60)

            # Start tick streamer for all equity symbols
            await self._start_tick_streamer(active_symbols)
            await self._start_news_streamer(active_symbols)
            streamer = self._streamer

            has_tick = self._streamer is not None
            if has_tick:
                tick_label = 'Schwab streaming' if isinstance(self._streamer, SchwabTickStreamer) else 'Alpaca streaming'
            else:
                tick_label = f'bar/{bar_check_freq}s'
            await broadcast({'type': 'paper_status', 'status': 'running',
                             'message': f'Live — {tick_label} — {len(active_symbols)} symbol(s)'})

            while self.status in ('running', 'paused', 'market_closed'):
                if self.status == 'paused':
                    await asyncio.sleep(2)
                    continue

                # Check if ANY symbol's market is open
                any_open = any(_is_market_open(sym) for sym in active_symbols)
                if not any_open:
                    if self.status != 'market_closed':
                        self.status = 'market_closed'
                        await broadcast({'type': 'paper_status', 'status': 'market_closed',
                                         'message': 'All markets closed — waiting...'})
                    await asyncio.sleep(30)
                    if self.status == 'market_closed':
                        if any(_is_market_open(sym) for sym in active_symbols):
                            self.status = 'running'
                            await broadcast({'type': 'paper_status', 'status': 'running',
                                             'message': f'Market open — {tick_label}'})
                    continue

                # Fetch new bars for all open-market symbols concurrently
                open_syms = [sym for sym in active_symbols if _is_market_open(sym)]
                fetch_tasks2 = {
                    sym: loop.run_in_executor(None, _fetch_recent_bars, sym, self.interval, 10)
                    for sym in open_syms
                }
                fetch_results2 = {}
                for sym, coro in fetch_tasks2.items():
                    try:
                        fetch_results2[sym] = await coro
                    except Exception as e:
                        logger.warning(f"Paper poll fetch error for {sym}: {e}")

                any_new = False
                for sym in open_syms:
                    ss = self.symbol_states.get(sym)
                    if not ss or ss.df is None:
                        continue
                    new_df = fetch_results2.get(sym)
                    if new_df is None or new_df.empty:
                        continue

                    # Find new bars since last processed
                    if ss.last_bar_time:
                        try:
                            cutoff = pd.Timestamp(ss.last_bar_time)
                            new_bars = new_df[new_df.index > cutoff]
                        except Exception:
                            new_bars = new_df.tail(1)
                    else:
                        new_bars = new_df.tail(1)

                    if not new_bars.empty:
                        ss.forming = None  # reset forming candle
                        for bar_idx in range(len(new_bars)):
                            await self._process_bar(sym, new_bars, bar_idx, trading_style)
                        ss.bars_processed += len(new_bars)
                        any_new = True

                if any_new:
                    self.total_bars_processed = max(
                        ss.bars_processed for ss in self.symbol_states.values() if ss.df is not None)
                    await self._broadcast_aggregate_stats()
                    self._save_state()

                await asyncio.sleep(bar_check_freq)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Paper trader loop crashed: {e}\n{traceback.format_exc()}")
            self.status = 'stopped'
            await broadcast({'type': 'paper_status', 'status': 'error',
                             'message': f'Error: {e}'})
        finally:
            if self._streamer:
                await self._streamer.stop()
                self._streamer = None
            if self._news_streamer:
                await self._news_streamer.stop()
                self._news_streamer = None
            self._save_state()

    # --- Equity computation ---
    def _current_equity_all(self) -> float:
        """Aggregate equity across all symbols."""
        equity = self.cash
        for sym, ss in self.symbol_states.items():
            if ss.df is not None and len(ss.df) > 0:
                cp = float(ss.df['close'].iloc[-1])
                if ss.pos.position > 0:
                    equity += ss.pos.position * cp
                elif ss.pos.position < 0:
                    equity -= abs(ss.pos.position) * cp
        return equity

    def _current_equity(self, current_price: float) -> float:
        """Backwards-compat: aggregate equity (ignores current_price arg, uses latest df prices)."""
        return self._current_equity_all()

    def _total_buy_hold(self) -> float:
        """Sum of buy-hold values across all symbols at current prices."""
        total = 0.0
        for ss in self.symbol_states.values():
            if ss.bh_shares and ss.df is not None and len(ss.df) > 0:
                total += ss.bh_shares * float(ss.df['close'].iloc[-1])
        return round(total, 2)

    async def _broadcast_aggregate_stats(self):
        """Broadcast aggregate stats across all symbols."""
        eq = self._current_equity_all()
        total_pnl = eq - self.initial_capital
        total_return = (total_pnl / self.initial_capital) * 100

        # Peak / drawdown
        peak = self.initial_capital
        for e in self.equity_curve:
            if e['value'] > peak:
                peak = e['value']
        if eq > peak:
            peak = eq
        dd = ((peak - eq) / peak * 100) if peak > 0 else 0

        # Win rate
        closed = [t for t in self.trades if t.get('pnl', 0) != 0]
        wins = sum(1 for t in closed if t['pnl'] > 0)
        wr = (wins / len(closed) * 100) if closed else 0

        # Per-symbol positions
        positions = {}
        active_positions = 0
        for sym, ss in self.symbol_states.items():
            cp = float(ss.df['close'].iloc[-1]) if ss.df is not None and len(ss.df) > 0 else 0
            if ss.pos.position > 0:
                unreal = (cp - ss.pos.entry_price) * ss.pos.position
                positions[sym] = {'side': 'long', 'shares': ss.pos.position,
                                  'entry': ss.pos.entry_price, 'unrealized': round(unreal, 2)}
                active_positions += 1
            elif ss.pos.position < 0:
                unreal = (ss.pos.entry_price - cp) * abs(ss.pos.position)
                positions[sym] = {'side': 'short', 'shares': abs(ss.pos.position),
                                  'entry': ss.pos.entry_price, 'unrealized': round(unreal, 2)}
                active_positions += 1
            else:
                positions[sym] = {'side': 'flat', 'shares': 0, 'entry': 0, 'unrealized': 0}

        # Summary position text
        if active_positions == 0:
            pos_text = 'FLAT'
            pos_side = 'flat'
            total_unreal = 0
        elif active_positions == 1:
            # Show the one active position
            for sym, p in positions.items():
                if p['side'] != 'flat':
                    pos_text = f"{sym}: {p['side'].upper()} {p['shares']}"
                    pos_side = p['side']
                    total_unreal = p['unrealized']
                    break
        else:
            # Show each active position with symbol prefix
            parts = []
            for sym, p in positions.items():
                if p['side'] != 'flat':
                    parts.append(f"{sym}:{p['side'][0].upper()}{p['shares']}")
            pos_text = '  '.join(parts)
            pos_side = 'long'  # use long color for multi
            total_unreal = sum(p['unrealized'] for p in positions.values())

        # Get latest prices for ALL symbols
        symbol_prices = {}
        for sym_p, ss_p in self.symbol_states.items():
            if ss_p.df is not None and len(ss_p.df) > 0:
                symbol_prices[sym_p] = round(float(ss_p.df['close'].iloc[-1]), 2)

        await broadcast({
            'type': 'stats',
            'capital': round(eq, 2),
            'pnl': round(total_pnl, 2),
            'return_pct': round(total_return, 2),
            'position_side': pos_side,
            'position_text': pos_text,
            'unrealized_pnl': round(total_unreal, 2),
            'num_trades': len(closed),
            'win_rate': round(wr, 1),
            'peak': round(peak, 2),
            'drawdown_pct': round(dd, 2),
            'bar': self.total_bars_processed,
            'total_bars': 'LIVE',
            'price': symbol_prices,
            'symbols': self.symbols,
            'positions': positions,
        })

    async def _broadcast_stats(self, cp: float, bh_shares: float):
        """Backwards-compat wrapper."""
        await self._broadcast_aggregate_stats()

    def get_state(self) -> dict:
        # Per-symbol position info
        positions = {}
        for sym, ss in self.symbol_states.items():
            positions[sym] = {
                'position': ss.pos.position,
                'entry_price': ss.pos.entry_price,
                'bars_processed': ss.bars_processed,
                'last_bar_time': ss.last_bar_time,
            }
        # Backwards-compat: single-symbol fields from first symbol
        first_ss = self.symbol_states.get(self.active_chart_symbol, SymbolState())
        return {
            'status': self.status,
            'symbols': self.symbols,
            'active_chart_symbol': self.active_chart_symbol,
            'symbol': self.active_chart_symbol,  # backwards compat
            'interval': self.interval,
            'initial_capital': self.initial_capital,
            'cash': round(self.cash, 2),
            'position': first_ss.pos.position,  # backwards compat
            'entry_price': first_ss.pos.entry_price,  # backwards compat
            'bars_processed': self.total_bars_processed,
            'num_trades': len(self.trades),
            'equity': round(self.equity_curve[-1]['value'], 2) if self.equity_curve else self.initial_capital,
            'trades': self.trades[-20:],
            'positions': positions,
        }


# Singleton paper trader
paper_trader = PaperTrader()


# =============================================================================
# AUTONOMOUS API ENDPOINTS
# =============================================================================

@app.post("/api/auto/start")
async def auto_start():
    """Start the autonomous backtesting loop."""
    if auto_engine.status == 'running':
        return {'status': 'already_running'}
    auto_engine.start()
    await broadcast({'type': 'auto_status', 'status': 'running',
                     'message': 'Starting autonomous engine...',
                     'cycle': auto_engine.total_cycles,
                     'total_opts': auto_engine.total_optimizations})
    return {'status': 'started'}


@app.post("/api/auto/stop")
async def auto_stop():
    """Stop the autonomous backtesting loop."""
    auto_engine.stop()
    await broadcast({'type': 'auto_status', 'status': 'stopped', 'message': 'Engine stopped.'})
    return {'status': 'stopped'}


@app.post("/api/auto/pause")
async def auto_pause():
    """Toggle pause/resume on the autonomous engine."""
    auto_engine.pause()
    msg = 'Paused' if auto_engine.status == 'paused' else 'Resumed'
    await broadcast({'type': 'auto_status', 'status': auto_engine.status, 'message': msg})
    return {'status': auto_engine.status}


@app.get("/api/auto/state")
async def auto_state():
    """Get current autonomous engine state + leaderboard."""
    return auto_engine.get_state()


@app.get("/api/auto/leaderboard")
async def auto_leaderboard():
    """Get top 20 best parameter sets across all symbols."""
    return {'leaderboard': auto_engine.global_best[:20]}


@app.post("/api/auto/symbols")
async def auto_set_symbols(config: dict = {}):
    """Update which symbols the autonomous engine optimizes."""
    symbols = config.get('symbols')
    if symbols is None:
        return {'enabled_symbols': list(auto_engine.enabled_symbols), 'all_symbols': AUTO_SYMBOLS}
    # Validate: only allow known symbols
    valid = set(AUTO_SYMBOLS)
    enabled = set(s for s in symbols if s in valid)
    if not enabled:
        return {'error': 'At least one valid symbol required'}
    auto_engine.enabled_symbols = enabled
    auto_engine._save_state()
    logger.info(f"Auto symbols updated: {len(enabled)} enabled")
    await broadcast({'type': 'auto_symbols', 'enabled_symbols': list(enabled)})
    return {'enabled_symbols': list(enabled)}


@app.post("/api/auto/apply-best")
async def auto_apply_best(config: dict = {}):
    """Return the best params for a given symbol to apply in manual optimizer."""
    symbol = config.get('symbol', '')
    interval = config.get('interval', '')
    sr = auto_engine.symbol_results.get(symbol, {}).get(interval, {})
    if sr.get('best_params'):
        return {'params': sr['best_params'], 'return_pct': sr.get('best_return', 0)}
    return {'error': f'No auto results for {symbol} {interval}'}


# =============================================================================
# PAPER TRADING API ENDPOINTS
# =============================================================================

@app.post("/api/paper/start")
async def paper_start(config: dict = {}):
    """Start live paper trading."""
    if paper_trader.status == 'running':
        return {'status': 'already_running', 'symbols': paper_trader.symbols,
                'symbol': paper_trader.symbol}
    paper_trader.start(config)
    return {'status': 'started', 'symbols': paper_trader.symbols,
            'symbol': paper_trader.symbol, 'interval': paper_trader.interval}


@app.post("/api/paper/stop")
async def paper_stop():
    """Stop live paper trading."""
    paper_trader.stop()
    await broadcast({'type': 'paper_status', 'status': 'stopped', 'message': 'Paper trading stopped.'})
    return {'status': 'stopped'}


@app.post("/api/paper/pause")
async def paper_pause():
    """Toggle pause/resume on paper trading."""
    paper_trader.pause()
    msg = 'Paused' if paper_trader.status == 'paused' else 'Resumed'
    await broadcast({'type': 'paper_status', 'status': paper_trader.status, 'message': msg})
    return {'status': paper_trader.status}


@app.post("/api/paper/reset")
async def paper_reset():
    """Reset paper trading account to initial state."""
    paper_trader.reset()
    cap = paper_trader.initial_capital
    await broadcast({'type': 'paper_status', 'status': 'stopped',
                     'message': f'Account reset to ${cap:,.0f}.'})
    return {'status': 'reset', 'cash': paper_trader.cash}


@app.get("/api/paper/state")
async def paper_state():
    """Get current paper trading state."""
    return paper_trader.get_state()


@app.post("/api/paper/switch_symbol")
async def paper_switch_symbol(config: dict = {}):
    """Switch which symbol's chart is displayed."""
    sym = config.get('symbol', '').upper()
    if not sym or sym not in paper_trader.symbol_states:
        return {'error': f'Symbol {sym} not in active session'}
    await paper_trader.switch_chart_symbol(sym)
    return {'status': 'switched', 'active_chart_symbol': sym}


@app.get("/api/alerts/status")
async def alerts_status():
    """Check if Telegram alerts are configured and connected."""
    _load_env_file()
    token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
    configured = bool(token and chat_id)
    return {
        'configured': configured,
        'imports_ok': TELEGRAM_IMPORTS_OK,
        'token_set': bool(token),
        'chat_id_set': bool(chat_id),
    }


# =============================================================================
# NEWS / SENTIMENT
# =============================================================================

async def fetch_news_sentiment(symbol: str) -> dict:
    """Fetch and score news headlines for a symbol using RSS + VADER."""
    clean_symbol = symbol.replace('-USD', '').replace('-', '')  # BTC-USD -> BTC
    headlines = []

    rss_urls = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={clean_symbol}+stock&hl=en-US&gl=US&ceid=US:en",
    ]

    try:
        timeout = _aiohttp.ClientTimeout(total=8)
        async with _aiohttp.ClientSession(timeout=timeout) as session:
            for url in rss_urls:
                try:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            continue
                        text = await resp.text()
                        if _feedparser is None:
                            continue
                        feed = _feedparser.parse(text)
                        source = 'Yahoo Finance' if 'yahoo' in url else 'Google News'
                        for entry in (feed.entries or [])[:10]:
                            title = entry.get('title', '')
                            if not title:
                                continue
                            # VADER score
                            score = 0
                            if _vader:
                                score = _vader.polarity_scores(title)['compound'] * 100  # -100..+100
                            # Age in minutes
                            age_min = 0
                            published = entry.get('published_parsed')
                            if published:
                                from time import mktime
                                age_min = int((datetime.now().timestamp() - mktime(published)) / 60)
                            headlines.append({
                                'title': title,
                                'source': source,
                                'score': round(score, 1),
                                'age_min': max(0, age_min),
                                'url': entry.get('link', ''),
                            })
                except Exception as e:
                    logger.debug(f"RSS fetch error for {url}: {e}")
    except Exception as e:
        logger.warning(f"News fetch failed for {symbol}: {e}")

    # Deduplicate by title
    seen = set()
    unique = []
    for h in headlines:
        key = h['title'].lower()[:60]
        if key not in seen:
            seen.add(key)
            unique.append(h)
    headlines = unique[:15]

    # Overall score
    if headlines:
        overall = round(sum(h['score'] for h in headlines) / len(headlines), 1)
    else:
        overall = 0

    label = 'Bullish' if overall > 15 else 'Bearish' if overall < -15 else 'Neutral'

    return {
        'symbol': symbol,
        'overall_score': overall,
        'overall_label': label,
        'headlines': headlines,
        'count': len(headlines),
        'feedparser_ok': FEEDPARSER_OK,
        'vader_ok': VADER_OK,
    }


@app.get("/api/news/{symbol:path}")
async def get_news(symbol: str):
    """Get news headlines with sentiment scores for a symbol."""
    return await fetch_news_sentiment(symbol.upper())


# =============================================================================
# EMBEDDED HTML DASHBOARD
# =============================================================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claude AI Backtest Dashboard</title>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            line-height: 1.6;
            overflow-x: hidden;
        }

        /* Header */
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 24px;
            border-bottom: 1px solid #222;
            background: #111;
        }
        .header h1 {
            font-size: 1.3em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header-right {
            display: flex;
            align-items: center;
            gap: 16px;
            font-size: 0.85em;
        }
        .status-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
        }
        .status-dot.connected { background: #4ade80; }
        .status-dot.disconnected { background: #f87171; }
        .api-status { color: #aaa; }
        .api-status.set { color: #4ade80; }

        /* Layout */
        .main-layout {
            display: grid;
            grid-template-columns: 300px 1fr;
            min-height: calc(100vh - 52px);
        }

        /* Left Panel */
        .left-panel {
            background: #111;
            border-right: 1px solid #222;
            padding: 20px;
            overflow-y: auto;
        }
        .panel-section {
            margin-bottom: 20px;
        }
        .panel-section h3 {
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #888;
            margin-bottom: 10px;
        }

        /* Form controls */
        label {
            display: block;
            font-size: 0.85em;
            color: #aaa;
            margin-bottom: 4px;
        }
        input[type="text"], input[type="number"], input[type="date"] {
            width: 100%;
            padding: 8px 10px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 0.9em;
            margin-bottom: 10px;
        }
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        .symbol-presets {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin-bottom: 10px;
        }
        .symbol-preset {
            padding: 3px 8px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            color: #aaa;
            font-size: 0.75em;
            cursor: pointer;
            transition: all 0.2s;
        }
        .symbol-preset:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }

        /* Interval selector */
        .interval-group {
            display: flex;
            gap: 4px;
            margin-bottom: 6px;
        }
        .interval-btn {
            padding: 6px 12px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 6px;
            color: #aaa;
            font-size: 0.82em;
            cursor: pointer;
            transition: all 0.2s;
            flex: 1;
            text-align: center;
        }
        .interval-btn:hover { border-color: #667eea; color: #e0e0e0; }
        .interval-btn.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
        }
        .interval-note {
            font-size: 0.7em;
            color: #555;
        }

        /* Checkboxes */
        .checkbox-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .checkbox-item {
            display: flex;
            align-items: flex-start;
            gap: 8px;
        }
        .checkbox-item input[type="checkbox"] {
            margin-top: 3px;
            accent-color: #667eea;
        }
        .checkbox-item .cb-label {
            font-size: 0.85em;
            color: #e0e0e0;
        }
        .checkbox-item .cb-desc {
            font-size: 0.72em;
            color: #666;
        }

        /* Radio buttons */
        .radio-group {
            display: flex;
            gap: 12px;
        }
        .radio-group label {
            display: flex;
            align-items: center;
            gap: 5px;
            color: #e0e0e0;
            cursor: pointer;
        }
        .radio-group input[type="radio"] {
            accent-color: #667eea;
        }

        /* Buttons */
        .btn-run {
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        .btn-run:hover { opacity: 0.9; }
        .btn-run:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .cost-estimate {
            font-size: 0.72em;
            color: #888;
            text-align: center;
            margin-top: 6px;
        }

        /* Progress bar */
        .progress-container {
            margin-top: 12px;
            display: none;
        }
        .progress-container.active { display: block; }
        .progress-bar-bg {
            width: 100%;
            height: 6px;
            background: #222;
            border-radius: 3px;
            overflow: hidden;
        }
        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 3px;
            transition: width 0.3s;
            width: 0%;
        }
        .progress-text {
            font-size: 0.72em;
            color: #888;
            margin-top: 4px;
        }

        /* Center Panel */
        .center-panel {
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }

        /* Live Stats Bar */
        .live-stats {
            display: none;
            position: sticky;
            top: 0;
            z-index: 10;
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 12px;
            flex-shrink: 0;
        }
        .live-stats.active { display: block; }
        .stats-row {
            display: flex;
            gap: 16px;
            align-items: center;
            flex-wrap: wrap;
        }
        .stat-item {
            display: flex;
            flex-direction: column;
            min-width: 90px;
        }
        .stat-label {
            font-size: 0.65em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-value {
            font-size: 1.05em;
            font-weight: 600;
            color: #ccc;
            font-variant-numeric: tabular-nums;
        }
        .stat-value.positive { color: #4ade80; }
        .stat-value.negative { color: #f87171; }
        .stat-value.neutral { color: #888; }
        .stat-divider {
            width: 1px;
            height: 32px;
            background: #333;
        }
        .position-badge {
            padding: 3px 10px;
            border-radius: 4px;
            font-weight: 700;
            font-size: 0.85em;
        }
        .position-badge.flat { background: #222; color: #888; }
        .position-badge.long { background: rgba(74,222,128,0.15); color: #4ade80; }
        .position-badge.short { background: rgba(248,113,113,0.15); color: #f87171; }

        /* Metric cards */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: #151515;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 14px;
            text-align: center;
        }
        .metric-card .metric-value {
            font-size: 1.5em;
            font-weight: 700;
        }
        .metric-card .metric-label {
            font-size: 0.75em;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value.positive { color: #4ade80; }
        .metric-value.negative { color: #f87171; }
        .metric-value.neutral { color: #60a5fa; }

        /* Charts */
        .chart-container {
            background: #151515;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
        }
        .chart-container h3 {
            font-size: 0.85em;
            color: #aaa;
            margin-bottom: 12px;
        }
        .tv-chart-panel {
            width: 100%;
            border-radius: 6px;
            overflow: hidden;
            background: #151515;
        }

        /* Trade table */
        .trade-table-container {
            background: #151515;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
            overflow-x: auto;
        }
        .trade-table-container h3 {
            font-size: 0.85em;
            color: #aaa;
            margin-bottom: 12px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82em;
        }
        th {
            background: #1a1a1a;
            padding: 8px 10px;
            text-align: left;
            color: #888;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75em;
            letter-spacing: 0.5px;
            border-bottom: 1px solid #333;
        }
        td {
            padding: 8px 10px;
            border-bottom: 1px solid #1a1a1a;
        }
        tr:hover { background: #1a1a1a; }
        tr.trade-buy td:first-child { border-left: 3px solid #4ade80; }
        tr.trade-sell td:first-child { border-left: 3px solid #f87171; }
        tr.trade-short td:first-child { border-left: 3px solid #f59e0b; }
        tr.trade-cover td:first-child { border-left: 3px solid #8b5cf6; }
        tr.trade-stop td:first-child { border-left: 3px solid #fbbf24; }
        tr.trade-tp td:first-child { border-left: 3px solid #60a5fa; }
        tr.expandable { cursor: pointer; }

        /* Reasoning viewer */
        .reasoning-panel {
            background: #0d0d0d;
            border: 1px solid #333;
            border-radius: 6px;
            padding: 14px;
            margin: 8px 0;
            display: none;
            font-size: 0.82em;
        }
        .reasoning-panel.active { display: block; }
        .reasoning-panel h4 {
            color: #667eea;
            font-size: 0.85em;
            margin-bottom: 8px;
        }
        .reasoning-section {
            margin-bottom: 10px;
        }
        .reasoning-section .rs-label {
            color: #888;
            font-size: 0.78em;
            text-transform: uppercase;
        }
        .reasoning-section .rs-content {
            color: #ccc;
            padding: 6px 0;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .reasoning-section pre {
            background: #111;
            border: 1px solid #222;
            border-radius: 4px;
            padding: 10px;
            overflow-x: auto;
            color: #aaa;
            font-size: 0.9em;
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 80px 20px;
            color: #555;
        }
        .empty-state .es-icon {
            font-size: 3em;
            margin-bottom: 16px;
        }
        .empty-state h2 {
            font-size: 1.2em;
            color: #888;
            margin-bottom: 8px;
        }
        .empty-state p {
            font-size: 0.85em;
        }

        /* Autonomous Engine Panel */
        .auto-panel {
            background: #0d0d0d;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 16px;
        }
        .auto-panel h3 {
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #888;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .auto-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            display: inline-block;
            background: #555;
        }
        .auto-dot.running {
            background: #4ade80;
            animation: pulse 1.5s ease-in-out infinite;
        }
        .auto-dot.paused { background: #fbbf24; }
        .auto-dot.stopped { background: #555; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .auto-status-text {
            font-size: 0.8em;
            color: #888;
            margin-bottom: 8px;
            min-height: 1.4em;
        }
        .auto-buttons {
            display: flex;
            gap: 6px;
        }
        .auto-buttons button {
            flex: 1;
            padding: 7px 10px;
            border-radius: 6px;
            border: 1px solid #333;
            font-size: 0.82em;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-auto-start {
            background: #4ade8022;
            color: #4ade80;
            border-color: #4ade8044;
        }
        .btn-auto-start:hover { background: #4ade8033; }
        .btn-auto-stop {
            background: #f8717122;
            color: #f87171;
            border-color: #f8717144;
        }
        .btn-auto-stop:hover { background: #f8717133; }
        .btn-auto-pause {
            background: #fbbf2422;
            color: #fbbf24;
            border-color: #fbbf2444;
        }
        .btn-auto-pause:hover { background: #fbbf2433; }

        /* Auto Symbol Picker */
        .auto-symbols-picker {
            margin-top: 10px;
            border-top: 1px solid #222;
            padding-top: 8px;
        }
        .auto-symbols-picker .group-label {
            font-size: 0.65em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #555;
            margin: 6px 0 3px 0;
        }
        .auto-symbols-picker .group-label:first-child { margin-top: 0; }
        .auto-sym-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
        }
        .auto-sym-chip {
            font-size: 0.7em;
            padding: 2px 7px;
            border-radius: 4px;
            border: 1px solid #333;
            background: #111;
            color: #555;
            cursor: pointer;
            transition: all 0.15s;
            user-select: none;
        }
        .auto-sym-chip.enabled {
            background: #4ade8015;
            color: #4ade80;
            border-color: #4ade8044;
        }
        .auto-sym-chip:hover { border-color: #4ade80; }
        .auto-sym-toggle-row {
            display: flex;
            justify-content: flex-end;
            gap: 8px;
            margin-bottom: 4px;
        }
        .auto-sym-toggle {
            font-size: 0.65em;
            color: #60a5fa;
            cursor: pointer;
            background: none;
            border: none;
            padding: 0;
        }
        .auto-sym-toggle:hover { text-decoration: underline; }

        /* News/Sentiment Panel */
        .news-panel {
            background: #0d0d0d;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 16px;
        }
        .news-panel h3 {
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #888;
            margin-bottom: 10px;
        }
        .news-score {
            font-size: 1.4em;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .news-score.bullish { color: #4ade80; }
        .news-score.bearish { color: #f87171; }
        .news-score.neutral { color: #888; }
        .news-headline {
            font-size: 0.78em;
            padding: 5px 0;
            border-bottom: 1px solid #1a1a1a;
            line-height: 1.35;
        }
        .news-headline a {
            color: #bbb;
            text-decoration: none;
        }
        .news-headline a:hover { color: #e0e0e0; }
        .news-headline .hl-score {
            display: inline-block;
            min-width: 36px;
            text-align: right;
            margin-right: 6px;
            font-weight: 600;
            font-size: 0.9em;
        }
        .news-headline .hl-source {
            color: #555;
            font-size: 0.85em;
        }
        #liveDecisionFeed a { transition: color 0.15s ease; }
        #liveDecisionFeed a:hover { color: #fff !important; text-decoration: underline; }

        /* Leaderboard */
        .leaderboard-container {
            background: #151515;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
            display: none;
        }
        .leaderboard-container.active { display: block; }
        .leaderboard-container h3 {
            font-size: 0.85em;
            color: #aaa;
            margin-bottom: 12px;
        }
        .lb-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8em;
        }
        .lb-table th {
            background: #1a1a1a;
            padding: 6px 8px;
            text-align: left;
            color: #888;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75em;
            letter-spacing: 0.3px;
            border-bottom: 1px solid #333;
        }
        .lb-table td {
            padding: 6px 8px;
            border-bottom: 1px solid #1a1a1a;
        }
        .lb-table tr:hover { background: #1a1a1a; }
        .lb-use-btn {
            background: #667eea22;
            color: #667eea;
            border: 1px solid #667eea44;
            border-radius: 3px;
            padding: 2px 8px;
            cursor: pointer;
            font-size: 0.9em;
        }
        .lb-use-btn:hover { background: #667eea33; }
        .overfit-tag {
            background: #f8717122;
            color: #f87171;
            padding: 1px 5px;
            border-radius: 3px;
            font-size: 0.75em;
        }

        /* Chart grid cells */
        .chart-grid-cell {
            background: #151515;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 8px;
            overflow: hidden;
            resize: both;
            min-width: 300px;
            min-height: 200px;
        }
        .chart-grid-cell .cell-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 4px;
            font-size: 0.8em;
        }
        .chart-grid-cell .cell-symbol {
            font-weight: 700;
            color: #fbbf24;
        }
        .chart-grid-cell .cell-price {
            color: #ccc;
        }

        /* Responsive */
        @media (max-width: 900px) {
            .main-layout {
                grid-template-columns: 1fr;
            }
            .left-panel {
                border-right: none;
                border-bottom: 1px solid #222;
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <div class="header">
        <h1>Claude AI Backtest</h1>
        <div class="header-right">
            <span><span class="status-dot disconnected" id="wsDot"></span><span id="wsStatus">Connecting...</span></span>
            <span class="api-status" id="apiStatus">API Key: checking...</span>
            <span class="api-status" id="dataSource" style="margin-left:12px;">Data: checking...</span>
        </div>
    </div>

    <div class="main-layout">
        <!-- Left Panel: Controls -->
        <div class="left-panel">
            <div class="panel-section">
                <h3>Symbol</h3>
                <input type="text" id="symbol" value="SPY" placeholder="SPY, AAPL, BTC-USD...">
                <div style="font-size:0.7em;color:#555;margin-top:2px;">Comma-separated for multi-symbol backtesting/paper trading</div>
                <div class="symbol-presets" id="symbolPresets"></div>
            </div>

            <div class="panel-section">
                <h3>Timeframe</h3>
                <div class="interval-group" id="intervalGroup">
                    <span class="interval-btn" data-val="1m">1m</span>
                    <span class="interval-btn active" data-val="5m">5m</span>
                    <span class="interval-btn" data-val="10m">10m</span>
                    <span class="interval-btn" data-val="15m">15m</span>
                    <span class="interval-btn" data-val="30m">30m</span>
                    <span class="interval-btn" data-val="1h">1H</span>
                    <span class="interval-btn" data-val="1d">1D</span>
                </div>
                <div class="interval-note" id="intervalNote">5m bars — max 60 days</div>
            </div>

            <div class="panel-section">
                <h3>Date Range</h3>
                <label>Start Date</label>
                <input type="date" id="startDate">
                <label>End Date</label>
                <input type="date" id="endDate">
            </div>

            <div class="panel-section">
                <h3>Analysis Factors</h3>
                <div class="checkbox-group">
                    <div class="checkbox-item">
                        <input type="checkbox" id="fPrice" checked>
                        <div>
                            <div class="cb-label">Price Action</div>
                            <div class="cb-desc">OHLCV candles, price structure</div>
                        </div>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="fTech" checked>
                        <div>
                            <div class="cb-label">Technical Indicators</div>
                            <div class="cb-desc">RSI, MACD, BB, ADX, ATR</div>
                        </div>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="fNews">
                        <div>
                            <div class="cb-label">News & Sentiment</div>
                            <div class="cb-desc">Headlines, sentiment scores</div>
                        </div>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="fRegime" checked>
                        <div>
                            <div class="cb-label">Market Regime</div>
                            <div class="cb-desc">Trend, volatility, S/R levels</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="panel-section">
                <h3>Backtest Mode</h3>
                <div class="radio-group" style="flex-direction:column;gap:6px;">
                    <label><input type="radio" name="mode" value="rules" checked> Rules Engine (backtest, free)</label>
                    <label><input type="radio" name="mode" value="live_api"> Live LLM (backtest, per-bar AI)</label>
                    <label><input type="radio" name="mode" value="fallback"> Fallback (backtest, Adaptive)</label>
                    <label style="color:#fbbf24;"><input type="radio" name="mode" value="paper"> Live Paper Trade (real-time, rules engine)</label>
                </div>
                <div id="llmProviderSection" style="display:none;margin-top:8px;">
                    <label>LLM Provider</label>
                    <select id="llmProvider" style="width:100%;padding:4px;background:#1a1a1a;color:#e0e0e0;border:1px solid #333;border-radius:4px;">
                        <option value="ollama">Local Ollama (free, GPU)</option>
                        <option value="anthropic">Claude API ($)</option>
                    </select>
                    <div id="ollamaModelSection" style="margin-top:4px;">
                        <label>Ollama Model</label>
                        <select id="ollamaModel" style="width:100%;padding:4px;background:#1a1a1a;color:#e0e0e0;border:1px solid #333;border-radius:4px;">
                            <option>Loading...</option>
                        </select>
                    </div>
                    <div id="anthropicNote" style="display:none;font-size:0.75em;color:#f59e0b;margin-top:4px;">
                        Requires ANTHROPIC_API_KEY environment variable
                    </div>
                </div>
            </div>

            <div class="panel-section">
                <h3>Trading Style</h3>
                <div class="radio-group">
                    <label><input type="radio" name="trading_style" value="swing" checked> Swing / Position (hold overnight)</label>
                    <label><input type="radio" name="trading_style" value="day"> Day Trading (close at EOD)</label>
                </div>
            </div>

            <div class="panel-section">
                <h3>Risk Protection</h3>
                <label>Max Trade Loss (%)</label>
                <input type="number" id="maxTradeLossPct" value="3" min="0" max="20" step="0.5" title="Force close if trade loss exceeds this % of equity (0=disabled). Works even without adaptive risk.">
                <div style="font-size:0.75em;color:#f59e0b;margin-bottom:6px;">Safety net: active even with adaptive risk OFF. 0 = disabled.</div>
                <label>Max Gap Exit (%)</label>
                <input type="number" id="maxGapPct" value="2" min="0" max="10" step="0.5" title="Force exit on adverse overnight gap exceeding this % (0=disabled).">
                <div style="font-size:0.75em;color:#888;margin-bottom:6px;">Exit at open price on adverse gap. Best for swing trades. 0 = disabled.</div>
                <label>News Sentiment Filter</label>
                <input type="number" id="newsSentimentThreshold" value="0" min="0" max="80" step="5" title="Block entries when news sentiment opposes trade direction (0=disabled). Paper trading only.">
                <div style="font-size:0.75em;color:#888;margin-bottom:6px;">Paper trading only. 0=off, 30=moderate, 50=strict.</div>
                <label>Counter-Trend Penalty</label>
                <input type="number" id="counterTrendPenalty" value="1" min="0" max="3" step="1" title="Score penalty for trades against the medium-term trend (EMA21 vs EMA50). Higher = fewer counter-trend entries.">
                <div style="font-size:0.75em;color:#888;margin-bottom:6px;">Penalizes entries against EMA50 trend. 0=off, 1=moderate, 2=strict.</div>
            </div>

            <div class="panel-section">
                <h3>Adaptive Risk <span style="font-size:0.7em;color:#4ade80;font-weight:normal;">NEW</span></h3>
                <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
                    <input type="checkbox" id="adaptiveRisk" checked>
                    <span>Enable adaptive risk management</span>
                </label>
                <div id="adaptiveRiskParams" style="margin-top:8px;">
                    <label>Risk Per Trade (%)</label>
                    <input type="number" id="riskPerTrade" value="2" min="0.5" max="5" step="0.5" title="Max risk per trade as % of equity">
                    <label>Max Stop Distance (%)</label>
                    <input type="number" id="maxStopPct" value="4" min="1" max="10" step="0.5" title="Hard cap on stop distance as % of price">
                    <label>Drawdown Throttle</label>
                    <div style="font-size:0.75em;color:#888;">Half risk at 5% DD, skip trades at 10% DD</div>
                    <label>Cooldown After Losses</label>
                    <input type="number" id="cooldownLosses" value="3" min="1" max="10" step="1" title="Skip trades after N consecutive losses">
                    <label>Re-entry Cooldown (bars)</label>
                    <input type="number" id="reentryCooldown" value="5" min="0" max="60" step="1" title="Wait N bars after closing a position before re-entering">
                    <div style="font-size:0.75em;color:#888;margin-top:4px;">
                        Includes: vol-normalized sizing, ATR-regime stops, SPY/VIX filter, drawdown throttle, stop-price fills
                    </div>
                </div>
            </div>

            <div class="panel-section">
                <h3>Chart Options</h3>
                <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
                    <input type="checkbox" id="showPatterns" checked>
                    <span>Show detected patterns on chart</span>
                </label>
                <div style="font-size:0.7em;color:#555;margin-top:2px;">
                    <span style="color:#22c55e;">&#9679;</span> Bullish &nbsp;
                    <span style="color:#ef4444;">&#9679;</span> Bearish &nbsp;
                    <span style="color:#eab308;">&#9679;</span> Neutral
                    &mdash; Rules mode only
                </div>
            </div>

            <div class="panel-section">
                <h3>Settings</h3>
                <label>Initial Capital ($)</label>
                <input type="number" id="capital" value="100000" min="1000" step="1000">
                <label>Commission (%)</label>
                <input type="number" id="commission" value="0.1" min="0" max="5" step="0.01">
                <label>Slippage (x ATR)</label>
                <input type="number" id="slippage" value="0.1" min="0" max="1" step="0.01" title="Slippage as fraction of ATR applied adversely to each fill (0=none, 0.1=realistic)">
                <label>Call Claude Every N Bars</label>
                <select id="callFrequency">
                    <option value="1">Every bar (slowest, most precise)</option>
                    <option value="3">Every 3 bars</option>
                    <option value="5" selected>Every 5 bars (recommended)</option>
                    <option value="10">Every 10 bars (fast)</option>
                    <option value="15">Every 15 bars (fastest)</option>
                </select>
                <div id="callEstimate" style="font-size:0.75em;color:#888;margin-top:4px;"></div>
            </div>

            <div class="panel-section">
                <button class="btn-run" id="btnRun" onclick="runBacktest()">Run Backtest</button>
                <button class="btn-run" id="btnOptimize" onclick="runOptimizer()" style="background:linear-gradient(135deg,#764ba2,#667eea);margin-top:6px;">Optimize Parameters</button>
                <div class="cost-estimate" id="costEstimate"></div>
                <div class="progress-container" id="progressContainer">
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" id="progressFill"></div>
                    </div>
                    <div class="progress-text" id="progressText">Starting...</div>
                </div>
            </div>

            <!-- Paper Trading Controls (hidden by default) -->
            <div id="paperControls" style="display:none; margin-bottom:16px;">
                <div style="display:flex;gap:6px;margin-bottom:8px;">
                    <button id="btnPaperPause" onclick="paperPause()" style="flex:1;padding:8px;border-radius:6px;border:1px solid #fbbf2444;background:#fbbf2422;color:#fbbf24;cursor:pointer;font-size:0.85em;">Pause</button>
                    <button id="btnPaperStop" onclick="paperStop()" style="flex:1;padding:8px;border-radius:6px;border:1px solid #f8717144;background:#f8717122;color:#f87171;cursor:pointer;font-size:0.85em;">Stop</button>
                </div>
                <div id="paperStatusText" style="font-size:0.8em;color:#fbbf24;margin-bottom:6px;">Idle</div>
                <button onclick="paperReset()" style="width:100%;padding:6px;border-radius:6px;border:1px solid #333;background:#1a1a1a;color:#888;cursor:pointer;font-size:0.78em;">Reset Account ($100k)</button>
            </div>

            <!-- Autonomous Engine Controls -->
            <div class="auto-panel">
                <h3><span class="auto-dot stopped" id="autoDot"></span> AUTONOMOUS ENGINE</h3>
                <div class="auto-status-text" id="autoStatusText">Stopped</div>
                <div class="auto-buttons">
                    <button class="btn-auto-start" id="btnAutoStart" onclick="autoStart()">Start Auto</button>
                    <button class="btn-auto-pause" id="btnAutoPause" onclick="autoPause()" disabled>Pause</button>
                </div>
                <div id="telegramStatus" style="font-size:0.75em;color:#555;margin-top:6px;"></div>
                <div class="auto-symbols-picker" id="autoSymbolsPicker">
                    <div class="auto-sym-toggle-row">
                        <button class="auto-sym-toggle" onclick="autoSymSelectAll()">All</button>
                        <button class="auto-sym-toggle" onclick="autoSymSelectNone()">None</button>
                        <button class="auto-sym-toggle" onclick="autoSymSelectEquities()">Equities</button>
                        <button class="auto-sym-toggle" onclick="autoSymSelectCrypto()">Crypto</button>
                        <button class="auto-sym-toggle" onclick="autoSymSelectForex()">Forex</button>
                    </div>
                    <div id="autoSymChips"></div>
                </div>
            </div>

            <!-- News/Sentiment Panel -->
            <div class="news-panel" id="newsPanel" style="display:none;">
                <h3>NEWS SENTIMENT</h3>
                <div class="news-score neutral" id="newsScore">--</div>
                <div id="newsHeadlines"></div>
            </div>
        </div>

        <!-- Center Panel: Results -->
        <div class="center-panel" id="centerPanel">
            <!-- Live Stats Bar (sticky at top during run) -->
            <div class="live-stats" id="liveStats">
                <div class="stats-row">
                    <div class="stat-item">
                        <span class="stat-label">Capital</span>
                        <span class="stat-value" id="statCapital">$100,000</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">P&L</span>
                        <span class="stat-value neutral" id="statPnl">$0.00</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Return</span>
                        <span class="stat-value neutral" id="statReturn">0.00%</span>
                    </div>
                    <div class="stat-divider"></div>
                    <div class="stat-item">
                        <span class="stat-label">Position</span>
                        <span class="position-badge flat" id="statPosition">FLAT</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Unreal. P&L</span>
                        <span class="stat-value neutral" id="statUnrealPnl">—</span>
                    </div>
                    <div class="stat-divider"></div>
                    <div class="stat-item">
                        <span class="stat-label">Trades</span>
                        <span class="stat-value" id="statTrades">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Win Rate</span>
                        <span class="stat-value" id="statWinRate">—</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Peak</span>
                        <span class="stat-value" id="statPeak">$100,000</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Drawdown</span>
                        <span class="stat-value neutral" id="statDrawdown">0.00%</span>
                    </div>
                    <div class="stat-divider"></div>
                    <div class="stat-item">
                        <span class="stat-label">Bar</span>
                        <span class="stat-value" id="statBar">—</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Price</span>
                        <span class="stat-value" id="statPrice">—</span>
                    </div>
                </div>
            </div>

            <!-- Live Charts (visible during run) -->
            <div id="liveChartContainer" style="display:none;">
                <div id="chartGrid" style="display:grid; gap:8px; margin-bottom:8px;"></div>
                <div class="chart-container" style="padding:8px; margin-bottom:8px;">
                    <div id="tvEquityChart" class="tv-chart-panel" style="height:180px;"></div>
                </div>
            </div>

            <!-- Autonomous Engine Leaderboard -->
            <div class="leaderboard-container" id="leaderboardContainer">
                <h3>Autonomous Engine Leaderboard</h3>
                <table class="lb-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Symbol</th>
                            <th>TF</th>
                            <th>Return</th>
                            <th>Win%</th>
                            <th>MaxDD</th>
                            <th>PF</th>
                            <th>OOS</th>
                            <th>Tests</th>
                            <th></th>
                        </tr>
                    </thead>
                    <tbody id="leaderboardBody"></tbody>
                </table>
            </div>

            <div class="empty-state" id="emptyState">
                <div class="es-icon">&#x1f4ca;</div>
                <h2>No Backtest Results Yet</h2>
                <p>Configure your parameters and click "Run Backtest" to start.</p>
            </div>

            <!-- Live Decision Feed (visible during run) -->
            <div id="liveLogContainer" style="display:none;">
                <h3 style="margin:0 0 10px 0;color:#ccc;">Claude's Live Decisions</h3>
                <div id="liveDecisionFeed" style="max-height:calc(40vh - 80px); overflow-y:auto; display:flex; flex-direction:column; gap:8px; padding:4px;"></div>
            </div>

            <div id="resultsContainer" style="display:none;">
                <!-- Metrics -->
                <div class="metrics-grid" id="metricsGrid"></div>

                <!-- Equity Curve -->
                <div class="chart-container">
                    <h3>Equity Curve</h3>
                    <div id="equityChart" class="tv-chart-panel" style="height:300px;"></div>
                </div>

                <!-- Drawdown -->
                <div class="chart-container">
                    <h3>Drawdown</h3>
                    <div id="drawdownChart" class="tv-chart-panel" style="height:200px;"></div>
                </div>

                <!-- Trade Log -->
                <div class="trade-table-container">
                    <h3>Trade Log & Claude Reasoning</h3>
                    <table id="tradeTable">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Signal</th>
                                <th>Price</th>
                                <th>Shares</th>
                                <th>P&L</th>
                                <th>Return</th>
                                <th>Strength</th>
                            </tr>
                        </thead>
                        <tbody id="tradeBody"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
    // ======== State ========
    let ws = null;
    let currentResults = null;

    // === TradingView Lightweight Charts State ===
    // Per-symbol chart instances: { symbol: { el, priceEl, priceChart, candleSeries, volumeSeries, tradeMarkers, markersPrimitive } }
    let chartGrid = {};
    let equityChart = null;
    let equitySeries = null;
    let bhSeries = null;
    let postEquityChart = null;
    let postDrawdownChart = null;
    let syncingCharts = false;

    const LWC_CHART_OPTS = {
        layout: { background: { color: '#151515' }, textColor: '#aaa' },
        grid: { vertLines: { color: '#1a1a1a' }, horzLines: { color: '#1a1a1a' } },
        crosshair: { mode: 0 },
        rightPriceScale: { borderColor: '#222' },
        timeScale: { borderColor: '#222', timeVisible: true, secondsVisible: false },
    };

    function parseBarTime(dateStr) {
        // Convert bar date string to a timestamp for lightweight-charts.
        // Schwab/data sources send UTC timestamps (14:30 UTC = 9:30 AM ET).
        // LWC has no timezone setting, so we shift UTC → ET for intraday bars.
        if (!dateStr) return 0;
        const hasTime = dateStr.includes(' ') || (dateStr.includes('T') && dateStr.length > 10);
        if (!hasTime) {
            // Daily bar: date-only "2026-01-15" — keep as-is, no TZ shift
            const d = new Date(dateStr + 'T00:00:00Z');
            return Math.floor(d.getTime() / 1000);
        }
        // Intraday bar: convert UTC → Eastern Time for correct display
        const s = dateStr.replace(' ', 'T');
        const d = new Date(s + 'Z');
        const etStr = d.toLocaleString('sv-SE', {timeZone: 'America/New_York'});
        const etFake = new Date(etStr.replace(' ', 'T') + 'Z');
        return Math.floor(etFake.getTime() / 1000);
    }

    function initChartGrid(symbols) {
        // Clean up existing chart grid
        Object.values(chartGrid).forEach(c => {
            if (c.resizeObserver) c.resizeObserver.disconnect();
            if (c.priceChart) c.priceChart.remove();
        });
        chartGrid = {};
        if (equityChart) { equityChart.remove(); equityChart = null; }

        document.getElementById('liveChartContainer').style.display = 'block';

        const gridEl = document.getElementById('chartGrid');
        gridEl.innerHTML = '';
        const equityEl = document.getElementById('tvEquityChart');
        equityEl.innerHTML = '';

        // Set grid layout based on symbol count
        if (symbols.length === 1) {
            gridEl.style.gridTemplateColumns = '1fr';
        } else if (symbols.length === 2) {
            gridEl.style.gridTemplateColumns = '1fr 1fr';
        } else {
            gridEl.style.gridTemplateColumns = 'repeat(auto-fit, minmax(400px, 1fr))';
        }

        // Compute chart height based on symbol count
        const chartHeight = symbols.length === 1 ? 350 : 280;

        // Create a chart cell for each symbol
        symbols.forEach(sym => {
            const cell = document.createElement('div');
            cell.className = 'chart-grid-cell';

            // Header with symbol name + live price
            const header = document.createElement('div');
            header.className = 'cell-header';
            const symSpan = document.createElement('span');
            symSpan.className = 'cell-symbol';
            symSpan.textContent = sym;
            const priceSpan = document.createElement('span');
            priceSpan.className = 'cell-price';
            priceSpan.textContent = '—';
            header.appendChild(symSpan);
            header.appendChild(priceSpan);
            cell.appendChild(header);

            // Chart div
            const chartDiv = document.createElement('div');
            chartDiv.style.height = chartHeight + 'px';
            cell.appendChild(chartDiv);
            gridEl.appendChild(cell);

            // Create LWC chart
            const pc = LightweightCharts.createChart(chartDiv, {
                ...LWC_CHART_OPTS,
                height: chartHeight,
            });

            const cs = pc.addSeries(LightweightCharts.CandlestickSeries, {
                upColor: '#4ade80',
                downColor: '#f87171',
                borderUpColor: '#4ade80',
                borderDownColor: '#f87171',
                wickUpColor: '#4ade80aa',
                wickDownColor: '#f87171aa',
            });

            const vs = pc.addSeries(LightweightCharts.HistogramSeries, {
                priceFormat: { type: 'volume' },
                priceScaleId: 'volume',
            });
            pc.priceScale('volume').applyOptions({
                scaleMargins: { top: 0.8, bottom: 0 },
            });

            // ResizeObserver to handle cell resize
            const ro = new ResizeObserver(() => {
                if (pc) {
                    pc.applyOptions({ width: chartDiv.clientWidth, height: chartDiv.clientHeight });
                }
            });
            ro.observe(chartDiv);

            chartGrid[sym] = {
                el: cell,
                chartDiv: chartDiv,
                priceEl: priceSpan,
                priceChart: pc,
                candleSeries: cs,
                volumeSeries: vs,
                tradeMarkers: [],
                patternMarkers: [],
                lastPatternTime: 0,  // throttle: min 3 bars between pattern markers
                markersPrimitive: null,
                resizeObserver: ro,
            };
        });

        // Create single aggregate equity chart
        equityChart = LightweightCharts.createChart(equityEl, {
            ...LWC_CHART_OPTS,
            height: 180,
        });

        equitySeries = equityChart.addSeries(LightweightCharts.LineSeries, {
            color: '#667eea',
            lineWidth: 2,
            title: 'Strategy',
        });

        bhSeries = equityChart.addSeries(LightweightCharts.LineSeries, {
            color: '#888',
            lineWidth: 1,
            lineStyle: 2, // dashed
            title: 'Buy & Hold',
        });

        // ResizeObserver for equity chart
        const eqRo = new ResizeObserver(() => {
            if (equityChart) equityChart.applyOptions({ width: equityEl.clientWidth });
        });
        eqRo.observe(equityEl);
    }

    // Backwards-compat wrapper for backtest mode (single symbol)
    function initLiveChart() {
        const sym = document.getElementById('symbol').value.trim().toUpperCase().split(',')[0].trim() || 'SPY';
        initChartGrid([sym]);
    }

    let scrollRAF = null;
    let showPatterns = true;
    const PATTERN_THROTTLE_SECS = 0;  // min seconds between pattern markers (0 = no throttle)

    function _mergeAndSetMarkers(chart) {
        // Merge trade + pattern markers into one sorted array (LWC requires single primitive per series)
        const all = [...chart.tradeMarkers];
        if (showPatterns) all.push(...chart.patternMarkers);
        all.sort((a, b) => a.time - b.time);
        if (chart.markersPrimitive) {
            chart.markersPrimitive.setMarkers(all);
        } else if (all.length > 0) {
            chart.markersPrimitive = LightweightCharts.createSeriesMarkers(chart.candleSeries, all);
        }
    }

    function addBarToChart(msg) {
        const sym = msg.symbol;
        const chart = sym ? chartGrid[sym] : chartGrid[Object.keys(chartGrid)[0]];
        if (!chart) return;

        const t = parseBarTime(msg.date);
        if (!t) return;

        // Update candlestick on the correct symbol chart
        chart.candleSeries.update({ time: t, open: msg.open, high: msg.high, low: msg.low, close: msg.close });

        // Update volume
        const volColor = msg.close >= msg.open ? 'rgba(74,222,128,0.3)' : 'rgba(248,113,113,0.3)';
        chart.volumeSeries.update({ time: t, value: msg.volume || 0, color: volColor });

        // Update equity lines (aggregate, single chart)
        if (equitySeries) equitySeries.update({ time: t, value: msg.equity });
        if (bhSeries) bhSeries.update({ time: t, value: msg.buy_hold });

        // Pattern highlighting: add circle markers for detected patterns
        if (showPatterns && msg.patterns && msg.patterns.length > 0) {
            // Throttle: skip if too close to last pattern marker
            if (PATTERN_THROTTLE_SECS === 0 || t - chart.lastPatternTime >= PATTERN_THROTTLE_SECS) {
                // Pick the most important pattern (first bullish/bearish, then neutral)
                const bull = msg.patterns.find(p => p.type === 'bullish');
                const bear = msg.patterns.find(p => p.type === 'bearish');
                const pat = bull || bear || msg.patterns[0];
                const isBull = pat.type === 'bullish';
                const isBear = pat.type === 'bearish';
                chart.patternMarkers.push({
                    time: t,
                    position: isBull ? 'belowBar' : 'aboveBar',
                    color: isBull ? '#22c55e' : isBear ? '#ef4444' : '#eab308',
                    shape: 'circle',
                    text: pat.name + (msg.patterns.length > 1 ? '+' + (msg.patterns.length - 1) : ''),
                });
                chart.lastPatternTime = t;
                _mergeAndSetMarkers(chart);
            }
        }

        // Update header price
        if (chart.priceEl) chart.priceEl.textContent = '$' + msg.close.toFixed(2);

        // Batch scroll calls via rAF
        if (!scrollRAF) {
            scrollRAF = requestAnimationFrame(() => {
                scrollRAF = null;
                Object.values(chartGrid).forEach(c => { if (c.priceChart) c.priceChart.timeScale().scrollToRealTime(); });
                if (equityChart) equityChart.timeScale().scrollToRealTime();
            });
        }
    }

    function addTradeMarker(trade) {
        const sym = trade.symbol;
        const chart = sym ? chartGrid[sym] : chartGrid[Object.keys(chartGrid)[0]];
        if (!chart || !chart.candleSeries) return;

        const t = parseBarTime(trade.date);
        if (!t) return;

        const tradeType = trade.type;
        let shape, color, position, text;

        if (tradeType === 'BUY' || tradeType === 'COVER') {
            shape = 'arrowUp';
            color = '#4ade80';
            position = 'belowBar';
            text = tradeType;
        } else if (tradeType === 'SELL' || tradeType === 'SHORT') {
            shape = 'arrowDown';
            color = '#f87171';
            position = 'aboveBar';
            text = tradeType;
        } else if (tradeType === 'STOP_LOSS' || tradeType === 'STOP_LOSS_SHORT') {
            shape = 'arrowDown';
            color = '#fbbf24';
            position = 'aboveBar';
            text = 'SL';
        } else if (tradeType === 'TAKE_PROFIT' || tradeType === 'TAKE_PROFIT_SHORT') {
            shape = 'arrowUp';
            color = '#60a5fa';
            position = 'belowBar';
            text = 'TP';
        } else {
            // EOD_CLOSE, EOD_COVER, CLOSE, etc.
            shape = 'circle';
            color = '#fbbf24';
            position = 'aboveBar';
            text = tradeType;
        }

        chart.tradeMarkers.push({ time: t, position, color, shape, text });
        chart.tradeMarkers.sort((a, b) => a.time - b.time);
        _mergeAndSetMarkers(chart);
    }

    // ======== Init ========
    document.addEventListener('DOMContentLoaded', () => {
        initDates();
        loadSymbols();
        checkApiKey();
        checkDataSources();
        connectWebSocket();
        updateCostEstimate();
        checkAutoState();
        checkTelegramStatus();
        fetchNewsSentiment();
        checkPaperState();
        fetchOllamaModels();

        document.querySelectorAll('input[name="mode"]').forEach(r => {
            r.addEventListener('change', () => { updateCostEstimate(); updateModeUI(); });
        });
        document.getElementById('startDate').addEventListener('change', updateCostEstimate);
        document.getElementById('endDate').addEventListener('change', updateCostEstimate);
        document.getElementById('callFrequency').addEventListener('change', updateCostEstimate);

        // LLM provider toggle
        const llmProv = document.getElementById('llmProvider');
        if (llmProv) {
            llmProv.addEventListener('change', function() {
                const isOllama = this.value === 'ollama';
                document.getElementById('ollamaModelSection').style.display = isOllama ? '' : 'none';
                document.getElementById('anthropicNote').style.display = isOllama ? 'none' : '';
                updateCostEstimate();
            });
        }

        // Initial mode UI
        updateModeUI();
    });

    let selectedInterval = '5m';
    // Schwab allows much longer history than yfinance for intraday data
    const intervalLimits = {'1m': 10, '5m': 60, '10m': 60, '15m': 60, '30m': 120, '1h': 365, '1d': 365};

    function initDates() {
        setDatesForInterval(selectedInterval);

        document.querySelectorAll('.interval-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.interval-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                selectedInterval = btn.dataset.val;
                setDatesForInterval(selectedInterval);
                updateCostEstimate();
                const limit = intervalLimits[selectedInterval] || 60;
                const label = selectedInterval === '1d' ? 'Daily bars' : selectedInterval + ' bars';
                document.getElementById('intervalNote').textContent = label + ' — max ' + limit + ' days';
            });
        });
    }

    function setDatesForInterval(interval) {
        const end = new Date();
        const start = new Date();
        const days = intervalLimits[interval] || 30;
        start.setDate(start.getDate() - days);
        document.getElementById('startDate').value = start.toISOString().split('T')[0];
        document.getElementById('endDate').value = end.toISOString().split('T')[0];
    }

    async function loadSymbols() {
        try {
            const resp = await fetch('/api/symbols');
            const data = await resp.json();
            const container = document.getElementById('symbolPresets');
            data.symbols.forEach(s => {
                const btn = document.createElement('span');
                btn.className = 'symbol-preset';
                btn.textContent = s.symbol;
                btn.title = s.name;
                btn.onclick = () => { document.getElementById('symbol').value = s.symbol; lastNewsFetch = ''; fetchNewsSentiment(s.symbol); };
                container.appendChild(btn);
            });
        } catch(e) { console.error('Failed to load symbols:', e); }
    }

    async function checkApiKey() {
        try {
            const resp = await fetch('/api/key-status');
            const data = await resp.json();
            const el = document.getElementById('apiStatus');
            if (data.set) {
                el.textContent = 'API Key: ' + data.masked;
                el.className = 'api-status set';
            } else {
                el.textContent = 'API Key: not set';
                el.className = 'api-status';
            }
        } catch(e) { console.error('Failed to check API key:', e); }
    }

    async function checkDataSources() {
        try {
            const resp = await fetch('/api/data-sources');
            const data = await resp.json();
            const el = document.getElementById('dataSource');
            const sources = [];
            if (data.schwab) sources.push('Schwab');
            if (data.coinbase) sources.push('Coinbase');
            if (data.oanda) sources.push('OANDA');
            if (data.alpaca) sources.push('Alpaca');
            if (data.yfinance) sources.push('yfinance');
            if (sources.length > 0) {
                el.textContent = 'Data: ' + sources.join(' + ');
                el.className = 'api-status set';
            } else {
                el.textContent = 'Data: none!';
                el.style.color = '#f87171';
            }
        } catch(e) { console.error('Failed to check data sources:', e); }
    }

    async function fetchOllamaModels() {
        try {
            const resp = await fetch('/api/ollama/models');
            const data = await resp.json();
            const sel = document.getElementById('ollamaModel');
            if (!sel) return;
            sel.innerHTML = '';
            if (data.models && data.models.length > 0) {
                data.models.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.name;
                    opt.textContent = m.name + ' (' + m.size_gb + 'GB)';
                    sel.appendChild(opt);
                });
            } else {
                sel.innerHTML = '<option value="">No models found — is Ollama running?</option>';
            }
        } catch(e) { console.error('Ollama models fetch failed:', e); }
    }

    function updateCostEstimate() {
        const mode = document.querySelector('input[name="mode"]:checked').value;
        const el = document.getElementById('costEstimate');
        const callEl = document.getElementById('callEstimate');
        if (mode === 'live_api') {
            const start = new Date(document.getElementById('startDate').value);
            const end = new Date(document.getElementById('endDate').value);
            const days = Math.max(1, Math.round((end - start) / (1000 * 60 * 60 * 24)));
            const barsPerDay = {'1m': 390, '5m': 78, '10m': 39, '15m': 26, '30m': 13, '1d': 1}[selectedInterval] || 78;
            const tradingDays = Math.round(days * 0.7);
            const bars = tradingDays * barsPerDay;
            const freq = parseInt(document.getElementById('callFrequency').value) || 1;
            const apiCalls = Math.ceil(bars / freq);
            const provider = document.getElementById('llmProvider') ? document.getElementById('llmProvider').value : 'anthropic';
            if (provider === 'ollama') {
                const model = document.getElementById('ollamaModel') ? document.getElementById('ollamaModel').value : 'local';
                el.textContent = '~' + bars + ' bars, ~' + apiCalls + ' LLM calls (FREE, local GPU: ' + model + ')';
            } else {
                const costPerCall = 0.0003;
                const est = (apiCalls * costPerCall).toFixed(2);
                const estTimeSec = apiCalls * 1.8;
                const estTimeStr = estTimeSec > 120 ? '~' + Math.round(estTimeSec/60) + ' min' : '~' + Math.round(estTimeSec) + 's';
                el.textContent = '~' + bars + ' bars, ~' + apiCalls + ' API calls (~$' + est + ', ' + estTimeStr + ')';
            }
            if (callEl) callEl.textContent = freq > 1 ? 'Skipping ' + (freq-1) + ' of every ' + freq + ' bars' : '';
        } else if (mode === 'rules') {
            el.textContent = 'Rules engine: no API calls, runs instantly';
            if (callEl) callEl.textContent = '';
        } else if (mode === 'paper') {
            el.textContent = 'Live paper trading: polls real-time data, no API cost';
            if (callEl) callEl.textContent = '';
        } else {
            el.textContent = 'Fallback mode: no API calls needed';
            if (callEl) callEl.textContent = '';
        }
    }

    // ======== Mode UI Toggle ========
    let paperActive = false;
    let activeChartSymbol = '';
    let paperSymbols = [];

    function updateModeUI() {
        const mode = document.querySelector('input[name="mode"]:checked').value;
        const isPaper = mode === 'paper';
        const isLiveApi = mode === 'live_api';
        const btnRun = document.getElementById('btnRun');
        const dateSection = document.getElementById('startDate').closest('.panel-section');
        const callFreqSection = document.getElementById('callFrequency').closest('.panel-section');

        // Show/hide LLM provider section
        const llmSection = document.getElementById('llmProviderSection');
        if (llmSection) llmSection.style.display = isLiveApi ? '' : 'none';

        if (isPaper) {
            btnRun.textContent = 'Go Live';
            btnRun.style.background = 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)';
            btnRun.style.color = '#000';
            // Hide date range and call frequency
            if (dateSection) dateSection.style.display = 'none';
            if (callFreqSection) callFreqSection.style.display = 'none';
        } else {
            btnRun.textContent = 'Run Backtest';
            btnRun.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            btnRun.style.color = '#fff';
            if (dateSection) dateSection.style.display = '';
            if (callFreqSection) callFreqSection.style.display = '';
            // Hide paper controls
            document.getElementById('paperControls').style.display = 'none';
        }
    }

    // ======== Paper Trading Functions ========
    async function startPaperTrading() {
        // Parse comma-separated symbols
        const rawInput = document.getElementById('symbol').value.trim().toUpperCase();
        const symbols = rawInput.split(',').map(s => s.trim()).filter(s => s);
        const config = {
            symbols: symbols,
            symbol: symbols[0] || 'BTC-USD',  // backwards compat
            interval: selectedInterval,
            initial_capital: parseFloat(document.getElementById('capital').value),
            commission_pct: parseFloat(document.getElementById('commission').value),
            rules_params: _buildRulesParams(),
        };

        document.getElementById('btnRun').disabled = true;
        document.getElementById('btnRun').textContent = 'Starting...';
        document.getElementById('progressContainer').classList.add('active');
        document.getElementById('progressFill').style.width = '0%';
        document.getElementById('progressText').textContent = 'Starting paper trading...';
        document.getElementById('liveDecisionFeed').innerHTML = '';
        document.getElementById('liveLogContainer').style.display = 'none';
        document.getElementById('resultsContainer').style.display = 'none';
        document.getElementById('emptyState').style.display = 'none';

        // Init chart grid with all symbols
        initChartGrid(symbols);

        // Reset stats
        const initCap = parseFloat(document.getElementById('capital').value);
        updateStatsBar({
            capital: initCap, pnl: 0, return_pct: 0,
            position_side: 'flat', position_text: 'FLAT', unrealized_pnl: 0,
            num_trades: 0, win_rate: 0, peak: initCap, drawdown_pct: 0,
            bar: 0, total_bars: 'LIVE', price: 0
        });

        try {
            const resp = await fetch('/api/paper/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config)
            });
            const data = await resp.json();
            if (data.error) {
                alert('Error: ' + data.error);
                document.getElementById('btnRun').disabled = false;
                document.getElementById('btnRun').textContent = 'Go Live';
                return;
            }
            paperActive = true;
            paperSymbols = data.symbols || symbols;
            activeChartSymbol = paperSymbols[0] || '';
            document.getElementById('btnRun').disabled = true;
            document.getElementById('btnRun').textContent = 'Live';
            document.getElementById('paperControls').style.display = 'block';
            document.getElementById('paperStatusText').textContent = 'Starting...';
        } catch(e) {
            alert('Failed to start paper trading: ' + e);
            document.getElementById('btnRun').disabled = false;
            document.getElementById('btnRun').textContent = 'Go Live';
        }
    }

    async function paperPause() {
        try {
            const resp = await fetch('/api/paper/pause', {method: 'POST'});
            const data = await resp.json();
            const btn = document.getElementById('btnPaperPause');
            if (data.status === 'paused') {
                btn.textContent = 'Resume';
                document.getElementById('paperStatusText').textContent = 'Paused';
            } else {
                btn.textContent = 'Pause';
                document.getElementById('paperStatusText').textContent = 'Running';
            }
        } catch(e) { console.error('Paper pause error:', e); }
    }

    async function paperStop() {
        try {
            await fetch('/api/paper/stop', {method: 'POST'});
            paperActive = false;
            paperSymbols = [];
            activeChartSymbol = '';
            document.getElementById('paperControls').style.display = 'none';
            document.getElementById('btnRun').disabled = false;
            document.getElementById('btnRun').textContent = 'Go Live';
            document.getElementById('paperStatusText').textContent = 'Stopped';
            // Fit charts
            Object.values(chartGrid).forEach(c => { if (c.priceChart) c.priceChart.timeScale().fitContent(); });
            if (equityChart) equityChart.timeScale().fitContent();
        } catch(e) { console.error('Paper stop error:', e); }
    }

    async function paperReset() {
        if (!confirm('Reset paper trading account? This will clear all trades and positions.')) return;
        try {
            await fetch('/api/paper/reset', {method: 'POST'});
            paperActive = false;
            paperSymbols = [];
            activeChartSymbol = '';
            document.getElementById('paperControls').style.display = 'none';
            document.getElementById('btnRun').disabled = false;
            document.getElementById('btnRun').textContent = 'Go Live';
            document.getElementById('paperStatusText').textContent = 'Account reset';
            // Reset stats bar
            const initCap = parseFloat(document.getElementById('capital').value);
            updateStatsBar({
                capital: initCap, pnl: 0, return_pct: 0,
                position_side: 'flat', position_text: 'FLAT', unrealized_pnl: 0,
                num_trades: 0, win_rate: 0, peak: initCap, drawdown_pct: 0,
                bar: 0, total_bars: 0, price: 0
            });
        } catch(e) { console.error('Paper reset error:', e); }
    }

    async function checkPaperState() {
        try {
            const resp = await fetch('/api/paper/state');
            const data = await resp.json();
            if (data.status === 'running' || data.status === 'paused' || data.status === 'market_closed') {
                // Paper trader is active on server — restore UI
                paperActive = true;
                // Select paper radio
                document.querySelectorAll('input[name="mode"]').forEach(r => {
                    r.checked = r.value === 'paper';
                });
                updateModeUI();
                // Set symbol input — comma-separated for multi
                paperSymbols = data.symbols || [data.symbol];
                activeChartSymbol = data.active_chart_symbol || paperSymbols[0] || '';
                document.getElementById('symbol').value = paperSymbols.join(', ');
                // Show controls
                document.getElementById('paperControls').style.display = 'block';
                document.getElementById('btnRun').disabled = true;
                document.getElementById('btnRun').textContent = 'Live';
                const statusMsg = data.status === 'paused' ? 'Paused' :
                                  data.status === 'market_closed' ? 'Market closed' : 'Running';
                document.getElementById('paperStatusText').textContent = statusMsg;
                if (data.status === 'paused') {
                    document.getElementById('btnPaperPause').textContent = 'Resume';
                }
                // Init chart grid for all symbols — the server will re-send bars on next poll
                initChartGrid(paperSymbols);
            }
        } catch(e) { console.error('Failed to check paper state:', e); }
    }

    // ======== WebSocket ========
    function connectWebSocket() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/ws`);

        ws.onopen = () => {
            document.getElementById('wsDot').className = 'status-dot connected';
            document.getElementById('wsStatus').textContent = 'Connected';
        };

        ws.onclose = () => {
            document.getElementById('wsDot').className = 'status-dot disconnected';
            document.getElementById('wsStatus').textContent = 'Disconnected';
            setTimeout(connectWebSocket, 3000);
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            handleWSMessage(msg);
        };
    }

    const signalColors = {
        'BUY': '#4ade80', 'SELL': '#f87171', 'CLOSE_LONG': '#fbbf24',
        'CLOSE_SHORT': '#38bdf8', 'HOLD': '#888', 'NONE': '#555',
        'SKIP': '#333', 'SUMMARY': '#667eea', 'ERROR': '#f87171',
        'SHORT': '#f87171', 'COVER': '#38bdf8',
        'EOD_CLOSE': '#fbbf24', 'EOD_COVER': '#fbbf24',
    };
    const signalBg = {
        'BUY': 'rgba(74,222,128,0.08)', 'SELL': 'rgba(248,113,113,0.08)',
        'CLOSE_LONG': 'rgba(251,191,36,0.08)', 'CLOSE_SHORT': 'rgba(56,189,248,0.08)',
        'HOLD': 'rgba(136,136,136,0.04)', 'NONE': 'rgba(50,50,50,0.04)',
        'SHORT': 'rgba(248,113,113,0.08)', 'COVER': 'rgba(56,189,248,0.08)',
        'EOD_CLOSE': 'rgba(251,191,36,0.08)', 'EOD_COVER': 'rgba(251,191,36,0.08)',
    };

    function formatMoney(v) {
        const sign = v < 0 ? '-' : '';
        return sign + '$' + Math.abs(v).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    }

    function updateStatsBar(s) {
        const bar = document.getElementById('liveStats');
        bar.classList.add('active');

        document.getElementById('statCapital').textContent = formatMoney(s.capital);
        const pnlEl = document.getElementById('statPnl');
        pnlEl.textContent = formatMoney(s.pnl);
        pnlEl.className = 'stat-value ' + (s.pnl > 0 ? 'positive' : s.pnl < 0 ? 'negative' : 'neutral');

        const retEl = document.getElementById('statReturn');
        retEl.textContent = (s.return_pct > 0 ? '+' : '') + s.return_pct.toFixed(2) + '%';
        retEl.className = 'stat-value ' + (s.return_pct > 0 ? 'positive' : s.return_pct < 0 ? 'negative' : 'neutral');

        const posEl = document.getElementById('statPosition');
        posEl.textContent = s.position_text;
        posEl.className = 'position-badge ' + s.position_side;

        const upnlEl = document.getElementById('statUnrealPnl');
        if (s.position_side === 'flat') {
            upnlEl.textContent = '\u2014';
            upnlEl.className = 'stat-value neutral';
        } else {
            upnlEl.textContent = formatMoney(s.unrealized_pnl);
            upnlEl.className = 'stat-value ' + (s.unrealized_pnl > 0 ? 'positive' : s.unrealized_pnl < 0 ? 'negative' : 'neutral');
        }

        document.getElementById('statTrades').textContent = s.num_trades;
        document.getElementById('statWinRate').textContent = s.num_trades > 0 ? s.win_rate.toFixed(1) + '%' : '\u2014';
        document.getElementById('statPeak').textContent = formatMoney(s.peak);

        const ddEl = document.getElementById('statDrawdown');
        ddEl.textContent = s.drawdown_pct.toFixed(2) + '%';
        ddEl.className = 'stat-value ' + (s.drawdown_pct < -1 ? 'negative' : 'neutral');

        document.getElementById('statBar').textContent = s.total_bars === 'LIVE' ? s.bar + ' (LIVE)' : s.bar + '/' + s.total_bars;

        // Price: show all symbol prices if multi-ticker, single price otherwise
        const priceEl = document.getElementById('statPrice');
        if (s.price && typeof s.price === 'object') {
            const parts = Object.entries(s.price).map(([sym, p]) => sym + ': $' + p);
            priceEl.textContent = parts.join('  ');
        } else {
            priceEl.textContent = '$' + (s.price || 0);
        }
    }

    function handleWSMessage(msg) {
        if (msg.type === 'bar') {
            // Route bar to correct chart (addBarToChart handles per-symbol routing)
            addBarToChart(msg);
        } else if (msg.type === 'seed_complete') {
            // Fit content for the completed symbol's chart
            const chart = msg.symbol ? chartGrid[msg.symbol] : null;
            if (chart && chart.priceChart) chart.priceChart.timeScale().fitContent();
        } else if (msg.type === 'progress') {
            document.getElementById('progressFill').style.width = msg.progress + '%';
            document.getElementById('progressText').textContent = msg.message;
        } else if (msg.type === 'stats') {
            updateStatsBar(msg);
        } else if (msg.type === 'decision') {
            // Full Claude decision card
            const container = document.getElementById('liveLogContainer');
            const feed = document.getElementById('liveDecisionFeed');
            container.style.display = 'block';
            document.getElementById('emptyState').style.display = 'none';

            const color = signalColors[msg.signal] || '#888';
            const bg = signalBg[msg.signal] || 'rgba(50,50,50,0.04)';
            const isAction = msg.signal !== 'HOLD' && msg.signal !== 'NONE';
            const action = msg.action || msg.signal;

            // Determine action color (COVERING/CLOSING use different colors than raw signal)
            const isClosing = action.startsWith('COVERING') || action.startsWith('CLOSING');
            const isOpening = action.startsWith('OPENING');
            const actionColor = isClosing ? '#fbbf24' : isOpening ? color : '#888';

            const card = document.createElement('div');
            card.style.cssText = `background:${bg}; border:1px solid ${isAction ? color+'55' : '#222'}; border-left:3px solid ${isAction ? actionColor : '#333'}; border-radius:6px; padding:10px 12px; font-size:0.82em;`;

            // Top line: bar info + date + price + symbol badge
            let topLine = `<div style="display:flex; align-items:center; gap:8px; margin-bottom:4px; font-size:0.85em;">`;
            if (msg.symbol && paperSymbols.length > 1) {
                topLine += `<span style="background:#fbbf2422; color:#fbbf24; padding:1px 6px; border-radius:3px; font-weight:700; font-size:0.9em;">${msg.symbol}</span>`;
            }
            topLine += `<span style="color:#555;">Bar ${msg.bar}${msg.total_bars === 'LIVE' ? ' (LIVE)' : '/' + msg.total_bars}</span>`;
            topLine += `<span style="color:#888;">${msg.date}</span>`;
            topLine += `<span style="color:#ccc; font-weight:600;">$${msg.price}</span>`;
            if (msg.strength > 0) topLine += `<span style="color:#666;">str: ${msg.strength}</span>`;
            topLine += `<span style="color:#555; margin-left:auto;">via ${msg.source || 'strategy'}</span>`;
            topLine += `</div>`;

            // Action line — the main takeaway, big and clear
            let actionLine = `<div style="display:flex; align-items:center; gap:8px; margin:6px 0;">`;
            actionLine += `<span style="background:${actionColor}22; color:${actionColor}; padding:3px 10px; border-radius:4px; font-weight:700; font-size:1.05em;">${action}</span>`;
            actionLine += `</div>`;

            // Position context (smaller, secondary)
            let posLine = '';
            if (msg.position && msg.position !== 'FLAT') {
                posLine = `<div style="color:#777; font-size:0.85em; margin-bottom:4px;">Position before: ${msg.position}</div>`;
            }

            // Reasoning
            let body = '';
            if (msg.reasoning) {
                body += `<div style="color:#bbb; margin:6px 0; line-height:1.45;">${msg.reasoning}</div>`;
            }

            // Risk notes
            if (msg.risk_notes) {
                body += `<div style="color:#c9a040; font-size:0.9em; margin:4px 0;">Risk: ${msg.risk_notes}</div>`;
            }

            // Footer: indicators, regime
            let footer = `<div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; font-size:0.8em;">`;
            if (msg.indicators) footer += `<span style="color:#556; background:#0a0a0a; padding:1px 6px; border-radius:3px;">${msg.indicators}</span>`;
            if (msg.regime) footer += `<span style="color:#764ba2; background:#0a0a0a; padding:1px 6px; border-radius:3px;">${msg.regime}</span>`;
            footer += `</div>`;

            card.innerHTML = topLine + actionLine + posLine + body + footer;
            feed.appendChild(card);
            feed.scrollTop = feed.scrollHeight;

        } else if (msg.type === 'log') {
            // Lightweight skip/heartbeat line
            const container = document.getElementById('liveLogContainer');
            const feed = document.getElementById('liveDecisionFeed');
            container.style.display = 'block';
            document.getElementById('emptyState').style.display = 'none';

            const line = document.createElement('div');
            line.style.cssText = 'font-size:0.72em; color:#444; padding:1px 12px; font-family:monospace;';
            line.textContent = `Bar ${msg.bar} | ${msg.date} | $${msg.price} | ${msg.indicators || ''}`;
            feed.appendChild(line);
            // Only auto-scroll if near bottom
            if (feed.scrollHeight - feed.scrollTop - feed.clientHeight < 100) {
                feed.scrollTop = feed.scrollHeight;
            }

        } else if (msg.type === 'trade') {
            // Add marker to live chart
            addTradeMarker(msg.trade);

            // Flash trade notification at top
            const feed = document.getElementById('liveDecisionFeed');
            if (feed) {
                const t = msg.trade;
                const color = signalColors[t.type] || '#888';
                const note = document.createElement('div');
                note.style.cssText = `background:${color}18; border:1px solid ${color}44; border-radius:4px; padding:6px 12px; font-size:0.85em; font-weight:600; color:${color};`;
                const symPrefix = t.symbol && paperSymbols.length > 1 ? `[${t.symbol}] ` : '';
                let text = `${symPrefix}TRADE EXECUTED: ${t.type} ${t.shares} shares @ $${t.price?.toFixed(2)}`;
                if (t.pnl) text += ` | P&L: $${t.pnl.toFixed(2)}`;
                if (t.return_pct) text += ` (${t.return_pct > 0 ? '+' : ''}${t.return_pct.toFixed(2)}%)`;
                note.textContent = text;
                if (t.alpaca_order && !t.alpaca_order.error) {
                    const badge = document.createElement('span');
                    badge.style.cssText = 'background:#22c55e; color:#fff; padding:1px 6px; border-radius:3px; font-size:0.75em; margin-left:8px; vertical-align:middle;';
                    badge.textContent = 'ALPACA';
                    note.appendChild(badge);
                }
                feed.appendChild(note);
                feed.scrollTop = feed.scrollHeight;
            }

        } else if (msg.type === 'complete') {
            document.getElementById('btnRun').disabled = false;
            document.getElementById('btnRun').textContent = 'Run Backtest';
            // Fit all chart data into view
            Object.values(chartGrid).forEach(c => { if (c.priceChart) c.priceChart.timeScale().fitContent(); });
            if (equityChart) equityChart.timeScale().fitContent();
            fetchResults();
        } else if (msg.type === 'error') {
            document.getElementById('btnRun').disabled = false;
            document.getElementById('btnRun').textContent = 'Run Backtest';
            document.getElementById('progressText').textContent = 'Error: ' + msg.message;

        // === Autonomous engine messages ===
        } else if (msg.type === 'auto_status') {
            updateAutoUI(msg);
        } else if (msg.type === 'auto_result') {
            addAutoResultCard(msg.result);
        } else if (msg.type === 'auto_leaderboard') {
            renderLeaderboard(msg.leaderboard);

        // === Paper trading messages ===
        } else if (msg.type === 'paper_status') {
            const statusEl = document.getElementById('paperStatusText');
            if (statusEl) statusEl.textContent = msg.message || msg.status;
            if (msg.status === 'stopped' || msg.status === 'error') {
                paperActive = false;
                paperSymbols = [];
                activeChartSymbol = '';
                document.getElementById('paperControls').style.display = 'none';
                document.getElementById('btnRun').disabled = false;
                document.getElementById('btnRun').textContent = 'Go Live';
            } else if (msg.status === 'market_closed') {
                if (statusEl) statusEl.textContent = 'Market closed — waiting...';
            } else if (msg.status === 'running') {
                document.getElementById('progressContainer').classList.remove('active');
            }

        // === Real-time news from Alpaca ===
        } else if (msg.type === 'news') {
            const feed = document.getElementById('liveDecisionFeed');
            const container = document.getElementById('liveLogContainer');
            if (!feed || !container) return;
            container.style.display = 'block';
            const empty = document.getElementById('emptyState');
            if (empty) empty.style.display = 'none';

            const sentColors = {bullish:'#4ade80', bearish:'#f87171', neutral:'#888'};
            const sentColor = sentColors[msg.sentiment_label] || '#888';
            const card = document.createElement('div');
            card.style.cssText = 'background:linear-gradient(135deg,#1a0f2e,#0d0d0d);border:1px solid #3d2963;border-left:3px solid #8b5cf6;border-radius:6px;padding:10px 12px;font-size:0.82em;margin-bottom:4px;';

            // Header row: NEWS badge + symbol badges + time + source
            let h = '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:0.85em;flex-wrap:wrap;">';
            h += '<span style="background:#8b5cf622;color:#8b5cf6;padding:2px 8px;border-radius:3px;font-weight:700;font-size:0.85em;">NEWS</span>';
            if (msg.symbols && msg.symbols.length) {
                msg.symbols.filter(s => paperSymbols.includes(s)).slice(0,3).forEach(s => {
                    h += '<span style="background:#fbbf2422;color:#fbbf24;padding:1px 5px;border-radius:3px;font-weight:600;font-size:0.8em;">' + s + '</span>';
                });
            }
            if (msg.created_at) {
                try {
                    const t = new Date(msg.created_at);
                    h += '<span style="color:#888;font-size:0.9em;">' + t.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'}) + '</span>';
                } catch(e) {}
            }
            if (msg.source) h += '<span style="color:#666;font-size:0.85em;margin-left:auto;">' + msg.source + '</span>';
            h += '</div>';

            // Headline (clickable if URL provided)
            const safeHL = (msg.headline||'').replace(/</g,'&lt;').replace(/>/g,'&gt;');
            const hl = msg.url
                ? '<a href="' + msg.url + '" target="_blank" rel="noopener" style="color:#e0e0e0;text-decoration:none;font-weight:600;line-height:1.4;display:block;margin-bottom:6px;">' + safeHL + '</a>'
                : '<div style="color:#e0e0e0;font-weight:600;line-height:1.4;margin-bottom:6px;">' + safeHL + '</div>';

            // Summary (truncated)
            let sum = '';
            if (msg.summary) {
                const safeSum = msg.summary.replace(/</g,'&lt;').replace(/>/g,'&gt;');
                sum = '<div style="color:#999;font-size:0.9em;line-height:1.35;margin-bottom:6px;">' + (safeSum.length > 200 ? safeSum.slice(0,200) + '...' : safeSum) + '</div>';
            }

            // Sentiment footer
            const scoreStr = msg.sentiment_score > 0 ? '+' + msg.sentiment_score : '' + msg.sentiment_score;
            const sf = '<div style="font-size:0.8em;"><span style="color:' + sentColor + ';background:' + sentColor + '22;padding:2px 8px;border-radius:3px;font-weight:600;">' + msg.sentiment_label.toUpperCase() + ' ' + scoreStr + '</span></div>';

            card.innerHTML = h + hl + sum + sf;
            feed.appendChild(card);
            if (feed.scrollHeight - feed.scrollTop - feed.clientHeight < 100) {
                feed.scrollTop = feed.scrollHeight;
            }
        }
    }

    // ======== Run Backtest ========
    async function runBacktest() {
        // Branch: if paper mode, start paper trading instead
        const selectedMode = document.querySelector('input[name="mode"]:checked').value;
        if (selectedMode === 'paper') {
            return startPaperTrading();
        }

        const config = {
            symbol: document.getElementById('symbol').value.trim().toUpperCase(),
            start_date: document.getElementById('startDate').value,
            end_date: document.getElementById('endDate').value,
            interval: selectedInterval,
            initial_capital: parseFloat(document.getElementById('capital').value),
            commission_pct: parseFloat(document.getElementById('commission').value),
            call_frequency: parseInt(document.getElementById('callFrequency').value),
            mode: selectedMode,
            trading_style: document.querySelector('input[name="trading_style"]:checked').value,
            factors: {
                price_action: document.getElementById('fPrice').checked,
                technical_indicators: document.getElementById('fTech').checked,
                news_sentiment: document.getElementById('fNews').checked,
                market_regime: document.getElementById('fRegime').checked,
            },
            rules_params: _buildRulesParams(),
        };

        // Add LLM provider config for live_api mode
        if (selectedMode === 'live_api') {
            const provEl = document.getElementById('llmProvider');
            config.llm_provider = provEl ? provEl.value : 'anthropic';
            if (config.llm_provider === 'ollama') {
                const modelEl = document.getElementById('ollamaModel');
                config.ollama_model = modelEl ? modelEl.value : 'qwen3-coder:30b';
            }
        }

        document.getElementById('btnRun').disabled = true;
        document.getElementById('btnRun').textContent = 'Running...';
        document.getElementById('progressContainer').classList.add('active');
        document.getElementById('progressFill').style.width = '0%';
        document.getElementById('progressText').textContent = 'Starting...';
        document.getElementById('liveDecisionFeed').innerHTML = '';
        document.getElementById('liveLogContainer').style.display = 'none';
        document.getElementById('resultsContainer').style.display = 'none';
        document.getElementById('emptyState').style.display = 'none';
        // Initialize live chart — parse comma-separated symbols for multi-symbol grid
        const rawSymbols = config.symbol.split(',').map(s => s.trim()).filter(s => s);
        const backtestSymbols = rawSymbols.length > 0 ? rawSymbols : ['SPY'];
        initChartGrid(backtestSymbols);
        // Reset stats bar with initial capital
        const initCap = parseFloat(document.getElementById('capital').value);
        updateStatsBar({
            capital: initCap, pnl: 0, return_pct: 0,
            position_side: 'flat', position_text: 'FLAT', unrealized_pnl: 0,
            num_trades: 0, win_rate: 0, peak: initCap, drawdown_pct: 0,
            bar: 0, total_bars: 0, price: 0
        });

        try {
            const resp = await fetch('/api/backtest/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config)
            });
            const data = await resp.json();
            if (data.error) {
                alert('Error: ' + data.error);
                document.getElementById('btnRun').disabled = false;
                document.getElementById('btnRun').textContent = 'Run Backtest';
            }
        } catch(e) {
            alert('Failed to start backtest: ' + e);
            document.getElementById('btnRun').disabled = false;
            document.getElementById('btnRun').textContent = 'Run Backtest';
        }
    }

    async function runOptimizer() {
        const rawSymbol = document.getElementById('symbol').value.trim().toUpperCase();
        const symbols = rawSymbol.split(',').map(s => s.trim()).filter(s => s);
        const isMulti = symbols.length > 1;

        const config = {
            symbol: rawSymbol,
            start_date: document.getElementById('startDate').value,
            end_date: document.getElementById('endDate').value,
            interval: selectedInterval,
            initial_capital: parseFloat(document.getElementById('capital').value),
            commission_pct: parseFloat(document.getElementById('commission').value),
            trading_style: document.querySelector('input[name="trading_style"]:checked').value,
            slippage_factor: parseFloat(document.getElementById('slippage').value) || 0,
            ..._getAdaptiveRiskParams(),
        };

        document.getElementById('btnOptimize').disabled = true;
        document.getElementById('btnOptimize').textContent = isMulti ? `Optimizing ${symbols.length} symbols...` : 'Optimizing...';
        document.getElementById('progressContainer').classList.add('active');
        document.getElementById('progressFill').style.width = '0%';
        document.getElementById('progressText').textContent = isMulti ? `Starting multi-symbol optimizer (${symbols.join(', ')})...` : 'Starting optimizer...';

        try {
            const endpoint = isMulti ? '/api/optimize/multi' : '/api/optimize';
            const resp = await fetch(endpoint, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config)
            });
            const data = await resp.json();
            document.getElementById('btnOptimize').disabled = false;
            document.getElementById('btnOptimize').textContent = 'Optimize Parameters';

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            // Show results in the decision feed area
            const container = document.getElementById('liveLogContainer');
            const feed = document.getElementById('liveDecisionFeed');
            container.style.display = 'block';
            document.getElementById('emptyState').style.display = 'none';
            feed.innerHTML = '';

            if (isMulti && data.per_symbol) {
                // ── Multi-symbol results ──
                const header = document.createElement('div');
                header.style.cssText = 'color:#ccc;margin-bottom:12px;font-size:0.9em;';
                header.innerHTML = `<strong>Per-Symbol Optimization</strong> — ${symbols.join(', ')} ${selectedInterval}`;
                feed.appendChild(header);

                const allBestParams = {};
                for (const sym of symbols) {
                    const sr = data.per_symbol[sym];
                    const symDiv = document.createElement('div');
                    symDiv.style.cssText = 'margin-bottom:16px;';

                    if (sr && sr.error) {
                        symDiv.innerHTML = `<div style="color:#f87171;font-size:0.85em;"><strong>${sym}</strong>: ${sr.error}</div>`;
                        feed.appendChild(symDiv);
                        continue;
                    }
                    if (!sr || !sr.best_params) {
                        symDiv.innerHTML = `<div style="color:#888;font-size:0.85em;"><strong>${sym}</strong>: No results</div>`;
                        feed.appendChild(symDiv);
                        continue;
                    }

                    allBestParams[sym] = sr.best_params;
                    const retColor = sr.return_pct >= 0 ? '#4ade80' : '#f87171';
                    const bp = sr.best_params;

                    let html = `<div style="color:#ccc;font-weight:600;margin-bottom:4px;font-size:0.9em;">`;
                    html += `<span style="background:#667eea33;color:#667eea;padding:2px 8px;border-radius:3px;margin-right:6px;">${sym}</span>`;
                    html += `<span style="color:${retColor};">${sr.return_pct > 0 ? '+' : ''}${sr.return_pct}%</span>`;
                    html += ` · ${sr.num_trades} trades · ${sr.win_rate}% WR · PF ${sr.profit_factor}`;
                    html += ` · DD ${sr.max_drawdown}%`;
                    html += ` · ${sr.bars} bars · ${sr.valid_combos} valid combos</div>`;

                    // Compact param summary
                    html += `<div style="color:#888;font-size:0.8em;margin-left:4px;">`;
                    html += `Stop=${bp.atr_stop_mult}x · Target=${bp.atr_target_mult}x · Triggers=${bp.min_triggers} · PB=${bp.pullback_zone_atr}`;
                    if (bp.skip_open_minutes !== undefined) html += ` · OpenSkip=${bp.skip_open_minutes}m`;
                    if (bp.max_trades_per_day !== undefined) html += ` · MaxTPD=${bp.max_trades_per_day}`;
                    html += `</div>`;

                    // Top 3 alternatives
                    if (sr.top3 && sr.top3.length > 1) {
                        html += `<details style="margin-top:4px;"><summary style="color:#555;font-size:0.75em;cursor:pointer;">Top ${sr.top3.length} alternatives</summary>`;
                        html += `<table style="width:100%;border-collapse:collapse;font-size:0.75em;margin-top:4px;">`;
                        html += `<tr style="color:#555;"><th style="text-align:left;padding:2px 4px;">#</th><th>Return</th><th>WR</th><th>PF</th><th>Stop</th><th>Target</th><th>Trig</th><th>PB</th></tr>`;
                        sr.top3.forEach((alt, ai) => {
                            const ac = alt.return_pct >= 0 ? '#4ade80' : '#f87171';
                            const ap = alt.params;
                            html += `<tr style="border-top:1px solid #1a1a1a;">`;
                            html += `<td style="padding:2px 4px;color:#555;">${ai+1}</td>`;
                            html += `<td style="color:${ac};">${alt.return_pct > 0 ? '+' : ''}${alt.return_pct}%</td>`;
                            html += `<td>${alt.win_rate}%</td><td>${alt.profit_factor}</td>`;
                            html += `<td>${ap.atr_stop_mult}x</td><td>${ap.atr_target_mult}x</td>`;
                            html += `<td>${ap.min_triggers}</td><td>${ap.pullback_zone_atr}</td>`;
                            html += `</tr>`;
                        });
                        html += `</table></details>`;
                    }

                    symDiv.innerHTML = html;
                    feed.appendChild(symDiv);
                }

                // "Use All" button — applies per-symbol params
                if (Object.keys(allBestParams).length > 0) {
                    const btnDiv = document.createElement('div');
                    btnDiv.style.cssText = 'margin-top:12px;display:flex;gap:8px;align-items:center;';
                    const useAllBtn = document.createElement('button');
                    useAllBtn.textContent = `Use All ${Object.keys(allBestParams).length} Configs`;
                    useAllBtn.style.cssText = 'background:#4ade8022;color:#4ade80;border:1px solid #4ade8044;border-radius:4px;padding:6px 16px;cursor:pointer;font-size:0.9em;font-weight:600;';
                    useAllBtn.onclick = () => applyParams({per_symbol: allBestParams});
                    btnDiv.appendChild(useAllBtn);

                    // Individual "Use" buttons per symbol
                    for (const [sym, bp] of Object.entries(allBestParams)) {
                        const btn = document.createElement('button');
                        btn.textContent = `Use ${sym}`;
                        btn.style.cssText = 'background:#667eea22;color:#667eea;border:1px solid #667eea44;border-radius:3px;padding:4px 10px;cursor:pointer;font-size:0.8em;';
                        btn.onclick = () => applyParams(bp);
                        btnDiv.appendChild(btn);
                    }
                    feed.appendChild(btnDiv);

                    const hint = document.createElement('div');
                    hint.style.cssText = 'color:#667eea;font-size:0.8em;margin-top:8px;';
                    hint.textContent = 'Use All applies per-symbol tuning. Individual buttons apply that symbol params globally.';
                    feed.appendChild(hint);
                }

            } else {
                // ── Single-symbol results (existing behavior) ──
                const header = document.createElement('div');
                header.style.cssText = 'color:#ccc;margin-bottom:12px;font-size:0.9em;';
                header.innerHTML = `<strong>Optimization Results</strong> — ${config.symbol} ${selectedInterval} — Tested ${data.total_tested} combinations on ${data.bars} bars, ${data.valid_results} valid`;
                feed.appendChild(header);

                if (data.best && data.best.length > 0) {
                    let table = '<table style="width:100%;border-collapse:collapse;font-size:0.8em;">';
                    table += '<tr style="color:#888;border-bottom:1px solid #333;text-align:left;">';
                    table += '<th style="padding:6px 4px;">#</th><th>Return</th><th>Trades</th><th>Win%</th><th>MaxDD</th><th>PF</th>';
                    table += '<th>Stop</th><th>Target</th><th>Triggers</th><th>PB Zone</th><th></th></tr>';

                    data.best.forEach((r, i) => {
                        const retColor = r.return_pct >= 0 ? '#4ade80' : '#f87171';
                        const p = r.params;
                        table += `<tr style="border-bottom:1px solid #1a1a1a;${i===0?'background:#4ade8010;':''}">`;
                        table += `<td style="padding:5px 4px;color:#555;">${i+1}</td>`;
                        table += `<td style="color:${retColor};font-weight:600;">${r.return_pct>0?'+':''}${r.return_pct}%</td>`;
                        table += `<td>${r.num_trades}</td>`;
                        table += `<td>${r.win_rate}%</td>`;
                        table += `<td style="color:${r.max_drawdown>10?'#f87171':'#888'};">${r.max_drawdown}%</td>`;
                        table += `<td>${r.profit_factor}</td>`;
                        table += `<td>${p.atr_stop_mult}x</td>`;
                        table += `<td>${p.atr_target_mult}x</td>`;
                        table += `<td>${p.min_triggers}</td>`;
                        table += `<td>${p.pullback_zone_atr}</td>`;
                        table += `<td><button onclick="applyParams(${JSON.stringify(p).replace(/"/g,'&quot;')})" style="background:#667eea22;color:#667eea;border:1px solid #667eea44;border-radius:3px;padding:2px 8px;cursor:pointer;font-size:0.9em;">Use</button></td>`;
                        table += '</tr>';
                    });
                    table += '</table>';

                    const tableDiv = document.createElement('div');
                    tableDiv.innerHTML = table;
                    feed.appendChild(tableDiv);

                    const hint = document.createElement('div');
                    hint.style.cssText = 'color:#667eea;font-size:0.8em;margin-top:10px;';
                    hint.textContent = 'Click "Use" to apply those parameters, then run a backtest to see detailed results.';
                    feed.appendChild(hint);
                } else {
                    feed.innerHTML += '<div style="color:#888;">No profitable configurations found. Try a different symbol or date range.</div>';
                }
            }
        } catch(e) {
            alert('Optimizer failed: ' + e);
            document.getElementById('btnOptimize').disabled = false;
            document.getElementById('btnOptimize').textContent = 'Optimize Parameters';
        }
    }

    // Adaptive risk params helper — reads checkbox + fields, returns object
    function _getAdaptiveRiskParams() {
        // Universal safety params — always included regardless of adaptive risk toggle
        const params = {};
        const maxLoss = parseFloat(document.getElementById('maxTradeLossPct').value);
        if (maxLoss > 0) params.max_trade_loss_pct = maxLoss / 100;
        const maxGap = parseFloat(document.getElementById('maxGapPct').value);
        if (maxGap > 0) params.max_gap_pct = maxGap / 100;
        const newsThresh = parseFloat(document.getElementById('newsSentimentThreshold').value);
        if (newsThresh > 0) params.news_sentiment_threshold = newsThresh;
        const ctPenalty = parseInt(document.getElementById('counterTrendPenalty').value);
        if (ctPenalty >= 0) params.counter_trend_penalty = ctPenalty;

        // Adaptive risk params — only when checkbox is on
        const on = document.getElementById('adaptiveRisk') && document.getElementById('adaptiveRisk').checked;
        if (!on) return params;
        params.adaptive_risk = true;
        params.risk_per_trade = parseFloat(document.getElementById('riskPerTrade').value) / 100 || 0.02;
        params.max_stop_pct = parseFloat(document.getElementById('maxStopPct').value) / 100 || 0.04;
        params.drawdown_half_risk = 0.05;
        params.drawdown_skip = 0.10;
        params.cooldown_losses = parseInt(document.getElementById('cooldownLosses').value) || 3;
        params.reentry_cooldown_bars = parseInt(document.getElementById('reentryCooldown').value) || 5;
        return params;
    }

    // Toggle adaptive risk param visibility
    if (document.getElementById('adaptiveRisk')) {
        document.getElementById('adaptiveRisk').addEventListener('change', function() {
            document.getElementById('adaptiveRiskParams').style.display = this.checked ? '' : 'none';
        });
    }

    // Show Patterns toggle
    if (document.getElementById('showPatterns')) {
        document.getElementById('showPatterns').addEventListener('change', function() {
            showPatterns = this.checked;
            // Re-render markers on all charts (toggle pattern markers on/off)
            Object.values(chartGrid).forEach(c => _mergeAndSetMarkers(c));
        });
    }

    // Apply optimized params (stores in a global, sent with backtest config)
    // Can be either flat params {atr_stop_mult: 2, ...} or per-symbol {per_symbol: {PLTR: {...}, NVDA: {...}}}
    let optimizedParams = null;
    function applyParams(params) {
        optimizedParams = params;
        const feed = document.getElementById('liveDecisionFeed');
        const note = document.createElement('div');
        note.style.cssText = 'color:#4ade80;font-size:0.85em;margin-top:8px;padding:8px;background:#4ade8010;border:1px solid #4ade8033;border-radius:4px;';
        if (params.per_symbol) {
            const syms = Object.keys(params.per_symbol);
            note.textContent = `Applied per-symbol params for ${syms.join(', ')} — now click "Run Backtest" with Rules Engine mode.`;
        } else {
            note.textContent = `Applied: Stop=${params.atr_stop_mult}xATR Target=${params.atr_target_mult}xATR Triggers=${params.min_triggers} PullbackZone=${params.pullback_zone_atr}ATR — now click "Run Backtest" with Rules Engine mode.`;
        }
        feed.appendChild(note);
        feed.scrollTop = feed.scrollHeight;
    }

    // Build rules_params merging optimized + adaptive risk.  Preserves per_symbol structure.
    function _buildRulesParams() {
        const adaptive = _getAdaptiveRiskParams();
        const opt = optimizedParams || {};
        if (opt.per_symbol) {
            // Per-symbol mode: merge adaptive risk into each symbol's params AND base
            const merged = Object.assign({}, adaptive);
            merged.per_symbol = {};
            for (const [sym, sp] of Object.entries(opt.per_symbol)) {
                merged.per_symbol[sym] = Object.assign({}, sp, adaptive);
            }
            return merged;
        }
        return Object.assign({}, opt, adaptive);
    }

    async function fetchResults() {
        try {
            const resp = await fetch('/api/backtest/results');
            const data = await resp.json();
            if (data.error) {
                console.error('Results error:', data.error);
                return;
            }
            currentResults = data;
            renderResults(data);
        } catch(e) { console.error('Failed to fetch results:', e); }
    }

    // ======== Render Results ========
    function renderResults(data) {
        document.getElementById('emptyState').style.display = 'none';
        document.getElementById('resultsContainer').style.display = 'block';

        renderMetrics(data.metrics);
        // Show per-symbol metrics if available
        if (data.per_symbol_metrics && Object.keys(data.per_symbol_metrics).length > 1) {
            renderPerSymbolMetrics(data.per_symbol_metrics);
        }
        renderEquityChart(data.equity_curve, data.buy_hold_curve);
        renderDrawdownChart(data.equity_curve, data.metrics.initial_capital);
        renderTradeTable(data.trades);
    }

    function renderPerSymbolMetrics(perSym) {
        // Add/update per-symbol breakdown after main metrics grid
        let el = document.getElementById('perSymbolMetrics');
        if (!el) {
            el = document.createElement('div');
            el.id = 'perSymbolMetrics';
            el.style.cssText = 'margin-top:12px;';
            const metricsGrid = document.getElementById('metricsGrid');
            if (metricsGrid) metricsGrid.parentNode.insertBefore(el, metricsGrid.nextSibling);
        }

        let html = '<details style="background:#111;border:1px solid #333;border-radius:6px;padding:8px 12px;"><summary style="cursor:pointer;color:#667eea;font-weight:600;font-size:0.9em;">Per-Symbol Breakdown</summary>';
        html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:8px;margin-top:8px;">';
        for (const [sym, m] of Object.entries(perSym)) {
            const retCls = m.total_return >= 0 ? '#4ade80' : '#f87171';
            html += `<div style="background:#0d0d0d;border:1px solid #222;border-radius:4px;padding:8px 10px;">
                <div style="font-weight:700;color:#667eea;margin-bottom:4px;">${sym}</div>
                <div style="display:flex;flex-wrap:wrap;gap:6px;font-size:0.82em;">
                    <span style="color:${retCls}">${m.total_return}% return</span>
                    <span style="color:#888">${m.total_trades} trades</span>
                    <span style="color:#888">${m.win_rate}% win</span>
                    <span style="color:#888">PF ${m.profit_factor}</span>
                    <span style="color:#f87171">${m.max_drawdown}% DD</span>
                </div>
            </div>`;
        }
        html += '</div></details>';
        el.innerHTML = html;
    }

    function renderMetrics(m) {
        const grid = document.getElementById('metricsGrid');
        const cards = [
            { label: 'Total Return', value: m.total_return + '%', cls: m.total_return >= 0 ? 'positive' : 'negative' },
            { label: 'Buy & Hold', value: m.buy_hold_return + '%', cls: m.buy_hold_return >= 0 ? 'positive' : 'negative' },
            { label: 'Sharpe Ratio', value: m.sharpe_ratio, cls: m.sharpe_ratio >= 1 ? 'positive' : m.sharpe_ratio >= 0 ? 'neutral' : 'negative' },
            { label: 'Max Drawdown', value: m.max_drawdown + '%', cls: 'negative' },
            { label: 'Win Rate', value: m.win_rate + '%', cls: m.win_rate >= 50 ? 'positive' : 'negative' },
            { label: 'Total Trades', value: m.total_trades, cls: 'neutral' },
            { label: 'Profit Factor', value: m.profit_factor, cls: m.profit_factor >= 1 ? 'positive' : 'negative' },
            { label: 'Final Value', value: '$' + m.final_value.toLocaleString(), cls: m.total_return >= 0 ? 'positive' : 'negative' },
        ];

        grid.innerHTML = cards.map(c => `
            <div class="metric-card">
                <div class="metric-value ${c.cls}">${c.value}</div>
                <div class="metric-label">${c.label}</div>
            </div>
        `).join('');
    }

    function renderEquityChart(equity, buyHold) {
        const el = document.getElementById('equityChart');
        el.innerHTML = '';
        if (postEquityChart) { postEquityChart.remove(); postEquityChart = null; }

        postEquityChart = LightweightCharts.createChart(el, {
            ...LWC_CHART_OPTS,
            height: 300,
        });

        const stratLine = postEquityChart.addSeries(LightweightCharts.LineSeries, {
            color: '#667eea',
            lineWidth: 2,
            title: 'Strategy',
        });
        const bhLine = postEquityChart.addSeries(LightweightCharts.LineSeries, {
            color: '#888',
            lineWidth: 1,
            lineStyle: 2,
            title: 'Buy & Hold',
        });

        stratLine.setData(equity.map(e => ({ time: parseBarTime(e.date), value: e.value })));
        if (buyHold && buyHold.length) {
            bhLine.setData(buyHold.map(e => ({ time: parseBarTime(e.date), value: e.value })));
        }
        postEquityChart.timeScale().fitContent();

        // Responsive resize
        const ro = new ResizeObserver(() => {
            if (postEquityChart) postEquityChart.applyOptions({ width: el.clientWidth });
        });
        ro.observe(el);
    }

    function renderDrawdownChart(equity, initialCapital) {
        const el = document.getElementById('drawdownChart');
        el.innerHTML = '';
        if (postDrawdownChart) { postDrawdownChart.remove(); postDrawdownChart = null; }

        postDrawdownChart = LightweightCharts.createChart(el, {
            ...LWC_CHART_OPTS,
            height: 200,
        });

        const ddSeries = postDrawdownChart.addSeries(LightweightCharts.AreaSeries, {
            topColor: 'rgba(248,113,113,0.01)',
            bottomColor: 'rgba(248,113,113,0.25)',
            lineColor: '#f87171',
            lineWidth: 1,
            title: 'Drawdown %',
        });

        let peak = initialCapital;
        const ddData = equity.map(e => {
            if (e.value > peak) peak = e.value;
            return { time: parseBarTime(e.date), value: -((peak - e.value) / peak * 100) };
        });

        ddSeries.setData(ddData);
        postDrawdownChart.timeScale().fitContent();

        // Responsive resize
        const ro = new ResizeObserver(() => {
            if (postDrawdownChart) postDrawdownChart.applyOptions({ width: el.clientWidth });
        });
        ro.observe(el);
    }

    function renderTradeTable(trades) {
        const tbody = document.getElementById('tradeBody');
        tbody.innerHTML = '';

        // Detect multi-symbol: check if trades have >1 unique symbol
        const uniqueSyms = new Set(trades.map(t => t.symbol).filter(Boolean));
        const isMultiSym = uniqueSyms.size > 1;

        // Update table header dynamically
        const thead = document.querySelector('#tradeTable thead tr');
        if (thead) {
            if (isMultiSym) {
                thead.innerHTML = '<th>Date</th><th>Symbol</th><th>Signal</th><th>Price</th><th>Shares</th><th>P&L</th><th>Return</th><th>Strength</th>';
            } else {
                thead.innerHTML = '<th>Date</th><th>Signal</th><th>Price</th><th>Shares</th><th>P&L</th><th>Return</th><th>Strength</th>';
            }
        }
        const colSpan = isMultiSym ? 8 : 7;

        trades.forEach((t, idx) => {
            const cls = t.type === 'BUY' ? 'trade-buy' :
                        t.type === 'SHORT' ? 'trade-short' :
                        t.type === 'COVER' ? 'trade-cover' :
                        t.type === 'SELL' || t.type === 'CLOSE' || t.type === 'CLOSE_SHORT' ? 'trade-sell' :
                        (t.type === 'STOP_LOSS' || t.type === 'STOP_LOSS_SHORT') ? 'trade-stop' :
                        (t.type === 'TAKE_PROFIT' || t.type === 'TAKE_PROFIT_SHORT') ? 'trade-tp' : '';
            const pnl = t.pnl ? (t.pnl >= 0 ? '+' : '') + t.pnl.toFixed(2) : '-';
            const retPct = t.return_pct ? (t.return_pct >= 0 ? '+' : '') + t.return_pct.toFixed(1) + '%' : '-';
            const pnlCls = t.pnl > 0 ? 'positive' : t.pnl < 0 ? 'negative' : '';

            const symCol = isMultiSym ? `<td><span style="background:#667eea22;color:#667eea;padding:1px 6px;border-radius:3px;font-weight:600;font-size:0.85em;">${t.symbol || '?'}</span></td>` : '';

            const tr = document.createElement('tr');
            tr.className = `${cls} expandable`;
            tr.innerHTML = `
                <td>${t.date.split(' ')[0]}</td>
                ${symCol}
                <td>${t.type}</td>
                <td>$${t.price.toFixed(2)}</td>
                <td>${t.shares}</td>
                <td class="${pnlCls}">${pnl}</td>
                <td class="${pnlCls}">${retPct}</td>
                <td>${(t.strength * 100).toFixed(0)}%</td>
            `;
            tr.onclick = () => toggleReasoning(idx);
            tbody.appendChild(tr);

            // Reasoning row (hidden)
            const reasoningTr = document.createElement('tr');
            reasoningTr.id = `reasoning-${idx}`;
            reasoningTr.style.display = 'none';
            reasoningTr.innerHTML = `
                <td colspan="${colSpan}">
                    <div class="reasoning-panel active">
                        <h4>Claude's Analysis</h4>
                        <div class="reasoning-section">
                            <div class="rs-label">Reasoning</div>
                            <div class="rs-content">${t.reasoning || 'N/A'}</div>
                        </div>
                        <div class="reasoning-section">
                            <div class="rs-label">Risk Notes</div>
                            <div class="rs-content">${t.risk_notes || 'N/A'}</div>
                        </div>
                        <div class="reasoning-section">
                            <div class="rs-label">Factors Used</div>
                            <div class="rs-content">${(t.factors_used || []).join(', ') || 'N/A'}</div>
                        </div>
                        ${t.indicators && Object.keys(t.indicators).length > 0 ? `
                        <div class="reasoning-section">
                            <div class="rs-label">Indicators at Entry</div>
                            <div class="rs-content" style="display:flex;flex-wrap:wrap;gap:8px;">
                                ${Object.entries(t.indicators).map(([k,v]) => {
                                    let color = '#888';
                                    if (k === 'rsi') color = v < 30 ? '#4ade80' : v > 70 ? '#f87171' : '#60a5fa';
                                    if (k === 'macd_hist') color = v > 0 ? '#4ade80' : '#f87171';
                                    if (k === 'ema_trend') color = v === 'BULL' ? '#4ade80' : '#f87171';
                                    if (k === 'adx') color = v > 25 ? '#fbbf24' : '#666';
                                    return '<span style="background:#1a1a1a;border:1px solid #333;border-radius:4px;padding:2px 8px;font-size:0.85em;">' +
                                           '<span style="color:#666">' + k + '</span> ' +
                                           '<span style="color:' + color + ';font-weight:600">' + v + '</span></span>';
                                }).join('')}
                            </div>
                        </div>` : ''}
                        ${t.raw_response && Object.keys(t.raw_response).length > 0 ? `
                        <div class="reasoning-section">
                            <div class="rs-label">Raw Response</div>
                            <pre>${JSON.stringify(t.raw_response, null, 2)}</pre>
                        </div>` : ''}
                    </div>
                </td>
            `;
            tbody.appendChild(reasoningTr);
        });
    }

    function toggleReasoning(idx) {
        const el = document.getElementById(`reasoning-${idx}`);
        if (el) {
            el.style.display = el.style.display === 'none' ? 'table-row' : 'none';
        }
    }

    // ======== Autonomous Engine ========
    let autoRunning = false;

    async function autoStart() {
        if (autoRunning) {
            // Stop
            await fetch('/api/auto/stop', { method: 'POST' });
            autoRunning = false;
            updateAutoUI({ status: 'stopped', message: 'Stopped' });
        } else {
            // Start
            await fetch('/api/auto/start', { method: 'POST' });
            autoRunning = true;
            updateAutoUI({ status: 'running', message: 'Starting...' });
        }
    }

    async function autoPause() {
        const resp = await fetch('/api/auto/pause', { method: 'POST' });
        const data = await resp.json();
        updateAutoUI({ status: data.status, message: data.status === 'paused' ? 'Paused' : 'Resumed' });
    }

    function updateAutoUI(msg) {
        const dot = document.getElementById('autoDot');
        const text = document.getElementById('autoStatusText');
        const startBtn = document.getElementById('btnAutoStart');
        const pauseBtn = document.getElementById('btnAutoPause');
        const status = msg.status || 'stopped';

        dot.className = 'auto-dot ' + status;

        let statusLine = msg.message || status;
        if (msg.current_job) statusLine = msg.current_job;
        if (msg.cycle) statusLine += ` | Cycle ${msg.cycle}`;
        if (msg.total_opts) statusLine += `, ${msg.total_opts} opts`;
        text.textContent = statusLine;

        if (status === 'running' || status === 'paused' || status === 'waiting') {
            autoRunning = true;
            startBtn.textContent = 'Stop Auto';
            startBtn.className = 'btn-auto-stop';
            pauseBtn.disabled = false;
            pauseBtn.textContent = status === 'paused' ? 'Resume' : 'Pause';
        } else {
            autoRunning = false;
            startBtn.textContent = 'Start Auto';
            startBtn.className = 'btn-auto-start';
            pauseBtn.disabled = true;
            pauseBtn.textContent = 'Pause';
        }
    }

    // ======== Auto Symbol Picker ========
    let autoAllSymbols = [];
    let autoEnabledSymbols = new Set();
    let autoSymSaveTimer = null;

    function classifySymbol(s) {
        if (s.includes('/')) return 'forex';
        if (s.endsWith('-USD')) return 'crypto';
        return 'equity';
    }

    function renderAutoSymChips() {
        const container = document.getElementById('autoSymChips');
        container.innerHTML = '';
        const groups = { equity: [], crypto: [], forex: [] };
        autoAllSymbols.forEach(s => groups[classifySymbol(s)].push(s));
        const labels = { equity: 'Equities', crypto: 'Crypto', forex: 'Forex' };
        for (const [key, syms] of Object.entries(groups)) {
            if (syms.length === 0) continue;
            const lbl = document.createElement('div');
            lbl.className = 'group-label';
            lbl.textContent = labels[key];
            container.appendChild(lbl);
            const row = document.createElement('div');
            row.className = 'auto-sym-chips';
            syms.forEach(s => {
                const chip = document.createElement('span');
                chip.className = 'auto-sym-chip' + (autoEnabledSymbols.has(s) ? ' enabled' : '');
                chip.textContent = s;
                chip.dataset.symbol = s;
                chip.onclick = () => toggleAutoSym(s);
                row.appendChild(chip);
            });
            container.appendChild(row);
        }
    }

    function toggleAutoSym(s) {
        if (autoEnabledSymbols.has(s)) {
            if (autoEnabledSymbols.size <= 1) return; // keep at least 1
            autoEnabledSymbols.delete(s);
        } else {
            autoEnabledSymbols.add(s);
        }
        renderAutoSymChips();
        scheduleAutoSymSave();
    }

    function scheduleAutoSymSave() {
        clearTimeout(autoSymSaveTimer);
        autoSymSaveTimer = setTimeout(saveAutoSymbols, 800);
    }

    async function saveAutoSymbols() {
        try {
            await fetch('/api/auto/symbols', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbols: [...autoEnabledSymbols] })
            });
        } catch(e) { console.error('Failed to save auto symbols:', e); }
    }

    function autoSymSelectAll() {
        autoEnabledSymbols = new Set(autoAllSymbols);
        renderAutoSymChips(); scheduleAutoSymSave();
    }
    function autoSymSelectNone() {
        // Keep at least one
        autoEnabledSymbols = new Set([autoAllSymbols[0]]);
        renderAutoSymChips(); scheduleAutoSymSave();
    }
    function autoSymSelectEquities() {
        autoEnabledSymbols = new Set(autoAllSymbols.filter(s => classifySymbol(s) === 'equity'));
        if (autoEnabledSymbols.size === 0) autoEnabledSymbols.add(autoAllSymbols[0]);
        renderAutoSymChips(); scheduleAutoSymSave();
    }
    function autoSymSelectCrypto() {
        autoEnabledSymbols = new Set(autoAllSymbols.filter(s => classifySymbol(s) === 'crypto'));
        if (autoEnabledSymbols.size === 0) autoEnabledSymbols.add(autoAllSymbols[0]);
        renderAutoSymChips(); scheduleAutoSymSave();
    }
    function autoSymSelectForex() {
        autoEnabledSymbols = new Set(autoAllSymbols.filter(s => classifySymbol(s) === 'forex'));
        if (autoEnabledSymbols.size === 0) autoEnabledSymbols.add(autoAllSymbols[0]);
        renderAutoSymChips(); scheduleAutoSymSave();
    }

    function addAutoResultCard(r) {
        const feed = document.getElementById('liveDecisionFeed');
        const container = document.getElementById('liveLogContainer');
        container.style.display = 'block';

        const retColor = r.return_pct >= 0 ? '#4ade80' : '#f87171';
        const card = document.createElement('div');
        card.style.cssText = `background:#0d1a0d; border:1px solid #4ade8033; border-left:3px solid #4ade80; border-radius:6px; padding:8px 12px; font-size:0.8em;`;
        let html = `<div style="display:flex;align-items:center;gap:8px;">`;
        html += `<span style="background:#4ade8022;color:#4ade80;padding:2px 6px;border-radius:3px;font-weight:700;font-size:0.85em;">AUTO</span>`;
        html += `<span style="color:#ccc;font-weight:600;">${r.symbol} ${r.interval}</span>`;
        html += `<span style="color:${retColor};font-weight:700;">${r.return_pct > 0 ? '+' : ''}${r.return_pct}%</span>`;
        html += `<span style="color:#888;">Win:${r.win_rate}% DD:${r.max_drawdown}% PF:${r.profit_factor}</span>`;
        if (r.oos_return !== null && r.oos_return !== undefined) {
            const oosColor = r.oos_return >= 0 ? '#60a5fa' : '#f87171';
            html += `<span style="color:${oosColor};">OOS:${r.oos_return > 0 ? '+' : ''}${r.oos_return}%</span>`;
        }
        if (r.overfit) html += `<span class="overfit-tag">OVERFIT?</span>`;
        if (r.sentiment_label && r.sentiment_label !== 'N/A') {
            const sentColor = r.sentiment_score > 15 ? '#4ade80' : r.sentiment_score < -15 ? '#f87171' : '#888';
            html += `<span style="color:${sentColor};font-size:0.9em;">Sent:${r.sentiment_label}</span>`;
        }
        html += `</div>`;
        card.innerHTML = html;
        feed.appendChild(card);
        if (feed.scrollHeight - feed.scrollTop - feed.clientHeight < 150) {
            feed.scrollTop = feed.scrollHeight;
        }
    }

    function renderLeaderboard(leaderboard) {
        const container = document.getElementById('leaderboardContainer');
        const tbody = document.getElementById('leaderboardBody');
        if (!leaderboard || leaderboard.length === 0) {
            container.classList.remove('active');
            return;
        }
        container.classList.add('active');
        tbody.innerHTML = '';
        leaderboard.forEach((r, i) => {
            const retColor = r.return_pct >= 0 ? '#4ade80' : '#f87171';
            const oosStr = (r.oos_return !== null && r.oos_return !== undefined)
                ? `${r.oos_return > 0 ? '+' : ''}${r.oos_return}%` : '-';
            const oosColor = (r.oos_return !== null && r.oos_return !== undefined)
                ? (r.oos_return >= 0 ? '#60a5fa' : '#f87171') : '#555';
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="color:#555;">${i + 1}</td>
                <td style="font-weight:600;">${r.symbol}</td>
                <td>${r.interval}</td>
                <td style="color:${retColor};font-weight:600;">${r.return_pct > 0 ? '+' : ''}${r.return_pct}%</td>
                <td>${r.win_rate}%</td>
                <td style="color:${r.max_drawdown > 10 ? '#f87171' : '#888'};">${r.max_drawdown}%</td>
                <td>${r.profit_factor}</td>
                <td style="color:${oosColor};">${oosStr}${r.overfit ? ' <span class="overfit-tag">!</span>' : ''}</td>
                <td>${r.test_count || 1}</td>
                <td><button class="lb-use-btn" onclick='applyAutoParams(${JSON.stringify(r).replace(/'/g,"&#39;")})'>Use</button></td>
            `;
            tbody.appendChild(tr);
        });
    }

    function applyAutoParams(r) {
        // Set symbol and interval in the form
        document.getElementById('symbol').value = r.symbol;
        // Set interval button
        document.querySelectorAll('.interval-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.val === r.interval) {
                btn.classList.add('active');
                selectedInterval = r.interval;
            }
        });
        setDatesForInterval(r.interval);
        // Apply params
        optimizedParams = r.params;
        // Select rules mode
        document.querySelectorAll('input[name="mode"]').forEach(radio => {
            radio.checked = radio.value === 'rules';
        });
        updateCostEstimate();

        // Visual feedback
        const feed = document.getElementById('liveDecisionFeed');
        if (feed) {
            const note = document.createElement('div');
            note.style.cssText = 'color:#4ade80;font-size:0.85em;margin-top:8px;padding:8px;background:#4ade8010;border:1px solid #4ade8033;border-radius:4px;';
            const p = r.params;
            note.textContent = `Applied ${r.symbol} ${r.interval}: Stop=${p.atr_stop_mult}x Tgt=${p.atr_target_mult}x Trig=${p.min_triggers} PB=${p.pullback_zone_atr} (${r.return_pct > 0 ? '+' : ''}${r.return_pct}%) — click "Run Backtest"`;
            feed.appendChild(note);
            feed.scrollTop = feed.scrollHeight;
        }
    }

    // ======== News/Sentiment ========
    let lastNewsFetch = '';
    async function fetchNewsSentiment(symbol) {
        if (!symbol) symbol = document.getElementById('symbol').value.trim().toUpperCase();
        if (!symbol || symbol === lastNewsFetch) return;
        lastNewsFetch = symbol;
        const panel = document.getElementById('newsPanel');
        const scoreEl = document.getElementById('newsScore');
        const headlinesEl = document.getElementById('newsHeadlines');
        try {
            panel.style.display = 'block';
            scoreEl.textContent = 'Loading...';
            scoreEl.className = 'news-score neutral';
            headlinesEl.innerHTML = '';
            const resp = await fetch('/api/news/' + encodeURIComponent(symbol));
            const data = await resp.json();
            // Score
            const s = data.overall_score;
            const cls = s > 15 ? 'bullish' : s < -15 ? 'bearish' : 'neutral';
            scoreEl.textContent = (s > 0 ? '+' : '') + s.toFixed(1) + ' ' + data.overall_label;
            scoreEl.className = 'news-score ' + cls;
            // Headlines
            if (data.headlines && data.headlines.length > 0) {
                data.headlines.forEach(h => {
                    const div = document.createElement('div');
                    div.className = 'news-headline';
                    const sColor = h.score > 10 ? '#4ade80' : h.score < -10 ? '#f87171' : '#888';
                    const ageStr = h.age_min < 60 ? h.age_min + 'm' : Math.round(h.age_min/60) + 'h';
                    let html = `<span class="hl-score" style="color:${sColor};">${h.score > 0 ? '+' : ''}${h.score}</span>`;
                    if (h.url) {
                        html += `<a href="${h.url}" target="_blank" rel="noopener">${h.title}</a>`;
                    } else {
                        html += `<span style="color:#bbb;">${h.title}</span>`;
                    }
                    html += ` <span class="hl-source">${h.source} ${ageStr}</span>`;
                    div.innerHTML = html;
                    headlinesEl.appendChild(div);
                });
            } else {
                headlinesEl.innerHTML = '<div style="color:#555;font-size:0.8em;">No headlines found</div>';
            }
        } catch(e) {
            scoreEl.textContent = 'Error';
            console.error('News fetch error:', e);
        }
    }

    // Auto-fetch news when symbol changes
    const symbolInput = document.getElementById('symbol');
    if (symbolInput) {
        let newsTimer = null;
        symbolInput.addEventListener('change', () => {
            lastNewsFetch = '';
            fetchNewsSentiment();
        });
        symbolInput.addEventListener('input', () => {
            clearTimeout(newsTimer);
            newsTimer = setTimeout(() => { lastNewsFetch = ''; fetchNewsSentiment(); }, 800);
        });
    }

    async function checkAutoState() {
        try {
            const resp = await fetch('/api/auto/state');
            const data = await resp.json();
            if (data.status === 'running' || data.status === 'paused') {
                autoRunning = true;
                updateAutoUI({
                    status: data.status,
                    message: data.current_job || data.status,
                    cycle: data.total_cycles,
                    total_opts: data.total_optimizations,
                });
            }
            if (data.global_best && data.global_best.length > 0) {
                renderLeaderboard(data.global_best);
            }
            // Initialize symbol picker
            if (data.all_symbols) {
                autoAllSymbols = data.all_symbols;
                autoEnabledSymbols = new Set(data.enabled_symbols || data.all_symbols);
                renderAutoSymChips();
            }
        } catch(e) { console.error('Failed to check auto state:', e); }
    }

    async function checkTelegramStatus() {
        try {
            const resp = await fetch('/api/alerts/status');
            const data = await resp.json();
            const el = document.getElementById('telegramStatus');
            if (data.configured && data.imports_ok) {
                el.innerHTML = '<span style="color:#4ade80;">Telegram alerts: configured</span>';
            } else if (data.configured) {
                el.innerHTML = '<span style="color:#fbbf24;">Telegram: missing observability module</span>';
            } else {
                el.innerHTML = '<span style="color:#555;">Telegram: not configured (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env)</span>';
            }
        } catch(e) { /* silent */ }
    }
    </script>
</body>
</html>"""


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Claude AI Backtest Dashboard")
    print("  Open http://localhost:8001 in your browser")
    print("=" * 60)

    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if api_key:
        print(f"  API Key: {api_key[:8]}...{api_key[-4:]}")
    else:
        print("  API Key: NOT SET (fallback mode only)")
        print("  Set ANTHROPIC_API_KEY env var for live API mode")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
