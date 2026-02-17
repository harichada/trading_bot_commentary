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
import time as _time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import aiohttp

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


def fetch_alpaca_snapshots(symbols: List[str]) -> dict:
    """Batch fetch latest quotes/trades via Alpaca snapshots endpoint.
    Returns {symbol: {latestTrade: {p, s, t}, dailyBar: {o,h,l,c,v}, prevDailyBar: {o,h,l,c,v}}}
    """
    cfg = _get_alpaca_config()
    if cfg is None:
        return {}

    headers = _alpaca_headers(cfg)
    results = {}

    # Batch 100 symbols per request
    for i in range(0, len(symbols), 100):
        batch = symbols[i:i+100]
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

        if i + 100 < len(symbols):
            _time.sleep(0.3)

    return results


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
    # Gap detection
    gap_threshold: float = 0.05        # minimum gap-up % to consider (5%)
    vol_ratio_max: float = 1.5         # only short when gap-day vol < this × avg vol (IEX ~noisier than SIP)
    min_avg_volume: int = 5_000         # minimum 20d avg daily volume (IEX ~2-3% of consolidated)
    min_price: float = 3.0             # minimum stock price

    # Position sizing
    initial_capital: float = 25_000
    risk_pct: float = 0.02             # risk 2% of equity per trade
    kelly_fraction: float = 0.25       # quarter Kelly
    max_positions: int = 5             # max concurrent shorts

    # Stops and targets
    stop_pct: float = 0.05             # 5% stop loss above entry (study: 5% > 3%)
    partial_target_pct: float = 0.50   # cover half when price drops to midpoint
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

    # Backtest
    backtest_years: int = 3            # default lookback for backtests


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
# SECTION 4: SCANNER
# =============================================================================

class GapScanner:
    """Pre-market gap detection and filtering."""

    def __init__(self, config: GapFadeConfig, universe: List[str] = None):
        self.config = config
        self.universe = universe or UNIVERSE
        self._avg_volumes: Dict[str, float] = {}     # 20d avg volume cache
        self._prev_closes: Dict[str, float] = {}     # previous day close cache
        self._shortable_cache: Dict[str, Tuple[bool, bool]] = {}
        self._cache_date: str = ''

    def _ensure_volume_cache(self):
        """Fetch 30d daily bars to compute 20d avg volume (cached per day)."""
        today = datetime.now(ET).strftime('%Y-%m-%d')
        if self._cache_date == today and self._avg_volumes:
            return

        logger.info(f"Building volume cache for {len(self.universe)} symbols...")
        end = datetime.now(ET)
        start = end - timedelta(days=45)
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')

        for i, sym in enumerate(self.universe):
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

    def scan_premarket(self) -> List[GapCandidate]:
        """Scan for gap-up candidates using Alpaca snapshots."""
        self._ensure_volume_cache()

        # Fetch snapshots in batch
        snapshots = fetch_alpaca_snapshots(self.universe)
        if not snapshots:
            logger.warning("No snapshot data returned")
            return []

        candidates = []
        for sym, snap in snapshots.items():
            try:
                # Get latest trade price and previous daily bar
                latest_trade = snap.get('latestTrade', {})
                prev_bar = snap.get('prevDailyBar', {})

                if not latest_trade or not prev_bar:
                    continue

                current_price = float(latest_trade.get('p', 0))
                prev_close = float(prev_bar.get('c', 0))

                if prev_close <= 0 or current_price <= 0:
                    continue

                # Compute gap
                gap_pct = (current_price - prev_close) / prev_close
                if gap_pct < self.config.gap_threshold:
                    continue

                # Price filter
                if current_price < self.config.min_price:
                    continue

                # Volume filter
                avg_vol = self._avg_volumes.get(sym, 0)
                if avg_vol < self.config.min_avg_volume:
                    continue

                # Volume ratio from daily bar (if available) or estimate
                daily_bar = snap.get('dailyBar', {})
                today_vol = float(daily_bar.get('v', 0)) if daily_bar else 0
                vol_ratio = today_vol / avg_vol if avg_vol > 0 and today_vol > 0 else 0.5

                candidates.append(GapCandidate(
                    symbol=sym,
                    gap_pct=gap_pct,
                    prev_close=prev_close,
                    premarket_price=current_price,
                    avg_vol_20d=avg_vol,
                    vol_ratio=vol_ratio,
                    shortable=True,   # will verify before entry
                    easy_to_borrow=True,
                    score=0.0,
                ))
            except Exception as e:
                logger.debug(f"Snapshot parse error for {sym}: {e}")

        # Filter by volume ratio
        candidates = [c for c in candidates if c.vol_ratio <= self.config.vol_ratio_max]

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
        logger.info(f"Scanner found {len(candidates)} gap-up candidates")
        return candidates

    def scan_historical(self, symbol: str, start_date: str, end_date: str) -> List[dict]:
        """Find historical gap-up days for backtesting."""
        df = fetch_alpaca_bars(symbol, start_date, end_date, '1Day', 'iex')
        if df is None or len(df) < 25:
            return []

        opens = df['open'].values
        closes = df['close'].values
        volumes = df['volume'].values
        dates = df.index

        avg_vol = pd.Series(volumes).rolling(20).mean().values

        gaps = []
        for i in range(1, len(df)):
            prev_c = closes[i-1]
            if prev_c <= 0 or opens[i] <= 0:
                continue
            gap = (opens[i] - prev_c) / prev_c
            if gap < self.config.gap_threshold:
                continue
            if opens[i] < self.config.min_price:
                continue

            cur_avg_vol = avg_vol[i] if not np.isnan(avg_vol[i]) else volumes[i]
            if cur_avg_vol < self.config.min_avg_volume:
                continue

            vol_ratio = volumes[i] / cur_avg_vol if cur_avg_vol > 0 else 1.0

            gaps.append({
                'date': dates[i].strftime('%Y-%m-%d') if hasattr(dates[i], 'strftime') else str(dates[i])[:10],
                'symbol': symbol,
                'open': float(opens[i]),
                'high': float(df['high'].values[i]),
                'low': float(df['low'].values[i]),
                'close': float(closes[i]),
                'prev_close': float(prev_c),
                'gap_pct': round(gap, 4),
                'volume': int(volumes[i]),
                'avg_vol': int(cur_avg_vol),
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

        # Cap at reasonable position size (max 20% of equity in one position)
        max_shares = int(self.equity * 0.20 / entry_price)
        shares = min(shares, max_shares)

        return max(0, shares)

    def should_enter(self, candidate: GapCandidate) -> Tuple[bool, str]:
        """Check if we should enter a new short. Returns (ok, reason)."""
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
        """Create a new short position."""
        stop_price = entry_price * (1 + self.config.stop_pct)
        half_target = (entry_price + candidate.prev_close) / 2
        full_target = candidate.prev_close

        shares = self.compute_position_size(entry_price, stop_price)
        if shares <= 0:
            return None

        pos = GapPosition(
            symbol=candidate.symbol,
            shares=shares,
            entry_price=entry_price,
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

        # Parse entry time for holding duration
        try:
            entry_dt = datetime.strptime(pos.entry_time, '%Y-%m-%d %H:%M')
            holding_min = int((current_time - entry_dt).total_seconds() / 60)
        except (ValueError, TypeError):
            holding_min = 0

        # 1. Stop loss — use HIGH for honest stop check
        if high >= pos.stop_price:
            exit_price = pos.stop_price  # fill at stop price
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

        # 2. Partial profit — cover 50% when price hits midpoint target
        if not pos.partial_filled and price <= pos.half_target:
            cover_shares = pos.remaining_shares // 2
            if cover_shares > 0:
                pnl = (pos.entry_price - price) * cover_shares
                pnl_pct = (pos.entry_price - price) / pos.entry_price
                trades.append(TradeRecord(
                    symbol=symbol, entry_price=pos.entry_price, exit_price=price,
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
            pnl = (pos.entry_price - price) * pos.remaining_shares
            pnl_pct = (pos.entry_price - price) / pos.entry_price
            trades.append(TradeRecord(
                symbol=symbol, entry_price=pos.entry_price, exit_price=price,
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
                pnl = (pos.entry_price - price) * pos.remaining_shares
                pnl_pct = (pos.entry_price - price) / pos.entry_price
                trades.append(TradeRecord(
                    symbol=symbol, entry_price=pos.entry_price, exit_price=price,
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
                pnl = (pos.entry_price - price) * pos.remaining_shares
                pnl_pct = (pos.entry_price - price) / pos.entry_price
                trades.append(TradeRecord(
                    symbol=symbol, entry_price=pos.entry_price, exit_price=price,
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

        # Step 1: Find gap days for each symbol using daily bars
        total_symbols = len(syms)
        await _log('info', f'Scanning {total_symbols} symbols for gap-ups >= {self.config.gap_threshold:.0%}, '
                   f'vol_ratio <= {self.config.vol_ratio_max}x, stop = {self.config.stop_pct:.0%}')

        for idx, sym in enumerate(syms):
            if self._cancel:
                break

            self.progress = (idx / total_symbols) * 30  # 0-30% for scanning
            if progress_callback:
                await progress_callback(self.progress, f"Scanning {sym} ({idx+1}/{total_symbols})...")

            gaps = scanner.scan_historical(sym, start_date, end_date)
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
            return {'error': 'No gap days found', 'total_trades': 0, 'bt_log': self.bt_log}

        # Sort by date
        all_gap_days.sort(key=lambda g: g['date'])

        use_1min = kwargs.get('use_1min', False)
        mode_label = '1-min bars (slow, detailed)' if use_1min else 'daily bars (fast)'
        await _log('info', f'Found {len(all_gap_days)} qualifying gap days '
                   f'(from {raw_gap_count} raw). Simulating with {mode_label}...')

        # Step 2: Simulate each gap day
        total_gaps = len(all_gap_days)
        trades_by_day = []
        bars_missing = 0

        for gi, gap in enumerate(all_gap_days):
            if self._cancel:
                break

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
                    _time.sleep(0.35)
                    continue
                day_trades, day_log = self._simulate_day_verbose(engine, gap, min_df)
                _time.sleep(0.35)
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

        await _log('info', f'Backtest complete: {metrics["total_trades"]} trades, '
                   f'{metrics.get("win_rate",0):.1%} win rate, '
                   f'${metrics.get("total_pnl",0):,.0f} P&L ({metrics.get("return_pct",0):.1f}%)')

        return metrics

    def _simulate_day_daily(self, engine: GapFadeEngine,
                           gap: dict) -> Tuple[List[TradeRecord], List[dict]]:
        """Fast simulation from daily OHLCV — no extra API calls.

        Uses the gap dict which already has open/high/low/close from daily bars.
        Order of checks within the bar (conservative for shorts):
          1. Stop: high >= stop_price  → loss at stop_price
          2. Partial: low <= midpoint  → cover half
          3. Full target: close <= prev_close → cover rest at prev_close
          4. Otherwise: exit at close (time_exit)
        """
        sym = gap['symbol']
        prev_close = gap['prev_close']
        day_open = gap['open']
        day_high = gap['high']
        day_low = gap['low']
        day_close = gap['close']
        day_trades = []
        log = []

        candidate = GapCandidate(
            symbol=sym, gap_pct=gap['gap_pct'], prev_close=prev_close,
            premarket_price=day_open, avg_vol_20d=gap.get('avg_vol', 0),
            vol_ratio=gap['vol_ratio'], shortable=True, easy_to_borrow=True,
        )

        # Entry at the open
        entry_price = day_open
        entry_time_str = f'{gap["date"]} 09:31'

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
        max_shares = int(engine.equity * 0.20 / entry_price) if entry_price > 0 else 0
        shares = min(shares, max_shares)
        if shares <= 0:
            log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: position size = 0'})
            return day_trades, log

        stop_dist = (stop_price - entry_price) / entry_price
        target_dist = (entry_price - full_target) / entry_price
        log.append({
            'level': 'entry',
            'msg': (f'{sym} {gap["date"]}: SHORT {shares}sh @ ${entry_price:.2f} '
                    f'| gap {gap["gap_pct"]:.1%} vol {gap["vol_ratio"]:.2f}x '
                    f'| stop ${stop_price:.2f} (+{stop_dist:.1%}) '
                    f'| half ${half_target:.2f} full ${full_target:.2f} (-{target_dist:.1%})'),
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

        if stopped and not partial_hit:
            # Pure stop-out: entire position lost at stop
            exit_price = stop_price
            pnl = (entry_price - exit_price) * remaining
            pnl_pct = (entry_price - exit_price) / entry_price
            day_trades.append(TradeRecord(
                symbol=sym, entry_price=entry_price, exit_price=exit_price,
                shares=remaining, pnl=pnl, pnl_pct=pnl_pct,
                entry_time=entry_time_str, exit_time=f'{gap["date"]} stop',
                exit_reason='stop',
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

        elif stopped and partial_hit:
            # Partial fill happened before stop (low hit target, then high hit stop)
            cover_shares = shares // 2
            if cover_shares > 0:
                pnl1 = (entry_price - half_target) * cover_shares
                pnl_pct1 = (entry_price - half_target) / entry_price
                day_trades.append(TradeRecord(
                    symbol=sym, entry_price=entry_price, exit_price=half_target,
                    shares=cover_shares, pnl=pnl1, pnl_pct=pnl_pct1,
                    entry_time=entry_time_str, exit_time=f'{gap["date"]} partial',
                    exit_reason='partial',
                ))
                engine._record_trade(day_trades[-1])
                total_pnl += pnl1
                remaining -= cover_shares
                log.append({
                    'level': 'exit',
                    'msg': (f'{sym} {gap["date"]}: PARTIAL {cover_shares}sh @ ${half_target:.2f} '
                            f'| P&L +${pnl1:.0f} ({pnl_pct1:+.1%})'),
                    'data': {'pnl': pnl1, 'reason': 'partial'}
                })

            # Remaining stopped at breakeven (stop moved to entry after partial)
            if remaining > 0:
                be_price = entry_price  # stop moved to breakeven
                pnl2 = (entry_price - be_price) * remaining
                day_trades.append(TradeRecord(
                    symbol=sym, entry_price=entry_price, exit_price=be_price,
                    shares=remaining, pnl=pnl2, pnl_pct=0.0,
                    entry_time=entry_time_str, exit_time=f'{gap["date"]} stop',
                    exit_reason='stop',
                ))
                engine._record_trade(day_trades[-1])
                total_pnl += pnl2
                be_shares = remaining
                remaining = 0
                log.append({
                    'level': 'stop',
                    'msg': (f'{sym} {gap["date"]}: STOP (BE) {be_shares}sh @ ${be_price:.2f} '
                            f'| P&L $0'),
                    'data': {'pnl': 0, 'reason': 'stop_be'}
                })

        elif partial_hit:
            # Partial fill, no stop hit
            cover_shares = shares // 2
            if cover_shares > 0:
                pnl1 = (entry_price - half_target) * cover_shares
                pnl_pct1 = (entry_price - half_target) / entry_price
                day_trades.append(TradeRecord(
                    symbol=sym, entry_price=entry_price, exit_price=half_target,
                    shares=cover_shares, pnl=pnl1, pnl_pct=pnl_pct1,
                    entry_time=entry_time_str, exit_time=f'{gap["date"]} partial',
                    exit_reason='partial',
                ))
                engine._record_trade(day_trades[-1])
                total_pnl += pnl1
                remaining -= cover_shares
                log.append({
                    'level': 'exit',
                    'msg': (f'{sym} {gap["date"]}: PARTIAL {cover_shares}sh @ ${half_target:.2f} '
                            f'| P&L +${pnl1:.0f} ({pnl_pct1:+.1%})'),
                    'data': {'pnl': pnl1, 'reason': 'partial'}
                })

            # Check if remaining hits full target
            if remaining > 0 and full_hit:
                exit_p = full_target
                pnl2 = (entry_price - exit_p) * remaining
                pnl_pct2 = (entry_price - exit_p) / entry_price
                day_trades.append(TradeRecord(
                    symbol=sym, entry_price=entry_price, exit_price=exit_p,
                    shares=remaining, pnl=pnl2, pnl_pct=pnl_pct2,
                    entry_time=entry_time_str, exit_time=f'{gap["date"]} full',
                    exit_reason='full_target',
                ))
                engine._record_trade(day_trades[-1])
                total_pnl += pnl2
                remaining = 0
                log.append({
                    'level': 'exit',
                    'msg': (f'{sym} {gap["date"]}: FULL TARGET {day_trades[-1].shares}sh @ ${exit_p:.2f} '
                            f'| P&L +${pnl2:.0f} ({pnl_pct2:+.1%})'),
                    'data': {'pnl': pnl2, 'reason': 'full_target'}
                })
            elif remaining > 0:
                # Exit at close (time exit)
                pnl2 = (entry_price - day_close) * remaining
                pnl_pct2 = (entry_price - day_close) / entry_price
                day_trades.append(TradeRecord(
                    symbol=sym, entry_price=entry_price, exit_price=day_close,
                    shares=remaining, pnl=pnl2, pnl_pct=pnl_pct2,
                    entry_time=entry_time_str, exit_time=f'{gap["date"]} close',
                    exit_reason='time_exit',
                ))
                engine._record_trade(day_trades[-1])
                total_pnl += pnl2
                remaining = 0
                log.append({
                    'level': 'exit' if pnl2 >= 0 else 'stop',
                    'msg': (f'{sym} {gap["date"]}: TIME EXIT {day_trades[-1].shares}sh @ ${day_close:.2f} '
                            f'| P&L ${pnl2:.0f} ({pnl_pct2:+.1%})'),
                    'data': {'pnl': pnl2, 'reason': 'time_exit'}
                })

        else:
            # No targets hit, no stop — exit at close
            pnl = (entry_price - day_close) * remaining
            pnl_pct = (entry_price - day_close) / entry_price
            day_trades.append(TradeRecord(
                symbol=sym, entry_price=entry_price, exit_price=day_close,
                shares=remaining, pnl=pnl, pnl_pct=pnl_pct,
                entry_time=entry_time_str, exit_time=f'{gap["date"]} close',
                exit_reason='time_exit',
            ))
            engine._record_trade(day_trades[-1])
            total_pnl += pnl
            remaining = 0
            log.append({
                'level': 'exit' if pnl >= 0 else 'stop',
                'msg': (f'{sym} {gap["date"]}: EXIT @ CLOSE ${day_close:.2f} '
                        f'| P&L ${pnl:.0f} ({pnl_pct:+.1%}) | H ${day_high:.2f} L ${day_low:.2f}'),
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

        # Summary line for this gap day
        total_pnl = sum(t.pnl for t in day_trades)
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

                # After hours — save state and sleep
                if now.hour >= 16:
                    self._save_state()
                    self.engine.reset_daily()
                    self.status = 'scanning'
                    await asyncio.sleep(3600)
                    continue

                await asyncio.sleep(30)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Trading loop error: {e}\n{traceback.format_exc()}")
            self._add_message('error', f'Trading loop error: {e}')

    async def _run_scan(self):
        """Run the pre-market scanner."""
        self._add_message('scan', 'Running pre-market scan...')
        await broadcast({'type': 'live_status', 'status': 'scanning'})

        self.candidates = self.scanner.scan_premarket()
        self.last_scan_time = datetime.now(ET).strftime('%H:%M:%S')

        self._add_message('scan', f'Found {len(self.candidates)} candidates', {
            'candidates': [asdict(c) for c in self.candidates[:10]]
        })
        await broadcast({
            'type': 'scan_results',
            'candidates': [asdict(c) for c in self.candidates[:10]],
            'scan_time': self.last_scan_time,
        })

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
            result = alpaca_place_order(candidate.symbol, pos.shares, 'sell')
            if 'error' in result:
                self._add_message('error', f'Order failed for {candidate.symbol}: {result["error"]}')
                del self.engine.positions[candidate.symbol]
                continue

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
            result = alpaca_place_order(symbol, trade.shares, 'buy')
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

# Singletons
live_trader = GapFadeLiveTrader()
backtester = GapFadeBacktester()


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the embedded HTML dashboard."""
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/api/state")
async def get_state():
    """Get current live trader state."""
    return live_trader.get_state()


@app.post("/api/scan")
async def run_scan():
    """Trigger a pre-market scan."""
    scanner = GapScanner(live_trader.config)
    candidates = scanner.scan_premarket()
    live_trader.candidates = candidates
    live_trader.last_scan_time = datetime.now(ET).strftime('%H:%M:%S')
    return {
        'candidates': [asdict(c) for c in candidates[:15]],
        'scan_time': live_trader.last_scan_time,
        'count': len(candidates),
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
    live_trader.engine.config = config
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
    for key in ['gap_threshold', 'vol_ratio_max', 'stop_pct', 'risk_pct',
                'kelly_fraction', 'max_positions', 'initial_capital',
                'daily_loss_limit', 'max_consec_losses', 'max_drawdown',
                'time_exit_hour', 'min_avg_volume', 'min_price']:
        if key in body:
            field_type = type(getattr(config, key))
            try:
                setattr(config, key, field_type(body[key]))
            except (ValueError, TypeError):
                pass

    if symbols and isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(',')]

    async def progress_cb(pct, msg):
        await broadcast({'type': 'backtest_progress', 'progress': round(pct, 1), 'message': msg})

    backtester = GapFadeBacktester(config)
    use_1min = body.get('use_1min', False)
    result = await backtester.run(
        symbol=symbol, symbols=symbols,
        start_date=start_date, end_date=end_date,
        config=config, progress_callback=progress_cb,
        use_1min=use_1min,
    )
    return result


@app.get("/api/backtest/status")
async def backtest_status():
    return {'progress': backtester.progress, 'status': backtester.status}


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
      <h2>Gap-Up Candidates <span style="color:var(--muted);font-size:11px;" id="scanTime"></span></h2>
      <table>
        <thead><tr>
          <th>Symbol</th><th>Gap %</th><th>Prev Close</th><th>Current</th><th>Vol Ratio</th>
          <th>Avg Vol</th><th>Shortable</th><th>ETB</th><th>Score</th>
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
          <input id="btSymbol" value="" placeholder="TSLA,NVDA..." style="width:160px;">
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
          <input type="number" id="btGap" value="5" step="1" style="width:60px;">
        </div>
        <div class="config-item">
          <label>Vol Max:</label>
          <input type="number" id="btVol" value="1.5" step="0.1" style="width:60px;">
        </div>
        <div class="config-item">
          <label>Stop %:</label>
          <input type="number" id="btStop" value="5" step="0.5" style="width:60px;">
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
      <div class="config-grid" id="configGrid"></div>
      <div style="margin-top:12px;">
        <button class="primary" onclick="saveConfig()">Save Config</button>
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
  } else if (msg.type === 'trade') {
    fetchState();
  } else if (msg.type === 'backtest_progress') {
    document.getElementById('btProgress').style.display = 'block';
    document.getElementById('btProgressMsg').textContent = msg.message || 'Running...';
    document.getElementById('btProgressBar').style.width = msg.progress + '%';
  } else if (msg.type === 'bt_log') {
    appendBtLog(msg.entry);
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
  const r = await api('scan', 'POST');
  renderCandidates(r.candidates || []);
  if (r.scan_time) document.getElementById('scanTime').textContent = `Last scan: ${r.scan_time}`;
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
    vol_ratio_max: parseFloat(document.getElementById('btVol').value),
    stop_pct: parseFloat(document.getElementById('btStop').value) / 100,
  };
  if (syms) body.symbols = syms;
  if (document.getElementById('bt1min').checked) body.use_1min = true;

  document.getElementById('btProgress').style.display = 'block';
  document.getElementById('btProgressMsg').textContent = 'Starting backtest...';
  document.getElementById('btProgressBar').style.width = '0%';
  document.getElementById('btResults').innerHTML = '';
  document.getElementById('btLog').innerHTML = '';
  document.getElementById('btLog').style.display = 'block';

  try {
    const r = await api('backtest', 'POST', body);
    document.getElementById('btProgress').style.display = 'none';
    renderBacktestResults(r);
  } catch(e) {
    document.getElementById('btProgressMsg').textContent = 'Error: ' + e.message;
  }
}

async function cancelBacktest() { await api('backtest/cancel', 'POST'); }

async function saveConfig() {
  const inputs = document.querySelectorAll('#configGrid input');
  const body = {};
  inputs.forEach(inp => {
    const key = inp.dataset.key;
    const val = inp.type === 'number' ? parseFloat(inp.value) : inp.value;
    body[key] = val;
  });
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
  ];
  grid.innerHTML = fields.map(([key, label, type]) => `
    <div class="config-item">
      <label>${label}:</label>
      <input type="${type}" data-key="${key}" value="${config[key] !== undefined ? config[key] : ''}" step="any">
    </div>
  `).join('');
}

function renderBacktestResults(r) {
  if (r.error) {
    document.getElementById('btResults').innerHTML = `<div style="color:var(--red);">${r.error}</div>`;
    return;
  }

  const html = `
    <div style="margin-bottom:10px;padding:8px 12px;background:rgba(168,85,247,0.08);border-radius:6px;font-size:12px;color:var(--muted);">
      Funnel: <span style="color:var(--text)">${r.raw_gaps||'?'}</span> raw gaps
      &rarr; <span style="color:var(--text)">${r.gap_days_found||0}</span> vol-filtered
      ${r.bars_missing ? `&rarr; <span style="color:var(--orange)">${r.bars_missing} no 1m data</span>` : ''}
      &rarr; <span style="color:var(--green)">${r.gap_days_traded||0}</span> traded
    </div>
    <div class="stats-grid" style="margin-bottom:12px;">
      <div class="stat"><div class="label">Trades</div><div class="value">${r.total_trades||0}</div></div>
      <div class="stat"><div class="label">Win Rate</div><div class="value">${((r.win_rate||0)*100).toFixed(1)}%</div></div>
      <div class="stat"><div class="label">Total P&L</div><div class="value ${(r.total_pnl||0)>=0?'green':'red'}">$${(r.total_pnl||0).toFixed(0)}</div></div>
      <div class="stat"><div class="label">Return</div><div class="value ${(r.return_pct||0)>=0?'green':'red'}">${(r.return_pct||0).toFixed(1)}%</div></div>
      <div class="stat"><div class="label">Profit Factor</div><div class="value">${(r.profit_factor||0).toFixed(2)}</div></div>
      <div class="stat"><div class="label">Max DD</div><div class="value red">${(r.max_drawdown_pct||0).toFixed(1)}%</div></div>
      <div class="stat"><div class="label">Gap Days</div><div class="value">${r.gap_days_found||0}</div></div>
      <div class="stat"><div class="label">Traded</div><div class="value">${r.gap_days_traded||0}</div></div>
    </div>
    ${r.trades && r.trades.length > 0 ? `
    <h2 style="margin-top:12px;">Recent Trades</h2>
    <table>
      <thead><tr><th>Time</th><th>Symbol</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Reason</th></tr></thead>
      <tbody>${r.trades.slice(-50).reverse().map(t => `<tr>
        <td>${t.exit_time||''}</td>
        <td style="font-weight:600;">${t.symbol}</td>
        <td>$${(t.entry_price||0).toFixed(2)}</td>
        <td>$${(t.exit_price||0).toFixed(2)}</td>
        <td class="${(t.pnl||0)>=0?'pnl-pos':'pnl-neg'}">${pnlFmt(t.pnl||0)}</td>
        <td>${t.exit_reason||''}</td>
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

// ── Helpers ──────────────────────────────────────────────────────
function pnlFmt(v) {
  const sign = v >= 0 ? '+' : '';
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

  connectWS();
  fetchState();
  setInterval(updateClock, 1000);
  setInterval(fetchState, 10000);
  updateClock();
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
    print()

    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
