#!/usr/bin/env python3
"""
Rudra Intraday Scalper — Standalone FastAPI Application

A high-frequency intraday trading application that manages its own positions,
state, and tick stream. Runs independently from gap_fade_app.py on port 8004.

Strategies (from gap_fade_strategies/):
  - VWAP Mean Reversion (primary)
  - Gap Continuation (proven PF 1.14)
  - Gap Bounce (proven PF 1.09)
  - Micro Scalp, Momentum Surge, etc. (disabled by default)

Sections:
  1. Imports & Config
  2. Alpaca Helpers (self-contained)
  3. Data Classes
  4. State Management
  5. Indicator & Tick Engine
  6. Strategy Manager
  7. Position Manager
  8. Trading Engine
  9. EOD Watchdog
 10. WebSocket & Broadcast
 11. FastAPI Endpoints
 12. Dashboard HTML
 13. Main
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
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone, time as dt_time
from typing import Any, Dict, List, Optional, Tuple

import requests
import aiohttp

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Strategy imports
from gap_fade_strategies import (
    IntradayStrategyRegistry,
    IntradaySetup,
    IntradayStrategy,
    TickIndicatorEngine,
    ExitSignal,
)

logger = logging.getLogger('IntradayScalper')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)

# Eastern timezone
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo('America/New_York')
except ImportError:
    import pytz
    ET = pytz.timezone('America/New_York')

# App config from env
INTRADAY_PORT = int(os.environ.get('INTRADAY_PORT', '8004'))
DATABASE_URL = os.environ.get(
    'DATABASE_URL',
    'postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev',
)
GAP_FADE_URL = os.environ.get('GAP_FADE_URL', 'http://localhost:8002')
STATE_FILE = os.path.join(os.path.dirname(__file__) or '.', 'intraday_state.json')

# Trading parameters
TOTAL_CAPITAL = 12_500.0
RISK_PER_TRADE_PCT = 0.005  # 0.5% = $62.50
MAX_POSITIONS = 5
MAX_TRADES_PER_DAY = 20
SCAN_INTERVAL_SEC = 5.0
SCAN_START_HOUR = 9
SCAN_START_MIN = 45
EOD_CLOSE_HOUR = 15
EOD_CLOSE_MIN = 50

# Default strategy enablement
DEFAULT_ENABLED = {'vwap_mean_reversion', 'gap_continuation', 'gap_bounce'}
DEFAULT_DISABLED = {
    'micro_scalp', 'momentum_surge', 'orb_breakout', 'pullback_entry',
    'connors_rsi2', 'vwap_bounce', 'catalyst_momentum', 'opening_trend',
    'first_hour_breakout', 'range_trade',
}

# Top 50 liquid symbols for watchlist
LIQUID_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "AVGO",
    # Semi & hardware
    "AMD", "INTC", "MU", "QCOM", "AMAT", "LRCX", "KLAC",
    # Software & cloud
    "CRM", "ORCL", "NFLX", "ADBE", "NOW", "UBER", "SHOP",
    # Finance
    "JPM", "BAC", "GS", "MS", "V", "MA",
    # Healthcare
    "UNH", "LLY", "ABBV", "JNJ", "PFE", "MRK",
    # Consumer
    "WMT", "COST", "HD", "MCD", "NKE", "DIS",
    # Energy
    "XOM", "CVX",
    # Industrial
    "CAT", "BA", "GE",
    # ETFs
    "SPY", "QQQ", "IWM",
]


# =============================================================================
# SECTION 2: ALPACA HELPERS (self-contained, no gap_fade_app imports)
# =============================================================================

def _load_env_file() -> None:
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
        logger.debug("Could not load .env: %s", e)


def _get_alpaca_config() -> Optional[dict]:
    """Get Alpaca paper trading config from env vars."""
    _load_env_file()
    api_key = os.environ.get('ALPACA_API_KEY', '')
    secret_key = os.environ.get('ALPACA_SECRET_KEY', '')
    if not api_key or not secret_key:
        return None
    trade_api_key = os.environ.get('ALPACA_TRADE_API_KEY', '')
    trade_secret_key = os.environ.get('ALPACA_TRADE_SECRET_KEY', '')
    return {
        'api_key': api_key,
        'secret_key': secret_key,
        'trade_api_key': trade_api_key or api_key,
        'trade_secret_key': trade_secret_key or secret_key,
        'base_url': 'https://paper-api.alpaca.markets',
        'data_url': 'https://data.alpaca.markets',
    }


def _alpaca_trade_headers(cfg: dict) -> dict:
    """Headers for ORDER/ACCOUNT requests (paper key)."""
    return {
        'APCA-API-KEY-ID': cfg['trade_api_key'],
        'APCA-API-SECRET-KEY': cfg['trade_secret_key'],
    }


def _alpaca_data_headers(cfg: dict) -> dict:
    """Headers for DATA requests (live key with SIP access)."""
    return {
        'APCA-API-KEY-ID': cfg['api_key'],
        'APCA-API-SECRET-KEY': cfg['secret_key'],
    }


@dataclass
class OrderResult:
    """Structured result from order submission + fill verification."""
    order_id: str = ''
    status: str = ''
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
        return self.status in ('error', 'rejected')


def alpaca_place_order(
    symbol: str, qty: int, side: str, order_type: str = 'market',
    limit_price: Optional[float] = None, stop_price: Optional[float] = None,
    time_in_force: str = 'day',
) -> dict:
    """Place an order on Alpaca paper account (sync)."""
    cfg = _get_alpaca_config()
    if cfg is None:
        return {'error': 'Alpaca not configured'}

    headers = {**_alpaca_trade_headers(cfg), 'Content-Type': 'application/json'}
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

    t0 = _time.monotonic()
    try:
        resp = requests.post(
            f'{cfg["base_url"]}/v2/orders',
            headers=headers, json=payload, timeout=10,
        )
        elapsed = (_time.monotonic() - t0) * 1000
        if resp.status_code in (200, 201):
            data = resp.json()
            logger.info(
                "Alpaca order placed: %s %s %s -> %s (%.0fms)",
                side, qty, symbol, data.get('id', '?'), elapsed,
            )
            return {
                'id': data.get('id'), 'status': data.get('status'),
                'symbol': symbol, 'qty': qty, 'side': side,
            }
        else:
            err = resp.text[:200]
            logger.warning("Alpaca order failed (%d): %s", resp.status_code, err)
            return {'error': f"HTTP {resp.status_code}: {err}"}
    except Exception as e:
        logger.warning("Alpaca order error: %s", e)
        return {'error': str(e)}


def alpaca_get_order(order_id: str) -> Optional[dict]:
    """Get order status from Alpaca by order ID."""
    cfg = _get_alpaca_config()
    if cfg is None:
        return None
    try:
        resp = requests.get(
            f'{cfg["base_url"]}/v2/orders/{order_id}',
            headers=_alpaca_trade_headers(cfg), timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        logger.warning("Get order %s failed: HTTP %d", order_id, resp.status_code)
    except Exception as e:
        logger.warning("Get order %s error: %s", order_id, e)
    return None


def alpaca_cancel_order(order_id: str) -> bool:
    """Cancel an open order on Alpaca. Returns True if cancelled."""
    cfg = _get_alpaca_config()
    if cfg is None:
        return False
    try:
        resp = requests.delete(
            f'{cfg["base_url"]}/v2/orders/{order_id}',
            headers=_alpaca_trade_headers(cfg), timeout=10,
        )
        return resp.status_code in (200, 204)
    except Exception as e:
        logger.warning("Cancel order %s error: %s", order_id, e)
        return False


async def alpaca_submit_and_confirm(
    symbol: str, qty: int, side: str,
    order_type: str = 'market', limit_price: Optional[float] = None,
    timeout_sec: float = 30.0, poll_interval: float = 0.5,
    cancel_on_timeout: bool = False,
) -> OrderResult:
    """Submit order and poll until filled, rejected, or timeout."""
    result = OrderResult(symbol=symbol, side=side)

    order_resp = await asyncio.to_thread(
        alpaca_place_order, symbol, qty, side, order_type, limit_price,
    )
    if 'error' in order_resp:
        result.status = 'error'
        result.error = order_resp['error']
        logger.error("Order submit failed: %s %s %s: %s", side, qty, symbol, result.error)
        return result

    order_id = order_resp.get('id', '')
    result.order_id = order_id
    logger.info(
        "Order submitted: %s %s %s (%s) id=%s%s",
        side, qty, symbol, order_type, order_id,
        f" limit=${limit_price:.2f}" if limit_price else "",
    )

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
            logger.info(
                "Order FILLED: %s %s %s @ $%.2f (id=%s)",
                side, result.filled_qty, symbol, result.filled_avg_price, order_id,
            )
            return result

        if status in ('cancelled', 'canceled', 'expired', 'suspended'):
            result.status = 'cancelled'
            result.error = f"Order {status} (by broker)"
            logger.warning("Order %s: %s %s %s (id=%s)", status, side, qty, symbol, order_id)
            return result

        if status == 'rejected':
            result.status = 'rejected'
            result.error = f"Rejected: {order_data.get('reject_reason', 'unknown')}"
            logger.warning("Order REJECTED: %s %s %s: %s", side, qty, symbol, result.error)
            return result

        if status == 'partially_filled':
            result.filled_qty = int(order_data.get('filled_qty', 0))
            result.filled_avg_price = float(order_data.get('filled_avg_price', 0))

    # Timeout — final check
    order_data = await asyncio.to_thread(alpaca_get_order, order_id)
    if order_data and order_data.get('status') == 'filled':
        result.status = 'filled'
        result.filled_qty = int(order_data.get('filled_qty', qty))
        result.filled_avg_price = float(order_data.get('filled_avg_price', 0))
        return result

    filled_qty = int(order_data.get('filled_qty', 0)) if order_data else 0
    should_cancel = cancel_on_timeout or order_type == 'market'

    if should_cancel:
        logger.warning(
            "Order TIMEOUT after %.0fs, CANCELLING: %s %s %s id=%s",
            timeout_sec, side, qty, symbol, order_id,
        )
        await asyncio.to_thread(alpaca_cancel_order, order_id)
        if filled_qty > 0:
            result.status = 'partially_filled'
            result.filled_qty = filled_qty
            result.filled_avg_price = float(order_data.get('filled_avg_price', 0))
            result.error = f"Partial fill: {filled_qty}/{qty}, rest cancelled"
        else:
            result.status = 'timeout'
            result.error = f"No fill after {timeout_sec}s, order cancelled"
    else:
        if filled_qty > 0:
            result.status = 'partially_filled'
            result.filled_qty = filled_qty
            result.filled_avg_price = float(order_data.get('filled_avg_price', 0))
        else:
            result.status = 'pending'
            result.error = f"Still pending after {timeout_sec}s"

    return result


def alpaca_place_stop_order(
    symbol: str, qty: int, stop_price: float,
    limit_offset_pct: float = 0.003, direction: str = 'short',
) -> dict:
    """Place a broker-side stop-limit order (GTC)."""
    cfg = _get_alpaca_config()
    if cfg is None:
        return {'error': 'Alpaca not configured'}

    if direction == 'long':
        side = 'sell'
        limit_price = round(stop_price * (1 - limit_offset_pct), 2)
    else:
        side = 'buy'
        limit_price = round(stop_price * (1 + limit_offset_pct), 2)

    headers = {**_alpaca_trade_headers(cfg), 'Content-Type': 'application/json'}
    payload = {
        'symbol': symbol,
        'qty': str(qty),
        'side': side,
        'type': 'stop_limit',
        'stop_price': str(round(stop_price, 2)),
        'limit_price': str(limit_price),
        'time_in_force': 'gtc',
    }

    try:
        resp = requests.post(
            f'{cfg["base_url"]}/v2/orders',
            headers=headers, json=payload, timeout=10,
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            logger.info(
                "Stop placed: %s %s %s stop=$%.2f limit=$%.2f -> id=%s",
                side, qty, symbol, stop_price, limit_price, data.get('id', '?'),
            )
            return {
                'id': data.get('id'), 'status': data.get('status'),
                'symbol': symbol, 'stop_price': stop_price,
            }
        else:
            err = resp.text[:200]
            logger.warning("Stop order failed (%d): %s", resp.status_code, err)
            return {'error': f"HTTP {resp.status_code}: {err}"}
    except Exception as e:
        logger.warning("Stop order error: %s", e)
        return {'error': str(e)}


def alpaca_cancel_open_orders_for_symbol(symbol: str) -> int:
    """Cancel all open orders for a specific symbol. Returns count cancelled."""
    cfg = _get_alpaca_config()
    if cfg is None:
        return 0
    try:
        resp = requests.get(
            f'{cfg["base_url"]}/v2/orders',
            headers=_alpaca_trade_headers(cfg),
            params={'status': 'open', 'symbols': symbol, 'limit': 50},
            timeout=10,
        )
        if resp.status_code != 200:
            return 0
        orders = resp.json()
        cancelled = 0
        for order in orders:
            oid = order.get('id', '')
            if oid:
                try:
                    r = requests.delete(
                        f'{cfg["base_url"]}/v2/orders/{oid}',
                        headers=_alpaca_trade_headers(cfg), timeout=10,
                    )
                    if r.status_code in (200, 204):
                        cancelled += 1
                except Exception:
                    pass
        return cancelled
    except Exception as e:
        logger.warning("Cancel orders for %s error: %s", symbol, e)
        return 0


def alpaca_get_positions() -> List[dict]:
    """Get current positions on Alpaca paper account."""
    cfg = _get_alpaca_config()
    if cfg is None:
        return []
    try:
        resp = requests.get(
            f'{cfg["base_url"]}/v2/positions',
            headers=_alpaca_trade_headers(cfg), timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning("Get positions error: %s", e)
    return []


def alpaca_get_account() -> Optional[dict]:
    """Get Alpaca paper account info."""
    cfg = _get_alpaca_config()
    if cfg is None:
        return None
    try:
        resp = requests.get(
            f'{cfg["base_url"]}/v2/account',
            headers=_alpaca_trade_headers(cfg), timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning("Get account error: %s", e)
    return None


def alpaca_get_latest_quotes(symbols: List[str]) -> Dict[str, dict]:
    """Get latest quotes for multiple symbols via Alpaca REST."""
    cfg = _get_alpaca_config()
    if cfg is None or not symbols:
        return {}
    try:
        resp = requests.get(
            f'{cfg["data_url"]}/v2/stocks/snapshots',
            headers=_alpaca_data_headers(cfg),
            params={'symbols': ','.join(symbols[:50])},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning("Get snapshots error: %s", e)
    return {}


# =============================================================================
# SECTION 3: DATA CLASSES
# =============================================================================

@dataclass
class IntradayPosition:
    """Active intraday position."""
    symbol: str
    direction: str                # 'long' or 'short'
    qty: int = 0
    entry_price: float = 0.0
    entry_time: str = ''          # ISO format
    current_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    stop_order_id: str = ''
    entry_order_id: str = ''
    strategy_id: str = ''
    setup_type: str = ''
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    indicators_at_entry: Dict = field(default_factory=dict)
    notes: str = ''

    def update_pnl(self, price: float) -> None:
        """Update unrealized P&L from current price."""
        self.current_price = price
        if self.direction == 'long':
            self.unrealized_pnl = (price - self.entry_price) * self.qty
            self.unrealized_pnl_pct = (
                (price - self.entry_price) / self.entry_price
                if self.entry_price > 0 else 0.0
            )
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.qty
            self.unrealized_pnl_pct = (
                (self.entry_price - price) / self.entry_price
                if self.entry_price > 0 else 0.0
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradeRecord:
    """Completed trade record."""
    symbol: str
    direction: str
    qty: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    entry_time: str = ''
    exit_time: str = ''
    pnl: float = 0.0
    pnl_pct: float = 0.0
    strategy_id: str = ''
    setup_type: str = ''
    exit_reason: str = ''
    hold_seconds: int = 0
    indicators_at_entry: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StrategyState:
    """Runtime state for each strategy."""
    strategy_id: str
    enabled: bool = True
    status: str = 'idle'          # 'idle', 'scanning', 'triggered', 'cooldown'
    trades_today: int = 0
    pnl_today: float = 0.0
    wins_today: int = 0
    losses_today: int = 0
    last_signal_time: str = ''
    cooldown_until: str = ''

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RiskState:
    """Daily risk tracking."""
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    consecutive_losses: int = 0
    trades_today: int = 0
    is_halted: bool = False
    halt_reason: str = ''
    # Limits
    daily_loss_limit: float = -250.0   # -2% of $12,500
    max_drawdown_limit: float = -375.0 # -3% of $12,500
    max_consecutive_losses: int = 5

    def check_limits(self) -> Tuple[bool, str]:
        """Check if risk limits are breached. Returns (should_halt, reason)."""
        if self.daily_pnl <= self.daily_loss_limit:
            return True, f"Daily loss limit hit: ${self.daily_pnl:.2f} <= ${self.daily_loss_limit:.2f}"
        drawdown = self.daily_pnl - self.peak_equity
        if drawdown <= self.max_drawdown_limit:
            return True, f"Max drawdown hit: ${drawdown:.2f} <= ${self.max_drawdown_limit:.2f}"
        if self.consecutive_losses >= self.max_consecutive_losses:
            return True, f"Consecutive loss limit: {self.consecutive_losses} >= {self.max_consecutive_losses}"
        return False, ''

    def record_trade(self, pnl: float) -> None:
        """Update risk state after a trade closes."""
        self.daily_pnl += pnl
        self.trades_today += 1
        if self.daily_pnl > self.peak_equity:
            self.peak_equity = self.daily_pnl
        self.max_drawdown = min(self.max_drawdown, self.daily_pnl - self.peak_equity)
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        should_halt, reason = self.check_limits()
        if should_halt:
            self.is_halted = True
            self.halt_reason = reason

    def reset_daily(self) -> None:
        """Reset for new trading day."""
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        self.consecutive_losses = 0
        self.trades_today = 0
        self.is_halted = False
        self.halt_reason = ''


# =============================================================================
# SECTION 4: STATE MANAGEMENT
# =============================================================================

class StateManager:
    """Atomic state persistence to JSON file + PostgreSQL."""

    def __init__(self, filepath: str = STATE_FILE) -> None:
        self._filepath = filepath
        self._lock = asyncio.Lock()

    async def save(self, state: dict) -> None:
        """Atomically save state to file."""
        async with self._lock:
            try:
                tmp = self._filepath + '.tmp'
                data = json.dumps(state, indent=2, default=str)
                await asyncio.to_thread(self._write_file, tmp, data)
                await asyncio.to_thread(os.replace, tmp, self._filepath)
            except Exception as e:
                logger.error("State save failed: %s", e)

    @staticmethod
    def _write_file(path: str, data: str) -> None:
        with open(path, 'w') as f:
            f.write(data)

    def load(self) -> dict:
        """Load state from file (sync, for startup)."""
        if not os.path.exists(self._filepath):
            return {}
        try:
            with open(self._filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("State load failed: %s", e)
            return {}

    async def save_to_db(self, state: dict) -> None:
        """Save state to PostgreSQL trader_state table."""
        try:
            import psycopg2
            import psycopg2.extras

            def _do_save():
                conn = psycopg2.connect(DATABASE_URL)
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO trader_state (key, state_json, updated_at)
                            VALUES ('intraday', %s, NOW())
                            ON CONFLICT (key) DO UPDATE
                            SET state_json = EXCLUDED.state_json,
                                updated_at = EXCLUDED.updated_at
                        """, (json.dumps(state, default=str),))
                    conn.commit()
                finally:
                    conn.close()

            await asyncio.to_thread(_do_save)
        except Exception as e:
            logger.debug("DB state save failed (non-critical): %s", e)


# =============================================================================
# SECTION 5: TICK STREAMER
# =============================================================================

class IntradayTickStreamer:
    """Tick streamer that proxies through gap_fade_app's WebSocket.

    Instead of opening a separate Alpaca WebSocket (which hits the
    1-connection-per-key limit), this connects to gap_fade_app's
    /ws endpoint on localhost and receives tick data from there.

    Falls back to REST polling if the gap fade WS is unavailable.
    """

    GAP_FADE_WS = 'ws://localhost:{port}/ws'
    GAP_FADE_REST = 'http://localhost:{port}/api/positions'
    THROTTLE_SEC = 0.25

    def __init__(
        self, symbols: List[str],
        on_tick: Optional[Any] = None,
        gap_fade_port: int = None,
    ) -> None:
        self.symbols = list(symbols)
        self.on_tick = on_tick
        self.latest_prices: Dict[str, float] = {}
        self.connected = False
        self._task: Optional[asyncio.Task] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_push: Dict[str, float] = {}
        self._ws = None
        self._tick_count = 0
        self._last_tick_time = 0.0
        self._gf_port = gap_fade_port or int(os.environ.get('GAP_FADE_PORT', '8003'))

    async def start(self) -> None:
        """Start: connect to gap_fade_app WS for shared tick data."""
        self._task = asyncio.create_task(self._run())
        self._poll_task = asyncio.create_task(self._rest_fallback())

    async def stop(self) -> None:
        """Stop the WebSocket connection."""
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

    async def add_symbols(self, symbols: List[str]) -> None:
        """Dynamically subscribe to additional symbols."""
        new_syms = [s for s in symbols if s not in self.symbols]
        if not new_syms:
            return
        self.symbols.extend(new_syms)
        if self._ws and not self._ws.closed:
            await self._ws.send_json({'action': 'subscribe', 'trades': new_syms})
            logger.info("Intraday stream: subscribed to %s (%d total)", new_syms, len(self.symbols))

    async def remove_symbols(self, symbols: List[str]) -> None:
        """Dynamically unsubscribe from symbols."""
        rm_syms = [s for s in symbols if s in self.symbols]
        if not rm_syms:
            return
        self.symbols = [s for s in self.symbols if s not in rm_syms]
        if self._ws and not self._ws.closed:
            await self._ws.send_json({'action': 'unsubscribe', 'trades': rm_syms})
            logger.info("Intraday stream: unsubscribed from %s", rm_syms)

    async def _rest_fallback(self) -> None:
        """Fallback: poll gap_fade_app REST API for prices every 5s."""
        url = f'http://localhost:{self._gf_port}/api/health'
        while True:
            try:
                if not self.connected:
                    async with aiohttp.ClientSession() as session:
                        # Get snapshot prices from Alpaca REST as fallback
                        cfg = _get_alpaca_config()
                        if cfg and self.symbols:
                            headers = {'APCA-API-KEY-ID': cfg['api_key'],
                                       'APCA-API-SECRET-KEY': cfg['secret_key']}
                            batch = ','.join(self.symbols[:50])
                            async with session.get(
                                f'{cfg["data_url"]}/v2/stocks/snapshots',
                                headers=headers,
                                params={'symbols': batch, 'feed': 'sip'},
                                timeout=aiohttp.ClientTimeout(total=10)
                            ) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    for sym, snap in data.items():
                                        lt = snap.get('latestTrade', {})
                                        p = lt.get('p', 0)
                                        if p and p > 0:
                                            self.latest_prices[sym] = float(p)
                                            if self.on_tick and sym in self.symbols:
                                                now = _time.monotonic()
                                                last = self._last_push.get(sym, 0)
                                                if now - last >= self.THROTTLE_SEC:
                                                    self._last_push[sym] = now
                                                    self.on_tick(sym, float(p))
                            self._tick_count += 1
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.debug("REST fallback error: %s", e)
            await asyncio.sleep(5)

    async def _run(self) -> None:
        """Primary data loop: poll Alpaca REST snapshots every 3 seconds.

        Uses the LIVE key for SIP snapshot data (no WebSocket needed).
        This avoids the 1-connection-per-key limit entirely.
        """
        cfg = _get_alpaca_config()
        if cfg is None:
            logger.warning("IntradayTickStreamer: no Alpaca config")
            return

        headers = {
            'APCA-API-KEY-ID': cfg['api_key'],
            'APCA-API-SECRET-KEY': cfg['secret_key'],
        }
        data_url = cfg['data_url']
        poll_interval = 3.0  # seconds between polls

        logger.info("Intraday stream: starting REST snapshot polling (%d symbols, %.0fs interval)",
                     len(self.symbols), poll_interval)
        self.connected = True
        last_health_log = _time.monotonic()

        while True:
            try:
                if not self.symbols:
                    await asyncio.sleep(poll_interval)
                    continue

                async with aiohttp.ClientSession() as session:
                    batch = ','.join(self.symbols[:50])
                    async with session.get(
                        f'{data_url}/v2/stocks/snapshots',
                        headers=headers,
                        params={'symbols': batch, 'feed': 'sip'},
                        timeout=aiohttp.ClientTimeout(total=8),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            for sym, snap in data.items():
                                lt = snap.get('latestTrade', {})
                                p = lt.get('p', 0)
                                if p and p > 0:
                                    old_price = self.latest_prices.get(sym, 0)
                                    self.latest_prices[sym] = float(p)
                                    self._last_tick_time = _time.monotonic()
                                    self._tick_count += 1

                                    if self.on_tick and sym in self.symbols:
                                        now = _time.monotonic()
                                        last = self._last_push.get(sym, 0)
                                        if now - last >= self.THROTTLE_SEC:
                                            self._last_push[sym] = now
                                            try:
                                                if asyncio.iscoroutinefunction(self.on_tick):
                                                    await self.on_tick(sym, float(p), 0)
                                                else:
                                                    self.on_tick(sym, float(p), 0)
                                            except Exception as e:
                                                logger.warning("on_tick error for %s: %s", sym, e)

                # Health log every 60s
                now_mono = _time.monotonic()
                if now_mono - last_health_log >= 60:
                    logger.info("Intraday stream: %d ticks/min (%d symbols, REST mode)",
                               self._tick_count, len(self.symbols))
                    self._tick_count = 0
                    last_health_log = now_mono

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("Intraday stream poll error: %s", e)

            await asyncio.sleep(poll_interval)


# =============================================================================
# SECTION 6: STRATEGY MANAGER
# =============================================================================

class StrategyManager:
    """Manages all intraday strategy instances and their state."""

    def __init__(self) -> None:
        self._strategies: Dict[str, IntradayStrategy] = {}
        self._states: Dict[str, StrategyState] = {}
        self._indicator_engine: Optional[TickIndicatorEngine] = None

    def initialize(self) -> None:
        """Create strategy instances and build unified indicator engine."""
        all_ids = IntradayStrategyRegistry.get_all_ids()
        all_indicators = set()
        all_ema_periods = set()

        for sid in all_ids:
            try:
                strategy = IntradayStrategyRegistry.create_strategy(sid)
                self._strategies[sid] = strategy
                enabled = sid in DEFAULT_ENABLED
                self._states[sid] = StrategyState(
                    strategy_id=sid,
                    enabled=enabled,
                )
                # Collect required indicators
                if enabled:
                    all_indicators.update(strategy.get_required_indicators())
                    all_ema_periods.update(strategy.get_ema_periods())
                logger.info(
                    "Strategy loaded: %s (%s) %s",
                    sid, strategy.name,
                    'ENABLED' if enabled else 'disabled',
                )
            except Exception as e:
                logger.warning("Failed to load strategy %s: %s", sid, e)

        # Build unified indicator engine with all required indicators
        if not all_indicators:
            all_indicators = {'vwap', 'rsi', 'atr', 'ema_multi', 'volume_profile',
                              'day_high', 'day_low', 'bar_history', 'opening_range'}
        if not all_ema_periods:
            all_ema_periods = {9, 20}

        self._indicator_engine = TickIndicatorEngine(
            indicators=list(all_indicators),
            ema_periods=sorted(all_ema_periods),
            rsi_period=5,    # fast RSI for intraday
            atr_period=14,
            or_minutes=5,
        )
        logger.info(
            "Indicator engine initialized: indicators=%s, ema_periods=%s",
            sorted(all_indicators), sorted(all_ema_periods),
        )

    @property
    def indicator_engine(self) -> Optional[TickIndicatorEngine]:
        return self._indicator_engine

    def get_enabled_strategies(self) -> Dict[str, IntradayStrategy]:
        """Return dict of enabled strategies."""
        return {
            sid: strat for sid, strat in self._strategies.items()
            if self._states.get(sid, StrategyState(strategy_id=sid)).enabled
        }

    def get_strategy(self, strategy_id: str) -> Optional[IntradayStrategy]:
        return self._strategies.get(strategy_id)

    def get_state(self, strategy_id: str) -> Optional[StrategyState]:
        return self._states.get(strategy_id)

    def get_all_states(self) -> Dict[str, StrategyState]:
        return dict(self._states)

    def enable_strategy(self, strategy_id: str) -> bool:
        """Enable a strategy. Returns True if found."""
        state = self._states.get(strategy_id)
        if state is None:
            return False
        state.enabled = True
        # Add indicators for newly enabled strategy
        strat = self._strategies.get(strategy_id)
        if strat and self._indicator_engine:
            for ind in strat.get_required_indicators():
                self._indicator_engine._indicators.add(ind)
        logger.info("Strategy enabled: %s", strategy_id)
        return True

    def disable_strategy(self, strategy_id: str) -> bool:
        """Disable a strategy. Returns True if found."""
        state = self._states.get(strategy_id)
        if state is None:
            return False
        state.enabled = False
        logger.info("Strategy disabled: %s", strategy_id)
        return True

    def reset_daily(self) -> None:
        """Reset all strategy states for a new trading day."""
        for state in self._states.values():
            state.trades_today = 0
            state.pnl_today = 0.0
            state.wins_today = 0
            state.losses_today = 0
            state.status = 'idle'
            state.last_signal_time = ''
            state.cooldown_until = ''
        for strategy in self._strategies.values():
            try:
                strategy.on_day_start()
            except Exception as e:
                logger.warning("Strategy day_start error for %s: %s", strategy.strategy_id, e)
        if self._indicator_engine:
            self._indicator_engine.reset()

    def record_trade(self, strategy_id: str, pnl: float) -> None:
        """Update strategy state after trade completion."""
        state = self._states.get(strategy_id)
        if state is None:
            return
        state.trades_today += 1
        state.pnl_today += pnl
        if pnl >= 0:
            state.wins_today += 1
        else:
            state.losses_today += 1


# =============================================================================
# SECTION 7: AGENT LOG
# =============================================================================

class AgentLog:
    """Ring buffer for real-time strategy decision logging."""

    def __init__(self, maxlen: int = 500) -> None:
        self._entries: deque = deque(maxlen=maxlen)

    def log(self, message: str, level: str = 'info') -> None:
        """Add a log entry."""
        now = datetime.now(ET)
        entry = {
            'time': now.strftime('%H:%M:%S'),
            'timestamp': now.isoformat(),
            'message': message,
            'level': level,
        }
        self._entries.append(entry)
        if level == 'error':
            logger.error("[AgentLog] %s", message)
        elif level == 'warning':
            logger.warning("[AgentLog] %s", message)
        else:
            logger.info("[AgentLog] %s", message)

    def get_recent(self, n: int = 100) -> List[dict]:
        """Get the last N entries."""
        entries = list(self._entries)
        return entries[-n:]

    def clear(self) -> None:
        self._entries.clear()


# =============================================================================
# SECTION 8: TRADING ENGINE
# =============================================================================

class IntradayTradingEngine:
    """Core trading engine that coordinates strategies, positions, and execution."""

    def __init__(self) -> None:
        # Components
        self.strategy_mgr = StrategyManager()
        self.state_mgr = StateManager()
        self.agent_log = AgentLog()
        self.risk = RiskState()

        # Positions & trades
        self.positions: Dict[str, IntradayPosition] = {}
        self.trades: List[TradeRecord] = []
        self.pending_orders: Dict[str, str] = {}  # symbol -> order_id

        # Tick stream
        self.streamer: Optional[IntradayTickStreamer] = None

        # Control
        self.is_trading = False
        self.is_running = False
        self._scan_task: Optional[asyncio.Task] = None
        self._eod_task: Optional[asyncio.Task] = None
        self._state_save_task: Optional[asyncio.Task] = None
        self._startup_time: Optional[datetime] = None
        self._last_scan_time: Optional[datetime] = None
        self._trading_date: str = ''

        # WebSocket clients
        self.ws_clients: List[WebSocket] = []

        # Gap fade position cache (cross-system check)
        self._gap_fade_symbols: set = set()
        self._gap_fade_check_time = 0.0

        # Phase 3: Agent avatar statuses (war room)
        self.agent_statuses: Dict[str, str] = {
            'scout': 'idle',
            'analyst': 'idle',
            'executor': 'idle',
            'risk': 'ok',
        }

        # Phase 3: Manual order share size
        self.manual_share_size: int = 100

        # Phase 3: Sound muted flag
        self.sound_muted: bool = False

        # Phase 3: Last manual order ID (for cancel-last)
        self._last_order_id: str = ''

        # Phase 3: Volume-at-price from ticks (for DOM)
        self._volume_at_price: Dict[str, Dict[float, int]] = defaultdict(lambda: defaultdict(int))

    # ── Initialization ────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize all components."""
        self._startup_time = datetime.now(ET)
        self.strategy_mgr.initialize()

        # Load saved state
        saved = self.state_mgr.load()
        if saved:
            self._restore_state(saved)
            self.agent_log.log("State restored from disk")

        # Start tick streamer
        self.streamer = IntradayTickStreamer(
            symbols=LIQUID_UNIVERSE[:50],
            on_tick=self._on_tick,
        )
        await self.streamer.start()

        # Check for orphaned positions on Alpaca
        await self._reconcile_positions()

        self.agent_log.log(
            f"Engine initialized. {len(self.strategy_mgr.get_enabled_strategies())} "
            f"strategies enabled, {len(LIQUID_UNIVERSE)} symbols tracked"
        )

    def _restore_state(self, saved: dict) -> None:
        """Restore state from saved dict."""
        today = datetime.now(ET).strftime('%Y-%m-%d')

        # Only restore positions and trades from today
        saved_date = saved.get('trading_date', '')
        if saved_date != today:
            logger.info("State is from %s, today is %s — starting fresh", saved_date, today)
            return

        # Restore positions
        for pos_data in saved.get('positions', []):
            try:
                pos = IntradayPosition(**{
                    k: v for k, v in pos_data.items()
                    if k in IntradayPosition.__dataclass_fields__
                })
                self.positions[pos.symbol] = pos
            except Exception as e:
                logger.warning("Failed to restore position: %s", e)

        # Restore trade records
        for trade_data in saved.get('trades', []):
            try:
                trade = TradeRecord(**{
                    k: v for k, v in trade_data.items()
                    if k in TradeRecord.__dataclass_fields__
                })
                self.trades.append(trade)
            except Exception as e:
                logger.warning("Failed to restore trade: %s", e)

        # Restore risk state
        risk_data = saved.get('risk', {})
        for k, v in risk_data.items():
            if hasattr(self.risk, k):
                setattr(self.risk, k, v)

        # Restore strategy enablement
        for sid, enabled in saved.get('strategy_enabled', {}).items():
            if enabled:
                self.strategy_mgr.enable_strategy(sid)
            else:
                self.strategy_mgr.disable_strategy(sid)

        logger.info(
            "Restored: %d positions, %d trades, daily P&L $%.2f",
            len(self.positions), len(self.trades), self.risk.daily_pnl,
        )

    async def _reconcile_positions(self) -> None:
        """Check Alpaca for positions that match our tracked ones."""
        try:
            broker_positions = await asyncio.to_thread(alpaca_get_positions)
            broker_symbols = {p['symbol'] for p in broker_positions}

            # Check for orphaned positions (on broker but not tracked)
            our_symbols = set(self.positions.keys())
            orphaned = broker_symbols - our_symbols
            if orphaned:
                self.agent_log.log(
                    f"WARNING: Found {len(orphaned)} untracked positions on broker: "
                    f"{', '.join(sorted(orphaned))}",
                    level='warning',
                )

            # Check for ghost positions (tracked but not on broker)
            ghosts = our_symbols - broker_symbols
            for symbol in ghosts:
                self.agent_log.log(
                    f"Ghost position removed: {symbol} (not on broker)",
                    level='warning',
                )
                del self.positions[symbol]

        except Exception as e:
            logger.warning("Reconciliation error: %s", e)

    # ── Tick Processing ───────────────────────────────────────────────

    async def _on_tick(self, symbol: str, price: float, size: int) -> None:
        """Process an incoming tick from the streamer."""
        now = datetime.now(ET)

        # Update indicator engine
        engine = self.strategy_mgr.indicator_engine
        if engine:
            engine.on_tick(symbol, price, size, now)

        # Phase 3: Accumulate volume at price for DOM ladder
        rounded_price = round(price, 2)
        self._volume_at_price[symbol][rounded_price] += size

        # Update position P&L if we hold this symbol
        if symbol in self.positions:
            self.positions[symbol].update_pnl(price)

            # Check strategy exit signals
            pos = self.positions[symbol]
            strategy = self.strategy_mgr.get_strategy(pos.strategy_id)
            if strategy and engine:
                tick_data = engine.get_data(symbol)
                exit_signal = strategy.evaluate_exit(
                    pos.to_dict(), price, tick_data, now,
                )
                if exit_signal and exit_signal.action == 'close':
                    self.agent_log.log(
                        f"EXIT SIGNAL {symbol}: {exit_signal.reason}",
                    )
                    await self._close_position(symbol, exit_signal.reason)
                    return

                # Check trailing stop update
                new_stop = strategy.update_trailing_stop(
                    pos.to_dict(), price, tick_data, now,
                )
                if new_stop is not None and new_stop != pos.stop_price:
                    old_stop = pos.stop_price
                    pos.stop_price = new_stop
                    self.agent_log.log(
                        f"STOP TIGHTENED {symbol}: ${old_stop:.2f} -> ${new_stop:.2f}"
                    )
                    # Update broker stop order
                    if pos.stop_order_id:
                        await self._update_broker_stop(pos)

        # Broadcast price to WS clients (throttled by streamer)
        await self._broadcast({
            'type': 'tick',
            'symbol': symbol,
            'price': round(price, 2),
            'time': now.strftime('%H:%M:%S'),
        })

    # ── Scanning Loop ─────────────────────────────────────────────────

    async def start_trading(self) -> None:
        """Start the scanning/trading loop."""
        if self.is_trading:
            return
        self.is_trading = True
        self.is_running = True
        self._trading_date = datetime.now(ET).strftime('%Y-%m-%d')

        # Reset for new day if needed
        today = datetime.now(ET).strftime('%Y-%m-%d')
        if self._trading_date != today:
            self._new_day_reset()
            self._trading_date = today

        self._scan_task = asyncio.create_task(self._scan_loop())
        self._eod_task = asyncio.create_task(self._eod_watchdog())
        self._state_save_task = asyncio.create_task(self._periodic_state_save())

        self.agent_log.log("Trading STARTED")
        await self._broadcast({'type': 'status', 'trading': True})

    async def stop_trading(self) -> None:
        """Stop the scanning loop (does not close positions)."""
        self.is_trading = False
        if self._scan_task and not self._scan_task.done():
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass

        self.agent_log.log("Trading STOPPED")
        await self._broadcast({'type': 'status', 'trading': False})

    def _new_day_reset(self) -> None:
        """Reset all daily state."""
        self.risk.reset_daily()
        self.trades.clear()
        self.strategy_mgr.reset_daily()
        self.agent_log.log("New trading day: all daily state reset")

    async def _scan_loop(self) -> None:
        """Main scanning loop — runs every SCAN_INTERVAL_SEC."""
        self.agent_log.log(
            f"Scan loop started. Interval={SCAN_INTERVAL_SEC}s, "
            f"max_positions={MAX_POSITIONS}, max_trades={MAX_TRADES_PER_DAY}"
        )

        while self.is_trading:
            try:
                now = datetime.now(ET)
                self._last_scan_time = now

                # Check if within trading hours
                if not self._is_scan_time(now):
                    await asyncio.sleep(SCAN_INTERVAL_SEC)
                    continue

                # Check risk halt
                if self.risk.is_halted:
                    await asyncio.sleep(SCAN_INTERVAL_SEC)
                    continue

                # Check daily trade limit
                if self.risk.trades_today >= MAX_TRADES_PER_DAY:
                    await asyncio.sleep(SCAN_INTERVAL_SEC)
                    continue

                # Check max positions
                if len(self.positions) >= MAX_POSITIONS:
                    await asyncio.sleep(SCAN_INTERVAL_SEC)
                    continue

                # Update gap fade positions (cross-system check)
                await self._update_gap_fade_positions()

                # Phase 3: Update agent statuses for war room
                await self._set_agent_status('scout', 'scanning')

                # Scan all enabled strategies
                await self._run_scan_cycle(now)

                # Phase 3: Revert scout status after scan
                await self._set_agent_status('scout', 'idle')

                await asyncio.sleep(SCAN_INTERVAL_SEC)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Scan loop error: %s\n%s", e, traceback.format_exc())
                self.agent_log.log(f"Scan error: {e}", level='error')
                await asyncio.sleep(SCAN_INTERVAL_SEC)

    def _is_scan_time(self, now: datetime) -> bool:
        """Check if current time is within scanning window."""
        t = now.time()
        start = dt_time(SCAN_START_HOUR, SCAN_START_MIN)
        end = dt_time(EOD_CLOSE_HOUR, EOD_CLOSE_MIN)
        return start <= t <= end

    async def _update_gap_fade_positions(self) -> None:
        """Query gap_fade_app to get its current positions (avoid conflicts)."""
        now = _time.monotonic()
        if now - self._gap_fade_check_time < 30.0:
            return
        self._gap_fade_check_time = now

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'{GAP_FADE_URL}/api/health', timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        positions = data.get('positions', [])
                        self._gap_fade_symbols = {
                            p.get('symbol', '') for p in positions if p.get('symbol')
                        }
        except Exception:
            pass  # Non-critical — gap fade may not be running

    async def _run_scan_cycle(self, now: datetime) -> None:
        """Run one scan cycle across all enabled strategies and symbols."""
        engine = self.strategy_mgr.indicator_engine
        if engine is None:
            return

        enabled = self.strategy_mgr.get_enabled_strategies()
        if not enabled:
            return

        # Get symbols with live data
        streamer = self.streamer
        if streamer is None:
            return

        symbols_with_data = [
            sym for sym in streamer.latest_prices
            if sym not in self.positions  # skip symbols we already hold
            and sym not in self._gap_fade_symbols  # skip gap fade symbols
            and sym not in self.pending_orders  # skip pending entries
        ]

        setups_found = []

        for strategy_id, strategy in enabled.items():
            state = self.strategy_mgr.get_state(strategy_id)
            if state:
                state.status = 'scanning'

            # Check strategy active window
            h1, m1, h2, m2 = strategy.get_active_window()
            if now.hour < h1 or (now.hour == h1 and now.minute < m1):
                if state:
                    state.status = 'idle'
                continue
            if now.hour > h2 or (now.hour == h2 and now.minute > m2):
                if state:
                    state.status = 'idle'
                continue

            # Check cooldown
            if state and state.cooldown_until:
                try:
                    cooldown_end = datetime.fromisoformat(state.cooldown_until)
                    if now < cooldown_end:
                        state.status = 'cooldown'
                        continue
                    else:
                        state.cooldown_until = ''
                except (ValueError, TypeError):
                    state.cooldown_until = ''

            for symbol in symbols_with_data:
                try:
                    tick_data = engine.get_data(symbol)
                    if not tick_data:
                        continue

                    snapshot = {
                        'price': streamer.latest_prices.get(symbol, 0),
                    }

                    setup = strategy.scan_for_setups(symbol, tick_data, snapshot, now)
                    if setup is None:
                        continue

                    # Validate setup
                    valid, reason = strategy.validate_setup(setup, tick_data, now)
                    if not valid:
                        continue

                    setups_found.append(setup)

                except Exception as e:
                    logger.debug(
                        "Scan error %s/%s: %s", strategy_id, symbol, e,
                    )

            if state:
                state.status = 'idle'

        # Process best setups (limit to remaining position slots)
        if setups_found:
            # Sort by confidence (highest first)
            setups_found.sort(key=lambda s: s.confidence, reverse=True)
            slots_available = MAX_POSITIONS - len(self.positions)

            for setup in setups_found[:slots_available]:
                if self.risk.trades_today >= MAX_TRADES_PER_DAY:
                    break
                if setup.symbol in self.positions:
                    continue  # another strategy already filled this symbol

                state = self.strategy_mgr.get_state(setup.strategy_id)
                if state:
                    state.status = 'triggered'
                    state.last_signal_time = now.isoformat()

                self.agent_log.log(
                    f"{setup.strategy_id.upper()}: {setup.symbol} "
                    f"{setup.direction.upper()} setup detected. "
                    f"Entry ~${setup.entry_price:.2f}, "
                    f"Stop ${setup.stop_price:.2f}, "
                    f"Target ${setup.target_price:.2f}, "
                    f"R:R {setup.risk_reward:.1f}, "
                    f"Conf {setup.confidence:.0%}. {setup.notes}"
                )

                await self._enter_position(setup)

                if state:
                    state.status = 'idle'

    # ── Position Entry ────────────────────────────────────────────────

    async def _enter_position(self, setup: IntradaySetup) -> None:
        """Enter a position based on a strategy setup."""
        symbol = setup.symbol
        now = datetime.now(ET)

        # Phase 3: Agent statuses
        await self._set_agent_status('analyst', 'evaluating')

        # Position sizing: risk-based
        risk_amount = TOTAL_CAPITAL * RISK_PER_TRADE_PCT
        risk_per_share = abs(setup.entry_price - setup.stop_price)
        if risk_per_share <= 0:
            self.agent_log.log(
                f"SKIP {symbol}: zero risk per share", level='warning',
            )
            return

        qty = int(risk_amount / risk_per_share)
        if qty <= 0:
            self.agent_log.log(
                f"SKIP {symbol}: calculated qty=0 (risk_per_share=${risk_per_share:.2f})",
                level='warning',
            )
            return

        # Cap position size to 20% of capital
        max_position_value = TOTAL_CAPITAL * 0.20
        if qty * setup.entry_price > max_position_value:
            qty = int(max_position_value / setup.entry_price)
            if qty <= 0:
                return

        # Determine order side
        if setup.direction == 'long':
            side = 'buy'
        else:
            side = 'sell'

        await self._set_agent_status('analyst', 'approved')
        await self._set_agent_status('executor', 'executing')

        self.agent_log.log(
            f"ENTERING {symbol}: {side.upper()} {qty} shares @ ~${setup.entry_price:.2f} "
            f"(strategy={setup.strategy_id})"
        )

        # Submit market order
        result = await alpaca_submit_and_confirm(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type='market',
            timeout_sec=15.0,
            cancel_on_timeout=True,
        )

        if not result.is_filled:
            await self._set_agent_status('executor', 'idle')
            await self._set_agent_status('analyst', 'idle')
            self.agent_log.log(
                f"ENTRY FAILED {symbol}: {result.status} — {result.error}",
                level='warning',
            )
            return

        # Create position
        position = IntradayPosition(
            symbol=symbol,
            direction=setup.direction,
            qty=result.filled_qty,
            entry_price=result.filled_avg_price,
            entry_time=now.isoformat(),
            current_price=result.filled_avg_price,
            stop_price=setup.stop_price,
            target_price=setup.target_price,
            entry_order_id=result.order_id,
            strategy_id=setup.strategy_id,
            setup_type=setup.setup_type,
            indicators_at_entry=setup.indicators,
            notes=setup.notes,
        )

        # Place broker-side stop order
        stop_result = await asyncio.to_thread(
            alpaca_place_stop_order,
            symbol, result.filled_qty, setup.stop_price,
            direction=setup.direction,
        )
        if 'id' in stop_result:
            position.stop_order_id = stop_result['id']

        self.positions[symbol] = position

        # Update strategy tracking
        strategy = self.strategy_mgr.get_strategy(setup.strategy_id)
        if strategy and hasattr(strategy, 'record_entry'):
            strategy.record_entry(symbol, now)

        await self._set_agent_status('executor', 'idle')
        await self._set_agent_status('analyst', 'idle')

        self.agent_log.log(
            f"FILLED {symbol}: {setup.direction.upper()} {result.filled_qty} "
            f"@ ${result.filled_avg_price:.2f} "
            f"(stop=${setup.stop_price:.2f}, target=${setup.target_price:.2f})"
        )

        # Phase 3: Sound alert for entry fill
        await self._broadcast({'type': 'sound', 'sound': 'entry'})

        await self._save_state()
        await self._broadcast({
            'type': 'position_open',
            'position': position.to_dict(),
        })

        # Broadcast entry marker for charting
        marker_color = '#22c55e' if setup.direction == 'long' else '#ef4444'
        marker_shape = 'arrowUp' if setup.direction == 'long' else 'arrowDown'
        marker_pos = 'belowBar' if setup.direction == 'long' else 'aboveBar'
        marker_text = f'{setup.direction.upper()} ${result.filled_avg_price:.2f}'
        now_epoch = int(datetime.now(ET).timestamp())
        # Round to nearest minute for chart alignment
        now_epoch = (now_epoch // 60) * 60
        await self._broadcast({
            'type': 'marker',
            'symbol': symbol,
            'time': now_epoch,
            'position': marker_pos,
            'color': marker_color,
            'shape': marker_shape,
            'text': marker_text,
        })

    # ── Position Exit ─────────────────────────────────────────────────

    async def _close_position(
        self, symbol: str, reason: str = 'manual',
    ) -> Optional[TradeRecord]:
        """Close a position and record the trade."""
        pos = self.positions.get(symbol)
        if pos is None:
            return None

        now = datetime.now(ET)

        # Cancel existing stop order
        if pos.stop_order_id:
            await asyncio.to_thread(alpaca_cancel_order, pos.stop_order_id)

        # Cancel any open orders for this symbol
        await asyncio.to_thread(alpaca_cancel_open_orders_for_symbol, symbol)

        # Submit close order
        if pos.direction == 'long':
            close_side = 'sell'
        else:
            close_side = 'buy'

        self.agent_log.log(
            f"CLOSING {symbol}: {close_side.upper()} {pos.qty} shares "
            f"(reason={reason})"
        )

        result = await alpaca_submit_and_confirm(
            symbol=symbol,
            qty=pos.qty,
            side=close_side,
            order_type='market',
            timeout_sec=15.0,
            cancel_on_timeout=True,
        )

        exit_price = result.filled_avg_price if result.is_filled else pos.current_price

        # Calculate P&L
        if pos.direction == 'long':
            pnl = (exit_price - pos.entry_price) * pos.qty
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0
        else:
            pnl = (pos.entry_price - exit_price) * pos.qty
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price if pos.entry_price > 0 else 0

        # Calculate hold time
        try:
            entry_dt = datetime.fromisoformat(pos.entry_time)
            hold_seconds = int((now - entry_dt).total_seconds())
        except (ValueError, TypeError):
            hold_seconds = 0

        # Create trade record
        trade = TradeRecord(
            symbol=symbol,
            direction=pos.direction,
            qty=pos.qty,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=now.isoformat(),
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            strategy_id=pos.strategy_id,
            setup_type=pos.setup_type,
            exit_reason=reason,
            hold_seconds=hold_seconds,
            indicators_at_entry=pos.indicators_at_entry,
        )
        self.trades.append(trade)

        # Update risk state
        self.risk.record_trade(pnl)

        # Update strategy state
        self.strategy_mgr.record_trade(pos.strategy_id, pnl)

        # Remove position
        del self.positions[symbol]

        self.agent_log.log(
            f"CLOSED {symbol}: {pos.direction.upper()} {pos.qty} "
            f"@ ${exit_price:.2f} "
            f"P&L ${pnl:+.2f} ({pnl_pct:+.2%}) "
            f"Hold {hold_seconds}s "
            f"(reason={reason})"
        )

        # Phase 3: Sound alert based on exit reason
        if 'stop' in reason.lower():
            await self._broadcast({'type': 'sound', 'sound': 'stop'})
        else:
            await self._broadcast({'type': 'sound', 'sound': 'exit'})

        if self.risk.is_halted:
            await self._set_agent_status('risk', 'halted')
            await self._broadcast({'type': 'sound', 'sound': 'killswitch'})
            self.agent_log.log(
                f"RISK HALT: {self.risk.halt_reason}", level='warning',
            )

        await self._save_state()
        await self._broadcast({
            'type': 'trade_closed',
            'trade': trade.to_dict(),
        })

        # Broadcast exit marker for charting
        now_epoch = int(now.timestamp())
        now_epoch = (now_epoch // 60) * 60
        await self._broadcast({
            'type': 'marker',
            'symbol': symbol,
            'time': now_epoch,
            'position': 'aboveBar' if pos.direction == 'long' else 'belowBar',
            'color': '#ffffff',
            'shape': 'circle',
            'text': f'EXIT ${exit_price:.2f} ({reason})',
        })

        return trade

    async def _update_broker_stop(self, pos: IntradayPosition) -> None:
        """Update the broker-side stop order for a position."""
        if pos.stop_order_id:
            await asyncio.to_thread(alpaca_cancel_order, pos.stop_order_id)

        stop_result = await asyncio.to_thread(
            alpaca_place_stop_order,
            pos.symbol, pos.qty, pos.stop_price,
            direction=pos.direction,
        )
        if 'id' in stop_result:
            pos.stop_order_id = stop_result['id']

    async def close_all_positions(self, reason: str = 'manual') -> int:
        """Close all open positions. Returns count closed."""
        symbols = list(self.positions.keys())
        closed = 0
        for symbol in symbols:
            try:
                trade = await self._close_position(symbol, reason)
                if trade:
                    closed += 1
            except Exception as e:
                self.agent_log.log(
                    f"Failed to close {symbol}: {e}", level='error',
                )
        return closed

    # ── EOD Watchdog ──────────────────────────────────────────────────

    async def _eod_watchdog(self) -> None:
        """End-of-day watchdog: close all positions before market close."""
        while self.is_trading:
            try:
                now = datetime.now(ET)
                t = now.time()

                # Phase 1: Warning at 3:45 PM
                if t >= dt_time(15, 45) and t < dt_time(15, 50):
                    if self.positions:
                        self.agent_log.log(
                            f"EOD WARNING: {len(self.positions)} positions open, "
                            f"closing at 3:50 PM",
                            level='warning',
                        )

                # Phase 2: Close all at 3:50 PM
                if t >= dt_time(15, 50) and t < dt_time(15, 55):
                    if self.positions:
                        self.agent_log.log(
                            f"EOD CLOSE: Closing {len(self.positions)} positions",
                            level='warning',
                        )
                        await self.close_all_positions('eod_close')
                        self.is_trading = False

                # Phase 3: Emergency at 3:55 PM
                if t >= dt_time(15, 55):
                    if self.positions:
                        self.agent_log.log(
                            f"EMERGENCY: Force closing {len(self.positions)} positions",
                            level='error',
                        )
                        await self.close_all_positions('eod_emergency')
                        self.is_trading = False

                await asyncio.sleep(10)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("EOD watchdog error: %s", e)
                await asyncio.sleep(10)

    # ── State Persistence ─────────────────────────────────────────────

    async def _save_state(self) -> None:
        """Save current state to file and DB."""
        state = {
            'trading_date': self._trading_date,
            'positions': [p.to_dict() for p in self.positions.values()],
            'trades': [t.to_dict() for t in self.trades[-100:]],
            'risk': {
                'daily_pnl': self.risk.daily_pnl,
                'max_drawdown': self.risk.max_drawdown,
                'peak_equity': self.risk.peak_equity,
                'consecutive_losses': self.risk.consecutive_losses,
                'trades_today': self.risk.trades_today,
                'is_halted': self.risk.is_halted,
                'halt_reason': self.risk.halt_reason,
            },
            'strategy_enabled': {
                sid: state.enabled
                for sid, state in self.strategy_mgr.get_all_states().items()
            },
            'saved_at': datetime.now(ET).isoformat(),
        }
        await self.state_mgr.save(state)
        await self.state_mgr.save_to_db(state)

    async def _periodic_state_save(self) -> None:
        """Save state every 5 minutes."""
        while self.is_trading:
            try:
                await asyncio.sleep(300)
                await self._save_state()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Periodic state save error: %s", e)

    # ── Broadcast ─────────────────────────────────────────────────────

    async def _broadcast(self, data: dict) -> None:
        """Send data to all connected WebSocket clients."""
        if not self.ws_clients:
            return
        message = json.dumps(data, default=str)
        stale = []
        for ws in self.ws_clients:
            try:
                await ws.send_text(message)
            except Exception:
                stale.append(ws)
        for ws in stale:
            try:
                self.ws_clients.remove(ws)
            except ValueError:
                pass

    # ── Phase 3: Agent Status ────────────────────────────────────────

    async def _set_agent_status(self, agent: str, status: str) -> None:
        """Update an agent avatar status and broadcast to clients."""
        self.agent_statuses[agent] = status
        await self._broadcast({
            'type': 'agent_status',
            'agents': dict(self.agent_statuses),
        })

    # ── Phase 3: Manual Order Execution ──────────────────────────────

    async def place_manual_order(
        self, symbol: str, side: str, qty: int,
        order_type: str = 'market', limit_price: Optional[float] = None,
    ) -> dict:
        """Place a manual order (from one-click trading or hotkeys)."""
        symbol = symbol.upper()

        # Validate
        if qty <= 0:
            return {'error': 'Invalid quantity'}
        if side not in ('buy', 'sell'):
            return {'error': 'Side must be buy or sell'}

        # Risk check
        if self.risk.is_halted:
            return {'error': f'Trading halted: {self.risk.halt_reason}'}

        await self._set_agent_status('executor', 'executing')

        self.agent_log.log(
            f"MANUAL ORDER: {side.upper()} {qty} {symbol} ({order_type})"
        )

        result = await alpaca_submit_and_confirm(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            timeout_sec=15.0,
            cancel_on_timeout=(order_type == 'market'),
        )

        self._last_order_id = result.order_id

        if result.is_filled:
            await self._set_agent_status('executor', 'idle')

            # Broadcast sound event
            await self._broadcast({
                'type': 'sound',
                'sound': 'entry',
            })

            self.agent_log.log(
                f"MANUAL FILLED: {side.upper()} {result.filled_qty} {symbol} "
                f"@ ${result.filled_avg_price:.2f}"
            )
            return {
                'status': 'filled',
                'order_id': result.order_id,
                'filled_qty': result.filled_qty,
                'filled_avg_price': result.filled_avg_price,
            }
        else:
            await self._set_agent_status('executor', 'idle')
            self.agent_log.log(
                f"MANUAL ORDER FAILED: {result.status} — {result.error}",
                level='warning',
            )
            return {
                'status': result.status,
                'error': result.error,
                'order_id': result.order_id,
            }

    async def flatten_symbol(self, symbol: str) -> dict:
        """Flatten (close) all positions for a symbol."""
        symbol = symbol.upper()
        if symbol in self.positions:
            trade = await self._close_position(symbol, 'manual_flatten')
            if trade:
                return {'status': 'closed', 'pnl': trade.pnl}
            return {'error': 'Close failed'}
        return {'error': f'No position in {symbol}'}

    async def cancel_last_order(self) -> dict:
        """Cancel the most recent manual order."""
        if not self._last_order_id:
            return {'error': 'No recent order to cancel'}
        ok = await asyncio.to_thread(alpaca_cancel_order, self._last_order_id)
        if ok:
            self.agent_log.log(f"Cancelled last order: {self._last_order_id}")
            return {'status': 'cancelled', 'order_id': self._last_order_id}
        return {'error': f'Failed to cancel order {self._last_order_id}'}

    async def update_position_stop(self, symbol: str, new_stop: float) -> dict:
        """Update stop price for a position (draggable stop)."""
        symbol = symbol.upper()
        pos = self.positions.get(symbol)
        if pos is None:
            return {'error': f'No position in {symbol}'}
        if new_stop <= 0:
            return {'error': 'Invalid stop price'}

        old_stop = pos.stop_price
        pos.stop_price = round(new_stop, 2)
        self.agent_log.log(
            f"STOP MOVED {symbol}: ${old_stop:.2f} -> ${pos.stop_price:.2f} (manual)"
        )
        await self._update_broker_stop(pos)
        await self._save_state()
        await self._broadcast({
            'type': 'update',
            'health': self.get_health(),
            'positions': self.get_positions_list(),
        })
        return {'status': 'ok', 'symbol': symbol, 'new_stop': pos.stop_price}

    async def update_position_target(self, symbol: str, new_target: float) -> dict:
        """Update target price for a position (draggable target)."""
        symbol = symbol.upper()
        pos = self.positions.get(symbol)
        if pos is None:
            return {'error': f'No position in {symbol}'}
        if new_target <= 0:
            return {'error': 'Invalid target price'}

        old_target = pos.target_price
        pos.target_price = round(new_target, 2)
        self.agent_log.log(
            f"TARGET MOVED {symbol}: ${old_target:.2f} -> ${pos.target_price:.2f} (manual)"
        )
        await self._save_state()
        await self._broadcast({
            'type': 'update',
            'health': self.get_health(),
            'positions': self.get_positions_list(),
        })
        return {'status': 'ok', 'symbol': symbol, 'new_target': pos.target_price}

    def get_dom_data(self, symbol: str) -> dict:
        """Get DOM ladder data for a symbol."""
        symbol = symbol.upper()
        streamer = self.streamer
        current_price = streamer.latest_prices.get(symbol, 0) if streamer else 0
        if current_price <= 0:
            return {'symbol': symbol, 'levels': [], 'current_price': 0}

        # Calculate tick size (penny stocks = 0.01, else auto)
        if current_price < 1:
            tick_size = 0.0001
        elif current_price < 10:
            tick_size = 0.01
        elif current_price < 100:
            tick_size = 0.01
        else:
            tick_size = 0.01

        # Build 20 levels centered on current price
        levels = []
        vol_at_price = self._volume_at_price.get(symbol, {})
        center = round(current_price / tick_size) * tick_size

        for i in range(10, -10, -1):
            price_level = round(center + i * tick_size, 4)
            vol = vol_at_price.get(round(price_level, 2), 0)
            levels.append({
                'price': round(price_level, 2),
                'volume': vol,
            })

        # Position info
        pos = self.positions.get(symbol)
        pos_info = None
        if pos:
            pos_info = {
                'entry_price': pos.entry_price,
                'stop_price': pos.stop_price,
                'target_price': pos.target_price,
                'direction': pos.direction,
            }

        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'levels': levels,
            'position': pos_info,
        }

    # ── Public API helpers ────────────────────────────────────────────

    def get_health(self) -> dict:
        """Get engine health status."""
        now = datetime.now(ET)
        uptime = (
            (now - self._startup_time).total_seconds()
            if self._startup_time else 0
        )
        return {
            'status': 'running' if self.is_trading else 'stopped',
            'uptime_seconds': int(uptime),
            'positions': len(self.positions),
            'trades_today': self.risk.trades_today,
            'daily_pnl': round(self.risk.daily_pnl, 2),
            'is_halted': self.risk.is_halted,
            'halt_reason': self.risk.halt_reason,
            'streamer_connected': self.streamer.connected if self.streamer else False,
            'symbols_tracked': len(self.streamer.symbols) if self.streamer else 0,
            'last_scan': self._last_scan_time.isoformat() if self._last_scan_time else None,
            'strategies_enabled': len(self.strategy_mgr.get_enabled_strategies()),
            'trading_date': self._trading_date,
        }

    def get_positions_list(self) -> List[dict]:
        """Get list of current positions as dicts."""
        return [p.to_dict() for p in self.positions.values()]

    def get_trades_list(self) -> List[dict]:
        """Get list of today's trades as dicts."""
        return [t.to_dict() for t in self.trades]

    def get_strategies_list(self) -> List[dict]:
        """Get all strategy states."""
        result = []
        for sid, state in self.strategy_mgr.get_all_states().items():
            strategy = self.strategy_mgr.get_strategy(sid)
            result.append({
                **state.to_dict(),
                'name': strategy.name if strategy else sid,
                'description': strategy.description if strategy else '',
                'version': strategy.version if strategy else '?',
            })
        return result

    def get_watchlist_data(self) -> List[dict]:
        """Get watchlist with live prices and indicator data."""
        engine = self.strategy_mgr.indicator_engine
        streamer = self.streamer
        if not streamer or not engine:
            return []

        result = []
        for sym in LIQUID_UNIVERSE[:50]:
            price = streamer.latest_prices.get(sym, 0)
            data = engine.get_data(sym)
            vwap = data.get('vwap', 0)
            vwap_dev = 0.0
            if vwap > 0 and price > 0:
                vwap_dev = (price - vwap) / vwap

            result.append({
                'symbol': sym,
                'price': round(price, 2),
                'vwap': round(vwap, 2) if vwap else 0,
                'vwap_deviation_pct': round(vwap_dev * 100, 2),
                'rsi': round(data.get('rsi', 50), 1),
                'volume_surge': round(data.get('volume_surge_ratio', 0), 1),
                'in_position': sym in self.positions,
            })
        return result


# =============================================================================
# SECTION 9: FASTAPI APPLICATION
# =============================================================================

app = FastAPI(title='Rudra Intraday Scalper', version='1.0')
engine = IntradayTradingEngine()


@app.on_event('startup')
async def startup() -> None:
    """Initialize engine on startup."""
    await engine.initialize()
    logger.info("Rudra Intraday Scalper started on port %d", INTRADAY_PORT)


@app.on_event('shutdown')
async def shutdown() -> None:
    """Cleanup on shutdown."""
    if engine.is_trading:
        await engine.stop_trading()
    if engine.streamer:
        await engine.streamer.stop()
    await engine._save_state()
    logger.info("Rudra Intraday Scalper shutdown complete")


# ── Health ────────────────────────────────────────────────────────────

@app.get('/api/health')
async def api_health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(engine.get_health())


# ── Positions ─────────────────────────────────────────────────────────

@app.get('/api/positions')
async def api_positions() -> JSONResponse:
    """Get current positions."""
    return JSONResponse(engine.get_positions_list())


# ── Trades ────────────────────────────────────────────────────────────

@app.get('/api/trades')
async def api_trades() -> JSONResponse:
    """Get today's trades."""
    return JSONResponse(engine.get_trades_list())


# ── Strategies ────────────────────────────────────────────────────────

@app.get('/api/strategies')
async def api_strategies() -> JSONResponse:
    """Get strategy status."""
    return JSONResponse(engine.get_strategies_list())


@app.post('/api/strategies/{strategy_id}/enable')
async def api_enable_strategy(strategy_id: str) -> JSONResponse:
    """Enable a strategy."""
    ok = engine.strategy_mgr.enable_strategy(strategy_id)
    if not ok:
        return JSONResponse({'error': f'Strategy {strategy_id} not found'}, status_code=404)
    return JSONResponse({'status': 'enabled', 'strategy_id': strategy_id})


@app.post('/api/strategies/{strategy_id}/disable')
async def api_disable_strategy(strategy_id: str) -> JSONResponse:
    """Disable a strategy."""
    ok = engine.strategy_mgr.disable_strategy(strategy_id)
    if not ok:
        return JSONResponse({'error': f'Strategy {strategy_id} not found'}, status_code=404)
    return JSONResponse({'status': 'disabled', 'strategy_id': strategy_id})


# ── Trading Controls ──────────────────────────────────────────────────

@app.post('/api/start')
async def api_start() -> JSONResponse:
    """Start trading."""
    await engine.start_trading()
    return JSONResponse({'status': 'started'})


@app.post('/api/stop')
async def api_stop() -> JSONResponse:
    """Stop trading (does not close positions)."""
    await engine.stop_trading()
    return JSONResponse({'status': 'stopped'})


@app.post('/api/close/{symbol}')
async def api_close_position(symbol: str) -> JSONResponse:
    """Force close a specific position."""
    symbol = symbol.upper()
    if symbol not in engine.positions:
        return JSONResponse({'error': f'No position in {symbol}'}, status_code=404)
    trade = await engine._close_position(symbol, 'manual_close')
    if trade:
        return JSONResponse(trade.to_dict())
    return JSONResponse({'error': 'Close failed'}, status_code=500)


@app.post('/api/close_all')
async def api_close_all() -> JSONResponse:
    """Close all positions."""
    count = await engine.close_all_positions('manual_close_all')
    return JSONResponse({'status': 'ok', 'closed': count})


# ── Watchlist ─────────────────────────────────────────────────────────

@app.get('/api/watchlist')
async def api_watchlist() -> JSONResponse:
    """Get watchlist data with live indicators."""
    return JSONResponse(engine.get_watchlist_data())


# ── Bars (for charting) ───────────────────────────────────────────────

@app.get('/api/bars/{symbol}')
async def api_bars(symbol: str, timeframe: str = '1m', limit: int = 390) -> JSONResponse:
    """Return recent 1-min bars for charting from the indicator engine's bar_history."""
    symbol = symbol.upper()
    bars: List[dict] = []

    # Try to get bars from the indicator engine
    ind_engine = engine.strategy_mgr.indicator_engine
    if ind_engine:
        data = ind_engine.get_data(symbol)
        bar_history = data.get('bar_history', [])
        now = datetime.now(ET)
        today_str = now.strftime('%Y-%m-%d')

        for bar in bar_history[-limit:]:
            # FiveMinBar has .bar_time (H:MM), .open, .high, .low, .close, .volume
            bar_time_str = bar.bar_time if hasattr(bar, 'bar_time') else ''
            if not bar_time_str:
                continue
            try:
                # Parse bar_time like '9:30' or '10:25'
                parts = bar_time_str.split(':')
                h_val = int(parts[0])
                m_val = int(parts[1]) if len(parts) > 1 else 0
                bar_dt = datetime(
                    now.year, now.month, now.day,
                    h_val, m_val, 0,
                    tzinfo=ET,
                )
                # Convert to UTC epoch for lightweight-charts
                epoch = int(bar_dt.timestamp())
                bars.append({
                    'time': epoch,
                    'open': round(getattr(bar, 'open', 0), 4),
                    'high': round(getattr(bar, 'high', 0), 4),
                    'low': round(getattr(bar, 'low', 0), 4),
                    'close': round(getattr(bar, 'close', 0), 4),
                    'volume': getattr(bar, 'volume', 0),
                })
            except (ValueError, TypeError):
                continue

    return JSONResponse(bars)


# ── Agent Log ─────────────────────────────────────────────────────────

@app.get('/api/log')
async def api_log() -> JSONResponse:
    """Get recent agent log entries."""
    return JSONResponse(engine.agent_log.get_recent(200))


# ── Phase 3: Manual Order ─────────────────────────────────────────────

@app.post('/api/order')
async def api_place_order(request: Request) -> JSONResponse:
    """Place a manual order. Body: {symbol, side, qty, order_type, limit_price}."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({'error': 'Invalid JSON body'}, status_code=400)

    symbol = str(body.get('symbol', '')).upper()
    side = str(body.get('side', '')).lower()
    qty = int(body.get('qty', 0))
    order_type = str(body.get('order_type', 'market')).lower()
    limit_price = body.get('limit_price')

    if not symbol:
        return JSONResponse({'error': 'Missing symbol'}, status_code=400)
    if side not in ('buy', 'sell'):
        return JSONResponse({'error': 'Side must be buy or sell'}, status_code=400)
    if qty <= 0:
        return JSONResponse({'error': 'Quantity must be positive'}, status_code=400)

    if limit_price is not None:
        try:
            limit_price = float(limit_price)
        except (ValueError, TypeError):
            return JSONResponse({'error': 'Invalid limit_price'}, status_code=400)

    result = await engine.place_manual_order(
        symbol=symbol, side=side, qty=qty,
        order_type=order_type, limit_price=limit_price,
    )
    status_code = 200 if result.get('status') == 'filled' else 400
    return JSONResponse(result, status_code=status_code)


@app.post('/api/flatten/{symbol}')
async def api_flatten_symbol(symbol: str) -> JSONResponse:
    """Flatten all positions for a symbol."""
    result = await engine.flatten_symbol(symbol)
    if 'error' in result:
        return JSONResponse(result, status_code=404)
    return JSONResponse(result)


@app.post('/api/cancel_last')
async def api_cancel_last() -> JSONResponse:
    """Cancel the most recent manual order."""
    result = await engine.cancel_last_order()
    if 'error' in result:
        return JSONResponse(result, status_code=400)
    return JSONResponse(result)


# ── Phase 3: Draggable Stop/Target ──────────────────────────────────

@app.post('/api/positions/{symbol}/stop')
async def api_update_stop(symbol: str, request: Request) -> JSONResponse:
    """Update stop price. Body: {new_stop: float}."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({'error': 'Invalid JSON body'}, status_code=400)

    new_stop = body.get('new_stop')
    if new_stop is None:
        return JSONResponse({'error': 'Missing new_stop'}, status_code=400)
    try:
        new_stop = float(new_stop)
    except (ValueError, TypeError):
        return JSONResponse({'error': 'Invalid new_stop value'}, status_code=400)

    result = await engine.update_position_stop(symbol, new_stop)
    if 'error' in result:
        return JSONResponse(result, status_code=400)
    return JSONResponse(result)


@app.post('/api/positions/{symbol}/target')
async def api_update_target(symbol: str, request: Request) -> JSONResponse:
    """Update target price. Body: {new_target: float}."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({'error': 'Invalid JSON body'}, status_code=400)

    new_target = body.get('new_target')
    if new_target is None:
        return JSONResponse({'error': 'Missing new_target'}, status_code=400)
    try:
        new_target = float(new_target)
    except (ValueError, TypeError):
        return JSONResponse({'error': 'Invalid new_target value'}, status_code=400)

    result = await engine.update_position_target(symbol, new_target)
    if 'error' in result:
        return JSONResponse(result, status_code=400)
    return JSONResponse(result)


# ── Phase 3: DOM Data ────────────────────────────────────────────────

@app.get('/api/dom/{symbol}')
async def api_dom(symbol: str) -> JSONResponse:
    """Get DOM ladder data for a symbol."""
    return JSONResponse(engine.get_dom_data(symbol.upper()))


# ── Phase 3: Share Size Control ──────────────────────────────────────

@app.post('/api/share_size')
async def api_share_size(request: Request) -> JSONResponse:
    """Update manual share size. Body: {size: int}."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({'error': 'Invalid JSON body'}, status_code=400)
    size = int(body.get('size', 0))
    if size <= 0:
        return JSONResponse({'error': 'Size must be positive'}, status_code=400)
    engine.manual_share_size = size
    return JSONResponse({'status': 'ok', 'size': size})


# ── WebSocket ─────────────────────────────────────────────────────────

@app.websocket('/ws')
async def websocket_endpoint(ws: WebSocket) -> None:
    """WebSocket for live dashboard updates."""
    await ws.accept()
    engine.ws_clients.append(ws)
    _last_bar_counts: Dict[str, int] = {}  # track bars sent per symbol
    _bar_tick_counter = 0
    try:
        # Send initial state
        await ws.send_json({
            'type': 'init',
            'health': engine.get_health(),
            'positions': engine.get_positions_list(),
            'trades': engine.get_trades_list()[-20:],
            'strategies': engine.get_strategies_list(),
            'log': engine.agent_log.get_recent(50),
            'agent_statuses': dict(engine.agent_statuses),
            'share_size': engine.manual_share_size,
        })

        # Periodic updates
        while True:
            await asyncio.sleep(1)
            _bar_tick_counter += 1
            await ws.send_json({
                'type': 'update',
                'health': engine.get_health(),
                'positions': engine.get_positions_list(),
                'risk': {
                    'daily_pnl': round(engine.risk.daily_pnl, 2),
                    'max_drawdown': round(engine.risk.max_drawdown, 2),
                    'consecutive_losses': engine.risk.consecutive_losses,
                    'is_halted': engine.risk.is_halted,
                    'halt_reason': engine.risk.halt_reason,
                    'trades_today': engine.risk.trades_today,
                    'daily_loss_limit': engine.risk.daily_loss_limit,
                },
            })

            # Every 5 seconds, send bar/vwap updates for watched symbols
            if _bar_tick_counter % 5 == 0:
                ind_engine = engine.strategy_mgr.indicator_engine
                if ind_engine:
                    now = datetime.now(ET)
                    for sym in list(engine.positions.keys()) + ['SPY', 'QQQ']:
                        try:
                            data = ind_engine.get_data(sym)
                            bar_history = data.get('bar_history', [])
                            cur_count = len(bar_history)
                            prev_count = _last_bar_counts.get(sym, 0)

                            # Send new/updated bars
                            if cur_count > 0 and cur_count != prev_count:
                                # Send last bar (latest)
                                bar = bar_history[-1]
                                bar_time_str = getattr(bar, 'bar_time', '')
                                if bar_time_str:
                                    parts = bar_time_str.split(':')
                                    h_val = int(parts[0])
                                    m_val = int(parts[1]) if len(parts) > 1 else 0
                                    bar_dt = datetime(
                                        now.year, now.month, now.day,
                                        h_val, m_val, 0, tzinfo=ET,
                                    )
                                    epoch = int(bar_dt.timestamp())
                                    await ws.send_json({
                                        'type': 'bar',
                                        'symbol': sym,
                                        'time': epoch,
                                        'open': round(getattr(bar, 'open', 0), 4),
                                        'high': round(getattr(bar, 'high', 0), 4),
                                        'low': round(getattr(bar, 'low', 0), 4),
                                        'close': round(getattr(bar, 'close', 0), 4),
                                        'volume': getattr(bar, 'volume', 0),
                                    })
                                _last_bar_counts[sym] = cur_count

                            # Send VWAP
                            vwap_val = data.get('vwap', 0)
                            if vwap_val > 0 and cur_count > 0:
                                bar = bar_history[-1]
                                bar_time_str = getattr(bar, 'bar_time', '')
                                if bar_time_str:
                                    parts = bar_time_str.split(':')
                                    h_val = int(parts[0])
                                    m_val = int(parts[1]) if len(parts) > 1 else 0
                                    bar_dt = datetime(
                                        now.year, now.month, now.day,
                                        h_val, m_val, 0, tzinfo=ET,
                                    )
                                    epoch = int(bar_dt.timestamp())
                                    await ws.send_json({
                                        'type': 'vwap',
                                        'symbol': sym,
                                        'time': epoch,
                                        'value': round(vwap_val, 4),
                                    })
                        except Exception:
                            pass

            # Every 60 seconds, send volume profile for the chart symbol
            if _bar_tick_counter % 60 == 0:
                ind_engine = engine.strategy_mgr.indicator_engine
                if ind_engine:
                    for sym in list(engine.positions.keys()) + ['SPY', 'QQQ']:
                        try:
                            data = ind_engine.get_data(sym)
                            bar_history = data.get('bar_history', [])
                            if bar_history:
                                # Build simple volume profile from bars
                                levels: Dict[float, int] = {}
                                for bar in bar_history:
                                    mid = round((getattr(bar, 'high', 0) + getattr(bar, 'low', 0)) / 2, 2)
                                    vol = getattr(bar, 'volume', 0)
                                    levels[mid] = levels.get(mid, 0) + vol
                                vp = [
                                    {'price': p, 'volume': v}
                                    for p, v in sorted(levels.items())
                                ]
                                if vp:
                                    await ws.send_json({
                                        'type': 'volume_profile',
                                        'symbol': sym,
                                        'levels': vp,
                                    })
                        except Exception:
                            pass

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            engine.ws_clients.remove(ws)
        except ValueError:
            pass


# =============================================================================
# SECTION 10: DASHBOARD HTML
# =============================================================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rudra Intraday Scalper</title>
<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
    --bg: #000000;
    --surface: #0a0a0a;
    --surface2: #111111;
    --border: #222222;
    --text: #e0e0e0;
    --text-dim: #888888;
    --green: #00c853;
    --red: #ff1744;
    --blue: #2979ff;
    --yellow: #ffd600;
    --orange: #ff9100;
    --font: 'Segoe UI', system-ui, -apple-system, sans-serif;
    --mono: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
}
body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 13px;
    line-height: 1.4;
    overflow-x: hidden;
}

/* Header Bar */
.header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 16px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
}
.header-left {
    display: flex;
    align-items: center;
    gap: 16px;
}
.app-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--blue);
    letter-spacing: 0.5px;
}
.pnl-display {
    font-size: 20px;
    font-weight: 700;
    font-family: var(--mono);
}
.pnl-positive { color: var(--green); }
.pnl-negative { color: var(--red); }
.pnl-zero { color: var(--text-dim); }
.header-right {
    display: flex;
    align-items: center;
    gap: 16px;
}
.stat-box {
    text-align: center;
    padding: 2px 10px;
}
.stat-label {
    font-size: 10px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.stat-value {
    font-size: 14px;
    font-weight: 600;
    font-family: var(--mono);
}
.status-dots {
    display: flex;
    gap: 6px;
    align-items: center;
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--red);
}
.status-dot.connected { background: var(--green); }
.status-dot-label {
    font-size: 9px;
    color: var(--text-dim);
}
.eod-timer {
    font-family: var(--mono);
    font-size: 14px;
    color: var(--yellow);
}
.btn {
    padding: 4px 12px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--surface2);
    color: var(--text);
    cursor: pointer;
    font-size: 12px;
    transition: background 0.15s;
}
.btn:hover { background: #1a1a1a; }
.btn-start { border-color: var(--green); color: var(--green); }
.btn-stop { border-color: var(--red); color: var(--red); }
.btn-danger { background: #2a0000; border-color: var(--red); color: var(--red); }

/* Main Grid */
.main-grid {
    display: grid;
    grid-template-columns: 60% 40%;
    gap: 1px;
    background: var(--border);
    min-height: calc(100vh - 46px - 140px);
}
.panel {
    background: var(--bg);
    padding: 12px;
}
.panel-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--blue);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}
th {
    text-align: left;
    padding: 4px 6px;
    color: var(--text-dim);
    font-weight: 500;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    background: var(--bg);
}
td {
    padding: 4px 6px;
    border-bottom: 1px solid #0d0d0d;
    font-family: var(--mono);
    font-size: 11px;
}
tr:hover td { background: var(--surface); }
.text-green { color: var(--green); }
.text-red { color: var(--red); }
.text-dim { color: var(--text-dim); }
.text-blue { color: var(--blue); }
.text-yellow { color: var(--yellow); }

/* Strategy Cards */
.strategy-card {
    padding: 6px 8px;
    margin-bottom: 4px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--surface);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.strategy-card.disabled { opacity: 0.4; }
.strategy-name { font-weight: 600; font-size: 12px; }
.strategy-status {
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 3px;
    text-transform: uppercase;
}
.status-scanning { background: #0a2a00; color: var(--green); }
.status-triggered { background: #2a2a00; color: var(--yellow); }
.status-cooldown { background: #2a1500; color: var(--orange); }
.status-idle { background: var(--surface2); color: var(--text-dim); }

/* Agent Log */
.agent-log {
    height: 220px;
    overflow-y: auto;
    font-family: var(--mono);
    font-size: 11px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px;
}
.log-entry {
    padding: 1px 0;
    border-bottom: 1px solid #0a0a0a;
}
.log-time { color: var(--text-dim); }
.log-info { color: var(--text); }
.log-warning { color: var(--yellow); }
.log-error { color: var(--red); }

/* Risk Monitor */
.risk-bar-container {
    margin: 6px 0;
}
.risk-bar-label {
    font-size: 10px;
    color: var(--text-dim);
    margin-bottom: 2px;
}
.risk-bar-track {
    height: 16px;
    background: var(--surface2);
    border-radius: 3px;
    overflow: hidden;
    position: relative;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s;
}
.risk-bar-value {
    position: absolute;
    right: 4px;
    top: 1px;
    font-size: 10px;
    font-family: var(--mono);
    color: var(--text);
}
.halted-banner {
    background: #2a0000;
    border: 2px solid var(--red);
    color: var(--red);
    text-align: center;
    padding: 8px;
    font-weight: 700;
    font-size: 14px;
    border-radius: 4px;
    margin: 8px 0;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Bottom Bar (Watchlist) */
.bottom-bar {
    border-top: 1px solid var(--border);
    padding: 8px 12px;
    background: var(--surface);
    max-height: 140px;
    overflow-y: auto;
}
.watchlist-title {
    font-size: 11px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.watchlist-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
}
.watchlist-item {
    font-family: var(--mono);
    font-size: 10px;
    padding: 2px 6px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 3px;
    display: flex;
    gap: 6px;
    align-items: center;
    min-width: 145px;
}
.watchlist-item.in-position { border-color: var(--blue); background: #0a0a2a; }
.wl-sym { font-weight: 600; color: var(--text); min-width: 40px; }
.wl-price { color: var(--text-dim); }
.wl-vwap { font-size: 9px; }

/* SVG Chart */
.chart-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px;
    height: 120px;
}
.chart-container svg {
    width: 100%;
    height: 100%;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #333; }

/* Main Price Chart */
.chart-wrapper {
    position: relative;
    margin-bottom: 12px;
}
.chart-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
}
.chart-symbol-label {
    font-size: 14px;
    font-weight: 700;
    color: var(--text);
    font-family: var(--mono);
}
.tf-buttons {
    display: flex;
    gap: 4px;
}
.tf-btn {
    padding: 2px 10px;
    border: 1px solid var(--border);
    border-radius: 3px;
    background: var(--surface2);
    color: var(--text-dim);
    cursor: pointer;
    font-size: 11px;
    font-family: var(--mono);
    transition: background 0.15s, color 0.15s;
}
.tf-btn:hover { background: #1a1a1a; color: var(--text); }
.tf-btn.active { background: var(--blue); color: #fff; border-color: var(--blue); }
#mainChart {
    height: 350px;
    width: 100%;
    position: relative;
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
}
#volumeProfileOverlay {
    position: absolute;
    top: 0;
    right: 0;
    width: 80px;
    height: 350px;
    pointer-events: none;
    z-index: 5;
}

/* Risk Gauges */
.gauge-row {
    display: flex;
    gap: 12px;
    margin-top: 8px;
    align-items: stretch;
}
.gauge-card {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px;
    text-align: center;
}
.gauge-label {
    font-size: 10px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.gauge-value {
    font-size: 18px;
    font-weight: 700;
    font-family: var(--mono);
}
.pnl-meter-track {
    height: 14px;
    background: linear-gradient(to right, var(--red), #333 50%, var(--green));
    border-radius: 7px;
    position: relative;
    margin-top: 4px;
    overflow: visible;
}
.pnl-meter-needle {
    position: absolute;
    top: -2px;
    width: 4px;
    height: 18px;
    background: #fff;
    border-radius: 2px;
    transition: left 0.3s;
}
.pnl-meter-mark {
    position: absolute;
    top: -2px;
    width: 2px;
    height: 18px;
    background: var(--yellow);
    opacity: 0.7;
}
.dd-gauge-svg {
    width: 100px;
    height: 55px;
    margin: 0 auto;
    display: block;
}
.sharpe-value {
    font-size: 22px;
}
.sharpe-green { color: var(--green); }
.sharpe-yellow { color: var(--yellow); }
.sharpe-red { color: var(--red); }

/* Agent Decision Timeline */
.timeline-container {
    height: 220px;
    overflow-y: auto;
    padding: 6px 6px 6px 20px;
    position: relative;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
}
.timeline-line {
    position: absolute;
    left: 14px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--border);
}
.timeline-event {
    position: relative;
    padding: 3px 0 3px 16px;
    font-size: 11px;
    font-family: var(--mono);
}
.timeline-dot {
    position: absolute;
    left: -12px;
    top: 6px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.dot-scan { background: var(--blue); }
.dot-signal { background: var(--yellow); }
.dot-entry { background: var(--green); }
.dot-exit { background: var(--red); }
.dot-skip { background: #555; }
.timeline-time {
    color: var(--text-dim);
    font-size: 9px;
}
.timeline-msg {
    color: var(--text);
}
.timeline-event:hover .timeline-detail {
    display: block;
}
.timeline-detail {
    display: none;
    position: absolute;
    left: 16px;
    top: 100%;
    background: var(--surface2);
    border: 1px solid var(--border);
    padding: 4px 8px;
    border-radius: 3px;
    font-size: 10px;
    color: var(--text-dim);
    z-index: 10;
    white-space: nowrap;
    max-width: 350px;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Watchlist sparkline and VWAP flash */
.wl-sparkline {
    display: inline-block;
    vertical-align: middle;
}
.watchlist-item {
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
}
.watchlist-item:hover { border-color: var(--blue); background: #0a0a1a; }
.watchlist-item.vwap-flash {
    animation: vwapFlash 0.6s ease-out;
}
@keyframes vwapFlash {
    0% { box-shadow: 0 0 8px var(--blue); }
    100% { box-shadow: none; }
}

/* ═══ Phase 3: War Room (Agent Avatars) ═══ */
.war-room {
    display: flex;
    gap: 8px;
    padding: 8px;
    background: #0a0a0a;
    border-radius: 8px;
    margin-bottom: 12px;
    border: 1px solid var(--border);
}
.agent-avatar {
    text-align: center;
    padding: 8px 12px;
    border-radius: 6px;
    background: #111;
    min-width: 70px;
    flex: 1;
}
.agent-icon { font-size: 24px; }
.agent-name { font-size: 10px; color: #94a3b8; margin-top: 2px; }
.agent-status-badge {
    font-size: 10px;
    font-weight: 700;
    margin-top: 4px;
    padding: 2px 6px;
    border-radius: 3px;
    display: inline-block;
}
.agent-status-badge.scanning { color: #06b6d4; background: rgba(6,182,212,0.15); }
.agent-status-badge.idle { color: #64748b; background: rgba(100,116,139,0.15); }
.agent-status-badge.evaluating { color: #f59e0b; background: rgba(245,158,11,0.15); }
.agent-status-badge.approved { color: #22c55e; background: rgba(34,197,94,0.15); }
.agent-status-badge.rejected { color: #ef4444; background: rgba(239,68,68,0.15); }
.agent-status-badge.executing { color: #22c55e; background: rgba(34,197,94,0.15); animation: pulse 1s infinite; }
.agent-status-badge.ok { color: #22c55e; background: rgba(34,197,94,0.15); }
.agent-status-badge.warning { color: #f59e0b; background: rgba(245,158,11,0.15); }
.agent-status-badge.halted { color: #ef4444; background: rgba(239,68,68,0.15); animation: pulse 1s infinite; }

/* ═══ Phase 3: Quick Order Panel ═══ */
.quick-order-panel {
    display: flex;
    gap: 8px;
    padding: 8px;
    background: #111;
    border: 1px solid var(--border);
    border-radius: 4px;
    margin-bottom: 8px;
    align-items: center;
}
.quick-order-panel .btn-buy {
    flex: 1;
    padding: 12px;
    background: #22c55e;
    color: #000;
    font-weight: bold;
    font-size: 14px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.15s;
}
.quick-order-panel .btn-buy:hover { background: #16a34a; }
.quick-order-panel .btn-sell {
    flex: 1;
    padding: 12px;
    background: #ef4444;
    color: #fff;
    font-weight: bold;
    font-size: 14px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.15s;
}
.quick-order-panel .btn-sell:hover { background: #dc2626; }
.quick-order-panel .btn-flatten {
    padding: 12px;
    background: #f59e0b;
    color: #000;
    font-weight: bold;
    font-size: 13px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.15s;
}
.quick-order-panel .btn-flatten:hover { background: #d97706; }
.share-size-control {
    display: flex;
    align-items: center;
    gap: 4px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text);
}
.share-size-control input {
    width: 60px;
    text-align: center;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 3px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 12px;
    padding: 4px;
}
.share-size-control button {
    width: 24px;
    height: 24px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 3px;
    color: var(--text);
    cursor: pointer;
    font-size: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.share-size-control button:hover { background: #1a1a1a; }
.hotkey-badge {
    font-size: 9px;
    color: var(--text-dim);
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 0 3px;
    margin-right: 4px;
    font-family: var(--mono);
}

/* ═══ Phase 3: DOM Ladder ═══ */
.dom-ladder {
    height: 400px;
    overflow: hidden;
    font-family: var(--mono);
    font-size: 11px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    margin-bottom: 8px;
}
.dom-header {
    padding: 4px 8px;
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
    font-size: 11px;
    font-weight: 600;
    color: var(--blue);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.dom-body {
    height: calc(400px - 26px);
    overflow-y: auto;
}
.dom-row {
    display: flex;
    align-items: center;
    padding: 1px 6px;
    border-bottom: 1px solid #0a0a0a;
    cursor: pointer;
    transition: background 0.1s;
}
.dom-row:hover { background: #1a1a1a; }
.dom-row.current-price { border: 1px solid #ffd600; }
.dom-row.current-price .dom-price { color: #ffd600; font-weight: 700; }
.dom-row.stop-level { background: rgba(239,68,68,0.15); }
.dom-row.target-level { background: rgba(34,197,94,0.15); }
.dom-row.entry-level { background: rgba(41,121,255,0.15); }
.dom-row.above-entry { background: rgba(239,68,68,0.04); }
.dom-row.below-entry { background: rgba(34,197,94,0.04); }
.dom-price { width: 70px; text-align: right; padding-right: 8px; }
.dom-volume { width: 60px; text-align: right; padding-right: 8px; }
.dom-vol-bar {
    flex: 1;
    height: 12px;
    background: var(--surface2);
    border-radius: 2px;
    overflow: hidden;
    position: relative;
}
.dom-vol-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s;
}
.dom-level-label {
    width: 40px;
    text-align: center;
    font-size: 9px;
    font-weight: 700;
}

/* ═══ Phase 3: Sound Mute Toggle ═══ */
.sound-toggle {
    cursor: pointer;
    font-size: 16px;
    padding: 2px 6px;
    border-radius: 3px;
    user-select: none;
}
.sound-toggle:hover { background: var(--surface2); }
.sound-toggle.muted { opacity: 0.4; }
</style>
</head>
<body>

<!-- Header Bar -->
<div class="header">
    <div class="header-left">
        <span class="app-title">RUDRA INTRADAY SCALPER</span>
        <span id="pnl-display" class="pnl-display pnl-zero">$0.00</span>
    </div>
    <div class="header-right">
        <div class="stat-box">
            <div class="stat-label">Positions</div>
            <div class="stat-value" id="pos-count">0</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Trades</div>
            <div class="stat-value" id="trade-count">0</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">EOD</div>
            <div class="eod-timer" id="eod-timer">--:--:--</div>
        </div>
        <div class="status-dots">
            <div class="status-dot" id="dot-ws" title="WebSocket"></div>
            <span class="status-dot-label">WS</span>
            <div class="status-dot" id="dot-alpaca" title="Alpaca Stream"></div>
            <span class="status-dot-label">ALP</span>
        </div>
        <span class="sound-toggle" id="sound-toggle" onclick="toggleSound()" title="Toggle Sound Alerts">&#x1f50a;</span>
        <button class="btn btn-start" id="btn-start" onclick="startTrading()">START</button>
        <button class="btn btn-stop" id="btn-stop" onclick="stopTrading()">STOP</button>
        <button class="btn btn-danger" onclick="closeAll()"><span class="hotkey-badge">Shift+X</span>CLOSE ALL</button>
    </div>
</div>

<!-- Main Grid -->
<div class="main-grid">

    <!-- Left Column (60%) -->
    <div class="panel">
        <!-- Main Price Chart -->
        <div class="chart-wrapper">
            <div class="chart-header">
                <div>
                    <span class="panel-title" style="display:inline;border:none;margin:0;padding:0;">Price Chart</span>
                    <span class="chart-symbol-label" id="chart-symbol-label">SPY</span>
                </div>
                <div class="tf-buttons">
                    <button class="tf-btn active" data-tf="1" onclick="setTimeframe(1, this)">1m</button>
                    <button class="tf-btn" data-tf="5" onclick="setTimeframe(5, this)">5m</button>
                    <button class="tf-btn" data-tf="15" onclick="setTimeframe(15, this)">15m</button>
                </div>
            </div>
            <div id="mainChart">
                <canvas id="volumeProfileOverlay"></canvas>
            </div>
        </div>

        <div class="panel-title">Active Positions</div>
        <div style="max-height: 200px; overflow-y: auto;">
            <table id="positions-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Entry</th>
                        <th>Current</th>
                        <th>P&L</th>
                        <th>P&L%</th>
                        <th>Stop</th>
                        <th>Target</th>
                        <th>Strategy</th>
                        <th>Hold</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
            <div id="no-positions" class="text-dim" style="text-align:center;padding:20px;">No open positions</div>
        </div>

        <div style="margin-top: 12px;">
            <div class="panel-title">Recent Trades</div>
            <div style="max-height: 200px; overflow-y: auto;">
                <table id="trades-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Qty</th>
                            <th>Entry</th>
                            <th>Exit</th>
                            <th>P&L</th>
                            <th>P&L%</th>
                            <th>Strategy</th>
                            <th>Hold</th>
                            <th>Reason</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
                <div id="no-trades" class="text-dim" style="text-align:center;padding:20px;">No trades today</div>
            </div>
        </div>

        <div style="margin-top: 12px;">
            <div class="panel-title">Daily P&L Chart</div>
            <div class="chart-container" id="pnl-chart">
                <svg viewBox="0 0 600 100" preserveAspectRatio="none">
                    <line x1="0" y1="50" x2="600" y2="50" stroke="#222" stroke-width="0.5" stroke-dasharray="4,4"/>
                    <polyline id="pnl-line" fill="none" stroke="#2979ff" stroke-width="1.5" points=""/>
                </svg>
            </div>
        </div>
    </div>

    <!-- Right Column (40%) -->
    <div class="panel">
        <!-- Phase 3: Agent War Room -->
        <div class="war-room" id="warRoom">
            <div class="agent-avatar" id="agentScout">
                <div class="agent-icon">&#x1f50d;</div>
                <div class="agent-name">Scout</div>
                <div class="agent-status-badge idle" id="agent-scout-status">Idle</div>
            </div>
            <div class="agent-avatar" id="agentAnalyst">
                <div class="agent-icon">&#x1f4ca;</div>
                <div class="agent-name">Analyst</div>
                <div class="agent-status-badge idle" id="agent-analyst-status">Idle</div>
            </div>
            <div class="agent-avatar" id="agentExecutor">
                <div class="agent-icon">&#x26a1;</div>
                <div class="agent-name">Executor</div>
                <div class="agent-status-badge idle" id="agent-executor-status">Idle</div>
            </div>
            <div class="agent-avatar" id="agentRisk">
                <div class="agent-icon">&#x1f6e1;</div>
                <div class="agent-name">Risk</div>
                <div class="agent-status-badge ok" id="agent-risk-status">OK</div>
            </div>
        </div>

        <!-- Phase 3: Quick Order Panel -->
        <div class="quick-order-panel" id="quickOrderPanel">
            <div class="share-size-control">
                <button onclick="adjustShares(-100)">-</button>
                <input type="number" id="share-size-input" value="100" min="1" step="100">
                <button onclick="adjustShares(100)">+</button>
            </div>
            <button class="btn-buy" id="btnBuy" onclick="quickBuy()">
                <span class="hotkey-badge">B</span>BUY
            </button>
            <button class="btn-sell" id="btnSell" onclick="quickSell()">
                <span class="hotkey-badge">S</span>SELL
            </button>
            <button class="btn-flatten" id="btnFlatten" onclick="flattenSelected()">
                <span class="hotkey-badge">F</span>FLAT
            </button>
        </div>

        <!-- Phase 3: DOM Ladder -->
        <div class="dom-ladder" id="domLadder">
            <div class="dom-header" id="domHeader">DOM Ladder &mdash; <span id="domSymbolLabel">SPY</span></div>
            <div class="dom-body" id="domBody">
                <!-- Populated by JS -->
            </div>
        </div>

        <div class="panel-title">Strategy Status</div>
        <div id="strategy-cards" style="max-height: 180px; overflow-y: auto;"></div>

        <div style="margin-top: 12px;">
            <div class="panel-title">Agent Decision Timeline</div>
            <div class="timeline-container" id="agent-timeline">
                <div class="timeline-line"></div>
            </div>
        </div>

        <div style="margin-top: 12px;">
            <div class="panel-title">Risk Monitor</div>
            <div id="halted-banner" class="halted-banner" style="display:none;">TRADING HALTED</div>

            <!-- Visual Risk Gauges -->
            <div class="gauge-row">
                <div class="gauge-card">
                    <div class="gauge-label">Daily P&L (limit -$250)</div>
                    <div class="gauge-value" id="gauge-pnl-value" style="color:var(--text-dim);">$0.00</div>
                    <div class="pnl-meter-track">
                        <div class="pnl-meter-mark" id="pnl-limit-mark" style="left:25%;"></div>
                        <div class="pnl-meter-needle" id="pnl-meter-needle" style="left:50%;"></div>
                    </div>
                </div>
                <div class="gauge-card">
                    <div class="gauge-label">Drawdown</div>
                    <svg class="dd-gauge-svg" viewBox="0 0 100 55" id="dd-gauge-svg">
                        <path d="M10 50 A40 40 0 0 1 90 50" fill="none" stroke="#222" stroke-width="6" stroke-linecap="round"/>
                        <path d="M10 50 A40 40 0 0 1 90 50" fill="none" stroke="var(--green)" stroke-width="6" stroke-linecap="round"
                              id="dd-gauge-arc" stroke-dasharray="0 126"/>
                        <line x1="50" y1="50" x2="50" y2="15" stroke="#fff" stroke-width="2" stroke-linecap="round"
                              id="dd-gauge-needle" transform="rotate(-90, 50, 50)"/>
                        <circle cx="50" cy="50" r="3" fill="#fff"/>
                    </svg>
                    <div class="gauge-value" id="dd-gauge-value" style="font-size:12px;color:var(--text-dim);">0.0%</div>
                </div>
                <div class="gauge-card">
                    <div class="gauge-label">Live Sharpe</div>
                    <div class="gauge-value sharpe-value sharpe-red" id="sharpe-value">--</div>
                </div>
            </div>

            <div style="display:flex; gap:16px; margin-top:8px;">
                <div class="stat-box">
                    <div class="stat-label">Consecutive Losses</div>
                    <div class="stat-value" id="consec-losses">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Win Rate</div>
                    <div class="stat-value" id="win-rate">--</div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Bottom Bar: Watchlist -->
<div class="bottom-bar">
    <div class="watchlist-title">Watchlist (Top 50 Liquid Symbols)</div>
    <div class="watchlist-grid" id="watchlist"></div>
</div>

<script>
// ══════════════════════════════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════════════════════════════
let ws = null;
let wsConnected = false;
let pnlHistory = [0];
let logEntries = [];

// Chart state
window._chart = null;
window._candleSeries = null;
window._vwapLine = null;
window._volumeSeries = null;
window._stopLine = null;
window._targetLine = null;
window._orBox = null;
window._chartSymbol = 'SPY';
window._chartTimeframe = 1;
window._rawBars = {};       // symbol -> [{time,open,high,low,close,volume}]
window._rawVwap = {};       // symbol -> [{time, value}]
window._chartMarkers = {};  // symbol -> [{time,position,color,shape,text}]
window._volumeProfile = {}; // symbol -> [{price, volume}]
window._sparklines = {};    // symbol -> [price, price, ...]
window._prevVwapSide = {};  // symbol -> 'above'|'below' for flash detection
window._tradePnls = [];     // for Sharpe calculation

// ══════════════════════════════════════════════════════════════════════
// LIGHTWEIGHT CHARTS SETUP
// ══════════════════════════════════════════════════════════════════════
function initChart() {
    const container = document.getElementById('mainChart');
    if (!container || !window.LightweightCharts) return;
    const chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: 350,
        layout: {
            background: { type: 'solid', color: '#000000' },
            textColor: '#888',
            fontSize: 11,
        },
        grid: {
            vertLines: { color: '#111' },
            horzLines: { color: '#111' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        rightPriceScale: {
            borderColor: '#222',
        },
        timeScale: {
            borderColor: '#222',
            timeVisible: true,
            secondsVisible: false,
        },
    });

    const candleSeries = chart.addCandlestickSeries({
        upColor: '#00c853',
        downColor: '#ff1744',
        borderUpColor: '#00c853',
        borderDownColor: '#ff1744',
        wickUpColor: '#00c853',
        wickDownColor: '#ff1744',
    });

    const vwapLine = chart.addLineSeries({
        color: '#2979ff',
        lineWidth: 2,
        lineType: LightweightCharts.LineType.Simple,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
    });

    const volumeSeries = chart.addHistogramSeries({
        color: '#2979ff33',
        priceFormat: { type: 'volume' },
        priceScaleId: '',
        scaleMargins: { top: 0.85, bottom: 0 },
    });

    window._chart = chart;
    window._candleSeries = candleSeries;
    window._vwapLine = vwapLine;
    window._volumeSeries = volumeSeries;

    // Resize handler
    const ro = new ResizeObserver(() => {
        chart.applyOptions({ width: container.clientWidth });
    });
    ro.observe(container);
}

// ══════════════════════════════════════════════════════════════════════
// TIMEFRAME AGGREGATION
// ══════════════════════════════════════════════════════════════════════
function aggregateBars(bars, tfMinutes) {
    if (tfMinutes <= 1) return bars;
    const tfSec = tfMinutes * 60;
    const agg = [];
    let current = null;
    for (const b of bars) {
        const bucket = Math.floor(b.time / tfSec) * tfSec;
        if (!current || current.time !== bucket) {
            if (current) agg.push(current);
            current = { time: bucket, open: b.open, high: b.high, low: b.low, close: b.close, volume: b.volume || 0 };
        } else {
            current.high = Math.max(current.high, b.high);
            current.low = Math.min(current.low, b.low);
            current.close = b.close;
            current.volume = (current.volume || 0) + (b.volume || 0);
        }
    }
    if (current) agg.push(current);
    return agg;
}

function setTimeframe(tf, btnEl) {
    window._chartTimeframe = tf;
    document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
    if (btnEl) btnEl.classList.add('active');
    refreshChart();
}

// ══════════════════════════════════════════════════════════════════════
// CHART REFRESH
// ══════════════════════════════════════════════════════════════════════
function refreshChart() {
    if (!window._candleSeries) return;
    const sym = window._chartSymbol;
    const raw = window._rawBars[sym] || [];
    const bars = aggregateBars(raw, window._chartTimeframe);

    window._candleSeries.setData(bars);

    // VWAP
    const vwap = window._rawVwap[sym] || [];
    if (window._chartTimeframe <= 1) {
        window._vwapLine.setData(vwap);
    } else {
        // Downsample VWAP to match tf
        const tfSec = window._chartTimeframe * 60;
        const ds = [];
        for (const v of vwap) {
            const bucket = Math.floor(v.time / tfSec) * tfSec;
            if (ds.length === 0 || ds[ds.length - 1].time !== bucket) {
                ds.push({ time: bucket, value: v.value });
            } else {
                ds[ds.length - 1].value = v.value;
            }
        }
        window._vwapLine.setData(ds);
    }

    // Volume
    const volData = bars.map(b => ({
        time: b.time,
        value: b.volume || 0,
        color: b.close >= b.open ? '#00c85333' : '#ff174433',
    }));
    window._volumeSeries.setData(volData);

    // Markers
    const markers = (window._chartMarkers[sym] || []).slice().sort((a, b) => a.time - b.time);
    window._candleSeries.setMarkers(markers);

    // Opening Range box
    drawOpeningRange(sym, bars);

    // Stop/Target lines for active positions
    drawPositionLines(sym);

    // Volume profile overlay
    drawVolumeProfile(sym);

    // Auto-scroll
    window._chart.timeScale().scrollToRealTime();
}

function drawOpeningRange(sym, bars) {
    // Remove old OR lines
    if (window._orBox) {
        try { window._chart.removeSeries(window._orBox.high); } catch(e) {}
        try { window._chart.removeSeries(window._orBox.low); } catch(e) {}
        window._orBox = null;
    }
    // Find 9:30-9:45 bars (first 15 bars in 1m)
    let orHigh = -Infinity, orLow = Infinity;
    let orStart = null, orEnd = null;
    for (const b of bars) {
        const d = new Date(b.time * 1000);
        const h = d.getUTCHours(), m = d.getUTCMinutes();
        // Bars are in UTC; 9:30 ET = 14:30 UTC (EST) or 13:30 UTC (EDT)
        // Use the first 15 bars heuristic instead
    }
    // Simpler: use first 15 1-min bars
    const rawBars = window._rawBars[sym] || [];
    if (rawBars.length >= 15) {
        const first15 = rawBars.slice(0, 15);
        orHigh = Math.max(...first15.map(b => b.high));
        orLow = Math.min(...first15.map(b => b.low));
        orStart = first15[0].time;
        orEnd = first15[14].time;
    }
    if (orHigh > orLow && orStart && orEnd) {
        // Draw OR as area between two lines
        const orHighLine = window._chart.addLineSeries({
            color: 'rgba(255,214,0,0.3)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
        });
        const orLowLine = window._chart.addLineSeries({
            color: 'rgba(255,214,0,0.3)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
        });
        // Extend OR lines across the full chart
        const allTimes = (window._rawBars[sym] || []);
        if (allTimes.length > 0) {
            const endTime = allTimes[allTimes.length - 1].time;
            orHighLine.setData([
                { time: orStart, value: orHigh },
                { time: endTime, value: orHigh },
            ]);
            orLowLine.setData([
                { time: orStart, value: orLow },
                { time: endTime, value: orLow },
            ]);
        }
        window._orBox = { high: orHighLine, low: orLowLine };
    }
}

function drawPositionLines(sym) {
    // Remove old lines
    if (window._stopLine) {
        try { window._chart.removeSeries(window._stopLine); } catch(e) {}
        window._stopLine = null;
    }
    if (window._targetLine) {
        try { window._chart.removeSeries(window._targetLine); } catch(e) {}
        window._targetLine = null;
    }
    // Find position for this symbol in current positions
    const posRows = document.querySelectorAll('#positions-table tbody tr');
    let stopPrice = null, targetPrice = null;
    posRows.forEach(row => {
        const cells = row.querySelectorAll('td');
        if (cells.length > 0 && cells[0].textContent.trim() === sym) {
            stopPrice = parseFloat(cells[6].textContent.replace('$', ''));
            targetPrice = parseFloat(cells[7].textContent.replace('$', ''));
        }
    });
    const allTimes = window._rawBars[sym] || [];
    if (stopPrice && allTimes.length > 1) {
        const t0 = allTimes[0].time;
        const t1 = allTimes[allTimes.length - 1].time;
        const sl = window._chart.addLineSeries({
            color: '#ff1744',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            priceLineVisible: false,
            lastValueVisible: true,
            crosshairMarkerVisible: false,
        });
        sl.setData([{ time: t0, value: stopPrice }, { time: t1, value: stopPrice }]);
        window._stopLine = sl;
    }
    if (targetPrice && allTimes.length > 1) {
        const t0 = allTimes[0].time;
        const t1 = allTimes[allTimes.length - 1].time;
        const tl = window._chart.addLineSeries({
            color: '#00c853',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            priceLineVisible: false,
            lastValueVisible: true,
            crosshairMarkerVisible: false,
        });
        tl.setData([{ time: t0, value: targetPrice }, { time: t1, value: targetPrice }]);
        window._targetLine = tl;
    }
}

// ══════════════════════════════════════════════════════════════════════
// VOLUME PROFILE (Canvas overlay)
// ══════════════════════════════════════════════════════════════════════
function drawVolumeProfile(sym) {
    const canvas = document.getElementById('volumeProfileOverlay');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = 80, h = 350;
    canvas.width = w;
    canvas.height = h;
    ctx.clearRect(0, 0, w, h);

    const levels = window._volumeProfile[sym];
    if (!levels || levels.length === 0) return;

    // Get visible price range from chart
    const bars = window._rawBars[sym] || [];
    if (bars.length === 0) return;
    let minP = Infinity, maxP = -Infinity;
    for (const b of bars) {
        if (b.low < minP) minP = b.low;
        if (b.high > maxP) maxP = b.high;
    }
    const priceRange = maxP - minP;
    if (priceRange <= 0) return;

    const maxVol = Math.max(...levels.map(l => l.volume));
    if (maxVol <= 0) return;

    for (const lv of levels) {
        const y = h - ((lv.price - minP) / priceRange) * h;
        const barW = (lv.volume / maxVol) * (w - 4);
        const intensity = lv.volume / maxVol;
        const r = Math.round(30 * (1 - intensity));
        const g = Math.round(200 * intensity + 80 * (1 - intensity));
        const b2 = Math.round(220 * intensity + 100 * (1 - intensity));
        ctx.fillStyle = `rgba(${r},${g},${b2},0.6)`;
        ctx.fillRect(w - barW - 2, y - 1, barW, 2);
    }
}

// ══════════════════════════════════════════════════════════════════════
// SYMBOL SELECTION
// ══════════════════════════════════════════════════════════════════════
function selectChartSymbol(sym) {
    window._chartSymbol = sym;
    document.getElementById('chart-symbol-label').textContent = sym;
    // Load bars from REST if we do not have them
    if (!window._rawBars[sym] || window._rawBars[sym].length === 0) {
        fetchBarsForSymbol(sym);
    } else {
        refreshChart();
    }
}

async function fetchBarsForSymbol(sym) {
    try {
        const resp = await fetch(`/api/bars/${sym}?timeframe=1m&limit=390`);
        const data = await resp.json();
        if (Array.isArray(data)) {
            window._rawBars[sym] = data;
        }
    } catch(e) {}
    refreshChart();
}

// ══════════════════════════════════════════════════════════════════════
// RISK GAUGES
// ══════════════════════════════════════════════════════════════════════
function updateRiskGauges(r) {
    const pnl = r.daily_pnl || 0;
    const limit = Math.abs(r.daily_loss_limit || 250);

    // P&L meter
    const gv = document.getElementById('gauge-pnl-value');
    gv.textContent = (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
    gv.style.color = pnl >= 0 ? 'var(--green)' : 'var(--red)';

    // Needle: center at 50%, full left = -2*limit, full right = +2*limit
    const range = limit * 2;
    const pct = Math.min(100, Math.max(0, ((pnl + range / 2) / range) * 100));
    document.getElementById('pnl-meter-needle').style.left = pct + '%';

    // Mark at -$250 position
    const markPct = Math.min(100, Math.max(0, ((-limit + range / 2) / range) * 100));
    document.getElementById('pnl-limit-mark').style.left = markPct + '%';

    // Drawdown gauge (semicircle)
    const dd = Math.abs(r.max_drawdown || 0);
    const ddPct = Math.min(100, (dd / (TOTAL_CAPITAL * 0.1)) * 100); // 10% max scale
    const ddAngle = -90 + (ddPct / 100) * 180; // -90 to +90
    document.getElementById('dd-gauge-needle').setAttribute(
        'transform', `rotate(${ddAngle}, 50, 50)`
    );
    // Color the arc
    const arcLen = (ddPct / 100) * 126; // 126 = approx arc length
    const arcEl = document.getElementById('dd-gauge-arc');
    arcEl.setAttribute('stroke-dasharray', `${arcLen} 126`);
    arcEl.setAttribute('stroke', ddPct > 50 ? 'var(--red)' : ddPct > 25 ? 'var(--yellow)' : 'var(--green)');
    document.getElementById('dd-gauge-value').textContent = (dd / TOTAL_CAPITAL * 100).toFixed(1) + '%';

    // Consecutive losses
    document.getElementById('consec-losses').textContent = r.consecutive_losses || 0;

    // Halted banner
    const banner = document.getElementById('halted-banner');
    if (r.is_halted) {
        banner.style.display = 'block';
        banner.textContent = 'TRADING HALTED: ' + (r.halt_reason || '');
    } else {
        banner.style.display = 'none';
    }
}

const TOTAL_CAPITAL = 12500;

function computeSharpe() {
    if (window._tradePnls.length < 2) {
        document.getElementById('sharpe-value').textContent = '--';
        document.getElementById('sharpe-value').className = 'gauge-value sharpe-value sharpe-red';
        return;
    }
    const pnls = window._tradePnls;
    const mean = pnls.reduce((a, b) => a + b, 0) / pnls.length;
    const variance = pnls.reduce((s, v) => s + (v - mean) ** 2, 0) / (pnls.length - 1);
    const std = Math.sqrt(variance);
    const sharpe = std > 0 ? (mean / std) * Math.sqrt(252) : 0;
    const el = document.getElementById('sharpe-value');
    el.textContent = sharpe.toFixed(2);
    if (sharpe >= 2) {
        el.className = 'gauge-value sharpe-value sharpe-green';
    } else if (sharpe >= 1) {
        el.className = 'gauge-value sharpe-value sharpe-yellow';
    } else {
        el.className = 'gauge-value sharpe-value sharpe-red';
    }
}

// ══════════════════════════════════════════════════════════════════════
// AGENT DECISION TIMELINE
// ══════════════════════════════════════════════════════════════════════
function classifyLogEvent(msg) {
    const m = (msg || '').toUpperCase();
    if (m.includes('SCAN') || m.includes('CHECKING')) return 'scan';
    if (m.includes('SIGNAL') || m.includes('SETUP')) return 'signal';
    if (m.includes('FILLED') || m.includes('ENTRY') || m.includes('POSITION_OPEN') || m.includes('LONG') || m.includes('SHORT')) return 'entry';
    if (m.includes('CLOSED') || m.includes('EXIT') || m.includes('STOP') || m.includes('TARGET')) return 'exit';
    if (m.includes('SKIP') || m.includes('REJECT') || m.includes('NO ')) return 'skip';
    return 'scan';
}

function renderTimeline() {
    const container = document.getElementById('agent-timeline');
    if (!container) return;
    const recent = logEntries.slice(-80);
    let html = '<div class="timeline-line"></div>';
    for (const e of recent) {
        const cls = classifyLogEvent(e.message);
        html += `<div class="timeline-event">
            <div class="timeline-dot dot-${cls}"></div>
            <span class="timeline-time">${e.time || ''}</span>
            <span class="timeline-msg">${truncate(e.message || '', 60)}</span>
            <div class="timeline-detail">${escHtml(e.message || '')}</div>
        </div>`;
    }
    container.innerHTML = html;
    container.scrollTop = container.scrollHeight;
}

function truncate(s, n) {
    return s.length > n ? s.slice(0, n) + '...' : s;
}
function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ══════════════════════════════════════════════════════════════════════
// SPARKLINES (mini Canvas per watchlist item)
// ══════════════════════════════════════════════════════════════════════
function drawSparkline(canvasId, prices) {
    const c = document.getElementById(canvasId);
    if (!c || !prices || prices.length < 2) return;
    const ctx = c.getContext('2d');
    const w = c.width, h = c.height;
    ctx.clearRect(0, 0, w, h);
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const range = max - min || 1;
    ctx.beginPath();
    ctx.strokeStyle = prices[prices.length - 1] >= prices[0] ? '#00c853' : '#ff1744';
    ctx.lineWidth = 1;
    for (let i = 0; i < prices.length; i++) {
        const x = (i / (prices.length - 1)) * w;
        const y = h - ((prices[i] - min) / range) * (h - 2) - 1;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
}

// ══════════════════════════════════════════════════════════════════════
// WEBSOCKET
// ══════════════════════════════════════════════════════════════════════
function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws`);

    ws.onopen = () => {
        wsConnected = true;
        document.getElementById('dot-ws').classList.add('connected');
    };

    ws.onclose = () => {
        wsConnected = false;
        document.getElementById('dot-ws').classList.remove('connected');
        setTimeout(connectWS, 3000);
    };

    ws.onerror = () => {
        ws.close();
    };

    ws.onmessage = (evt) => {
        const data = JSON.parse(evt.data);
        handleMessage(data);
    };
}

function handleMessage(data) {
    if (data.type === 'init') {
        updateHealth(data.health);
        updatePositions(data.positions);
        updateTrades(data.trades);
        updateStrategies(data.strategies);
        if (data.log) {
            logEntries = data.log;
            renderTimeline();
        }
        // Phase 3: Init agent statuses and share size
        if (data.agent_statuses) updateAgentStatuses(data.agent_statuses);
        if (data.share_size) {
            manualShareSize = data.share_size;
            const input = document.getElementById('share-size-input');
            if (input) input.value = data.share_size;
        }
    } else if (data.type === 'update') {
        updateHealth(data.health);
        updatePositions(data.positions);
        if (data.risk) updateRiskGauges(data.risk);
    } else if (data.type === 'tick') {
        updateWatchlistPrice(data.symbol, data.price);
        // Update sparkline data
        if (!window._sparklines[data.symbol]) window._sparklines[data.symbol] = [];
        window._sparklines[data.symbol].push(data.price);
        if (window._sparklines[data.symbol].length > 30) window._sparklines[data.symbol].shift();
    } else if (data.type === 'bar') {
        // 1-min bar from server
        if (!window._rawBars[data.symbol]) window._rawBars[data.symbol] = [];
        const bar = { time: data.time, open: data.open, high: data.high, low: data.low, close: data.close, volume: data.volume || 0 };
        const arr = window._rawBars[data.symbol];
        // Replace or append
        if (arr.length > 0 && arr[arr.length - 1].time === bar.time) {
            arr[arr.length - 1] = bar;
        } else {
            arr.push(bar);
        }
        if (data.symbol === window._chartSymbol) {
            refreshChart();
        }
    } else if (data.type === 'vwap') {
        if (!window._rawVwap[data.symbol]) window._rawVwap[data.symbol] = [];
        const vArr = window._rawVwap[data.symbol];
        const pt = { time: data.time, value: data.value };
        if (vArr.length > 0 && vArr[vArr.length - 1].time === pt.time) {
            vArr[vArr.length - 1] = pt;
        } else {
            vArr.push(pt);
        }
        if (data.symbol === window._chartSymbol) {
            refreshChart();
        }
    } else if (data.type === 'marker') {
        if (!window._chartMarkers[data.symbol]) window._chartMarkers[data.symbol] = [];
        window._chartMarkers[data.symbol].push({
            time: data.time,
            position: data.position,
            color: data.color,
            shape: data.shape,
            text: data.text,
        });
        if (data.symbol === window._chartSymbol) {
            refreshChart();
        }
    } else if (data.type === 'volume_profile') {
        window._volumeProfile[data.symbol] = data.levels || [];
        if (data.symbol === window._chartSymbol) {
            drawVolumeProfile(data.symbol);
        }
    } else if (data.type === 'trade_closed') {
        fetchTrades();
        if (data.trade && typeof data.trade.pnl === 'number') {
            window._tradePnls.push(data.trade.pnl);
            computeSharpe();
        }
    } else if (data.type === 'position_open') {
        // Will be reflected in next update; add entry marker
        if (data.position) {
            const p = data.position;
            selectChartSymbol(p.symbol);
        }
    } else if (data.type === 'agent_status') {
        // Phase 3: Update war room agent avatars
        updateAgentStatuses(data.agents);
    } else if (data.type === 'sound') {
        // Phase 3: Play sound alert
        handleSoundEvent(data.sound);
    }

    // Phase 3: Track latest tick prices for DOM
    if (data.type === 'tick' && data.symbol && data.price) {
        if (!window._lastTickPrices) window._lastTickPrices = {};
        window._lastTickPrices[data.symbol] = data.price;
    }
}

// ══════════════════════════════════════════════════════════════════════
// UPDATE FUNCTIONS (existing, preserved)
// ══════════════════════════════════════════════════════════════════════
function updateHealth(h) {
    const pnl = h.daily_pnl || 0;
    const pnlEl = document.getElementById('pnl-display');
    pnlEl.textContent = '$' + pnl.toFixed(2);
    pnlEl.className = 'pnl-display ' + (pnl > 0 ? 'pnl-positive' : pnl < 0 ? 'pnl-negative' : 'pnl-zero');

    document.getElementById('pos-count').textContent = h.positions || 0;
    document.getElementById('trade-count').textContent = h.trades_today || 0;

    const alpacaDot = document.getElementById('dot-alpaca');
    if (h.streamer_connected) {
        alpacaDot.classList.add('connected');
    } else {
        alpacaDot.classList.remove('connected');
    }

    pnlHistory.push(pnl);
    if (pnlHistory.length > 200) pnlHistory.shift();
    drawPnlChart();
}

function updatePositions(positions) {
    const tbody = document.querySelector('#positions-table tbody');
    const noPos = document.getElementById('no-positions');
    if (!positions || positions.length === 0) {
        tbody.innerHTML = '';
        noPos.style.display = 'block';
        return;
    }
    noPos.style.display = 'none';

    tbody.innerHTML = positions.map(p => {
        const pnlClass = p.unrealized_pnl >= 0 ? 'text-green' : 'text-red';
        const sideClass = p.direction === 'long' ? 'text-green' : 'text-red';
        let holdTime = '';
        if (p.entry_time) {
            const entry = new Date(p.entry_time);
            const now = new Date();
            const secs = Math.floor((now - entry) / 1000);
            if (secs < 60) holdTime = secs + 's';
            else holdTime = Math.floor(secs / 60) + 'm';
        }
        return `<tr onclick="selectChartSymbol('${p.symbol}')" style="cursor:pointer;">
            <td style="font-weight:600;">${p.symbol}</td>
            <td class="${sideClass}">${p.direction.toUpperCase()}</td>
            <td>$${p.entry_price.toFixed(2)}</td>
            <td>$${p.current_price.toFixed(2)}</td>
            <td class="${pnlClass}">$${p.unrealized_pnl.toFixed(2)}</td>
            <td class="${pnlClass}">${(p.unrealized_pnl_pct * 100).toFixed(2)}%</td>
            <td class="text-red">$${p.stop_price.toFixed(2)}</td>
            <td class="text-green">$${p.target_price.toFixed(2)}</td>
            <td class="text-dim">${p.strategy_id}</td>
            <td class="text-dim">${holdTime}</td>
            <td><button class="btn" onclick="event.stopPropagation();closePosition('${p.symbol}')" style="font-size:10px;padding:2px 6px;">CLOSE</button></td>
        </tr>`;
    }).join('');
}

function updateTrades(trades) {
    const tbody = document.querySelector('#trades-table tbody');
    const noTrades = document.getElementById('no-trades');
    if (!trades || trades.length === 0) {
        tbody.innerHTML = '';
        noTrades.style.display = 'block';
        return;
    }
    noTrades.style.display = 'none';

    // Rebuild Sharpe pnls
    window._tradePnls = trades.map(t => t.pnl);
    computeSharpe();

    const recent = trades.slice(-20).reverse();
    tbody.innerHTML = recent.map(t => {
        const pnlClass = t.pnl >= 0 ? 'text-green' : 'text-red';
        const sideClass = t.direction === 'long' ? 'text-green' : 'text-red';
        let exitTime = '';
        if (t.exit_time) {
            const d = new Date(t.exit_time);
            exitTime = d.toLocaleTimeString('en-US', {hour12:false, hour:'2-digit', minute:'2-digit', second:'2-digit'});
        }
        let holdStr = '';
        if (t.hold_seconds < 60) holdStr = t.hold_seconds + 's';
        else holdStr = Math.floor(t.hold_seconds / 60) + 'm';
        return `<tr>
            <td class="text-dim">${exitTime}</td>
            <td style="font-weight:600;">${t.symbol}</td>
            <td class="${sideClass}">${t.direction.toUpperCase()}</td>
            <td>${t.qty}</td>
            <td>$${t.entry_price.toFixed(2)}</td>
            <td>$${t.exit_price.toFixed(2)}</td>
            <td class="${pnlClass}">$${t.pnl.toFixed(2)}</td>
            <td class="${pnlClass}">${(t.pnl_pct * 100).toFixed(2)}%</td>
            <td class="text-dim">${t.strategy_id}</td>
            <td class="text-dim">${holdStr}</td>
            <td class="text-dim">${t.exit_reason}</td>
        </tr>`;
    }).join('');
}

function updateStrategies(strategies) {
    const container = document.getElementById('strategy-cards');
    if (!strategies) return;

    container.innerHTML = strategies.map(s => {
        const statusClass = 'status-' + (s.status || 'idle');
        const disabledClass = s.enabled ? '' : ' disabled';
        const toggleBtn = s.enabled
            ? `<button class="btn" onclick="toggleStrategy('${s.strategy_id}', false)" style="font-size:9px;padding:1px 4px;">OFF</button>`
            : `<button class="btn" onclick="toggleStrategy('${s.strategy_id}', true)" style="font-size:9px;padding:1px 4px;">ON</button>`;
        const pnlStr = s.pnl_today !== 0 ? (s.pnl_today >= 0 ? '+' : '') + '$' + s.pnl_today.toFixed(2) : '';
        const pnlClass = s.pnl_today >= 0 ? 'text-green' : 'text-red';
        return `<div class="strategy-card${disabledClass}">
            <div>
                <span class="strategy-name">${s.name || s.strategy_id}</span>
                <span class="strategy-status ${statusClass}">${s.status || 'idle'}</span>
                <span class="text-dim" style="font-size:10px;margin-left:8px;">
                    T:${s.trades_today} W:${s.wins_today} L:${s.losses_today}
                </span>
                <span class="${pnlClass}" style="font-size:10px;margin-left:4px;">${pnlStr}</span>
            </div>
            <div>${toggleBtn}</div>
        </div>`;
    }).join('');
}

function drawPnlChart() {
    const line = document.getElementById('pnl-line');
    if (!pnlHistory.length) return;

    const maxAbs = Math.max(1, ...pnlHistory.map(Math.abs));
    const w = 600;
    const h = 100;
    const mid = h / 2;

    const points = pnlHistory.map((v, i) => {
        const x = (i / Math.max(1, pnlHistory.length - 1)) * w;
        const y = mid - (v / maxAbs) * (mid - 5);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');

    line.setAttribute('points', points);
    line.setAttribute('stroke', pnlHistory[pnlHistory.length - 1] >= 0 ? '#00c853' : '#ff1744');
}

function updateWatchlistPrice(symbol, price) {
    const el = document.getElementById('wl-' + symbol);
    if (!el) return;
    const priceEl = el.querySelector('.wl-price');
    if (priceEl) priceEl.textContent = '$' + price.toFixed(2);

    // VWAP flash detection
    const vwapEl = el.querySelector('.wl-vwap');
    if (vwapEl) {
        const vwapVal = parseFloat(vwapEl.dataset.vwap || '0');
        if (vwapVal > 0) {
            const side = price > vwapVal ? 'above' : 'below';
            const prev = window._prevVwapSide[symbol];
            if (prev && prev !== side) {
                el.classList.add('vwap-flash');
                setTimeout(() => el.classList.remove('vwap-flash'), 600);
            }
            window._prevVwapSide[symbol] = side;
        }
    }
}

// ── EOD Timer ──
function updateEodTimer() {
    const now = new Date();
    const target = new Date(now);
    target.setHours(15, 50, 0, 0);
    const diff = target - now;
    if (diff <= 0) {
        document.getElementById('eod-timer').textContent = '00:00:00';
        document.getElementById('eod-timer').style.color = 'var(--red)';
        return;
    }
    const h = Math.floor(diff / 3600000);
    const m = Math.floor((diff % 3600000) / 60000);
    const s = Math.floor((diff % 60000) / 1000);
    document.getElementById('eod-timer').textContent =
        String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
}

// ══════════════════════════════════════════════════════════════════════
// API CALLS
// ══════════════════════════════════════════════════════════════════════
async function startTrading() {
    await fetch('/api/start', {method: 'POST'});
}
async function stopTrading() {
    await fetch('/api/stop', {method: 'POST'});
}
async function closePosition(symbol) {
    await fetch('/api/close/' + symbol, {method: 'POST'});
}
async function closeAll() {
    if (confirm('Close ALL positions?')) {
        await fetch('/api/close_all', {method: 'POST'});
    }
}
async function toggleStrategy(id, enable) {
    const action = enable ? 'enable' : 'disable';
    await fetch(`/api/strategies/${id}/${action}`, {method: 'POST'});
    fetchStrategies();
}
async function fetchTrades() {
    try {
        const resp = await fetch('/api/trades');
        const trades = await resp.json();
        updateTrades(trades);
    } catch(e) {}
}
async function fetchStrategies() {
    try {
        const resp = await fetch('/api/strategies');
        const data = await resp.json();
        updateStrategies(data);
    } catch(e) {}
}
async function fetchWatchlist() {
    try {
        const resp = await fetch('/api/watchlist');
        const items = await resp.json();
        const container = document.getElementById('watchlist');
        container.innerHTML = items.map(w => {
            const vwapClass = w.vwap_deviation_pct > 1 ? 'text-green'
                : w.vwap_deviation_pct < -1 ? 'text-red' : 'text-dim';
            const posClass = w.in_position ? ' in-position' : '';
            const sparkId = 'spark-' + w.symbol;
            // Store VWAP in data attribute for flash detection
            return `<div class="watchlist-item${posClass}" id="wl-${w.symbol}" onclick="selectChartSymbol('${w.symbol}')">
                <span class="wl-sym">${w.symbol}</span>
                <canvas class="wl-sparkline" id="${sparkId}" width="50" height="16"></canvas>
                <span class="wl-price">$${w.price.toFixed(2)}</span>
                <span class="wl-vwap ${vwapClass}" data-vwap="${w.vwap || 0}">${w.vwap_deviation_pct > 0 ? '+' : ''}${w.vwap_deviation_pct.toFixed(1)}%</span>
            </div>`;
        }).join('');
        // Draw sparklines
        for (const w of items) {
            const prices = window._sparklines[w.symbol];
            if (prices && prices.length >= 2) {
                drawSparkline('spark-' + w.symbol, prices);
            }
        }
    } catch(e) {}
}
async function fetchLog() {
    try {
        const resp = await fetch('/api/log');
        logEntries = await resp.json();
        renderTimeline();
    } catch(e) {}
}

// ── Win rate ──
function updateWinRate() {
    fetch('/api/trades').then(r => r.json()).then(trades => {
        if (trades.length === 0) {
            document.getElementById('win-rate').textContent = '--';
            return;
        }
        const wins = trades.filter(t => t.pnl >= 0).length;
        const wr = (wins / trades.length * 100).toFixed(0);
        document.getElementById('win-rate').textContent = wr + '%';
    }).catch(() => {});
}

// ══════════════════════════════════════════════════════════════════════
// PHASE 3: SOUND ALERTS (Web Audio API)
// ══════════════════════════════════════════════════════════════════════
let soundMuted = false;
let audioCtx = null;

function getAudioCtx() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    return audioCtx;
}

function playTone(freq, duration, type, rampTo) {
    if (soundMuted) return;
    try {
        const ctx = getAudioCtx();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.value = freq;
        osc.type = type || 'sine';
        gain.gain.value = 0.3;
        if (rampTo) {
            osc.frequency.linearRampToValueAtTime(rampTo, ctx.currentTime + duration);
        }
        osc.start();
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
        osc.stop(ctx.currentTime + duration);
    } catch(e) {}
}

function playSoundEntry() {
    // Ascending tone 440->880 Hz, 200ms
    playTone(440, 0.2, 'sine', 880);
}
function playSoundExit() {
    // Descending tone 880->440 Hz, 200ms
    playTone(880, 0.2, 'sine', 440);
}
function playSoundStop() {
    // Two short beeps 660Hz, 100ms each
    playTone(660, 0.1, 'square');
    setTimeout(() => playTone(660, 0.1, 'square'), 150);
}
function playSoundKillswitch() {
    // Long low tone 220Hz, 500ms
    playTone(220, 0.5, 'sawtooth');
}
function playSoundAlert() {
    playTone(550, 0.15, 'sine', 700);
}

function handleSoundEvent(soundName) {
    switch(soundName) {
        case 'entry': playSoundEntry(); break;
        case 'exit': playSoundExit(); break;
        case 'stop': playSoundStop(); break;
        case 'killswitch': playSoundKillswitch(); break;
        case 'alert': playSoundAlert(); break;
    }
}

function toggleSound() {
    soundMuted = !soundMuted;
    const el = document.getElementById('sound-toggle');
    if (soundMuted) {
        el.innerHTML = '&#x1f507;';
        el.classList.add('muted');
    } else {
        el.innerHTML = '&#x1f50a;';
        el.classList.remove('muted');
    }
}

// ══════════════════════════════════════════════════════════════════════
// PHASE 3: AGENT WAR ROOM
// ══════════════════════════════════════════════════════════════════════
function updateAgentStatuses(agents) {
    if (!agents) return;
    const mapping = {
        scout: 'agent-scout-status',
        analyst: 'agent-analyst-status',
        executor: 'agent-executor-status',
        risk: 'agent-risk-status',
    };
    for (const [agent, elId] of Object.entries(mapping)) {
        const status = agents[agent] || 'idle';
        const el = document.getElementById(elId);
        if (!el) continue;
        el.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        el.className = 'agent-status-badge ' + status;
    }
}

// ══════════════════════════════════════════════════════════════════════
// PHASE 3: ONE-CLICK TRADING
// ══════════════════════════════════════════════════════════════════════
let manualShareSize = 100;

function getShareSize() {
    const input = document.getElementById('share-size-input');
    return parseInt(input.value) || manualShareSize;
}

function adjustShares(delta) {
    const input = document.getElementById('share-size-input');
    let val = parseInt(input.value) || 100;
    val = Math.max(1, val + delta);
    input.value = val;
    manualShareSize = val;
}

async function quickBuy() {
    const sym = window._chartSymbol;
    const qty = getShareSize();
    try {
        const resp = await fetch('/api/order', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({symbol: sym, side: 'buy', qty: qty, order_type: 'market'}),
        });
        const data = await resp.json();
        if (data.error) {
            console.warn('Buy failed:', data.error);
        }
    } catch(e) { console.error('Buy error:', e); }
}

async function quickSell() {
    const sym = window._chartSymbol;
    const qty = getShareSize();
    try {
        const resp = await fetch('/api/order', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({symbol: sym, side: 'sell', qty: qty, order_type: 'market'}),
        });
        const data = await resp.json();
        if (data.error) {
            console.warn('Sell failed:', data.error);
        }
    } catch(e) { console.error('Sell error:', e); }
}

async function flattenSelected() {
    const sym = window._chartSymbol;
    try {
        await fetch('/api/flatten/' + sym, {method: 'POST'});
    } catch(e) { console.error('Flatten error:', e); }
}

async function flattenAll() {
    if (confirm('FLATTEN ALL positions?')) {
        await fetch('/api/close_all', {method: 'POST'});
    }
}

async function cancelLastOrder() {
    try {
        await fetch('/api/cancel_last', {method: 'POST'});
    } catch(e) { console.error('Cancel error:', e); }
}

// ══════════════════════════════════════════════════════════════════════
// PHASE 3: DOM LADDER
// ══════════════════════════════════════════════════════════════════════
let domThrottleTimer = null;
let lastDomSymbol = '';

function updateDOM(data) {
    if (!data || !data.levels || data.levels.length === 0) return;

    const body = document.getElementById('domBody');
    if (!body) return;

    document.getElementById('domSymbolLabel').textContent = data.symbol || '';

    const maxVol = Math.max(1, ...data.levels.map(l => l.volume));
    const pos = data.position;
    const entryPrice = pos ? pos.entry_price : null;
    const stopPrice = pos ? pos.stop_price : null;
    const targetPrice = pos ? pos.target_price : null;
    const direction = pos ? pos.direction : null;
    const currentPrice = data.current_price;

    let html = '';
    for (const lv of data.levels) {
        let rowClass = 'dom-row';
        let label = '';
        let labelColor = '';

        // Current price row
        if (currentPrice > 0 && Math.abs(lv.price - currentPrice) < 0.005) {
            rowClass += ' current-price';
        }

        // Stop/Target/Entry markers
        if (stopPrice && Math.abs(lv.price - stopPrice) < 0.005) {
            rowClass += ' stop-level';
            label = 'STOP';
            labelColor = 'color:var(--red);';
        } else if (targetPrice && Math.abs(lv.price - targetPrice) < 0.005) {
            rowClass += ' target-level';
            label = 'TGT';
            labelColor = 'color:var(--green);';
        } else if (entryPrice && Math.abs(lv.price - entryPrice) < 0.005) {
            rowClass += ' entry-level';
            label = 'ENTRY';
            labelColor = 'color:var(--blue);';
        } else if (entryPrice) {
            // Color above/below entry
            if (direction === 'short') {
                rowClass += lv.price > entryPrice ? ' above-entry' : ' below-entry';
            } else {
                rowClass += lv.price > entryPrice ? ' below-entry' : ' above-entry';
            }
        }

        const volPct = maxVol > 0 ? (lv.volume / maxVol) * 100 : 0;
        const volColor = volPct > 50 ? 'var(--blue)' : 'rgba(41,121,255,0.4)';

        html += '<div class="' + rowClass + '" onclick="domClickPrice(' + lv.price + ')">' +
            '<span class="dom-price">$' + lv.price.toFixed(2) + '</span>' +
            '<span class="dom-volume">' + formatVolume(lv.volume) + '</span>' +
            '<div class="dom-vol-bar"><div class="dom-vol-fill" style="width:' + volPct.toFixed(1) + '%;background:' + volColor + ';"></div></div>' +
            '<span class="dom-level-label" style="' + labelColor + '">' + label + '</span>' +
            '</div>';
    }
    body.innerHTML = html;
}

function formatVolume(v) {
    if (v >= 1000000) return (v / 1000000).toFixed(1) + 'M';
    if (v >= 1000) return (v / 1000).toFixed(1) + 'K';
    return v.toString();
}

async function domClickPrice(price) {
    // Click a DOM level to place a limit order at that price
    const sym = window._chartSymbol;
    const qty = getShareSize();
    const currentPrice = window._lastTickPrices ? (window._lastTickPrices[sym] || 0) : 0;
    // If clicking below current, it's a buy limit; above = sell limit
    const side = price < currentPrice ? 'buy' : 'sell';
    try {
        await fetch('/api/order', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({symbol: sym, side: side, qty: qty, order_type: 'limit', limit_price: price}),
        });
    } catch(e) { console.error('DOM order error:', e); }
}

function refreshDOM() {
    const sym = window._chartSymbol;
    if (domThrottleTimer && sym === lastDomSymbol) return;
    lastDomSymbol = sym;
    domThrottleTimer = setTimeout(() => { domThrottleTimer = null; }, 250);
    fetch('/api/dom/' + sym).then(r => r.json()).then(updateDOM).catch(() => {});
}

// ══════════════════════════════════════════════════════════════════════
// PHASE 3: HOTKEY SUPPORT
// ══════════════════════════════════════════════════════════════════════
window._selectedPositionIndex = 0;
window._lastTickPrices = {};

document.addEventListener('keydown', function(e) {
    // Don't intercept when typing in inputs
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;

    switch(e.key.toLowerCase()) {
        case 'b':
            e.preventDefault();
            quickBuy();
            break;
        case 's':
            e.preventDefault();
            quickSell();
            break;
        case 'f':
            e.preventDefault();
            flattenSelected();
            break;
        case 'x':
            if (e.shiftKey) {
                e.preventDefault();
                flattenAll();
            }
            break;
        case 'escape':
            e.preventDefault();
            cancelLastOrder();
            break;
        case '1': case '2': case '3': case '4': case '5':
            e.preventDefault();
            selectPositionByIndex(parseInt(e.key) - 1);
            break;
        case '+': case '=':
            e.preventDefault();
            adjustShares(100);
            break;
        case '-':
            e.preventDefault();
            adjustShares(-100);
            break;
        case ' ':
            e.preventDefault();
            toggleDomFocus();
            break;
    }
});

function selectPositionByIndex(idx) {
    // Select a position by its index in the positions table
    const rows = document.querySelectorAll('#positions-table tbody tr');
    if (idx >= 0 && idx < rows.length) {
        const cells = rows[idx].querySelectorAll('td');
        if (cells.length > 0) {
            const sym = cells[0].textContent.trim();
            selectChartSymbol(sym);
            window._selectedPositionIndex = idx;
        }
    }
}

let domFocused = false;
function toggleDomFocus() {
    domFocused = !domFocused;
    const dom = document.getElementById('domLadder');
    const chart = document.getElementById('mainChart');
    if (domFocused) {
        dom.style.border = '2px solid var(--blue)';
        chart.style.border = '1px solid var(--border)';
    } else {
        dom.style.border = '1px solid var(--border)';
        chart.style.border = '2px solid var(--blue)';
    }
}

// ══════════════════════════════════════════════════════════════════════
// PHASE 3: DRAGGABLE STOP/TARGET ON CHART
// ══════════════════════════════════════════════════════════════════════
let isDraggingStop = false;
let isDraggingTarget = false;
let dragStartY = 0;
let dragStartPrice = 0;

function setupDragHandlers() {
    const container = document.getElementById('mainChart');
    if (!container) return;

    container.addEventListener('mousedown', function(e) {
        const sym = window._chartSymbol;
        const bars = window._rawBars[sym] || [];
        if (bars.length === 0) return;

        // Check if clicking near stop or target line
        const rect = container.getBoundingClientRect();
        const y = e.clientY - rect.top;
        const chartHeight = rect.height;

        // Get price range from visible bars
        let minP = Infinity, maxP = -Infinity;
        for (const b of bars) {
            if (b.low < minP) minP = b.low;
            if (b.high > maxP) maxP = b.high;
        }
        const priceRange = maxP - minP;
        if (priceRange <= 0) return;

        // Convert y to price
        const clickPrice = maxP - (y / chartHeight) * priceRange;

        // Find position stop/target
        const posRows = document.querySelectorAll('#positions-table tbody tr');
        let stopPrice = null, targetPrice = null;
        posRows.forEach(row => {
            const cells = row.querySelectorAll('td');
            if (cells.length > 0 && cells[0].textContent.trim() === sym) {
                stopPrice = parseFloat(cells[6].textContent.replace('$', ''));
                targetPrice = parseFloat(cells[7].textContent.replace('$', ''));
            }
        });

        const threshold = priceRange * 0.02; // 2% of visible range

        if (stopPrice && Math.abs(clickPrice - stopPrice) < threshold) {
            isDraggingStop = true;
            dragStartY = e.clientY;
            dragStartPrice = stopPrice;
            container.style.cursor = 'ns-resize';
            e.preventDefault();
        } else if (targetPrice && Math.abs(clickPrice - targetPrice) < threshold) {
            isDraggingTarget = true;
            dragStartY = e.clientY;
            dragStartPrice = targetPrice;
            container.style.cursor = 'ns-resize';
            e.preventDefault();
        }
    });

    document.addEventListener('mousemove', function(e) {
        if (!isDraggingStop && !isDraggingTarget) return;
        e.preventDefault();

        const container = document.getElementById('mainChart');
        const rect = container.getBoundingClientRect();
        const sym = window._chartSymbol;
        const bars = window._rawBars[sym] || [];

        let minP = Infinity, maxP = -Infinity;
        for (const b of bars) {
            if (b.low < minP) minP = b.low;
            if (b.high > maxP) maxP = b.high;
        }
        const priceRange = maxP - minP;
        if (priceRange <= 0) return;

        const pixelDelta = e.clientY - dragStartY;
        const priceDelta = -(pixelDelta / rect.height) * priceRange;
        const newPrice = Math.round((dragStartPrice + priceDelta) * 100) / 100;

        // Show tooltip
        showDragTooltip(e.clientX, e.clientY, newPrice, isDraggingStop ? 'STOP' : 'TARGET');
    });

    document.addEventListener('mouseup', function(e) {
        hideDragTooltip();
        const container = document.getElementById('mainChart');
        if (container) container.style.cursor = '';

        if (isDraggingStop || isDraggingTarget) {
            const rect = document.getElementById('mainChart').getBoundingClientRect();
            const sym = window._chartSymbol;
            const bars = window._rawBars[sym] || [];

            let minP = Infinity, maxP = -Infinity;
            for (const b of bars) {
                if (b.low < minP) minP = b.low;
                if (b.high > maxP) maxP = b.high;
            }
            const priceRange = maxP - minP;
            if (priceRange > 0) {
                const pixelDelta = e.clientY - dragStartY;
                const priceDelta = -(pixelDelta / rect.height) * priceRange;
                const newPrice = Math.round((dragStartPrice + priceDelta) * 100) / 100;

                if (isDraggingStop) {
                    updateStopPrice(sym, newPrice);
                } else {
                    updateTargetPrice(sym, newPrice);
                }
            }

            isDraggingStop = false;
            isDraggingTarget = false;
        }
    });
}

function showDragTooltip(x, y, price, label) {
    let tip = document.getElementById('drag-tooltip');
    if (!tip) {
        tip = document.createElement('div');
        tip.id = 'drag-tooltip';
        tip.style.cssText = 'position:fixed;z-index:9999;background:#222;color:#fff;padding:4px 8px;border-radius:3px;font-family:var(--mono);font-size:11px;pointer-events:none;';
        document.body.appendChild(tip);
    }
    tip.textContent = label + ': $' + price.toFixed(2);
    tip.style.left = (x + 15) + 'px';
    tip.style.top = (y - 10) + 'px';
    tip.style.display = 'block';
}

function hideDragTooltip() {
    const tip = document.getElementById('drag-tooltip');
    if (tip) tip.style.display = 'none';
}

async function updateStopPrice(sym, newStop) {
    try {
        await fetch('/api/positions/' + sym + '/stop', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({new_stop: newStop}),
        });
    } catch(e) { console.error('Update stop error:', e); }
}

async function updateTargetPrice(sym, newTarget) {
    try {
        await fetch('/api/positions/' + sym + '/target', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({new_target: newTarget}),
        });
    } catch(e) { console.error('Update target error:', e); }
}

// ══════════════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    fetchBarsForSymbol(window._chartSymbol);
    // Phase 3: Setup drag handlers for stop/target lines
    setupDragHandlers();
});
connectWS();
setInterval(updateEodTimer, 1000);
setInterval(fetchWatchlist, 10000);
setInterval(fetchLog, 5000);
setInterval(updateWinRate, 15000);
// Phase 3: Refresh DOM ladder every 500ms
setInterval(refreshDOM, 500);
// Redraw sparklines periodically
setInterval(() => {
    const items = document.querySelectorAll('.wl-sparkline');
    items.forEach(c => {
        const sym = c.id.replace('spark-', '');
        const prices = window._sparklines[sym];
        if (prices && prices.length >= 2) drawSparkline(c.id, prices);
    });
}, 5000);
fetchWatchlist();
fetchLog();
updateEodTimer();
</script>
</body>
</html>"""


@app.get('/', response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the dashboard."""
    return HTMLResponse(DASHBOARD_HTML)


# =============================================================================
# SECTION 11: MAIN
# =============================================================================

def main() -> None:
    """Entry point."""
    _load_env_file()

    host = '0.0.0.0' if os.environ.get('INTRADAY_BIND_ALL') else '127.0.0.1'
    logger.info(
        "Starting Rudra Intraday Scalper on %s:%d",
        host, INTRADAY_PORT,
    )
    uvicorn.run(
        app,
        host=host,
        port=INTRADAY_PORT,
        log_level='info',
        access_log=False,
    )


if __name__ == '__main__':
    main()
