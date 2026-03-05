"""Abstract broker interface for multi-broker trading.

Defines the contract that every broker adapter must implement,
plus shared data classes (OrderResult, BrokerConfig, TickStreamer).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import pandas as pd

    from brokers.market_hours import MarketHoursPolicy


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BrokerConfig:
    """Broker connection configuration."""
    broker_id: str = ''            # e.g. 'alpaca', 'oanda'
    api_key: str = ''
    api_secret: str = ''
    base_url: str = ''
    data_url: str = ''
    account_id: str = ''           # OANDA needs explicit account ID
    environment: str = 'paper'     # 'paper' or 'live'
    extra: Dict = field(default_factory=dict)


@dataclass
class OrderResult:
    """Structured result from order submission + fill verification.

    Mirrors the existing OrderResult in gap_fade_app.py so the adapter
    can return it directly without conversion.
    """
    order_id: str = ''
    status: str = ''               # 'filled', 'partially_filled', 'rejected', 'cancelled', 'error', 'timeout'
    filled_qty: float = 0
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


# ---------------------------------------------------------------------------
# Tick streamer ABC
# ---------------------------------------------------------------------------

class TickStreamer(ABC):
    """Real-time price stream interface."""

    @abstractmethod
    async def start(self) -> None:
        """Connect and begin streaming."""

    @abstractmethod
    async def stop(self) -> None:
        """Graceful shutdown."""

    @abstractmethod
    async def add_symbols(self, symbols: List[str]) -> None:
        """Subscribe to additional symbols mid-stream."""

    @abstractmethod
    async def remove_symbols(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols mid-stream."""

    @property
    @abstractmethod
    def latest_prices(self) -> Dict[str, float]:
        """Current best-known price per symbol."""

    @property
    @abstractmethod
    def symbols(self) -> List[str]:
        """Currently subscribed symbols."""

    @property
    @abstractmethod
    def connected(self) -> bool:
        """Whether the stream is currently connected."""


# ---------------------------------------------------------------------------
# Abstract broker
# ---------------------------------------------------------------------------

class AbstractBroker(ABC):
    """Unified broker interface.

    Every concrete adapter (Alpaca, OANDA, …) must implement these methods.
    Async methods that wrap sync HTTP calls should use
    ``await asyncio.to_thread(...)`` internally.
    """

    def __init__(self, config: BrokerConfig) -> None:
        self.config = config

    @property
    def broker_id(self) -> str:
        return self.config.broker_id

    # -- Connection / Account ------------------------------------------------

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if required env vars / credentials are present."""

    @abstractmethod
    async def get_account(self) -> Optional[Dict]:
        """Fetch account info (equity, buying power, cash, etc.)."""

    @abstractmethod
    async def get_positions(self) -> List[Dict]:
        """Fetch open positions.

        Each dict must include at minimum:
            symbol, qty (float), avg_entry_price, side, broker_id
        """

    # -- Order Execution -----------------------------------------------------

    @abstractmethod
    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        timeout_sec: float = 30.0,
    ) -> OrderResult:
        """Submit order and wait for fill / timeout / rejection."""

    @abstractmethod
    async def place_stop_order(
        self,
        symbol: str,
        qty: float,
        stop_price: float,
        limit_offset_pct: float = 0.003,
        direction: str = 'short',
    ) -> Dict:
        """Place a broker-side stop-limit (or stop-loss) order.

        Returns dict with at least ``{'id': order_id}`` on success
        or ``{'error': msg}`` on failure.
        """

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.  Returns True on success."""

    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Dict]:
        """Fetch order status by ID.

        Returns dict with at least ``{'status': ...}`` or None if not found.
        """

    @abstractmethod
    async def replace_stop_order(
        self,
        old_order_id: str,
        symbol: str,
        qty: float,
        new_stop_price: float,
        limit_offset_pct: float = 0.003,
    ) -> Dict:
        """Cancel old stop and place a new one at *new_stop_price*."""

    # -- Market Data ---------------------------------------------------------

    @abstractmethod
    def create_tick_streamer(
        self,
        symbols: List[str],
        on_tick: Optional[Callable] = None,
    ) -> TickStreamer:
        """Create (but don't start) a real-time tick streamer."""

    @abstractmethod
    async def fetch_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1Day',
    ) -> Optional['pd.DataFrame']:
        """Fetch historical OHLCV bars for a single symbol."""

    @abstractmethod
    async def fetch_bars_multi(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = '1Day',
    ) -> Dict[str, 'pd.DataFrame']:
        """Fetch historical bars for multiple symbols."""

    # -- Symbol Info ----------------------------------------------------------

    @abstractmethod
    async def check_tradeable(self, symbol: str) -> Tuple[bool, bool]:
        """Check if *symbol* can be traded.

        Returns (shortable, easy_to_borrow).
        For forex brokers both should be True (bidirectional).
        """

    @abstractmethod
    async def get_tradeable_symbols(
        self,
        min_price: float = 1.0,
        max_price: float = 0.0,
    ) -> List[str]:
        """Fetch the tradeable symbol universe."""

    # -- Snapshot Prices -----------------------------------------------------

    @abstractmethod
    async def get_snapshot_prices(self, symbols: List[str]) -> Dict[str, float]:
        """REST price fallback — fetch latest prices for *symbols*.

        Returns ``{symbol: price}`` for all symbols that have a valid price.
        Used when the WebSocket streamer is unavailable.
        """

    # -- Market Hours --------------------------------------------------------

    @abstractmethod
    def get_market_hours(self) -> 'MarketHoursPolicy':
        """Return the market-hours policy for this broker's asset class."""
