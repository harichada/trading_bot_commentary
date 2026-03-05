"""Alpaca broker adapter — thin wrapper around existing functions in gap_fade_app.

This adapter does NOT re-implement any Alpaca logic.  It imports the
existing functions from ``gap_fade_app`` and delegates, conforming to
the :class:`AbstractBroker` interface.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from brokers.base import AbstractBroker, BrokerConfig, OrderResult, TickStreamer
from brokers.market_hours import MarketHoursPolicy, USEquityHours

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import helper — gap_fade_app is large; import on first use
# ---------------------------------------------------------------------------

_gf = None  # module reference, populated lazily


def _gap_fade():
    """Lazy-import gap_fade_app to avoid circular imports at module load."""
    global _gf
    if _gf is None:
        import gap_fade_app as gf
        _gf = gf
    return _gf


# ---------------------------------------------------------------------------
# Tick streamer adapter
# ---------------------------------------------------------------------------

class AlpacaTickStreamerAdapter(TickStreamer):
    """Wraps :class:`gap_fade_app.AlpacaTickStreamer`."""

    def __init__(self, symbols: List[str], on_tick: Optional[Callable] = None):
        gf = _gap_fade()
        self._inner = gf.AlpacaTickStreamer(symbols, on_tick=on_tick)

    async def start(self) -> None:
        await self._inner.start()

    async def stop(self) -> None:
        await self._inner.stop()

    async def add_symbols(self, symbols: List[str]) -> None:
        await self._inner.add_symbols(symbols)

    async def remove_symbols(self, symbols: List[str]) -> None:
        await self._inner.remove_symbols(symbols)

    @property
    def symbols(self) -> List[str]:
        return self._inner.symbols if hasattr(self._inner, 'symbols') else []

    @property
    def latest_prices(self) -> Dict[str, float]:
        return self._inner.latest_prices

    @property
    def connected(self) -> bool:
        return self._inner.connected


# ---------------------------------------------------------------------------
# Alpaca broker adapter
# ---------------------------------------------------------------------------

class AlpacaBrokerAdapter(AbstractBroker):
    """Delegates to existing ``alpaca_*`` functions in gap_fade_app.py."""

    def __init__(self, config: Optional[BrokerConfig] = None):
        if config is None:
            config = BrokerConfig(broker_id='alpaca', environment='paper')
        super().__init__(config)
        self._market_hours = USEquityHours()

    # -- Connection / Account ------------------------------------------------

    def is_configured(self) -> bool:
        gf = _gap_fade()
        return gf._get_alpaca_config() is not None

    async def get_account(self) -> Optional[Dict]:
        gf = _gap_fade()
        return await asyncio.to_thread(gf.alpaca_get_account)

    async def get_positions(self) -> List[Dict]:
        gf = _gap_fade()
        raw = await asyncio.to_thread(gf.alpaca_get_positions)
        # Normalize into the canonical format
        positions = []
        for p in raw:
            positions.append({
                'symbol': p.get('symbol', ''),
                'qty': float(p.get('qty', 0)),
                'avg_entry_price': float(p.get('avg_entry_price', 0)),
                'side': p.get('side', 'long'),
                'broker_id': 'alpaca',
                'raw': p,  # preserve full Alpaca response
            })
        return positions

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
        gf = _gap_fade()
        # Alpaca uses int qty for equities
        int_qty = int(qty)
        result = await gf.alpaca_submit_and_confirm(
            symbol, int_qty, side,
            order_type=order_type,
            limit_price=limit_price,
            timeout_sec=timeout_sec,
        )
        # Convert gap_fade_app.OrderResult → brokers.base.OrderResult
        return OrderResult(
            order_id=result.order_id,
            status=result.status,
            filled_qty=float(result.filled_qty),
            filled_avg_price=result.filled_avg_price,
            symbol=result.symbol,
            side=result.side,
            error=result.error,
        )

    async def place_stop_order(
        self,
        symbol: str,
        qty: float,
        stop_price: float,
        limit_offset_pct: float = 0.003,
        direction: str = 'short',
    ) -> Dict:
        gf = _gap_fade()
        int_qty = int(qty)
        return await asyncio.to_thread(
            gf.alpaca_place_stop_order,
            symbol, int_qty, stop_price, limit_offset_pct, direction,
        )

    async def cancel_order(self, order_id: str) -> bool:
        gf = _gap_fade()
        return await asyncio.to_thread(gf.alpaca_cancel_order, order_id)

    async def get_order(self, order_id: str) -> Optional[Dict]:
        gf = _gap_fade()
        return await asyncio.to_thread(gf.alpaca_get_order, order_id)

    async def replace_stop_order(
        self,
        old_order_id: str,
        symbol: str,
        qty: float,
        new_stop_price: float,
        limit_offset_pct: float = 0.003,
    ) -> Dict:
        gf = _gap_fade()
        int_qty = int(qty)
        return await gf.alpaca_replace_stop_order(
            old_order_id, symbol, int_qty, new_stop_price, limit_offset_pct,
        )

    # -- Market Data ---------------------------------------------------------

    def create_tick_streamer(
        self,
        symbols: List[str],
        on_tick: Optional[Callable] = None,
    ) -> TickStreamer:
        return AlpacaTickStreamerAdapter(symbols, on_tick=on_tick)

    async def fetch_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1Day',
    ) -> Optional[pd.DataFrame]:
        gf = _gap_fade()
        return await asyncio.to_thread(
            gf.fetch_alpaca_bars, symbol, start_date, end_date, interval,
        )

    async def fetch_bars_multi(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = '1Day',
    ) -> Dict[str, pd.DataFrame]:
        gf = _gap_fade()
        return await asyncio.to_thread(
            gf.fetch_alpaca_bars_multi, symbols, start_date, end_date, interval,
        )

    # -- Symbol Info ----------------------------------------------------------

    async def check_tradeable(self, symbol: str) -> Tuple[bool, bool]:
        gf = _gap_fade()
        return await asyncio.to_thread(gf.alpaca_check_shortable, symbol)

    async def get_tradeable_symbols(
        self,
        min_price: float = 1.0,
        max_price: float = 0.0,
    ) -> List[str]:
        gf = _gap_fade()
        return await asyncio.to_thread(gf.fetch_alpaca_assets, min_price)

    # -- Snapshot Prices -----------------------------------------------------

    async def get_snapshot_prices(self, symbols: List[str]) -> Dict[str, float]:
        gf = _gap_fade()
        snaps = await asyncio.to_thread(gf.fetch_alpaca_snapshots, symbols)
        prices: Dict[str, float] = {}
        for sym, snap in snaps.items():
            lt = snap.get('latestTrade', {})
            if lt.get('p'):
                prices[sym] = lt['p']
        return prices

    # -- Market Hours --------------------------------------------------------

    def get_market_hours(self) -> MarketHoursPolicy:
        return self._market_hours
