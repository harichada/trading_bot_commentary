"""IBKR real-time tick streamer using ib_insync market data.

Subscribes to real-time trades via ``reqMktData`` and provides
latest prices through the TickStreamer interface.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional

from ib_insync import Ticker

from brokers.base import TickStreamer
from brokers.ibkr_connection import IBKRConnectionManager

logger = logging.getLogger(__name__)


class IBKRTickStreamer(TickStreamer):
    """Real-time price stream via IB Gateway market data."""

    THROTTLE_SEC = 0.25  # Min interval between price updates per symbol
    MAX_SYMBOLS = 100    # IBKR market data line limit

    def __init__(
        self,
        symbols: List[str],
        conn: IBKRConnectionManager,
        on_tick: Optional[Callable] = None,
    ) -> None:
        self._symbols = list(symbols)
        self._conn = conn
        self._on_tick = on_tick
        self._latest_prices: Dict[str, float] = {}
        self._connected = False
        self._tickers: Dict[str, Ticker] = {}
        self._last_push: Dict[str, float] = {}
        self._task: Optional[asyncio.Task] = None

    @property
    def latest_prices(self) -> Dict[str, float]:
        return dict(self._latest_prices)

    @property
    def symbols(self) -> List[str]:
        return list(self._symbols)

    @property
    def connected(self) -> bool:
        return self._connected

    async def start(self) -> None:
        """Subscribe to market data for all symbols."""
        if not await self._conn.ensure_connected():
            logger.error("IBKRTickStreamer: cannot start — not connected")
            return

        ib = self._conn.ib
        ib.pendingTickersEvent += self._on_pending_tickers

        contracts = await self._conn.qualify_contracts_batch(self._symbols)
        for sym, contract in contracts.items():
            ticker = ib.reqMktData(contract, genericTickList='', snapshot=False)
            self._tickers[sym] = ticker

        subscribed = len(self._tickers)
        logger.info(
            f"IBKRTickStreamer: subscribed to {subscribed}/{len(self._symbols)} symbols"
        )
        self._connected = True

    async def stop(self) -> None:
        """Cancel all market data subscriptions."""
        if not self._conn.is_connected:
            self._tickers.clear()
            self._connected = False
            return

        ib = self._conn.ib
        try:
            ib.pendingTickersEvent -= self._on_pending_tickers
        except ValueError:
            pass  # handler already removed

        for sym, ticker in self._tickers.items():
            try:
                ib.cancelMktData(ticker.contract)
            except Exception:
                pass

        self._tickers.clear()
        self._connected = False
        logger.info("IBKRTickStreamer: stopped")

    async def add_symbols(self, symbols: List[str]) -> None:
        """Dynamically subscribe to additional symbols."""
        new_syms = [s for s in symbols if s not in self._tickers]
        if not new_syms:
            return

        room = self.MAX_SYMBOLS - len(self._tickers)
        if room <= 0:
            logger.warning(
                f"IBKRTickStreamer: at {self.MAX_SYMBOLS} limit, cannot add {new_syms}"
            )
            return
        new_syms = new_syms[:room]

        if not self._conn.is_connected:
            self._symbols.extend(new_syms)
            return

        contracts = await self._conn.qualify_contracts_batch(new_syms)
        ib = self._conn.ib
        for sym, contract in contracts.items():
            ticker = ib.reqMktData(contract, genericTickList='', snapshot=False)
            self._tickers[sym] = ticker
            if sym not in self._symbols:
                self._symbols.append(sym)

        logger.info(
            f"IBKRTickStreamer: added {list(contracts.keys())} "
            f"({len(self._tickers)}/{self.MAX_SYMBOLS})"
        )

    async def remove_symbols(self, symbols: List[str]) -> None:
        """Dynamically unsubscribe from symbols."""
        if not self._conn.is_connected:
            self._symbols = [s for s in self._symbols if s not in symbols]
            return

        ib = self._conn.ib
        removed = []
        for sym in symbols:
            ticker = self._tickers.pop(sym, None)
            if ticker:
                try:
                    ib.cancelMktData(ticker.contract)
                    removed.append(sym)
                except Exception:
                    pass

        self._symbols = [s for s in self._symbols if s not in symbols]
        if removed:
            logger.info(f"IBKRTickStreamer: removed {removed}")

    def _on_pending_tickers(self, tickers: list) -> None:
        """Callback fired by ib_insync when tickers update."""
        now = time.monotonic()
        for ticker in tickers:
            sym = ticker.contract.symbol
            price = ticker.marketPrice()
            if price != price or price <= 0:  # NaN check
                # Fall back to last trade or close
                price = ticker.last if ticker.last > 0 else ticker.close
            if price <= 0 or price != price:
                continue

            # Throttle updates
            last = self._last_push.get(sym, 0)
            if now - last < self.THROTTLE_SEC:
                continue

            self._latest_prices[sym] = price
            self._last_push[sym] = now

            if self._on_tick:
                try:
                    self._on_tick(sym, price, now)
                except Exception as e:
                    logger.warning(f"IBKRTickStreamer on_tick error: {e}")
