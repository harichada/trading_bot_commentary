"""IB Gateway connection manager with auto-reconnect.

Wraps ``ib_insync.IB`` to provide a shared, reconnectable connection
used by both the broker adapter and the tick streamer.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

from ib_insync import IB, Contract, Stock, util

# Patch asyncio for ib_insync compatibility — but only if nest_asyncio
# can handle the current loop type (fails with uvloop used by uvicorn).
try:
    util.patchAsyncio()
except (ValueError, RuntimeError):
    pass  # uvloop — ib_insync works fine without nest_asyncio in this context

logger = logging.getLogger(__name__)


class IBKRConnectionManager:
    """Manages a single IB Gateway connection with auto-reconnect."""

    def __init__(
        self,
        host: str = '',
        port: int = 0,
        client_id: int = 0,
    ) -> None:
        self._host = host or os.environ.get('IBKR_HOST', '127.0.0.1')
        self._port = port or int(os.environ.get('IBKR_PORT', '4002'))
        self._client_id = client_id or int(os.environ.get('IBKR_CLIENT_ID', '1'))
        self._ib = IB()
        self._connected = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._contract_cache: dict[str, Contract] = {}
        self._last_connect_attempt = 0.0
        self._intentional_disconnect = False

        # Register disconnect handler
        self._ib.disconnectedEvent += self._on_disconnect

    @property
    def ib(self) -> IB:
        """Raw ib_insync.IB instance."""
        return self._ib

    @property
    def is_connected(self) -> bool:
        return self._ib.isConnected()

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    async def connect(self, timeout: float = 15.0) -> bool:
        """Connect to IB Gateway. Returns True on success."""
        if self._ib.isConnected():
            return True

        self._last_connect_attempt = time.monotonic()
        self._intentional_disconnect = False
        try:
            await self._ib.connectAsync(
                self._host, self._port, self._client_id, timeout=timeout,
            )
            self._connected = True
            logger.info(
                f"IBKR connected: {self._host}:{self._port} "
                f"(clientId={self._client_id})"
            )
            return True
        except Exception as e:
            logger.error(f"IBKR connect failed: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Graceful disconnect."""
        self._intentional_disconnect = True
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
        if self._ib.isConnected():
            self._ib.disconnect()
        self._connected = False
        self._contract_cache.clear()
        logger.info("IBKR disconnected")

    async def ensure_connected(self) -> bool:
        """Connect if not already connected. Returns connection status."""
        if self._ib.isConnected():
            return True
        return await self.connect()

    async def qualify_contract(self, symbol: str) -> Optional[Contract]:
        """Get a qualified US equity contract, with caching."""
        if symbol in self._contract_cache:
            return self._contract_cache[symbol]

        contract = Stock(symbol, 'SMART', 'USD')
        try:
            qualified = await self._ib.qualifyContractsAsync(contract)
            if qualified:
                self._contract_cache[symbol] = qualified[0]
                return qualified[0]
        except Exception as e:
            logger.warning(f"IBKR qualify contract {symbol} failed: {e}")
        return None

    async def qualify_contracts_batch(
        self, symbols: list[str],
    ) -> dict[str, Contract]:
        """Qualify multiple contracts at once."""
        result: dict[str, Contract] = {}
        to_qualify: list[Contract] = []
        for sym in symbols:
            if sym in self._contract_cache:
                result[sym] = self._contract_cache[sym]
            else:
                to_qualify.append(Stock(sym, 'SMART', 'USD'))

        if to_qualify:
            try:
                qualified = await self._ib.qualifyContractsAsync(*to_qualify)
                for c in qualified:
                    if c.conId:  # successfully qualified
                        self._contract_cache[c.symbol] = c
                        result[c.symbol] = c
            except Exception as e:
                logger.warning(f"IBKR batch qualify failed: {e}")

        return result

    def _on_disconnect(self) -> None:
        """Handle unexpected disconnection — schedule reconnect."""
        self._connected = False

        if self._intentional_disconnect:
            self._intentional_disconnect = False
            return

        logger.warning("IBKR disconnected unexpectedly")

        # Don't reconnect if too recently attempted
        elapsed = time.monotonic() - self._last_connect_attempt
        if elapsed < 5.0:
            return

        try:
            loop = asyncio.get_running_loop()
            if self._reconnect_task is None or self._reconnect_task.done():
                self._reconnect_task = loop.create_task(self._reconnect_loop())
        except RuntimeError:
            pass  # no running loop

    async def _reconnect_loop(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        delays = [5, 10, 30, 60, 60]
        for attempt, delay in enumerate(delays, 1):
            logger.info(f"IBKR reconnect attempt {attempt} in {delay}s...")
            await asyncio.sleep(delay)
            if await self.connect():
                logger.info("IBKR reconnected successfully")
                return
        logger.error("IBKR reconnect failed after all attempts")
