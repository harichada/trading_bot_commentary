import json
import asyncio
import logging
from typing import Set, Any

from fastapi import WebSocket

logger = logging.getLogger('TradingBot')


class ConnectionManager:
    """Manages WebSocket connections"""
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Send message to all connected clients.

        v-ws-broadcast-parallel-2026-04-30: this method used to iterate
        connections serially WHILE holding an asyncio.Lock — meaning every
        broadcast would await send_text() per client one after another.
        Accumulated zombie connections (browser tabs closed without clean
        TCP shutdown) caused each broadcast to hang for tens of seconds
        waiting for TCP timeouts. With dozens of commentary broadcasts
        per trading cycle, the FastAPI event loop spent most of its time
        awaiting writes to dead sockets — every HTTP request to '/' timed
        out at 4s even though the dashboard handler is just a memory read.
        Fix: snapshot the connection set under the lock (cheap), then
        release it; send to all clients concurrently with per-send
        timeout; remove dead ones in a single follow-up critical section.
        """
        if not self.active_connections:
            return

        message_str = json.dumps(message)

        async with self._lock:
            connections_snapshot = list(self.active_connections)

        async def _send_one(conn):
            try:
                await asyncio.wait_for(conn.send_text(message_str), timeout=2.0)
                return None  # success
            except (asyncio.TimeoutError, Exception) as exc:
                logger.debug(f"ws send failed (will drop): {exc}")
                return conn  # mark for removal

        results = await asyncio.gather(
            *(_send_one(c) for c in connections_snapshot),
            return_exceptions=False,
        )
        disconnected = {r for r in results if r is not None}

        if disconnected:
            async with self._lock:
                self.active_connections -= disconnected
            logger.info(f"WebSocket cleanup: dropped {len(disconnected)} dead connections; {len(self.active_connections)} active")
