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
        """Send message to all connected clients"""
        if not self.active_connections:
            return

        message_str = json.dumps(message)
        disconnected = set()

        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_text(message_str)
                except Exception as e:
                    logger.debug(f"Failed to send to connection: {e}")
                    disconnected.add(connection)

        if disconnected:
            async with self._lock:
                self.active_connections -= disconnected
