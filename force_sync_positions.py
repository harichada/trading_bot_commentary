#!/usr/bin/env python3
"""
Force sync positions with Schwab to clear phantom positions
"""

import asyncio
import json
from datetime import datetime

async def force_sync():
    """Force the trading bot to sync positions with Schwab"""
    
    # Send a WebSocket command to force position sync
    import websockets
    
    try:
        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            # Request position sync
            command = {
                "type": "force_sync_positions"
            }
            await websocket.send(json.dumps(command))
            
            response = await websocket.recv()
            result = json.loads(response)
            
            print(f"Position sync result: {result}")
            
    except Exception as e:
        print(f"Error connecting to trading bot: {e}")
        print("\nAlternative: Restart the trading bot to clear phantom positions")
        print("The bot will automatically sync with Schwab on startup")

if __name__ == "__main__":
    asyncio.run(force_sync())