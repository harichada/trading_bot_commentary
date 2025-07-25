#!/usr/bin/env python3
"""Test script to verify WebSocket commentary broadcasting fix"""

import asyncio
import websockets
import json
import sys

async def test_websocket_connection():
    """Test WebSocket connection and commentary reception"""
    uri = "ws://localhost:8000/ws"
    
    try:
        print("Connecting to WebSocket...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully!")
            print("Waiting for commentary updates...")
            
            # Listen for messages for 30 seconds
            start_time = asyncio.get_event_loop().time()
            message_count = 0
            
            while asyncio.get_event_loop().time() - start_time < 30:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    message_count += 1
                    
                    if data.get('type') == 'commentary':
                        print(f"\n📢 Commentary received: {data['data'].get('title', 'No title')}")
                        print(f"   Message: {data['data'].get('message', 'No message')[:100]}...")
                    elif data.get('type') == 'dashboard_update':
                        print("📊 Dashboard update received")
                    else:
                        print(f"📨 Message type: {data.get('type', 'unknown')}")
                        
                except asyncio.TimeoutError:
                    # No message received in 1 second, continue
                    pass
                except Exception as e:
                    print(f"❌ Error receiving message: {e}")
            
            print(f"\n✅ Test completed! Received {message_count} messages in 30 seconds")
            
    except ConnectionRefusedError:
        print("❌ Could not connect to WebSocket. Is the trading bot running?")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("WebSocket Commentary Broadcasting Test")
    print("=====================================")
    print("Make sure the trading bot is running first!")
    print()
    asyncio.run(test_websocket_connection())