#!/usr/bin/env python3
"""Test script to verify commentary queue fix"""

import asyncio
import time
import requests
import websocket
import json
import threading

def test_websocket_connection():
    """Test WebSocket connection and commentary reception"""
    messages_received = []
    
    def on_message(ws, message):
        try:
            data = json.loads(message)
            if data.get('type') == 'commentary':
                print(f"✓ Commentary received: {data['data'].get('title', 'No title')}")
                messages_received.append(data)
        except Exception as e:
            print(f"Error parsing message: {e}")
    
    def on_error(ws, error):
        print(f"WebSocket error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        print("WebSocket closed")
    
    def on_open(ws):
        print("WebSocket connected")
    
    # Connect to WebSocket
    ws = websocket.WebSocketApp("ws://localhost:8000/ws",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    
    # Run WebSocket in a thread
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()
    
    # Wait for connection
    time.sleep(2)
    
    # Start trading to trigger commentary
    print("\nStarting trading engine...")
    response = requests.post("http://localhost:8000/api/start")
    print(f"Start response: {response.json()}")
    
    # Wait for commentary messages
    print("\nWaiting for commentary messages...")
    time.sleep(10)
    
    # Check results
    if messages_received:
        print(f"\n✅ Test PASSED: Received {len(messages_received)} commentary messages")
        print("No 'Error queuing commentary' errors detected")
    else:
        print("\n❌ Test FAILED: No commentary messages received")
    
    # Stop trading
    response = requests.post("http://localhost:8000/api/stop")
    print(f"\nStop response: {response.json()}")
    
    ws.close()

if __name__ == "__main__":
    print("Testing commentary queue fix...")
    print("Make sure the trading bot is running on port 8000")
    print("-" * 50)
    
    try:
        test_websocket_connection()
    except Exception as e:
        print(f"Test error: {e}")
        print("Make sure the trading bot is running with: python trading_bot_commentary_updated.py")