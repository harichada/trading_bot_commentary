#!/usr/bin/env python3
"""
Test script to place some paper trades
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def place_test_trades():
    """Place some test trades to populate positions"""
    
    print("Placing test paper trades...")
    
    # Test orders to place
    test_orders = [
        {
            "type": "bracket",
            "symbol": "SPY",
            "side": "BUY",
            "quantity": 100,
            "entry_price": 450.00,
            "take_profit": 460.00,
            "stop_loss": 445.00
        },
        {
            "type": "bracket",
            "symbol": "QQQ",
            "side": "BUY", 
            "quantity": 50,
            "entry_price": 380.00,
            "take_profit": 390.00,
            "stop_loss": 375.00
        },
        {
            "type": "bracket",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 25,
            "entry_price": 195.00,
            "take_profit": 200.00,
            "stop_loss": 192.00
        }
    ]
    
    for order in test_orders:
        try:
            response = requests.post(
                f"{BASE_URL}/api/professional/orders/advanced",
                json=order
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Placed {order['symbol']} order: {result.get('order_id')}")
            else:
                print(f"✗ Failed to place {order['symbol']} order: {response.text}")
                
        except Exception as e:
            print(f"✗ Error placing {order['symbol']} order: {e}")
        
        time.sleep(0.5)  # Small delay between orders
    
    print("\nChecking positions...")
    
    # Get current positions
    try:
        response = requests.get(f"{BASE_URL}/api/professional/paper/positions")
        if response.status_code == 200:
            positions = response.json()
            print(f"\nCurrent positions: {len(positions)}")
            for symbol, pos in positions.items():
                print(f"  {symbol}: {pos['quantity']} shares @ ${pos['entry_price']}")
        else:
            print(f"Failed to get positions: {response.text}")
    except Exception as e:
        print(f"Error getting positions: {e}")
    
    # Get account summary
    try:
        response = requests.get(f"{BASE_URL}/api/professional/paper/account")
        if response.status_code == 200:
            account = response.json()
            print(f"\nAccount Summary:")
            print(f"  Equity: ${account['equity']:,.2f}")
            print(f"  Positions: {account['positions']}")
            print(f"  Unrealized P&L: ${account['unrealized_pnl']:,.2f}")
    except Exception as e:
        print(f"Error getting account: {e}")

if __name__ == "__main__":
    # Check if API is available
    try:
        response = requests.get(f"{BASE_URL}/api/dashboard-version")
        if response.status_code == 200:
            print(f"API is running: {response.json()}")
            place_test_trades()
        else:
            print("API is not responding. Make sure the bot is running.")
    except:
        print("Cannot connect to API. Start the bot with: python run_professional_bot.py")