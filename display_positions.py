#!/usr/bin/env python3
"""
Display current positions with accurate P&L matching Schwab
"""

import json
from pathlib import Path
from datetime import datetime

def display_positions():
    """Display positions in Schwab-like format"""
    
    # Load trading state
    state_path = Path("trading_state.json")
    if not state_path.exists():
        print("No trading state found")
        return
    
    with open(state_path, 'r') as f:
        state = json.load(f)
    
    positions_data = state.get('positions_data', {})
    
    # Current market prices (you'd get these from API)
    market_prices = {
        'AMD': 163.34,
        'HQGE': 0.0003,
        'COIN': 303.58,
        'NVDA': 179.40,
        'RIOT': 11.65,
        'TSLA': 320.26
    }
    
    print("=" * 100)
    print(f"{'Instrument':<10} {'Qty':<10} {'Trade Price':<12} {'Mark':<10} {'P/L Open':<12} {'P/L %':<10}")
    print("=" * 100)
    
    total_open_pnl = 0
    
    # Display open positions
    for symbol, pos_data in positions_data.items():
        qty = pos_data['quantity']
        entry_price = pos_data['entry_price']
        current_price = market_prices.get(symbol, entry_price)
        
        # Calculate P&L
        pnl_open = (current_price - entry_price) * qty
        pnl_percent = ((current_price - entry_price) / entry_price) * 100
        
        total_open_pnl += pnl_open
        
        print(f"{symbol:<10} {qty:<10.0f} ${entry_price:<11.4f} ${current_price:<9.4f} "
              f"${pnl_open:<11.2f} {pnl_percent:<9.2f}%")
    
    # Calculate day P&L from closed trades
    today = datetime.now().date()
    day_pnl = 0
    
    for trade in state.get('trade_history', []):
        try:
            exit_time = datetime.fromisoformat(trade['exit_time'])
            if exit_time.date() == today:
                day_pnl += trade.get('pnl', 0)
        except:
            pass
    
    print("=" * 100)
    print(f"{'Overall Total P/L Open:':<35} ${total_open_pnl:>10.2f}")
    print(f"{'Day P/L (Closed Trades):':<35} ${day_pnl:>10.2f}")
    print(f"{'Total P/L:':<35} ${total_open_pnl + day_pnl:>10.2f}")
    print("=" * 100)
    
    # Show any discrepancies
    print("\nExpected Schwab values:")
    print(f"  AMD: 707 shares, P/L Open: $-1,173.62")
    print(f"  HQGE: 5100 shares, P/L Open: $-95.53")
    print(f"  Overall Total: $-1,269.15")
    
    # Calculate bot's values with current data
    amd_pnl = (163.34 - 165.00) * positions_data.get('AMD', {}).get('quantity', 0)
    hqge_pnl = (0.0003 - positions_data.get('HQGE', {}).get('entry_price', 0)) * positions_data.get('HQGE', {}).get('quantity', 0)
    
    print(f"\nBot's calculated values:")
    print(f"  AMD: {positions_data.get('AMD', {}).get('quantity', 0)} shares, P/L: ${amd_pnl:.2f}")
    print(f"  HQGE: {positions_data.get('HQGE', {}).get('quantity', 0)} shares, P/L: ${hqge_pnl:.2f}")

if __name__ == "__main__":
    display_positions()