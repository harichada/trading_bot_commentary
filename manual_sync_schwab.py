#!/usr/bin/env python3
"""
Manually sync positions with correct Schwab data
"""

import json
from pathlib import Path
from datetime import datetime

def manual_sync():
    """Manually update positions to match Schwab exactly"""
    
    # Correct data from Schwab UI
    schwab_positions = {
        'AMD': {
            'quantity': 707,
            'average_price': 165.00,
            'current_price': 163.34,
            'pnl_open': -1173.62
        },
        'HQGE': {
            'quantity': 5100,
            'average_price': 0.0163,  # Schwab shows this as trade price
            'current_price': 0.0003,
            'pnl_open': -95.53
        }
    }
    
    # Load current state
    state_path = Path("trading_state.json")
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)
    else:
        state = {'positions_data': {}, 'trade_history': []}
    
    print("Current positions in bot:")
    positions_data = state.get('positions_data', {})
    for symbol, data in positions_data.items():
        print(f"  {symbol}: {data['quantity']} @ ${data['entry_price']:.4f}")
    
    print("\nUpdating to match Schwab...")
    
    # Update positions to match Schwab
    for symbol, schwab_data in schwab_positions.items():
        if symbol in positions_data:
            # Update existing position
            old_qty = positions_data[symbol]['quantity']
            old_price = positions_data[symbol]['entry_price']
            
            positions_data[symbol]['quantity'] = schwab_data['quantity']
            positions_data[symbol]['entry_price'] = schwab_data['average_price']
            
            print(f"  {symbol}: {old_qty} @ ${old_price:.4f} → "
                  f"{schwab_data['quantity']} @ ${schwab_data['average_price']:.4f}")
        else:
            # Add new position
            positions_data[symbol] = {
                'is_long_term': False,
                'entry_price': schwab_data['average_price'],
                'quantity': schwab_data['quantity'],
                'side': 'long',
                'entry_time': datetime.now().isoformat()
            }
            print(f"  {symbol}: Added {schwab_data['quantity']} @ ${schwab_data['average_price']:.4f}")
    
    # Update state
    state['positions_data'] = positions_data
    state['last_save'] = datetime.now().isoformat()
    
    # Save updated state
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)
    
    print("\n✅ Positions synced with Schwab!")
    print("\nNew positions:")
    for symbol, data in positions_data.items():
        print(f"  {symbol}: {data['quantity']} @ ${data['entry_price']:.4f}")
    
    # Calculate P&L
    print("\nP&L Calculations:")
    total_pnl = 0
    for symbol, schwab_data in schwab_positions.items():
        pnl = schwab_data['pnl_open']
        total_pnl += pnl
        print(f"  {symbol}: ${pnl:.2f}")
    print(f"  Total Open P&L: ${total_pnl:.2f}")
    
    print("\n⚠️  IMPORTANT: Restart the trading bot to load these changes!")

if __name__ == "__main__":
    manual_sync()