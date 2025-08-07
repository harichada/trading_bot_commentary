#!/usr/bin/env python3
"""
Fix position sync issues with Schwab
"""

import json
from pathlib import Path
from datetime import datetime

def fix_positions():
    """Update positions to match Schwab account"""
    state_path = Path("trading_state.json")
    
    if not state_path.exists():
        print("Error: trading_state.json not found")
        return
    
    # Load current state
    with open(state_path, 'r') as f:
        state = json.load(f)
    
    print("Current positions:")
    positions_data = state.get('positions_data', {})
    for symbol, data in positions_data.items():
        print(f"  {symbol}: {data['quantity']} shares @ ${data['entry_price']:.4f}")
    
    # Fix AMD quantity
    if 'AMD' in positions_data:
        print(f"\nFixing AMD: changing from {positions_data['AMD']['quantity']} to 707 shares")
        positions_data['AMD']['quantity'] = 707.0
        
    # Update HQGE entry price to match Schwab
    if 'HQGE' in positions_data:
        print(f"Fixing HQGE: changing entry price from ${positions_data['HQGE']['entry_price']:.4f} to $0.0163")
        positions_data['HQGE']['entry_price'] = 0.0163
    
    # Save updated state
    state['positions_data'] = positions_data
    state['last_save'] = datetime.now().isoformat()
    
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)
    
    print("\n✅ Positions updated!")
    print("\nNew positions:")
    for symbol, data in positions_data.items():
        print(f"  {symbol}: {data['quantity']} shares @ ${data['entry_price']:.4f}")
    
    print("\n⚠️  Note: You should restart the trading bot to reload these changes")
    print("⚠️  The bot should use Schwab's P&L calculations directly rather than calculating its own")

if __name__ == "__main__":
    fix_positions()