#!/usr/bin/env python3
"""
Fix phantom PLTR position in trading bot
"""

import json
from pathlib import Path

def fix_pltr_position():
    """Remove any phantom PLTR position from trading state"""
    state_path = Path("trading_state.json")
    
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        # Check positions_data
        positions_data = state.get('positions_data', {})
        
        if 'PLTR' in positions_data:
            print(f"Found PLTR in positions_data: {positions_data['PLTR']}")
            del positions_data['PLTR']
            state['positions_data'] = positions_data
            
            # Save updated state
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
            print("✅ Removed PLTR from positions_data")
        else:
            print("PLTR not found in positions_data")
        
        # Check trade history for any open PLTR positions
        trade_history = state.get('trade_history', [])
        pltr_trades = [t for t in trade_history if t['symbol'] == 'PLTR']
        
        if pltr_trades:
            latest_pltr = pltr_trades[-1]
            print(f"\nLatest PLTR trade:")
            print(f"  Entry: {latest_pltr['entry_time']}")
            print(f"  Exit: {latest_pltr.get('exit_time', 'STILL OPEN')}")
            print(f"  Exit reason: {latest_pltr.get('exit_reason', 'N/A')}")
            
            if 'exit_time' in latest_pltr:
                print("\n✅ PLTR position was properly closed")
            else:
                print("\n⚠️ PLTR position appears to be open in trade history!")
    
    print("\nTo fix the in-memory position, you'll need to restart the trading bot.")
    print("The phantom position is likely stuck in the bot's memory.")

if __name__ == "__main__":
    fix_pltr_position()