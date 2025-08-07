#!/usr/bin/env python3
"""
Fix trading state before market open
- Remove blacklisted HQGE position
- Reset consecutive losses
"""

import json
from pathlib import Path
from datetime import datetime

def fix_trading_state():
    """Clean up trading state"""
    state_path = Path("trading_state.json")
    
    if not state_path.exists():
        print("No trading state found")
        return
        
    # Load state
    with open(state_path, 'r') as f:
        state = json.load(f)
    
    print("Current state:")
    print(f"- Consecutive losses: {state.get('consecutive_losses', 0)}")
    print(f"- Positions: {list(state.get('positions_data', {}).keys())}")
    
    # Remove HQGE position if exists
    if 'positions_data' in state and 'HQGE' in state['positions_data']:
        del state['positions_data']['HQGE']
        print("✅ Removed HQGE from positions")
    
    # Reset consecutive losses for fresh start
    if state.get('consecutive_losses', 0) >= 5:
        state['consecutive_losses'] = 0
        print("✅ Reset consecutive losses to 0")
    
    # Update last save
    state['last_save'] = datetime.now().isoformat()
    
    # Save updated state
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)
    
    print("\nUpdated state:")
    print(f"- Consecutive losses: {state.get('consecutive_losses', 0)}")
    print(f"- Positions: {list(state.get('positions_data', {}).keys())}")
    print("\n✅ State cleaned up for market open!")

if __name__ == "__main__":
    fix_trading_state()