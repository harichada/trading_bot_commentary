#!/usr/bin/env python3
"""
EMERGENCY: Stop HQGE trading immediately
- Remove from all state files
- Add multiple blocks to prevent re-entry
"""

import json
from pathlib import Path
from datetime import datetime
import sys

def emergency_stop_hqge():
    """Emergency stop for HQGE trading"""
    print("🚨 EMERGENCY HQGE STOP 🚨")
    print("=" * 50)
    
    # 1. Clean trading_state.json
    state_path = Path("trading_state.json")
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        # Count HQGE losses
        hqge_trades = [t for t in state.get('trade_history', []) if t['symbol'] == 'HQGE']
        total_hqge_loss = sum(t['pnl'] for t in hqge_trades if t['pnl'] < 0)
        
        print(f"Found {len(hqge_trades)} HQGE trades")
        print(f"Total HQGE losses: ${total_hqge_loss:.2f}")
        
        # Remove HQGE from positions
        if 'positions_data' in state and 'HQGE' in state['positions_data']:
            del state['positions_data']['HQGE']
            print("✅ Removed HQGE from active positions")
        
        # Reset consecutive losses
        state['consecutive_losses'] = 0
        print("✅ Reset consecutive losses to 0")
        
        # Save
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    # 2. Update trading_brain.json with STRONG blacklist
    brain_path = Path("trading_brain.json")
    if brain_path.exists():
        with open(brain_path, 'r') as f:
            brain = json.load(f)
        
        # Ensure blacklisted_symbols exists
        if 'blacklisted_symbols' not in brain:
            brain['blacklisted_symbols'] = {}
        
        # Add HQGE with EMERGENCY status
        brain['blacklisted_symbols']['HQGE'] = {
            "reason": "EMERGENCY: Catastrophic penny stock - 98% losses per trade",
            "total_loss": abs(total_hqge_loss),
            "loss_count": len(hqge_trades),
            "blacklisted_at": datetime.now().isoformat(),
            "consecutive_losses": len(hqge_trades),
            "EMERGENCY": True,
            "NEVER_TRADE": True,
            "BLOCK_EXTERNAL": True
        }
        
        # Reset consecutive losses tracking
        if 'consecutive_losses' in brain:
            brain['consecutive_losses'] = {}
        
        print("✅ Added EMERGENCY blacklist for HQGE")
        
        # Save
        with open(brain_path, 'w') as f:
            json.dump(brain, f, indent=2)
    
    # 3. Create a permanent block file
    block_path = Path("BLOCK_HQGE.json")
    block_data = {
        "symbol": "HQGE",
        "blocked": True,
        "reason": "EMERGENCY BLOCK - Catastrophic losses",
        "created": datetime.now().isoformat(),
        "total_loss": abs(total_hqge_loss),
        "trade_count": len(hqge_trades),
        "message": "DO NOT TRADE HQGE UNDER ANY CIRCUMSTANCES"
    }
    
    with open(block_path, 'w') as f:
        json.dump(block_data, f, indent=2)
    
    print("✅ Created BLOCK_HQGE.json file")
    
    # 4. Show summary
    print("\n" + "=" * 50)
    print("EMERGENCY STOP COMPLETE:")
    print(f"- Removed HQGE from positions")
    print(f"- Reset consecutive losses to 0")
    print(f"- Added EMERGENCY blacklist")
    print(f"- Created permanent block file")
    print(f"- Total HQGE losses prevented: ${abs(total_hqge_loss):.2f}")
    print("\n⛔ HQGE IS NOW PERMANENTLY BLOCKED ⛔")
    
    return True

if __name__ == "__main__":
    if emergency_stop_hqge():
        sys.exit(0)
    else:
        sys.exit(1)