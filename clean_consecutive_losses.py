#!/usr/bin/env python3
"""
Clean up consecutive losses tracking and remove HQGE from all state files
"""

import json
from pathlib import Path
from datetime import datetime

def clean_all_state():
    """Clean trading state and brain from consecutive losses and HQGE"""
    
    # 1. Clean trading_state.json
    state_path = Path("trading_state.json")
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        print("=== TRADING STATE ===")
        print(f"Current consecutive losses: {state.get('consecutive_losses', 0)}")
        print(f"Current positions: {list(state.get('positions_data', {}).keys())}")
        
        # Reset consecutive losses
        state['consecutive_losses'] = 0
        
        # Remove HQGE from positions
        if 'positions_data' in state and 'HQGE' in state['positions_data']:
            del state['positions_data']['HQGE']
            print("✅ Removed HQGE from positions")
        
        # Update last save
        state['last_save'] = datetime.now().isoformat()
        
        # Save
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        print("✅ Reset consecutive losses to 0")
        print(f"✅ Updated positions: {list(state.get('positions_data', {}).keys())}")
    
    # 2. Clean trading_brain.json
    brain_path = Path("trading_brain.json")
    if brain_path.exists():
        with open(brain_path, 'r') as f:
            brain = json.load(f)
        
        print("\n=== TRADING BRAIN ===")
        
        # Reset consecutive losses tracking
        if 'consecutive_losses' in brain:
            brain['consecutive_losses'] = {}
            print("✅ Reset brain consecutive losses tracking")
        
        # Ensure HQGE stays blacklisted
        if 'blacklisted_symbols' not in brain:
            brain['blacklisted_symbols'] = {}
        
        if 'HQGE' not in brain['blacklisted_symbols']:
            brain['blacklisted_symbols']['HQGE'] = {
                "reason": "Catastrophic penny stock losses",
                "total_loss": 7360.38,
                "loss_count": 77,
                "blacklisted_at": datetime.now().isoformat(),
                "consecutive_losses": 77
            }
            print("✅ Re-added HQGE to blacklist")
        else:
            print("✅ HQGE still blacklisted")
        
        # Save
        with open(brain_path, 'w') as f:
            json.dump(brain, f, indent=2)
    
    # 3. Clean any HQGE references from memory
    print("\n=== CHECKING FOR HQGE IN MEMORIES ===")
    hqge_memories = 0
    if brain_path.exists():
        with open(brain_path, 'r') as f:
            brain = json.load(f)
        
        if 'memories' in brain:
            # Count HQGE memories
            for memory in brain['memories']:
                if memory.get('symbol') == 'HQGE':
                    hqge_memories += 1
            
            if hqge_memories > 0:
                # Remove HQGE memories
                brain['memories'] = [m for m in brain['memories'] if m.get('symbol') != 'HQGE']
                
                with open(brain_path, 'w') as f:
                    json.dump(brain, f, indent=2)
                
                print(f"✅ Removed {hqge_memories} HQGE memories")
            else:
                print("✅ No HQGE memories found")
    
    print("\n✅ CLEANUP COMPLETE!")
    print("- Consecutive losses reset to 0")
    print("- HQGE removed from positions")
    print("- HQGE confirmed blacklisted")
    print("- Brain state cleaned")

if __name__ == "__main__":
    clean_all_state()