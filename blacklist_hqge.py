#!/usr/bin/env python3
"""
Emergency script to blacklist HQGE due to catastrophic losses
"""

import json
from pathlib import Path
from datetime import datetime

def blacklist_hqge():
    """Add HQGE to the blacklist in trading_brain.json"""
    brain_path = Path("trading_brain.json")
    
    # Load existing brain data
    if brain_path.exists():
        with open(brain_path, 'r') as f:
            brain_data = json.load(f)
    else:
        print("Warning: trading_brain.json not found, creating new one")
        brain_data = {
            'memories': [],
            'pattern_success_rates': {},
            'symbol_behaviors': {},
            'emotional_state': {},
            'blacklisted_symbols': {},
            'consecutive_losses': {}
        }
    
    # Calculate total losses from trading_state.json
    state_path = Path("trading_state.json")
    total_loss = 0
    loss_count = 0
    
    if state_path.exists():
        with open(state_path, 'r') as f:
            state_data = json.load(f)
            
        for trade in state_data.get('trade_history', []):
            if trade['symbol'] == 'HQGE' and trade['pnl'] < 0:
                total_loss += abs(trade['pnl'])
                loss_count += 1
    
    # Ensure blacklisted_symbols exists
    if 'blacklisted_symbols' not in brain_data:
        brain_data['blacklisted_symbols'] = {}
    
    # Add HQGE to blacklist
    brain_data['blacklisted_symbols']['HQGE'] = {
        'reason': 'Emergency blacklist: 80+ consecutive stop losses',
        'total_loss': total_loss,
        'loss_count': loss_count,
        'blacklisted_at': datetime.now().isoformat(),
        'consecutive_losses': 80  # Approximate from the trading history
    }
    
    # Remove from consecutive losses counter if present
    if 'HQGE' in brain_data.get('consecutive_losses', {}):
        del brain_data['consecutive_losses']['HQGE']
    
    # Save updated brain data
    with open(brain_path, 'w') as f:
        json.dump(brain_data, f, indent=2)
    
    print(f"✅ HQGE has been blacklisted!")
    print(f"   Total loss: ${total_loss:.2f}")
    print(f"   Number of losing trades: {loss_count}")
    print(f"   Blacklisted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    blacklist_hqge()