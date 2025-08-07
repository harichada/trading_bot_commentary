import json
from datetime import datetime

# Load current state
with open('trading_state.json', 'r') as f:
    state = json.load(f)

print("=== Marking HQGE as Long-Term Position ===\n")

# Check if HQGE is in positions_data
if 'HQGE' in state.get('positions_data', {}):
    print(f"Found HQGE position: {state['positions_data']['HQGE']}")
    
    # Mark as long-term
    state['positions_data']['HQGE']['is_long_term'] = True
    print("✅ Marked HQGE as long-term position")
    
    # Recalculate P&L excluding long-term positions
    long_term_symbols = {sym for sym, data in state['positions_data'].items() if data.get('is_long_term', False)}
    print(f"\nLong-term positions: {list(long_term_symbols)}")
    
    recalculated_pnl = 0
    hqge_pnl = 0
    
    for trade in state['trade_history']:
        if trade['symbol'] not in long_term_symbols:
            recalculated_pnl += trade.get('pnl', 0)
        else:
            hqge_pnl += trade.get('pnl', 0)
    
    print(f"\nTotal HQGE P&L: ${hqge_pnl:.2f}")
    print(f"Recalculated daily P&L (excluding long-term): ${recalculated_pnl:.2f}")
    
    # Update state
    state['daily_pnl'] = recalculated_pnl
    state['last_save'] = datetime.now().isoformat()
    
    # Save updated state
    with open('trading_state.json', 'w') as f:
        json.dump(state, f, indent=2)
    
    print("\n✅ State updated successfully!")
    print(f"✅ Daily P&L updated to: ${recalculated_pnl:.2f}")
    print("\nHQGE is now marked as long-term and will be:")
    print("- Excluded from P&L calculations")
    print("- Protected from auto-closing")
    print("- Ignored in trading decisions")
else:
    print("No HQGE position found in positions_data")

print("\nYou can toggle long-term status using the checkbox in the web interface.")