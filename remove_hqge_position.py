import json
from datetime import datetime

# Load current state
with open('trading_state.json', 'r') as f:
    state = json.load(f)

print("=== Removing HQGE from Active Positions ===\n")

# Check if HQGE is in positions_data
if 'HQGE' in state.get('positions_data', {}):
    print(f"Found HQGE position: {state['positions_data']['HQGE']}")
    del state['positions_data']['HQGE']
    print("✅ Removed HQGE from positions_data")
else:
    print("No HQGE found in positions_data")

# Recalculate P&L without HQGE
blacklisted_symbols = ['HQGE']
recalculated_pnl = 0
hqge_trades = 0
total_hqge_pnl = 0

for trade in state['trade_history']:
    if trade['symbol'] not in blacklisted_symbols:
        recalculated_pnl += trade.get('pnl', 0)
    else:
        hqge_trades += 1
        total_hqge_pnl += trade.get('pnl', 0)

print(f"\nTotal HQGE trades in history: {hqge_trades}")
print(f"Total HQGE P&L: ${total_hqge_pnl:.2f}")
print(f"Recalculated daily P&L (excluding HQGE): ${recalculated_pnl:.2f}")

# Update state
state['daily_pnl'] = recalculated_pnl
state['last_save'] = datetime.now().isoformat()

# Save updated state
with open('trading_state.json', 'w') as f:
    json.dump(state, f, indent=2)

print("\n✅ State updated successfully!")
print(f"✅ Daily P&L updated to: ${recalculated_pnl:.2f}")
print("\nHQGE has been completely removed from active tracking.")
print("\nIMPORTANT: The bot needs to be restarted to load the blacklist configuration!")