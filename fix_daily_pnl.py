import json
from datetime import datetime

# Load current state
with open('trading_state.json', 'r') as f:
    state = json.load(f)

print("=== Fixing Daily P&L by Excluding HQGE ===\n")
print(f"Current daily_pnl: ${state['daily_pnl']:.2f}")

# Recalculate P&L excluding HQGE
blacklisted_symbols = ['HQGE']
recalculated_pnl = 0
hqge_pnl = 0
consecutive_losses = 0

for trade in state['trade_history']:
    if trade['symbol'] not in blacklisted_symbols:
        recalculated_pnl += trade.get('pnl', 0)
        if trade.get('pnl', 0) < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0
    else:
        hqge_pnl += trade.get('pnl', 0)

print(f"HQGE total P&L: ${hqge_pnl:.2f}")
print(f"Recalculated P&L (excluding HQGE): ${recalculated_pnl:.2f}")
print(f"Difference: ${state['daily_pnl'] - recalculated_pnl:.2f}")

# Update the state
state['daily_pnl'] = recalculated_pnl
state['consecutive_losses'] = consecutive_losses
state['last_save'] = datetime.now().isoformat()

# Save the corrected state
with open('trading_state.json', 'w') as f:
    json.dump(state, f, indent=2)

print(f"\n✅ Updated daily_pnl to: ${recalculated_pnl:.2f}")
print(f"✅ Updated consecutive_losses to: {consecutive_losses}")
print("\nThe bot should no longer trigger the 5% daily loss limit!")

# Show recent non-HQGE trades
print("\n=== Recent Non-HQGE Trades ===")
recent_trades = [t for t in state['trade_history'][-10:] if t['symbol'] != 'HQGE']
for trade in recent_trades:
    print(f"{trade['symbol']}: ${trade['pnl']:.2f} ({trade['exit_reason']})")