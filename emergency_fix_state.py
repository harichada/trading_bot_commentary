import json
from datetime import datetime

print("=== EMERGENCY STATE FIX ===\n")

# Load current state
with open('trading_state.json', 'r') as f:
    state = json.load(f)

print(f"Current daily_pnl: ${state['daily_pnl']:.2f} (EMERGENCY!)")
print(f"Consecutive losses: {state['consecutive_losses']}")

# Get all long-term positions
long_term_symbols = {sym for sym, data in state['positions_data'].items() if data.get('is_long_term', False)}
print(f"\nLong-term positions: {list(long_term_symbols)}")

# Recalculate P&L excluding ALL trades from long-term positions
recalculated_pnl = 0
recalculated_losses = 0
excluded_pnl = 0
excluded_trades = 0

for trade in state['trade_history']:
    if trade['symbol'] not in long_term_symbols:
        recalculated_pnl += trade.get('pnl', 0)
        if trade.get('pnl', 0) < 0:
            recalculated_losses += 1
        else:
            recalculated_losses = 0
    else:
        excluded_pnl += trade.get('pnl', 0)
        excluded_trades += 1

print(f"\nExcluded {excluded_trades} trades from long-term positions")
print(f"Excluded P&L: ${excluded_pnl:.2f}")
print(f"Recalculated P&L: ${recalculated_pnl:.2f}")

# Update the state
state['daily_pnl'] = recalculated_pnl
state['consecutive_losses'] = recalculated_losses
state['last_save'] = datetime.now().isoformat()

# Save the fixed state
with open('trading_state.json', 'w') as f:
    json.dump(state, f, indent=2)

print("\n✅ EMERGENCY FIX APPLIED!")
print(f"✅ Daily P&L corrected to: ${recalculated_pnl:.2f}")
print(f"✅ Consecutive losses corrected to: {recalculated_losses}")

# Show the actual trading performance
print("\n=== ACTUAL TRADING PERFORMANCE ===")
symbol_pnl = {}
for trade in state['trade_history']:
    symbol = trade['symbol']
    if symbol not in symbol_pnl:
        symbol_pnl[symbol] = 0
    symbol_pnl[symbol] += trade.get('pnl', 0)

for symbol, pnl in sorted(symbol_pnl.items(), key=lambda x: x[1], reverse=True):
    if symbol not in long_term_symbols:
        print(f"{symbol}: ${pnl:.2f}")

print("\n⚠️  CRITICAL: The bot MUST be restarted to load the new code!")
print("⚠️  Until restart, it will continue trying to close HQGE!")
print("⚠️  The long-term protection is NOT active in the running bot!")