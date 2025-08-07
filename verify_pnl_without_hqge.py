import json

# Load the trading state
with open('trading_state.json', 'r') as f:
    data = json.load(f)

# Calculate P&L with and without HQGE
all_trades_pnl = 0
non_hqge_pnl = 0
hqge_count = 0
non_hqge_count = 0

print("=== P&L Analysis Excluding HQGE ===\n")

for trade in data['trade_history']:
    pnl = trade.get('pnl', 0)
    all_trades_pnl += pnl
    
    if trade['symbol'] == 'HQGE':
        hqge_count += 1
    else:
        non_hqge_pnl += pnl
        non_hqge_count += 1

print(f"Total trades: {len(data['trade_history'])}")
print(f"HQGE trades: {hqge_count}")
print(f"Other trades: {non_hqge_count}")
print()
print(f"All trades P&L: ${all_trades_pnl:.2f}")
print(f"Non-HQGE P&L: ${non_hqge_pnl:.2f}")
print(f"HQGE impact: ${all_trades_pnl - non_hqge_pnl:.2f}")
print()
print(f"Account balance (assumed): $100,000")
print(f"Daily loss % (all trades): {abs(all_trades_pnl) / 100000 * 100:.2f}%")
print(f"Daily loss % (excluding HQGE): {abs(non_hqge_pnl) / 100000 * 100:.2f}%")
print()
print(f"5% of $100,000 = $5,000")
print(f"Exceeds 5% limit (all trades): {abs(all_trades_pnl) > 5000}")
print(f"Exceeds 5% limit (excluding HQGE): {abs(non_hqge_pnl) > 5000}")

# Show breakdown by symbol
print("\n=== P&L Breakdown by Symbol ===")
symbol_pnl = {}
for trade in data['trade_history']:
    symbol = trade['symbol']
    if symbol not in symbol_pnl:
        symbol_pnl[symbol] = {'count': 0, 'total_pnl': 0}
    symbol_pnl[symbol]['count'] += 1
    symbol_pnl[symbol]['total_pnl'] += trade.get('pnl', 0)

for symbol, stats in sorted(symbol_pnl.items(), key=lambda x: x[1]['total_pnl']):
    print(f"{symbol}: {stats['count']} trades, P&L: ${stats['total_pnl']:.2f}")