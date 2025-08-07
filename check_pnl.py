import json

with open('trading_state.json', 'r') as f:
    data = json.load(f)
    
total_pnl = sum(trade['pnl'] for trade in data['trade_history'])
print(f'Total P&L from all trades: ${total_pnl:.2f}')
print(f'Daily P&L stored in state: ${data["daily_pnl"]:.2f}')
print(f'Number of trades: {len(data["trade_history"])}')

# Count trades with HQGE
hqge_trades = [t for t in data['trade_history'] if t['symbol'] == 'HQGE']
hqge_pnl = sum(t['pnl'] for t in hqge_trades)
print(f'\nHQGE trades: {len(hqge_trades)}')
print(f'HQGE total P&L: ${hqge_pnl:.2f}')

# Count other trades
other_trades = [t for t in data['trade_history'] if t['symbol'] != 'HQGE']
other_pnl = sum(t['pnl'] for t in other_trades)
print(f'\nOther trades: {len(other_trades)}')
print(f'Other total P&L: ${other_pnl:.2f}')

# Check account balance
print(f'\nAssuming account balance: $100,000')
print(f'Daily loss percentage: {abs(data["daily_pnl"]) / 100000 * 100:.2f}%')
print(f'5% of $100,000 = $5,000')
print(f'Daily loss exceeds 5% limit: {abs(data["daily_pnl"]) > 5000}')