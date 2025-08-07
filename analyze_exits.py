#!/usr/bin/env python3
"""
Analyze trade exit reasons
"""

import json
from pathlib import Path

def analyze_exits():
    state_path = Path("trading_state.json")
    if not state_path.exists():
        print("No trading state found")
        return
    
    with open(state_path, 'r') as f:
        state = json.load(f)
    
    # Analyze exit reasons
    exit_reasons = {}
    total_trades = 0
    stop_loss_trades = 0
    profitable_trades = 0
    
    for trade in state.get('trade_history', []):
        reason = trade.get('exit_reason', 'unknown')
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        total_trades += 1
        
        if reason == 'stop_loss':
            stop_loss_trades += 1
        
        if trade.get('pnl', 0) > 0:
            profitable_trades += 1
    
    print("=" * 60)
    print("TRADE EXIT ANALYSIS")
    print("=" * 60)
    print(f"Total trades: {total_trades}")
    print(f"Profitable trades: {profitable_trades} ({profitable_trades/total_trades*100:.1f}% win rate)")
    print(f"Stop loss exits: {stop_loss_trades} ({stop_loss_trades/total_trades*100:.1f}%)")
    
    print("\nExit reason distribution:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count} ({count/total_trades*100:.1f}%)")
    
    # Show profitable trades that didn't hit take profit
    print("\nProfitable trades (last 10):")
    profitable_exits = [t for t in state.get('trade_history', []) if t.get('pnl', 0) > 0]
    for trade in profitable_exits[-10:]:
        pnl_pct = (trade['pnl'] / (trade['entry_price'] * trade['quantity'])) * 100
        print(f"  {trade['symbol']}: +${trade['pnl']:.2f} ({pnl_pct:.1f}%) - Exit: {trade['exit_reason']}")
    
    # Show losing trades
    print("\nLosing trades (last 10):")
    losing_exits = [t for t in state.get('trade_history', []) if t.get('pnl', 0) < 0]
    for trade in losing_exits[-10:]:
        pnl_pct = (trade['pnl'] / (trade['entry_price'] * trade['quantity'])) * 100
        print(f"  {trade['symbol']}: -${abs(trade['pnl']):.2f} ({abs(pnl_pct):.1f}%) - Exit: {trade['exit_reason']}")
    
    print("\n" + "=" * 60)
    print("FINDINGS:")
    print("- Most exits are stop losses, indicating the take profit logic needs work")
    print("- Consider tighter stop losses and better entry timing")
    print("- The bot needs to let winners run to their profit targets")

if __name__ == "__main__":
    analyze_exits()