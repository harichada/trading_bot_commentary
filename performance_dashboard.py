#!/usr/bin/env python3
"""
Performance Analytics Dashboard for Trading Bot
Provides comprehensive metrics and insights
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

def calculate_performance_metrics():
    """Calculate comprehensive performance metrics"""
    state_path = Path("trading_state.json")
    if not state_path.exists():
        print("No trading state found")
        return
    
    with open(state_path, 'r') as f:
        state = json.load(f)
    
    trades = state.get('trade_history', [])
    if not trades:
        print("No trades to analyze")
        return
    
    # Basic metrics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]
    
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    # P&L metrics
    total_pnl = sum(t.get('pnl', 0) for t in trades)
    avg_win = statistics.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = statistics.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
    
    # Profit factor
    gross_profit = sum(t['pnl'] for t in winning_trades)
    gross_loss = sum(abs(t['pnl']) for t in losing_trades)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Strategy performance
    strategy_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
    for trade in trades:
        strategy = trade.get('reasoning', {}).get('strategy', 'unknown')
        if trade['pnl'] > 0:
            strategy_stats[strategy]['wins'] += 1
        else:
            strategy_stats[strategy]['losses'] += 1
        strategy_stats[strategy]['pnl'] += trade['pnl']
    
    # Symbol performance
    symbol_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0})
    for trade in trades:
        symbol = trade['symbol']
        symbol_stats[symbol]['trades'] += 1
        if trade['pnl'] > 0:
            symbol_stats[symbol]['wins'] += 1
        symbol_stats[symbol]['pnl'] += trade['pnl']
    
    # Exit reason analysis
    exit_stats = defaultdict(int)
    for trade in trades:
        exit_stats[trade.get('exit_reason', 'unknown')] += 1
    
    # Time analysis
    hourly_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0})
    for trade in trades:
        try:
            entry_time = datetime.fromisoformat(trade['entry_time'])
            hour = entry_time.hour
            hourly_stats[hour]['trades'] += 1
            hourly_stats[hour]['pnl'] += trade['pnl']
        except:
            pass
    
    # Risk metrics
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_streak = 0
    
    for trade in trades:
        if trade['pnl'] > 0:
            if current_streak >= 0:
                current_streak += 1
            else:
                current_streak = 1
            max_consecutive_wins = max(max_consecutive_wins, current_streak)
        else:
            if current_streak <= 0:
                current_streak -= 1
            else:
                current_streak = -1
            max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))
    
    # Display dashboard
    print("=" * 80)
    print("                     TRADING PERFORMANCE DASHBOARD")
    print("=" * 80)
    
    print("\n📊 OVERALL PERFORMANCE")
    print("-" * 40)
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {len(winning_trades)} ({win_rate*100:.1f}%)")
    print(f"Losing Trades: {len(losing_trades)} ({(1-win_rate)*100:.1f}%)")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Average Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Win/Loss Ratio: {avg_win/avg_loss if avg_loss > 0 else 'N/A':.2f}")
    
    print("\n🎯 STRATEGY PERFORMANCE")
    print("-" * 40)
    for strategy, stats in sorted(strategy_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
        total = stats['wins'] + stats['losses']
        win_rate = stats['wins'] / total if total > 0 else 0
        print(f"{strategy:20} | Trades: {total:3d} | Win Rate: {win_rate*100:5.1f}% | P&L: ${stats['pnl']:8.2f}")
    
    print("\n💹 TOP SYMBOLS")
    print("-" * 40)
    sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)
    for symbol, stats in sorted_symbols[:10]:
        win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
        print(f"{symbol:6} | Trades: {stats['trades']:3d} | Win Rate: {win_rate*100:5.1f}% | P&L: ${stats['pnl']:8.2f}")
    
    print("\n🚪 EXIT REASON DISTRIBUTION")
    print("-" * 40)
    for reason, count in sorted(exit_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_trades * 100
        print(f"{reason:20} | {count:3d} trades ({percentage:5.1f}%)")
    
    print("\n⏰ BEST TRADING HOURS (ET)")
    print("-" * 40)
    sorted_hours = sorted(hourly_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)
    for hour, stats in sorted_hours[:5]:
        if stats['trades'] > 0:
            avg_pnl = stats['pnl'] / stats['trades']
            print(f"{hour:02d}:00-{hour:02d}:59 | Trades: {stats['trades']:3d} | Total P&L: ${stats['pnl']:7.2f} | Avg: ${avg_pnl:6.2f}")
    
    print("\n⚠️  RISK METRICS")
    print("-" * 40)
    print(f"Max Consecutive Wins: {max_consecutive_wins}")
    print(f"Max Consecutive Losses: {max_consecutive_losses}")
    print(f"Current Streak: {'Win' if current_streak > 0 else 'Loss'} streak of {abs(current_streak)}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    
    if win_rate < 0.5:
        print("⚠️  Win rate below 50% - Review entry criteria")
    
    if profit_factor < 1.5:
        print("⚠️  Profit factor below 1.5 - Need better risk/reward")
    
    if exit_stats.get('stop_loss', 0) / total_trades > 0.6:
        print("⚠️  Too many stop loss exits - Consider wider stops or better entries")
    
    problem_symbols = [sym for sym, stats in symbol_stats.items() if stats['pnl'] < -100]
    if problem_symbols:
        print(f"⚠️  Problem symbols to avoid: {', '.join(problem_symbols)}")
    
    best_strategy = max(strategy_stats.items(), key=lambda x: x[1]['pnl'])[0]
    print(f"✅ Best performing strategy: {best_strategy}")
    
    best_hour = max(hourly_stats.items(), key=lambda x: x[1]['pnl'])[0]
    print(f"✅ Most profitable hour: {best_hour:02d}:00-{best_hour:02d}:59")

if __name__ == "__main__":
    calculate_performance_metrics()