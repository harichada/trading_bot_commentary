#!/usr/bin/env python3
"""
Simple backtesting using existing trade history
Analyzes past performance to optimize strategies
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

def analyze_strategy_performance():
    """Analyze historical trades by strategy"""
    
    state_path = Path("trading_state.json")
    if not state_path.exists():
        print("No trading state found")
        return
    
    with open(state_path, 'r') as f:
        state = json.load(f)
    
    trades = state.get('trade_history', [])
    
    # Group by strategy
    strategy_results = defaultdict(lambda: {
        'trades': [],
        'by_hour': defaultdict(list),
        'by_symbol': defaultdict(list),
        'by_exit': defaultdict(list)
    })
    
    for trade in trades:
        strategy = trade.get('reasoning', {}).get('strategy', 'unknown')
        pnl = trade.get('pnl', 0)
        
        # Add to strategy
        strategy_results[strategy]['trades'].append(trade)
        
        # By hour
        try:
            entry_time = datetime.fromisoformat(trade['entry_time'])
            hour = entry_time.hour
            strategy_results[strategy]['by_hour'][hour].append(pnl)
        except:
            pass
        
        # By symbol
        symbol = trade.get('symbol', 'unknown')
        strategy_results[strategy]['by_symbol'][symbol].append(pnl)
        
        # By exit reason
        exit_reason = trade.get('exit_reason', 'unknown')
        strategy_results[strategy]['by_exit'][exit_reason].append(pnl)
    
    print("=" * 80)
    print("STRATEGY BACKTEST ANALYSIS")
    print("=" * 80)
    
    # Overall strategy performance
    print("\n📊 STRATEGY PERFORMANCE COMPARISON")
    print("-" * 60)
    print(f"{'Strategy':<20} {'Trades':<10} {'Win Rate':<12} {'Avg Win':<12} {'Avg Loss':<12} {'Total P&L':<12}")
    print("-" * 60)
    
    strategy_scores = {}
    
    for strategy, data in strategy_results.items():
        trades = data['trades']
        if not trades:
            continue
        
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = statistics.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = statistics.mean([t['pnl'] for t in losses]) if losses else 0
        total_pnl = sum(t['pnl'] for t in trades)
        
        # Calculate strategy score
        score = (win_rate * 100) + (avg_win / max(abs(avg_loss), 1) * 50) + (total_pnl / 100)
        strategy_scores[strategy] = score
        
        print(f"{strategy:<20} {len(trades):<10} {win_rate*100:<11.1f}% "
              f"${avg_win:<11.2f} ${avg_loss:<11.2f} ${total_pnl:<11.2f}")
    
    # Best hours by strategy
    print("\n⏰ BEST TRADING HOURS BY STRATEGY")
    print("-" * 60)
    
    for strategy, data in strategy_results.items():
        if len(data['trades']) < 5:
            continue
            
        print(f"\n{strategy}:")
        hour_performance = []
        
        for hour, pnls in data['by_hour'].items():
            if len(pnls) >= 2:  # Need at least 2 trades
                avg_pnl = statistics.mean(pnls)
                hour_performance.append((hour, avg_pnl, len(pnls)))
        
        # Sort by average P&L
        hour_performance.sort(key=lambda x: x[1], reverse=True)
        
        for hour, avg_pnl, count in hour_performance[:3]:
            print(f"  {hour:02d}:00-{hour:02d}:59 - Avg P&L: ${avg_pnl:>7.2f} ({count} trades)")
    
    # Best symbols by strategy
    print("\n💹 BEST SYMBOLS BY STRATEGY")
    print("-" * 60)
    
    for strategy, data in strategy_results.items():
        if len(data['trades']) < 5:
            continue
            
        print(f"\n{strategy}:")
        symbol_performance = []
        
        for symbol, pnls in data['by_symbol'].items():
            total_pnl = sum(pnls)
            win_rate = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
            symbol_performance.append((symbol, total_pnl, win_rate, len(pnls)))
        
        # Sort by total P&L
        symbol_performance.sort(key=lambda x: x[1], reverse=True)
        
        for symbol, total_pnl, win_rate, count in symbol_performance[:5]:
            print(f"  {symbol:<6} - P&L: ${total_pnl:>8.2f}, Win Rate: {win_rate*100:>5.1f}% ({count} trades)")
    
    # Exit reason analysis
    print("\n🚪 EXIT REASON ANALYSIS BY STRATEGY")
    print("-" * 60)
    
    for strategy, data in strategy_results.items():
        if len(data['trades']) < 5:
            continue
            
        print(f"\n{strategy}:")
        
        for exit_reason, pnls in data['by_exit'].items():
            if pnls:
                avg_pnl = statistics.mean(pnls)
                win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
                print(f"  {exit_reason:<20} - Avg: ${avg_pnl:>7.2f}, Win Rate: {win_rate*100:>5.1f}% ({len(pnls)} trades)")
    
    # Recommendations
    print("\n💡 OPTIMIZATION RECOMMENDATIONS")
    print("-" * 60)
    
    # Find best strategy
    if strategy_scores:
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        print(f"✅ Focus on {best_strategy} strategy - highest overall score")
    
    # Find problematic exit reasons
    all_stop_losses = []
    for strategy, data in strategy_results.items():
        stop_losses = data['by_exit'].get('stop_loss', [])
        all_stop_losses.extend(stop_losses)
    
    if all_stop_losses:
        stop_loss_rate = len(all_stop_losses) / len(trades) if trades else 0
        if stop_loss_rate > 0.6:
            print(f"⚠️  {stop_loss_rate*100:.1f}% of trades exit via stop loss - consider:")
            print("   - Wider initial stops")
            print("   - Better entry timing")
            print("   - Trailing stops instead of fixed")
    
    # Symbol recommendations
    print("\n📊 SYMBOL RECOMMENDATIONS")
    all_symbol_pnl = defaultdict(float)
    all_symbol_trades = defaultdict(int)
    
    for trade in trades:
        symbol = trade.get('symbol', 'unknown')
        all_symbol_pnl[symbol] += trade.get('pnl', 0)
        all_symbol_trades[symbol] += 1
    
    # Sort by total P&L
    sorted_symbols = sorted(all_symbol_pnl.items(), key=lambda x: x[1], reverse=True)
    
    print("\nBest performing symbols:")
    for symbol, total_pnl in sorted_symbols[:5]:
        if all_symbol_trades[symbol] >= 3:  # At least 3 trades
            avg_pnl = total_pnl / all_symbol_trades[symbol]
            print(f"  {symbol}: ${total_pnl:.2f} total, ${avg_pnl:.2f} average ({all_symbol_trades[symbol]} trades)")
    
    print("\nWorst performing symbols to avoid:")
    for symbol, total_pnl in sorted_symbols[-5:]:
        if total_pnl < -50:  # Significant losses
            print(f"  ❌ {symbol}: ${total_pnl:.2f} loss ({all_symbol_trades[symbol]} trades)")

def simulate_improvements():
    """Simulate potential improvements"""
    
    state_path = Path("trading_state.json")
    if not state_path.exists():
        return
    
    with open(state_path, 'r') as f:
        state = json.load(f)
    
    trades = state.get('trade_history', [])
    
    print("\n" + "=" * 80)
    print("IMPROVEMENT SIMULATIONS")
    print("=" * 80)
    
    # Simulate tighter stop losses
    print("\n📊 SIMULATION: Impact of New Stop Loss Logic")
    print("-" * 60)
    
    original_stop_losses = 0
    improved_stop_losses = 0
    
    for trade in trades:
        if trade.get('exit_reason') == 'stop_loss':
            original_stop_losses += abs(trade['pnl'])
            
            # Simulate new stop loss (max 5% instead of 98%)
            entry = trade['entry_price']
            exit = trade['exit_price']
            qty = trade['quantity']
            
            # Calculate what stop would have been
            if entry > 0:
                loss_pct = abs((exit - entry) / entry)
                if loss_pct > 0.05:  # If loss was more than 5%
                    new_exit = entry * 0.95  # 5% stop
                    new_pnl = (new_exit - entry) * qty
                    improved_stop_losses += abs(new_pnl)
                else:
                    improved_stop_losses += abs(trade['pnl'])
    
    savings = original_stop_losses - improved_stop_losses
    print(f"Original stop loss total: ${original_stop_losses:.2f}")
    print(f"With 5% max stop loss: ${improved_stop_losses:.2f}")
    print(f"Potential savings: ${savings:.2f} ({savings/original_stop_losses*100:.1f}% reduction)")
    
    # Simulate profit targets
    print("\n📊 SIMULATION: Impact of Profit Targets")
    print("-" * 60)
    
    trades_without_profit_exit = 0
    potential_profit_exits = 0
    
    for trade in trades:
        if trade.get('exit_reason') != 'take_profit' and trade['pnl'] > 0:
            trades_without_profit_exit += 1
            
            # Check if trade could have hit 3% profit target
            entry = trade['entry_price']
            exit = trade['exit_price']
            
            if entry > 0:
                max_gain = (exit - entry) / entry
                if max_gain >= 0.03:  # Could have hit 3% target
                    potential_profit_exits += 1
    
    print(f"Profitable trades that didn't hit profit target: {trades_without_profit_exit}")
    print(f"Trades that could have hit 3% target: {potential_profit_exits}")
    print(f"Potential improvement: {potential_profit_exits/trades_without_profit_exit*100:.1f}% more profit target exits")

if __name__ == "__main__":
    print("Running strategy backtest analysis...")
    print("Analyzing historical performance to optimize strategies")
    print("-" * 80)
    
    analyze_strategy_performance()
    simulate_improvements()
    
    print("\n✅ Backtest analysis complete!")
    print("Use these insights to refine your trading strategies")