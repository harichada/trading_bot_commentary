#!/usr/bin/env python3
"""
Analyze position holding times to understand overnight risk
"""

import json
from pathlib import Path
from datetime import datetime

def analyze_holding_times():
    state_path = Path("trading_state.json")
    if not state_path.exists():
        print("No trading state found")
        return
    
    with open(state_path, 'r') as f:
        state = json.load(f)
    
    holding_times = []
    overnight_positions = 0
    intraday_positions = 0
    
    for trade in state.get('trade_history', []):
        try:
            entry_time = datetime.fromisoformat(trade['entry_time'])
            exit_time = datetime.fromisoformat(trade['exit_time'])
            
            # Calculate holding time
            holding_duration = exit_time - entry_time
            holding_minutes = holding_duration.total_seconds() / 60
            holding_times.append(holding_minutes)
            
            # Check if position was held overnight
            if entry_time.date() != exit_time.date():
                overnight_positions += 1
            else:
                intraday_positions += 1
                
        except Exception as e:
            continue
    
    if not holding_times:
        print("No completed trades found")
        return
    
    print("=" * 60)
    print("POSITION HOLDING TIME ANALYSIS")
    print("=" * 60)
    
    avg_holding = sum(holding_times) / len(holding_times)
    min_holding = min(holding_times)
    max_holding = max(holding_times)
    
    print(f"Total trades analyzed: {len(holding_times)}")
    print(f"Intraday trades: {intraday_positions} ({intraday_positions/len(holding_times)*100:.1f}%)")
    print(f"Overnight trades: {overnight_positions} ({overnight_positions/len(holding_times)*100:.1f}%)")
    
    print(f"\nHolding times:")
    print(f"  Average: {avg_holding:.1f} minutes ({avg_holding/60:.1f} hours)")
    print(f"  Minimum: {min_holding:.1f} minutes")
    print(f"  Maximum: {max_holding:.1f} minutes ({max_holding/60:.1f} hours)")
    
    # Distribution
    print("\nHolding time distribution:")
    under_5min = sum(1 for t in holding_times if t < 5)
    under_30min = sum(1 for t in holding_times if t < 30)
    under_1hr = sum(1 for t in holding_times if t < 60)
    under_2hr = sum(1 for t in holding_times if t < 120)
    over_2hr = sum(1 for t in holding_times if t >= 120)
    
    print(f"  < 5 minutes: {under_5min} ({under_5min/len(holding_times)*100:.1f}%)")
    print(f"  < 30 minutes: {under_30min} ({under_30min/len(holding_times)*100:.1f}%)")
    print(f"  < 1 hour: {under_1hr} ({under_1hr/len(holding_times)*100:.1f}%)")
    print(f"  < 2 hours: {under_2hr} ({under_2hr/len(holding_times)*100:.1f}%)")
    print(f"  >= 2 hours: {over_2hr} ({over_2hr/len(holding_times)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    
    if overnight_positions > 0:
        print(f"⚠️  WARNING: {overnight_positions} positions were held overnight!")
        print("   This adds significant gap risk for day trading")
    
    if avg_holding > 120:
        print("⚠️  Average holding time is over 2 hours")
        print("   Consider tighter exit criteria for day trading")
    
    print("\nWith new time-based exits implemented:")
    print("✅ No new trades in first 15 minutes (9:30-9:45 AM)")
    print("✅ No new trades after 3:30 PM")
    print("✅ Auto-close profits in last 30 minutes")
    print("✅ Force-close all positions in last 10 minutes")
    print("✅ Exit flat positions during lunch hour")

if __name__ == "__main__":
    analyze_holding_times()