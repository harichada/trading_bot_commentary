#!/usr/bin/env python3
"""
Analyze profit targets and why they're not being hit
"""

import json
from pathlib import Path

def analyze_targets():
    """Analyze take profit targets in current positions"""
    
    # Current market prices (approximate)
    current_prices = {
        'AMD': 163.34,
        'HQGE': 0.0003
    }
    
    # Load positions
    state_path = Path("trading_state.json")
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)
    
    print("=" * 80)
    print("CURRENT POSITIONS - PROFIT TARGET ANALYSIS")
    print("=" * 80)
    
    positions_data = state.get('positions_data', {})
    
    for symbol, pos_data in positions_data.items():
        entry = pos_data['entry_price']
        current = current_prices.get(symbol, entry)
        
        # Default take profit is 10% (from the code)
        take_profit = entry * 1.10
        
        # Calculate how far we are from take profit
        if pos_data['side'] == 'long':
            current_pnl_pct = ((current - entry) / entry) * 100
            target_pnl_pct = ((take_profit - entry) / entry) * 100
            distance_to_target = take_profit - current
            distance_pct = (distance_to_target / current) * 100
        else:
            current_pnl_pct = ((entry - current) / entry) * 100
            target_pnl_pct = ((entry - take_profit) / entry) * 100
            distance_to_target = current - take_profit
            distance_pct = (distance_to_target / current) * 100
        
        print(f"\n{symbol}:")
        print(f"  Entry: ${entry:.4f}")
        print(f"  Current: ${current:.4f}")
        print(f"  Take Profit: ${take_profit:.4f} ({target_pnl_pct:.1f}% gain)")
        print(f"  Current P/L: {current_pnl_pct:.1f}%")
        print(f"  Distance to target: ${distance_to_target:.4f} ({distance_pct:.1f}% move needed)")
        
        # Analysis
        if symbol == 'HQGE':
            print(f"  ⚠️  WARNING: This penny stock would need a {distance_pct:.0f}% move to hit target!")
            print(f"  💡 Recommendation: Use much smaller targets for penny stocks (2-5%)")
        elif distance_pct > 10:
            print(f"  ⚠️  Target is quite far ({distance_pct:.1f}% move needed)")
            print(f"  💡 Recommendation: Consider scaling out at smaller gains (3-5%)")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR BETTER PROFIT TAKING:")
    print("=" * 80)
    print("1. Implement SCALED EXITS:")
    print("   - Take 50% off at 2% gain")
    print("   - Take another 25% at 4% gain")
    print("   - Let final 25% run to 6%+ with trailing stop")
    print("\n2. ADJUST TARGETS BY STOCK PRICE:")
    print("   - Penny stocks (<$5): 2-5% targets")
    print("   - Low price ($5-50): 3-7% targets")
    print("   - High price (>$50): 2-4% targets")
    print("\n3. USE DYNAMIC TARGETS based on:")
    print("   - ATR (Average True Range)")
    print("   - Recent volatility")
    print("   - Time of day")
    print("\n4. IMPLEMENT TRAILING PROFIT TARGETS:")
    print("   - If stock moves 3% in favor, move target to 5%")
    print("   - If stock moves 5% in favor, move target to 7%")

if __name__ == "__main__":
    analyze_targets()