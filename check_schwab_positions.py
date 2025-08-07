#!/usr/bin/env python3
"""
Diagnostic script to check Schwab positions and P&L calculations
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

# Import the necessary modules
import sys
sys.path.append(str(Path(__file__).parent))

async def check_positions():
    """Check positions and compare calculations"""
    
    # Expected positions from user's Schwab account
    expected_positions = {
        'AMD': {'qty': 707, 'trade_price': 165.00, 'mark': 163.34, 'pnl_open': -1173.62},
        'HQGE': {'qty': 5100, 'trade_price': 0.0163, 'mark': 0.0003, 'pnl_open': -95.53},
        'COIN': {'qty': 0, 'pnl_day': 113.60},
        'NVDA': {'qty': 0, 'pnl_day': 254.30},
        'RIOT': {'qty': 0, 'pnl_day': 341.29},
        'TSLA': {'qty': 0, 'pnl_day': -4.76}
    }
    
    print("Expected positions from Schwab UI:")
    print("-" * 80)
    for symbol, data in expected_positions.items():
        if data.get('qty', 0) > 0:
            print(f"{symbol}: {data['qty']} shares @ ${data['trade_price']:.2f}, "
                  f"Mark: ${data['mark']:.2f}, P/L Open: ${data.get('pnl_open', 0):.2f}")
    print(f"\nExpected Overall Total P/L: $-1,269.15")
    print(f"Expected P/L Day: $-469.19")
    
    # Check what's in trading_state.json
    state_path = Path("trading_state.json")
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)
            
        print("\n" + "=" * 80)
        print("Positions in trading_state.json:")
        print("-" * 80)
        
        positions_data = state.get('positions_data', {})
        for symbol, pos_data in positions_data.items():
            print(f"{symbol}: {pos_data['quantity']} shares @ ${pos_data['entry_price']:.2f}, "
                  f"Side: {pos_data['side']}, Entry: {pos_data['entry_time'][:10]}")
            
            # Calculate expected P&L
            if symbol in expected_positions:
                expected = expected_positions[symbol]
                if expected.get('qty', 0) > 0:
                    calc_pnl = (expected['mark'] - pos_data['entry_price']) * pos_data['quantity']
                    print(f"  Calculated P&L: ${calc_pnl:.2f}")
                    print(f"  Expected P&L: ${expected.get('pnl_open', 'N/A')}")
                    print(f"  Difference: ${calc_pnl - expected.get('pnl_open', 0):.2f}")
    
    print("\n" + "=" * 80)
    print("Issues found:")
    print("-" * 80)
    
    # Check quantity mismatches
    if 'AMD' in positions_data:
        amd_qty_bot = positions_data['AMD']['quantity']
        amd_qty_schwab = expected_positions['AMD']['qty']
        if amd_qty_bot != amd_qty_schwab:
            print(f"❌ AMD quantity mismatch: Bot has {amd_qty_bot}, Schwab shows {amd_qty_schwab}")
            print(f"   Missing {amd_qty_schwab - amd_qty_bot} shares")
    
    # Check HQGE calculation
    if 'HQGE' in positions_data:
        hqge_data = positions_data['HQGE']
        hqge_expected = expected_positions['HQGE']
        hqge_calc_pnl = (hqge_expected['mark'] - hqge_data['entry_price']) * hqge_data['quantity']
        print(f"\n❌ HQGE P&L calculation:")
        print(f"   Entry: ${hqge_data['entry_price']:.4f}, Current: ${hqge_expected['mark']:.4f}")
        print(f"   Calculated: ${hqge_calc_pnl:.2f}")
        print(f"   Schwab shows: ${hqge_expected['pnl_open']:.2f}")
        print(f"   Note: Bot entry price ${hqge_data['entry_price']:.4f} vs Schwab ${hqge_expected['trade_price']:.4f}")
    
    print("\n" + "=" * 80)
    print("Recommendations:")
    print("-" * 80)
    print("1. The bot needs to sync the correct quantities from Schwab")
    print("2. The bot's entry prices may not match Schwab's average prices")
    print("3. Consider using Schwab's unrealizedProfitLoss field directly")
    print("4. The bot should handle closed positions (COIN, NVDA, etc.) for day P&L")

if __name__ == "__main__":
    asyncio.run(check_positions())