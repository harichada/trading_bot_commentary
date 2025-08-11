#!/usr/bin/env python3
"""
Simple debug script to check Schwab P&L using existing bot infrastructure
"""

import asyncio
import logging
import sys
import os

# Add parent directory to path to import trading bot modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def debug_pnl():
    """Debug P&L using the trading bot's existing connection"""
    
    try:
        # Import the trading engine which already has Schwab configured
        from trading_bot_commentary_updated import TradingEngineWithCommentary, TradingMode
        
        logger.info("Initializing trading engine...")
        
        # Create engine instance
        engine = TradingEngineWithCommentary(mode=TradingMode.LIVE)
        
        # Check if Schwab client is initialized
        if not engine.schwab_client:
            logger.error("Schwab client not initialized. Check your token configuration.")
            return
        
        logger.info("Schwab client initialized successfully")
        
        # Get account info multiple times to see consistency
        print("\n" + "="*80)
        print("CHECKING P&L VALUES (3 attempts)")
        print("="*80)
        
        for i in range(3):
            print(f"\n--- Attempt {i+1} ---")
            
            # Get account info
            account_info = await engine._get_real_account_info()
            
            if account_info:
                print(f"Balance: ${account_info.get('balance', 0):,.2f}")
                print(f"Buying Power: ${account_info.get('buying_power', 0):,.2f}")
                print(f"Cash: ${account_info.get('cash', 0):,.2f}")
                print(f"Daily P&L: ${account_info.get('day_pnl', 0):,.2f}")
                print(f"Initial Balance: ${account_info.get('initial_balance', 0):,.2f}")
                
                # Check if P&L makes sense
                day_pnl = account_info.get('day_pnl', 0)
                balance = account_info.get('balance', 0)
                
                if balance > 0:
                    pnl_percent = (day_pnl / balance) * 100
                    print(f"P&L as % of balance: {pnl_percent:.2f}%")
                    
                    if abs(pnl_percent) > 10:
                        print("⚠️  WARNING: P&L is more than 10% of account - this seems wrong!")
                
                # Get positions to check individual P&L
                positions = await engine.get_schwab_positions()
                if positions:
                    print(f"\nPositions ({len(positions)}):")
                    total_position_pnl = 0
                    for pos in positions:
                        symbol = pos.get('symbol', 'Unknown')
                        qty = pos.get('quantity', 0)
                        pnl = pos.get('total_pnl', 0)
                        pnl_pct = pos.get('pnl_percent', 0)
                        print(f"  {symbol}: {qty} shares, P&L=${pnl:.2f} ({pnl_pct:.1f}%)")
                        total_position_pnl += pnl
                    
                    print(f"\nTotal Position P&L: ${total_position_pnl:.2f}")
                    print(f"Reported Daily P&L: ${day_pnl:.2f}")
                    
                    diff = abs(total_position_pnl - day_pnl)
                    if diff > 100:
                        print(f"⚠️  WARNING: Position P&L and Daily P&L differ by ${diff:.2f}")
            else:
                print("Failed to get account info")
            
            # Wait a bit between attempts
            if i < 2:
                await asyncio.sleep(2)
        
        print("\n" + "="*80)
        print("ANALYSIS")
        print("="*80)
        
        # Check risk manager values
        print(f"\nRisk Manager Values:")
        print(f"  Account Balance: ${engine.risk_manager.account_balance:.2f}")
        print(f"  Daily P&L (internal): ${engine.risk_manager.daily_pnl:.2f}")
        print(f"  Schwab Daily P&L: ${engine.risk_manager.schwab_daily_pnl:.2f}")
        print(f"  5% Loss Limit: ${engine.risk_manager.account_balance * 0.05:.2f}")
        
        # Check if emergency stop would trigger
        pnl_check = engine.risk_manager.schwab_daily_pnl if engine.risk_manager.schwab_daily_pnl != 0 else engine.risk_manager.daily_pnl
        would_trigger = pnl_check < 0 and abs(pnl_check) > engine.risk_manager.account_balance * 0.05
        
        print(f"\nEmergency Stop Check:")
        print(f"  P&L being checked: ${pnl_check:.2f}")
        print(f"  Would trigger emergency stop: {'YES ⚠️' if would_trigger else 'NO ✅'}")
        
        if would_trigger:
            print(f"  ⚠️  Emergency stop would trigger with loss of ${abs(pnl_check):.2f}")
            print(f"  ⚠️  This exceeds 5% limit of ${engine.risk_manager.account_balance * 0.05:.2f}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        if 'engine' in locals():
            engine.is_running = False

if __name__ == "__main__":
    print("Starting P&L debug...")
    asyncio.run(debug_pnl())