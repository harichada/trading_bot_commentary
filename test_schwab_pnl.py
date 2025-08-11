#!/usr/bin/env python3
"""
Test script to verify Schwab P&L calculation is working correctly
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_schwab_pnl():
    """Test the Schwab P&L calculation"""
    
    try:
        # Import the trading engine
        from trading_bot_commentary_updated import TradingEngineWithCommentary, Config, TradingMode
        
        # Initialize the engine
        config = Config()
        engine = TradingEngineWithCommentary(config)
        
        # Initialize in simulation mode first
        await engine.initialize(mode=TradingMode.SIMULATION_WITH_COMMENTARY)
        
        # Check if Schwab client is available
        if not engine.schwab_client:
            logger.error("Schwab client not initialized. Please check your token configuration.")
            return
        
        logger.info("Schwab client initialized successfully")
        
        # Get account info with P&L
        logger.info("Fetching account information from Schwab...")
        account_info = await engine._get_real_account_info()
        
        if account_info:
            logger.info("=" * 60)
            logger.info("SCHWAB ACCOUNT INFORMATION")
            logger.info("=" * 60)
            logger.info(f"Account Balance: ${account_info.get('balance', 0):,.2f}")
            logger.info(f"Buying Power: ${account_info.get('buying_power', 0):,.2f}")
            logger.info(f"Cash: ${account_info.get('cash', 0):,.2f}")
            logger.info(f"Initial Balance: ${account_info.get('initial_balance', 0):,.2f}")
            logger.info("=" * 60)
            logger.info(f"DAILY P&L: ${account_info.get('day_pnl', 0):,.2f}")
            logger.info("=" * 60)
            
            # Test risk manager sync
            logger.info("\nTesting Risk Manager Sync...")
            engine.risk_manager.sync_with_schwab_data(account_info)
            
            logger.info(f"Risk Manager Daily P&L: ${engine.risk_manager.daily_pnl:.2f}")
            logger.info(f"Risk Manager Schwab P&L: ${engine.risk_manager.schwab_daily_pnl:.2f}")
            logger.info(f"Risk Manager Buying Power: ${engine.risk_manager.buying_power:.2f}")
            logger.info(f"Risk Manager Account Balance: ${engine.risk_manager.account_balance:.2f}")
            
            # Get positions to check position-level P&L
            logger.info("\nFetching positions...")
            positions = await engine.get_schwab_positions()
            
            if positions:
                logger.info(f"Found {len(positions)} positions")
                total_position_pnl = 0
                
                for pos in positions:
                    symbol = pos.get('symbol', 'Unknown')
                    quantity = pos.get('quantity', 0)
                    avg_price = pos.get('averagePrice', 0)
                    current_price = pos.get('currentPrice', 0)
                    market_value = pos.get('marketValue', 0)
                    
                    # Calculate position P&L
                    position_pnl = (current_price - avg_price) * quantity
                    total_position_pnl += position_pnl
                    
                    logger.info(f"  {symbol}: Qty={quantity}, Avg=${avg_price:.2f}, "
                               f"Current=${current_price:.2f}, P&L=${position_pnl:.2f}")
                
                logger.info(f"\nTotal Position P&L: ${total_position_pnl:.2f}")
                logger.info(f"Schwab Reported P&L: ${account_info.get('day_pnl', 0):.2f}")
                
                # Check if they match
                diff = abs(total_position_pnl - account_info.get('day_pnl', 0))
                if diff > 1.0:  # Allow $1 difference for rounding
                    logger.warning(f"P&L Mismatch: Position calc=${total_position_pnl:.2f}, "
                                 f"Schwab=${account_info.get('day_pnl', 0):.2f}")
                else:
                    logger.info("✓ P&L calculations match!")
            else:
                logger.info("No positions found")
                
        else:
            logger.error("Failed to get account information from Schwab")
            
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        # Clean up
        if 'engine' in locals():
            engine.is_running = False
            logger.info("\nTest completed")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_schwab_pnl())