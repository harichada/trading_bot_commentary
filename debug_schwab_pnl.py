#!/usr/bin/env python3
"""
Debug script to check actual Schwab P&L values
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
from schwab import auth, client

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def debug_schwab_pnl():
    """Debug Schwab P&L calculation"""
    
    try:
        # Load token
        token_path = Path("token_1.json")
        if not token_path.exists():
            logger.error("Token file not found")
            return
            
        # Initialize Schwab client using auth.client_from_token_file
        # This is the correct way to initialize from a saved token
        from schwab import auth
        
        # You need to provide app_key and app_secret from your Schwab app
        # These should match what's in your main trading bot
        app_key = "your_app_key_here"  # You'll need to get this from your config
        app_secret = "your_app_secret_here"  # You'll need to get this from your config
        
        # For now, let's try to load from the existing token file
        try:
            schwab_client = auth.client_from_token_file(
                token_path,
                app_key,
                app_secret
            )
        except Exception as e:
            # Alternative: Use manual token loading
            logger.info("Trying alternative client initialization...")
            with open(token_path, 'r') as f:
                token_data = json.load(f)
            
            # Create client with just the token data
            schwab_client = client.Client(app_key, app_secret)
            schwab_client.set_token(token_data)
        
        # Get account numbers
        response = schwab_client.get_account_numbers()
        if response.status_code != 200:
            logger.error(f"Failed to get account numbers: {response.status_code}")
            return
            
        accounts = response.json()
        if not accounts:
            logger.error("No accounts found")
            return
            
        account_hash = accounts[0]['hashValue']
        logger.info(f"Using account hash: {account_hash[:8]}...")
        
        # Get full account details
        from schwab.client import Client
        response = schwab_client.get_account(
            account_hash,
            fields=[Client.Account.Fields.POSITIONS]
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to get account: {response.status_code}")
            return
            
        data = response.json()
        account = data.get('securitiesAccount', {})
        
        print("\n" + "="*80)
        print("SCHWAB ACCOUNT DATA STRUCTURE")
        print("="*80)
        
        # Current balances
        current_balances = account.get('currentBalances', {})
        print("\nCurrent Balances:")
        for key, value in current_balances.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: ${value:,.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Initial balances
        initial_balances = account.get('initialBalances', {})
        print("\nInitial Balances:")
        for key, value in initial_balances.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: ${value:,.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Projected balances
        projected = account.get('projectedBalances', {})
        if projected:
            print("\nProjected Balances:")
            for key, value in projected.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: ${value:,.2f}")
                else:
                    print(f"  {key}: {value}")
        
        # Calculate P&L different ways
        print("\n" + "="*80)
        print("P&L CALCULATIONS")
        print("="*80)
        
        # Method 1: Direct field
        day_pnl_direct = projected.get('dayTradingGainLoss', 0)
        print(f"\n1. Direct dayTradingGainLoss: ${day_pnl_direct:,.2f}")
        
        # Method 2: From liquidation value
        current_liq = current_balances.get('liquidationValue', 0)
        prev_liq = initial_balances.get('liquidationValue', 0)
        pnl_from_liq = current_liq - prev_liq if prev_liq > 0 else 0
        print(f"\n2. From liquidation value change:")
        print(f"   Current: ${current_liq:,.2f}")
        print(f"   Previous: ${prev_liq:,.2f}")
        print(f"   Change: ${pnl_from_liq:,.2f}")
        
        # Method 3: From account value
        current_acct = current_balances.get('accountValue', 0)
        prev_acct = initial_balances.get('accountValue', 0)
        pnl_from_acct = current_acct - prev_acct if prev_acct > 0 else 0
        print(f"\n3. From account value change:")
        print(f"   Current: ${current_acct:,.2f}")
        print(f"   Previous: ${prev_acct:,.2f}")
        print(f"   Change: ${pnl_from_acct:,.2f}")
        
        # Cash changes
        current_cash = current_balances.get('cashBalance', 0)
        initial_cash = initial_balances.get('cashBalance', 0)
        cash_change = current_cash - initial_cash
        print(f"\n4. Cash movement:")
        print(f"   Current cash: ${current_cash:,.2f}")
        print(f"   Initial cash: ${initial_cash:,.2f}")
        print(f"   Change: ${cash_change:,.2f}")
        
        # Position P&L
        positions = account.get('positions', [])
        if positions:
            print(f"\n5. Position P&L:")
            total_position_pnl = 0
            for pos in positions:
                symbol = pos.get('instrument', {}).get('symbol', 'Unknown')
                long_qty = pos.get('longQuantity', 0)
                short_qty = pos.get('shortQuantity', 0)
                avg_price = pos.get('averagePrice', 0)
                market_value = pos.get('marketValue', 0)
                
                # Calculate cost basis
                net_qty = long_qty - short_qty
                cost_basis = avg_price * abs(net_qty)
                position_pnl = market_value - cost_basis if net_qty > 0 else cost_basis - market_value
                
                print(f"   {symbol}: Qty={net_qty}, Avg=${avg_price:.2f}, Value=${market_value:.2f}, P&L=${position_pnl:.2f}")
                total_position_pnl += position_pnl
            
            print(f"   TOTAL Position P&L: ${total_position_pnl:,.2f}")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        # Determine most likely correct P&L
        if day_pnl_direct != 0:
            print(f"Using direct field: ${day_pnl_direct:,.2f}")
        elif abs(cash_change) < 1000:  # No large cash movements
            print(f"Using liquidation value change: ${pnl_from_liq:,.2f}")
        else:
            print(f"Large cash movement detected (${cash_change:,.2f})")
            print(f"P&L calculation unreliable - need position-level data")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(debug_schwab_pnl())