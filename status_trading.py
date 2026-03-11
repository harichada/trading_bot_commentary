#!/usr/bin/env python3
"""
Trading Bot Status
==================
Check the current status of the trading bot.
"""

import os
import sys
import json
import logging
import requests
from datetime import datetime

logger = logging.getLogger('TradingBot')

def get_status():
    """Get bot status from API"""
    try:
        response = requests.get("http://localhost:8000/api/status", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException as e:
        logger.debug(f"Status API unavailable: {e}")
        return None

def get_positions():
    """Get current positions"""
    try:
        response = requests.get("http://localhost:8000/api/positions", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException as e:
        logger.debug(f"Positions API unavailable: {e}")
        return None

def get_pnl():
    """Get P&L"""
    try:
        response = requests.get("http://localhost:8000/api/pnl", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException as e:
        logger.debug(f"P&L API unavailable: {e}")
        return None

def main():
    print("\n" + "═" * 60)
    print("       TRADING BOT STATUS")
    print("═" * 60)

    status = get_status()

    if not status:
        print("\n❌ Bot is NOT running or not responding")
        print("   Start with: python start_trading.py")
        return

    print(f"\n✅ Bot is RUNNING")
    print(f"   Mode: {status.get('mode', 'Unknown')}")
    print(f"   State: {status.get('state', 'Unknown')}")
    print(f"   Uptime: {status.get('uptime', 'Unknown')}")

    # Positions
    positions = get_positions()
    if positions:
        print(f"\n📊 POSITIONS ({len(positions)} open)")
        print("   " + "-" * 50)
        for pos in positions:
            symbol = pos.get('symbol', '?')
            qty = pos.get('quantity', 0)
            pnl = pos.get('unrealized_pnl', 0)
            pnl_color = '🟢' if pnl >= 0 else '🔴'
            print(f"   {symbol}: {qty} shares | P&L: {pnl_color} ${pnl:,.2f}")

    # P&L
    pnl_data = get_pnl()
    if pnl_data:
        daily = pnl_data.get('daily_pnl', 0)
        total = pnl_data.get('total_pnl', 0)
        print(f"\n💰 P&L")
        print("   " + "-" * 50)
        print(f"   Today: {'🟢' if daily >= 0 else '🔴'} ${daily:,.2f}")
        print(f"   Total: {'🟢' if total >= 0 else '🔴'} ${total:,.2f}")

    print("\n" + "═" * 60)
    print(f"   Dashboard: http://localhost:8000")
    print(f"   Checked at: {datetime.now().strftime('%H:%M:%S')}")
    print("═" * 60 + "\n")

if __name__ == "__main__":
    main()
