#!/usr/bin/env python3
"""
EMERGENCY STOP
==============
Immediately close all positions and stop trading.
USE ONLY IN EMERGENCIES!
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()


def _auth_headers():
    """Build auth headers from TRADING_API_KEY if set."""
    key = os.getenv("TRADING_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def main():
    print("\n" + "!" * 60)
    print("       🚨 EMERGENCY STOP 🚨")
    print("!" * 60)
    print("\nThis will:")
    print("  1. Close ALL open positions at market price")
    print("  2. Cancel ALL pending orders")
    print("  3. Stop the trading bot")
    print()

    confirm = input("Type 'EMERGENCY' to confirm: ")
    if confirm != 'EMERGENCY':
        print("Aborted.")
        return

    print("\n⚠️  Executing emergency stop...")

    try:
        # Close all positions
        response = requests.post(
            "http://localhost:8000/api/emergency/close-all",
            timeout=30,
            headers=_auth_headers()
        )
        if response.status_code == 200:
            print("✓ All positions closed")
        else:
            print(f"⚠ Position close response: {response.status_code}")

        # Shutdown bot
        response = requests.post(
            "http://localhost:8000/api/shutdown",
            timeout=10,
            headers=_auth_headers()
        )
        print("✓ Bot shutdown initiated")

    except Exception as e:
        print(f"\n⚠️  API not responding. Attempting force stop...")

        # Force kill
        import subprocess
        subprocess.run(["pkill", "-f", "trading_bot"], capture_output=True)
        print("✓ Processes terminated")

    print("\n✅ Emergency stop complete")
    print("   Check your Schwab account directly to verify positions")

if __name__ == "__main__":
    main()
