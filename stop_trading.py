#!/usr/bin/env python3
"""
Stop Trading Bot
================
Safely stop the trading bot.
"""

import os
import sys
import signal
import logging
import requests

logger = logging.getLogger('TradingBot')

def main():
    print("🛑 Stopping Trading Bot...")

    # Try graceful shutdown via API
    try:
        response = requests.post("http://localhost:8000/api/shutdown", timeout=5)
        if response.status_code == 200:
            print("✓ Graceful shutdown initiated")
            return
    except requests.RequestException as e:
        logger.debug(f"API shutdown unavailable, falling back to signal: {e}")

    # Find and kill process
    import subprocess
    result = subprocess.run(
        ["pgrep", "-f", "trading_bot_commentary_updated"],
        capture_output=True, text=True
    )

    if result.stdout.strip():
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
                print(f"✓ Sent SIGTERM to process {pid}")
            except (ProcessLookupError, PermissionError, ValueError) as e:
                logger.warning(f"Failed to send SIGTERM to {pid}: {e}")
    else:
        print("No trading bot process found")

if __name__ == "__main__":
    main()
