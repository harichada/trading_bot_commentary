#!/usr/bin/env python3
"""
Start Trading Bot
=================
Easy launcher for the institutional trading bot.
Usage: python start_trading.py [--paper] [--live] [--monitor-only]
"""

import os
import sys
import signal
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def print_banner():
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║         INSTITUTIONAL TRADING BOT v2.0                       ║
    ║         ────────────────────────────────────                 ║
    ║         Press Ctrl+C to stop safely                          ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_prerequisites():
    """Check all required files and credentials"""
    errors = []

    # Check .env file
    if not Path(".env").exists():
        errors.append("Missing .env file - run setup_wizard.py first")

    # Check token file
    if not Path("token_1.json").exists():
        errors.append("Missing token_1.json - run setup_wizard.py first")

    # Check API credentials
    if not os.getenv("SCHWAB_API_KEY"):
        errors.append("SCHWAB_API_KEY not set")
    if not os.getenv("SCHWAB_APP_SECRET"):
        errors.append("SCHWAB_APP_SECRET not set")

    return errors

def main():
    parser = argparse.ArgumentParser(description='Start the Trading Bot')
    parser.add_argument('--paper', action='store_true', help='Run in paper trading mode (default)')
    parser.add_argument('--live', action='store_true', help='Run in live trading mode')
    parser.add_argument('--monitor-only', action='store_true', help='Monitor only, no trading')
    parser.add_argument('--port', type=int, default=8000, help='Web interface port')
    args = parser.parse_args()

    print_banner()

    # Check prerequisites
    errors = check_prerequisites()
    if errors:
        print("\n❌ Setup incomplete:")
        for error in errors:
            print(f"   • {error}")
        print("\nRun: python setup_wizard.py")
        sys.exit(1)

    # Determine trading mode
    if args.live:
        os.environ['PAPER_TRADING'] = 'false'
        os.environ['TRADING_ENV'] = 'production'
        mode = "🔴 LIVE TRADING"
        print(f"\n⚠️  WARNING: Running in LIVE mode - real money at risk!")
        confirm = input("Type 'LIVE' to confirm: ")
        if confirm != 'LIVE':
            print("Aborted.")
            sys.exit(0)
    else:
        os.environ['PAPER_TRADING'] = 'true'
        os.environ['TRADING_ENV'] = 'paper'
        mode = "📝 PAPER TRADING"

    if args.monitor_only:
        os.environ['MONITOR_ONLY'] = 'true'
        mode += " (Monitor Only)"

    print(f"\n🚀 Starting in {mode} mode...")
    print(f"📊 Dashboard: http://localhost:{args.port}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "─" * 60)

    # Register signal handlers for graceful shutdown
    def shutdown_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n🛑 Received {sig_name}, shutting down gracefully...")
        print("   Saving state and closing positions if needed...")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Import and run the trading bot
    try:
        from trading_bot_commentary_updated import app
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")

    except SystemExit:
        pass  # Expected from shutdown_handler
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
