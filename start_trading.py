#!/usr/bin/env python3
"""
Start Trading Bot
=================
Easy launcher for the institutional trading bot.
Usage: python start_trading.py [--paper] [--live] [--monitor-only]
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars are set another way


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
    warnings = []

    # Check .env file
    if not Path(".env").exists():
        warnings.append("No .env file - using environment variables")

    # Check token file
    if not Path("token_1.json").exists():
        errors.append("Missing token_1.json - run setup_wizard.py first")

    # Check API credentials
    if not os.getenv("SCHWAB_API_KEY") and not os.getenv("SCHWAB_APP_KEY"):
        errors.append("SCHWAB_API_KEY or SCHWAB_APP_KEY not set")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description='Start the Trading Bot')
    parser.add_argument('--paper', action='store_true',
                        help='Run in paper trading mode (default)')
    parser.add_argument('--live', action='store_true',
                        help='Run in live trading mode')
    parser.add_argument('--monitor-only', action='store_true',
                        help='Monitor only, no trading')
    parser.add_argument('--port', type=int, default=8000,
                        help='Web interface port')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Host to bind to')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open browser on start')
    args = parser.parse_args()

    print_banner()

    # Check prerequisites
    errors, warnings = check_prerequisites()

    for warning in warnings:
        print(f"⚠️  {warning}")

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

    # Import and run the trading bot
    try:
        from trading_bot_commentary_updated import app
        import uvicorn

        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info"
        )

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down gracefully...")
        print("   Saving state and closing positions if needed...")
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
