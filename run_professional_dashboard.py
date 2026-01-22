#!/usr/bin/env python3
"""
Professional Trading Bot Runner

Starts the trading bot with the professional dashboard interface.

Usage:
    python run_professional_dashboard.py [--port PORT] [--mode MODE]

Example:
    python run_professional_dashboard.py --port 9000 --mode simulation
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Professional Trading Bot")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=9000,
        help="Port to run the server on (default: 9000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["simulation", "paper", "live"],
        default="simulation",
        help="Trading mode (default: simulation)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║            Professional Trading Bot Dashboard                 ║
╠══════════════════════════════════════════════════════════════╣
║  Mode: {args.mode.upper():54} ║
║  Server: http://{args.host}:{args.port:<43} ║
║  Dashboard: http://localhost:{args.port:<30} ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Import and create the app
    try:
        from core.config import Config, TradingMode
        from core.engine import TradingEngine
        from api.app import create_app

        # Create configuration
        config = Config()
        config.trading.mode = TradingMode(args.mode)

        # Create engine
        engine = TradingEngine(config)

        # Create FastAPI app with engine
        app = create_app(engine=engine)

        logger.info(f"Starting server on {args.host}:{args.port}")
        logger.info(f"Trading mode: {args.mode}")
        logger.info(f"Open http://localhost:{args.port} in your browser")

        # Run server
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            reload=args.reload
        )

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Falling back to standalone mode...")

        # Fallback: run API without full engine
        from api.app import create_app
        app = create_app()

        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info"
        )


if __name__ == "__main__":
    main()
