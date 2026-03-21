#!/usr/bin/env python3
"""EOD Dead Man's Switch — Independent position liquidation.

Standalone script that runs via cron/systemd at 15:55 ET.
Closes ALL open positions on Alpaca regardless of bot state.
This is the last line of defense if the trading bot dies or hangs.

Usage:
    python eod_dead_man_switch.py              # Normal: only runs 15:50-16:10 ET
    python eod_dead_man_switch.py --force      # Bypass time check
    python eod_dead_man_switch.py --dry-run    # Show positions, don't close
    python eod_dead_man_switch.py --force --dry-run  # Test anytime

Env vars required:
    ALPACA_TRADE_API_KEY / ALPACA_TRADE_SECRET_KEY (paper account)
    or ALPACA_API_KEY / ALPACA_SECRET_KEY (fallback)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

ET = ZoneInfo('US/Eastern')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/eod_dead_man_switch.log', mode='a'),
    ]
)
logger = logging.getLogger('DeadManSwitch')


def get_config() -> dict:
    """Load Alpaca credentials from env."""
    load_dotenv()
    api_key = os.environ.get('ALPACA_TRADE_API_KEY') or os.environ.get('ALPACA_API_KEY', '')
    secret = os.environ.get('ALPACA_TRADE_SECRET_KEY') or os.environ.get('ALPACA_SECRET_KEY', '')
    base_url = os.environ.get('ALPACA_TRADE_BASE_URL', 'https://paper-api.alpaca.markets')
    if not api_key or not secret:
        logger.error("No Alpaca credentials found in environment")
        sys.exit(1)
    return {
        'api_key': api_key,
        'secret': secret,
        'base_url': base_url,
        'headers': {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': secret,
        },
    }


def get_positions(cfg: dict) -> list:
    """Get all open positions from Alpaca."""
    resp = requests.get(f'{cfg["base_url"]}/v2/positions',
                        headers=cfg['headers'], timeout=10)
    if resp.status_code == 200:
        return resp.json()
    logger.error(f"GET /v2/positions failed: {resp.status_code} {resp.text[:200]}")
    return []


def cancel_all_orders(cfg: dict) -> int:
    """Cancel all open orders."""
    resp = requests.delete(f'{cfg["base_url"]}/v2/orders',
                           headers=cfg['headers'], timeout=10)
    if resp.status_code in (200, 204, 207):
        cancelled = resp.json() if resp.status_code == 207 else []
        logger.info(f"Cancelled {len(cancelled)} open orders")
        return len(cancelled)
    logger.warning(f"DELETE /v2/orders: {resp.status_code}")
    return 0


def close_position(cfg: dict, symbol: str) -> bool:
    """Close a single position via Alpaca DELETE API."""
    resp = requests.delete(f'{cfg["base_url"]}/v2/positions/{symbol}',
                           headers=cfg['headers'], timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        logger.info(f"CLOSED {symbol}: {data.get('qty')} shares")
        return True
    logger.error(f"CLOSE {symbol} failed: {resp.status_code} {resp.text[:200]}")
    return False


def send_alert(message: str):
    """Send alert via Discord webhook if configured."""
    webhook = os.environ.get('DISCORD_WEBHOOK_URL', '')
    if not webhook:
        return
    try:
        requests.post(webhook, json={'content': f'🚨 **Dead Man Switch**: {message}'},
                      timeout=5)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description='EOD Dead Man Switch')
    parser.add_argument('--force', action='store_true',
                        help='Bypass time window check')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show positions without closing')
    args = parser.parse_args()

    now = datetime.now(ET)
    logger.info(f"Dead Man Switch triggered at {now.strftime('%Y-%m-%d %H:%M:%S ET')}")

    # Time gate: only run 15:50-16:10 ET on weekdays
    if not args.force:
        if now.weekday() >= 5:
            logger.info("Weekend — skipping")
            sys.exit(0)
        minutes_since_midnight = now.hour * 60 + now.minute
        if not (950 <= minutes_since_midnight <= 970):  # 15:50 - 16:10
            logger.info(f"Outside EOD window ({now.strftime('%H:%M')} ET) — skipping")
            sys.exit(0)

    cfg = get_config()

    # Check positions
    positions = get_positions(cfg)
    if not positions:
        logger.info("No open positions — nothing to do")
        sys.exit(0)

    # Report positions
    total_value = 0.0
    total_pl = 0.0
    for p in positions:
        sym = p.get('symbol', '?')
        qty = p.get('qty', '?')
        side = p.get('side', '?')
        upl = float(p.get('unrealized_pl', 0))
        mv = float(p.get('market_value', 0))
        total_value += abs(mv)
        total_pl += upl
        logger.info(f"  {side.upper()} {qty} {sym} | MV: ${mv:,.2f} | P&L: ${upl:+,.2f}")

    logger.info(f"Total: {len(positions)} positions, ${total_value:,.2f} value, ${total_pl:+,.2f} unrealized P&L")

    if args.dry_run:
        logger.info("DRY RUN — no actions taken")
        sys.exit(0)

    # LIQUIDATE
    alert_msg = f"{len(positions)} positions open at {now.strftime('%H:%M')} ET — liquidating"
    logger.warning(alert_msg)
    send_alert(alert_msg)

    # Cancel all orders first (frees held shares)
    cancel_all_orders(cfg)

    # Close each position
    import time
    success = 0
    for p in positions:
        sym = p.get('symbol', '')
        if not sym:
            continue
        time.sleep(0.5)  # brief pause between closes
        if close_position(cfg, sym):
            success += 1

    # Verify
    time.sleep(2)
    remaining = get_positions(cfg)
    if remaining:
        rem_syms = [p.get('symbol') for p in remaining]
        msg = f"CRITICAL: {len(remaining)} positions still open after liquidation: {rem_syms}"
        logger.error(msg)
        send_alert(msg)
        sys.exit(1)

    msg = f"All {success} positions closed successfully"
    logger.info(msg)
    send_alert(msg)
    sys.exit(0)


if __name__ == '__main__':
    main()
