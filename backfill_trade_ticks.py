#!/usr/bin/env python3
"""Download historical trade ticks from Alpaca and store in PostgreSQL.

Usage:
    python backfill_trade_ticks.py                    # All 50 symbols, last 6 months
    python backfill_trade_ticks.py --symbols AAPL MSFT # Specific symbols
    python backfill_trade_ticks.py --days 30           # Last 30 trading days only
    python backfill_trade_ticks.py --start 2025-01-01 --end 2025-03-07
"""

import argparse
import asyncio
import json
import logging
import os
import io
import time
from datetime import datetime, timedelta, date
from typing import List, Optional, Tuple

import psycopg2
import psycopg2.extras
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-6s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────

DB_URL = os.environ.get('DATABASE_URL', 'postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev')

ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', '')
ALPACA_DATA_URL = 'https://data.alpaca.markets/v2/stocks'

# Rate limit: 10,000 req/min (paid SIP plan) — no throttle needed
RATE_LIMIT_PER_SEC = 0  # 0 = no delay between requests
PAGE_LIMIT = 10000  # Max trades per API call

# 50 liquid large-cap symbols with high trade volume
DEFAULT_SYMBOLS = [
    # Mega-cap tech
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'META', 'TSLA',
    # Semiconductors
    'AMD', 'AVGO', 'MU', 'QCOM', 'INTC', 'MRVL', 'AMAT',
    # Finance
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C',
    # ETFs
    'SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'SOXL', 'TQQQ',
    # High ADR / popular day-trading
    'NFLX', 'CRM', 'SHOP', 'SQ', 'COIN', 'MARA', 'RIOT',
    'PLTR', 'UBER', 'SNAP', 'ROKU', 'DKNG', 'SOFI',
    # Energy / industrials
    'XOM', 'CVX', 'BA', 'CAT',
    # Healthcare
    'UNH', 'JNJ', 'PFE', 'MRNA',
    # Other liquid
    'DIS', 'NKE', 'PYPL',
]

# ── Database ────────────────────────────────────────────────────────────

def create_table(conn):
    """Create trade_ticks table if not exists."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trade_ticks (
                ts          TIMESTAMPTZ NOT NULL,
                symbol      TEXT NOT NULL,
                price       NUMERIC(12,4) NOT NULL,
                size        INTEGER NOT NULL,
                exchange    CHAR(1),
                conditions  TEXT
            );
        """)
        # Create index for efficient queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_trade_ticks_symbol_ts
            ON trade_ticks (symbol, ts);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_trade_ticks_ts
            ON trade_ticks (ts);
        """)
    conn.commit()
    logger.info("trade_ticks table ready")


def get_existing_dates(conn, symbol: str) -> set:
    """Get dates we already have data for (to skip re-downloading)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ts::date
            FROM trade_ticks
            WHERE symbol = %s
        """, (symbol,))
        return {row[0] for row in cur.fetchall()}


def bulk_insert_trades(conn, trades: list):
    """Bulk insert trades using COPY for maximum speed."""
    if not trades:
        return 0

    buf = io.StringIO()
    for t in trades:
        # ts, symbol, price, size, exchange, conditions
        conditions = t.get('conditions', '')
        if isinstance(conditions, list):
            conditions = ','.join(conditions)
        line = f"{t['ts']}\t{t['symbol']}\t{t['price']}\t{t['size']}\t{t.get('exchange', '')}\t{conditions}\n"
        buf.write(line)

    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_from(buf, 'trade_ticks',
                      columns=('ts', 'symbol', 'price', 'size', 'exchange', 'conditions'),
                      null='')
    conn.commit()
    return len(trades)


# ── Alpaca API ──────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    """Create a requests session with retries and auth headers."""
    session = requests.Session()
    session.headers.update({
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
    })
    retry = Retry(total=3, backoff_factor=1.0,
                  status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    return session


# Global session (created in main)
_session: Optional[requests.Session] = None


def fetch_and_insert_day(conn, symbol: str, trade_date: date) -> int:
    """Fetch all trades for a symbol on a given day, inserting page-by-page.
    Returns total trades inserted."""
    # Regular market hours only: 9:30-16:00 ET
    start = f"{trade_date}T09:30:00-05:00"
    end = f"{trade_date}T16:00:00-05:00"

    page_token = None
    total = 0
    pages = 0

    while True:
        params = {
            'start': start,
            'end': end,
            'limit': str(PAGE_LIMIT),
            'feed': 'sip',
        }
        if page_token:
            params['page_token'] = page_token

        url = f"{ALPACA_DATA_URL}/{symbol}/trades"

        try:
            resp = _session.get(url, params=params, timeout=60)
            if resp.status_code == 422:
                return 0
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError:
            logger.error(f"  HTTP {resp.status_code} for {symbol} {trade_date}")
            return total
        except Exception as e:
            logger.error(f"  Error fetching {symbol} {trade_date}: {e}")
            return total

        trades_raw = data.get('trades', [])
        if trades_raw:
            batch = []
            for t in trades_raw:
                batch.append({
                    'ts': t['t'],
                    'symbol': symbol,
                    'price': t['p'],
                    'size': t['s'],
                    'exchange': t.get('x', ''),
                    'conditions': t.get('c', []),
                })
            inserted = bulk_insert_trades(conn, batch)
            total += inserted
            pages += 1

            if pages % 20 == 0:
                logger.info(f"      {symbol} {trade_date}: page {pages}, {total:,} trades so far...")

        # Rate limiting (skip if no limit set)
        if RATE_LIMIT_PER_SEC > 0:
            time.sleep(1.0 / RATE_LIMIT_PER_SEC)

        # Pagination
        page_token = data.get('next_page_token')
        if not page_token:
            break

    return total


# ── Trading Calendar ────────────────────────────────────────────────────

def get_trading_days(start_date: date, end_date: date) -> List[date]:
    """Get trading days from Alpaca calendar API."""
    url = "https://paper-api.alpaca.markets/v2/calendar"
    params = {'start': start_date.isoformat(), 'end': end_date.isoformat()}
    try:
        resp = _session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return [date.fromisoformat(d['date']) for d in resp.json()]
    except Exception as e:
        logger.error(f"Failed to get trading calendar: {e}")
        # Fallback: weekdays only
        days = []
        d = start_date
        while d <= end_date:
            if d.weekday() < 5:  # Mon-Fri
                days.append(d)
            d += timedelta(days=1)
        return days


# ── Main Loop ───────────────────────────────────────────────────────────

def backfill_symbol(conn, symbol: str, trading_days: List[date]) -> Tuple[int, int]:
    """Download all missing days for one symbol. Returns (trades_inserted, days_processed)."""
    existing = get_existing_dates(conn, symbol)
    missing_days = [d for d in trading_days if d not in existing]

    if not missing_days:
        logger.info(f"  {symbol}: all {len(trading_days)} days already downloaded, skipping")
        return 0, 0

    logger.info(f"  {symbol}: {len(missing_days)} days to download ({len(existing)} already done)")

    total_trades = 0
    days_done = 0

    for i, trade_date in enumerate(missing_days):
        day_trades = fetch_and_insert_day(conn, symbol, trade_date)
        total_trades += day_trades

        if day_trades > 0 or (i + 1) % 10 == 0 or i == len(missing_days) - 1:
            logger.info(f"    {symbol} [{i+1}/{len(missing_days)}] "
                       f"{trade_date}: {day_trades:,} trades (total: {total_trades:,})")
        days_done += 1

    return total_trades, days_done


def main():
    parser = argparse.ArgumentParser(description='Backfill trade ticks from Alpaca')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Symbols to download (default: 50 liquid stocks)')
    parser.add_argument('--days', type=int, default=None,
                       help='Number of trading days to look back')
    parser.add_argument('--start', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    # Validate API keys
    global ALPACA_API_KEY, ALPACA_SECRET_KEY
    if not ALPACA_API_KEY:
        # Try loading from .env
        env_path = os.path.join(os.path.dirname(__file__) or '.', '.env')
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('ALPACA_API_KEY='):
                        ALPACA_API_KEY = line.split('=', 1)[1]
                    elif line.startswith('ALPACA_SECRET_KEY='):
                        ALPACA_SECRET_KEY = line.split('=', 1)[1]

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logger.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        return

    # Symbols
    symbols = args.symbols or DEFAULT_SYMBOLS
    logger.info(f"Symbols: {len(symbols)} — {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

    # Date range
    if args.start:
        start_date = date.fromisoformat(args.start)
    elif args.days:
        start_date = date.today() - timedelta(days=int(args.days * 1.5))  # Extra for weekends
    else:
        start_date = date.today() - timedelta(days=270)  # ~6 months

    end_date = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)

    logger.info(f"Date range: {start_date} to {end_date}")

    # Create HTTP session (needed for calendar + data calls)
    global _session
    _session = _make_session()

    # Get trading calendar
    trading_days = get_trading_days(start_date, end_date)
    logger.info(f"Trading days: {len(trading_days)}")

    if not trading_days:
        logger.error("No trading days in range")
        return

    # Connect to DB
    conn = psycopg2.connect(DB_URL)
    create_table(conn)

    # Download
    t0 = time.time()
    grand_total_trades = 0
    grand_total_days = 0

    for idx, symbol in enumerate(symbols):
        logger.info(f"[{idx+1}/{len(symbols)}] Processing {symbol}...")
        try:
            trades, days = backfill_symbol(conn, symbol, trading_days)
            grand_total_trades += trades
            grand_total_days += days
        except Exception as e:
            logger.error(f"  FAILED {symbol}: {e}")
            conn.rollback()
            continue

    elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"  DONE in {elapsed/60:.1f} minutes")
    logger.info(f"  Symbols: {len(symbols)}")
    logger.info(f"  Days processed: {grand_total_days}")
    logger.info(f"  Trades inserted: {grand_total_trades:,}")
    logger.info(f"{'='*60}")

    # Summary query
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(ts::date), MAX(ts::date) FROM trade_ticks")
        row = cur.fetchone()
        logger.info(f"  DB totals: {row[0]:,} ticks, {row[1]} symbols, {row[2]} to {row[3]}")

    conn.close()


if __name__ == '__main__':
    main()
