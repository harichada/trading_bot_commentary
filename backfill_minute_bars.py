#!/usr/bin/env python3
"""
Backfill 1-minute bars from Alpaca into PostgreSQL minute_bars table.

Features:
  - Converts all timestamps to naive Eastern Time (US/Eastern)
  - Filters to RTH only (9:30-15:59 ET)
  - Multi-day batch mode: up to 25 trading days per API request (25x speedup)
  - --top-liquid N: query DB for top N symbols by avg daily volume
  - --purge: delete existing data before re-downloading
  - --dry-run: show what would be downloaded
  - --rate-limit: optional delay between requests (default: no delay for paid plan)
  - Skips existing data by default (resumable)
  - Progress reporting with ETA

Usage:
    python backfill_minute_bars.py --liquid --start 2024-01-01
    python backfill_minute_bars.py --top-liquid 200 --start 2024-01-01
    python backfill_minute_bars.py --symbols AAPL MSFT --start 2024-06-01
    python backfill_minute_bars.py --start 2024-01-01 --end 2024-12-31 --purge
    python backfill_minute_bars.py --dry-run --top-liquid 500
"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_1min")

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev"
)

ALPACA_DATA_URL = "https://data.alpaca.markets"
ET = ZoneInfo("US/Eastern")

# Max trading days per API request: 390 bars/day * 25 = 9,750 < 10,000 limit
MAX_DAYS_PER_REQUEST = 25

# RTH bounds (inclusive)
RTH_OPEN_HOUR, RTH_OPEN_MIN = 9, 30
RTH_CLOSE_HOUR, RTH_CLOSE_MIN = 15, 59

# Curated liquid large/mid-cap symbols for intraday strategy backtesting
LIQUID_LARGE_CAPS = [
    # Mega-cap tech
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "AVGO",
    # Semi & hardware
    "AMD", "INTC", "MU", "QCOM", "AMAT", "LRCX", "KLAC",
    # Software & cloud
    "CRM", "ORCL", "NFLX", "ADBE", "NOW", "UBER", "SHOP",
    # Finance
    "JPM", "BAC", "GS", "MS", "V", "MA", "AXP",
    # Healthcare
    "UNH", "LLY", "ABBV", "JNJ", "PFE", "MRK",
    # Consumer
    "WMT", "COST", "HD", "MCD", "SBUX", "NKE", "DIS",
    # Staples
    "PG", "KO", "PEP",
    # Energy
    "XOM", "CVX", "SLB",
    # Industrial
    "CAT", "BA", "GE", "HON", "UPS",
    # ETFs (market benchmarks)
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV",
]


def load_env_keys() -> tuple[str, str]:
    """Load Alpaca API keys from env vars or .env file."""
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

    if not api_key or not secret_key:
        env_path = os.path.join(os.path.dirname(__file__) or ".", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        v = v.strip().strip('"').strip("'")
                        k = k.strip()
                        if k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
                            os.environ[k] = v
            api_key = os.environ.get("ALPACA_API_KEY", "")
            secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

    return api_key, secret_key


def get_gap_candidates(
    conn, start_date: str, end_date: str,
    min_gap_pct: float = 0.03, min_avg_vol: float = 50000,
) -> list[tuple[str, str]]:
    """Find (symbol, date) pairs where the stock gapped >min_gap_pct."""
    lookback_date = (
        datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=40)
    ).strftime("%Y-%m-%d")
    sql = """
    WITH bars_ext AS (
        SELECT symbol, date, open, close, volume,
               LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS prev_close,
               AVG(volume) OVER (PARTITION BY symbol ORDER BY date
                                 ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS avg_vol_20
        FROM daily_bars
        WHERE date >= %s AND date <= %s
    )
    SELECT symbol, date
    FROM bars_ext
    WHERE prev_close IS NOT NULL
      AND date >= %s
      AND ABS((open - prev_close) / NULLIF(prev_close, 0)) > %s
      AND avg_vol_20 > %s
    ORDER BY date, symbol
    """
    cur = conn.cursor()
    cur.execute(sql, (lookback_date, end_date, start_date, min_gap_pct, min_avg_vol))
    return [(row[0], str(row[1])) for row in cur.fetchall()]


def get_top_liquid_symbols(conn, n: int) -> list[str]:
    """Query DB for top N symbols by average daily volume in last year."""
    sql = """
    SELECT symbol, AVG(volume) as avg_vol FROM daily_bars
    WHERE date >= (CURRENT_DATE - INTERVAL '1 year')::text
    GROUP BY symbol HAVING COUNT(*) > 100
    ORDER BY avg_vol DESC LIMIT %s
    """
    cur = conn.cursor()
    cur.execute(sql, (n,))
    symbols = [row[0] for row in cur.fetchall()]
    log.info(f"Top-liquid query returned {len(symbols)} symbols (requested {n})")
    return symbols


def get_trading_dates(conn, start_date: str, end_date: str) -> list[str]:
    """Get trading dates from daily_bars (using SPY as reference)."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT date FROM daily_bars
        WHERE symbol = 'SPY' AND date >= %s AND date <= %s
        ORDER BY date
    """,
        (start_date, end_date),
    )
    return [str(row[0]) for row in cur.fetchall()]


def get_existing_symbol_dates(conn, symbols: list[str], start_date: str, end_date: str) -> set[tuple[str, str]]:
    """Get (symbol, date_str) pairs already in minute_bars.

    Uses a single efficient query with date range filtering.
    Returns set of (symbol, 'YYYY-MM-DD') tuples.
    """
    if not symbols:
        return set()

    cur = conn.cursor()
    # Chunk symbols to avoid overly large IN clauses
    existing = set()
    chunk_size = 500
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        placeholders = ",".join(["%s"] * len(chunk))
        cur.execute(
            f"""
            SELECT DISTINCT symbol, ts::date::text
            FROM minute_bars
            WHERE symbol IN ({placeholders})
              AND ts >= %s::timestamp
              AND ts < (%s::date + 1)::timestamp
            """,
            (*chunk, start_date, end_date),
        )
        for row in cur.fetchall():
            existing.add((row[0], str(row[1])))

    return existing


def purge_minute_bars(conn, symbols: list[str], start_date: str, end_date: str) -> int:
    """Delete existing minute_bars for the given symbols and date range."""
    if not symbols:
        return 0

    cur = conn.cursor()
    total_deleted = 0
    chunk_size = 500
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        placeholders = ",".join(["%s"] * len(chunk))
        cur.execute(
            f"""
            DELETE FROM minute_bars
            WHERE symbol IN ({placeholders})
              AND ts >= %s::timestamp
              AND ts < (%s::date + 1)::timestamp
            """,
            (*chunk, start_date, end_date),
        )
        total_deleted += cur.rowcount

    conn.commit()
    return total_deleted


def fetch_bars_multi_day(
    symbol: str,
    start_date: str,
    end_date: str,
    api_key: str,
    secret_key: str,
) -> list[dict]:
    """Fetch 1-min bars for a symbol across a date range from Alpaca.

    Handles pagination via next_page_token. Converts timestamps to naive ET.
    Filters to RTH only (9:30-15:59 ET).

    Args:
        symbol: Ticker symbol
        start_date: First trading date (YYYY-MM-DD)
        end_date: Last trading date (YYYY-MM-DD)
        api_key: Alpaca API key
        secret_key: Alpaca secret key

    Returns:
        List of dicts with keys: ts, open, high, low, close, volume
    """
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }

    # Build start/end as ET-aware timestamps
    dt_start = datetime.strptime(start_date, "%Y-%m-%d")
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")

    # Start at market open on start_date
    market_open = dt_start.replace(hour=4, minute=0, tzinfo=ET)
    # End after market close on end_date (next day midnight)
    market_end = (dt_end + timedelta(days=1)).replace(hour=0, minute=0, tzinfo=ET)

    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    all_bars = []
    page_token: Optional[str] = None

    while True:
        params = {
            "timeframe": "1Min",
            "start": market_open.isoformat(),
            "end": market_end.isoformat(),
            "limit": 10000,
            "feed": "sip",
            "adjustment": "raw",
        }
        if page_token:
            params["page_token"] = page_token

        data = _api_request_with_retry(url, headers, params, symbol, f"{start_date}..{end_date}")
        if data is None:
            break

        raw_bars = data.get("bars") or []
        for b in raw_bars:
            # Parse the Alpaca timestamp and convert to naive ET
            dt_utc = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
            dt_et = dt_utc.astimezone(ET).replace(tzinfo=None)

            # Filter to RTH only: 9:30 - 15:59
            bar_time = dt_et.hour * 100 + dt_et.minute
            if bar_time < RTH_OPEN_HOUR * 100 + RTH_OPEN_MIN:
                continue
            if bar_time > RTH_CLOSE_HOUR * 100 + RTH_CLOSE_MIN:
                continue

            all_bars.append({
                "ts": dt_et.strftime("%Y-%m-%d %H:%M:%S"),
                "open": b["o"],
                "high": b["h"],
                "low": b["l"],
                "close": b["c"],
                "volume": int(b["v"]),
            })

        page_token = data.get("next_page_token")
        if not page_token:
            break

    return all_bars


def _api_request_with_retry(
    url: str, headers: dict, params: dict, symbol: str, date_label: str,
    max_retries: int = 5,
) -> Optional[dict]:
    """Make an API request with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:
                wait = min(2 ** (attempt + 1), 60)
                log.warning(f"Rate limited for {symbol} {date_label}, waiting {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code == 422:
                # Unprocessable — symbol likely doesn't exist for this period
                log.debug(f"Alpaca 422 for {symbol} {date_label} — skipping")
                return None
            if resp.status_code != 200:
                log.error(f"Alpaca {resp.status_code} for {symbol} {date_label}: {resp.text[:200]}")
                return None
            return resp.json()
        except requests.exceptions.Timeout:
            wait = min(2 ** (attempt + 1), 30)
            log.warning(f"Timeout for {symbol} {date_label}, retry {attempt + 1}/{max_retries}")
            time.sleep(wait)
        except requests.exceptions.RequestException as exc:
            log.error(f"Request error for {symbol} {date_label}: {exc}")
            time.sleep(2)

    log.error(f"All retries exhausted for {symbol} {date_label}")
    return None


def insert_bars_bulk(conn, symbol: str, bars: list[dict]) -> int:
    """Insert bars into minute_bars using execute_values for speed."""
    if not bars:
        return 0

    rows = [
        (symbol, b["ts"], b["open"], b["high"], b["low"], b["close"], b["volume"])
        for b in bars
    ]

    sql = (
        "INSERT INTO minute_bars (symbol, ts, open, high, low, close, volume) "
        "VALUES %s ON CONFLICT (symbol, ts) DO NOTHING"
    )
    psycopg2.extras.execute_values(conn.cursor(), sql, rows, page_size=2000)
    conn.commit()
    return len(rows)


def format_eta(elapsed_seconds: float, done: int, total: int) -> str:
    """Format ETA string based on progress."""
    if done == 0:
        return "calculating..."
    rate = elapsed_seconds / done
    remaining = rate * (total - done)
    if remaining < 60:
        return f"{remaining:.0f}s"
    if remaining < 3600:
        return f"{remaining / 60:.1f}min"
    return f"{remaining / 3600:.1f}hr"


def build_batches(
    symbol_dates: dict[str, list[str]],
) -> list[tuple[str, str, str]]:
    """Group (symbol, dates) into multi-day batches of up to MAX_DAYS_PER_REQUEST.

    Args:
        symbol_dates: dict mapping symbol -> sorted list of date strings

    Returns:
        List of (symbol, batch_start_date, batch_end_date) tuples
    """
    batches = []
    for symbol, dates in sorted(symbol_dates.items()):
        sorted_dates = sorted(dates)
        for i in range(0, len(sorted_dates), MAX_DAYS_PER_REQUEST):
            chunk = sorted_dates[i : i + MAX_DAYS_PER_REQUEST]
            batches.append((symbol, chunk[0], chunk[-1]))
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill 1-min bars into PostgreSQL minute_bars table"
    )
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end", default=datetime.now().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--min-gap", type=float, default=0.03, help="Min gap %% for gap scan (default 0.03 = 3%%)"
    )
    parser.add_argument(
        "--min-vol", type=float, default=50000, help="Min 20-day avg volume (default 50K)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded, no fetching")
    parser.add_argument("--batch-size", type=int, default=180, help="Requests per batch before pause (legacy)")
    parser.add_argument(
        "--symbols", nargs="+", default=None, help="Specific symbols to backfill"
    )
    parser.add_argument(
        "--liquid", action="store_true", help="Backfill curated liquid large-cap symbols"
    )
    parser.add_argument(
        "--top-liquid",
        nargs="?",
        const=200,
        type=int,
        default=None,
        metavar="N",
        help="Query DB for top N symbols by avg daily volume (default: 200)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Delay between API requests in seconds (default: 0 for paid plan)",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Delete existing minute_bars for target symbols/date range before downloading",
    )
    args = parser.parse_args()

    api_key, secret_key = load_env_keys()
    if not api_key:
        log.error("ALPACA_API_KEY not set. Set env var or add to .env")
        sys.exit(1)

    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False

    # ── Determine symbols and candidate (symbol, date) pairs ──

    # Priority: --symbols > --top-liquid > --liquid > gap scan
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
        log.info(f"Explicit symbols mode: {len(symbols)} symbols")
    elif args.top_liquid is not None:
        symbols = get_top_liquid_symbols(conn, args.top_liquid)
        if not symbols:
            log.error("No symbols found from top-liquid query")
            conn.close()
            sys.exit(1)
    elif args.liquid:
        symbols = LIQUID_LARGE_CAPS[:]
        log.info(f"Curated liquid mode: {len(symbols)} symbols")
    else:
        symbols = None  # will use gap scan

    if symbols is not None:
        # Symbol-list mode: fetch every trading day for each symbol
        trading_dates = get_trading_dates(conn, args.start, args.end)
        if not trading_dates:
            log.error(f"No trading dates found between {args.start} and {args.end}")
            conn.close()
            sys.exit(1)
        log.info(
            f"Date range: {trading_dates[0]} to {trading_dates[-1]} ({len(trading_dates)} trading days)"
        )

        # Build symbol -> dates mapping
        symbol_dates: dict[str, list[str]] = {sym: list(trading_dates) for sym in symbols}
        total_pairs = len(symbols) * len(trading_dates)
        log.info(f"Total symbol-date pairs: {total_pairs}")

    else:
        # Gap candidate scan mode
        log.info(
            f"Gap scan mode: {args.min_gap * 100:.0f}%+ gaps, avg vol > {args.min_vol:.0f}, "
            f"{args.start} to {args.end}"
        )
        candidates = get_gap_candidates(conn, args.start, args.end, args.min_gap, args.min_vol)
        unique_symbols = set(s for s, _ in candidates)
        unique_dates = set(d for _, d in candidates)
        log.info(
            f"Found {len(candidates)} gap events across {len(unique_symbols)} symbols, "
            f"{len(unique_dates)} dates"
        )

        # Build symbol -> dates mapping
        symbol_dates = defaultdict(list)
        for sym, d in candidates:
            symbol_dates[sym].append(str(d))
        symbols = list(symbol_dates.keys())
        total_pairs = len(candidates)

    # ── Dry run ──
    if args.dry_run:
        batches = build_batches(symbol_dates)
        log.info(f"Symbols: {len(symbols)}")
        if len(symbols) <= 20:
            log.info(f"  {', '.join(sorted(symbols))}")
        else:
            log.info(f"  First 20: {', '.join(sorted(symbols)[:20])} ...")
        log.info(f"Total symbol-date pairs: {total_pairs}")
        log.info(f"API requests needed (multi-day batches): {len(batches)}")
        est_minutes = len(batches) * (args.rate_limit or 0.5) / 60
        log.info(f"Estimated time: ~{max(est_minutes, len(batches) * 0.3 / 60):.1f} min")
        conn.close()
        return

    # ── Purge if requested ──
    if args.purge:
        log.info(f"Purging existing minute_bars for {len(symbols)} symbols, {args.start} to {args.end}...")
        deleted = purge_minute_bars(conn, symbols, args.start, args.end)
        log.info(f"Purged {deleted} rows")

    # ── Check existing data to skip ──
    if not args.purge:
        log.info("Checking existing data to skip...")
        existing = get_existing_symbol_dates(conn, symbols, args.start, args.end)
        log.info(f"Found {len(existing)} existing symbol-date pairs in minute_bars")

        # Remove already-fetched dates from symbol_dates
        for sym in list(symbol_dates.keys()):
            symbol_dates[sym] = [d for d in symbol_dates[sym] if (sym, d) not in existing]
            if not symbol_dates[sym]:
                del symbol_dates[sym]

        remaining_pairs = sum(len(dates) for dates in symbol_dates.values())
        log.info(f"Need to fetch {remaining_pairs} symbol-date pairs ({total_pairs - remaining_pairs} skipped)")
    else:
        remaining_pairs = total_pairs

    if not symbol_dates:
        log.info("Nothing to fetch -- all done!")
        conn.close()
        return

    # ── Build multi-day batches ──
    batches = build_batches(symbol_dates)
    log.info(
        f"Created {len(batches)} API requests (multi-day batches of up to {MAX_DAYS_PER_REQUEST} days)"
    )

    # ── Fetch and insert ──
    total_bars_inserted = 0
    successful_requests = 0
    failed_requests = 0
    start_time = time.monotonic()

    for i, (symbol, batch_start, batch_end) in enumerate(batches):
        bars = fetch_bars_multi_day(symbol, batch_start, batch_end, api_key, secret_key)

        if bars:
            n = insert_bars_bulk(conn, symbol, bars)
            total_bars_inserted += n
            successful_requests += 1
        elif bars is not None:
            # Empty list = no bars for this period (not an error)
            successful_requests += 1
        else:
            failed_requests += 1

        # Progress reporting
        done = i + 1
        if done % 10 == 0 or done == len(batches):
            elapsed = time.monotonic() - start_time
            pct = done / len(batches) * 100
            eta = format_eta(elapsed, done, len(batches))
            log.info(
                f"[{pct:5.1f}%] {done}/{len(batches)} requests | "
                f"{total_bars_inserted:,} bars | "
                f"{failed_requests} errors | "
                f"ETA: {eta}"
            )

        # Optional rate limiting
        if args.rate_limit > 0:
            time.sleep(args.rate_limit)

    elapsed_total = time.monotonic() - start_time
    log.info(
        f"Done! {successful_requests}/{len(batches)} successful requests, "
        f"{total_bars_inserted:,} bars inserted, {failed_requests} errors, "
        f"{elapsed_total:.0f}s elapsed"
    )
    conn.close()


if __name__ == "__main__":
    main()
