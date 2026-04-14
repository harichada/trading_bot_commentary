"""CLI for Alpaca 1-minute bar ingestion.

Examples:
    # Top 150 most-traded symbols, last 180 days (recommended retrain dataset)
    python ingest_alpaca_bars.py --top-n 150 --days 180

    # Specific basket, last 30 days
    python ingest_alpaca_bars.py --symbols NVDA TSLA AAPL --days 30

    # Resumable: rerun with same args after interruption — uses checkpoint
    python ingest_alpaca_bars.py --top-n 150 --days 180

    # Dry-run: resolve symbols + show window, no network / DB writes
    python ingest_alpaca_bars.py --top-n 10 --days 7 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from data_providers.alpaca_ingest import (
    IngestConfig,
    IngestJob,
    top_symbols_by_volume,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sym = p.add_mutually_exclusive_group(required=True)
    sym.add_argument("--symbols", nargs="+", metavar="SYM",
                     help="Explicit symbol list")
    sym.add_argument("--top-n", type=int, metavar="N",
                     help="Top-N most-traded symbols from current minute_bars")

    p.add_argument("--days", type=int, default=180,
                   help="Lookback window (default 180)")
    p.add_argument("--end", type=str, default=None,
                   help="Window end ISO8601 UTC (default: now)")

    p.add_argument("--batch-size", type=int, default=50,
                   help="Symbols per API request (max 200, default 50)")
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent HTTP requests (default 4)")
    p.add_argument("--feed", choices=["sip", "iex"], default=None,
                   help="Override ALPACA_DATA_FEED")

    p.add_argument("--dry-run", action="store_true",
                   help="Resolve symbols + window, show plan, exit 0")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def _main_async(args: argparse.Namespace) -> int:
    load_dotenv()
    overrides: dict = {
        "symbols_per_request": args.batch_size,
        "concurrent_requests": args.workers,
    }
    if args.feed:
        overrides["feed"] = args.feed
    cfg = IngestConfig.from_env(**overrides)

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = top_symbols_by_volume(cfg.dsn, args.top_n, days=args.days)

    end = (
        datetime.fromisoformat(args.end.replace("Z", "+00:00"))
        if args.end else datetime.now(timezone.utc)
    )
    start = end - timedelta(days=args.days)

    logger = logging.getLogger("alpaca_ingest_cli")
    logger.info(
        "plan symbols=%d batches=%d start=%s end=%s feed=%s workers=%d",
        len(symbols),
        (len(symbols) + cfg.symbols_per_request - 1) // cfg.symbols_per_request,
        start.isoformat(), end.isoformat(), cfg.feed, cfg.concurrent_requests,
    )

    if args.dry_run:
        logger.info("dry_run first_5=%s last_5=%s",
                    symbols[:5], symbols[-5:] if len(symbols) > 5 else [])
        return 0

    stats = await IngestJob(cfg).run(symbols, start, end)
    logger.info("ingest_summary %s", " ".join(f"{k}={v}" for k, v in stats.items()))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.log_level)
    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        logging.getLogger("alpaca_ingest_cli").warning("interrupted — "
            "checkpoint preserved, rerun to resume")
        return 130


if __name__ == "__main__":
    sys.exit(main())
