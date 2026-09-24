"""CLI for Alpaca 1-minute bar ingestion.

v-ingest-traded-2026-09-24: adds --include-traded/--no-include-traded and
--traded-date flags to union the top-N volume universe with symbols the bot
traded or signaled that ET day. Default behavior is to include traded symbols
(purely additive, idempotent INSERT ... ON CONFLICT DO NOTHING).

Examples:
    # Top 150 most-traded symbols, last 180 days (recommended retrain dataset)
    python ingest_alpaca_bars.py --top-n 150 --days 180

    # Specific basket, last 30 days
    python ingest_alpaca_bars.py --symbols NVDA TSLA AAPL --days 30

    # Resumable: rerun with same args after interruption — uses checkpoint
    python ingest_alpaca_bars.py --top-n 150 --days 180

    # Dry-run: resolve symbols + show window, no network / DB writes
    python ingest_alpaca_bars.py --top-n 10 --days 7 --dry-run

    # Disable traded-symbol inclusion (legacy behavior)
    python ingest_alpaca_bars.py --top-n 150 --days 7 --no-include-traded

    # Explicit traded date (useful for backfills)
    python ingest_alpaca_bars.py --top-n 150 --days 7 --traded-date 2026-09-23
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from data_providers.alpaca_ingest import (
    IngestConfig,
    IngestJob,
    top_symbols_by_volume,
)
from data_providers.ingest_universe import (
    resolve_ingest_universe,
    symbol_list_hash,
    traded_symbols_for_day,
)

_ET = ZoneInfo("America/New_York")
_STALL_RETRY_LOOKBACK_DAYS = 30


def _get_include_traded_default() -> bool:
    """Get default for --include-traded from config flag."""
    try:
        from core.config import Config
        return Config().MINUTE_BARS_INGEST_INCLUDE_TRADED
    except Exception:
        return True


def _get_traded_actions() -> list[str]:
    """Get action allowlist from config flag."""
    try:
        from core.config import Config
        return Config().MINUTE_BARS_INGEST_TRADED_ACTIONS
    except Exception:
        return [
            'received', 'accepted', 'bracket_placed', 'position_created',
            'suppressed', 'skip', 'signal_buy', 'signal_sell',
        ]


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
    p.add_argument("--window-days", type=int, default=30,
                   help="Days per checkpointed unit for visible progress "
                        "(default 30 = monthly)")
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent HTTP requests (default 4)")
    p.add_argument("--feed", choices=["sip", "iex"], default=None,
                   help="Override ALPACA_DATA_FEED")

    p.add_argument("--dry-run", action="store_true",
                   help="Resolve symbols + window, show plan, exit 0")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    include_grp = p.add_mutually_exclusive_group()
    include_default = _get_include_traded_default()
    include_grp.add_argument(
        "--include-traded", dest="include_traded", action="store_true",
        default=include_default,
        help="Include bot-traded/signaled symbols in universe "
             f"(default: {'yes' if include_default else 'no'})")
    include_grp.add_argument(
        "--no-include-traded", dest="include_traded", action="store_false",
        help="Disable traded-symbol inclusion (legacy top-N only behavior)")

    p.add_argument("--traded-date", type=str, default=None,
                   help="Date for traded symbols (YYYY-MM-DD, default: ET date of --end/now)")

    return p.parse_args(argv)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _resolve_traded_date(args: argparse.Namespace, end: datetime) -> date:
    """Determine the ET date for traded-symbol resolution."""
    if args.traded_date:
        return date.fromisoformat(args.traded_date)
    return end.astimezone(_ET).date()


def _resolve_symbols_with_stall_guard(
    cfg: IngestConfig,
    args: argparse.Namespace,
    end: datetime,
    logger: logging.Logger,
) -> tuple[list[str], dict]:
    """Resolve symbols with stall guard and traded-symbol inclusion.

    Returns:
        (symbols, info_dict) where info_dict contains:
          - top_n_count: number of top-N volume symbols
          - traded_count: number of traded symbols (if include_traded)
          - added_by_traded: list of symbols added by traded inclusion
          - final_count: final universe size
          - used_wide_lookback: whether stall guard widened lookback
    """
    info = {
        "top_n_count": 0,
        "traded_count": 0,
        "added_by_traded": [],
        "final_count": 0,
        "used_wide_lookback": False,
    }

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
        if args.include_traded:
            traded_date = _resolve_traded_date(args, end)
            traded = traded_symbols_for_day(
                cfg.dsn,
                traded_date,
                decision_actions=_get_traded_actions(),
            )
            info["traded_count"] = len(traded)
            top_n_set = set(symbols)
            info["added_by_traded"] = sorted(traded - top_n_set)
            symbols = resolve_ingest_universe(symbols, traded)
        info["final_count"] = len(symbols)
        return symbols, info

    top_n_symbols = top_symbols_by_volume(cfg.dsn, args.top_n, days=args.days)
    info["top_n_count"] = len(top_n_symbols)

    traded: set[str] = set()
    if args.include_traded:
        traded_date = _resolve_traded_date(args, end)
        traded = traded_symbols_for_day(
            cfg.dsn,
            traded_date,
            decision_actions=_get_traded_actions(),
        )
        info["traded_count"] = len(traded)

    if not top_n_symbols:
        logger.warning(
            "stall_guard top_n_empty top_n=%d traded=%d days=%d",
            info["top_n_count"], info["traded_count"], args.days,
        )

        if traded:
            logger.info(
                "stall_guard_fallback using_traded_only count=%d",
                len(traded),
            )
            top_n_set: set[str] = set()
        else:
            logger.warning(
                "stall_guard_retry widening_lookback from=%d to=%d",
                args.days, _STALL_RETRY_LOOKBACK_DAYS,
            )
            top_n_symbols = top_symbols_by_volume(
                cfg.dsn, args.top_n, days=_STALL_RETRY_LOOKBACK_DAYS
            )
            info["top_n_count"] = len(top_n_symbols)
            info["used_wide_lookback"] = True

            if not top_n_symbols and not traded:
                logger.error(
                    "stall_guard_exhausted symbols=0 after wide lookback and "
                    "no traded symbols — refusing to run with empty universe"
                )
                raise SystemExit(1)

    top_n_set = set(s.upper() for s in top_n_symbols)
    info["added_by_traded"] = sorted(traded - top_n_set)

    if args.include_traded:
        symbols = resolve_ingest_universe(top_n_symbols, traded)
    else:
        symbols = [s.upper() for s in top_n_symbols]

    info["final_count"] = len(symbols)

    if info["final_count"] == 0:
        logger.error("resolved_symbols=0 — refusing to run with empty universe")
        raise SystemExit(1)

    return symbols, info


async def _main_async(args: argparse.Namespace) -> int:
    load_dotenv()
    overrides: dict = {
        "symbols_per_request": args.batch_size,
        "concurrent_requests": args.workers,
    }
    if args.feed:
        overrides["feed"] = args.feed
    cfg = IngestConfig.from_env(**overrides)

    end = (
        datetime.fromisoformat(args.end.replace("Z", "+00:00"))
        if args.end else datetime.now(timezone.utc)
    )
    start = end - timedelta(days=args.days)

    logger = logging.getLogger("alpaca_ingest_cli")

    symbols, info = _resolve_symbols_with_stall_guard(cfg, args, end, logger)

    sym_hash = symbol_list_hash(symbols)
    logger.info(
        "plan top_n_count=%d traded_count=%d added_by_traded=%d final_count=%d "
        "sym_hash=%s start=%s end=%s feed=%s workers=%d",
        info["top_n_count"], info["traded_count"], len(info["added_by_traded"]),
        info["final_count"], sym_hash, start.isoformat(), end.isoformat(),
        cfg.feed, cfg.concurrent_requests,
    )
    if info["added_by_traded"]:
        logger.debug("added_by_traded=%s", info["added_by_traded"])
    if info["used_wide_lookback"]:
        logger.info("note: used wide lookback (%d days) due to stall guard",
                    _STALL_RETRY_LOOKBACK_DAYS)

    if args.dry_run:
        logger.info("dry_run first_5=%s last_5=%s",
                    symbols[:5], symbols[-5:] if len(symbols) > 5 else [])
        return 0

    stats = await IngestJob(cfg).run(
        symbols, start, end,
        window_days=args.window_days,
        sym_hash=sym_hash,
    )
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
