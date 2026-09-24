"""Universe resolution for minute_bars ingest: traded/signaled symbols.

v-ingest-traded-2026-09-24: combines the top-N volume symbols with symbols
the bot traded or signaled that ET day. Each source query is fault-tolerant
(one missing table never breaks the ingest).

Usage:
    from data_providers.ingest_universe import (
        traded_symbols_for_day,
        resolve_ingest_universe,
    )
    traded = traded_symbols_for_day(dsn, date(2026, 9, 24))
    final_universe = resolve_ingest_universe(top_n_symbols, traded)

DB sources for "traded or signaled on day D":
  - bot_trades: entry_time::date = D OR exit_time::date = D
  - bot_positions: currently open positions (any)
  - bot_decisions: ts::date = D with action in configurable allowlist
  - bot_decision_snapshots: ts (timestamptz)::date in America/New_York = D
  - signals_intraday: best-effort (stale since 2026-03)
"""
from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime
from typing import Optional

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # type: ignore

try:
    import sqlalchemy
except ImportError:
    sqlalchemy = None  # type: ignore


__all__ = [
    "traded_symbols_for_day",
    "resolve_ingest_universe",
    "symbol_list_hash",
]

_logger = logging.getLogger("ingest_universe")

# Default action allowlist for bot_decisions.
# Excludes shadow_pass (~290 symbols/day with no trading intent).
DEFAULT_DECISION_ACTIONS = frozenset({
    'received', 'accepted', 'bracket_placed', 'position_created',
    'suppressed', 'skip', 'signal_buy', 'signal_sell',
})


def traded_symbols_for_day(
    dsn: str,
    day: date,
    *,
    decision_actions: Optional[list[str]] = None,
    sources: Optional[list[str]] = None,
) -> set[str]:
    """Return symbols the bot traded or signaled on the given ET day.

    Args:
        dsn: Postgres connection string (sync driver).
        day: The ET date to query.
        decision_actions: Allowlist of bot_decisions.action values to include.
            Defaults to DEFAULT_DECISION_ACTIONS (excludes shadow_pass).
        sources: Which sources to query. Defaults to all:
            ['bot_trades', 'bot_positions', 'bot_decisions',
             'bot_decision_snapshots', 'signals_intraday'].
            Each source is queried independently; failures are logged and skipped.

    Returns:
        Set of uppercase symbol strings.
    """
    if decision_actions is None:
        decision_actions = list(DEFAULT_DECISION_ACTIONS)
    action_set = frozenset(a.lower() for a in decision_actions)

    if sources is None:
        sources = [
            'bot_trades', 'bot_positions', 'bot_decisions',
            'bot_decision_snapshots', 'signals_intraday',
        ]
    source_set = frozenset(sources)

    if sqlalchemy is None:
        _logger.warning("sqlalchemy not available, cannot query traded symbols")
        return set()

    symbols: set[str] = set()

    try:
        engine = sqlalchemy.create_engine(dsn)
    except Exception as exc:
        _logger.warning("traded_symbols_for_day dsn_error=%s", exc)
        return symbols

    with engine.connect() as conn:
        # bot_trades: entry_time::date = day OR exit_time::date = day
        if 'bot_trades' in source_set:
            symbols |= _query_bot_trades(conn, day)

        # bot_positions: currently open positions
        if 'bot_positions' in source_set:
            symbols |= _query_bot_positions(conn)

        # bot_decisions: ts::date = day with action in allowlist
        if 'bot_decisions' in source_set:
            symbols |= _query_bot_decisions(conn, day, action_set)

        # bot_decision_snapshots: ts AT TIME ZONE 'America/New_York' date = day
        if 'bot_decision_snapshots' in source_set:
            symbols |= _query_bot_decision_snapshots(conn, day)

        # signals_intraday: best-effort (stale since 2026-03)
        if 'signals_intraday' in source_set:
            symbols |= _query_signals_intraday(conn, day)

    return symbols


def _query_bot_trades(conn, day: date) -> set[str]:
    """Query bot_trades for symbols traded on day."""
    try:
        rows = conn.execute(
            sqlalchemy.text(
                "SELECT DISTINCT symbol FROM bot_trades "
                "WHERE (entry_time::date = :day OR exit_time::date = :day)"
            ),
            {"day": day},
        ).fetchall()
        result = {r[0].upper() for r in rows if r[0]}
        _logger.debug("bot_trades day=%s symbols=%d", day, len(result))
        return result
    except Exception as exc:
        _logger.warning("bot_trades query failed: %s", exc)
        return set()


def _query_bot_positions(conn) -> set[str]:
    """Query bot_positions for currently open positions."""
    try:
        rows = conn.execute(
            sqlalchemy.text("SELECT DISTINCT symbol FROM bot_positions")
        ).fetchall()
        result = {r[0].upper() for r in rows if r[0]}
        _logger.debug("bot_positions symbols=%d", len(result))
        return result
    except Exception as exc:
        _logger.warning("bot_positions query failed: %s", exc)
        return set()


def _query_bot_decisions(conn, day: date, action_set: frozenset[str]) -> set[str]:
    """Query bot_decisions for symbols with matching actions on day."""
    if not action_set:
        return set()
    try:
        rows = conn.execute(
            sqlalchemy.text(
                "SELECT DISTINCT symbol FROM bot_decisions "
                "WHERE ts::date = :day AND LOWER(action) = ANY(:actions) "
                "AND symbol IS NOT NULL"
            ),
            {"day": day, "actions": list(action_set)},
        ).fetchall()
        result = {r[0].upper() for r in rows if r[0]}
        _logger.debug("bot_decisions day=%s actions=%d symbols=%d",
                      day, len(action_set), len(result))
        return result
    except Exception as exc:
        _logger.warning("bot_decisions query failed: %s", exc)
        return set()


def _query_bot_decision_snapshots(conn, day: date) -> set[str]:
    """Query bot_decision_snapshots for symbols on day (timestamptz → ET)."""
    try:
        rows = conn.execute(
            sqlalchemy.text(
                "SELECT DISTINCT symbol FROM bot_decision_snapshots "
                "WHERE (ts AT TIME ZONE 'America/New_York')::date = :day "
                "AND action IN ('signal_buy', 'signal_sell', 'veto')"
            ),
            {"day": day},
        ).fetchall()
        result = {r[0].upper() for r in rows if r[0]}
        _logger.debug("bot_decision_snapshots day=%s symbols=%d", day, len(result))
        return result
    except Exception as exc:
        _logger.warning("bot_decision_snapshots query failed: %s", exc)
        return set()


def _query_signals_intraday(conn, day: date) -> set[str]:
    """Query signals_intraday table (best-effort, may be stale)."""
    try:
        rows = conn.execute(
            sqlalchemy.text(
                "SELECT DISTINCT symbol FROM signals_intraday "
                "WHERE ts::date = :day"
            ),
            {"day": day},
        ).fetchall()
        result = {r[0].upper() for r in rows if r[0]}
        _logger.debug("signals_intraday day=%s symbols=%d", day, len(result))
        return result
    except Exception as exc:
        _logger.debug("signals_intraday query failed (best-effort): %s", exc)
        return set()


def resolve_ingest_universe(
    top_n_symbols: list[str],
    traded_symbols: set[str],
) -> list[str]:
    """Combine top-N volume symbols with traded symbols, preserving order.

    Args:
        top_n_symbols: Ordered list of top-N volume symbols (from top_symbols_by_volume).
        traded_symbols: Set of symbols the bot traded/signaled.

    Returns:
        Ordered list: top-N symbols first (in original order), then any extra
        traded symbols (sorted alphabetically), all uppercase and deduplicated.
    """
    seen: set[str] = set()
    result: list[str] = []

    # Top-N symbols first, in their original volume-sorted order
    for sym in top_n_symbols:
        s = sym.upper()
        if s not in seen:
            seen.add(s)
            result.append(s)

    # Extra traded symbols, sorted alphabetically
    extras = sorted(traded_symbols - seen)
    for s in extras:
        result.append(s.upper())

    return result


def symbol_list_hash(symbols: list[str], length: int = 8) -> str:
    """Compute a short deterministic hash of the symbol list.

    Used in checkpoint keys so that re-runs with different symbol universes
    don't incorrectly skip work.

    Args:
        symbols: Ordered list of symbols.
        length: Number of hex characters to return (default 8).

    Returns:
        Hex string of the first `length` characters of MD5(joined symbols).
    """
    joined = ",".join(s.upper() for s in symbols)
    return hashlib.md5(joined.encode()).hexdigest()[:length]
