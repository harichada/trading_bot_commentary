"""v-quote-cache-2026-04-30 (Phase 2): in-memory per-symbol quote cache.

Architectural role: the `position_loop` (1Hz exit dispatcher) reads from
this cache instead of calling the broker directly. The `quote_streamer`
(separate task) refreshes the cache every QUOTE_REFRESH_SEC. Reading
from a cache is what keeps the position loop from blocking on the
network — that's the property Phase 2 is buying us.

Invariants:
  - get/put are atomic under a single asyncio.Lock
  - never holds the lock across an await on a network call
    (the streamer awaits the broker OUTSIDE the lock, then puts)
  - is_stale() is read-only; cheap; safe to call on every position tick
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List

logger = logging.getLogger("TradingBot")


@dataclass
class CachedQuote:
    """One symbol's most-recently-observed market quote."""
    symbol: str
    last: float
    bid: float = 0.0
    ask: float = 0.0
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "schwab"   # "schwab" | "fallback" | "stale"

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.fetched_at).total_seconds()

    def is_stale(self, max_age_sec: float) -> bool:
        return self.age_seconds() > max_age_sec


class QuoteCache:
    """Process-local quote store. Single instance lives on the engine.

    Designed to be the only place position_loop reads price from. The
    contract is: anyone calling .get() gets either a fresh enough
    CachedQuote, a stale one (with source='stale' or aged), or None
    if the streamer hasn't seeded that symbol yet. The caller decides
    whether to act on staleness — see Config.QUOTE_MAX_STALE_SEC.
    """

    def __init__(self) -> None:
        self._d: Dict[str, CachedQuote] = {}
        self._lock = asyncio.Lock()

    async def put(self, q: CachedQuote) -> None:
        """Atomic write. Caller has already done the network fetch."""
        async with self._lock:
            self._d[q.symbol] = q

    async def get(self, symbol: str) -> Optional[CachedQuote]:
        """Atomic read. Returns None if symbol never seeded."""
        async with self._lock:
            return self._d.get(symbol)

    async def all_symbols(self) -> List[str]:
        async with self._lock:
            return list(self._d.keys())

    async def has(self, symbol: str) -> bool:
        async with self._lock:
            return symbol in self._d

    async def mark_stale(self, symbol: str, reason: str = "fetch_failed") -> None:
        """Streamer calls this on fetch failure. Keeps the last-good
        price/bid/ask but bumps source so consumers can detect."""
        async with self._lock:
            q = self._d.get(symbol)
            if q is not None:
                q.source = f"stale:{reason}"

    async def evict(self, symbol: str) -> None:
        """Called when a position closes — keeps the cache lean."""
        async with self._lock:
            self._d.pop(symbol, None)

    async def snapshot(self) -> Dict[str, CachedQuote]:
        """Read-only snapshot for diagnostics / dashboard."""
        async with self._lock:
            return dict(self._d)
