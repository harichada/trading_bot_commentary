"""v-pricebook-2026-05-01: canonical price source for the entire engine.

Architectural role: REPLACES QuoteCache + the various stored-on-Position
mark fields (current_price, unrealized_pnl). The PriceBook is the ONE
place anyone reads a price from. Producers (Schwab stream, REST poll
fallback) write here. Consumers (FSM exit loop, dashboard WS handler,
risk manager, screener UI) read here.

Why this exists (the lesson the bot was teaching us):
  - Distributed mark state was the root cause of "PLTR ticks then
    freezes": the WS handler was reading pos.current_price as a
    fallback when the cache went stale, and pos.current_price was
    never being refreshed continuously — only on exit decisions.
  - Each surface (sim positions, real positions, ticker tape, watchlist)
    had its own private mark cache with its own staleness rules. Five
    caches → five different failure modes. After this refactor: one
    cache, one staleness rule, one place to fix bugs.

Subscription reconciliation is a first-class concern of this class.
The set of symbols we want streaming = union(positions, watchlist,
chart-viewer's selected symbol). PriceBook owns reconciliation; the
Schwab stream task is just the I/O layer.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Set, List, Awaitable

logger = logging.getLogger("TradingBot")


# ──────────────────────────────────────────────────────────────────────
# PriceObject — the "what does this symbol cost right now?" answer.
#
# Returned by PriceBook.get_mark(symbol). Always returned (never None) —
# absence is encoded as price=None + is_stale=True so callers don't have
# to special-case the missing-symbol path.
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PriceObject:
    symbol: str
    price: Optional[float]        # None if no quote ever received
    bid: float = 0.0
    ask: float = 0.0
    last_trade: Optional[float] = None  # last traded price (sparse, may be old)
    timestamp: Optional[datetime] = None  # when the price was set
    source: str = "none"          # "stream" | "rest" | "stale" | "none"
    is_stale: bool = True         # True when older than STALE_THRESHOLD_SEC

    @property
    def has_price(self) -> bool:
        return self.price is not None and self.price > 0

    def age_seconds(self) -> Optional[float]:
        if self.timestamp is None:
            return None
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()


# ──────────────────────────────────────────────────────────────────────
# Internal storage record — mutable, lives inside PriceBook.
# Public callers receive frozen PriceObject snapshots, never the raw row.
# ──────────────────────────────────────────────────────────────────────
@dataclass
class _PriceRow:
    symbol: str
    last: Optional[float] = None       # raw last trade
    bid: float = 0.0
    ask: float = 0.0
    updated_at: Optional[datetime] = None
    source: str = "none"


class PriceBook:
    """Singleton-managed canonical mark per symbol.

    Producers call:
      apply_tick(symbol, last=?, bid=?, ask=?, source="stream") — partial
        update; missing fields are preserved from the prior row.

    Consumers call:
      get_mark(symbol) -> PriceObject
      get_marks([symbols...]) -> dict[symbol, PriceObject]

    Subscription concerns:
      register_interest(scope, symbols) — declares which symbols the
        scope (e.g. "positions", "watchlist", "chart") wants priced.
        The book recomputes the union and notifies the producer (Schwab
        stream) of subscribe/unsubscribe diffs.

    Staleness: configurable threshold; defaults to 30s (matches Schwab's
    "quiet stock" silence window observed in production).
    """

    DEFAULT_STALE_THRESHOLD_SEC: float = 30.0

    # Class-level singleton handle. The engine sets this in __init__;
    # downstream modules that don't have an engine reference (notably
    # core.calculations) read PriceBook.instance() to get the singleton.
    _singleton: Optional["PriceBook"] = None

    def __init__(self, stale_threshold_sec: float = DEFAULT_STALE_THRESHOLD_SEC) -> None:
        self._rows: dict[str, _PriceRow] = {}
        self._lock = asyncio.Lock()
        # Per-scope interest sets. Scopes: "positions" / "watchlist" /
        # "chart" / anything the engine wants to declare.
        self._interest: dict[str, Set[str]] = {}
        self._interest_lock = asyncio.Lock()
        # Subscriber called whenever the union-of-interest changes; the
        # engine wires this to schwab_stream.update_subscriptions().
        self._on_subscription_change: Optional[Callable[[Set[str]], Awaitable[None]]] = None
        # Tunables
        self._stale_threshold_sec = stale_threshold_sec
        # Diagnostic
        self._tick_count = 0
        self._last_tick_at: Optional[datetime] = None

    # ── singleton wiring ──────────────────────────────────────────────

    @classmethod
    def install(cls, book: "PriceBook") -> None:
        """Engine calls this once after constructing the book."""
        cls._singleton = book

    @classmethod
    def instance(cls) -> Optional["PriceBook"]:
        """For modules that can't easily plumb an engine reference."""
        return cls._singleton

    # ── producer API ──────────────────────────────────────────────────

    def apply_tick_sync(
        self,
        symbol: str,
        *,
        last: Optional[float] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        source: str = "stream",
    ) -> None:
        """Sync write path. Used by the Schwab stream tick handler
        (which is invoked from the asyncio loop and doesn't need a lock
        for single-write atomicity — dict assignment is GIL-protected).
        Async writers may use apply_tick().

        Partial-merge: any field passed as None is preserved from the
        prior row. This is critical because Schwab sends partial
        updates (most ticks are bid/ask only, no last_trade).
        """
        if not symbol:
            return
        row = self._rows.get(symbol)
        if row is None:
            row = _PriceRow(symbol=symbol)
            self._rows[symbol] = row
        if last is not None:
            try:
                lf = float(last)
                if lf > 0:
                    row.last = lf
            except (TypeError, ValueError):
                pass
        if bid is not None:
            try:
                row.bid = float(bid)
            except (TypeError, ValueError):
                pass
        if ask is not None:
            try:
                row.ask = float(ask)
            except (TypeError, ValueError):
                pass
        row.source = source
        row.updated_at = datetime.now(timezone.utc)
        self._tick_count += 1
        self._last_tick_at = row.updated_at

    async def apply_tick(self, symbol: str, **kwargs) -> None:
        """Async wrapper. Same semantics as apply_tick_sync."""
        async with self._lock:
            self.apply_tick_sync(symbol, **kwargs)

    # ── consumer API ──────────────────────────────────────────────────

    def _row_to_object(self, row: Optional[_PriceRow], symbol: str) -> PriceObject:
        """Internal: convert a row to a frozen PriceObject. Computes
        the canonical 'price' field as bid/ask mid when available
        (continuously updating), else last trade (sparse). Computes
        is_stale per threshold."""
        if row is None:
            return PriceObject(
                symbol=symbol, price=None, last_trade=None,
                timestamp=None, source="none", is_stale=True,
            )
        # Canonical price preference: mid > last
        if row.bid > 0 and row.ask > 0:
            price = (row.bid + row.ask) / 2.0
        elif row.last is not None and row.last > 0:
            price = row.last
        else:
            price = None
        # Staleness
        if row.updated_at is None:
            is_stale = True
            source = "none"
        else:
            age = (datetime.now(timezone.utc) - row.updated_at).total_seconds()
            is_stale = age > self._stale_threshold_sec
            source = row.source if not is_stale else "stale"
        return PriceObject(
            symbol=symbol,
            price=price,
            bid=row.bid,
            ask=row.ask,
            last_trade=row.last,
            timestamp=row.updated_at,
            source=source,
            is_stale=is_stale,
        )

    def get_mark(self, symbol: str) -> PriceObject:
        """Read the canonical mark for a symbol. Always returns a
        PriceObject — never None. is_stale=True signals the caller
        that the data is unsafe to act on (e.g. exit decisions skip
        evaluation, dashboard greys out the value)."""
        if not symbol:
            return PriceObject(symbol="", price=None, source="none", is_stale=True)
        return self._row_to_object(self._rows.get(symbol), symbol)

    def get_marks(self, symbols: List[str]) -> dict[str, PriceObject]:
        """Bulk read. Single-pass, no per-symbol lock — _rows reads
        are GIL-atomic at the dict level for our usage pattern."""
        return {s: self._row_to_object(self._rows.get(s), s) for s in symbols}

    def all_symbols(self) -> List[str]:
        return list(self._rows.keys())

    # ── interest / subscription reconciliation ───────────────────────

    def set_subscription_callback(
        self, fn: Optional[Callable[[Set[str]], Awaitable[None]]],
    ) -> None:
        """Engine wires this to schwab_stream.update_subscriptions()."""
        self._on_subscription_change = fn

    async def register_interest(self, scope: str, symbols: Set[str]) -> None:
        """A subsystem declares 'these are the symbols I care about
        right now'. The book recomputes the global union; if it
        changed, fires the subscription callback so the stream
        producer can subscribe/unsubscribe diff-only.

        Scope examples: 'positions', 'watchlist', 'chart'.
        Calling with an empty set removes interest from that scope.
        """
        symbols = set(s.upper() for s in symbols if s)
        async with self._interest_lock:
            prior_union = set().union(*self._interest.values()) if self._interest else set()
            if symbols:
                self._interest[scope] = symbols
            else:
                self._interest.pop(scope, None)
            new_union = set().union(*self._interest.values()) if self._interest else set()
            if new_union == prior_union:
                return
            cb = self._on_subscription_change
        # Fire outside the interest lock so the producer's I/O can't
        # deadlock against another register_interest call.
        if cb is not None:
            try:
                await cb(new_union)
            except Exception as exc:
                logger.warning("price_book: subscription callback failed: %s", exc)

    async def desired_subscription_set(self) -> Set[str]:
        """The current union, as the producer would see it."""
        async with self._interest_lock:
            return set().union(*self._interest.values()) if self._interest else set()

    # ── diagnostics ──────────────────────────────────────────────────

    def status(self) -> dict:
        """Snapshot for /api endpoints, monitoring."""
        rows = []
        for sym, row in self._rows.items():
            obj = self._row_to_object(row, sym)
            rows.append({
                "symbol": sym,
                "price": obj.price,
                "bid": obj.bid,
                "ask": obj.ask,
                "last_trade": obj.last_trade,
                "age_sec": (round(obj.age_seconds(), 2)
                            if obj.age_seconds() is not None else None),
                "source": obj.source,
                "is_stale": obj.is_stale,
            })
        return {
            "stale_threshold_sec": self._stale_threshold_sec,
            "tick_count": self._tick_count,
            "last_tick_at": (self._last_tick_at.isoformat()
                              if self._last_tick_at else None),
            "interest_scopes": {k: sorted(v) for k, v in self._interest.items()},
            "rows": rows,
        }
