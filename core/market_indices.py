"""v-market-indices-strip-2026-05-27: live market regime indicators.

Background-refreshed cache of major US equity index quotes (S&P 500,
Dow, Nasdaq, Russell 2000, VIX). Used by the dashboard to render an
at-a-glance regime strip above the ticker tape.

Threading model
---------------
The bot runs as a single-threaded asyncio process. Every consumer of
this cache (the engine background loop, the FastAPI route) lives on
the same event loop, so no locking is needed. Do NOT call ``refresh``
from a worker thread.

Wiring
------
The engine spawns ``_market_indices_loop`` which calls
``MarketIndicesCache.instance().refresh(schwab_client)`` every 10s.
The REST endpoint ``GET /api/market-indices`` returns
``snapshot()``. Frontend polls every 10s.

Why a singleton + a refresh loop (not on-demand per request):
Polling-per-request would multiply Schwab calls by the number of
open dashboard tabs and rate-limit us. One refresh per 10s feeds
all viewers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger("TradingBot")

# Module-level singleton.
_INSTANCE: Optional["MarketIndicesCache"] = None

# Symbol → display name. The leading "$" on $VIX signals to Schwab
# that this is an index, not an equity; the response payload shape
# differs (no assetSubType, no extended-hours fields).
_DISPLAY_NAMES: Dict[str, str] = {
    "SPY": "S&P 500",
    "DIA": "Dow Jones",
    "QQQ": "Nasdaq",
    "IWM": "Russell 2000",
    "$VIX": "VIX",
}

# Considered stale after this many seconds without a successful refresh.
# 30s = 3x the refresh cadence; below that we'd flag transient hiccups
# as stale.
_STALE_AFTER_SEC: float = 30.0


@dataclass(frozen=True)
class IndexQuote:
    """One index's latest price and intraday % change."""

    symbol: str
    display_name: str
    last: float
    change_pct: float
    ts: datetime  # UTC timestamp of the refresh that produced this quote


class MarketIndicesCache:
    """Singleton holding the latest snapshot for the dashboard."""

    def __init__(self) -> None:
        self._snapshot: Dict[str, IndexQuote] = {}
        self.last_updated: Optional[datetime] = None
        self.last_error: Optional[str] = None

    @classmethod
    def instance(cls) -> "MarketIndicesCache":
        """Process-wide singleton accessor."""
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = cls()
        return _INSTANCE

    @classmethod
    def reset_for_tests(cls) -> None:
        """Drop the singleton so tests can start fresh."""
        global _INSTANCE
        _INSTANCE = None

    def refresh(self, schwab_client) -> None:
        """Pull the latest quotes from Schwab in one batched call.

        Best-effort: per-symbol parse failures keep the prior value.
        Only a transport-level failure (non-200, exception) clears
        ``last_updated`` (via the absence of an update). A single bad
        response should not blank the entire strip.
        """
        symbols = list(_DISPLAY_NAMES.keys())
        try:
            response = schwab_client.get_quotes(symbols)
            if getattr(response, "status_code", None) != 200:
                self.last_error = f"http_{getattr(response, 'status_code', 'unknown')}"
                return
            data = response.json()
        except Exception as exc:
            self.last_error = f"fetch_error: {str(exc)[:200]}"
            logger.debug("market_indices: refresh failed: %s", exc)
            return

        now = datetime.now(timezone.utc)
        any_updated = False
        for sym in symbols:
            try:
                sym_data = data.get(sym, {})
                # Both equities and the $VIX index nest the actual
                # numbers under a "quote" subkey in schwab-py's
                # current response shape. data_providers/schwab.py
                # parses VIX with the same `data['$VIX']['quote']`
                # pattern (line ~228); mirror it.
                quote = sym_data.get("quote", {})
                if not quote:
                    continue

                last = quote.get("lastPrice")
                # Equities: netPercentChangeInDouble (and the alias
                # netPercentChange in some versions). VIX index: same.
                # We accept either to be resilient to schwab-py version
                # drift.
                change_pct = quote.get(
                    "netPercentChange",
                    quote.get("netPercentChangeInDouble"),
                )
                if last is None or change_pct is None:
                    continue

                self._snapshot[sym] = IndexQuote(
                    symbol=sym,
                    display_name=_DISPLAY_NAMES[sym],
                    last=float(last),
                    change_pct=float(change_pct),
                    ts=now,
                )
                any_updated = True
            except Exception as exc:
                # Don't let one bad symbol kill the whole refresh.
                logger.debug(
                    "market_indices: parse failed for %s: %s", sym, exc,
                )
                continue

        if any_updated:
            self.last_updated = now
            self.last_error = None

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot for the REST endpoint.

        Shape:
            {
              "indices": [
                {"symbol": str, "name": str, "last": float, "change_pct": float},
                ...
              ],
              "last_updated": str | None,  # ISO 8601 UTC
              "stale": bool,
              "error": str | None,
            }
        """
        now = datetime.now(timezone.utc)
        stale = (
            self.last_updated is None
            or (now - self.last_updated).total_seconds() > _STALE_AFTER_SEC
        )
        # Order indices in the same order as _DISPLAY_NAMES so the
        # frontend can render them consistently (S&P first, VIX last).
        ordered = [
            self._snapshot[sym] for sym in _DISPLAY_NAMES if sym in self._snapshot
        ]
        return {
            "indices": [
                {
                    "symbol": q.symbol,
                    "name": q.display_name,
                    "last": round(q.last, 2),
                    "change_pct": round(q.change_pct, 2),
                }
                for q in ordered
            ],
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "stale": stale,
            "error": self.last_error,
        }
