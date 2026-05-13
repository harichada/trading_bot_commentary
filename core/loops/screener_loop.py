"""v-parallel-screener-loop-2026-05-12: extract the screener cycle
from `engine._analyze_markets_with_commentary` into its own
supervised task.

Why this exists
---------------
The previous design ran the screener as a cadence-gated block inside
`_analysis_loop_body`. That body is a single try/except scope spanning
emergency-stop check → market analysis → screener → signal generation
→ ML retrain → state save. The 2026-05-11 `pnl_to_check` NameError
fired in the emergency-stop block and aborted every downstream step in
the same iteration — including the screener — for ~6 hours of RTH.

With the screener as its own supervised task, a crash anywhere in the
signal pipeline can no longer freeze the watchlist refresh. The
`TaskSupervisor` already provides exponential backoff and a
circuit-break after N crashes in T seconds, so the loop body only
needs a tight per-iteration try/except.

What this owns
--------------
The new loop is the *sole writer* of:
  - `engine.dynamic_watchlist`              (list[str])
  - `engine.screener.top_movers`            (delegated to StockScreener)
  - `engine.last_screener_run`              (datetime; retained for
    backward compat with code that inspects it)
  - PriceBook interest scopes "watchlist" and "ticker_tape"

The signal-generation pipeline reads `engine.dynamic_watchlist` and
`engine.screener.top_movers`; those reads are unchanged.

Migrated v-tags (must remain greppable for the inventory test):
  - v-watchlist-size-2026-04-29: cap is driven by Config.WATCHLIST_SIZE
  - v-watchlist-stream-2026-05-01: register watchlist symbols as a
    PriceBook interest scope so the ticker tape ticks on every Schwab
    stream update.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from core.config import Config
from core.models import CommentaryType
from core.commentary import TradingCommentary

if TYPE_CHECKING:  # pragma: no cover
    from core.engine import TradingEngineWithCommentary

logger = logging.getLogger("TradingBot")

_DEFAULT_ANCHORS = ("NVDA", "TSLA", "PLTR")


class ScreenerLoop:
    """Runs `screener.screen_stocks()` on its own cadence and propagates
    the resulting watchlist to the engine and the PriceBook.

    The loop never raises out of `run()`. Per-iteration failures are
    logged and swallowed so each tick is independent — only structural
    errors (e.g., the engine handle disappearing) would surface to the
    supervisor.
    """

    def __init__(
        self,
        engine: "TradingEngineWithCommentary",
        cadence_sec: float | None = None,
    ) -> None:
        self.engine = engine
        self.cadence_sec = (
            cadence_sec if cadence_sec is not None else Config().SCREENER_LOOP_SEC
        )

    async def run(self) -> None:
        while getattr(self.engine, "is_running", False):
            try:
                await self._tick()
            except Exception as exc:
                # Per-iteration guard. The supervisor only needs to see
                # structural failures, not transient screener errors.
                logger.warning("screener_loop: tick failed: %s", exc, exc_info=True)
            await asyncio.sleep(self.cadence_sec)

    async def _tick(self) -> None:
        screener = getattr(self.engine, "screener", None)
        if screener is None:
            return  # No screener configured (e.g., test harness)

        await screener.screen_stocks()
        self.engine.last_screener_run = datetime.now()

        # v-watchlist-size-2026-04-29: cap driven centrally by Config.
        wl_size = Config().WATCHLIST_SIZE
        screener_symbols = screener.get_watchlist_symbols(limit=wl_size)
        if not screener_symbols:
            return

        combined = list(dict.fromkeys(list(screener_symbols) + list(_DEFAULT_ANCHORS)))
        self.engine.dynamic_watchlist = combined[:wl_size]

        # v-watchlist-stream-2026-05-01: register watchlist + ticker
        # tape interest scopes so streamed ticks update prices in both
        # tiles without an extra REST round-trip.
        price_book = getattr(self.engine, "price_book", None)
        if price_book is not None:
            try:
                await price_book.register_interest(
                    "watchlist", set(self.engine.dynamic_watchlist)
                )
            except Exception as exc:
                logger.debug(f"register_interest watchlist failed: {exc}")

            try:
                ticker_syms = [
                    m.get("symbol")
                    for m in (screener.top_movers or [])
                    if m.get("symbol")
                ]
                if ticker_syms:
                    await price_book.register_interest(
                        "ticker_tape", set(ticker_syms)
                    )
            except Exception as exc:
                logger.debug(f"register_interest ticker_tape failed: {exc}")

        commentary = getattr(self.engine, "commentary", None)
        if commentary is not None:
            commentary.add_commentary(
                TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.MARKET_ANALYSIS,
                    symbol=None,
                    title="📊 Watchlist Updated",
                    message=f"Now tracking: {', '.join(self.engine.dynamic_watchlist)}",
                    importance=6,
                )
            )
