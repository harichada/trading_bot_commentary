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
  - v-pinned-watchlist-2026-09-17: reserve seats for pinned symbols
    before mover-filler. Pinned names (liquid megas) always get slots;
    remaining slots filled by dollar-volume-ranked movers.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, List, Set

from core.config import Config
from core.models import CommentaryType
from core.commentary import TradingCommentary

if TYPE_CHECKING:  # pragma: no cover
    from core.engine import TradingEngineWithCommentary

logger = logging.getLogger("TradingBot")


class ScreenerLoop:
    """Runs `screener.screen_stocks()` on its own cadence and propagates
    the resulting watchlist to the engine and the PriceBook.

    The loop never raises out of `run()`. Per-iteration failures are
    logged and swallowed so each tick is independent — only structural
    errors (e.g., the engine handle disappearing) would surface to the
    supervisor.

    v-pinned-watchlist-2026-09-17: reserves seats for pinned symbols
    (liquid megas) before filling remaining slots with movers ranked
    by dollar-volume. HANDS_OFF symbols are excluded from all slots.
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
                logger.warning("screener_loop: tick failed: %s", exc, exc_info=True)
            await asyncio.sleep(self.cadence_sec)

    def _get_hands_off_symbols(self) -> Set[str]:
        """Get the HANDS_OFF denylist — symbols never auto-traded/managed."""
        try:
            return set(Config().HANDS_OFF_DENYLIST)
        except Exception:
            return {'MU', 'HQGE', 'SPCX'}

    def _get_pinned_watchlist(self, hands_off: Set[str]) -> List[str]:
        """Get pinned watchlist with HANDS_OFF symbols excluded.

        v-pinned-watchlist-2026-09-17: pinned symbols always get reserved
        seats in the dynamic watchlist, but HANDS_OFF symbols are never
        included even if configured in the pinned list.
        """
        cfg = Config()
        if not cfg.ENABLE_PINNED_WATCHLIST:
            return []
        pinned = cfg.PINNED_WATCHLIST
        return [s for s in pinned if s not in hands_off]

    async def _tick(self) -> None:
        screener = getattr(self.engine, "screener", None)
        if screener is None:
            return

        await screener.screen_stocks()
        self.engine.last_screener_run = datetime.now()

        cfg = Config()
        wl_size = cfg.WATCHLIST_SIZE
        hands_off = self._get_hands_off_symbols()

        # v-pinned-watchlist-2026-09-17: reserve seats for pinned symbols first.
        pinned = self._get_pinned_watchlist(hands_off)
        pinned_in_watchlist = pinned[:wl_size]

        # Fill remaining slots with screener movers (excluding pinned + HANDS_OFF).
        # v-pinned-watchlist-2026-09-17: apply dollar-volume filter to filler slots
        # to demote micro lottery names.
        remaining_slots = wl_size - len(pinned_in_watchlist)
        pinned_set = set(pinned_in_watchlist)
        min_dollar_volume = cfg.MIN_DOLLAR_VOLUME_FLOOR

        if remaining_slots > 0:
            movers = screener.get_movers_with_dollar_volume()
            filler_symbols = []
            filler_rejected_dv = 0

            for mover in movers:
                if len(filler_symbols) >= remaining_slots:
                    break
                sym = mover.get('symbol')
                if not sym or sym in pinned_set or sym in hands_off:
                    continue
                dv = mover.get('dollar_volume', 0)
                if min_dollar_volume > 0 and dv < min_dollar_volume:
                    filler_rejected_dv += 1
                    continue
                filler_symbols.append(sym)

            if filler_rejected_dv > 0:
                logger.debug(
                    "screener_loop: filler rejected %d movers below $%.1fM dollar-volume",
                    filler_rejected_dv, min_dollar_volume / 1_000_000,
                )
        else:
            filler_symbols = []

        # Combine: pinned first, then filler movers (preserves order, deduped).
        combined = list(dict.fromkeys(pinned_in_watchlist + filler_symbols))
        self.engine.dynamic_watchlist = combined[:wl_size]

        # Log composition for observability.
        if pinned_in_watchlist:
            logger.info(
                "screener_loop: watchlist pinned=%d filler=%d total=%d pinned_syms=%s",
                len(pinned_in_watchlist),
                len(filler_symbols),
                len(self.engine.dynamic_watchlist),
                ",".join(pinned_in_watchlist[:5]),
            )

        # v-watchlist-stream-2026-05-01: register watchlist + ticker tape interest.
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
            pinned_count = len(pinned_in_watchlist)
            filler_count = len(filler_symbols)
            commentary.add_commentary(
                TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.MARKET_ANALYSIS,
                    symbol=None,
                    title="📊 Watchlist Updated",
                    message=(
                        f"Now tracking: {', '.join(self.engine.dynamic_watchlist)} "
                        f"(pinned: {pinned_count}, movers: {filler_count})"
                    ),
                    importance=6,
                )
            )
