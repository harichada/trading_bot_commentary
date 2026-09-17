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
  - v-pinned-watchlist-2026-09-17: reserve slots for pinned liquid core
    before filler movers. Dollar-volume ranking for non-pinned.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, List, Dict, Any

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

        cfg = Config()
        wl_size = cfg.WATCHLIST_SIZE

        # v-pinned-watchlist-2026-09-17: build watchlist with pinned
        # symbols first, then fill remaining slots with dollar-volume
        # ranked movers.
        final_watchlist = self._build_pinned_watchlist(screener, cfg, wl_size)
        if not final_watchlist:
            return

        self.engine.dynamic_watchlist = final_watchlist

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
            # Include info about pinned vs mover composition
            pinned_set = set(cfg.PINNED_WATCHLIST) if cfg.ENABLE_PINNED_WATCHLIST else set()
            pinned_in_wl = [s for s in final_watchlist if s in pinned_set]
            movers_in_wl = [s for s in final_watchlist if s not in pinned_set]
            
            msg = f"Now tracking: {', '.join(final_watchlist)}"
            if pinned_in_wl and cfg.ENABLE_PINNED_WATCHLIST:
                msg += f"\n  Pinned ({len(pinned_in_wl)}): {', '.join(pinned_in_wl)}"
                msg += f"\n  Movers ({len(movers_in_wl)}): {', '.join(movers_in_wl[:5])}{'...' if len(movers_in_wl) > 5 else ''}"
            
            commentary.add_commentary(
                TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.MARKET_ANALYSIS,
                    symbol=None,
                    title="📊 Watchlist Updated",
                    message=msg,
                    importance=6,
                )
            )

    def _build_pinned_watchlist(
        self,
        screener,
        cfg: Config,
        wl_size: int,
    ) -> List[str]:
        """v-pinned-watchlist-2026-09-17: build watchlist with pins first.

        1. If ENABLE_PINNED_WATCHLIST, reserve slots for pinned symbols
        2. Fill remaining slots with dollar-volume ranked movers
        3. Fallback to old behavior if pinned watchlist disabled
        """
        # Get all screener movers with their data for dollar-volume ranking
        top_movers: List[Dict[str, Any]] = screener.top_movers or []
        
        if not cfg.ENABLE_PINNED_WATCHLIST:
            # Fallback to legacy behavior: screener order + default anchors
            screener_symbols = screener.get_watchlist_symbols(limit=wl_size)
            if not screener_symbols:
                return []
            combined = list(dict.fromkeys(
                list(screener_symbols) + list(_DEFAULT_ANCHORS)
            ))
            return combined[:wl_size]

        # Build pinned set (uppercase for case-insensitive matching)
        pinned_list = cfg.PINNED_WATCHLIST
        pinned_set = set(s.upper() for s in pinned_list)
        
        # Index movers by symbol for O(1) lookup
        mover_by_sym: Dict[str, Dict[str, Any]] = {}
        for m in top_movers:
            sym = m.get('symbol', '').upper()
            if sym and sym not in mover_by_sym:
                mover_by_sym[sym] = m

        # Step 1: Collect pinned symbols (in config order, preserving priority)
        final: List[str] = []
        pinned_included: List[str] = []
        for sym in pinned_list:
            sym_upper = sym.upper()
            if len(final) >= wl_size:
                break
            # Pinned symbols get included even if not in today's movers
            # (they'll be fetched by the analysis loop regardless)
            if sym_upper not in final:
                final.append(sym_upper)
                pinned_included.append(sym_upper)

        # Step 2: Fill remaining slots with dollar-volume ranked movers
        remaining_slots = wl_size - len(final)
        if remaining_slots > 0:
            # Filter and rank non-pinned movers by dollar-volume
            min_dollar_vol = cfg.MIN_DOLLAR_VOLUME
            candidates: List[Dict[str, Any]] = []
            
            for m in top_movers:
                sym = m.get('symbol', '').upper()
                if not sym or sym in pinned_set or sym in final:
                    continue
                
                # Calculate dollar-volume (price × volume)
                price = float(m.get('last', 0) or m.get('price', 0) or 0)
                volume = float(m.get('volume', 0) or 0)
                dollar_vol = price * volume
                
                # Apply dollar-volume floor for non-pinned
                if dollar_vol < min_dollar_vol:
                    logger.debug(
                        "screener_loop: demoted %s: dollar_vol $%.1fM < $%.1fM floor",
                        sym, dollar_vol / 1e6, min_dollar_vol / 1e6,
                    )
                    continue
                
                candidates.append({
                    'symbol': sym,
                    'dollar_vol': dollar_vol,
                    'mover': m,
                })
            
            # Sort by dollar-volume descending (highest liquidity first)
            candidates.sort(key=lambda x: x['dollar_vol'], reverse=True)
            
            # Fill remaining slots
            movers_added: List[str] = []
            for c in candidates[:remaining_slots]:
                sym = c['symbol']
                if sym not in final:
                    final.append(sym)
                    movers_added.append(sym)

            if movers_added:
                logger.info(
                    "screener_loop: filled %d mover slots (by dollar-vol): %s",
                    len(movers_added),
                    ", ".join(movers_added[:5]) + ("..." if len(movers_added) > 5 else ""),
                )

        # Log the composition
        if final:
            logger.info(
                "screener_loop: watchlist built — %d pinned, %d movers, %d total: %s",
                len(pinned_included),
                len(final) - len(pinned_included),
                len(final),
                ", ".join(final[:10]) + ("..." if len(final) > 10 else ""),
            )

        return final
