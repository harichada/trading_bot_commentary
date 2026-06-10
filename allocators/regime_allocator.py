"""Regime allocator — SPY trend-efficiency decides breakout vs mean-rev.

v-regime-allocator-shadow-2026-06-10 (P1 of the profitability roadmap).

Evidence: research/regime_allocated_report_2026-06-10.json. Six-window
walk-forward (Apr 2025 → Apr 2026): allocating between breakout and
mean-reversion by SPY 20-day Kaufman Efficiency Ratio lifted PF
1.04 → 1.11 and summed returns +106% → +188% with 45% fewer trades.
The lift is disaster-avoidance: the allocated line dodged mean-rev's
−104.7% window (Jun–Aug 2025) and breakout's −56.5% window (Oct–Dec
2025). Stable across ER cuts 0.30–0.40; degrades below 0.25.

SHADOW MODE ONLY in this version: the engine logs what the allocator
WOULD have done for every routed signal. It gates nothing. Promotion
criterion (roadmap §4): two weeks of live shadow data agreeing with
the walk-forward before any gating wire-up.

Pure functions (`efficiency_ratio`, `allocate`) are kept free of I/O
so they are trivially testable and reusable by the backtest harness.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

logger = logging.getLogger("TradingBot")

DEFAULT_THRESHOLD = 0.30
DEFAULT_LEDGER = "regime_allocator_shadow.ndjson"
# Daily closes move once per session; re-fetching more often than a few
# hours is pure API noise.
_CLOSES_CACHE_TTL_SEC = 4 * 3600.0


def efficiency_ratio(
    closes: Sequence[float], lookback: int = 20
) -> Optional[float]:
    """Kaufman Efficiency Ratio over the last `lookback` steps.

    ER = |net change| / Σ|step changes|. 1.0 = perfectly directional,
    0.0 = pure oscillation. Returns None when history is insufficient.
    """
    if closes is None or len(closes) < lookback + 1:
        return None
    window = list(closes)[-(lookback + 1):]
    net = abs(window[-1] - window[0])
    path = sum(abs(window[i + 1] - window[i]) for i in range(lookback))
    if path <= 0:
        return 0.0
    return net / path


@dataclass(frozen=True)
class RegimeAllocation:
    er: Optional[float]
    threshold: float
    tape: str               # 'trending' | 'choppy' | 'unknown'
    breakout_allowed: bool
    mean_rev_allowed: bool
    reason: str

    def allows(self, strategy: str) -> bool:
        """Would this allocation permit `strategy` to trade?

        Strategies outside the breakout/mean-rev pair (news, momentum)
        are not allocated and always pass — the walk-forward evidence
        only covers the anti-correlated pair.
        """
        if strategy == "breakout":
            return self.breakout_allowed
        if strategy in ("mean_reversion", "oversold_v2"):
            return self.mean_rev_allowed
        return True


def allocate(
    er: Optional[float], threshold: float = DEFAULT_THRESHOLD
) -> RegimeAllocation:
    """Map an efficiency ratio to per-strategy permissions.

    Fail-open on unknown ER: a data hiccup must never read as
    "block everything" — in shadow that would poison the ledger, and
    if this is ever promoted to a gate, fail-closed on missing data
    would halt the bot on every Schwab blip.
    """
    if er is None:
        return RegimeAllocation(
            er=None, threshold=threshold, tape="unknown",
            breakout_allowed=True, mean_rev_allowed=True,
            reason="no ER (insufficient history or fetch failure) — fail-open",
        )
    if er >= threshold:
        return RegimeAllocation(
            er=er, threshold=threshold, tape="trending",
            breakout_allowed=True, mean_rev_allowed=False,
            reason=f"ER {er:.2f} >= {threshold:.2f}: directional tape — "
                   f"breakout on, mean-rev off",
        )
    return RegimeAllocation(
        er=er, threshold=threshold, tape="choppy",
        breakout_allowed=False, mean_rev_allowed=True,
        reason=f"ER {er:.2f} < {threshold:.2f}: choppy tape — "
               f"mean-rev on, breakout off",
    )


class RegimeAllocatorShadow:
    """Shadow wrapper: caches SPY daily closes, evaluates signals,
    appends one NDJSON line per routed signal. Never raises into the
    signal path."""

    def __init__(
        self,
        fetch_daily_closes: Callable[[], Sequence[float]],
        threshold: float = DEFAULT_THRESHOLD,
        ledger_path: Path | str = DEFAULT_LEDGER,
        lookback: int = 20,
    ) -> None:
        self._fetch = fetch_daily_closes
        self._threshold = threshold
        self._ledger_path = Path(ledger_path)
        self._lookback = lookback
        self._cached_er: Optional[float] = None
        self._cached_at: float = 0.0

    async def _current_er(self) -> Optional[float]:
        """Cached ER. The fetch is a synchronous Schwab HTTP call —
        run it in the default thread executor with a hard timeout so a
        hung API can never block the engine's event loop (same pattern
        as strategies/news_strategy.py _run_sync_with_timeout; the
        MRVL 2026-04-29 incident is why this is non-negotiable)."""
        now = time.monotonic()
        if self._cached_at and now - self._cached_at < _CLOSES_CACHE_TTL_SEC:
            return self._cached_er
        try:
            loop = asyncio.get_event_loop()
            closes = await asyncio.wait_for(
                loop.run_in_executor(None, self._fetch), timeout=8.0
            )
            self._cached_er = efficiency_ratio(closes, self._lookback)
        except Exception as exc:
            logger.warning("regime_allocator: SPY closes fetch failed: %s",
                           exc)
            self._cached_er = None
        self._cached_at = now
        return self._cached_er

    async def evaluate(
        self, strategy: str, symbol: str, signal_side: str
    ) -> Optional[RegimeAllocation]:
        """Evaluate a routed signal and append the would-be decision to
        the ledger. Returns the allocation, or None on total failure
        (callers treat None as 'no opinion')."""
        try:
            allocation = allocate(await self._current_er(), self._threshold)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "strategy": strategy,
                "symbol": symbol,
                "signal_side": signal_side,
                "er": None if allocation.er is None
                      else round(allocation.er, 4),
                "threshold": allocation.threshold,
                "tape": allocation.tape,
                "would_allow": allocation.allows(strategy),
                "reason": allocation.reason,
            }
            try:
                # NDJSON append — O_APPEND < 4 KiB is atomic on Linux,
                # same rationale as shadow_short_log.ndjson.
                with open(self._ledger_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except OSError as io_exc:
                # Disk full / permissions: shadow data is the whole
                # point of this module — surface it, don't debug-bury.
                logger.warning("regime_allocator: ledger write failed: %s",
                               io_exc)
            return allocation
        except Exception as exc:
            logger.debug("regime_allocator shadow evaluate error: %s", exc)
            return None
