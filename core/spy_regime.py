"""v-oversold-v2-spy-regime-2026-05-19: SPY market-regime cache.

Singleton cache that keeps a short rolling window of SPY 5-min closes and
exposes an EMA-slope helper used by gate E of the OversoldBounceV2 strategy.
Other strategies are free to consume `ema_slope_pct(period)` as a coarse
risk-on/risk-off filter without having to re-fetch SPY data themselves.

Threading model
---------------
The trading bot runs as a single-threaded asyncio process. Every consumer
of this cache (engine ticks, strategies, REST/WS handlers) is awaited on
the same loop, so no locking is needed. Do NOT call `update()` from a
worker thread.

Wiring
------
The engine is expected to call ``SpyRegimeCache.instance().update(close)``
once per SPY 5-min bar. This module does NOT subscribe to SPY itself.
Until that wiring lands the cache stays empty and `ema_slope_pct` returns
0.0 (treated as flat by callers).
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

# Module-level singleton storage; created lazily by `instance()`.
_INSTANCE: Optional["SpyRegimeCache"] = None

# Keep ~150 minutes of 5-min closes (30 bars). EMA periods 9/21 fit comfortably
# inside this window with room for a couple of historical EMA samples used to
# compute the slope.
_MAXLEN = 30

# Minimum closes before we trust the EMA-slope output. Below this we return
# 0.0 (flat) so downstream gates don't punish symbols on a cold-start cache.
_MIN_CLOSES_FOR_SLOPE = 10


class SpyRegimeCache:
    """Rolling cache of recent SPY 5-min closes with an EMA-slope helper.

    Use :meth:`instance` to obtain the process-wide singleton. The cache is
    intentionally tiny — it answers a single question (is SPY trending down
    aggressively right now?) and is not a substitute for full market data.
    """

    def __init__(self, maxlen: int = _MAXLEN) -> None:
        self._closes: Deque[float] = deque(maxlen=maxlen)

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------
    @classmethod
    def instance(cls) -> "SpyRegimeCache":
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = cls()
        return _INSTANCE

    @classmethod
    def reset_for_tests(cls) -> None:
        """Wipe the singleton — test-only helper, never call in production."""
        global _INSTANCE
        _INSTANCE = None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def update(self, spy_close: float) -> None:
        """Append a new SPY 5-min close to the cache.

        Silently ignores non-finite or non-positive values so a single
        bad tick can't poison the EMA computation.
        """
        try:
            value = float(spy_close)
        except (TypeError, ValueError):
            return
        if not (value > 0):
            return
        self._closes.append(value)

    # ------------------------------------------------------------------
    # Read-only helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._closes)

    def ema_slope_pct(self, period: int = 9) -> float:
        """Return the percentage change between the first and last EMA values.

        Calculates an EMA(`period`) over the rolling close window and reports
        ``(ema[-1] - ema[0]) / ema[0] * 100``. Returns 0.0 when there are
        fewer than ``_MIN_CLOSES_FOR_SLOPE`` cached closes — callers treat
        that as a flat regime.
        """
        if period <= 0:
            return 0.0
        if len(self._closes) < _MIN_CLOSES_FOR_SLOPE:
            return 0.0

        alpha = 2.0 / (period + 1.0)
        closes = list(self._closes)
        # Seed EMA with the first close — standard exponential smoother init.
        ema_values = [closes[0]]
        for c in closes[1:]:
            ema_values.append(alpha * c + (1.0 - alpha) * ema_values[-1])

        ema_first = ema_values[0]
        ema_last = ema_values[-1]
        if ema_first <= 0:
            return 0.0
        return (ema_last - ema_first) / ema_first * 100.0
