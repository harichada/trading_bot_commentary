"""Symbol Intelligence Hub — the bot's realtime per-symbol read.

v-symbol-intel-2026-06-10. Operator request (2026-06-09/10): "analyse
each symbol in realtime — its direction, price movement, news
sentiment, indicators and everything which determines the trade
quality." Everything below was ALREADY computed inside the engine per
analysis cycle; this module assembles it into one snapshot per symbol
and serves two consumers from a single build:

  1. The dashboard's Symbol Intelligence panel — the live snapshot
     rides the existing dashboard_update WebSocket payload.
  2. Composite View Phase 1 (docs/COMPOSITE_VIEW_ARCHITECTURE.md) —
     the same views append to symbol_intel_log.ndjson as the
     read-only evidence ledger the ADR's migration plan starts with.

Quality score v1 is a transparent heuristic, not a trained model:
direction conviction (50%), trend strength via ADX (25%), volume
participation (25%). It exists so the panel can sort; the Composite
View phases will replace it with the evidence-weighted composite.
Sentiment is None in v1 — the news pipeline's per-symbol cache wiring
lands with Composite View Phase 3 (the panel renders "—").
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("TradingBot")

DEFAULT_LEDGER = "symbol_intel_log.ndjson"
# Ledger cadence per symbol. The UI snapshot updates every analysis
# cycle regardless — this throttle only bounds disk growth.
_LEDGER_THROTTLE_SEC = 10 * 60.0


def _default_session_gate() -> bool:
    """Ledger writes only during trusted-data hours (10:00–16:00 ET
    weekdays) — same rationale and window as the SHORT shadow gate
    (v-shadow-log-gating-2026-06-10): pre-10:00 indicators ride too
    few fresh bars, after-close marks go stale/corrupt."""
    from zoneinfo import ZoneInfo
    et = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return False
    minutes = et.hour * 60 + et.minute
    return 10 * 60 <= minutes < 16 * 60


def build_symbol_view(
    symbol: str,
    close: float,
    indicators: Dict[str, Any],
    tape_er: Optional[float] = None,
    sentiment: Optional[float] = None,
) -> Dict[str, Any]:
    """Assemble one symbol's realtime intelligence snapshot.

    Pure function over already-computed inputs — no I/O, no engine
    coupling. Degrades gracefully on missing indicators (scores lean
    neutral, never raises)."""
    def _f(key: str, default: float = 0.0) -> float:
        try:
            v = float(indicators.get(key, default))
            return default if v != v else v  # NaN → default
        except (TypeError, ValueError):
            return default

    from core.direction_reader import read_direction
    try:
        dr = read_direction(close, indicators)
        direction, phase, components = dr.direction, dr.phase, dr.components
        dir_reason = dr.reason
    except Exception as exc:  # reader must never break the panel
        logger.debug("symbol_intel: direction read failed for %s: %s",
                     symbol, exc)
        direction, phase, components, dir_reason = 0.0, "unknown", {}, "n/a"

    adx = _f("adx")
    vol_ratio = _f("volume_ratio", 1.0)
    atr = _f("atr")
    rsi = _f("rsi", 50.0)
    macd_hist = _f("macd_histogram")
    bb_upper, bb_lower = _f("bb_upper"), _f("bb_lower")
    bb_pos = None
    if bb_upper > bb_lower > 0:
        bb_pos = round((close - bb_lower) / (bb_upper - bb_lower), 2)

    # Quality v1: conviction (|direction|) 50% + trend strength 25% +
    # volume participation 25%. Transparent and replaceable.
    quality = round(
        (min(abs(direction), 10.0) / 10.0) * 50.0
        + (min(adx, 50.0) / 50.0) * 25.0
        + (min(vol_ratio, 3.0) / 3.0) * 25.0,
        1,
    )

    return {
        "symbol": symbol,
        "price": close,
        "direction": round(direction, 2),
        "phase": phase,
        "direction_components": components,
        "direction_reason": dir_reason,
        "tape_er": tape_er,
        "rsi": round(rsi, 1),
        "macd_hist": round(macd_hist, 4),
        "bb_position": bb_pos,
        "atr_pct": round(atr / close * 100, 2) if close > 0 else 0.0,
        "volume_ratio": round(vol_ratio, 2),
        "sentiment": sentiment,
        "quality": quality,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


class SymbolIntelHub:
    """Latest view per symbol + throttled NDJSON evidence ledger.

    Single-writer by design: updated from the engine's analysis loop
    only. snapshot() is read by the WS payload builder on the FastAPI
    loop — a plain dict swap, no locking needed under the GIL for
    whole-value replacement."""

    def __init__(
        self,
        ledger_path: Path | str = DEFAULT_LEDGER,
        session_gate: Callable[[], bool] = _default_session_gate,
        max_symbols: int = 200,
    ) -> None:
        self._views: Dict[str, Dict[str, Any]] = {}
        self._ledger_path = Path(ledger_path)
        self._session_gate = session_gate
        self._max_symbols = max_symbols
        self._last_logged: Dict[str, float] = {}

    def update(self, view: Dict[str, Any]) -> None:
        """Store the latest view; append to the ledger if the session
        gate is open and the per-symbol throttle has elapsed. Never
        raises into the analysis loop."""
        try:
            symbol = view["symbol"]
            self._views[symbol] = view
            # Bound memory: drop the stalest symbols beyond the cap.
            if len(self._views) > self._max_symbols:
                stalest = sorted(self._views.values(),
                                 key=lambda v: v.get("updated_at", ""))
                for v in stalest[: len(self._views) - self._max_symbols]:
                    self._views.pop(v["symbol"], None)

            if not self._session_gate():
                return
            now = time.monotonic()
            last = self._last_logged.get(symbol, 0.0)
            if now - last < _LEDGER_THROTTLE_SEC:
                return
            self._last_logged[symbol] = now
            # Ledger rows exclude the verbose reason/components text —
            # the resolver needs numbers, the UI shows the prose.
            row = {k: v for k, v in view.items()
                   if k not in ("direction_reason", "direction_components")}
            with open(self._ledger_path, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception as exc:
            logger.debug("symbol_intel update error: %s", exc)

    def snapshot(self) -> List[Dict[str, Any]]:
        """All current views, best quality first (panel default sort)."""
        return sorted(self._views.values(),
                      key=lambda v: v.get("quality", 0), reverse=True)
