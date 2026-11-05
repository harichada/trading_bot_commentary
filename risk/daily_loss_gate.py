"""v-daily-loss-pause-entries-2026-11-05: daily-loss circuit = pause NEW entries.

Owner requirement (2026-11-05): when the daily-loss limit (risk-manager
``MAX_DAILY_LOSS`` circuit) or the 5% emergency stop trips, the bot must ONLY
stop opening new positions. It must keep analysing, monitoring positions,
running exits / stops / re-brackets, syncing positions, refreshing P&L,
pulsing the heartbeat, logging decisions and serving the dashboard.

This module holds the latch. It is deliberately tiny and engine-free so it
is unit-testable:

  * ``trip(reason, ...)`` latches "entries paused" for the current ET date.
  * ``is_paused()`` -> (bool, reason); the latch clears itself at the ET
    date rollover (a fresh session starts un-paused).
  * The latch is mirrored to a small JSON sidecar (``daily_loss_pause.json``
    next to ``trading_state.json``) so a restart on the same ET day starts
    paused instead of re-opening entries until the first P&L refresh.
    A sidecar from another day is ignored. Sidecar I/O failures are logged
    and never raised — the gate must not crash any loop.

It never touches positions, orders, consecutive_losses, LIVE flags or
``day_trade_live_entries_enabled``.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - zoneinfo is stdlib on 3.9+
    _ET = None

logger = logging.getLogger("TradingBot")

DEFAULT_STATE_PATH = "daily_loss_pause.json"


def _et_now() -> datetime:
    return datetime.now(_ET) if _ET is not None else datetime.now()


class DailyLossGate:
    """Per-ET-day latch for "new entries paused by the daily-loss circuit"."""

    def __init__(self, state_path: Optional[str] = DEFAULT_STATE_PATH,
                 clock: Optional[Callable[[], datetime]] = None):
        self._state_path = Path(state_path) if state_path else None
        self._clock = clock or _et_now
        self._lock = threading.RLock()
        self._date: Optional[str] = None
        self._reason: Optional[str] = None
        self._since: Optional[str] = None
        self._detail: dict = {}
        self._load()

    # ── internals ────────────────────────────────────────────────────
    def _today(self) -> str:
        return self._clock().date().isoformat()

    def _load(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text())
            if data.get("date") == self._today() and data.get("reason"):
                self._date = data["date"]
                self._reason = str(data["reason"])
                self._since = data.get("since")
                self._detail = dict(data.get("detail") or {})
                logger.warning(
                    "daily_loss_gate restored: new entries PAUSED for %s "
                    "(reason=%s since=%s)", self._date, self._reason, self._since)
        except Exception as exc:  # never raise
            logger.warning("daily_loss_gate: could not read %s: %s",
                           self._state_path, exc)

    def _persist(self) -> None:
        if self._state_path is None:
            return
        try:
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps({
                "date": self._date, "reason": self._reason,
                "since": self._since, "detail": self._detail,
            }, default=str))
            os.replace(tmp, self._state_path)
        except Exception as exc:  # never raise
            logger.warning("daily_loss_gate: could not write %s: %s",
                           self._state_path, exc)

    def _rollover(self) -> None:
        if self._date is not None and self._date != self._today():
            logger.info("daily_loss_gate: ET day rollover %s -> %s — entries "
                        "un-paused", self._date, self._today())
            self._date = self._reason = self._since = None
            self._detail = {}

    # ── public API ───────────────────────────────────────────────────
    def trip(self, reason: str, **detail) -> bool:
        """Latch the pause for today. Returns True only on a NEW trip
        (so callers emit their commentary/alert once)."""
        with self._lock:
            self._rollover()
            if self._reason is not None:
                return False
            self._date = self._today()
            self._reason = str(reason)
            self._since = self._clock().isoformat()
            self._detail = {k: v for k, v in detail.items()}
            self._persist()
            logger.warning(
                "daily_loss_gate TRIPPED: new entries PAUSED for %s reason=%s "
                "detail=%s (exits/stops/monitoring continue)",
                self._date, self._reason, self._detail)
            return True

    def is_paused(self) -> Tuple[bool, Optional[str]]:
        with self._lock:
            self._rollover()
            return (self._reason is not None), self._reason

    def status(self) -> dict:
        with self._lock:
            self._rollover()
            return {
                "entries_paused": self._reason is not None,
                "entries_paused_reason": self._reason,
                "entries_paused_since": self._since,
                "entries_paused_detail": dict(self._detail),
            }
