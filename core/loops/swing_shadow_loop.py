"""v-swing-lane-2026-10-23: engine-side scheduler for the watch-only swing shadow.

Registered by engine._build_supervisor ONLY when Config().ENABLE_SWING_SHADOW is true
(default false => this module is never imported). Talks to the lane solely through
core.swing_lane_gate. No orders, no broker calls; the gate's own flag check is a
second guard. Every step runs in a worker thread and is exception-isolated.

ET schedule (weekdays):
  >= 09:35        maybe_record_fills(today)            once per day
  16:05-16:19     maybe_refresh_earnings(today)        once per day (network, ~30-90s)
  16:20-16:30     maybe_run_eod(now_et, held, realized) once per day
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time
from zoneinfo import ZoneInfo

logger = logging.getLogger("swing_lane")
ET = ZoneInfo("America/New_York")
FILLS_AFTER = time(9, 35)
REFRESH_WINDOW = (time(16, 5), time(16, 19))
EOD_WINDOW = (time(16, 20), time(16, 30))
TICK_SEC = 30


class SwingShadowLoop:
    def __init__(self, engine, cfg_factory=None, clock=None):
        self.engine = engine
        self._cfg = cfg_factory
        self._clock = clock or (lambda: datetime.now(ET))
        self.done = {}  # step -> date it last ran

    def cfg(self):
        if self._cfg is not None:
            return self._cfg()
        from core.config import Config
        return Config()

    def _held(self):
        try:
            return set(self.engine._symbols_with_open_positions())
        except Exception:
            return set()

    def _realized(self, cfg, today):
        try:
            from core.learning.risk_caps import realized_bot_pnl_today
            return realized_bot_pnl_today(getattr(self.engine, "trade_history", []) or [],
                                          today, cfg.HANDS_OFF_DENYLIST)
        except Exception:
            return None  # risk.py treats None as "unknown" (no daily-loss block info)

    async def tick(self) -> list:
        """One scheduling pass. Returns the step names that ran (for tests)."""
        cfg = self.cfg()
        if not cfg.ENABLE_SWING_SHADOW:
            return []
        from core import swing_lane_gate as gate
        now = self._clock().astimezone(ET)
        today, t = now.date(), now.time()
        ran = []
        if now.weekday() >= 5:
            return ran
        if t >= FILLS_AFTER and self.done.get("fills") != today:
            self.done["fills"] = today
            await asyncio.to_thread(gate.maybe_record_fills, cfg, today)
            ran.append("fills")
        if REFRESH_WINDOW[0] <= t <= REFRESH_WINDOW[1] and self.done.get("earnings") != today:
            self.done["earnings"] = today
            await asyncio.to_thread(gate.maybe_refresh_earnings, cfg, today)
            ran.append("earnings")
        if EOD_WINDOW[0] <= t <= EOD_WINDOW[1] and self.done.get("eod") != today:
            self.done["eod"] = today
            if self.done.get("earnings") != today:  # bot started late: refresh first
                self.done["earnings"] = today
                await asyncio.to_thread(gate.maybe_refresh_earnings, cfg, today)
                ran.append("earnings")
            await asyncio.to_thread(gate.maybe_run_eod, cfg, now, held_symbols=self._held(),
                                    realized_today=self._realized(cfg, today))
            ran.append("eod")
        return ran

    async def run(self) -> None:
        while getattr(self.engine, "is_running", False):
            try:
                await self.tick()
            except Exception as exc:  # never let the lane affect the bot
                logger.warning("swing_shadow_loop: tick failed (isolated): %s", exc, exc_info=True)
            await asyncio.sleep(TICK_SEC)
