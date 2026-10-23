"""v-swing-lane-2026-10-23: the ONLY entry point to core.swing_lane.

Safe off-path: with ENABLE_SWING_SHADOW false (default) this returns before
importing anything from core.swing_lane. ENABLE_SWING_LIVE has no order path:
it makes the gate refuse (SwingLiveNotSupported). Nothing in the engine calls
this yet — wiring is an explicit, separate decision.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("swing_lane")


class SwingLiveNotSupported(RuntimeError):
    """Raised when ENABLE_SWING_LIVE is set: the swing lane cannot place orders."""


def check_startup(cfg) -> bool:
    """-> True if the shadow lane may run. Raises if LIVE is requested (with or without SHADOW)."""
    if cfg.ENABLE_SWING_LIVE:
        if not cfg.ENABLE_SWING_SHADOW:
            raise SwingLiveNotSupported("ENABLE_SWING_LIVE=true requires ENABLE_SWING_SHADOW=true; refusing")
        raise SwingLiveNotSupported("swing lane has no order path; LIVE requires promotion sign-off + new code")
    return bool(cfg.ENABLE_SWING_SHADOW)


def maybe_run_eod(cfg, now_et, **kw) -> list:
    """Call from a scheduler at 16:20-16:30 ET. Never raises into the caller (except LIVE refusal)."""
    if not check_startup(cfg):
        return []
    try:
        from core.swing_lane.config import LaneSettings
        from core.swing_lane import scheduler
        s = LaneSettings.from_config(cfg)
        rows = scheduler.run_eod_scan(s, now_et, **kw)
        rows += scheduler.update_exits(s)
        return rows
    except Exception as exc:
        logger.warning("swing_lane: EOD run failed (isolated): %s", exc, exc_info=True)
        return []


def maybe_record_fills(cfg, today) -> list:
    if not check_startup(cfg):
        return []
    try:
        from core.swing_lane.config import LaneSettings
        from core.swing_lane import scheduler
        return scheduler.record_fills(LaneSettings.from_config(cfg), today)
    except Exception as exc:
        logger.warning("swing_lane: fill record failed (isolated): %s", exc, exc_info=True)
        return []
