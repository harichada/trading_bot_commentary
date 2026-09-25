"""v-playbook-desk-b2-2026-09-25: file store for the desk's daily plan.

One JSON file per session under `<DESK_DATA_DIR>/plans/<date>.json`, written
atomically (temp file in the same directory + `os.replace`) so a reader — the
API, the monitor, a restarted loop — never sees a half-written plan.

Neither function raises: the loop that calls `save_plan` must survive a full
disk, and every `load_plan` caller treats "no plan" as a normal state, so a
missing or corrupt file degrades to `None` with a log line rather than an
exception on the trading path.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Optional

from core.desk.models import DailyPlan

logger = logging.getLogger(__name__)

__all__ = ["plan_path", "plans_dir", "save_plan", "load_plan"]

PLANS_SUBDIR = "plans"


def plans_dir(desk_dir: str) -> str:
    """Directory holding the dated plan files."""
    return os.path.join(desk_dir or ".", PLANS_SUBDIR)


def plan_path(desk_dir: str, date: str) -> str:
    """Absolute-or-relative path of one session's plan file."""
    return os.path.join(plans_dir(desk_dir), f"{date}.json")


def save_plan(plan: DailyPlan, desk_dir: str) -> Optional[str]:
    """Write `plan` atomically; return the path written, or None on failure.

    The temp file is created in the destination directory so `os.replace` is a
    same-filesystem rename, and is removed if anything goes wrong — a failed
    save must not leave a `.tmp` file for the next reader to trip over.
    """
    path = plan_path(desk_dir, plan.date)
    directory = os.path.dirname(path) or "."
    tmp_path = None
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".plan-", suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(plan.to_dict(), fh, indent=2, sort_keys=True, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
        return path
    except (OSError, TypeError, ValueError) as exc:
        logger.error("desk save_plan failed date=%s dir=%s err=%s", plan.date, desk_dir, exc)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as exc:  # pragma: no cover - best-effort cleanup
                logger.warning("desk save_plan could not remove temp file %s: %s",
                               tmp_path, exc)


def load_plan(desk_dir: str, date: str) -> Optional[DailyPlan]:
    """Read one session's plan; None when absent or unusable (never raises)."""
    path = plan_path(desk_dir, date)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        logger.warning("desk load_plan unreadable path=%s err=%s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("desk load_plan payload is not an object path=%s type=%s",
                       path, type(payload).__name__)
        return None
    try:
        return DailyPlan.from_dict(payload)
    except Exception as exc:  # pragma: no cover - from_dict is defensive already
        logger.warning("desk load_plan could not parse path=%s err=%s", path, exc)
        return None
