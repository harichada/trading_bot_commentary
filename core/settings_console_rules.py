"""v-risk-console-2026-09-28: the pure rules the settings-console store
applies — session-day arithmetic for the 09:25 promotion, and the pre-write
check that a candidate value would not stop the bot from loading its config.

No file I/O, no core.config import (the manager passed in supplies
`validate_candidate` and `config`).
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
from typing import Any

PROMOTE_TIME_ET = (9, 25)


def is_session_day(now: datetime) -> bool:
    try:
        return now.weekday() < 5
    except (AttributeError, TypeError):
        return False


def promote_time_of(cfg: Any) -> tuple[int, int]:
    """(hour, minute) from `cfg.SETTINGS_PROMOTE_TIME_ET` ("HH:MM"), falling
    back to PROMOTE_TIME_ET — the same source the desk loop schedules from,
    so the badge and the promotion never disagree."""
    raw = getattr(cfg, "SETTINGS_PROMOTE_TIME_ET", None)
    try:
        hh, mm = str(raw).strip().split(":")
        hour, minute = int(hh), int(mm)
        if 0 <= hour < 24 and 0 <= minute < 60:
            return hour, minute
    except (AttributeError, TypeError, ValueError):
        pass
    return PROMOTE_TIME_ET


def next_promotion(now: datetime, promote_time: tuple[int, int] = PROMOTE_TIME_ET) -> datetime:
    """Today at `promote_time` ET if `now` is before that on a session day,
    else the next session day (weekday-only) at that time."""
    hour, minute = promote_time
    today_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if is_session_day(now) and now < today_at:
        return today_at
    candidate = today_at + timedelta(days=1)
    while not is_session_day(candidate):
        candidate += timedelta(days=1)
    return candidate



def candidate_error(manager: Any, yaml_path: str, value: Any) -> str:
    """Would writing `value` at `yaml_path` produce a config the bot refuses to
    load? Runs the manager's own `validate_candidate` (if it has one) on a deep
    copy; returns the error message, or "" when the write is safe."""
    validate = getattr(manager, "validate_candidate", None)
    config = getattr(manager, "config", None)
    if validate is None or not isinstance(config, dict):
        return ""
    candidate = copy.deepcopy(config)
    node = candidate
    keys = yaml_path.split(".")
    for part in keys[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            return f"{yaml_path} is not a mapping"
    node[keys[-1]] = value
    try:
        validate(candidate)
    except ValueError as exc:
        return f"config_validation: {exc}"
    return ""


