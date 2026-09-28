"""Risk Console change submission, pending queue, promotion and audit trail.

Built on the pure registry in `core/settings_console.py` (Knob, coerce,
classify, env_shadowed, yaml_value, effective_value). This module owns the
ratchet workflow: a "tighten" (more conservative) change applies immediately
via `manager.update`; a "loosen" (more permissive) change is queued as a
pending row and only takes effect at the next session open (09:25 ET),
giving a human a window to review or cancel it before it lands.

No import of `core.config` here — callers pass `manager` (a config manager
exposing `get`/`update`) and `cfg` (the live Config singleton, or a fake in
tests) explicitly. All state lives under `desk_dir` as two files:
`pending_settings.json` (a JSON list, written atomically) and
`settings_audit.ndjson` (one JSON object per line, append-only). Per the
task brief, `list_pending`, `read_audit` and `promote_pending` never raise:
any I/O or parse failure is logged and degrades to an empty/kept result.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from core.settings_console import (ALLOCATION_CEILING_PATH,
                                    DEFAULT_ALLOCATION_CEILING, KNOB_BY_KEY,
                                    classify, coerce, effective_value,
                                    env_shadowed, yaml_value)

logger = logging.getLogger(__name__)

PENDING_FILE = "pending_settings.json"
AUDIT_FILE = "settings_audit.ndjson"
PROMOTE_TIME_ET = (9, 25)

BLOCK_NEW_ENTRIES_PATH = "trading.block_new_entries"
_BLOCK_KEY = "block_new_entries"


@dataclass(frozen=True)
class ChangeResult:
    status: str            # "applied" | "pending" | "rejected" | "noop"
    key: str
    old: Any
    new: Any
    direction: str          # "tighten" | "loosen" | "noop" | ""
    effective_at: str       # ISO ET timestamp; "" when applied/rejected
    pending_id: str         # "" unless status == "pending"
    reason: str             # user-supplied reason, or the rejection message


# ────────────────────────────────────────────────────────────────────
# clock — weekday-only session days, matching core/loops/desk_steps.is_session_day
# ────────────────────────────────────────────────────────────────────

def _is_session_day(now: datetime) -> bool:
    try:
        return now.weekday() < 5
    except (AttributeError, TypeError):
        return False


def next_promotion(now: datetime) -> datetime:
    """Today 09:25 ET if `now` is before that on a session day, else the next
    session day (weekday-only) at 09:25 ET."""
    hour, minute = PROMOTE_TIME_ET
    today_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if _is_session_day(now) and now < today_at:
        return today_at
    candidate = today_at + timedelta(days=1)
    while not _is_session_day(candidate):
        candidate += timedelta(days=1)
    return candidate


# ────────────────────────────────────────────────────────────────────
# file plumbing — atomic writes, never-raise reads
# ────────────────────────────────────────────────────────────────────

def _ensure_dir(desk_dir: str) -> None:
    try:
        os.makedirs(desk_dir, exist_ok=True)
    except OSError as exc:
        logger.error("settings_console_store: cannot create desk_dir %s: %s", desk_dir, exc)


def _pending_path(desk_dir: str) -> str:
    return os.path.join(desk_dir, PENDING_FILE)


def _audit_path(desk_dir: str) -> str:
    return os.path.join(desk_dir, AUDIT_FILE)


def _atomic_write_json(path: str, data: Any) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(tmp_path, path)


def list_pending(desk_dir: str) -> list[dict]:
    """Read the pending queue. Never raises: any failure degrades to []."""
    path = _pending_path(desk_dir)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.error("settings_console_store: failed to read pending file %s: %s", path, exc)
        return []
    return data if isinstance(data, list) else []


def _write_pending(desk_dir: str, rows: list[dict]) -> None:
    _ensure_dir(desk_dir)
    try:
        _atomic_write_json(_pending_path(desk_dir), rows)
    except OSError as exc:
        logger.error("settings_console_store: failed to write pending file: %s", exc)


def _append_audit(desk_dir: str, row: dict) -> None:
    _ensure_dir(desk_dir)
    path = _audit_path(desk_dir)
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str))
            fh.write("\n")
    except OSError as exc:
        logger.error("settings_console_store: failed to append audit row: %s", exc)


def _audit(desk_dir: str, now: datetime, actor: str, event: str, key: str, old: Any,
           new: Any, direction: str, effective_at: str, reason: str) -> None:
    row = {
        "ts": now.isoformat(),
        "actor": actor,
        "event": event,
        "key": key,
        "old": old,
        "new": new,
        "direction": direction,
        "effective_at": effective_at,
        "reason": reason,
    }
    _append_audit(desk_dir, row)


def read_audit(desk_dir: str, limit: int = 50) -> list[dict]:
    """Read the last `limit` audit rows, oldest of the window first. Never
    raises: any failure degrades to []; malformed lines are skipped."""
    path = _audit_path(desk_dir)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.error("settings_console_store: failed to read audit file %s: %s", path, exc)
        return []
    rows: list[dict] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            logger.error("settings_console_store: skipping malformed audit row: %s", exc)
    return rows[-limit:] if limit is not None and limit >= 0 else rows


# ────────────────────────────────────────────────────────────────────
# submission
# ────────────────────────────────────────────────────────────────────

def _rejected(desk_dir: str, now: datetime, actor: str, key: str, raw_value: Any,
              reason: str) -> ChangeResult:
    _audit(desk_dir, now, actor, "rejected", key, None, raw_value, "", "", reason)
    return ChangeResult("rejected", key, None, raw_value, "", "", "", reason)


def submit_change(manager: Any, cfg: Any, desk_dir: str, key: str, raw_value: Any, *,
                   now: datetime, actor: str, reason: str) -> ChangeResult:
    """Validate and classify a change; apply tightens immediately, queue
    loosens until the next session open."""
    knob = KNOB_BY_KEY.get(key)
    if knob is None:
        return _rejected(desk_dir, now, actor, key, raw_value, "unknown setting")
    if knob.restart_required:
        return _rejected(desk_dir, now, actor, key, raw_value, "restart_required")
    if env_shadowed(knob):
        msg = (f"env_shadowed: {knob.env_name} is set in .env; the yaml value "
               f"is ignored until it is removed")
        return _rejected(desk_dir, now, actor, key, raw_value, msg)

    ceiling = manager.get(ALLOCATION_CEILING_PATH, DEFAULT_ALLOCATION_CEILING)
    try:
        new = coerce(knob, raw_value, ceiling=ceiling)
    except ValueError as exc:
        return _rejected(desk_dir, now, actor, key, raw_value, str(exc))

    old = effective_value(cfg, knob)
    if old is None:
        old = yaml_value(manager, knob)
    direction = classify(knob, old, new)
    stored = list(new) if knob.kind == "csv" else new

    if direction == "noop":
        return ChangeResult("noop", key, old, new, "noop", "", "", reason)
    if direction == "tighten":
        manager.update(knob.yaml_path, stored)
        _audit(desk_dir, now, actor, "applied", key, old, stored, direction, "", reason)
        return ChangeResult("applied", key, old, stored, direction, "", "", reason)
    return _queue_loosen(desk_dir, now, actor, key, old, stored, direction, reason)


def _queue_loosen(desk_dir: str, now: datetime, actor: str, key: str, old: Any, stored: Any,
                   direction: str, reason: str) -> ChangeResult:
    effective_at = next_promotion(now).isoformat()
    pending_id = uuid.uuid4().hex[:12]
    row = {
        "id": pending_id,
        "key": key,
        "old": old,
        "new": stored,
        "submitted_at": now.isoformat(),
        "effective_at": effective_at,
        "actor": actor,
        "reason": reason,
    }
    rows = [r for r in list_pending(desk_dir) if r.get("key") != key]
    rows.append(row)
    _write_pending(desk_dir, rows)
    _audit(desk_dir, now, actor, "pending", key, old, stored, direction, effective_at, reason)
    return ChangeResult("pending", key, old, stored, direction, effective_at, pending_id, reason)


def cancel_pending(desk_dir: str, pending_id: str, *, now: datetime, actor: str) -> bool:
    rows = list_pending(desk_dir)
    match = next((r for r in rows if r.get("id") == pending_id), None)
    if match is None:
        return False
    remaining = [r for r in rows if r.get("id") != pending_id]
    _write_pending(desk_dir, remaining)
    _audit(desk_dir, now, actor, "cancelled", match.get("key"), match.get("old"),
           match.get("new"), "loosen", match.get("effective_at", ""), match.get("reason", ""))
    return True


def _promote_row(manager: Any, desk_dir: str, now: datetime, row: dict) -> bool:
    """Try to apply one due pending row. Returns True on success. Never raises."""
    knob = KNOB_BY_KEY.get(row.get("key"))
    if knob is None:
        logger.error("settings_console_store: promote_pending unknown key %r; keeping row",
                      row.get("key"))
        return False
    try:
        manager.update(knob.yaml_path, row.get("new"))
    except Exception as exc:  # pragma: no cover - defensive, manager is caller-supplied
        logger.error("settings_console_store: failed to apply pending change for %s: %s",
                      row.get("key"), exc)
        return False
    _audit(desk_dir, now, row.get("actor", ""), "promoted", row.get("key"), row.get("old"),
           row.get("new"), "loosen", row.get("effective_at", ""), row.get("reason", ""))
    return True


def promote_pending(manager: Any, desk_dir: str, *, now: datetime) -> list[dict]:
    """Apply every pending row with `effective_at <= now`. Never raises: a row
    that fails to parse or apply is kept and logged, not dropped."""
    try:
        rows = list_pending(desk_dir)
        applied: list[dict] = []
        remaining: list[dict] = []
        for row in rows:
            try:
                effective_dt = datetime.fromisoformat(row.get("effective_at", ""))
            except (TypeError, ValueError) as exc:
                logger.error("settings_console_store: unparsable effective_at for pending "
                             "row %s: %s", row.get("id"), exc)
                remaining.append(row)
                continue
            if effective_dt > now or not _promote_row(manager, desk_dir, now, row):
                remaining.append(row)
                continue
            applied.append(row)
        _write_pending(desk_dir, remaining)
        return applied
    except Exception as exc:  # pragma: no cover - last-resort guard
        logger.error("settings_console_store: promote_pending failed: %s", exc)
        return []


# ────────────────────────────────────────────────────────────────────
# block / unblock new entries — always immediate
# ────────────────────────────────────────────────────────────────────

def set_block_new_entries(manager: Any, desk_dir: str, block: bool, *, now: datetime,
                           actor: str, reason: str, confirm: str = "") -> ChangeResult:
    if block:
        manager.update(BLOCK_NEW_ENTRIES_PATH, True)
        _audit(desk_dir, now, actor, "block_entries", _BLOCK_KEY, None, True, "", "", reason)
        return ChangeResult("applied", _BLOCK_KEY, None, True, "", "", "", reason)
    if confirm != "YES":
        return _rejected(desk_dir, now, actor, _BLOCK_KEY, False, "confirm required")
    manager.update(BLOCK_NEW_ENTRIES_PATH, False)
    _audit(desk_dir, now, actor, "unblock_entries", _BLOCK_KEY, None, False, "", "", reason)
    return ChangeResult("applied", _BLOCK_KEY, None, False, "", "", "", reason)
