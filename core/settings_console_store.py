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

import contextlib
import fcntl
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterator

from core.settings_console_rules import (PROMOTE_TIME_ET, candidate_error,  # noqa: F401
                                         next_promotion, promote_time_of)
from core.settings_console import (ALLOCATION_CEILING_PATH,
                                    DEFAULT_ALLOCATION_CEILING, KNOB_BY_KEY,
                                    classify, coerce, effective_value,
                                    env_shadowed, yaml_value)

logger = logging.getLogger(__name__)

PENDING_FILE = "pending_settings.json"
AUDIT_FILE = "settings_audit.ndjson"
BLOCK_NEW_ENTRIES_PATH = "trading.block_new_entries"
_BLOCK_KEY = "block_new_entries"
_LOCK_FILE = "pending_settings.lock"

# One threading.Lock per desk_dir, shared by every in-process caller (the
# FastAPI worker and the desk loop thread both live in one process). Guarded
# by its own lock since the dict itself is mutated lazily on first use.
_process_locks: dict[str, threading.Lock] = {}
_process_locks_guard = threading.Lock()


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


def _lock_path(desk_dir: str) -> str:
    return os.path.join(desk_dir, _LOCK_FILE)


def _get_process_lock(desk_dir: str) -> threading.Lock:
    abs_dir = os.path.abspath(desk_dir)
    with _process_locks_guard:
        lock = _process_locks.get(abs_dir)
        if lock is None:
            lock = threading.Lock()
            _process_locks[abs_dir] = lock
        return lock


@contextlib.contextmanager
def _locked(desk_dir: str) -> Iterator[None]:
    """Serialise a pending-file read → mutate → write cycle: a threading.Lock
    per desk_dir for in-process callers (desk loop thread + API worker),
    plus an flock on `<desk_dir>/pending_settings.lock` as cheap
    belt-and-braces against a second process. Always releases both."""
    _ensure_dir(desk_dir)
    with _get_process_lock(desk_dir):
        fh = open(_lock_path(desk_dir), "a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()


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
    except (OSError, ValueError) as exc:
        # ValueError also catches UnicodeDecodeError from a corrupted file.
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
    if limit is None or limit < 0:
        return rows
    if limit == 0:
        return []
    return rows[-limit:]


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
    problem = candidate_error(manager, knob.yaml_path, stored)
    if problem:
        return _rejected(desk_dir, now, actor, key, raw_value, problem)
    if direction == "tighten":
        _cancel_pending_for_key(desk_dir, now, actor, key)
        manager.update(knob.yaml_path, stored)
        _audit(desk_dir, now, actor, "applied", key, old, stored, direction, "", reason)
        return ChangeResult("applied", key, old, stored, direction, "", "", reason)
    return _queue_loosen(desk_dir, now, actor, key, old, stored, direction, reason,
                         promote_time_of(cfg))


def _cancel_pending_for_key(desk_dir: str, now: datetime, actor: str, key: str) -> None:
    """A tighten on `key` makes any queued loosen for the same key stale —
    drop it before applying, so a cancelled/replaced request never lands
    later and undoes the tighten."""
    with _locked(desk_dir):
        rows = list_pending(desk_dir)
        match = next((r for r in rows if r.get("key") == key), None)
        if match is None:
            return
        _write_pending(desk_dir, [r for r in rows if r.get("key") != key])
    _audit(desk_dir, now, actor, "cancelled", key, match.get("old"), match.get("new"),
           "loosen", match.get("effective_at", ""), "superseded_by_tighten")


def _queue_loosen(desk_dir: str, now: datetime, actor: str, key: str, old: Any, stored: Any,
                   direction: str, reason: str,
                   promote_time: tuple[int, int] = PROMOTE_TIME_ET) -> ChangeResult:
    effective_at = next_promotion(now, promote_time).isoformat()
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
    with _locked(desk_dir):
        rows = [r for r in list_pending(desk_dir) if r.get("key") != key]
        rows.append(row)
        _write_pending(desk_dir, rows)
    _audit(desk_dir, now, actor, "pending", key, old, stored, direction, effective_at, reason)
    return ChangeResult("pending", key, old, stored, direction, effective_at, pending_id, reason)


def cancel_pending(desk_dir: str, pending_id: str, *, now: datetime, actor: str) -> bool:
    with _locked(desk_dir):
        rows = list_pending(desk_dir)
        match = next((r for r in rows if r.get("id") == pending_id), None)
        if match is None:
            return False
        _write_pending(desk_dir, [r for r in rows if r.get("id") != pending_id])
    _audit(desk_dir, now, actor, "cancelled", match.get("key"), match.get("old"),
           match.get("new"), "loosen", match.get("effective_at", ""), match.get("reason", ""))
    return True


def _promote_row(manager: Any, desk_dir: str, now: datetime, row: dict) -> str:
    """Try to apply one due pending row. Returns "applied", "keep" (transient
    failure, retry later) or "drop" (the queued value is no longer valid —
    audited as `promotion_rejected`). Never raises."""
    knob = KNOB_BY_KEY.get(row.get("key"))
    if knob is None:
        logger.error("settings_console_store: promote_pending unknown key %r; keeping row",
                      row.get("key"))
        return "keep"
    # Re-validate against TODAY's ceiling and config band: the row was checked
    # when it was queued, and the world may have moved since (I4).
    ceiling = manager.get(ALLOCATION_CEILING_PATH, DEFAULT_ALLOCATION_CEILING)
    try:
        value = coerce(knob, row.get("new"), ceiling=ceiling)
    except ValueError as exc:
        problem = str(exc)
    else:
        value = list(value) if knob.kind == "csv" else value
        problem = candidate_error(manager, knob.yaml_path, value)
    if problem:
        _audit(desk_dir, now, row.get("actor", ""), "promotion_rejected", row.get("key"),
               row.get("old"), row.get("new"), "loosen", row.get("effective_at", ""), problem)
        return "drop"
    try:
        manager.update(knob.yaml_path, value)
    except Exception as exc:  # pragma: no cover - defensive, manager is caller-supplied
        logger.error("settings_console_store: failed to apply pending change for %s: %s",
                      row.get("key"), exc)
        return "keep"
    _audit(desk_dir, now, row.get("actor", ""), "promoted", row.get("key"), row.get("old"),
           value, "loosen", row.get("effective_at", ""), row.get("reason", ""))
    return "applied"


def promote_pending(manager: Any, desk_dir: str, *, now: datetime) -> list[dict]:
    """Apply every pending row with `effective_at <= now`. Never raises: a row
    that fails to parse or apply is kept and logged, not dropped."""
    try:
        with _locked(desk_dir):
            rows = list_pending(desk_dir)
            applied: list[dict] = []
            remaining: list[dict] = []
            for row in rows:
                try:
                    effective_dt = datetime.fromisoformat(row.get("effective_at", ""))
                except (TypeError, ValueError) as exc:
                    logger.error("settings_console_store: unparsable effective_at for "
                                 "pending row %s: %s", row.get("id"), exc)
                    remaining.append(row)
                    continue
                if effective_dt > now:
                    remaining.append(row)
                    continue
                outcome = _promote_row(manager, desk_dir, now, row)
                if outcome == "keep":
                    remaining.append(row)
                elif outcome == "applied":
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
