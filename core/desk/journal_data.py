"""v-playbook-desk-b2-2026-09-25 (Task 4): data adapters for the desk journal —
today's decisions and closed trades read from Postgres through `DbLogger`,
and the atomic file writer for `<DESK_DATA_DIR>/journal/<date>.md`.

Both read adapters are async (they await `DbLogger` coroutines) and never
raise: a Postgres outage on the 16:05 ET journal job must still produce a
journal from whatever local plan/trailing data is available, not crash the
loop. `render_journal` (`core/desk/journal.py`) is pure and untouched by any
of this — these adapters only shape rows for it.

Family naming: `family` on each trade comes from
`core.strategy_fit_gate.family_for` — the live strategy -> family map. (The
B2 plan cites this as `strategies/fit_lane.py::LANE_MAP`; that module
re-exports nothing of the sort — the map actually lives in
`core/strategy_fit_gate.py`, as established by Task 3's
`research/strategy_hunt/playbook_update.py`.) Importing it here is a
core-internal import (`core.desk` -> `core.strategy_fit_gate`), not a
`core` -> `strategies` dependency, so it needs no parameter escape hatch.

`decisions_for` is limited by what `DbLogger` exposes today:
`get_todays_bot_decision_entries` only returns *entry* signals (symbol ->
side/ts/strategy/confidence) for a fixed set of managed strategies, not the
strategy-fit / playbook gate's skip/shadow audit rows. Those aren't
queryable through `DbLogger` yet, so the "skips" and "shadow audits that
would have blocked" journal sections stay empty against the live adapter
until a future task adds that read — `render_journal` itself already
handles decision dicts carrying `action="skip"`/`"shadow"` correctly (see
its tests), so no rendering change will be needed then.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, List, Optional, Tuple

from core.strategy_fit_gate import family_for

logger = logging.getLogger(__name__)

__all__ = ["decisions_for", "trades_for", "journal_dir", "journal_path", "write_journal"]

JOURNAL_SUBDIR = "journal"


# ────────────────────────────────────────────────────────────────────
# Read adapters
# ────────────────────────────────────────────────────────────────────

async def decisions_for(date: str, db_logger: Any) -> List[dict]:
    """Today's governed decisions, shaped for `render_journal`.

    `date` is accepted for a future date-scoped query and otherwise unused:
    `DbLogger.get_todays_bot_decision_entries` only ever reads "today" in
    Postgres. Never raises: [] on a missing method, a disabled logger, or
    any read failure.
    """
    getter = getattr(db_logger, "get_todays_bot_decision_entries", None)
    if not callable(getter):
        return []
    try:
        raw = await getter()
    except Exception as exc:
        logger.warning("desk journal_data decisions_for failed date=%s err=%s", date, exc)
        return []
    if not isinstance(raw, dict):
        return []

    out: List[dict] = []
    for symbol, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        out.append({
            "symbol": symbol,
            "component": entry.get("strategy") or "unknown",
            "action": "entry",
            "reason": "",
            "side": entry.get("side"),
            "ts": entry.get("ts"),
            "confidence": entry.get("confidence"),
        })
    return out


async def trades_for(date_range: Tuple[Any, Any], db_logger: Any) -> List[dict]:
    """Closed `bot_trades` rows for `date_range = (start, end)` (inclusive),
    shaped for `render_journal`: each row keeps the raw `strategy` and gains
    `family` (via `family_for`) and `r` (an alias of `pnl_r`).

    Never raises: [] on a missing method, a malformed `date_range`, or any
    read failure.
    """
    getter = getattr(db_logger, "get_bot_trades_between", None)
    if not callable(getter):
        return []
    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        return []
    start, end = date_range
    try:
        raw = await getter(start, end)
    except Exception as exc:
        logger.warning("desk journal_data trades_for failed range=%s err=%s", date_range, exc)
        return []
    if not isinstance(raw, list):
        return []

    out: List[dict] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        strategy = row.get("strategy")
        merged = dict(row)
        merged["family"] = family_for(strategy) if strategy else None
        if not merged["family"]:
            merged["family"] = "unknown"
        merged["r"] = row.get("pnl_r")
        out.append(merged)
    return out


# ────────────────────────────────────────────────────────────────────
# Journal file store (mirrors core/desk/store.py's atomic-write pattern)
# ────────────────────────────────────────────────────────────────────

def journal_dir(desk_dir: str) -> str:
    """Directory holding the dated journal files."""
    return os.path.join(desk_dir or ".", JOURNAL_SUBDIR)


def journal_path(desk_dir: str, date: str) -> str:
    """Absolute-or-relative path of one session's journal file."""
    return os.path.join(journal_dir(desk_dir), f"{date}.md")


def write_journal(text: str, desk_dir: str, date: str) -> Optional[str]:
    """Write `text` atomically to `<desk_dir>/journal/<date>.md` (temp file in
    the same directory + `os.replace`); return the path written, or None on
    failure. Never raises — mirrors `core/desk/store.save_plan`.
    """
    path = journal_path(desk_dir, date)
    directory = os.path.dirname(path) or "."
    tmp_path = None
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".journal-", suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            fh.write(text if isinstance(text, str) else "")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
        return path
    except (OSError, TypeError, ValueError) as exc:
        logger.error("desk write_journal failed date=%s dir=%s err=%s", date, desk_dir, exc)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as exc:  # pragma: no cover - best-effort cleanup
                logger.warning("desk write_journal could not remove temp file %s: %s",
                               tmp_path, exc)
