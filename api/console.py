"""Risk Console API: GET/PUT knob snapshot, block switch, pending, audit.

Thin FastAPI router over `core.settings_console` (the pure knob registry)
and `core.settings_console_store` (submit_change / list_pending /
cancel_pending / read_audit / set_block_new_entries). The app-level
`APIAuthMiddleware` (api/routes.py) already enforces the bearer key on
every /api/* route, so this router adds no auth of its own.

`actor` is always the literal "dashboard" — the bearer key identifies the
operator but is never logged or echoed here.

Reads never raise: any failure degrades to `{"ok": false, "error": ...}`
with HTTP 200. Writes use the status codes from the Task 4 brief: 200 for
applied/pending/noop, 400 for a validation (or block-confirm) rejection,
404 for an unknown key/pending id, 409 for env_shadowed/restart_required.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core.settings_console import (ALLOCATION_CEILING_PATH,
                                    DEFAULT_ALLOCATION_CEILING, KNOBS,
                                    effective_value, env_shadowed, yaml_value)
from core.settings_console_store import (cancel_pending, list_pending,
                                          read_audit, set_block_new_entries,
                                          submit_change)

logger = logging.getLogger("TradingBot")

router = APIRouter(prefix="/api/console", tags=["console"])

_ACTOR = "dashboard"


def _manager() -> Any:
    from core.config import config_manager
    return config_manager


def _cfg() -> Any:
    from core.config import Config
    return Config()


def _now():
    from core.loops.desk_steps import now_et
    return now_et()


def _engine_audit(event: str, key: str, **fields: Any) -> None:
    """Best-effort mirror of a console write into the engine's own audit
    trail. Imported lazily to avoid a circular import with api.routes."""
    try:
        import api.routes as routes
        if routes.trading_engine is not None:
            routes.trading_engine._audit("settings_console", None, event, key, **fields)
    except Exception:
        logger.exception("settings_console: engine audit failed for %s/%s", event, key)


def _knob_row(knob, manager: Any, cfg: Any, pending_by_key: dict) -> dict:
    pending = pending_by_key.get(knob.key)
    pending_out = None
    if pending is not None:
        pending_out = {"id": pending.get("id"), "new": pending.get("new"),
                        "effective_at": pending.get("effective_at")}
    value = effective_value(cfg, knob)
    if value is None:
        value = yaml_value(manager, knob)
    return {
        "key": knob.key, "label": knob.label, "group": knob.group,
        "kind": knob.kind, "value": value, "yaml_value": yaml_value(manager, knob),
        "env_shadowed": env_shadowed(knob), "restart_required": knob.restart_required,
        "minimum": knob.minimum, "maximum": knob.maximum,
        "allowed": list(knob.allowed), "pending": pending_out,
    }


_REASON_MAX = 500


def _reason(body: dict) -> str:
    """Operator-supplied reason, coerced to a bounded string so the audit
    file cannot be grown without limit from the dashboard."""
    return str(body.get("reason", "") or "")[:_REASON_MAX]


def _rejection_status(reason: str) -> int:
    if reason == "unknown setting":
        return 404
    if reason.startswith("env_shadowed") or reason == "restart_required":
        return 409
    return 400


@router.get("")
def get_console() -> Any:
    try:
        manager, cfg = _manager(), _cfg()
        desk_dir = cfg.DESK_DATA_DIR
        pending_rows = list_pending(desk_dir)
        pending_by_key = {row.get("key"): row for row in pending_rows}
        knobs = [_knob_row(knob, manager, cfg, pending_by_key) for knob in KNOBS]
        ceiling = manager.get(ALLOCATION_CEILING_PATH, DEFAULT_ALLOCATION_CEILING)
        return {
            "ok": True, "block_new_entries": bool(cfg.BLOCK_NEW_ENTRIES),
            "allocation_ceiling": float(ceiling),
            "promote_time_et": cfg.SETTINGS_PROMOTE_TIME_ET,
            # Pending loosens are promoted by the desk loop; without it they
            # would sit "pending" forever (final review I1).
            "promotion_enabled": bool(getattr(cfg, "ENABLE_DESK", False)),
            "knobs": knobs, "pending": pending_rows,
            "audit": read_audit(desk_dir, limit=20),
        }
    except Exception as exc:
        logger.exception("settings_console: snapshot failed")
        return {"ok": False, "error": str(exc)}


@router.put("/{key}")
def put_console(key: str, body: dict) -> Any:
    manager, cfg = _manager(), _cfg()
    desk_dir = cfg.DESK_DATA_DIR
    reason = _reason(body)
    result = submit_change(manager, cfg, desk_dir, key, body.get("value"),
                            now=_now(), actor=_ACTOR, reason=reason)
    result_dict = asdict(result)
    if result.status != "rejected":
        _engine_audit(result.status, key, direction=result.direction)
        return {"ok": True, "result": result_dict}
    _engine_audit("rejected", key, reason=result.reason)
    status_code = _rejection_status(result.reason)
    return JSONResponse(status_code=status_code,
                         content={"ok": False, "error": result.reason, "result": result_dict})


@router.post("/block-entries")
def post_block_entries(body: dict) -> Any:
    manager, cfg = _manager(), _cfg()
    result = set_block_new_entries(
        manager, cfg.DESK_DATA_DIR, bool(body.get("block")), now=_now(), actor=_ACTOR,
        reason=_reason(body), confirm=body.get("confirm", ""),
    )
    result_dict = asdict(result)
    event = "block_entries" if result.status == "applied" else "rejected"
    _engine_audit(event, "block_new_entries", reason=result.reason)
    if result.status == "applied":
        return {"ok": True, "result": result_dict}
    return JSONResponse(status_code=400,
                         content={"ok": False, "error": result.reason, "result": result_dict})


@router.delete("/pending/{pending_id}")
def delete_pending(pending_id: str) -> Any:
    cfg = _cfg()
    cancelled = cancel_pending(cfg.DESK_DATA_DIR, pending_id, now=_now(), actor=_ACTOR)
    if cancelled:
        _engine_audit("cancelled", pending_id)
        return {"ok": True}
    return JSONResponse(status_code=404, content={"ok": False, "error": "unknown pending id"})


@router.get("/audit")
def get_audit(limit: int = 50) -> Any:
    try:
        cfg = _cfg()
        return {"ok": True, "rows": read_audit(cfg.DESK_DATA_DIR, limit=limit)}
    except Exception as exc:
        logger.exception("settings_console: audit read failed")
        return {"ok": False, "error": str(exc)}
