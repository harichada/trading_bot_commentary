"""v-algorithms-page-2026-10-06: `/api/algorithms` — every registered strategy
(live or shadow) with its latest decisions, for the dashboard Algorithms tab.

Read-only. Never 500: failures degrade to `{ok: false, error}` with HTTP 200.
The bearer middleware in api/routes.py covers these paths.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

from strategies.decision_book import DecisionBook

logger = logging.getLogger("TradingBot")
router = APIRouter(prefix="/api/algorithms", tags=["algorithms"])

# How each strategy's LIVE flag is decided. Fit lanes are handled separately.
_LIVE_FLAGS = {
    "day_trade_momentum": "DAY_TRADE_LIVE_ENTRIES_ENABLED",
    "day_trade_momentum_short": "DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED",
    "orb_contraction_rvol": "ORB_LIVE_ENTRIES_ENABLED",
    "mean_reversion": "MEAN_REV_LIVE_ENTRIES_ENABLED",
}
_LEGACY_ALWAYS_LIVE = ("momentum", "breakout", "news")


def _engine():
    import api.routes as routes
    return routes.trading_engine


def _cfg():
    from core.config import Config
    return Config()


def lane_status(name: str, cfg: Any, mode: str) -> Dict[str, Any]:
    """live | shadow, plus the flag that decides it. Pure."""
    try:
        if name.startswith("fit:"):
            fam = name.split(":", 1)[1]
            live = fam in tuple(cfg.FIT_LANE_LIVE_ENTRIES_ENABLED)
            return {"status": "live" if live else "shadow", "flag": "FIT_LANE_LIVE_ENTRIES_ENABLED", "family": fam,
                    "kind": "fit_lane"}
        flag = _LIVE_FLAGS.get(name)
        if flag:
            return {"status": "live" if bool(getattr(cfg, flag)) else "shadow", "flag": flag, "family": name,
                    "kind": "builtin"}
        if name in _LEGACY_ALWAYS_LIVE:
            return {"status": "live" if str(mode).lower().startswith("live") else "shadow", "flag": "engine mode",
                    "family": name, "kind": "legacy"}
    except Exception:
        pass
    return {"status": "unknown", "flag": "", "family": name, "kind": "unknown"}


def _strategy_rows(engine: Any, cfg: Any, limit: int) -> List[dict]:
    book = DecisionBook.instance()
    mode = str(getattr(getattr(engine, "mode", None), "value", getattr(engine, "mode", "")) or "")
    seen = set()
    rows: List[dict] = []
    for strat in list(getattr(engine, "strategies", []) or []):
        name = str(getattr(strat, "name", type(strat).__name__))
        seen.add(name)
        rows.append({"name": name, "class": type(strat).__name__, **lane_status(name, cfg, mode),
                     **book.counts(name), "recent": book.recent(name, limit)})
    for name in book.strategies():        # decisions from strategies the engine no longer lists
        if name not in seen:
            rows.append({"name": name, "class": "", **lane_status(name, cfg, mode), **book.counts(name),
                         "recent": book.recent(name, limit)})
    order = {"live": 0, "shadow": 1, "unknown": 2}
    rows.sort(key=lambda r: (order.get(r["status"], 3), r["name"]))
    return rows


@router.get("")
def get_algorithms(limit: int = 15) -> Any:
    try:
        engine = _engine()
        cfg = _cfg()
        mode = str(getattr(getattr(engine, "mode", None), "value", getattr(engine, "mode", "")) or "")
        rows = _strategy_rows(engine, cfg, max(1, min(int(limit), 100)))
        return {"ok": True, "mode": mode, "engine_running": bool(getattr(engine, "is_running", False)),
                "block_new_entries": bool(getattr(cfg, "BLOCK_NEW_ENTRIES", False)),
                "n_live": sum(1 for r in rows if r["status"] == "live"),
                "n_shadow": sum(1 for r in rows if r["status"] == "shadow"), "algorithms": rows}
    except Exception as exc:
        logger.exception("algorithms: snapshot failed")
        return {"ok": False, "error": str(exc), "algorithms": []}


@router.get("/{name}")
def get_algorithm(name: str, limit: int = 100) -> Any:
    try:
        engine = _engine()
        cfg = _cfg()
        book = DecisionBook.instance()
        mode = str(getattr(getattr(engine, "mode", None), "value", getattr(engine, "mode", "")) or "")
        return {"ok": True, "name": name, **lane_status(name, cfg, mode), **book.counts(name),
                "recent": book.recent(name, max(1, min(int(limit), 300)))}
    except Exception as exc:
        logger.exception("algorithms: detail failed")
        return {"ok": False, "error": str(exc), "name": name, "recent": []}
