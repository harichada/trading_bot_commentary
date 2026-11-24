"""API for the event-driven scalp live engine.

  GET  /api/scalp/live/status
  GET  /api/scalp/live/history?date=YYYY-MM-DD
  POST /api/scalp/live/start
  POST /api/scalp/live/stop
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

logger = logging.getLogger("TradingBot")
_REPO_ROOT = Path(__file__).resolve().parent.parent
NDJSON_PATH = _REPO_ROOT / "data" / "live_scalp_reversal.ndjson"


def register(app: FastAPI) -> None:
    # scalp_live does NOT auto-start on boot — it trades the same Alpaca account
    # as scalp_paper, so only one engine should run at a time.
    # Start manually via POST /api/scalp/live/start when scalp_paper is NOT running.

    @app.get("/api/scalp/live/status")
    async def get_live_status() -> dict:
        try:
            from core.scalp_live import get_engine
            return get_engine().get_status()
        except Exception as exc:
            raise HTTPException(500, str(exc))

    @app.post("/api/scalp/live/start")
    async def start_live() -> dict:
        try:
            from core.scalp_live import get_engine
            e = get_engine()
            if not e._running:
                e.start()
                return {"ok": True, "message": "started"}
            return {"ok": True, "message": "already running"}
        except Exception as exc:
            raise HTTPException(500, str(exc))

    @app.post("/api/scalp/live/stop")
    async def stop_live() -> dict:
        try:
            from core.scalp_live import get_engine
            get_engine().stop()
            return {"ok": True}
        except Exception as exc:
            raise HTTPException(500, str(exc))

    @app.get("/api/scalp/live/history")
    async def get_live_history(date: str = Query(default="")) -> dict:
        from core.scalp_shadow import _today_et
        target  = date or _today_et()
        records: list[dict] = []
        if NDJSON_PATH.exists():
            for line in NDJSON_PATH.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    if r.get("date") == target:
                        records.append(r)
                except json.JSONDecodeError:
                    pass
        wins      = [r for r in records if r.get("dollar_pnl", 0) > 0]
        total_pnl = sum(r.get("dollar_pnl", 0) for r in records)
        return {
            "date":         target,
            "trades":       records,
            "n":            len(records),
            "n_wins":       len(wins),
            "total_pnl":    round(total_pnl, 2),
            "win_rate_pct": round(len(wins) / len(records) * 100, 1) if records else 0.0,
        }
