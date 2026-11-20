"""Alpaca paper trading API for the scalp reversal strategy.

  GET  /api/scalp/paper/status    live summary + open positions
  GET  /api/scalp/paper/history   completed trades for a given date
  POST /api/scalp/paper/start     start the engine (idempotent)
  POST /api/scalp/paper/stop      stop the engine

Registered via register(app) in api/routes.py at startup.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

logger = logging.getLogger("TradingBot")

_REPO_ROOT = Path(__file__).resolve().parent.parent
NDJSON_PATH = _REPO_ROOT / "data" / "paper_scalp_reversal.ndjson"


def register(app: FastAPI) -> None:

    @app.on_event("startup")
    async def _start_paper() -> None:
        try:
            from core.scalp_paper import get_engine
            engine = get_engine()
            engine.start()
            logger.info("scalp_paper_api: paper engine started")
        except Exception as exc:
            logger.warning("scalp_paper_api: paper start failed: %s", exc)

    @app.get("/api/scalp/paper/status")
    async def get_paper_status() -> dict:
        try:
            from core.scalp_paper import get_engine
            return get_engine().get_status()
        except Exception as exc:
            logger.error("paper_status failed: %s", exc)
            raise HTTPException(500, f"paper status unavailable: {exc}")

    @app.post("/api/scalp/paper/start")
    async def start_paper() -> dict:
        try:
            from core.scalp_paper import get_engine
            engine = get_engine()
            if engine._task is None or engine._task.done():
                engine.start()
                return {"ok": True, "message": "paper engine started"}
            return {"ok": True, "message": "already running"}
        except Exception as exc:
            logger.error("paper_start failed: %s", exc)
            raise HTTPException(500, str(exc))

    @app.post("/api/scalp/paper/stop")
    async def stop_paper() -> dict:
        try:
            from core.scalp_paper import get_engine
            get_engine().stop()
            return {"ok": True, "message": "paper engine stopped"}
        except Exception as exc:
            logger.error("paper_stop failed: %s", exc)
            raise HTTPException(500, str(exc))

    @app.get("/api/scalp/paper/history")
    async def get_paper_history(date: str = Query(default="")) -> dict:
        from core.scalp_shadow import _today_et
        target = date or _today_et()
        records: list[dict] = []
        if NDJSON_PATH.exists():
            try:
                with NDJSON_PATH.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            if rec.get("date") == target:
                                records.append(rec)
                        except json.JSONDecodeError:
                            pass
            except Exception as exc:
                logger.error("paper_history read failed: %s", exc)
                raise HTTPException(500, "failed to read paper log")
        wins = [r for r in records if r.get("dollar_pnl", 0) > 0]
        total_pnl = sum(r.get("dollar_pnl", 0) for r in records)
        return {
            "date": target,
            "trades": records,
            "n": len(records),
            "n_wins": len(wins),
            "total_pnl": round(total_pnl, 2),
            "win_rate_pct": round(len(wins) / len(records) * 100, 1) if records else 0.0,
        }
