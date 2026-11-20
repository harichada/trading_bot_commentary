"""Scalp reversal shadow API.

  GET  /api/scalp/shadow/status      live summary + open positions
  GET  /api/scalp/shadow/history     all ndjson records for a given date

Registered via register(app) in api/routes.py at startup.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

logger = logging.getLogger("TradingBot")

_REPO_ROOT = Path(__file__).resolve().parent.parent
NDJSON_PATH = _REPO_ROOT / "data" / "shadow_scalp_reversal.ndjson"


def register(app: FastAPI) -> None:

    @app.on_event("startup")
    async def _start_shadow() -> None:
        try:
            from core.scalp_shadow import get_engine
            engine = get_engine()
            engine.start()
            logger.info("scalp_shadow_api: shadow engine started")
        except Exception as exc:
            logger.warning("scalp_shadow_api: shadow start failed: %s", exc)

    @app.get("/api/bar_cache_stats")
    async def get_bar_cache_stats() -> dict:
        """Return live SchwabBarCache state: symbols, bar counts, last bar time."""
        from core.schwab_bar_cache import SchwabBarCache
        from datetime import timezone
        cache = SchwabBarCache.instance()
        syms = cache.symbols()
        result = {"total_symbols": len(syms), "symbols": {}}
        for sym in sorted(syms):
            df = cache.get_today_df(sym)
            if df.empty:
                result["symbols"][sym] = {"bars": 0, "last_bar_et": None}
            else:
                last_et = df.index[-1].tz_convert("America/New_York")
                result["symbols"][sym] = {
                    "bars": len(df),
                    "last_bar_et": last_et.strftime("%H:%M"),
                    "last_close": round(float(df["Close"].iloc[-1]), 2),
                }
        return result

    @app.get("/api/scalp/shadow/status")
    async def get_shadow_status() -> dict:
        try:
            from core.scalp_shadow import get_engine
            return get_engine().get_status()
        except Exception as exc:
            logger.error("shadow_status failed: %s", exc)
            raise HTTPException(500, f"shadow status unavailable: {exc}")

    @app.patch("/api/scalp/shadow/config")
    async def patch_shadow_config(body: dict) -> dict:
        """Update shadow engine params at runtime. Any subset of ScalpParams fields accepted."""
        from dataclasses import replace, fields
        from backtest.scalp_reversal import ScalpParams
        try:
            from core.scalp_shadow import get_engine
            engine = get_engine()
            p = engine.params
            updates: dict = {}
            field_map = {f.name: f for f in fields(ScalpParams)}
            for key, val in body.items():
                if key not in field_map:
                    continue
                ft = field_map[key].type
                # coerce type
                if ft in ("bool", bool) or ft == "bool":
                    updates[key] = bool(val)
                elif ft in ("float", float):
                    updates[key] = float(val)
                elif ft in ("int", int):
                    updates[key] = int(val)
                elif ft in ("str", str):
                    updates[key] = str(val)
                else:
                    try:
                        updates[key] = float(val)
                    except (TypeError, ValueError):
                        updates[key] = val
            if "sides" in updates and updates["sides"] not in ("both", "long_only", "short_only"):
                raise ValueError(f"Invalid sides: {updates['sides']!r}")
            # Handle max_slots separately — not a ScalpParams field
            if "max_slots" in body:
                try:
                    ms = int(body["max_slots"])
                    if 1 <= ms <= 50:
                        engine.max_slots = ms
                        logger.info("scalp_shadow max_slots updated: %d", ms)
                except (TypeError, ValueError):
                    pass
            # Handle slot_size separately — not a ScalpParams field
            if "slot_size" in body:
                try:
                    sz = int(body["slot_size"])
                    if sz >= 1000:
                        engine.slot_size = sz
                        logger.info("scalp_shadow slot_size updated: %d", sz)
                except (TypeError, ValueError):
                    pass
            # Handle symbols separately — it's not a ScalpParams field
            if "symbols" in body and isinstance(body["symbols"], list):
                syms = [str(s).strip().upper() for s in body["symbols"] if str(s).strip()]
                if syms:
                    engine.symbols = syms
                    logger.info("scalp_shadow symbols updated: %s", syms)
            if updates:
                engine.params = replace(p, **updates)
                logger.info("scalp_shadow config updated: %s", updates)
            from dataclasses import asdict
            return {"ok": True, "params": asdict(engine.params), "symbols": engine.symbols, "slot_size": engine.slot_size, "max_slots": engine.max_slots}
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:
            logger.error("shadow_config patch failed: %s", exc)
            raise HTTPException(500, str(exc))

    @app.get("/api/scalp/shadow/history")
    async def get_shadow_history(date: str = Query(default="")) -> dict:
        """Return all ndjson records for a given date (default: today)."""
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
                logger.error("shadow_history read failed: %s", exc)
                raise HTTPException(500, "failed to read shadow log")
        return {"date": target, "trades": records, "n": len(records)}
