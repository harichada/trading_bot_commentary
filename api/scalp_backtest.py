"""Scalp reversal backtest API: HTTP endpoint + WebSocket progress stream.

  POST /api/backtest/scalp/run       kick off a scalp run, return {run_id}
  WS   /ws/backtest/scalp/{run_id}   stream progress / complete / error
  GET  /api/backtest/scalp/latest    return last completed scalp report

Mirrors api/backtest.py structure. scalp_reversal.run() is synchronous
so it runs in asyncio.to_thread() to avoid blocking the event loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from backtest.scalp_reversal import (
    CAPITAL_USD,
    DEFAULT_SYMBOLS,
    MAX_SLOTS,
    ScalpParams,
    run as scalp_run,
)

logger = logging.getLogger("TradingBot")

RESULTS_DIR = Path("backtest_results")
SCALP_LATEST_FILE = RESULTS_DIR / "scalp_latest_report.json"
SYMBOL_RE = re.compile(r"^[A-Z.]{1,8}$")


# ---------------------------------------------------------------------------
# In-memory run state (single-flight, same pattern as api/backtest.py)
# ---------------------------------------------------------------------------

class _ScalpRunState:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._current_run_id: Optional[str] = None
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._last_event: dict[str, dict] = {}

    @property
    def current_run_id(self) -> Optional[str]:
        return self._current_run_id

    async def acquire(self, run_id: str) -> bool:
        async with self._lock:
            if self._current_run_id is not None:
                return False
            self._current_run_id = run_id
            self._subscribers[run_id] = []
            return True

    async def release(self, run_id: str) -> None:
        async with self._lock:
            if self._current_run_id == run_id:
                self._current_run_id = None

    def subscribe(self, run_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        self._subscribers.setdefault(run_id, []).append(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(run_id)
        if subs:
            try:
                subs.remove(q)
            except ValueError:
                pass

    def publish(self, run_id: str, event: dict) -> None:
        for q in list(self._subscribers.get(run_id, [])):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("scalp_bt_ws queue full run_id=%s", run_id)

    def cache_last(self, run_id: str, event: dict) -> None:
        if event.get("type") in ("complete", "error"):
            self._last_event[run_id] = event
            if len(self._last_event) > 8:
                oldest = next(iter(self._last_event))
                self._last_event.pop(oldest, None)

    def replay_last(self, run_id: str) -> Optional[dict]:
        return self._last_event.get(run_id)


_scalp_state = _ScalpRunState()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_scalp_request(body: dict) -> tuple[list[str], int, ScalpParams, int, int]:
    """Return (symbols, days, params, max_slots, capital_usd) or raise 400."""
    raw_symbols = body.get("symbols", DEFAULT_SYMBOLS)
    if not isinstance(raw_symbols, list) or not raw_symbols:
        raise HTTPException(400, "symbols: must be a non-empty list")
    symbols = [str(s).strip().upper() for s in raw_symbols]
    bad = [s for s in symbols if not SYMBOL_RE.match(s)]
    if bad:
        raise HTTPException(400, f"symbols: invalid tickers {bad}")

    days = body.get("days", 90)
    if not isinstance(days, int) or not (1 <= days <= 365):
        raise HTTPException(400, "days: must be int in [1, 365]")

    max_slots = body.get("max_slots", MAX_SLOTS)
    if not isinstance(max_slots, int) or not (1 <= max_slots <= 50):
        raise HTTPException(400, "max_slots: must be int in [1, 50]")

    capital_usd = body.get("capital_usd", CAPITAL_USD)
    if not isinstance(capital_usd, int) or capital_usd < 1000:
        raise HTTPException(400, "capital_usd: must be int >= 1000")

    def _flt(key: str, default: float, lo: float, hi: float) -> float:
        v = body.get(key, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise HTTPException(400, f"{key}: must be a number")
        if not (lo <= v <= hi):
            raise HTTPException(400, f"{key}: must be in [{lo}, {hi}]")
        return v

    def _int(key: str, default: int, lo: int, hi: int) -> int:
        v = body.get(key, default)
        if not isinstance(v, int) or not (lo <= v <= hi):
            raise HTTPException(400, f"{key}: must be int in [{lo}, {hi}]")
        return v

    end_date = body.get("end_date") or None
    if end_date is not None:
        import re as _re
        if not _re.match(r"^\d{4}-\d{2}-\d{2}$", str(end_date)):
            raise HTTPException(400, "end_date: must be YYYY-MM-DD")

    params = ScalpParams(
        rsi_long_thresh=_flt("rsi_long_thresh", 30.0, 5.0, 50.0),
        rsi_short_thresh=_flt("rsi_short_thresh", 75.0, 50.0, 95.0),
        vol_min=_flt("vol_min", 2.0, 0.5, 20.0),
        atr_stop_mult=_flt("atr_stop_mult", 1.0, 0.1, 10.0),
        atr_target_mult=_flt("atr_target_mult", 2.0, 0.1, 20.0),
        quick_exit_bars=_int("quick_exit_bars", 3, 1, 30),
        min_profit_pct=_flt("min_profit_pct", 0.15, 0.01, 5.0),
        max_hold_bars=_int("max_hold_bars", 15, 1, 120),
        adx_max=_flt("adx_max", 999.0, 10.0, 999.0),
        require_reversal_candle=bool(body.get("require_reversal_candle", False)),
        candle_wick_ratio=_flt("candle_wick_ratio", 1.5, 0.5, 10.0),
        entry_confirm=bool(body.get("entry_confirm", False)),
        rsi_turn=bool(body.get("rsi_turn", False)),
        sides=body.get("sides", "both") if body.get("sides") in ("both", "long_only", "short_only") else "both",
        from_open_drop_max=_flt("from_open_drop_max", 0.0, 0.0, 20.0),
    )
    return symbols, days, params, max_slots, capital_usd, end_date


# ---------------------------------------------------------------------------
# Run executor
# ---------------------------------------------------------------------------

def _make_run_id() -> str:
    return f"scalp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"


def _save_report(report: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"scalp_run_{report['run_id']}.json"
    payload = json.dumps(report, indent=2, default=str)
    path.write_text(payload)
    SCALP_LATEST_FILE.write_text(payload)
    return path


async def _execute_scalp_run(
    run_id: str,
    *,
    symbols: list[str],
    days: int,
    params: ScalpParams,
    max_slots: int,
    capital_usd: int,
    dsn: str,
    end_date: str | None = None,
) -> None:
    _scalp_state.publish(run_id, {
        "type": "progress",
        "run_id": run_id,
        "message": f"Running scalp reversal on {len(symbols)} symbol(s)…",
        "symbols_total": len(symbols),
        "symbols_done": 0,
    })
    try:
        report = await asyncio.to_thread(
            scalp_run,
            symbols=symbols,
            days=days,
            params=params,
            dsn=dsn,
            max_slots=max_slots,
            capital_usd=capital_usd,
            end_date=end_date,
        )
        report["run_id"] = run_id
        path = _save_report(report)
        complete_event = {
            "type": "complete",
            "run_id": run_id,
            "report_path": str(path),
            "report": report,
        }
        _scalp_state.cache_last(run_id, complete_event)
        _scalp_state.publish(run_id, complete_event)
        logger.info("scalp_bt_complete run_id=%s path=%s", run_id, path)
    except Exception as e:
        logger.exception("scalp_bt_failed run_id=%s", run_id)
        error_event = {
            "type": "error",
            "run_id": run_id,
            "message": str(e) or e.__class__.__name__,
        }
        _scalp_state.cache_last(run_id, error_event)
        _scalp_state.publish(run_id, error_event)
    finally:
        await _scalp_state.release(run_id)


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def register(app: FastAPI, dsn: Optional[str] = None) -> None:
    """Attach scalp backtest routes to the FastAPI app."""
    import os
    from backtest.scalp_reversal import DEFAULT_DSN
    effective_dsn = dsn or os.environ.get("POSTGRES_DSN", DEFAULT_DSN)

    @app.post("/api/backtest/scalp/run")
    async def post_scalp_run(body: dict) -> dict:
        symbols, days, params, max_slots, capital_usd, end_date = _validate_scalp_request(body)
        run_id = _make_run_id()
        if not await _scalp_state.acquire(run_id):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "scalp_backtest_running",
                    "run_id": _scalp_state.current_run_id,
                },
            )
        asyncio.create_task(_execute_scalp_run(
            run_id,
            symbols=symbols,
            days=days,
            params=params,
            max_slots=max_slots,
            capital_usd=capital_usd,
            dsn=effective_dsn,
            end_date=end_date,
        ))
        return {"status": "started", "run_id": run_id}

    @app.get("/api/backtest/scalp/latest")
    async def get_scalp_latest() -> dict:
        if not SCALP_LATEST_FILE.exists():
            raise HTTPException(404, "no scalp backtest has been run yet")
        try:
            return json.loads(SCALP_LATEST_FILE.read_text())
        except Exception as e:
            logger.error("scalp_latest_load_failed err=%s", e)
            raise HTTPException(500, "failed to load latest scalp report")

    @app.websocket("/ws/backtest/scalp/{run_id}")
    async def ws_scalp_run(websocket: WebSocket, run_id: str) -> None:
        from api.auth import verify_ws_token
        token = websocket.query_params.get("token")
        if not verify_ws_token(token):
            await websocket.close(code=1008, reason="Authentication required")
            return

        await websocket.accept()
        q = _scalp_state.subscribe(run_id)

        last = _scalp_state.replay_last(run_id)
        if last is not None:
            try:
                await websocket.send_json(last)
            except Exception:
                _scalp_state.unsubscribe(run_id, q)
                return
            if last.get("type") in ("complete", "error"):
                _scalp_state.unsubscribe(run_id, q)
                await websocket.close()
                return

        try:
            while True:
                event = await q.get()
                await websocket.send_json(event)
                if event.get("type") in ("complete", "error"):
                    break
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning("scalp_bt_ws_send_failed run_id=%s err=%s", run_id, e)
        finally:
            _scalp_state.unsubscribe(run_id, q)
            try:
                await websocket.close()
            except Exception:
                pass
