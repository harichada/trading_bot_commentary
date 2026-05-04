"""Backtest API: HTTP endpoint + WebSocket progress stream.

Wires the pure async `backtest_strategies.run_backtest()` into FastAPI:

  POST /api/backtest/run        kick off a run, return {run_id} immediately
  WS   /ws/backtest/{run_id}    stream progress / complete / error events
  GET  /api/backtest/latest     return last completed report (404 if none)

Single-flight: only one run executes at a time. A second concurrent POST
returns 429 with the in-flight run_id so the caller can attach to its WS.

Persistence: every successful run is written to
`backtest_results/run_YYYYMMDD_HHMMSS.json` and copied to
`backtest_results/latest_report.json`.
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

from backtest_strategies import ProgressEvent, run_backtest

logger = logging.getLogger("TradingBot")

RESULTS_DIR = Path("backtest_results")
LATEST_FILE = RESULTS_DIR / "latest_report.json"
SYMBOL_RE = re.compile(r"^[A-Z.]{1,8}$")
ALLOWED_FREQUENCIES = {1, 5, 15}


# ---------------------------------------------------------------------------
# In-memory run state (single-flight)
# ---------------------------------------------------------------------------

class _RunState:
    """Tracks the single in-flight backtest run.

    Holds:
      - current run_id (None if idle)
      - per-subscriber asyncio.Queues so multiple WS clients can attach
      - a last-event cache per run_id so a WS that connects after the run
        finishes still sees the `complete` (or `error`) event
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._current_run_id: Optional[str] = None
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._last_event: dict[str, dict] = {}

    @property
    def current_run_id(self) -> Optional[str]:
        return self._current_run_id

    async def acquire(self, run_id: str) -> bool:
        """Try to claim the single-flight slot. Returns False if busy."""
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
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self._subscribers.setdefault(run_id, []).append(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(run_id)
        if not subs:
            return
        try:
            subs.remove(q)
        except ValueError:
            pass

    def publish(self, run_id: str, event: dict) -> None:
        for q in list(self._subscribers.get(run_id, [])):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("backtest_ws subscriber queue full, dropping event")

    def cache_last(self, run_id: str, event: dict) -> None:
        # Only cache terminal events — that's all a late subscriber needs.
        if event.get("type") in ("complete", "error"):
            self._last_event[run_id] = event
            # Trim cache to most-recent 16 runs to bound memory.
            if len(self._last_event) > 16:
                oldest = next(iter(self._last_event))
                self._last_event.pop(oldest, None)

    def replay_last(self, run_id: str) -> Optional[dict]:
        return self._last_event.get(run_id)


_state = _RunState()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_request(body: dict) -> tuple[Optional[list[str]], Optional[int], int, int]:
    """Return (symbols, top_n, days, frequency) or raise HTTPException(400)."""
    mode = body.get("symbol_mode")
    if mode not in ("top_n", "explicit"):
        raise HTTPException(400, "symbol_mode: must be 'top_n' or 'explicit'")

    days = body.get("days", 90)
    if not isinstance(days, int) or not (1 <= days <= 365):
        raise HTTPException(400, "days: must be int in [1, 365]")

    frequency = body.get("frequency", 5)
    if frequency not in ALLOWED_FREQUENCIES:
        raise HTTPException(400, f"frequency: must be one of {sorted(ALLOWED_FREQUENCIES)}")

    if mode == "top_n":
        top_n = body.get("top_n", 30)
        if not isinstance(top_n, int) or not (1 <= top_n <= 200):
            raise HTTPException(400, "top_n: must be int in [1, 200]")
        return None, top_n, days, frequency

    raw_symbols = body.get("symbols")
    if not isinstance(raw_symbols, list) or not raw_symbols:
        raise HTTPException(400, "symbols: must be a non-empty list")
    symbols = [str(s).strip().upper() for s in raw_symbols]
    bad = [s for s in symbols if not SYMBOL_RE.match(s)]
    if bad:
        raise HTTPException(400, f"symbols: invalid tickers {bad}")
    return symbols, None, days, frequency


# ---------------------------------------------------------------------------
# Run executor
# ---------------------------------------------------------------------------

def _make_run_id() -> str:
    return f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"


def _save_report(report: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"run_{report['run_id']}.json"
    payload = json.dumps(report, indent=2, default=str)
    path.write_text(payload)
    LATEST_FILE.write_text(payload)
    return path


async def _execute_run(
    run_id: str,
    *,
    symbols: Optional[list[str]],
    top_n: Optional[int],
    days: int,
    frequency: int,
) -> None:
    """Run the backtest, publish progress events, persist final report."""
    async def _on_progress(event: ProgressEvent) -> None:
        _state.publish(run_id, {
            "type": "progress",
            "run_id": run_id,
            "stage": event.stage,
            "symbols_done": event.symbols_done,
            "symbols_total": event.symbols_total,
            "current_symbol": event.current_symbol,
        })

    try:
        report = await run_backtest(
            symbols=symbols,
            top_n=top_n,
            days=days,
            frequency=frequency,
            run_id=run_id,
            progress_cb=_on_progress,
        )
        path = _save_report(report)
        complete_event = {
            "type": "complete",
            "run_id": run_id,
            "report_path": str(path),
            "report": report,
        }
        _state.cache_last(run_id, complete_event)
        _state.publish(run_id, complete_event)
        logger.info("backtest_complete run_id=%s path=%s", run_id, path)
    except Exception as e:
        logger.exception("backtest_failed run_id=%s", run_id)
        error_event = {
            "type": "error",
            "run_id": run_id,
            "message": str(e) or e.__class__.__name__,
        }
        _state.cache_last(run_id, error_event)
        _state.publish(run_id, error_event)
    finally:
        await _state.release(run_id)


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def register(app: FastAPI) -> None:
    """Attach backtest routes to the FastAPI app. Call once from routes.py."""

    @app.post("/api/backtest/run")
    async def post_run(body: dict) -> dict:
        symbols, top_n, days, frequency = _validate_request(body)

        run_id = _make_run_id()
        if not await _state.acquire(run_id):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "backtest_running",
                    "run_id": _state.current_run_id,
                },
            )

        # Fire and forget; the WS or the saved file is how callers learn the result.
        asyncio.create_task(_execute_run(
            run_id, symbols=symbols, top_n=top_n,
            days=days, frequency=frequency,
        ))
        return {"status": "started", "run_id": run_id}

    @app.get("/api/backtest/latest")
    async def get_latest() -> dict:
        if not LATEST_FILE.exists():
            raise HTTPException(404, "no backtest has been run yet")
        try:
            return json.loads(LATEST_FILE.read_text())
        except Exception as e:
            logger.error("latest_report_load_failed err=%s", e)
            raise HTTPException(500, "failed to load latest report")

    @app.websocket("/ws/backtest/{run_id}")
    async def ws_run(websocket: WebSocket, run_id: str) -> None:
        # Best-effort token check using the same scheme as the main /ws.
        # Auth is enforced at the HTTP layer for /api/* routes; WS routes
        # use query-param token. Keep parity with the existing main /ws.
        from api.auth import verify_ws_token
        token = websocket.query_params.get("token")
        if not verify_ws_token(token):
            await websocket.close(code=1008, reason="Authentication required")
            return

        await websocket.accept()
        q = _state.subscribe(run_id)

        # Replay terminal event if the run already finished.
        last = _state.replay_last(run_id)
        if last is not None:
            try:
                await websocket.send_json(last)
            except Exception:
                _state.unsubscribe(run_id, q)
                return
            if last.get("type") in ("complete", "error"):
                _state.unsubscribe(run_id, q)
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
            logger.warning("backtest_ws send_failed run_id=%s err=%s", run_id, e)
        finally:
            _state.unsubscribe(run_id, q)
            try:
                await websocket.close()
            except Exception:
                pass
