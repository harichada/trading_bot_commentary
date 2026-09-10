"""FastAPI routes and WebSocket endpoints for the trading bot."""

import asyncio
import json
import logging
import os
import sys
import signal
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import secrets as _secrets
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from websockets.exceptions import ConnectionClosedError
import uvicorn

from core.models import TradingMode, CommentaryType, SignalType
from core.config import Config, config, logger
from core.commentary import TradingCommentary, CommentarySystem
from core.websocket_manager import ConnectionManager
from api.auth import verify_api_key, verify_ws_token

logger = logging.getLogger('TradingBot')

# ============================================================================
# FASTAPI APPLICATION WITH COMMENTARY
# ============================================================================

app = FastAPI(title="Trading Bot with Commentary API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:9000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:9000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class APIAuthMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces Bearer token auth on /api/* routes.

    Public routes (GET /, WebSocket /ws, /docs, /openapi.json) are excluded.
    When TRADING_API_KEY is unset, all requests are allowed (dev mode).
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        # Skip auth for public routes
        if path == "/" or path.startswith("/docs") or path.startswith("/openapi") or path.startswith("/redoc"):
            return await call_next(request)
        # Only protect /api/* routes
        if path.startswith("/api/"):
            api_key = os.getenv("TRADING_API_KEY")
            if api_key:
                auth_header = request.headers.get("authorization", "")
                if not auth_header.startswith("Bearer "):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Missing authorization header"},
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                token = auth_header[7:]
                if not _secrets.compare_digest(token, api_key):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid API key"},
                        headers={"WWW-Authenticate": "Bearer"},
                    )
        return await call_next(request)


app.add_middleware(APIAuthMiddleware)

# Register backtest endpoints (POST /api/backtest/run, WS /ws/backtest/{run_id},
# GET /api/backtest/latest). Implementation in api/backtest.py.
from api.backtest import register as _register_backtest
_register_backtest(app)

# Global instances
trading_engine = None
# v-shutdown-stops-engine-thread-2026-06-09: keep a handle to the
# engine thread so shutdown_handler can stop and join it. Without
# this, SIGINT exited only the main thread; the non-daemon engine
# thread kept the Schwab stream, token refreshes, and state writes
# alive as a zombie process, and the next bot instance fought it
# (token rotation race + concurrent state-file reads) — observed
# 2026-06-09 as the "engine init hang" on mid-session restarts.
trading_thread = None
connection_manager = ConnectionManager()


def _is_rth_now() -> bool:
    """True during regular trading hours (09:30–16:00 ET, weekdays).

    v-daypnl-accuracy-2026-06-11: stream-mark drift on top of the
    Schwab REST baseline is only trustworthy when the stream is dense.
    Pre/post-market ticks are sparse and lag the REST mark — applying
    drift there skewed day-P&L ±$25/position on 2026-06-11 08:20."""
    from zoneinfo import ZoneInfo
    et = datetime.now(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return False
    minutes = et.hour * 60 + et.minute
    return 9 * 60 + 30 <= minutes < 16 * 60

# Load dashboard HTML from template file
import os as _os
_template_dir = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), 'templates')
try:
    with open(_os.path.join(_template_dir, 'dashboard.html'), 'r') as _f:
        DASHBOARD_HTML_WITH_COMMENTARY = _f.read()
except FileNotFoundError:
    DASHBOARD_HTML_WITH_COMMENTARY = "<html><body><h1>Dashboard template not found</h1></body></html>"


@app.post("/api/toggle-mode")
async def toggle_trading_mode(request: dict):
    global trading_engine

    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}

    new_mode = request.get('mode', 'simulation')
    # v-mode-toggle-stale-state-2026-05-08: explicit `clear_inactive` flag
    # lets the operator opt into clearing the position collection that
    # belongs to the OPPOSITE mode of where they're going. Default False
    # (preserves prior behavior). Stale sim positions persisting across
    # toggles caused real bugs (leftover INTC sim blocked live QCOM via
    # correlation guard before mode-isolation fix). Surface the inactive
    # collection size in the response either way so the operator can see.
    clear_inactive = bool(request.get('clear_inactive', False))

    if new_mode == 'live':
        trading_engine.mode = TradingMode.LIVE
        # Add safety check
        if not trading_engine.schwab_client:
            return {"status": "error", "message": "Cannot switch to live mode - Schwab not connected"}

        # Sync external positions when switching to live mode
        await trading_engine._update_real_positions()

        # Inactive collection for live mode = simulated_positions
        sim_count = len(getattr(trading_engine, 'simulated_positions', {}) or {})
        sim_symbols = list((getattr(trading_engine, 'simulated_positions', {}) or {}).keys())
        cleared = 0
        if clear_inactive and sim_count > 0:
            try:
                trading_engine.simulated_positions.clear()
                cleared = sim_count
            except Exception as exc:
                logger.warning("toggle_mode: failed to clear sim positions: %s", exc)

    else:
        trading_engine.mode = TradingMode.SIMULATION_WITH_COMMENTARY
        sim_count = 0
        sim_symbols = []
        # Inactive collection for sim mode = positions (live)
        # Do NOT auto-clear live positions on a sim toggle — those are
        # real money on Schwab; clearing the bot's tracking is a no-op
        # for the broker but loses bot-side exit management. Operator
        # must use a separate explicit endpoint if they truly want to
        # forget about live positions.
        cleared = 0

    trading_engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.DECISION,
        symbol=None,
        title=f"\U0001f504 Mode Changed",
        message=(
            f"Switched to {'LIVE TRADING' if new_mode == 'live' else 'SIMULATION'} mode" +
            (f". Cleared {cleared} stale sim positions ({', '.join(sim_symbols)})."
             if cleared > 0 else
             (f". {sim_count} sim positions remain in tracking ({', '.join(sim_symbols)}). "
              "Pass clear_inactive=true to drop them."
              if sim_count > 0 and new_mode == 'live' else ""))
        ),
        importance=10
    ))
    if hasattr(trading_engine, '_audit'):
        trading_engine._audit(
            "mode_toggle", None, "changed",
            f"to_{new_mode}",
            sim_remaining=sim_count, cleared=cleared,
        )

    return {"status": "success", "mode": trading_engine.mode.value}

@app.post("/api/toggle-confirmations")
async def toggle_confirmations(request: dict):
    """Toggle close confirmation requirement"""
    global trading_engine

    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}

    enabled = request.get('enabled', True)
    trading_engine.require_confirmations = enabled

    # Add commentary about the change
    trading_engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.DECISION,
        symbol=None,
        title=f"\u2699\ufe0f Confirmation Settings Changed",
        message=f"Close confirmations {'enabled' if enabled else 'disabled'}",
        importance=7
    ))

    return {"status": "success", "enabled": enabled}

@app.get("/")
async def get_dashboard():
    # SECURITY NOTE: API key is injected into the page for single-user localhost use.
    # This is NOT safe for internet-facing deployments. For multi-user or remote access,
    # replace with a proper login flow (e.g. session cookie from POST /api/login).
    api_key = os.getenv("TRADING_API_KEY", "")
    auth_script = f'<script>window.TRADING_API_KEY={json.dumps(api_key)};</script>'

    # v-dashboard-hotreload-2026-05-28: re-read dashboard.html from
    # disk on every request instead of using the cached
    # DASHBOARD_HTML_WITH_COMMENTARY captured at module import time.
    #
    # Why: HTML/CSS changes are common during development and operator
    # tuning. The cached read meant every edit required a full bot
    # restart to land in the browser — exactly the trap that hid the
    # centering/restyle work today (operator hard-refreshing repeatedly
    # while the bot kept serving the morning-startup snapshot).
    #
    # Cost: one ~250KB file read per dashboard page load. Negligible
    # on local SSD. If this ever becomes a real perf concern in a
    # multi-user deployment, add an mtime-based cache. For single-user
    # localhost, always-fresh wins.
    try:
        with open(_os.path.join(_template_dir, 'dashboard.html'), 'r') as _f:
            template_html = _f.read()
    except FileNotFoundError:
        template_html = DASHBOARD_HTML_WITH_COMMENTARY  # fallback to import-time copy

    html = template_html.replace("</head>", f"{auth_script}</head>", 1)
    return HTMLResponse(content=html)


@app.get("/stream")
async def get_stream_view():
    """v-stream-monitor-2026-05-01: dedicated tick-level monitor at /stream.

    Pure live-price screen — one row per subscribed symbol with sub-second
    updates from the same WebSocket the dashboard uses. Diagnostic-first:
    "is the stream actually flowing?" answered at a glance.

    Renders rows via DOM API (no innerHTML for dynamic content) so tainted
    symbol strings cannot inject markup, even though upstream PriceBook
    constrains symbols to [A-Z0-9]{1,8}.
    """
    api_key = os.getenv("TRADING_API_KEY", "")
    api_json = json.dumps(api_key)
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stream Monitor</title>
<script>window.TRADING_API_KEY=__APIKEY__;</script>
<style>
  :root {
    --bg: oklch(0.10 0.012 270); --panel: oklch(0.13 0.014 270);
    --border: oklch(0.22 0.02 270); --fg: oklch(0.94 0.01 270);
    --muted: oklch(0.56 0.02 270); --green: oklch(0.72 0.20 155);
    --red: oklch(0.65 0.22 25); --amber: oklch(0.78 0.16 80);
    --mono: ui-monospace, "JetBrains Mono", "SF Mono", Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--fg); font-family: var(--sans); padding: 1rem; }
  header { display: flex; justify-content: space-between; align-items: center;
           padding: 0.75rem 1rem; background: var(--panel); border: 1px solid var(--border);
           border-radius: 8px; margin-bottom: 1rem; }
  header h1 { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.2rem; }
  header .meta { display: flex; gap: 1.5rem; font: 500 0.8rem var(--mono); color: var(--muted); }
  header .meta span b { color: var(--fg); font-weight: 600; }
  .controls { display: flex; gap: 0.75rem; padding: 0.75rem 1rem; background: var(--panel);
              border: 1px solid var(--border); border-radius: 8px; margin-bottom: 1rem; align-items: center; }
  .controls input[type="text"] { background: oklch(0.18 0.01 270); border: 1px solid var(--border);
                                  color: var(--fg); padding: 0.4rem 0.7rem; border-radius: 4px;
                                  font: 500 0.85rem var(--mono); width: 220px; }
  .controls select { background: oklch(0.18 0.01 270); border: 1px solid var(--border);
                     color: var(--fg); padding: 0.4rem 0.7rem; border-radius: 4px;
                     font: 500 0.8rem var(--sans); cursor: pointer; }
  .controls .stat { font: 500 0.8rem var(--mono); color: var(--muted); margin-left: auto; }
  .controls .stat b { color: var(--fg); }
  table { width: 100%; border-collapse: collapse; background: var(--panel);
          border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  th { text-align: left; padding: 0.6rem 0.85rem; font: 600 0.75rem var(--sans);
       text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted);
       background: oklch(0.16 0.014 270); border-bottom: 1px solid var(--border); }
  td { padding: 0.55rem 0.85rem; font: 500 0.875rem var(--mono);
       border-bottom: 1px solid oklch(0.18 0.01 270 / 0.5); }
  tr:hover td { background: oklch(0.16 0.014 270 / 0.5); }
  .sym { font-weight: 600; color: var(--fg); }
  .price { font-weight: 600; }
  .delta-up   { color: var(--green); }
  .delta-down { color: var(--red); }
  .arrow { display: inline-block; min-width: 0.8rem; text-align: center; }
  .age-fresh { color: var(--green); }
  .age-warm  { color: var(--amber); }
  .age-stale { color: var(--red); }
  .source-stream { color: var(--green); font-size: 0.7rem; }
  .source-rest   { color: var(--amber); font-size: 0.7rem; }
  .source-stale, .source-none { color: var(--muted); font-size: 0.7rem; }
  .flash      { animation: flash 0.4s ease-out; }
  .flash-down { animation: flash-down 0.4s ease-out; }
  @keyframes flash      { 0% { background: oklch(0.72 0.20 155 / 0.30); } 100% { background: transparent; } }
  @keyframes flash-down { 0% { background: oklch(0.65 0.22 25 / 0.30); } 100% { background: transparent; } }
  .ticks-bar { display: inline-block; height: 10px; background: var(--green);
               vertical-align: middle; border-radius: 2px; margin-right: 0.5rem; }
  a.back { color: var(--muted); text-decoration: none; font: 500 0.85rem var(--sans); }
  a.back:hover { color: var(--fg); }
  .pulse { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
           background: var(--green); animation: pulse 1s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  .pulse-off { background: var(--muted); animation: none; }
</style>
</head>
<body>
  <header>
    <div>
      <h1>📡 Stream Monitor</h1>
      <a class="back" href="/">← Back to Dashboard</a>
    </div>
    <div class="meta">
      <span><span id="conn-pulse" class="pulse"></span>&nbsp;<b id="conn-state">connecting…</b></span>
      <span>Symbols: <b id="sym-count">0</b></span>
      <span>Total ticks: <b id="tick-total">0</b></span>
      <span>Tick rate: <b id="tick-rate">0</b>/s</span>
      <span>Msgs: <b id="msg-count">0</b></span>
      <span>Last msg: <b id="last-msg-age">—</b></span>
    </div>
  </header>
  <div class="controls">
    <input type="text" id="filter" placeholder="Filter symbols (e.g. PLTR, AAPL)">
    <select id="sort">
      <option value="ticks">Sort: Most active first</option>
      <option value="age">Sort: Freshest first</option>
      <option value="sym">Sort: Symbol A-Z</option>
      <option value="price">Sort: Price desc</option>
    </select>
    <span class="stat">Last update: <b id="last-update">—</b></span>
  </div>
  <div id="log-container" style="background: var(--panel); border: 1px solid var(--border);
       border-radius: 8px; padding: 0.5rem; height: calc(100vh - 200px); overflow-y: auto;
       font: 12px/1.4 var(--mono);">
    <pre id="log" style="white-space: pre-wrap; word-break: break-all; margin: 0;"></pre>
  </div>

<script>
(function() {
  const TOKEN = window.TRADING_API_KEY;
  const logEl = document.getElementById('log');
  const logContainer = document.getElementById('log-container');
  const connState = document.getElementById('conn-state');
  const connPulse = document.getElementById('conn-pulse');
  const symCountEl = document.getElementById('sym-count');
  const tickTotalEl = document.getElementById('tick-total');
  const tickRateEl = document.getElementById('tick-rate');
  const lastUpdEl = document.getElementById('last-update');
  const filterIn = document.getElementById('filter');
  const sortSel = document.getElementById('sort');

  // Per-symbol state, accumulated for the page's lifetime
  const state = {};
  let totalTicks = 0;
  const tickTimestamps = [];
  // v-stream-heartbeat-2026-05-01: track WS health to recover from
  // silent drops (NAT timeout, idle proxies, server-side cull).
  let msgCount = 0;
  let lastMsgAt = Date.now();
  let heartbeatTimer = null;
  let watchdogTimer = null;

  // Symbols only ever come from the server's PriceBook (constrained
  // to [A-Z0-9]{1,8}); still defensive-validate in the browser.
  const SYM_RX = /^[A-Z0-9]{1,8}$/;
  const safeSymbol = (s) => (typeof s === 'string' && SYM_RX.test(s)) ? s : null;

  function fmtAge(sec) {
    if (sec == null) return '—';
    if (sec < 1)    return sec.toFixed(2) + 's';
    if (sec < 60)   return sec.toFixed(1) + 's';
    if (sec < 3600) return (sec/60).toFixed(1) + 'm';
    return (sec/3600).toFixed(1) + 'h';
  }
  function ageClass(sec) {
    if (sec == null) return 'age-stale';
    if (sec < 5)  return 'age-fresh';
    if (sec < 30) return 'age-warm';
    return 'age-stale';
  }
  function fmtPrice(p, dec) {
    if (p == null || p === 0) return '—';
    if (p < 0.01) return '$' + p.toFixed(6);
    if (p < 1)    return '$' + p.toFixed(4);
    return '$' + p.toFixed(dec ?? 2);
  }

  // v-stream-raw-2026-05-01: append one Schwab message as a JSON
  // block to the scrolling log. Trim the buffer to MAX_LINES so the
  // page doesn't grow unbounded over a long session.
  const MAX_LOG_ENTRIES = 500;
  const logEntries = [];
  function appendRaw(rawMsg) {
    const filterStr = filterIn.value.trim().toUpperCase();
    // Apply optional symbol filter — only include messages whose
    // content[*].key matches the filter. Empty filter = show all.
    if (filterStr) {
      try {
        const content = rawMsg && rawMsg.content;
        const hasMatch = Array.isArray(content) && content.some(r => {
          const k = (r && (r.key || r.KEY)) || '';
          return String(k).toUpperCase().includes(filterStr);
        });
        if (!hasMatch) return;   // doesn't mention any symbol matching filter
      } catch (e) { /* fall through */ }
    }

    const ts = new Date().toLocaleTimeString('en-US', {hour12: false}) +
               '.' + String(Date.now() % 1000).padStart(3, '0');
    const pretty = JSON.stringify(rawMsg, null, 2);
    const entry = '[' + ts + ']\\n' + pretty + '\\n';
    logEntries.push(entry);
    if (logEntries.length > MAX_LOG_ENTRIES) logEntries.shift();
    // Render — preserve scroll position UNLESS user is near bottom
    const wasNearBottom =
      logContainer.scrollHeight - logContainer.scrollTop - logContainer.clientHeight < 80;
    logEl.textContent = logEntries.join('\\n');
    if (wasNearBottom) logContainer.scrollTop = logContainer.scrollHeight;
    totalTicks += 1;
    tickTimestamps.push(Date.now());
    // Track unique symbols seen
    try {
      const content = rawMsg && rawMsg.content;
      if (Array.isArray(content)) {
        content.forEach(r => {
          const k = r && (r.key || r.KEY);
          if (k && safeSymbol(String(k))) state[k] = state[k] || true;
        });
      }
    } catch (e) {}
  }

  // v-stream-raw-2026-05-01: only header counters need refreshing on
  // a timer. The log body is appended to in appendRaw() directly when
  // schwab_raw messages arrive.
  function render() {
    const now = Date.now();
    symCountEl.textContent = Object.keys(state).length;
    tickTotalEl.textContent = totalTicks.toLocaleString();
    const fiveSecAgo = now - 5000;
    while (tickTimestamps.length && tickTimestamps[0] < fiveSecAgo) tickTimestamps.shift();
    tickRateEl.textContent = (tickTimestamps.length / 5).toFixed(1);
    lastUpdEl.textContent = new Date().toLocaleTimeString();
  }

  let ws = null, reconnectAttempt = 0;
  function connect() {
    const url = (location.protocol === 'https:' ? 'wss://' : 'ws://')
                + location.host + '/ws?token=' + encodeURIComponent(TOKEN);
    ws = new WebSocket(url);
    ws.onopen = () => {
      connState.textContent = 'connected';
      connPulse.classList.remove('pulse-off');
      reconnectAttempt = 0;
    };
    ws.onmessage = (ev) => {
      lastMsgAt = Date.now();
      msgCount += 1;
      try {
        const msg = JSON.parse(ev.data);
        // v-stream-raw-2026-05-01: only render schwab_raw messages —
        // ignore dashboard_update / commentary / etc. on this page.
        // The /stream view's job is to print exactly what the
        // streamer delivers, nothing else.
        if (msg.type !== 'schwab_raw') return;
        appendRaw(msg.data);
      } catch (e) { console.warn('parse error', e); }
    };
    ws.onclose = () => {
      connState.textContent = 'disconnected — reconnecting…';
      connPulse.classList.add('pulse-off');
      stopHeartbeat();
      const delay = Math.min(30000, 1000 * Math.pow(2, reconnectAttempt++));
      setTimeout(connect, delay);
    };
    ws.onerror = () => { connState.textContent = 'error'; };
    // v-stream-heartbeat-2026-05-01: send a ping every 10s. Keeps the
    // TCP connection from being idled out by NAT/proxies/uvicorn.
    // Server may or may not respond — the act of writing keeps it alive.
    stopHeartbeat();
    heartbeatTimer = setInterval(() => {
      try {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({type: 'ping', t: Date.now()}));
        }
      } catch (e) { /* swallow */ }
    }, 10000);
  }
  function stopHeartbeat() {
    if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
  }
  connect();

  // v-stream-heartbeat-2026-05-01: watchdog. If we haven't seen a
  // message in >8 seconds (server normally sends 4Hz, so ~32 should
  // arrive in 8s), the connection is dead even if onclose hasn't
  // fired yet (some proxies silently sink without RST). Force-close
  // and let onclose's reconnect kick in.
  watchdogTimer = setInterval(() => {
    const since = Date.now() - lastMsgAt;
    document.getElementById('msg-count').textContent = msgCount.toLocaleString();
    document.getElementById('last-msg-age').textContent =
      since < 1000 ? '<1s' :
      since < 60000 ? Math.round(since/1000) + 's' :
      Math.round(since/60000) + 'm';
    if (ws && ws.readyState === WebSocket.OPEN && since > 8000) {
      console.warn('stream watchdog: no msg for', since, 'ms — closing to reconnect');
      try { ws.close(); } catch (e) {}
    }
  }, 1000);

  filterIn.addEventListener('input', render);
  sortSel.addEventListener('change', render);
  setInterval(render, 1000);  // refresh age column even on quiet windows
})();
</script>
</body>
</html>
"""
    html = html.replace("__APIKEY__", api_json)
    return HTMLResponse(content=html)

@app.post("/api/start")
async def start_trading():
    """v-api-start-async-init-2026-04-30: previously this endpoint blocked
    for ~13 s while the engine constructor loaded the 2.7M-sample ML model
    and 354+ brain memories synchronously inside the FastAPI handler.
    The HTTP response only returned after that load completed.
    Fix: run the heavy constructor in a thread executor so the HTTP
    handler returns in milliseconds. The trading engine's own start()
    still runs in its dedicated thread (unchanged).
    """
    global trading_engine, connection_manager, trading_task, trading_thread

    # Import here to avoid circular imports
    from trading_bot_commentary_updated import TradingEngineWithCommentary, config_manager

    if not trading_engine:
        # Build the engine off the request thread — this is where the
        # 13s delay used to live. Run the synchronous constructor in
        # the default thread executor.
        _loop = asyncio.get_event_loop()
        trading_engine = await _loop.run_in_executor(
            None,
            lambda: TradingEngineWithCommentary(connection_manager=connection_manager),
        )

        # Subscribe to commentary updates
        async def broadcast_commentary(commentary):
            # Handle both TradingCommentary objects and dicts
            if hasattr(commentary, 'to_dict'):
                data = commentary.to_dict()
            elif isinstance(commentary, dict):
                data = commentary
            else:
                logger.error(f"Unexpected commentary type: {type(commentary)}")
                return

            await connection_manager.broadcast({
                'type': 'commentary',
                'data': data
            })

        # Create a thread-safe queue for commentary updates
        commentary_queue = asyncio.Queue()

        # Get the current event loop for cross-thread communication
        main_loop = asyncio.get_event_loop()
        # Engine may run on a background thread/loop; force WS broadcasts
        # back onto the FastAPI loop that owns WebSocket objects.
        trading_engine._ws_broadcast_loop = main_loop
        
        # v-fix-loop-safety-2026-09-10: Set db_logger owner loop so cross-loop
        # snapshot writes/reads route back to this (FastAPI) loop.
        if hasattr(trading_engine, 'db_logger') and trading_engine.db_logger is not None:
            trading_engine.db_logger.set_owner_loop(main_loop)

        # Background task to process commentary broadcasts
        async def commentary_broadcaster():
            while True:
                try:
                    commentary = await commentary_queue.get()
                    await broadcast_commentary(commentary)
                except Exception as e:
                    logger.error(f"Error broadcasting commentary: {e}")

        # Start the broadcaster task
        asyncio.create_task(commentary_broadcaster())

        def queue_commentary(commentary):
            try:
                # Check if the queue and event loop are still valid
                if commentary_queue is None or main_loop is None:
                    return

                # Check if the loop is still running
                if main_loop.is_closed():
                    return

                # Check if loop is running
                if not main_loop.is_running():
                    return

                # Use asyncio.run_coroutine_threadsafe for cross-thread communication
                future = asyncio.run_coroutine_threadsafe(
                    commentary_queue.put(commentary),
                    main_loop
                )
                # Wait for completion with timeout
                future.result(timeout=0.5)
            except asyncio.TimeoutError:
                pass  # Queue full, skip silently
            except RuntimeError as e:
                if "Event loop" not in str(e):
                    logger.debug(f"Commentary queue runtime error: {e}")
            except Exception as e:
                # Only log unexpected errors, not routine threading issues
                error_str = str(e)
                if error_str and "loop" not in error_str.lower():
                    logger.debug(f"Commentary queue error: {type(e).__name__}: {e}")

        trading_engine.commentary.subscribe(queue_commentary)

    if not trading_engine.is_running:
        # v-shutdown-stops-engine-thread-2026-06-09: daemon=True is the
        # backstop — a main-thread exit must never be blocked by this
        # thread. Graceful stop (is_running=False + join) lives in
        # shutdown_handler; state is saved there before exit, so a
        # daemon kill at interpreter teardown loses nothing.
        trading_thread = threading.Thread(
            target=lambda: asyncio.run(trading_engine.start()),
            daemon=True,
        )
        trading_thread.start()

        return {"status": "success", "message": "Trading with commentary started"}

    return {"status": "info", "message": "Already running"}

@app.post("/api/stop")
async def stop_trading():
    if trading_engine:
        trading_engine.is_running = False
        return {"status": "success", "message": "Trading stopped"}
    return {"status": "error", "message": "Not running"}

# ============================================================================
# SETTINGS MANAGEMENT API
# ============================================================================

@app.get("/api/settings")
async def get_all_settings():
    """Get all current settings"""
    try:
        from trading_bot_commentary_updated import config_manager
        config = config_manager.config
        return {
            "status": "success",
            "settings": config,
            "editable": {
                "trading": {
                    "max_positions": {"type": "int", "min": 1, "max": 20, "description": "Maximum number of positions"},
                    "max_position_value": {"type": "float", "min": 100, "max": 100000, "description": "Maximum value per position ($)"},
                    "max_risk_per_trade": {"type": "float", "min": 0.001, "max": 0.1, "description": "Maximum risk per trade (%)"},
                    "max_daily_loss": {"type": "float", "min": 0.01, "max": 0.2, "description": "Maximum daily loss (%)"},
                    "min_risk_reward_ratio": {"type": "float", "min": 1.0, "max": 5.0, "description": "Minimum risk/reward ratio"},
                    "ml_prediction_enabled": {"type": "bool", "description": "Enable ML predictions"},
                },
                "risk": {
                    "stop_loss_percent": {"type": "float", "min": 0.005, "max": 0.1, "description": "Default stop loss (%)"},
                    "take_profit_percent": {"type": "float", "min": 0.01, "max": 0.2, "description": "Default take profit (%)"},
                    "trailing_stop_enabled": {"type": "bool", "description": "Enable trailing stops"},
                },
                "strategies": {
                    "breakout_enabled": {"type": "bool", "description": "Enable breakout strategy"},
                    "mean_reversion_enabled": {"type": "bool", "description": "Enable mean reversion strategy"},
                    "momentum_enabled": {"type": "bool", "description": "Enable momentum strategy"},
                    "min_consensus": {"type": "int", "min": 1, "max": 5, "description": "Minimum strategy consensus"},
                }
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.put("/api/settings")
async def update_settings(settings: dict):
    """Update multiple settings at once"""
    try:
        from trading_bot_commentary_updated import config_manager
        updated = []
        for key, value in settings.items():
            config_manager.update(key, value)
            updated.append(key)

        return {
            "status": "success",
            "message": f"Updated {len(updated)} setting(s)",
            "updated": updated
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/settings/{category}")
async def get_category_settings(category: str):
    """Get settings for a specific category"""
    try:
        from trading_bot_commentary_updated import config_manager
        if category in config_manager.config:
            return {
                "status": "success",
                "category": category,
                "settings": config_manager.config[category]
            }
        return {"status": "error", "message": f"Category '{category}' not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.put("/api/settings/{category}")
async def update_category_settings(category: str, settings: dict):
    """Update settings for a specific category"""
    try:
        from trading_bot_commentary_updated import config_manager
        for key, value in settings.items():
            config_manager.update(f"{category}.{key}", value)

        return {
            "status": "success",
            "message": f"Updated {category} settings",
            "settings": config_manager.config.get(category, {})
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/settings/reset")
async def reset_settings():
    """Reset settings to defaults"""
    try:
        from trading_bot_commentary_updated import config_manager
        config_manager.config = config_manager._get_default_config()
        config_manager._save_config(config_manager.config)
        return {"status": "success", "message": "Settings reset to defaults"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/market-indices")
async def get_market_indices():
    """v-market-indices-strip-2026-05-27: regime strip data.

    Returns the latest cached snapshot of S&P 500, Dow, Nasdaq,
    Russell 2000, and VIX prices + intraday % change. The cache is
    populated by an engine background task every 10s; this route
    simply returns the snapshot. On cold start (before first refresh)
    returns `{"indices": [], "stale": true, ...}` with HTTP 200 so
    the frontend renders "—" placeholders instead of an error.
    """
    from core.market_indices import MarketIndicesCache
    return MarketIndicesCache.instance().snapshot()


@app.get("/api/status")
async def get_bot_status():
    """Get current bot status"""
    return {
        "status": "success",
        "bot_running": trading_engine.is_running if trading_engine else False,
        "mode": trading_engine.mode.value if trading_engine else "not_started",
        "positions_count": len(trading_engine.positions) if trading_engine else 0,
        "schwab_connected": trading_engine.schwab_client is not None if trading_engine else False,
        "uptime": str(datetime.now() - trading_engine.start_time) if trading_engine and hasattr(trading_engine, 'start_time') else "0:00:00"
    }


@app.get("/api/system-stats")
async def get_system_stats():
    """Get system observability stats: NewsBus, supervisor, profile.
    
    v-newsbus-observability-2026-09-08: exposes metrics for monitoring:
      - NewsBus item counts, high-impact alerts, refresh status
      - Supervisor task states and crash counts
      - Current autonomy profile and confirmation settings
    """
    from core.config import Config
    cfg = Config()
    
    result = {
        "status": "success",
        "profile": {
            "name": cfg.TRADING_PROFILE,
            "confirmation_timeout_sec": cfg.CONFIRMATION_TIMEOUT_SEC,
            "confirmation_timeout_action": cfg.CONFIRMATION_TIMEOUT_ACTION,
            "require_close_confirmation": cfg.REQUIRE_CLOSE_CONFIRMATION,
            "flatten_on_circuit": cfg.FLATTEN_ON_CIRCUIT,
        },
        "news_bus": None,
        "news_loop": None,
        "supervisor": None,
        "last_wake_reason": None,
    }
    
    if not trading_engine:
        return result
    
    # NewsBus stats
    bus = getattr(trading_engine, "_news_bus", None)
    if bus is not None:
        stats = bus.get_stats()
        result["news_bus"] = {
            "total_items": stats.total_items,
            "symbols_tracked": stats.symbols_tracked,
            "high_impact_items": stats.high_impact_items,
            "refresh_count": stats.refresh_count,
            "items_evicted": stats.items_evicted,
            "fetch_errors": stats.fetch_errors,
            "wake_events_fired": stats.wake_events_fired,
            "last_refresh": stats.last_refresh.isoformat() if stats.last_refresh else None,
            # v-newsbus-gates-2026-09-09: gate decision counters
            "gate_pass": stats.gate_pass,
            "gate_veto_stale": stats.gate_veto_stale,
            "gate_veto_low_tier": stats.gate_veto_low_tier,
            "gate_veto_no_corroboration": stats.gate_veto_no_corroboration,
            "gate_size_reduced": stats.gate_size_reduced,
        }
    
    # NewsLoop stats
    news_loop = getattr(trading_engine, "_news_loop", None)
    if news_loop is not None:
        result["news_loop"] = news_loop.get_status()
    
    # Supervisor stats
    supervisor = getattr(trading_engine, "_supervisor", None)
    if supervisor is not None:
        result["supervisor"] = supervisor.status()
    
    # Last wake reason
    result["last_wake_reason"] = getattr(trading_engine, "_last_wake_reason", None)
    
    return result


@app.get("/api/news-bus/{symbol}")
async def get_news_bus_symbol(symbol: str, max_age_sec: float = 14400):
    """Get NewsBus items for a specific symbol.
    
    v-newsbus-observability-2026-09-08: inspect cached news for a symbol.
    Useful for debugging why a news signal did or didn't fire.
    """
    if not trading_engine:
        return {"status": "error", "message": "Engine not running"}
    
    bus = getattr(trading_engine, "_news_bus", None)
    if bus is None:
        return {"status": "error", "message": "NewsBus not available"}
    
    items = await bus.get_items(symbol.upper(), max_age_sec=max_age_sec)
    aggregate = await bus.get_aggregate_sentiment(symbol.upper(), max_age_sec=max_age_sec)
    
    return {
        "status": "success",
        "symbol": symbol.upper(),
        "as_of": datetime.now().isoformat(),  # v-newsbus-gates-2026-09-09: timestamp for dashboard
        "item_count": len(items),
        "aggregate": aggregate,
        "items": [item.to_dict() for item in items[:20]],  # Limit to 20 for payload size
        "has_high_impact": bus.has_high_impact(symbol.upper()),
    }


@app.get("/api/account-stats")
async def get_account_stats():
    """Get detailed account statistics - returns real Schwab data when in live mode"""
    if not trading_engine:
        return {
            "status": "success",
            "source": "none",
            "stats": {
                "balance": 0,
                "buying_power": 0,
                "daily_pnl": 0,
                "total_pnl": 0,
                "position_count": 0,
                "cash": 0
            }
        }

    # v-account-stats-schwab-2026-04-30: previously this only fetched
    # Schwab data when mode==LIVE. The dashboard wants the user's REAL
    # account numbers visible even while paper-trading in sim, so fetch
    # whenever schwab_client is connected — regardless of mode.
    has_schwab = trading_engine.schwab_client is not None

    # v-day-pnl-schwab-only-2026-04-30: Day P&L is Schwab's number, period.
    # Schwab is the authoritative source for the user's actual account P&L.
    # Sim/paper trades are bot bookkeeping and shouldn't muddy the live
    # account dashboard.
    if has_schwab:
        try:
            account_info = await trading_engine._get_real_account_info()
            schwab_positions = await trading_engine.get_schwab_positions()

            total_pnl = sum(pos.get('total_pnl', 0) for pos in schwab_positions)

            if account_info:
                return {
                    "status": "success",
                    "source": "schwab",
                    "stats": {
                        "balance": account_info.get('balance', 0),
                        "buying_power": account_info.get('buying_power', 0),
                        "daily_pnl": account_info.get('day_pnl', 0),
                        "total_pnl": total_pnl,
                        "position_count": len(schwab_positions),
                        "cash": account_info.get('cash', 0)
                    }
                }
        except Exception as e:
            logger.error(f"Failed to get Schwab account stats: {e}")

    # Fall back to internal tracking (simulation mode or Schwab unavailable)
    positions = trading_engine.positions if hasattr(trading_engine, 'positions') else {}
    simulated_positions = trading_engine.simulated_positions if hasattr(trading_engine, 'simulated_positions') else {}

    # Use simulated positions in simulation mode
    active_positions = simulated_positions if trading_engine.mode == TradingMode.SIMULATION_WITH_COMMENTARY else positions

    # Calculate P&L from tracked positions
    total_pnl = sum(getattr(pos, 'unrealized_pnl', 0) for pos in active_positions.values())

    return {
        "status": "success",
        "source": "simulation" if trading_engine.mode == TradingMode.SIMULATION_WITH_COMMENTARY else "internal",
        "stats": {
            "balance": trading_engine.risk_manager.account_balance if hasattr(trading_engine, 'risk_manager') else 100000,
            "buying_power": trading_engine.risk_manager.buying_power if hasattr(trading_engine, 'risk_manager') else 50000,
            "daily_pnl": trading_engine.risk_manager.schwab_daily_pnl if hasattr(trading_engine.risk_manager, 'schwab_daily_pnl') else 0,
            "total_pnl": total_pnl,
            "position_count": len(active_positions),
            "cash": 0
        }
    }

@app.get("/api/quote-cache")
async def quote_cache_dump():
    """v-pricebook-2026-05-01: introspection of the canonical
    PriceBook (replaces the legacy QuoteCache dump). Each entry shows
    the canonical mark, last/bid/ask, age, source, and is_stale flag —
    everything a debugger needs to answer "why is this symbol's price
    not updating?". A symbol with is_stale=True for >30s means Schwab
    has stopped sending updates for it (quiet stock or subscription
    issue). A symbol absent from the rows list has never received a
    tick since the stream connected."""
    if not trading_engine or not getattr(trading_engine, 'price_book', None):
        return {"status": "error", "message": "price_book not initialised"}
    pb = trading_engine.price_book
    stream = getattr(trading_engine, '_schwab_quote_stream', None)
    return {
        "status": "success",
        "stream_available": (stream is not None and stream.is_available()) if stream else False,
        "stream_healthy":   stream.is_healthy() if stream else False,
        "stream_tick_count": getattr(stream, '_tick_count', 0) if stream else 0,
        "stream_last_tick_at": (stream._last_tick_at.isoformat() if stream and stream._last_tick_at else None),
        "price_book": pb.status(),
    }


@app.post("/api/refresh-positions")
async def refresh_positions():
    """Manually refresh positions from Schwab"""
    if not trading_engine or not trading_engine.schwab_client:
        return {"status": "error", "message": "Schwab not connected"}

    positions = await trading_engine.get_schwab_positions()

    return {
        "status": "success",
        "positions": positions,
        "count": len(positions)
    }

@app.get("/api/professional/models")
async def get_ml_models():
    """Get available ML models with their status"""
    if not trading_engine or not hasattr(trading_engine, 'ml_predictor'):
        return {'models': [
            {
                'key': 'ensemble',
                'name': 'Ensemble Model (RF + XGB + LGB)',
                'algorithm': 'VotingClassifier',
                'trained': True,
                'active': True,
                'performance': {
                    'accuracy': 0.68,
                    'precision': 0.65,
                    'recall': 0.70
                }
            },
            {
                'key': 'scalping',
                'name': 'Scalping ML Model',
                'algorithm': 'XGBoost + CatBoost',
                'trained': True,
                'active': False,
                'performance': {
                    'accuracy': 0.72,
                    'precision': 0.71,
                    'recall': 0.68
                }
            },
            {
                'key': 'neural',
                'name': 'Neural Network',
                'algorithm': 'MLP',
                'trained': False,
                'active': False,
                'performance': {}
            }
        ]}

    # Get actual model info if ML predictor is available
    models = []
    if hasattr(trading_engine, 'ml_predictor') and hasattr(trading_engine.ml_predictor, 'model'):
        models.append({
            'key': 'ensemble',
            'name': 'Active Ensemble Model',
            'algorithm': type(trading_engine.ml_predictor.model).__name__,
            'trained': True,
            'active': True,
            'performance': {
                'accuracy': 0.68,
                'precision': 0.65,
                'recall': 0.70
            }
        })

    return {'models': models}

@app.post("/api/professional/models/{model_key}/select")
async def select_model(model_key: str):
    """Select an ML model as active"""
    if not trading_engine:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")

    # Add commentary about model selection
    trading_engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.DECISION,
        symbol=None,
        title=f"\U0001f916 ML Model Selection",
        message=f"Selected {model_key} model for predictions",
        importance=8
    ))

    return {'success': True, 'active_model': model_key}

@app.get("/api/professional/risk")
async def get_risk_metrics():
    """Get current risk metrics"""
    if not trading_engine:
        return {
            'var_95': 0.0,
            'current_drawdown': 0.0,
            'leverage': 1.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0
        }

    # Calculate basic risk metrics
    positions = trading_engine.positions
    total_value = sum(pos.quantity * pos.current_price for pos in positions.values())

    # Mock risk metrics for now - in production these would be calculated
    return {
        'var_95': 0.015,  # 1.5% VaR
        'current_drawdown': 0.003,  # 0.3% current drawdown
        'leverage': 1.0,  # No leverage
        'sharpe_ratio': 1.2,
        'sortino_ratio': 1.5,
        'max_drawdown': 0.05,  # 5% max drawdown
        'total_exposure': total_value,
        'position_count': len(positions)
    }

@app.get("/api/professional/strategies")
async def get_strategies():
    """Get available trading strategies and their status"""
    # Get active states from trading engine if available
    active_states = {'momentum': True, 'mean_reversion': True, 'breakout': False, 'scalping': False}
    if trading_engine and hasattr(trading_engine, 'active_strategies'):
        active_states = trading_engine.active_strategies

    strategies = [
        {
            'key': 'momentum',
            'name': 'Momentum Trading',
            'description': 'Trades based on price momentum and trend following',
            'active': active_states.get('momentum', True),
            'performance': {'win_rate': 0.62, 'avg_return': 0.008}
        },
        {
            'key': 'mean_reversion',
            'name': 'Mean Reversion',
            'description': 'Trades oversold/overbought conditions expecting reversal',
            'active': active_states.get('mean_reversion', True),
            'performance': {'win_rate': 0.58, 'avg_return': 0.005}
        },
        {
            'key': 'breakout',
            'name': 'Breakout Strategy',
            'description': 'Trades breakouts from consolidation patterns',
            'active': active_states.get('breakout', False),
            'performance': {'win_rate': 0.55, 'avg_return': 0.007}
        },
        {
            'key': 'scalping',
            'name': 'ML Scalping',
            'description': 'High-frequency scalping with ML predictions',
            'active': active_states.get('scalping', False),
            'performance': {'win_rate': 0.68, 'avg_return': 0.003}
        }
    ]
    return {'strategies': strategies}

@app.get("/api/professional/performance")
async def get_professional_performance_metrics():
    """Get detailed performance metrics"""
    return {
        'daily_pnl': 125.50,
        'weekly_pnl': 680.25,
        'monthly_pnl': 1520.75,
        'total_pnl': 3250.00,
        'win_rate': 0.58,
        'profit_factor': 1.35,
        'sharpe_ratio': 1.2,
        'trades_today': 12,
        'trades_week': 45,
        'trades_month': 180
    }

@app.get("/api/professional/paper/account")
async def get_paper_account():
    """Get paper trading account information"""
    return {
        'balance': 100000,
        'buying_power': 100000,
        'positions_value': 0,
        'cash': 100000,
        'pnl_today': 0,
        'pnl_total': 0
    }

@app.get("/api/professional/paper/positions")
async def get_paper_positions():
    """Get paper trading positions"""
    if trading_engine and trading_engine.mode == TradingMode.SIMULATION_WITH_COMMENTARY:
        positions = []
        for symbol, pos in trading_engine.simulated_positions.items():
            if pos:
                positions.append({
                    'symbol': pos.symbol,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': 0,
                    'mode': getattr(pos, 'mode', 'simulation'),
                })
        return {'positions': positions}
    return {'positions': []}

@app.get("/api/professional/config")
async def get_professional_config():
    """Get professional trading configuration"""
    return {
        'trading_mode': 'simulation',
        'risk_limits': {
            'max_position_size': 0.1,
            'max_portfolio_risk': 0.02,
            'max_daily_loss': 0.02,
            'position_sizing_method': 'FIXED_PERCENTAGE'
        },
        'active_strategies': ['momentum', 'mean_reversion'],
        'ml_models': {
            'ensemble': True,
            'scalping': False
        }
    }

# ==================== Analytics API Endpoints ====================
# analytics_logger module was removed in cleanup commit 41ee4b7. These
# endpoints now return empty success envelopes so dashboard widgets render
# cleanly instead of 500ing. To restore real data, build a replacement
# reader that pulls from trading_commentary.json / trading_state.json.

@app.get("/api/analytics/summary")
async def get_analytics_summary():
    return {"status": "success", "summary": {}, "note": "analytics_logger removed"}

@app.get("/api/analytics/trades")
async def get_trade_history(days: int = 7, symbol: str = None):
    return {"status": "success", "count": 0, "trades": [], "note": "analytics_logger removed"}

@app.get("/api/analytics/decisions")
async def get_decision_analysis(days: int = 7, symbol: str = None):
    return {"status": "success", "analysis": {}, "note": "analytics_logger removed"}

@app.get("/api/analytics/performance")
async def get_performance_history(days: int = 7):
    return {"status": "success", "count": 0, "snapshots": [], "note": "analytics_logger removed"}

@app.get("/api/analytics/daily-stats")
async def get_daily_stats(date: str = None):
    return {"status": "success", "stats": {}, "note": "analytics_logger removed"}

@app.post("/api/analytics/export/{category}")
async def export_analytics(category: str, days: int = 30):
    return {"status": "error", "message": "analytics export disabled — analytics_logger removed"}

# =============================================================================
# NEWS SENTIMENT API ENDPOINTS
# =============================================================================

# Initialize sentiment engine
_sentiment_engine_instance = None

def get_sentiment_engine_instance():
    """Get or create sentiment engine instance"""
    global _sentiment_engine_instance
    if _sentiment_engine_instance is None:
        try:
            from news_sentiment_widget import get_sentiment_engine
            _sentiment_engine_instance = get_sentiment_engine({
                'watchlist': Config().WATCHLIST if hasattr(Config(), 'WATCHLIST') else ['TSLA', 'NVDA', 'AMD', 'AAPL', 'SPY', 'MARA']
            })
        except ImportError:
            logger.warning("News sentiment widget not available")
            return None
    return _sentiment_engine_instance

@app.get("/api/sentiment/update")
async def get_sentiment_update():
    """Get full sentiment update for all watchlist symbols"""
    engine = get_sentiment_engine_instance()
    if not engine:
        return {'status': 'error', 'message': 'Sentiment engine not available'}

    try:
        data = await engine.update()
        return {'status': 'success', 'data': data}
    except Exception as e:
        logger.error(f"Sentiment update failed: {e}")
        return {'status': 'error', 'message': str(e)}

@app.get("/api/sentiment/market")
async def get_market_sentiment():
    """Get overall market sentiment"""
    engine = get_sentiment_engine_instance()
    if not engine:
        return {'status': 'error', 'message': 'Sentiment engine not available'}

    try:
        data = await engine.update()
        return {
            'status': 'success',
            'market_sentiment': data.get('market_sentiment', {}),
            'last_updated': data.get('last_updated')
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.get("/api/sentiment/symbol/{symbol}")
async def get_symbol_sentiment(symbol: str):
    """Get sentiment for a specific symbol"""
    engine = get_sentiment_engine_instance()
    if not engine:
        return {'status': 'error', 'message': 'Sentiment engine not available'}

    try:
        sentiment = engine.get_symbol_sentiment(symbol.upper())
        if sentiment:
            return {'status': 'success', 'sentiment': sentiment}
        else:
            return {'status': 'error', 'message': f'No sentiment data for {symbol}'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.get("/api/sentiment/headlines")
async def get_sentiment_headlines(symbol: str = None, limit: int = 10):
    """Get recent headlines with sentiment scores"""
    engine = get_sentiment_engine_instance()
    if not engine:
        return {'status': 'error', 'message': 'Sentiment engine not available'}

    try:
        headlines = engine.get_headlines(symbol.upper() if symbol else None, limit)
        return {'status': 'success', 'headlines': headlines}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.get("/api/sentiment/alerts")
async def get_sentiment_alerts():
    """Get active sentiment alerts"""
    engine = get_sentiment_engine_instance()
    if not engine:
        return {'status': 'error', 'message': 'Sentiment engine not available'}

    try:
        alerts = [a.to_dict() for a in engine._alerts if not a.acknowledged][-10:]
        return {'status': 'success', 'alerts': alerts}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.post("/api/sentiment/alerts/{alert_id}/acknowledge")
async def acknowledge_sentiment_alert(alert_id: str):
    """Acknowledge a sentiment alert"""
    engine = get_sentiment_engine_instance()
    if not engine:
        return {'status': 'error', 'message': 'Sentiment engine not available'}

    engine.acknowledge_alert(alert_id)
    return {'status': 'success', 'message': f'Alert {alert_id} acknowledged'}

@app.get("/api/sentiment/correlation/{symbol}")
async def get_sentiment_correlation(symbol: str):
    """Get historical sentiment-price correlation for a symbol"""
    engine = get_sentiment_engine_instance()
    if not engine:
        return {'status': 'error', 'message': 'Sentiment engine not available'}

    try:
        correlation = engine.get_sentiment_price_correlation(symbol.upper())
        return {'status': 'success', 'correlation': correlation}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.get("/api/sentiment/social/{symbol}")
async def get_social_sentiment(symbol: str):
    """Get social media sentiment for a symbol"""
    engine = get_sentiment_engine_instance()
    if not engine:
        return {'status': 'error', 'message': 'Sentiment engine not available'}

    try:
        social = await engine.social_tracker.get_social_sentiment(symbol.upper())
        return {'status': 'success', 'social': social.to_dict()}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.get("/api/sentiment/earnings")
async def get_upcoming_earnings():
    """Get upcoming earnings for watchlist"""
    engine = get_sentiment_engine_instance()
    if not engine:
        return {'status': 'error', 'message': 'Sentiment engine not available'}

    try:
        earnings = await engine.earnings_calendar.get_upcoming_earnings(engine.watchlist)
        return {'status': 'success', 'earnings': [e.to_dict() for e in earnings]}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# ============================================================================
# BACKTESTING API ENDPOINTS
# ============================================================================

@app.post("/api/professional/risk/settings")
async def update_risk_settings(settings: dict):
    """Update risk management settings"""
    # In a real implementation, this would update the risk manager
    return {'success': True, 'settings': settings}

@app.post("/api/professional/strategies/{strategy_key}/toggle")
async def toggle_strategy(strategy_key: str):
    """Toggle a trading strategy on/off"""
    if not trading_engine:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")

    # Define available strategies
    available_strategies = ['momentum', 'mean_reversion', 'breakout', 'scalping']

    if strategy_key not in available_strategies:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_key} not found")

    # In a real implementation, this would enable/disable the strategy in the trading engine
    # For now, we'll just track it and add commentary
    if not hasattr(trading_engine, 'active_strategies'):
        trading_engine.active_strategies = {'momentum': True, 'mean_reversion': True, 'breakout': False, 'scalping': False}

    # Toggle the strategy
    current_state = trading_engine.active_strategies.get(strategy_key, False)
    trading_engine.active_strategies[strategy_key] = not current_state
    new_state = trading_engine.active_strategies[strategy_key]

    # Add commentary about the change
    trading_engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.DECISION,
        symbol=None,
        title=f"\U0001f4ca Strategy {'Enabled' if new_state else 'Disabled'}",
        message=f"{strategy_key.replace('_', ' ').title()} strategy has been {'enabled' if new_state else 'disabled'}",
        importance=7
    ))

    return {
        'success': True,
        'strategy': strategy_key,
        'active': new_state,
        'message': f"Strategy {strategy_key} {'enabled' if new_state else 'disabled'}"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """v-ws-cached-state-2026-04-30: WebSocket dashboard updater.

    Previously every WS tick (1 Hz, per client) made:
      - 1× sync schwab_client.get_quote per simulated position (sync, blocking)
      - 1× await trading_engine.get_schwab_positions() (Schwab API)
      - 1× await trading_engine._get_real_account_info() (Schwab API)
      - Sentiment update on a 30-tick counter

    With multiple WS clients and Schwab latency, the WS event loop spent
    most of its time awaiting Schwab — a single client's first message
    could take 3 minutes if Schwab calls were slow.

    Fix: read CACHED engine state. The engine's own main loop already
    syncs Schwab every 5 cycles and updates position.current_price /
    position.unrealized_pnl in memory. WS just reads what's already there.
    Schwab account info is similarly cached in risk_manager fields. No
    network calls inside the WS handler.
    """
    # Verify WebSocket auth token from query params
    token = websocket.query_params.get("token")
    if not verify_ws_token(token):
        await websocket.close(code=1008, reason="Authentication required")
        return
    await connection_manager.connect(websocket)

    try:
        # v-realtime-quote-overlay-2026-05-01: 4Hz dashboard tick (was 1Hz).
        # The QuoteCache is now fed by the websocket stream at tick rate;
        # broadcasting at 250ms means the browser sees price moves within
        # ~250ms of Schwab's tick. JSON payload is small (~3KB); 4Hz is
        # trivial CPU. Keeps the bot's own dashboard feeling as live as
        # the embedded TradingView chart.
        _DASH_TICK_SEC = 0.25
        while True:
            await asyncio.sleep(_DASH_TICK_SEC)

            if trading_engine:
                # Sim positions — read from PriceBook (single source of truth).
                # v-pricebook-2026-05-01: WS handler now pulls marks
                # from the canonical PriceBook via CalculationEngine.
                # No more reading pos.current_price (which is now a
                # reactive-pull property anyway) and no more touching
                # the legacy QuoteCache. is_stale comes through to the
                # frontend so it can grey out unfresh values.
                from core.calculations import CalculationEngine
                from core.price_book import PriceBook
                pb = getattr(trading_engine, 'price_book', None) or PriceBook.instance()
                sim_positions_data = []
                if trading_engine.mode == TradingMode.SIMULATION_WITH_COMMENTARY:
                    for symbol, pos in list(trading_engine.simulated_positions.items()):
                        if pos is None:
                            continue
                        if pb is not None:
                            mark = pb.get_mark(symbol)
                            pnl = CalculationEngine.get_pnl(pos, mark)
                            live_price = mark.price if mark.has_price else pos.entry_price
                            live_upnl = pnl.unrealized_pnl
                            is_stale = pnl.is_stale
                        else:
                            live_price = pos.entry_price
                            live_upnl = 0.0
                            is_stale = True
                        sim_positions_data.append({
                            'symbol': pos.symbol,
                            'quantity': pos.quantity,
                            'entry_price': pos.entry_price,
                            'current_price': live_price,
                            'stop_loss': getattr(pos, 'stop_loss', None),
                            'take_profit': getattr(pos, 'take_profit', None),
                            'unrealized_pnl': live_upnl,
                            'is_stale': is_stale,   # frontend can grey out
                            'type': 'simulated',
                            'mode': getattr(pos, 'mode', 'simulation'),
                            'managed_by_bot': bool(getattr(pos, 'managed_by_bot', True)),
                        })

                # Real Schwab positions — PriceBook-driven.
                # v-pricebook-2026-05-01: structural baseline (qty,
                # average_price, day_pnl baseline at the time of the
                # last REST snapshot) still comes from
                # `_schwab_positions_cache`. The MARK is canonical —
                # PriceBook.get_mark(). All derived P&L (unrealized,
                # percent, market value, day-P&L drift) flows through
                # the SAME CalculationEngine that the FSM uses, so
                # there can never be a divergence between "what the bot
                # decides" and "what the dashboard shows."
                real_positions_data = []
                # v-daypnl-accuracy-2026-06-11: accumulate the drift we
                # add to position rows so the account tile gets the SAME
                # correction — tile and rows previously disagreed by
                # construction (tile showed last sync verbatim).
                _total_drift = 0.0
                _rth = _is_rth_now()
                if trading_engine.schwab_client:
                    # v-daypnl-accuracy-2026-06-11 (part 2): the engine
                    # refreshes this cache every 5 ANALYSIS cycles —
                    # minutes each — so the baseline could be 10+ min
                    # old (COIN drifted $57 in that window pre-market).
                    # The dashboard now keeps its own 45s freshness:
                    # single-flight, fire-and-forget, executor-backed
                    # with a 6s timeout inside get_schwab_positions.
                    import time as _time
                    _cache_at = getattr(trading_engine,
                                        '_schwab_positions_cache_at', 0.0)
                    if (_time.time() - _cache_at > 45.0
                            and not getattr(trading_engine,
                                            '_pos_cache_refreshing', False)):
                        trading_engine._pos_cache_refreshing = True

                        async def _refresh_pos_cache():
                            try:
                                fresh = await trading_engine.get_schwab_positions()
                                if fresh:
                                    trading_engine._schwab_positions_cache = fresh
                                # v-tile-row-same-clock-2026-06-12: the
                                # account tile read risk_manager state
                                # refreshed only by the ENGINE sync
                                # (minutes), while the rows refresh here
                                # every 45s — measured 08:15: COIN's row
                                # moved +$11 while the tile sat frozen.
                                # Same refresher now syncs both. Fields
                                # set directly (mirrors
                                # sync_with_schwab_data) because that
                                # method INFO-logs per call — 45s
                                # cadence would add ~2k log lines/day.
                                acct = await trading_engine._get_real_account_info()
                                if acct:
                                    _rm = trading_engine.risk_manager
                                    _rm.schwab_daily_pnl = acct.get('day_pnl', 0)
                                    _rm.buying_power = acct.get(
                                        'buying_power', _rm.buying_power)
                                    _rm.account_balance = acct.get(
                                        'balance', _rm.account_balance)
                                trading_engine._schwab_positions_cache_at = _time.time()
                            except Exception as _exc:
                                logger.debug("dashboard pos-cache refresh "
                                             "failed: %s", _exc)
                            finally:
                                trading_engine._pos_cache_refreshing = False

                        asyncio.create_task(_refresh_pos_cache())
                    cache = getattr(trading_engine, '_schwab_positions_cache', None) or []
                    for sp in cache:
                        sym = sp.get('symbol')
                        tracked_pos = trading_engine.positions.get(sym)
                        is_long_term = getattr(tracked_pos, 'is_long_term', False) if tracked_pos else False
                        managed = bool(getattr(tracked_pos, 'managed_by_bot', False)) if tracked_pos else False

                        # Baseline from Schwab REST snapshot
                        qty = sp.get('quantity', 0)
                        avg = sp.get('average_price', 0) or 0
                        rest_price = sp.get('current_price', 0) or 0
                        rest_day_pnl = sp.get('day_pnl', 0) or 0

                        # Build a synthetic Position-shaped object the
                        # CalculationEngine can score. We don't have the
                        # tracked Position for some pre-existing Schwab
                        # holdings; this works whether tracked_pos is
                        # present or not.
                        side = 'short' if qty < 0 else 'long'
                        class _Synthetic:
                            symbol = sym
                            entry_price = avg
                            quantity = abs(qty) if side == 'short' else qty
                        _syn = _Synthetic()
                        _syn.side = side

                        # Reactive-pull mark from PriceBook
                        if pb is not None:
                            mark = pb.get_mark(sym)
                            pnl = CalculationEngine.get_pnl(_syn, mark)
                            live_price = mark.price if mark.has_price else rest_price
                            live_total_pnl = pnl.unrealized_pnl
                            live_pct = pnl.unrealized_pnl_pct
                            is_stale = pnl.is_stale
                            mv = pnl.market_value or (abs(qty) * live_price)
                        else:
                            live_price = rest_price
                            live_total_pnl = sp.get('total_pnl', 0) or 0
                            live_pct = sp.get('pnl_percent', 0) or 0
                            is_stale = True
                            mv = sp.get('market_value', 0)

                        # Day-P&L drift: tick the day-P&L value with
                        # the price move since the REST snapshot, so
                        # the day-P&L tile also feels live.
                        # v-daypnl-accuracy-2026-06-11: RTH only — see
                        # _is_rth_now(). Outside regular hours Schwab's
                        # REST number IS the truth; sparse pre-market
                        # stream marks made drift subtract accuracy.
                        if _rth and rest_price and live_price:
                            price_drift = (live_price - rest_price) * (qty if side == 'long' else -abs(qty))
                            live_day_pnl = rest_day_pnl + price_drift
                            _total_drift += price_drift
                        else:
                            live_day_pnl = rest_day_pnl

                        real_positions_data.append({
                            'symbol': sym,
                            'quantity': qty,
                            'entry_price': avg,
                            'current_price': live_price,
                            'unrealized_pnl': live_total_pnl,
                            'day_pnl': live_day_pnl,
                            'pnl_percent': live_pct,
                            'market_value': mv,
                            'is_stale': is_stale,
                            'is_long_term': is_long_term,
                            'type': 'real',
                            'mode': 'live',
                            'managed_by_bot': managed,
                        })

                # Account info — read CACHED risk_manager state, which is
                # synced from Schwab every 5 engine cycles (and once at
                # startup). day_pnl is Schwab's authoritative number for
                # the user's account; sim trade P&L is NOT mixed in.
                # v-day-pnl-schwab-only-2026-04-30.
                rm = trading_engine.risk_manager
                # v-daypnl-accuracy-2026-06-11: tile = last-sync Schwab
                # baseline + the same stream drift applied to the rows
                # above, so the tile always equals what the rows imply.
                _base_day_pnl = (getattr(rm, 'schwab_daily_pnl', None)
                                 or getattr(rm, 'daily_pnl', 0) or 0)
                account_info = {
                    'balance': getattr(rm, 'account_balance', 0),
                    'buying_power': getattr(rm, 'buying_power', 0),
                    'day_pnl': _base_day_pnl + _total_drift,
                    'cash': 0,
                }

                # Get screener data — overlay LIVE price from PriceBook.
                # v-watchlist-stream-2026-05-01: ticker tape used to
                # show prices from the screener's REST snapshot only —
                # which only refreshed every 60-120s. Now: structural
                # fields (volume, volatility, high/low/change) still
                # come from the screener snapshot, but `last` is
                # overlaid from the streaming PriceBook so the ticker
                # tape ticks in lockstep with the position rows.
                # v-watchlist-ui-cap-2026-05-11: dashboard had a hardcoded
                # `[:10]` slice even though the engine's WATCHLIST_SIZE
                # was 20 (and Config.yaml was the same). UI was clipping
                # half the analyzed names. Honor the same config knob.
                try:
                    _ui_cap = int(Config().WATCHLIST_SIZE or 20)
                except Exception:
                    _ui_cap = 20
                screener_data = []
                if trading_engine.screener and trading_engine.screener.top_movers:
                    for m in trading_engine.screener.top_movers[:_ui_cap]:
                        sym = m['symbol']
                        rest_last = m.get('last', 0) or 0
                        live_last = rest_last
                        if pb is not None:
                            obj = pb.get_mark(sym)
                            if obj.has_price and not obj.is_stale:
                                live_last = obj.price
                        screener_data.append({
                            'symbol': sym,
                            'last': live_last,
                            'change': m.get('percent_change', 0),
                            'volume': m.get('volume', 0),
                            'volatility': m.get('volatility', 0),
                            'high': m.get('high', 0),
                            'low': m.get('low', 0),
                        })

                # v-watchlist-stream-2026-05-01: live_quotes block —
                # the canonical "every symbol the bot has skin in,
                # right now, with its current mark." Frontend handlers
                # patch any element tagged data-price-symbol="X" by
                # looking up live_quotes[X].price. Single source of
                # truth for every price displayed on the dashboard.
                live_quotes = {}
                if pb is not None:
                    snap = pb.status().get('rows', [])
                    for r in snap:
                        sym = r.get('symbol')
                        if not sym:
                            continue
                        live_quotes[sym] = {
                            'price': r.get('price'),
                            'bid': r.get('bid'),
                            'ask': r.get('ask'),
                            'is_stale': r.get('is_stale', False),
                            'source': r.get('source'),
                            'age_sec': r.get('age_sec'),
                        }

                # Get sentiment data (every 30 seconds to avoid excessive API calls)
                sentiment_data = None
                if not hasattr(websocket, '_last_sentiment_update'):
                    websocket._last_sentiment_update = datetime.now() - timedelta(seconds=30)
                if not hasattr(websocket, '_sentiment_update_count'):
                    websocket._sentiment_update_count = 0

                websocket._sentiment_update_count += 1
                if websocket._sentiment_update_count % 30 == 0:  # Every 30 seconds
                    try:
                        sentiment_engine = get_sentiment_engine_instance()
                        if sentiment_engine:
                            sentiment_data = await sentiment_engine.update()
                            websocket._last_sentiment_update = datetime.now()
                    except Exception as e:
                        logger.debug(f"Sentiment update error: {e}")

                try:
                    await websocket.send_json({
                        'type': 'dashboard_update',
                        'data': {
                            'account': {
                                'balance': account_info.get('balance', 0),
                                'buying_power': account_info.get('buying_power', 0),
                                'daily_pnl': account_info.get('day_pnl', 0),
                                'margin_call': trading_engine.risk_manager.margin_call,
                                'cash': account_info.get('cash', 0)
                            },
                            'simulated_positions': sim_positions_data,
                            'real_positions': real_positions_data,
                            'trades': trading_engine.get_todays_trades(),
                            'screener': screener_data,
                            'live_quotes': live_quotes,  # v-watchlist-stream-2026-05-01
                            # v-symbol-intel-2026-06-10: per-symbol
                            # composite reads for the Symbol
                            # Intelligence panel. Top 40 by quality
                            # keeps the 4Hz payload light.
                            'symbol_intel': (
                                trading_engine._symbol_intel_hub.snapshot()[:40]
                                if getattr(trading_engine,
                                           '_symbol_intel_hub', None)
                                else []
                            ),
                        }
                    })

                    # Send sentiment update separately if available
                    if sentiment_data:
                        await websocket.send_json({
                            'type': 'sentiment_update',
                            'data': sentiment_data
                        })

                        # Send any critical alerts
                        alerts = sentiment_data.get('alerts', [])
                        critical_alerts = [a for a in alerts if a.get('urgency') in ['critical', 'high']]
                        for alert in critical_alerts:
                            await websocket.send_json({
                                'type': 'sentiment_alert',
                                'data': alert
                            })

                except (ConnectionClosedError, ConnectionResetError):
                    # Connection closed, break the loop
                    break
                except Exception as e:
                    # v-ws-broadcast-resilience-2026-05-04: starlette/uvicorn
                    # raises a plain RuntimeError ("Cannot call \"send\" once
                    # a close message has been sent.") on a half-closed
                    # socket — not a ConnectionClosedError. Without matching
                    # this string, the loop kept retrying the same dead
                    # socket every 250 ms, producing ~240 errors/minute and
                    # never shipping a real dashboard payload.
                    s = str(e)
                    if (
                        "Connection closed" in s
                        or "sent 1012" in s
                        or "close message has been sent" in s
                        or "WebSocket is not connected" in s
                        or "websocket.close" in s
                    ):
                        break
                    logger.debug(f"Error sending WebSocket data: {e}")



    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket)
    except asyncio.CancelledError:
        # Handle graceful shutdown
        await connection_manager.disconnect(websocket)
        logger.info("WebSocket connection cancelled during shutdown")
    except Exception as e:
        # Handle any other connection errors
        await connection_manager.disconnect(websocket)
        if "Connection closed" not in str(e) and "sent 1012" not in str(e):
            logger.error(f"WebSocket error: {e}")

@app.post("/api/confirm-close")
async def confirm_close(request: dict):
    """Handle user confirmation for closing positions"""
    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}

    request_id = request.get('request_id')
    confirmed = request.get('confirmed', False)

    if hasattr(trading_engine, 'pending_close_requests'):
        if request_id in trading_engine.pending_close_requests:
            trading_engine.pending_close_requests[request_id]['confirmed'] = confirmed
            return {"status": "success", "confirmed": confirmed}

    return {"status": "error", "message": "Invalid or expired request"}

@app.get("/api/get-trades-today")
async def get_trades_today():
    """Get today's actual trades from Schwab"""
    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}

    try:
        trades = trading_engine.get_todays_trades()
        return {"status": "success", "trades": trades}
    except Exception as e:
        logger.error(f"Error getting today's trades: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/trades")
async def get_trades(date: str = None, symbol: str = None, strategy: str = None, limit: int = 50):
    """Query trades from Postgres with optional filters.

    Examples:
      GET /api/trades                         → last 50 trades
      GET /api/trades?date=2026-04-20         → all trades on April 20
      GET /api/trades?symbol=TSLA             → all TSLA trades
      GET /api/trades?strategy=breakout       → breakout trades only
      GET /api/trades?date=2026-04-20&limit=5 → last 5 on that day
    """
    import os
    from sqlalchemy import create_engine, text as sa_text

    dsn = os.environ.get(
        "POSTGRES_DSN",
        "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev",
    )
    try:
        engine = create_engine(dsn)
        conditions = []
        params = {"lim": min(limit, 500)}

        if date:
            conditions.append("DATE(exit_time) = :dt")
            params["dt"] = date
        if symbol:
            conditions.append("symbol = :sym")
            params["sym"] = symbol.upper()
        if strategy:
            conditions.append("strategy = :strat")
            params["strat"] = strategy

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = sa_text(f"""
            SELECT id, symbol, side, strategy, entry_time::text, exit_time::text,
                   entry_price, exit_price, quantity, pnl,
                   ROUND(pnl_pct::numeric, 2) AS pnl_pct, exit_reason,
                   atr_at_entry, stop_loss, take_profit, confidence,
                   meta_proba, kelly_fraction, scaled_out, mode
            FROM bot_trades {where}
            ORDER BY exit_time DESC LIMIT :lim
        """)

        with engine.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()

        # v-fix-pool-leak-2026-09-10: dispose engine to release connection pool
        engine.dispose()

        trades = [dict(r) for r in rows]
        # v-json-nan-sanitize-trades-2026-05-05: same NaN/inf scrub as
        # /api/positions/db — Postgres can return non-finite floats
        # (pnl_pct, atr_at_entry, meta_proba) which json.dumps rejects
        # with "Out of range float values are not JSON compliant".
        import math as _math
        def _scrub(v):
            if isinstance(v, float):
                return None if (_math.isnan(v) or _math.isinf(v)) else v
            return v
        trades = [{k: _scrub(v) for k, v in t.items()} for t in trades]

        def _safe_pnl(t):
            v = t.get("pnl")
            try:
                f = float(v)
                if _math.isnan(f) or _math.isinf(f):
                    return 0.0
                return f
            except (TypeError, ValueError):
                return 0.0

        summary = {
            "count": len(trades),
            "total_pnl": round(sum(_safe_pnl(t) for t in trades), 2),
            "winners": sum(1 for t in trades if _safe_pnl(t) > 0),
            "losers": sum(1 for t in trades if _safe_pnl(t) < 0),
        }
        if trades:
            summary["win_rate"] = round(100 * summary["winners"] / summary["count"], 1)

        return {"status": "success", "summary": summary, "trades": trades}

    except Exception as e:
        logger.error(f"Error querying trades: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/decisions")
async def get_decisions(date: str = None, symbol: str = None, component: str = None, limit: int = 100):
    """Query engine decisions from Postgres.

    Examples:
      GET /api/decisions?component=meta_shadow    → all shadow evals
      GET /api/decisions?symbol=TSLA&date=2026-04-20
      GET /api/decisions?component=correlation_guard
    """
    import os
    from sqlalchemy import create_engine, text as sa_text

    dsn = os.environ.get(
        "POSTGRES_DSN",
        "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev",
    )
    try:
        engine = create_engine(dsn)
        conditions = []
        params = {"lim": min(limit, 500)}

        if date:
            conditions.append("DATE(ts) = :dt")
            params["dt"] = date
        if symbol:
            conditions.append("symbol = :sym")
            params["sym"] = symbol.upper()
        if component:
            conditions.append("component = :comp")
            params["comp"] = component

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = sa_text(f"""
            SELECT id, ts::text, component, symbol, action, reason,
                   confidence, meta_proba, atr, price, details_json
            FROM bot_decisions {where}
            ORDER BY ts DESC LIMIT :lim
        """)

        with engine.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()

        # v-fix-pool-leak-2026-09-10: dispose engine to release connection pool
        engine.dispose()

        decisions = [dict(r) for r in rows]
        import math as _math
        def _scrub(v):
            if isinstance(v, float):
                return None if (_math.isnan(v) or _math.isinf(v)) else v
            return v
        decisions = [{k: _scrub(v) for k, v in d.items()} for d in decisions]
        return {"status": "success", "count": len(decisions), "decisions": decisions}

    except Exception as e:
        logger.error(f"Error querying decisions: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/positions/db")
async def get_positions_db():
    """Current open positions.

    Prefer live in-memory + PriceBook-derived marks when engine is running.
    Fall back to Postgres mirror when engine is unavailable.
    """
    import os
    from sqlalchemy import create_engine, text as sa_text

    # v-live-positions-db-api-2026-05-04: `/api/positions/db` used to read
    # only the Postgres mirror, which lags the websocket path by one save
    # cycle and can make symbol price/PnL disagree across dashboard widgets.
    # Build from live engine state first so all surfaces use the same marks.
    if trading_engine is not None:
        try:
            from core.calculations import CalculationEngine
            from core.price_book import PriceBook

            pb = getattr(trading_engine, "price_book", None) or PriceBook.instance()
            # Live wins when same symbol exists in both collections.
            all_positions = {
                **getattr(trading_engine, "simulated_positions", {}),
                **getattr(trading_engine, "positions", {}),
            }

            positions = []
            for sym, pos in all_positions.items():
                if pos is None or getattr(pos, "quantity", 0) <= 0:
                    continue

                if pb is not None:
                    mark = pb.get_mark(sym)
                    pnl = CalculationEngine.get_pnl(pos, mark)
                    live_price = mark.price if mark.has_price else float(getattr(pos, "entry_price", 0) or 0)
                    live_upnl = float(pnl.unrealized_pnl or 0)
                else:
                    live_price = float(getattr(pos, "current_price", 0) or 0)
                    live_upnl = float(getattr(pos, "unrealized_pnl", 0) or 0)

                positions.append({
                    "symbol": sym,
                    "side": getattr(pos, "side", "long"),
                    "strategy": (getattr(pos, "reasoning", {}) or {}).get("strategy"),
                    "mode": getattr(pos, "mode", "simulation"),
                    "entry_time": getattr(pos, "entry_time", None).isoformat() if getattr(pos, "entry_time", None) else None,
                    "entry_price": float(getattr(pos, "entry_price", 0) or 0),
                    "current_price": live_price,
                    "quantity": int(getattr(pos, "quantity", 0) or 0),
                    "stop_loss": float(getattr(pos, "stop_loss", 0) or 0),
                    "take_profit": float(getattr(pos, "take_profit", 0) or 0),
                    "trailing_stop": getattr(pos, "trailing_stop", None),
                    "scaled_out": bool(getattr(pos, "scaled_out", False)),
                    "unrealized_pnl": round(live_upnl, 2),
                    "updated_at": datetime.now().isoformat(),
                    "managed_by_bot": bool(getattr(pos, "managed_by_bot", False)),
                })

            positions.sort(key=lambda p: p.get("entry_time") or "")
            # v-json-nan-sanitize-2026-05-04: Python's json.dumps rejects
            # NaN/inf with "Out of range float values are not JSON compliant",
            # which used to make /api/positions/db return 500 on any
            # position whose P&L or trailing_stop computed to inf (e.g.
            # HQGE with current_price=0). Replace non-finite floats with
            # None so the dashboard's positions feed always returns 200.
            import math as _math
            def _scrub(v):
                if isinstance(v, float):
                    return None if (_math.isnan(v) or _math.isinf(v)) else v
                return v
            positions = [{k: _scrub(v) for k, v in p.items()} for p in positions]
            return {"status": "success", "count": len(positions), "positions": positions}
        except Exception as exc:
            logger.debug(f"live positions build failed, falling back to DB: {exc}")

    dsn = os.environ.get(
        "POSTGRES_DSN",
        "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev",
    )
    try:
        engine = create_engine(dsn)
        with engine.connect() as conn:
            rows = conn.execute(sa_text("""
                SELECT symbol, side, strategy, mode, entry_time::text, entry_price,
                       current_price, quantity, stop_loss, take_profit,
                       trailing_stop, scaled_out,
                       ROUND(unrealized_pnl::numeric, 2) AS unrealized_pnl,
                       updated_at::text
                FROM bot_positions ORDER BY entry_time
            """)).mappings().all()

        # v-fix-pool-leak-2026-09-10: dispose engine to release connection pool
        engine.dispose()

        positions = [dict(r) for r in rows]
        # v-managed-by-bot-2026-04-28: enrich DB rows with per-position flag from
        # in-memory state so the dashboard checkbox reflects current truth (DB
        # mirror lags by up to one save cycle and doesn't store this field yet).
        if trading_engine is not None:
            for p in positions:
                sym = p['symbol']
                live_pos = (trading_engine.simulated_positions.get(sym)
                            or trading_engine.positions.get(sym))
                p['managed_by_bot'] = bool(getattr(live_pos, 'managed_by_bot', False)) if live_pos else False
        return {"status": "success", "count": len(positions), "positions": positions}

    except Exception as e:
        logger.error(f"Error querying positions: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# DECISION SNAPSHOT API — v-feature-snapshot-2026-09-09
# Exposes decision snapshots for ML training pipelines and tooling.
# ============================================================================

@app.get("/api/decision-snapshots")
async def get_decision_snapshots(
    symbol: str = None,
    strategy_id: str = None,
    action: str = None,
    limit: int = 100,
    offset: int = 0,
    since: str = None,
):
    """v-feature-snapshot-2026-09-09: query decision snapshots for ML training.
    
    Returns machine-readable snapshots capturing what the bot saw at each
    decision point (signal, skip, veto). These snapshots are the training
    data for GPU-based policy models.
    
    Query params:
        symbol: Filter by ticker symbol
        strategy_id: Filter by strategy name
        action: Filter by action (signal_buy, signal_sell, skip, veto, error)
        limit: Max results (default 100, max 1000)
        offset: Pagination offset
        since: ISO timestamp — only return snapshots after this time
    
    Returns:
        {
            "status": "success",
            "count": N,
            "snapshots": [...]
        }
    
    GPU inference sidecar integration:
        Poll this endpoint or connect to the WebSocket for real-time snapshots.
        Each snapshot contains price_vol (17-dim feature vector), news aggregate,
        and regime context — ready for model input.
    """
    import os
    from datetime import datetime
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text as sa_text
    
    limit = min(limit, 1000)
    
    # If engine has db_logger, use it
    if trading_engine is not None and getattr(trading_engine, "db_logger", None) is not None:
        try:
            since_dt = datetime.fromisoformat(since) if since else None
            snapshots = await trading_engine.db_logger.get_decision_snapshots(
                symbol=symbol,
                strategy_id=strategy_id,
                action=action,
                limit=limit,
                offset=offset,
                since=since_dt,
            )
            return {"status": "success", "count": len(snapshots), "snapshots": snapshots}
        except Exception as e:
            logger.error(f"Error querying snapshots via db_logger: {e}")
    
    # Fallback: direct DB query
    dsn = os.environ.get("POSTGRES_DSN", "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev")
    if "asyncpg" not in dsn:
        dsn = dsn.replace("postgresql://", "postgresql+asyncpg://")
    
    try:
        engine = create_async_engine(dsn, pool_size=1, max_overflow=0)
        
        filters = []
        params = {"limit": limit, "offset": offset}
        
        if symbol:
            filters.append("symbol = :symbol")
            params["symbol"] = symbol.upper()
        if strategy_id:
            filters.append("strategy_id = :strategy_id")
            params["strategy_id"] = strategy_id
        if action:
            filters.append("action = :action")
            params["action"] = action
        if since:
            filters.append("ts >= :since")
            params["since"] = since
        
        where = "WHERE " + " AND ".join(filters) if filters else ""
        
        async with engine.begin() as conn:
            rows = (await conn.execute(
                sa_text(f"""
                    SELECT snapshot_id, symbol, ts, mode, strategy_id,
                           action, reason, gate_name, confidence, price,
                           price_vol_json, news_json, regime_json,
                           would_entry_price, would_stop_loss, would_take_profit,
                           would_size_shares, would_size_mult, extra_json
                    FROM bot_decision_snapshots
                    {where}
                    ORDER BY ts DESC
                    LIMIT :limit OFFSET :offset
                """),
                params,
            )).mappings().all()
        
        await engine.dispose()
        
        import json
        snapshots = []
        for row in rows:
            snapshots.append({
                "snapshot_id": row["snapshot_id"],
                "symbol": row["symbol"],
                "ts": row["ts"].isoformat() if hasattr(row["ts"], "isoformat") else row["ts"],
                "mode": row["mode"],
                "strategy_id": row["strategy_id"],
                "action": row["action"],
                "reason": row["reason"],
                "gate_name": row["gate_name"],
                "confidence": row["confidence"],
                "price_vol": json.loads(row["price_vol_json"]) if row["price_vol_json"] else {},
                "news": json.loads(row["news_json"]) if row["news_json"] else {},
                "regime": json.loads(row["regime_json"]) if row["regime_json"] else {},
                "would_entry_price": row["would_entry_price"],
                "would_stop_loss": row["would_stop_loss"],
                "would_take_profit": row["would_take_profit"],
                "would_size_shares": row["would_size_shares"],
                "would_size_mult": row["would_size_mult"],
                "extra": json.loads(row["extra_json"]) if row["extra_json"] else {},
            })
        
        return {"status": "success", "count": len(snapshots), "snapshots": snapshots}
        
    except Exception as e:
        logger.error(f"Error querying snapshots: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/features/{symbol}")
async def get_symbol_features(symbol: str):
    """v-feature-snapshot-2026-09-09: get the latest feature snapshot for a symbol.
    
    Returns the most recent DecisionSnapshot for the given symbol, with the
    full feature vector (price_vol, news, regime) that can be fed directly
    to a policy model.
    
    Returns:
        {
            "status": "success",
            "symbol": "TSLA",
            "snapshot": {...},
            "feature_vector": [0.01, -0.02, ...]  // 17-dim price_vol vector
        }
    
    GPU inference sidecar integration:
        Call this endpoint to get the latest features for a symbol before
        making an inference decision. The feature_vector is in the same
        order as MLFeatureExtractor.feature_names.
    """
    import os
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text as sa_text
    
    symbol = symbol.upper()
    
    # If engine has db_logger, use it
    if trading_engine is not None and getattr(trading_engine, "db_logger", None) is not None:
        try:
            snapshot = await trading_engine.db_logger.get_latest_snapshot(symbol)
            if snapshot:
                # Extract feature vector from price_vol
                pv = snapshot.get("price_vol", {})
                feature_vector = [
                    pv.get("returns_1", 0), pv.get("returns_5", 0), pv.get("returns_20", 0),
                    pv.get("price_vs_sma20", 0), pv.get("price_vs_sma50", 0),
                    pv.get("rsi", 0.5), pv.get("macd", 0), pv.get("macd_signal", 0), pv.get("macd_hist", 0),
                    pv.get("bb_position", 0.5), pv.get("bb_width", 0),
                    pv.get("volume_ratio", 1), pv.get("volume_std_ratio", 0), pv.get("volume_trend", 0),
                    pv.get("atr_ratio", 0), pv.get("high_low_ratio", 0), pv.get("std_dev_ratio", 0),
                ]
                return {
                    "status": "success",
                    "symbol": symbol,
                    "snapshot": snapshot,
                    "feature_vector": feature_vector,
                }
            return {"status": "success", "symbol": symbol, "snapshot": None, "feature_vector": None}
        except Exception as e:
            logger.error(f"Error getting features via db_logger: {e}")
    
    # Fallback: direct DB query
    dsn = os.environ.get("POSTGRES_DSN", "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev")
    if "asyncpg" not in dsn:
        dsn = dsn.replace("postgresql://", "postgresql+asyncpg://")
    
    try:
        engine = create_async_engine(dsn, pool_size=1, max_overflow=0)
        
        async with engine.begin() as conn:
            row = (await conn.execute(
                sa_text("""
                    SELECT snapshot_id, symbol, ts, mode, strategy_id,
                           action, reason, gate_name, confidence, price,
                           price_vol_json, news_json, regime_json,
                           would_entry_price, would_stop_loss, would_take_profit,
                           would_size_shares, would_size_mult, extra_json
                    FROM bot_decision_snapshots
                    WHERE symbol = :symbol
                    ORDER BY ts DESC
                    LIMIT 1
                """),
                {"symbol": symbol},
            )).mappings().first()
        
        await engine.dispose()
        
        if not row:
            return {"status": "success", "symbol": symbol, "snapshot": None, "feature_vector": None}
        
        import json
        pv = json.loads(row["price_vol_json"]) if row["price_vol_json"] else {}
        feature_vector = [
            pv.get("returns_1", 0), pv.get("returns_5", 0), pv.get("returns_20", 0),
            pv.get("price_vs_sma20", 0), pv.get("price_vs_sma50", 0),
            pv.get("rsi", 0.5), pv.get("macd", 0), pv.get("macd_signal", 0), pv.get("macd_hist", 0),
            pv.get("bb_position", 0.5), pv.get("bb_width", 0),
            pv.get("volume_ratio", 1), pv.get("volume_std_ratio", 0), pv.get("volume_trend", 0),
            pv.get("atr_ratio", 0), pv.get("high_low_ratio", 0), pv.get("std_dev_ratio", 0),
        ]
        
        snapshot = {
            "snapshot_id": row["snapshot_id"],
            "symbol": row["symbol"],
            "ts": row["ts"].isoformat() if hasattr(row["ts"], "isoformat") else row["ts"],
            "mode": row["mode"],
            "strategy_id": row["strategy_id"],
            "action": row["action"],
            "reason": row["reason"],
            "gate_name": row["gate_name"],
            "confidence": row["confidence"],
            "price_vol": pv,
            "news": json.loads(row["news_json"]) if row["news_json"] else {},
            "regime": json.loads(row["regime_json"]) if row["regime_json"] else {},
            "would_entry_price": row["would_entry_price"],
            "would_stop_loss": row["would_stop_loss"],
            "would_take_profit": row["would_take_profit"],
            "would_size_shares": row["would_size_shares"],
            "would_size_mult": row["would_size_mult"],
            "extra": json.loads(row["extra_json"]) if row["extra_json"] else {},
        }
        
        return {
            "status": "success",
            "symbol": symbol,
            "snapshot": snapshot,
            "feature_vector": feature_vector,
        }
        
    except Exception as e:
        logger.error(f"Error querying features: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/reset-pnl")
async def reset_pnl():
    """Reset P&L values when they're incorrect"""
    global trading_engine

    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}

    # Reset Schwab P&L
    old_schwab = trading_engine.risk_manager.schwab_daily_pnl

    # Reset to 0 temporarily
    trading_engine.risk_manager.schwab_daily_pnl = 0

    # Try to get fresh P&L from Schwab
    new_pnl = 0
    if trading_engine.schwab_client:
        try:
            account_info = await trading_engine._get_real_account_info()
            if account_info:
                trading_engine.risk_manager.sync_with_schwab_data(account_info)
                new_pnl = account_info.get('day_pnl', 0)
        except Exception as e:
            logger.error(f"Error refreshing P&L: {e}")

    return {
        "status": "success",
        "old_pnl": old_schwab,
        "new_pnl": new_pnl,
        "message": f"P&L reset from ${old_schwab:.2f} to ${new_pnl:.2f}"
    }

@app.post("/api/request-close-position")
async def request_close_position(request: dict):
    """Handle manual position close request"""
    global trading_engine

    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}

    symbol = request.get('symbol')
    position_type = request.get('position_type', 'real')

    # Add commentary about manual close request
    trading_engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.DECISION,
        symbol=symbol,
        title=f"\U0001f527 Manual Close Requested",
        message=f"User requested to close {symbol} position",
        importance=8
    ))

    # Trigger the close
    await trading_engine.close_position_manually(symbol, position_type)

    return {"status": "success", "message": f"Close process initiated for {symbol}"}


# Concurrency guard so two operators can't fire close-all simultaneously
# (race would duplicate audit entries and double-tap the same symbol).
_close_all_in_flight = False


@app.post("/api/emergency/close-all")
async def emergency_close_all(request: dict):
    """EMERGENCY: close every open position in a mode.

    Request body (all fields required):
        mode:    "simulation" | "live" | "all"
        confirm: must be the literal string "YES"

    Returns a per-symbol report with closed[] and failed[]. Safe to
    re-call — the second call will find no positions left.

    Safety:
      * mode has no default; the caller MUST pass it explicitly.
      * confirm="YES" prevents accidental curl/fat-finger invocations.
      * A module-level flag blocks concurrent calls.
      * Every close goes through _close_position_with_commentary with
        reason="emergency_close_all", so audit logs, trade DB rows, and
        brain learning all see the same path as any other close.
      * Bearer auth (TRADING_API_KEY) is enforced by the existing
        /api/* middleware — no extra check needed here.
    """
    global trading_engine, _close_all_in_flight

    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}

    mode = request.get("mode")
    confirm = request.get("confirm")

    if mode not in ("simulation", "live", "all"):
        return {
            "status": "error",
            "message": "mode required: must be 'simulation', 'live', or 'all'",
        }
    if confirm != "YES":
        return {
            "status": "error",
            "message": 'confirm required: must be the literal string "YES"',
        }

    if _close_all_in_flight:
        return {"status": "error", "message": "another close-all is already running"}

    _close_all_in_flight = True
    try:
        targets: list[tuple[str, str, object]] = []  # (symbol, mode_tag, position)
        if mode in ("simulation", "all"):
            for sym, pos in list(trading_engine.simulated_positions.items()):
                if pos is not None and getattr(pos, "quantity", 0) > 0:
                    targets.append((sym, "simulation", pos))
        if mode in ("live", "all"):
            for sym, pos in list(trading_engine.positions.items()):
                if pos is None or getattr(pos, "quantity", 0) <= 0:
                    continue
                # External/manually-managed live positions are off-limits
                # even in an emergency — the user owns those, not the bot.
                if getattr(pos, "is_external", False) or getattr(pos, "is_manually_managed", False):
                    continue
                targets.append((sym, "live", pos))

        # Front-log the whole batch so the decision trail shows intent
        # even if some individual closes fail.
        trading_engine._audit(
            "emergency", None, "close_all_start",
            "manual_override",
            mode=mode,
            target_count=len(targets),
            symbols=",".join(t[0] for t in targets) or "none",
        )
        trading_engine.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.WARNING,
            symbol=None,
            title="\U0001f6a8 Emergency Close-All",
            message=f"Closing {len(targets)} position(s) in mode={mode}",
            importance=10,
        ))

        closed: list[dict] = []
        failed: list[dict] = []
        total_pnl = 0.0

        for sym, mode_tag, pos in targets:
            try:
                pnl_before = float(getattr(pos, "unrealized_pnl", 0) or 0)
                await trading_engine._close_position_with_commentary(
                    pos, reason="emergency_close_all"
                )
                closed.append({
                    "symbol": sym,
                    "mode": mode_tag,
                    "pnl_at_close": round(pnl_before, 2),
                    "reason": "emergency_close_all",
                })
                total_pnl += pnl_before
            except Exception as exc:
                logger.error("emergency_close_failed symbol=%s mode=%s err=%s",
                             sym, mode_tag, exc)
                failed.append({"symbol": sym, "mode": mode_tag, "error": str(exc)})

        trading_engine._audit(
            "emergency", None, "close_all_done",
            "manual_override",
            mode=mode,
            closed_count=len(closed),
            failed_count=len(failed),
            total_pnl=round(total_pnl, 2),
        )

        return {
            "status": "success" if not failed else "partial",
            "mode": mode,
            "closed": closed,
            "failed": failed,
            "total_pnl": round(total_pnl, 2),
        }
    finally:
        _close_all_in_flight = False


@app.get("/api/historical-data/{symbol}")
async def get_historical_data(symbol: str, frequency: int = 5, period: int = 1):
    """v-ui-price-chart-2026-04-29: minimal OHLC feed for the dashboard
    price chart. Reads from the running engine's data_provider so we don't
    add a new dependency.

    frequency=5 default → 5-minute bars
    period=1 default    → 1 day's worth
    Returns an array of {time, open, high, low, close, volume} dicts.
    """
    import re as _re
    if not _re.fullmatch(r"[A-Z0-9]{1,8}", symbol.upper()):
        return {"status": "error", "message": "invalid symbol"}
    if not trading_engine or not trading_engine.data_provider:
        return {"status": "error", "message": "data provider unavailable"}
    try:
        df = trading_engine.data_provider.get_market_data(
            symbol.upper(),
            frequency_type='minute',
            frequency=frequency,
            period_type='day',
            period=period,
        )
        if df is None or df.empty:
            return {"status": "success", "symbol": symbol.upper(), "bars": []}
        # Normalize column names (some providers Title-case)
        cols = {c.lower(): c for c in df.columns}
        get = lambda c: df[cols.get(c, c)] if (cols.get(c, c) in df.columns) else None
        bars = []
        opens = get('open'); highs = get('high'); lows = get('low')
        closes = get('close'); vols = get('volume')
        idx = df.index
        for i in range(len(df)):
            ts = idx[i]
            try:
                t_iso = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
            except Exception:
                t_iso = str(ts)
            bars.append({
                "time":  t_iso,
                "open":  float(opens.iloc[i])  if opens  is not None else None,
                "high":  float(highs.iloc[i])  if highs  is not None else None,
                "low":   float(lows.iloc[i])   if lows   is not None else None,
                "close": float(closes.iloc[i]) if closes is not None else None,
                "volume": int(vols.iloc[i])    if vols   is not None else 0,
            })
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "frequency_min": frequency,
            "period_days": period,
            "bars": bars[-200:],  # cap at last 200 points
        }
    except Exception as exc:
        logger.warning(f"historical-data error for {symbol}: {exc}")
        return {"status": "error", "message": str(exc)}


_logo_inflight: set = set()  # symbols currently being fetched in background

# v-logo-transparent-2026-04-30: smallest valid 1×1 transparent PNG (67 bytes).
# Used as a placeholder when a logo can't be fetched, so the browser caches a
# 200 response instead of repeatedly re-requesting the 404. Generated from:
#   PIL.Image.new("RGBA",(1,1),(0,0,0,0)).save("t.png","PNG")
import base64 as _b64_logo
_TRANSPARENT_PNG_1x1 = _b64_logo.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


async def _fetch_logo_background(symbol: str, cache_path):
    """v-logo-async-bg-2026-04-30: fetch one logo in the background.

    Runs out-of-band of any HTTP request so a 5-10s upstream wait can
    never freeze the dashboard. On success, writes the PNG to cache.
    On failure, writes a 0-byte sentinel so we don't keep retrying.
    """
    import aiohttp

    fetched = False
    try:
        # 1) FMP (free, no auth)
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=4)
            ) as sess:
                async with sess.get(
                    f"https://financialmodelingprep.com/image-stock/{symbol}.png"
                ) as r:
                    if r.status == 200:
                        content = await r.read()
                        ctype = r.headers.get("content-type", "")
                        if ctype.startswith("image/") and len(content) > 100:
                            cache_path.write_bytes(content)
                            fetched = True
        except Exception as exc:
            logger.debug(f"bg fmp logo fetch failed for {symbol}: {exc}")

        # 2) Alpaca (paid; 403 on free plan)
        if not fetched:
            api_key = os.getenv("ALPACA_API_KEY", "")
            api_sec = os.getenv("ALPACA_SECRET_KEY", "")
            if api_key and api_sec:
                try:
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=4)
                    ) as sess:
                        async with sess.get(
                            f"https://data.alpaca.markets/v1beta1/logos/{symbol}",
                            headers={
                                "APCA-API-KEY-ID": api_key,
                                "APCA-API-SECRET-KEY": api_sec,
                                "accept": "image/*",
                            },
                            params={"placeholder": "true"},
                        ) as r:
                            if r.status == 200:
                                content = await r.read()
                                ctype = r.headers.get("content-type", "")
                                if ctype.startswith("image/") and len(content) > 100:
                                    cache_path.write_bytes(content)
                                    fetched = True
                except Exception as exc:
                    logger.debug(f"bg alpaca logo fetch failed for {symbol}: {exc}")

        # 3) Sentinel: 0-byte file = "tried, failed". 30-day TTL prevents
        # repeated retries of impossible symbols.
        if not fetched:
            try:
                cache_path.write_bytes(b"")
            except Exception:
                pass
    finally:
        _logo_inflight.discard(symbol)


@app.get("/logos/{symbol}")
async def get_logo(symbol: str):
    """v-logo-cache-2026-04-28: serve cached company logo for a symbol.

    v-logo-async-bg-2026-04-30: the previous version awaited two upstream
    HTTP fetches (FMP 5s + Alpaca 5s) inside the request handler. On a
    fresh dashboard load with 10-20 watchlist symbols, those 10-second
    waits queued up and froze the FastAPI event loop for 30-100+ seconds.
    Now: cache hit → serve instantly. Cache miss → return 404 with a
    Cache-Control header AND fire the upstream fetch as a background
    task. The next dashboard refresh (or any subsequent request) gets
    the cached PNG. The user sees a missing logo for one cycle instead
    of a frozen page.

    Mounted under /logos/* (not /api/*) so it bypasses the Bearer-token
    middleware — browsers can't easily attach Authorization headers to
    <img src=...> requests.

    Symbol validation: A-Z + digits only, max 8 chars. Prevents path-
    traversal style attacks via symbol param (e.g. ../../etc/passwd).
    """
    import asyncio as _asyncio
    import re
    import time
    from pathlib import Path
    from fastapi.responses import FileResponse, Response

    symbol = symbol.upper()
    if not re.fullmatch(r"[A-Z0-9]{1,8}", symbol):
        return Response(status_code=404)

    cache_dir = Path("static/logos")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol}.png"

    cache_exists = cache_path.exists()
    cache_stale = cache_exists and (
        (time.time() - cache_path.stat().st_mtime) > 30 * 86400
    )

    # Cache hit: return immediately. This is the >99% path once warmed.
    if cache_exists and not cache_stale:
        if cache_path.stat().st_size > 0:
            return FileResponse(
                cache_path,
                media_type="image/png",
                headers={"Cache-Control": "public, max-age=86400"},
            )
        # Sentinel (0-byte) = "tried, failed". v-logo-transparent-2026-04-30:
        # serve a 1×1 transparent PNG with HTTP 200 so the browser actually
        # caches it. Cache-Control on 404 is ignored by Chrome/Firefox by
        # default — they don't cache error responses regardless of header.
        # That meant the dashboard re-fetched NFLX/CRM/SNAP every WS tick
        # (~1Hz) for the entire session. Returning a real 200 PNG fixes it.
        return Response(
            content=_TRANSPARENT_PNG_1x1,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # Cache miss or stale: kick off background fetch (idempotent — only
    # one in-flight task per symbol) and respond with a transparent PNG
    # immediately. Browser caches it for a short window; next dashboard
    # reload picks up the real logo if the background fetch succeeded.
    if symbol not in _logo_inflight:
        _logo_inflight.add(symbol)
        _asyncio.create_task(_fetch_logo_background(symbol, cache_path))

    return Response(
        content=_TRANSPARENT_PNG_1x1,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=60"},  # 1-min retry window
    )


@app.get("/api/news-vetoes/report")
async def news_vetoes_report():
    """v-news-veto-tracker-2026-04-28: scorecard for the news-veto gate.

    Tells you whether vetoes were correct (saved a loss) or wrong (missed
    a winner). Pull this every session to gauge gate health.
    """
    import os
    from sqlalchemy import create_engine, text as sa_text
    dsn = os.environ.get(
        "POSTGRES_DSN",
        "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev",
    )
    try:
        eng = create_engine(dsn)
        with eng.connect() as conn:
            summary = conn.execute(sa_text("""
                SELECT
                  COALESCE(outcome, 'open') AS outcome,
                  COUNT(*)                  AS n,
                  ROUND(AVG(outcome_pnl_pct)::numeric, 3) AS avg_pnl_pct,
                  ROUND(SUM(outcome_pnl_pct)::numeric, 2) AS total_pnl_pct
                FROM bot_shadow_news_vetoes
                WHERE veto_time > NOW() - INTERVAL '14 days'
                GROUP BY outcome
                ORDER BY n DESC
            """)).mappings().all()
            recent = conn.execute(sa_text("""
                SELECT id, veto_time::text, symbol, side, veto_reason,
                       veto_source, cached_sentiment, fresh_count,
                       fresh_avg_sentiment, latest_age_min,
                       would_entry_price, would_stop_loss, would_take_profit,
                       outcome, outcome_pnl_pct, outcome_hit_target, outcome_hit_stop
                FROM bot_shadow_news_vetoes
                WHERE veto_time > NOW() - INTERVAL '3 days'
                ORDER BY veto_time DESC
                LIMIT 50
            """)).mappings().all()
        # v-fix-pool-leak-2026-09-10: dispose engine to release connection pool
        eng.dispose()
        # Compute headline metrics
        n_correct = sum(r['n'] for r in summary if r['outcome'] == 'correct_veto')
        n_missed  = sum(r['n'] for r in summary if r['outcome'] == 'missed_winner')
        n_total_resolved = n_correct + n_missed
        gate_accuracy = (
            round(n_correct / n_total_resolved, 3) if n_total_resolved > 0 else None
        )
        return {
            "status": "success",
            "summary": [dict(r) for r in summary],
            "headline": {
                "correct_vetoes": n_correct,
                "missed_winners": n_missed,
                "gate_accuracy_pct": gate_accuracy,
                "interpretation": (
                    "gate is saving more losses than it costs"
                    if gate_accuracy is not None and gate_accuracy >= 0.6
                    else "gate is rejecting too many winners — loosen thresholds"
                    if gate_accuracy is not None and gate_accuracy < 0.4
                    else "gate is roughly break-even — keep monitoring"
                ),
            },
            "recent_vetoes": [dict(r) for r in recent],
        }
    except Exception as e:
        logger.error(f"news_vetoes_report error: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/toggle-managed-by-bot")
async def toggle_managed_by_bot(request: dict):
    """v-managed-by-bot-2026-04-28: turn bot management on/off for a single position.

    Body: { "symbol": "<SYM>", "enabled": <true|false> (optional toggle if omitted) }

    When OFF, the bot stops applying:
      - breakeven-stop ratchet
      - 1R partial scale-out
      - ATR trailing stop
      - proactive exit on indicator flip
      - take-profit / hard-stop auto-execution
    The position is otherwise untouched (still tracked for P&L display).
    """
    global trading_engine
    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}

    symbol = (request.get('symbol') or '').upper()
    if not symbol:
        return {"status": "error", "message": "symbol required"}

    # Find the position in either dict
    pos = trading_engine.simulated_positions.get(symbol) or trading_engine.positions.get(symbol)
    if pos is None:
        return {"status": "error", "message": f"position {symbol} not found"}

    if 'enabled' in request:
        new_state = bool(request['enabled'])
    else:
        new_state = not getattr(pos, 'managed_by_bot', False)

    pos.managed_by_bot = new_state
    trading_engine._save_state()

    trading_engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.DECISION,
        symbol=symbol,
        title=f"{'🤖 Bot Management ON' if new_state else '🖐  Bot Management OFF'} — {symbol}",
        message=(
            f"Position will {'be managed' if new_state else 'no longer be managed'} "
            f"by the bot (stops, trail, breakeven, exits)."
        ),
        importance=8,
    ))

    return {"status": "success", "symbol": symbol, "managed_by_bot": new_state}


@app.post("/api/toggle-long-term")
async def toggle_long_term(request: dict):
    """Toggle long-term status for a position"""
    global trading_engine

    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}

    symbol = request.get('symbol')
    if not symbol:
        return {"status": "error", "message": "Symbol required"}

    # Toggle the long-term flag
    if symbol in trading_engine.positions:
        current_status = getattr(trading_engine.positions[symbol], 'is_long_term', False)
        trading_engine.positions[symbol].is_long_term = not current_status

        # Save state
        trading_engine._save_state()

        # Add commentary
        trading_engine.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.DECISION,
            symbol=symbol,
            title=f"\U0001f512 Long-Term Status {'Enabled' if not current_status else 'Disabled'}",
            message=f"{symbol} marked as {'long-term hold (protected from auto-closing)' if not current_status else 'regular position (can be auto-closed)'}",
            importance=7
        ))

        return {"status": "success", "is_long_term": not current_status}

    return {"status": "error", "message": "Position not found"}

@app.post("/api/toggle-close-mode")
async def toggle_close_mode(request: dict):
    """Toggle between manual and automatic position closing"""
    global trading_engine

    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}

    manual_only = request.get('manual_only', True)
    trading_engine.manual_close_only = manual_only

    trading_engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.DECISION,
        symbol=None,
        title=f"\u2699\ufe0f Close Mode Changed",
        message=f"Position closing: {'Manual only' if manual_only else 'Automatic (stop loss, targets)'}",
        importance=7
    ))

    return {"status": "success", "manual_only": manual_only}

@app.post("/api/toggle-ml-prediction")
async def toggle_ml_prediction(request: dict):
    """Toggle ML prediction"""
    global trading_engine

    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}

    from trading_bot_commentary_updated import config_manager
    enabled = request.get('enabled', True)
    config_manager.update('trading.ml_prediction_enabled', enabled)

    trading_engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.DECISION,
        symbol=None,
        title=f"\U0001f916 ML Prediction Changed",
        message=f"ML prediction has been {'enabled' if enabled else 'disabled'}",
        importance=8
    ))

    return {"status": "success", "enabled": enabled}

@app.post("/api/reset-brain")
async def reset_brain():
    if trading_engine:
        trading_engine.reset_brain_state()
        return {"status": "success", "message": "Brain state reset"}
    return {"status": "error", "message": "Engine not running"}

@app.get("/api/performance-metrics")
async def get_performance_metrics():
    """Get comprehensive performance metrics"""
    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}

    try:
        metrics = trading_engine.performance_analyzer.calculate_metrics(
            list(trading_engine.positions.values()),
            trading_engine.trade_history
        )

        return {
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {"status": "error", "message": f"Error calculating metrics: {str(e)}"}

@app.post("/api/update-config")
async def update_config(request: dict):
    """Update configuration values"""
    try:
        from trading_bot_commentary_updated import config_manager
        key = request.get('key')
        value = request.get('value')

        if not key or value is None:
            return {"status": "error", "message": "Key and value are required"}

        config_manager.update(key, value)

        return {
            "status": "success",
            "message": f"Configuration updated: {key} = {value}"
        }
    except Exception as e:
        logger.error(f"Config update error: {e}")
        return {"status": "error", "message": f"Config update failed: {str(e)}"}

@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    try:
        from trading_bot_commentary_updated import config_manager
        return {
            "status": "success",
            "config": config_manager.config
        }
    except Exception as e:
        logger.error(f"Config retrieval error: {e}")
        return {"status": "error", "message": f"Config retrieval failed: {str(e)}"}


@app.get("/api/advanced-analytics")
async def get_advanced_analytics():
    """Get advanced analytics and insights"""
    try:
        if not trading_engine:
            return {"status": "error", "message": "Trading engine not initialized"}

        # Get performance report
        performance_report = trading_engine.performance_analyzer.calculate_metrics(
            list(trading_engine.positions.values()),
            trading_engine.trade_history
        )

        # Get behavioral analysis
        behavioral_patterns = trading_engine.behavioral_analyzer.analyze_trading_patterns(
            trading_engine.trade_history
        )

        # Get behavioral recommendations
        behavioral_recommendations = trading_engine.behavioral_analyzer.get_behavioral_recommendations()

        # Get portfolio optimization data
        positions = list(trading_engine.positions.values())
        historical_data = {}  # This would be populated with actual data

        # NOTE: Portfolio optimization is not fully implemented in this version
        # portfolio_weights = trading_engine.portfolio_optimizer.optimize_weights(
        #     positions, historical_data
        # )

        return {
            "status": "success",
            "performance_metrics": performance_report,
            "behavioral_patterns": behavioral_patterns,
            "behavioral_recommendations": behavioral_recommendations,
            # "portfolio_optimization": {
            #     "current_weights": portfolio_weights,
            #     "recommendations": performance_report.get("recommendations", [])
            # },
            "emotional_state": trading_engine.behavioral_analyzer.emotional_state
        }
    except Exception as e:
        logger.error(f"Error getting advanced analytics: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/alternative-data/{symbol}")
async def get_alternative_data(symbol: str):
    """Get alternative data for a specific symbol"""
    try:
        if not trading_engine:
            return {"status": "error", "message": "Trading engine not initialized"}

        # Get options flow
        options_flow = await trading_engine.alternative_data_integrator.get_options_flow(symbol)

        # Get insider trading
        insider_trading = await trading_engine.alternative_data_integrator.get_insider_trading(symbol)

        # Get social sentiment
        social_sentiment = await trading_engine.alternative_data_integrator.get_social_sentiment(symbol)

        # Get economic calendar
        economic_calendar = await trading_engine.alternative_data_integrator.get_economic_calendar()

        # Analyze alternative signals
        alternative_signals = trading_engine.alternative_data_integrator.analyze_alternative_signals(symbol)

        return {
            "status": "success",
            "symbol": symbol,
            "options_flow": options_flow,
            "insider_trading": insider_trading,
            "social_sentiment": social_sentiment,
            "economic_calendar": economic_calendar,
            "alternative_signals": alternative_signals
        }
    except Exception as e:
        logger.error(f"Error getting alternative data for {symbol}: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/market-neutral-opportunities")
async def get_market_neutral_opportunities():
    """Get market neutral trading opportunities"""
    try:
        if not trading_engine:
            return {"status": "error", "message": "Trading engine not initialized"}

        # Get current symbols
        symbols = list(trading_engine.simulated_positions.keys()) + list(trading_engine.real_positions.keys())
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Default symbols

        # Get historical data (this would be populated with actual data)
        historical_data = {}

        # Find pairs trading opportunities
        pairs_opportunities = trading_engine.market_neutral_strategies.find_pairs_trading_opportunities(
            symbols, historical_data
        )

        # Calculate sector weights
        sector_weights = trading_engine.market_neutral_strategies.calculate_sector_weights(symbols)

        # Get statistical arbitrage signals
        stat_arb_signals = {}
        for symbol in symbols[:5]:  # Limit to first 5 symbols
            if symbol in historical_data:
                stat_arb_signals[symbol] = trading_engine.market_neutral_strategies.calculate_statistical_arbitrage_signals(
                    symbol, historical_data[symbol]
                )

        return {
            "status": "success",
            "pairs_trading_opportunities": pairs_opportunities,
            "sector_weights": sector_weights,
            "statistical_arbitrage_signals": stat_arb_signals
        }
    except Exception as e:
        logger.error(f"Error getting market neutral opportunities: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/data-quality/{symbol}")
async def get_data_quality(symbol: str):
    """Get data quality analysis for a symbol"""
    try:
        if not trading_engine:
            return {"status": "error", "message": "Trading engine not initialized"}

        # Get market data
        if trading_engine.data_provider:
            data = trading_engine.data_provider.get_market_data(symbol)

            # Validate data quality
            quality_report = trading_engine.data_validator.validate_market_data(symbol, data)

            return {
                "status": "success",
                "symbol": symbol,
                "data_quality": quality_report,
                "historical_quality": trading_engine.data_validator.data_quality_history.get(symbol, {})
            }
        else:
            return {"status": "error", "message": "Data provider not available"}
    except Exception as e:
        logger.error(f"Error getting data quality for {symbol}: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/optimize-portfolio")
async def optimize_portfolio():
    """Trigger portfolio optimization"""
    try:
        if not trading_engine:
            return {"status": "error", "message": "Trading engine not initialized"}

        positions = list(trading_engine.positions.values())
        historical_data = {}  # This would be populated with actual data

        # NOTE: Portfolio optimization not fully implemented
        # optimized_weights = trading_engine.portfolio_optimizer.optimize_weights(
        #     positions, historical_data
        # )

        return {
            "status": "success",
            # "optimized_weights": optimized_weights,
            "message": "Portfolio optimization completed"
        }
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/record-execution")
async def record_execution(request: dict):
    """Record trade execution for performance monitoring"""
    try:
        if not trading_engine:
            return {"status": "error", "message": "Trading engine not initialized"}

        # NOTE: performance_monitor not fully implemented
        # trading_engine.performance_monitor.record_execution(
        #     symbol=request["symbol"],
        #     expected_price=request["expected_price"],
        #     actual_price=request["actual_price"],
        #     signal_time=datetime.fromisoformat(request["signal_time"]),
        #     execution_time=datetime.fromisoformat(request["execution_time"]),
        #     order_size=request["order_size"]
        # )

        return {"status": "success", "message": "Execution recorded successfully"}
    except Exception as e:
        logger.error(f"Error recording execution: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    logger.info("""
    \u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
    \u2551        Trading Bot with Live Commentary - Educational Mode           \u2551
    \u2551     Learn How Professional Trading Algorithms Make Decisions         \u2551
    \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d
    """)

    port = int(os.environ.get("TRADING_BOT_PORT", "9000"))

    logger.info("Starting Trading Bot in Commentary Mode...")
    logger.info("The bot will explain its thinking process in real-time.")
    logger.info(f"Open http://localhost:{port} in your browser")
    logger.info("Press Ctrl+C or send SIGTERM to stop")

    def shutdown_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, shutting down gracefully...")
        if trading_engine:
            # v-shutdown-stops-engine-thread-2026-06-09: stop the engine
            # loop and wait for its thread BEFORE saving state. The old
            # handler exited the main thread while the engine kept
            # trading on a non-daemon thread — the process never died,
            # and the zombie's Schwab stream / token refreshes / state
            # writes corrupted the next instance's startup ("init hang").
            trading_engine.is_running = False
            if trading_thread is not None and trading_thread.is_alive():
                trading_thread.join(timeout=15.0)
                if trading_thread.is_alive():
                    # Engine still mid-loop: saving now would race its
                    # own periodic state writes (torn JSON). The
                    # freshest consistent snapshot is already on disk
                    # from those periodic saves. os._exit skips
                    # interpreter teardown — sys.exit would clear
                    # module globals under the still-running thread.
                    logger.critical(
                        "Engine thread did not stop within 15s — "
                        "skipping final state save (engine owns the "
                        "state files) and force-exiting."
                    )
                    os._exit(1)
                logger.info("Engine thread stopped cleanly")
            try:
                trading_engine._save_state()
                logger.info("Trading state saved")
            except Exception as e:
                logger.error(f"Failed to save trading state: {e}")

            try:
                trading_engine.brain.save_memories()
                logger.info("Brain memories saved")
            except Exception as e:
                logger.error(f"Failed to save brain memories: {e}")

            try:
                if hasattr(trading_engine, 'ml_predictor') and trading_engine.ml_predictor:
                    if hasattr(trading_engine.ml_predictor, 'save_model'):
                        trading_engine.ml_predictor.save_model()
                        buffer_size = len(getattr(trading_engine.ml_predictor, 'online_buffer', []))
                        logger.info(f"ML model saved (buffer: {buffer_size} samples)")
            except Exception as e:
                logger.error(f"Failed to save ML model: {e}")

        logger.info("Shutdown complete")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
