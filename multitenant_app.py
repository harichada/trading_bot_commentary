"""
Rudra Trading Engine — Multi-tenant Application

Clean FastAPI app with per-user isolation. Every endpoint is user-scoped
via Depends(). No global trader singleton. Users must store their own
broker credentials to trade.

Usage: python multitenant_app.py
Port:  MULTITENANT_PORT env var (default 8010)
"""

import asyncio
import json
import logging
import os
import time as _time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse, FileResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.staticfiles import StaticFiles

# ── Import core trading classes from gap_fade_app ───────────────────────────
from gap_fade_app import (
    GapFadeConfig, GapFadeEngine, GapScanner, GapFadeLiveTrader,
    GapCandidate, GapPosition, TradeRecord, OrderResult,
    get_price_db, alpaca_submit_and_confirm,
)

# ── Import multi-tenant modules ────────────────────────────────────────────
from auth import router as auth_router, auth_enabled, get_current_user
from user_management import (
    router as user_router, admin_router,
    get_user_repo, get_cred_mgr, get_config_mgr,
    VALID_CONFIG_FIELDS,
)
from user_engine import (
    UserEngineManager, ProfitCapChecker, resolve_user_id,
    get_trader_for_request, init_engine_manager, get_engine_manager,
    NO_ENGINE_STATE, MAX_ENGINES,
)

logger = logging.getLogger("RudraMultitenant")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

_APP_START = _time.time()
_APP_VERSION = "2.0.0"

# ═══════════════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Rudra Trading Engine", version=_APP_VERSION)

# Session middleware (required by authlib OAuth)
_session_secret = os.environ.get("AUTH_JWT_SECRET", os.urandom(32).hex())
app.add_middleware(SessionMiddleware, secret_key=_session_secret)

# Mount auth and user management routers
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(admin_router)

# ═══════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════

_engine_mgr: Optional[UserEngineManager] = None


@app.on_event("startup")
async def startup():
    global _engine_mgr
    # No global trader — each user gets their own engine
    _engine_mgr = init_engine_manager(global_trader=None)
    logger.info("Multi-tenant engine pool ready (max %d engines)", MAX_ENGINES)


@app.on_event("shutdown")
async def shutdown():
    if _engine_mgr:
        active = _engine_mgr.get_all_active()
        for uid in list(active.keys()):
            await _engine_mgr.remove_trader(uid)
        logger.info("All user engines stopped")


# ═══════════════════════════════════════════════════════════════════════════
# DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════

async def require_auth(request: Request) -> dict:
    """Require authenticated user. Returns JWT payload or raises 401."""
    if not auth_enabled():
        raise _api_error(401, "Authentication is required for multi-tenant mode. Set AUTH_JWT_SECRET.")
    user = get_current_user(request)
    if not user:
        raise _api_error(401, "Not authenticated. Please log in.")
    return user


async def get_trader(request: Request) -> Optional[GapFadeLiveTrader]:
    """Resolve per-user trading engine. Returns None if no credentials stored."""
    return await get_trader_for_request(request)


async def require_trader(request: Request) -> GapFadeLiveTrader:
    """Require a provisioned trader. Raises 400 if no credentials."""
    trader = await get_trader_for_request(request)
    if trader is None:
        raise _api_error(400, "No broker credentials configured. Go to Account → Add Credential to start trading.")
    return trader


class _ApiError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message


def _api_error(status: int, message: str) -> _ApiError:
    return _ApiError(status, message)


@app.exception_handler(_ApiError)
async def api_error_handler(request: Request, exc: _ApiError):
    return JSONResponse({"error": exc.message, "needs_setup": exc.status == 400}, exc.status)


# ═══════════════════════════════════════════════════════════════════════════
# AUTH MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════

_AUTH_EXEMPT = frozenset({"/api/health", "/api/auth", "/docs", "/openapi.json", "/ws"})


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path.rstrip("/")

    # Allow exempt paths, static assets, SPA routes
    if (path in _AUTH_EXEMPT
            or path.startswith("/api/auth")
            or path.startswith("/assets")
            or path.startswith("/favicon")
            or not path.startswith("/api")):
        return await call_next(request)

    # Require auth on all /api/* paths
    if auth_enabled():
        user = get_current_user(request)
        if not user:
            return JSONResponse({"error": "Not authenticated"}, 401)

        # Viewer role: block mutations
        role = user.get("role", "viewer")
        if role != "admin" and request.method in ("POST", "PUT", "DELETE", "PATCH"):
            # Allow user management endpoints (credentials, config, profile)
            if not (path.startswith("/api/users") or path.startswith("/api/admin")):
                return JSONResponse({"error": "Read-only access. Contact admin."}, 403)

    return await call_next(request)


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH & STATE
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health(request: Request):
    trader = await get_trader(request)
    return {
        "status": "ok",
        "version": _APP_VERSION,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "uptime_seconds": int(_time.time() - _APP_START),
        "trader_status": trader.status if trader else "not_configured",
        "equity": round(trader.engine.equity, 2) if trader else 0,
        "daily_pnl": round(trader.engine.daily_stats.pnl, 2) if trader else 0,
        "open_positions": len(trader.engine.positions) if trader else 0,
        "trading_halted": trader._trading_halted if trader else False,
        "needs_setup": trader is None,
    }


@app.get("/api/state")
async def get_state(request: Request, user: dict = Depends(require_auth)):
    trader = await get_trader(request)
    if trader is None:
        return NO_ENGINE_STATE
    state = trader.get_state()

    # Enrich positions with broker prices
    if state.get("positions"):
        try:
            broker_positions = await trader._broker_get_positions()
            broker_map = {}
            for bp in broker_positions:
                sym = bp.get("symbol", "")
                if sym:
                    broker_map[sym] = {
                        "current_price": float(bp.get("current_price", 0)),
                        "unrealized_pl": float(bp.get("unrealized_pl", 0)),
                    }
            total_unrealized = 0.0
            for sym, pos in state["positions"].items():
                bp = broker_map.get(sym, {})
                current = bp.get("current_price", 0)
                if current > 0:
                    pos["current_price"] = current
                    entry = pos.get("entry_fill_price") or pos.get("entry_price", 0)
                    shares = pos.get("remaining_shares", pos.get("shares", 0))
                    direction = pos.get("direction", "short")
                    unrealized = (current - entry) * shares if direction == "long" else (entry - current) * shares
                    pos["pnl"] = round(unrealized, 2)
                    pos["pnl_pct"] = unrealized / (entry * shares) if entry * shares > 0 else 0
                    total_unrealized += unrealized
            state["unrealized_pnl"] = round(total_unrealized, 2)

            try:
                account = await trader._broker_get_account()
                if account:
                    state["equity"] = float(account.get("equity", state["equity"]))
            except Exception:
                pass
        except Exception as e:
            logger.debug("State enrichment failed: %s", e)

    return state


# ═══════════════════════════════════════════════════════════════════════════
# TRADING ACTIONS
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/start")
async def start_trading(trader: GapFadeLiveTrader = Depends(require_trader)):
    await trader.start()
    return {"status": trader.status}


@app.post("/api/stop")
async def stop_trading(trader: GapFadeLiveTrader = Depends(require_trader)):
    await trader.stop()
    return {"status": trader.status}


@app.post("/api/pause")
async def pause_trading(trader: GapFadeLiveTrader = Depends(require_trader)):
    await trader.pause()
    return {"status": trader.status}


@app.post("/api/resume")
async def resume_trading(trader: GapFadeLiveTrader = Depends(require_trader)):
    await trader.resume()
    return {"status": trader.status}


@app.post("/api/scan")
async def run_scan(trader: GapFadeLiveTrader = Depends(require_trader)):
    scanner = GapScanner(trader.config)
    candidates, universe_size = await asyncio.to_thread(scanner.scan_premarket)
    trader.candidates = candidates
    trader.last_scan_time = __import__("datetime").datetime.now(
        __import__("zoneinfo").ZoneInfo("America/New_York")
    ).strftime("%H:%M:%S")
    return {
        "candidates": [asdict(c) for c in candidates[:15]],
        "scan_time": trader.last_scan_time,
        "count": len(candidates),
        "universe_size": universe_size,
    }


@app.post("/api/enter")
async def enter_positions(request: Request, trader: GapFadeLiveTrader = Depends(require_trader)):
    if trader.status == "stopped":
        return JSONResponse({"error": "Trader is stopped. Start it first."}, 400)
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    llm_symbols = body.get("symbols")
    size_mult = body.get("size_mult", 1.0)
    if not trader.candidates:
        await trader._run_scan()
    await trader._enter_positions(llm_symbols=llm_symbols, size_mult=size_mult)
    return {
        "status": trader.status,
        "positions": {s: asdict(p) for s, p in trader.engine.positions.items()},
    }


@app.post("/api/reset")
async def reset_trader(trader: GapFadeLiveTrader = Depends(require_trader)):
    await trader.stop()
    trader.engine = GapFadeEngine(trader.config)
    trader.candidates = []
    trader.messages = []
    return {"status": "reset", "equity": trader.config.initial_capital}


# ═══════════════════════════════════════════════════════════════════════════
# DATA ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/trades")
async def get_trades(request: Request, user: dict = Depends(require_auth)):
    trader = await get_trader(request)
    if trader is None:
        return {"trades": [], "today": [], "total_trades": 0, "needs_setup": True}
    return {
        "trades": [asdict(t) for t in trader.engine.all_trade_log[-200:]],
        "today": [asdict(t) for t in trader.engine.trade_log],
        "total_trades": len(trader.engine.all_trade_log),
    }


@app.get("/api/trades/history")
async def get_trades_history(
    request: Request,
    start: str = None, end: str = None, symbol: str = None,
    limit: int = 100, offset: int = 0,
    user: dict = Depends(require_auth),
):
    user_id = resolve_user_id(request)
    db = get_price_db()
    limit = max(1, min(limit, 1000))
    offset = max(0, offset)
    return db.query_trades(start_date=start, end_date=end, symbol=symbol,
                           limit=limit, offset=offset, user_id=user_id)


@app.get("/api/metrics")
async def get_metrics(request: Request, user: dict = Depends(require_auth)):
    trader = await get_trader(request)
    if trader is None:
        return {"total_trades": 0, "needs_setup": True}
    return trader.engine.get_metrics()


@app.get("/api/equity_curve")
async def get_equity_curve(request: Request, user: dict = Depends(require_auth)):
    trader = await get_trader(request)
    if trader is None:
        return {"dates": [], "equity": [], "drawdown": []}
    trades = trader.engine.all_trade_log
    if not trades:
        return {"dates": [], "equity": [], "drawdown": []}
    dates, equity_vals, dd_vals = [], [], []
    eq = trader.config.initial_capital
    peak = eq
    for t in trades:
        eq += float(t.pnl)
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0
        dates.append(t.exit_time if hasattr(t, "exit_time") else "")
        equity_vals.append(round(eq, 2))
        dd_vals.append(round(dd, 4))
    return {"dates": dates, "equity": equity_vals, "drawdown": dd_vals}


@app.get("/api/scanner")
async def get_candidates(request: Request, user: dict = Depends(require_auth)):
    trader = await get_trader(request)
    if trader is None:
        return {"candidates": [], "needs_setup": True}
    return {"candidates": [asdict(c) for c in trader.candidates[:20]]}


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG & STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/config")
async def get_config(request: Request, user: dict = Depends(require_auth)):
    trader = await get_trader(request)
    if trader is None:
        return {"config": asdict(GapFadeConfig()), "needs_setup": True}
    return {"config": asdict(trader.config)}


@app.post("/api/config")
async def update_config(request: Request, user: dict = Depends(require_auth)):
    trader = await get_trader(request)
    body = await request.json()

    # Validate fields
    invalid = set(body.keys()) - VALID_CONFIG_FIELDS
    if invalid:
        return JSONResponse({"error": f"Invalid fields: {', '.join(sorted(invalid))}"}, 400)

    # Persist to DB
    user_id = resolve_user_id(request)
    if user_id:
        try:
            get_config_mgr().update_config(user_id, body)
        except Exception as e:
            logger.warning("Config persist failed: %s", e)

    # Apply to live trader if running
    if trader:
        for k, v in body.items():
            if hasattr(trader.config, k):
                setattr(trader.config, k, v)
        return {"config": asdict(trader.config), "applied": True}

    return {"config": body, "persisted": True, "needs_setup": True}


@app.get("/api/config/schema")
async def config_schema(user: dict = Depends(require_auth)):
    return {"fields": sorted(VALID_CONFIG_FIELDS)}


@app.get("/api/strategies")
async def list_strategies(request: Request, user: dict = Depends(require_auth)):
    trader = await get_trader(request)
    try:
        from gap_fade_strategies import GapFadeStrategyRegistry
        strategies = GapFadeStrategyRegistry.list_strategies()
    except Exception:
        strategies = [{"id": "classic_gap_fade", "name": "Classic Gap Fade", "description": "Default gap fade strategy"}]
    active = trader.config.active_strategy if trader else "classic_gap_fade"
    return {"strategies": strategies, "active": active}


# ═══════════════════════════════════════════════════════════════════════════
# TRACKER (BROKER POSITIONS & ORDERS)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/tracker/positions")
async def tracker_positions(request: Request, user: dict = Depends(require_auth)):
    trader = await get_trader(request)
    if trader is None:
        return {"positions": [], "equity": 0, "buying_power": 0, "needs_setup": True}
    positions = await trader._broker_get_positions()
    account = await trader._broker_get_account()
    return {
        "positions": positions or [],
        "equity": float(account.get("equity", 0)) if account else 0,
        "buying_power": float(account.get("buying_power", 0)) if account else 0,
    }


@app.post("/api/tracker/order")
async def tracker_order(request: Request, trader: GapFadeLiveTrader = Depends(require_trader)):
    body = await request.json()
    symbol = (body.get("symbol") or "").upper().strip()
    side = body.get("side", "")
    qty = int(body.get("qty", 0))

    if not symbol:
        return JSONResponse({"error": "Symbol required"}, 400)
    if side not in ("buy", "sell"):
        return JSONResponse({"error": "Side must be buy or sell"}, 400)
    if qty <= 0:
        return JSONResponse({"error": "Qty must be positive"}, 400)

    token = trader._activate_broker_context()
    try:
        result = await alpaca_submit_and_confirm(symbol, qty, side)
    finally:
        trader._deactivate_broker_context(token)

    return {
        "status": result.status,
        "order_id": result.order_id,
        "fill_price": result.fill_price,
        "error": result.error,
    }


@app.get("/api/account")
async def get_account(request: Request, user: dict = Depends(require_auth)):
    trader = await get_trader(request)
    if trader is None:
        return {"equity": 0, "buying_power": 0, "cash": 0, "needs_setup": True}
    account = await trader._broker_get_account()
    if account:
        return {
            "equity": account.get("equity"),
            "buying_power": account.get("buying_power"),
            "cash": account.get("cash"),
            "portfolio_value": account.get("portfolio_value"),
        }
    return {"error": "Could not fetch account"}


# ═══════════════════════════════════════════════════════════════════════════
# WEBSOCKET (user-scoped)
# ═══════════════════════════════════════════════════════════════════════════

_ws_connections: dict[str, list[WebSocket]] = {}


async def broadcast_to_user(user_id: str, msg: dict):
    """Send message to all WebSocket connections for a specific user."""
    sockets = _ws_connections.get(user_id, [])
    dead = []
    for ws in sockets:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        sockets.remove(ws)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Authenticate via cookie
    user = get_current_user(websocket) if auth_enabled() else None
    user_id = user.get("user_id", "") if user else ""

    if user_id not in _ws_connections:
        _ws_connections[user_id] = []
    _ws_connections[user_id].append(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        if user_id in _ws_connections:
            try:
                _ws_connections[user_id].remove(websocket)
            except ValueError:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# DB ENDPOINTS (shared, not per-user)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/db/stats")
async def db_stats(user: dict = Depends(require_auth)):
    db = get_price_db()
    try:
        return db.get_stats()
    except Exception:
        return {"total_rows": 0, "symbol_count": 0, "size_mb": 0}


# ═══════════════════════════════════════════════════════════════════════════
# FRONTEND SPA
# ═══════════════════════════════════════════════════════════════════════════

_FRONTEND_DIR = Path(__file__).parent / "frontend-v2-dist"

if _FRONTEND_DIR.is_dir():
    _assets_dir = _FRONTEND_DIR / "assets"
    if _assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="spa-assets")

    @app.get("/favicon.svg")
    async def favicon():
        return FileResponse(str(_FRONTEND_DIR / "favicon.svg"))

    @app.get("/")
    async def index():
        return FileResponse(str(_FRONTEND_DIR / "index.html"))

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = _FRONTEND_DIR / full_path
        if full_path and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_FRONTEND_DIR / "index.html"))

    logger.info("Frontend SPA serving from %s", _FRONTEND_DIR)
else:
    @app.get("/")
    async def index():
        return JSONResponse({"message": "Rudra Multi-tenant API", "docs": "/docs"})
    logger.info("No frontend-v2-dist/ — API-only mode")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("MULTITENANT_PORT", "8010"))
    bind = "0.0.0.0" if os.environ.get("GAP_FADE_BIND_ALL", "") == "1" else "127.0.0.1"

    print(f"\n  Rudra Trading Engine v{_APP_VERSION} (Multi-tenant)")
    print(f"  Port:     {bind}:{port}")
    print(f"  Auth:     {'ENABLED' if auth_enabled() else 'DISABLED'}")
    print(f"  Engines:  max {MAX_ENGINES}")
    print(f"  Frontend: {'READY' if _FRONTEND_DIR.is_dir() else 'NOT BUILT'}")
    print()

    uvicorn.run(app, host=bind, port=port, log_level="info")
