"""
FastAPI Application

Main API application with WebSocket support and professional dashboard.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Singleton app instance
_app: Optional[FastAPI] = None


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        data = json.dumps(message)
        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_text(data)
            except Exception:
                disconnected.add(connection)

        self.active_connections -= disconnected


# Connection manager instance
manager = ConnectionManager()


# Request/Response models
class StartRequest(BaseModel):
    mode: Optional[str] = None


class OrderRequest(BaseModel):
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    order_type: str = 'market'
    price: Optional[float] = None


class SettingUpdate(BaseModel):
    key: str
    value: Any


def create_app(
    engine=None,
    title: str = "Trading Bot API",
    version: str = "2.0.0"
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        engine: Trading engine instance (optional, can be set later)
        title: API title
        version: API version

    Returns:
        FastAPI application
    """
    global _app

    app = FastAPI(
        title=title,
        version=version,
        description="Professional Trading Bot API with real-time WebSocket updates"
    )

    # Store engine reference
    app.state.engine = engine

    # Static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ==================== Dashboard Routes ====================

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Serve the main dashboard"""
        dashboard_path = static_dir / "dashboard.html"
        if dashboard_path.exists():
            return FileResponse(dashboard_path)
        return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)

    # ==================== WebSocket ====================

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates"""
        await manager.connect(websocket)
        try:
            # Send initial state
            if app.state.engine:
                await websocket.send_json({
                    "type": "state_update",
                    "data": get_engine_status(app.state.engine)
                })

            while True:
                # Keep connection alive and handle incoming messages
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=30.0
                    )
                    # Handle client messages if needed
                    message = json.loads(data)
                    await handle_client_message(websocket, message, app.state.engine)
                except asyncio.TimeoutError:
                    # Send ping to keep alive
                    await websocket.send_json({"type": "ping"})

        except WebSocketDisconnect:
            manager.disconnect(websocket)

    # ==================== Status Endpoints ====================

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": version
        }

    @app.get("/api/status")
    async def get_status():
        """Get current bot status"""
        if not app.state.engine:
            return {
                "is_running": False,
                "mode": "simulation",
                "message": "Engine not initialized"
            }
        return get_engine_status(app.state.engine)

    # ==================== Control Endpoints ====================

    @app.post("/api/start")
    async def start_bot(request: StartRequest = None):
        """Start the trading bot"""
        if not app.state.engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")

        if app.state.engine.is_running:
            return {"status": "already_running"}

        success = await app.state.engine.start()
        if success:
            await broadcast_state_update(app.state.engine)
            return {"status": "started"}
        raise HTTPException(status_code=500, detail="Failed to start engine")

    @app.post("/api/stop")
    async def stop_bot():
        """Stop the trading bot"""
        if not app.state.engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")

        await app.state.engine.stop()
        await broadcast_state_update(app.state.engine)
        return {"status": "stopped"}

    @app.post("/api/analyze")
    async def run_analysis():
        """Trigger manual analysis"""
        if not app.state.engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")

        # This would trigger an analysis cycle
        return {"status": "analysis_triggered"}

    # ==================== Portfolio Endpoints ====================

    @app.get("/api/portfolio")
    async def get_portfolio():
        """Get portfolio summary"""
        if not app.state.engine:
            return {"error": "Engine not initialized"}

        return app.state.engine.get_portfolio_summary()

    @app.get("/api/positions")
    async def get_positions():
        """Get all positions"""
        if not app.state.engine or not app.state.engine.state:
            return {"positions": {}}

        return {
            "positions": {
                k: v.to_dict()
                for k, v in app.state.engine.state.portfolio.positions.items()
            }
        }

    @app.post("/api/positions/{symbol}/close")
    async def close_position(symbol: str):
        """Close a specific position"""
        if not app.state.engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")

        success = await app.state.engine.manual_sell(symbol)
        if success:
            await broadcast_state_update(app.state.engine)
            return {"status": "closing", "symbol": symbol}
        raise HTTPException(status_code=400, detail=f"Failed to close {symbol}")

    @app.post("/api/close-all")
    async def close_all_positions():
        """Close all positions"""
        if not app.state.engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")

        await app.state.engine.close_all_positions()
        await broadcast_state_update(app.state.engine)
        return {"status": "closing_all"}

    # ==================== Order Endpoints ====================

    @app.post("/api/orders")
    async def submit_order(order: OrderRequest):
        """Submit a new order"""
        if not app.state.engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")

        if order.side.lower() == 'buy':
            success = await app.state.engine.manual_buy(order.symbol, order.quantity)
        else:
            success = await app.state.engine.manual_sell(order.symbol, order.quantity)

        if success:
            await broadcast_state_update(app.state.engine)
            return {"status": "submitted", "order": order.dict()}
        raise HTTPException(status_code=400, detail="Failed to submit order")

    @app.get("/api/orders")
    async def get_orders():
        """Get pending orders"""
        if not app.state.engine or not app.state.engine.state:
            return {"orders": {}}

        return {
            "orders": {
                k: v.to_dict()
                for k, v in app.state.engine.state.portfolio.pending_orders.items()
            }
        }

    # ==================== Strategy Endpoints ====================

    @app.get("/api/strategies")
    async def get_strategies():
        """Get strategy status"""
        if not app.state.engine or not app.state.engine._strategy_manager:
            return {"strategies": []}

        return app.state.engine._strategy_manager.to_dict()

    @app.post("/api/strategies/{name}/enable")
    async def enable_strategy(name: str):
        """Enable a strategy"""
        if not app.state.engine or not app.state.engine._strategy_manager:
            raise HTTPException(status_code=500, detail="Strategy manager not available")

        app.state.engine._strategy_manager.enable_strategy(name)
        return {"status": "enabled", "strategy": name}

    @app.post("/api/strategies/{name}/disable")
    async def disable_strategy(name: str):
        """Disable a strategy"""
        if not app.state.engine or not app.state.engine._strategy_manager:
            raise HTTPException(status_code=500, detail="Strategy manager not available")

        app.state.engine._strategy_manager.disable_strategy(name)
        return {"status": "disabled", "strategy": name}

    # ==================== Risk Endpoints ====================

    @app.get("/api/risk")
    async def get_risk_status():
        """Get risk status"""
        if not app.state.engine or not app.state.engine._risk_manager:
            return {"error": "Risk manager not available"}

        return app.state.engine._risk_manager.get_risk_status()

    # ==================== Settings Endpoints ====================

    @app.get("/api/settings")
    async def get_settings():
        """Get current settings"""
        if not app.state.engine:
            return {"error": "Engine not initialized"}

        return app.state.engine.config.to_dict()

    @app.put("/api/settings")
    async def update_setting(update: SettingUpdate):
        """Update a setting"""
        # Settings updates would be handled here
        return {"status": "updated", "key": update.key}

    # ==================== Analytics Endpoints ====================

    @app.get("/api/analytics/trades")
    async def get_trade_history(limit: int = 100):
        """Get trade history"""
        # Would return trade history from analytics logger
        return {"trades": []}

    @app.get("/api/analytics/performance")
    async def get_performance():
        """Get performance metrics"""
        return {
            "total_pnl": 0,
            "win_rate": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0
        }

    _app = app
    return app


def get_app() -> Optional[FastAPI]:
    """Get the global app instance"""
    return _app


def get_engine_status(engine) -> Dict[str, Any]:
    """Get comprehensive engine status"""
    status = {
        "is_running": engine.is_running,
        "mode": engine.mode.value if hasattr(engine.mode, 'value') else str(engine.mode),
        "uptime_seconds": engine.status.uptime_seconds,
        "last_analysis": (
            engine.status.last_analysis_time.isoformat()
            if engine.status.last_analysis_time else None
        ),
        "error_count": engine.status.error_count
    }

    if engine.state and engine.state.portfolio:
        portfolio = engine.state.portfolio
        status["portfolio"] = {
            "account_balance": portfolio.account_balance,
            "buying_power": portfolio.buying_power,
            "cash": portfolio.cash,
            "total_market_value": portfolio.total_market_value,
            "total_unrealized_pnl": portfolio.total_unrealized_pnl,
            "daily_pnl": portfolio.daily_pnl,
            "total_pnl": portfolio.total_pnl,
            "position_count": portfolio.position_count
        }
        status["positions"] = {
            k: v.to_dict() for k, v in portfolio.positions.items()
        }

    return status


async def handle_client_message(websocket: WebSocket, message: Dict, engine):
    """Handle incoming client messages"""
    msg_type = message.get("type")

    if msg_type == "subscribe":
        # Handle subscription requests
        pass
    elif msg_type == "command":
        # Handle commands
        pass


async def broadcast_state_update(engine):
    """Broadcast state update to all clients"""
    await manager.broadcast({
        "type": "state_update",
        "data": get_engine_status(engine)
    })


async def broadcast_trade(trade_data: Dict):
    """Broadcast trade event"""
    await manager.broadcast({
        "type": "trade",
        "data": trade_data
    })


async def broadcast_signal(signal_data: Dict):
    """Broadcast signal event"""
    await manager.broadcast({
        "type": "signal",
        "data": signal_data
    })


async def broadcast_alert(alert_data: Dict):
    """Broadcast alert event"""
    await manager.broadcast({
        "type": "alert",
        "data": alert_data
    })
