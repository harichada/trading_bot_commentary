"""
FastAPI Application

Main API application with WebSocket support and professional dashboard.
"""

import asyncio
import json
import logging
import random
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


# Demo data state
class DemoState:
    """Demo trading state for UI demonstration"""

    def __init__(self):
        self.is_running = False
        self.mode = "simulation"
        self.start_time = None
        self.account_balance = 100000.0
        self.cash = 55000.0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0

        # Demo positions
        self.positions = {
            "AAPL": {
                "symbol": "AAPL",
                "side": "long",
                "quantity": 100,
                "entry_price": 185.50,
                "current_price": 189.25,
                "stop_loss": 180.0,
                "take_profit": 195.0,
                "strategy": "momentum",
                "unrealized_pnl": 375.0,
                "unrealized_pnl_pct": 2.02
            },
            "NVDA": {
                "symbol": "NVDA",
                "side": "long",
                "quantity": 50,
                "entry_price": 480.0,
                "current_price": 495.80,
                "stop_loss": 460.0,
                "take_profit": 520.0,
                "strategy": "breakout",
                "unrealized_pnl": 790.0,
                "unrealized_pnl_pct": 3.29
            },
            "GOOGL": {
                "symbol": "GOOGL",
                "side": "long",
                "quantity": 75,
                "entry_price": 142.30,
                "current_price": 140.15,
                "stop_loss": 135.0,
                "take_profit": 155.0,
                "strategy": "mean_reversion",
                "unrealized_pnl": -161.25,
                "unrealized_pnl_pct": -1.51
            }
        }

        self.strategies = {
            "momentum": {"enabled": True, "signal": "buy", "confidence": 0.75},
            "mean_reversion": {"enabled": True, "signal": "hold", "confidence": 0.45},
            "breakout": {"enabled": True, "signal": "sell", "confidence": 0.68}
        }

        self.activities = []

    def update_prices(self):
        """Simulate price movements"""
        for symbol, pos in self.positions.items():
            # Random price change (-1% to +1%)
            change_pct = random.uniform(-0.01, 0.01)
            pos["current_price"] = round(pos["current_price"] * (1 + change_pct), 2)
            pos["unrealized_pnl"] = round((pos["current_price"] - pos["entry_price"]) * pos["quantity"], 2)
            pos["unrealized_pnl_pct"] = round((pos["current_price"] - pos["entry_price"]) / pos["entry_price"] * 100, 2)

        # Update daily P&L
        self.daily_pnl = sum(p["unrealized_pnl"] for p in self.positions.values())

    def update_strategies(self):
        """Simulate strategy signal changes"""
        signals = ["buy", "sell", "hold"]
        for name, strategy in self.strategies.items():
            if random.random() < 0.1:  # 10% chance to change
                strategy["signal"] = random.choice(signals)
                strategy["confidence"] = round(random.uniform(0.3, 0.95), 2)

    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        total_market_value = sum(p["current_price"] * p["quantity"] for p in self.positions.values())

        return {
            "is_running": self.is_running,
            "mode": self.mode,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "portfolio": {
                "account_balance": self.account_balance,
                "buying_power": self.cash,
                "cash": self.cash,
                "total_market_value": round(total_market_value, 2),
                "total_unrealized_pnl": round(self.daily_pnl, 2),
                "daily_pnl": round(self.daily_pnl, 2),
                "total_pnl": round(self.total_pnl + self.daily_pnl, 2),
                "position_count": len(self.positions)
            },
            "positions": self.positions,
            "strategies": self.strategies
        }

    def add_activity(self, activity_type: str, data: Dict):
        """Add activity to feed"""
        self.activities.insert(0, {
            "type": activity_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.activities) > 50:
            self.activities = self.activities[:50]


# Global demo state
demo_state = DemoState()


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
    """Create FastAPI application."""
    global _app

    app = FastAPI(
        title=title,
        version=version,
        description="Professional Trading Bot API with real-time WebSocket updates"
    )

    # Store engine reference
    app.state.engine = engine
    app.state.update_task = None

    # Static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.on_event("startup")
    async def startup():
        """Start background update task"""
        app.state.update_task = asyncio.create_task(background_updates())

    @app.on_event("shutdown")
    async def shutdown():
        """Stop background update task"""
        if app.state.update_task:
            app.state.update_task.cancel()
            try:
                await app.state.update_task
            except asyncio.CancelledError:
                pass

    async def background_updates():
        """Background task to send periodic updates"""
        while True:
            try:
                await asyncio.sleep(2)  # Update every 2 seconds

                if demo_state.is_running:
                    # Update demo data
                    demo_state.update_prices()
                    demo_state.update_strategies()

                    # Occasionally generate activity
                    if random.random() < 0.15:  # 15% chance
                        activity_types = [
                            ("signal", {
                                "title": f"Signal: {random.choice(['AAPL', 'NVDA', 'GOOGL', 'MSFT', 'TSLA'])}",
                                "description": f"{random.choice(['Buy', 'Sell', 'Hold'])} signal (conf: {random.uniform(0.5, 0.95):.2f})"
                            }),
                            ("alert", {
                                "title": "Risk Update",
                                "description": f"Daily exposure: {random.uniform(30, 60):.1f}%"
                            })
                        ]
                        act_type, act_data = random.choice(activity_types)
                        demo_state.add_activity(act_type, act_data)

                        await manager.broadcast({
                            "type": act_type,
                            "data": act_data
                        })

                    # Broadcast state update
                    await manager.broadcast({
                        "type": "state_update",
                        "data": demo_state.get_status()
                    })

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background update error: {e}")

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
            await websocket.send_json({
                "type": "state_update",
                "data": demo_state.get_status()
            })

            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=30.0
                    )
                    message = json.loads(data)
                    await handle_client_message(websocket, message)
                except asyncio.TimeoutError:
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
        return demo_state.get_status()

    # ==================== Control Endpoints ====================

    @app.post("/api/start")
    async def start_bot(request: StartRequest = None):
        """Start the trading bot"""
        if demo_state.is_running:
            return {"status": "already_running"}

        demo_state.is_running = True
        demo_state.start_time = datetime.now()

        demo_state.add_activity("alert", {
            "title": "Bot Started",
            "description": f"Trading bot started in {demo_state.mode} mode"
        })

        await manager.broadcast({
            "type": "state_update",
            "data": demo_state.get_status()
        })

        return {"status": "started"}

    @app.post("/api/stop")
    async def stop_bot():
        """Stop the trading bot"""
        demo_state.is_running = False

        demo_state.add_activity("alert", {
            "title": "Bot Stopped",
            "description": "Trading bot has been stopped"
        })

        await manager.broadcast({
            "type": "state_update",
            "data": demo_state.get_status()
        })

        return {"status": "stopped"}

    @app.post("/api/analyze")
    async def run_analysis():
        """Trigger manual analysis"""
        demo_state.update_prices()
        demo_state.update_strategies()

        demo_state.add_activity("signal", {
            "title": "Analysis Complete",
            "description": f"Analyzed {len(demo_state.positions)} positions"
        })

        await manager.broadcast({
            "type": "state_update",
            "data": demo_state.get_status()
        })

        return {"status": "analysis_complete"}

    # ==================== Portfolio Endpoints ====================

    @app.get("/api/portfolio")
    async def get_portfolio():
        """Get portfolio summary"""
        return demo_state.get_status()["portfolio"]

    @app.get("/api/positions")
    async def get_positions():
        """Get all positions"""
        return {"positions": demo_state.positions}

    @app.post("/api/positions/{symbol}/close")
    async def close_position(symbol: str):
        """Close a specific position"""
        if symbol not in demo_state.positions:
            raise HTTPException(status_code=404, detail=f"Position {symbol} not found")

        pos = demo_state.positions[symbol]
        pnl = pos["unrealized_pnl"]

        del demo_state.positions[symbol]
        demo_state.total_pnl += pnl
        demo_state.cash += pos["current_price"] * pos["quantity"]

        demo_state.add_activity("trade" if pnl >= 0 else "alert", {
            "title": f"Closed {symbol}",
            "description": f"P&L: ${pnl:+.2f}"
        })

        await manager.broadcast({
            "type": "trade",
            "data": {
                "action": "sell",
                "title": f"Sold {symbol}",
                "description": f"{pos['quantity']} shares @ ${pos['current_price']:.2f} ({'+' if pnl >= 0 else ''}{pnl:.2f})"
            }
        })

        await manager.broadcast({
            "type": "state_update",
            "data": demo_state.get_status()
        })

        return {"status": "closed", "symbol": symbol, "pnl": pnl}

    @app.post("/api/close-all")
    async def close_all_positions():
        """Close all positions"""
        total_pnl = sum(p["unrealized_pnl"] for p in demo_state.positions.values())
        total_value = sum(p["current_price"] * p["quantity"] for p in demo_state.positions.values())

        demo_state.positions = {}
        demo_state.total_pnl += total_pnl
        demo_state.cash += total_value

        demo_state.add_activity("alert", {
            "title": "Closed All Positions",
            "description": f"Total P&L: ${total_pnl:+.2f}"
        })

        await manager.broadcast({
            "type": "state_update",
            "data": demo_state.get_status()
        })

        return {"status": "all_closed", "total_pnl": total_pnl}

    # ==================== Order Endpoints ====================

    @app.post("/api/orders")
    async def submit_order(order: OrderRequest):
        """Submit a new order"""
        # Simulate order fill
        price = random.uniform(100, 500)

        if order.side.lower() == 'buy':
            demo_state.positions[order.symbol] = {
                "symbol": order.symbol,
                "side": "long",
                "quantity": order.quantity,
                "entry_price": price,
                "current_price": price,
                "stop_loss": price * 0.95,
                "take_profit": price * 1.10,
                "strategy": "manual",
                "unrealized_pnl": 0,
                "unrealized_pnl_pct": 0
            }
            demo_state.cash -= price * order.quantity

            await manager.broadcast({
                "type": "trade",
                "data": {
                    "action": "buy",
                    "title": f"Bought {order.symbol}",
                    "description": f"{order.quantity} shares @ ${price:.2f}"
                }
            })

        await manager.broadcast({
            "type": "state_update",
            "data": demo_state.get_status()
        })

        return {"status": "filled", "order": order.dict(), "fill_price": price}

    @app.get("/api/orders")
    async def get_orders():
        """Get pending orders"""
        return {"orders": {}}

    # ==================== Strategy Endpoints ====================

    @app.get("/api/strategies")
    async def get_strategies():
        """Get strategy status"""
        return {"strategies": demo_state.strategies}

    @app.post("/api/strategies/{name}/enable")
    async def enable_strategy(name: str):
        """Enable a strategy"""
        if name in demo_state.strategies:
            demo_state.strategies[name]["enabled"] = True
        return {"status": "enabled", "strategy": name}

    @app.post("/api/strategies/{name}/disable")
    async def disable_strategy(name: str):
        """Disable a strategy"""
        if name in demo_state.strategies:
            demo_state.strategies[name]["enabled"] = False
        return {"status": "disabled", "strategy": name}

    # ==================== Risk Endpoints ====================

    @app.get("/api/risk")
    async def get_risk_status():
        """Get risk status"""
        total_value = sum(p["current_price"] * p["quantity"] for p in demo_state.positions.values())
        exposure = total_value / demo_state.account_balance * 100

        return {
            "daily_loss_pct": abs(min(0, demo_state.daily_pnl)) / demo_state.account_balance * 100,
            "daily_loss_limit": 3.0,
            "exposure_pct": exposure,
            "exposure_limit": 80.0,
            "drawdown_pct": 3.5,
            "drawdown_limit": 15.0,
            "consecutive_losses": 0,
            "circuit_breaker_active": False
        }

    # ==================== Settings Endpoints ====================

    @app.get("/api/settings")
    async def get_settings():
        """Get current settings"""
        return {
            "mode": demo_state.mode,
            "risk": {
                "max_position_size": 0.10,
                "max_daily_loss": 0.03,
                "max_drawdown": 0.15
            }
        }

    @app.put("/api/settings")
    async def update_setting(update: SettingUpdate):
        """Update a setting"""
        return {"status": "updated", "key": update.key}

    # ==================== Activity Feed ====================

    @app.get("/api/activities")
    async def get_activities(limit: int = 20):
        """Get recent activities"""
        return {"activities": demo_state.activities[:limit]}

    _app = app
    return app


def get_app() -> Optional[FastAPI]:
    """Get the global app instance"""
    return _app


async def handle_client_message(websocket: WebSocket, message: Dict):
    """Handle incoming client messages"""
    msg_type = message.get("type")

    if msg_type == "ping":
        await websocket.send_json({"type": "pong"})
