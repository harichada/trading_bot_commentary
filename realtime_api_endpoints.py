#!/usr/bin/env python3
"""
Enhanced API Endpoints for Real-Time Trading
Integrates streaming functionality with the main trading bot API
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from realtime_integration import RealTimeTradingBot
from trading_bot_commentary_updated import TradingMode, CommentaryType

logger = logging.getLogger(__name__)

# Global real-time trading bot instance
realtime_bot = None

# Create FastAPI app
app = FastAPI(
    title="Real-Time Trading Bot API",
    description="Advanced trading bot with real-time streaming data and intelligent decision making",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Add to real-time bot if available
        if realtime_bot:
            await realtime_bot.add_websocket_connection(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Remove from real-time bot if available
        if realtime_bot:
            asyncio.create_task(realtime_bot.remove_websocket_connection(websocket))
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                # Remove disconnected connection
                self.active_connections.remove(connection)

manager = ConnectionManager()

# ============================================================================
# REAL-TIME TRADING ENDPOINTS
# ============================================================================

@app.post("/api/realtime/start")
async def start_realtime_trading(background_tasks: BackgroundTasks):
    """Start the real-time trading bot with streaming data"""
    global realtime_bot
    
    try:
        if realtime_bot and realtime_bot.is_running:
            return JSONResponse(
                status_code=400,
                content={"error": "Real-time trading bot is already running"}
            )
        
        # Initialize Schwab client (you'll need to set up authentication)
        # This is a placeholder - you'll need to implement proper authentication
        schwab_client = None  # Replace with actual Schwab client initialization
        
        if not schwab_client:
            return JSONResponse(
                status_code=400,
                content={"error": "Schwab client not configured. Please set up authentication first."}
            )
        
        # Create and start the real-time bot
        realtime_bot = RealTimeTradingBot(schwab_client, TradingMode.SIMULATION_WITH_COMMENTARY)
        
        # Start the bot in background
        background_tasks.add_task(realtime_bot.start)
        
        return {
            "status": "success",
            "message": "Real-time trading bot started successfully",
            "mode": "simulation_with_commentary",
            "symbols": realtime_bot.trading_symbols
        }
        
    except Exception as e:
        logger.error(f"Failed to start real-time trading: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to start real-time trading: {str(e)}"}
        )

@app.post("/api/realtime/stop")
async def stop_realtime_trading():
    """Stop the real-time trading bot"""
    global realtime_bot
    
    try:
        if not realtime_bot or not realtime_bot.is_running:
            return JSONResponse(
                status_code=400,
                content={"error": "Real-time trading bot is not running"}
            )
        
        await realtime_bot.stop()
        
        return {
            "status": "success",
            "message": "Real-time trading bot stopped successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to stop real-time trading: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to stop real-time trading: {str(e)}"}
        )

@app.get("/api/realtime/status")
async def get_realtime_status():
    """Get real-time trading bot status"""
    global realtime_bot
    
    try:
        if not realtime_bot:
            return {
                "status": "not_initialized",
                "message": "Real-time trading bot not initialized"
            }
        
        status = realtime_bot.get_status()
        return {
            "status": "success",
            "data": status
        }
        
    except Exception as e:
        logger.error(f"Failed to get real-time status: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get status: {str(e)}"}
        )

@app.post("/api/realtime/symbols")
async def update_trading_symbols(symbols: List[str]):
    """Update the list of symbols to monitor"""
    global realtime_bot
    
    try:
        if not realtime_bot:
            return JSONResponse(
                status_code=400,
                content={"error": "Real-time trading bot not initialized"}
            )
        
        # Update symbols
        realtime_bot.trading_symbols = symbols
        
        # Resubscribe to new symbols if bot is running
        if realtime_bot.is_running:
            await realtime_bot.realtime_engine.subscribe_to_symbols(symbols)
        
        return {
            "status": "success",
            "message": f"Updated trading symbols to: {symbols}",
            "symbols": symbols
        }
        
    except Exception as e:
        logger.error(f"Failed to update symbols: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to update symbols: {str(e)}"}
        )

@app.post("/api/realtime/mode")
async def change_trading_mode(mode: str):
    """Change trading mode (simulation, paper, live)"""
    global realtime_bot
    
    try:
        if not realtime_bot:
            return JSONResponse(
                status_code=400,
                content={"error": "Real-time trading bot not initialized"}
            )
        
        # Validate mode
        valid_modes = ["simulation_commentary", "paper", "live"]
        if mode not in valid_modes:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid mode. Must be one of: {valid_modes}"}
            )
        
        # Convert mode string to TradingMode enum
        if mode == "simulation_commentary":
            new_mode = TradingMode.SIMULATION_WITH_COMMENTARY
        elif mode == "paper":
            new_mode = TradingMode.PAPER
        else:  # live
            new_mode = TradingMode.LIVE
        
        # Update mode
        realtime_bot.mode = new_mode
        
        return {
            "status": "success",
            "message": f"Trading mode changed to: {mode}",
            "mode": mode
        }
        
    except Exception as e:
        logger.error(f"Failed to change trading mode: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to change trading mode: {str(e)}"}
        )

@app.get("/api/realtime/positions")
async def get_realtime_positions():
    """Get current positions from real-time trading"""
    global realtime_bot
    
    try:
        if not realtime_bot:
            return {
                "status": "not_initialized",
                "positions": []
            }
        
        positions = realtime_bot.realtime_engine.positions
        
        # Convert positions to serializable format
        positions_data = []
        for symbol, position in positions.items():
            positions_data.append({
                "symbol": position.symbol,
                "side": position.side,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "unrealized_pnl": position.unrealized_pnl,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "entry_time": position.entry_time.isoformat(),
                "reasoning": position.reasoning
            })
        
        return {
            "status": "success",
            "positions": positions_data,
            "count": len(positions_data)
        }
        
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get positions: {str(e)}"}
        )

@app.get("/api/realtime/signals")
async def get_realtime_signals():
    """Get recent trading signals"""
    global realtime_bot
    
    try:
        if not realtime_bot:
            return {
                "status": "not_initialized",
                "signals": []
            }
        
        signals = realtime_bot.realtime_engine.signal_history[-50:]  # Last 50 signals
        
        # Convert signals to serializable format
        signals_data = []
        for signal in signals:
            signals_data.append({
                "symbol": signal.symbol,
                "signal_type": signal.signal_type,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "position_size": signal.position_size,
                "timestamp": signal.timestamp.isoformat(),
                "reasoning": signal.reasoning
            })
        
        return {
            "status": "success",
            "signals": signals_data,
            "count": len(signals_data)
        }
        
    except Exception as e:
        logger.error(f"Failed to get signals: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get signals: {str(e)}"}
        )

@app.get("/api/realtime/performance")
async def get_realtime_performance():
    """Get real-time performance statistics"""
    global realtime_bot
    
    try:
        if not realtime_bot:
            return {
                "status": "not_initialized",
                "performance": {}
            }
        
        stats = realtime_bot.realtime_engine.get_performance_stats()
        
        return {
            "status": "success",
            "performance": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get performance: {str(e)}"}
        )

@app.get("/api/realtime/streaming")
async def get_streaming_stats():
    """Get streaming connection statistics"""
    global realtime_bot
    
    try:
        if not realtime_bot:
            return {
                "status": "not_initialized",
                "streaming": {}
            }
        
        stream_stats = realtime_bot.realtime_engine.stream_client.get_connection_stats()
        
        return {
            "status": "success",
            "streaming": stream_stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get streaming stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get streaming stats: {str(e)}"}
        )

# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@app.websocket("/ws/realtime")
async def websocket_realtime_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time trading updates"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/commentary")
async def websocket_commentary_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time commentary"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# ============================================================================
# DASHBOARD ENDPOINTS
# ============================================================================

@app.get("/api/realtime/dashboard")
async def get_realtime_dashboard():
    """Get comprehensive dashboard data for real-time trading"""
    global realtime_bot
    
    try:
        if not realtime_bot:
            return {
                "status": "not_initialized",
                "dashboard": {}
            }
        
        # Get all relevant data
        status = realtime_bot.get_status()
        positions = realtime_bot.realtime_engine.positions
        signals = realtime_bot.realtime_engine.signal_history[-10:]  # Last 10 signals
        performance = realtime_bot.realtime_engine.get_performance_stats()
        stream_stats = realtime_bot.realtime_engine.stream_client.get_connection_stats()
        
        # Convert positions to serializable format
        positions_data = []
        for symbol, position in positions.items():
            positions_data.append({
                "symbol": position.symbol,
                "side": position.side,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "unrealized_pnl": position.unrealized_pnl,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "entry_time": position.entry_time.isoformat()
            })
        
        # Convert signals to serializable format
        signals_data = []
        for signal in signals:
            signals_data.append({
                "symbol": signal.symbol,
                "signal_type": signal.signal_type,
                "confidence": signal.confidence,
                "entry_price": signal.entry_price,
                "timestamp": signal.timestamp.isoformat()
            })
        
        dashboard_data = {
            "status": status,
            "positions": positions_data,
            "recent_signals": signals_data,
            "performance": performance,
            "streaming": stream_stats,
            "symbols": realtime_bot.trading_symbols,
            "last_updated": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "dashboard": dashboard_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get dashboard data: {str(e)}"}
        )

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/api/realtime/health")
async def health_check():
    """Health check for real-time trading system"""
    global realtime_bot
    
    try:
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "realtime_bot_initialized": realtime_bot is not None,
            "realtime_bot_running": realtime_bot.is_running if realtime_bot else False,
            "websocket_connections": len(manager.active_connections)
        }
        
        if realtime_bot:
            stream_stats = realtime_bot.realtime_engine.stream_client.get_connection_stats()
            health_data["streaming_connected"] = stream_stats.get("is_connected", False)
            health_data["streaming_messages"] = stream_stats.get("message_count", 0)
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Real-Time Trading Bot API",
        "version": "2.0.0",
        "description": "Advanced trading bot with real-time streaming data",
        "endpoints": {
            "realtime": "/api/realtime/*",
            "websocket": "/ws/*",
            "health": "/api/realtime/health"
        },
        "documentation": "/docs"
    }

# ============================================================================
# STARTUP AND SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup"""
    logger.info("Real-Time Trading Bot API starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global realtime_bot
    
    logger.info("Real-Time Trading Bot API shutting down...")
    
    if realtime_bot and realtime_bot.is_running:
        await realtime_bot.stop()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║        Real-Time Trading Bot API - Streaming Data Enabled           ║
    ║     Advanced algorithmic trading with real-time market data         ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n🚀 Starting Real-Time Trading Bot API...")
    print("📊 Features:")
    print("  • Real-time streaming data from Schwab")
    print("  • Instant signal generation and execution")
    print("  • Live commentary and analysis")
    print("  • WebSocket real-time updates")
    print("  • Advanced risk management")
    
    print("\n🌐 API Documentation: http://localhost:8000/docs")
    print("📱 WebSocket Endpoints:")
    print("  • Real-time updates: ws://localhost:8000/ws/realtime")
    print("  • Commentary: ws://localhost:8000/ws/commentary")
    print("❌ Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 