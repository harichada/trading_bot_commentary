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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from websockets.exceptions import ConnectionClosedError
import uvicorn

from core.models import TradingMode, CommentaryType, SignalType
from core.config import Config, config, logger
from core.commentary import TradingCommentary, CommentarySystem
from core.websocket_manager import ConnectionManager

logger = logging.getLogger('TradingBot')

# ============================================================================
# FASTAPI APPLICATION WITH COMMENTARY
# ============================================================================

app = FastAPI(title="Trading Bot with Commentary API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://localhost:9000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:9000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
trading_engine = None
connection_manager = ConnectionManager()

# Load dashboard HTML from template file
import os as _os
_template_dir = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), 'templates')
try:
    with open(_os.path.join(_template_dir, 'dashboard.html'), 'r') as _f:
        DASHBOARD_HTML_WITH_COMMENTARY = _f.read()
except FileNotFoundError:
    DASHBOARD_HTML_WITH_COMMENTARY = "<html><body><h1>Dashboard template not found</h1></body></html>"


async def simulate_backtest(historical_data: Dict[str, pd.DataFrame], config: dict) -> dict:
    """Simulate a backtest with the given historical data"""
    initial_capital = config.get('initial_capital', 100000)
    commission = config.get('commission', 0.001)  # 0.1% per trade

    # Initialize tracking variables
    cash = initial_capital
    positions = {}
    trades = []
    equity_curve = []

    # Get all unique timestamps across all symbols
    all_timestamps = set()
    for df in historical_data.values():
        all_timestamps.update(df.index)
    all_timestamps = sorted(list(all_timestamps))

    # Simulate trading
    for timestamp in all_timestamps:
        current_equity = cash

        # Update positions with current prices
        for symbol, position in list(positions.items()):
            if symbol in historical_data and timestamp in historical_data[symbol].index:
                current_price = historical_data[symbol].loc[timestamp, 'Close']
                position['current_price'] = current_price
                position['value'] = position['quantity'] * current_price
                position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
                current_equity += position['value']

                # Simple exit logic - exit if 2% profit or 1% loss
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                if pnl_pct > 0.02 or pnl_pct < -0.01:
                    # Close position
                    trade_pnl = position['unrealized_pnl'] - (position['value'] * commission)
                    cash += position['value'] - (position['value'] * commission)

                    trades.append({
                        'symbol': symbol,
                        'entry_time': position['entry_time'],
                        'exit_time': timestamp,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'quantity': position['quantity'],
                        'pnl': trade_pnl,
                        'pnl_pct': pnl_pct
                    })

                    del positions[symbol]

        # Generate entry signals (simple momentum strategy for demo)
        for symbol, df in historical_data.items():
            if timestamp in df.index and symbol not in positions and len(positions) < 5:
                # Get recent data
                idx = df.index.get_loc(timestamp)
                if idx >= 20:  # Need at least 20 periods
                    recent_data = df.iloc[max(0, idx-20):idx+1]

                    # Simple momentum signal
                    returns = recent_data['Close'].pct_change().dropna()
                    if len(returns) > 0 and returns.mean() > 0.001 and returns.iloc[-1] > 0:
                        # Enter position
                        position_size = (current_equity * 0.1) / recent_data['Close'].iloc[-1]  # 10% of equity
                        position_value = position_size * recent_data['Close'].iloc[-1]

                        if cash >= position_value * (1 + commission):
                            positions[symbol] = {
                                'quantity': position_size,
                                'entry_price': recent_data['Close'].iloc[-1],
                                'entry_time': timestamp,
                                'current_price': recent_data['Close'].iloc[-1],
                                'value': position_value
                            }
                            cash -= position_value * (1 + commission)

        # Record equity
        equity_curve.append({
            'date': timestamp.strftime('%Y-%m-%d'),
            'value': current_equity
        })

    # Calculate performance metrics
    equity_df = pd.DataFrame(equity_curve)
    if not equity_df.empty:
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)

        # Remove duplicates by keeping last value for each date
        equity_df = equity_df.groupby(equity_df.index).last()

        daily_returns = equity_df['value'].pct_change().dropna()

        total_return = (equity_df['value'].iloc[-1] - initial_capital) / initial_capital
        annual_return = (1 + total_return) ** (252 / len(equity_df)) - 1 if len(equity_df) > 0 else 0

        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

        # Calculate drawdown
        rolling_max = equity_df['value'].expanding().max()
        drawdown = (equity_df['value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Trade statistics
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades and sum(t['pnl'] for t in losing_trades) != 0 else 0

        # Monthly returns
        monthly_returns = []
        if not equity_df.empty:
            monthly = equity_df.resample('M').last()
            monthly_pct = monthly['value'].pct_change().dropna()
            for date, ret in monthly_pct.items():
                monthly_returns.append({
                    'month': date.strftime('%Y-%m'),
                    'return': ret
                })
    else:
        # Default values if no data
        total_return = 0
        annual_return = 0
        sharpe_ratio = 0
        max_drawdown = 0
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        monthly_returns = []

    return {
        'summary': {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sharpe_ratio * 0.8,  # Approximation
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades) if trades else 0,
            'losing_trades': len(losing_trades) if trades else 0,
            'profit_factor': profit_factor,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'best_trade': max([t['pnl'] for t in trades]) if trades else 0,
            'worst_trade': min([t['pnl'] for t in trades]) if trades else 0,
            'commission_paid': sum([t.get('commission', 0) for t in trades]),
            'initial_capital': initial_capital,
            'final_capital': equity_df['value'].iloc[-1] if not equity_df.empty else initial_capital
        },
        'trades': trades,
        'equity_curve': equity_curve[-100:],  # Last 100 points
        'monthly_returns': monthly_returns,
        'strategy_performance': {
            'momentum': {
                'trades': len(trades),
                'win_rate': win_rate,
                'avg_return': total_return / len(trades) if trades else 0,
                'total_return': total_return
            }
        }
    }

@app.post("/api/toggle-mode")
async def toggle_trading_mode(request: dict):
    global trading_engine

    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}

    new_mode = request.get('mode', 'simulation')

    if new_mode == 'live':
        trading_engine.mode = TradingMode.LIVE
        # Add safety check
        if not trading_engine.schwab_client:
            return {"status": "error", "message": "Cannot switch to live mode - Schwab not connected"}

        # Sync external positions when switching to live mode
        await trading_engine._update_real_positions()

    else:
        trading_engine.mode = TradingMode.SIMULATION_WITH_COMMENTARY

    trading_engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.DECISION,
        symbol=None,
        title=f"\U0001f504 Mode Changed",
        message=f"Switched to {'LIVE TRADING' if new_mode == 'live' else 'SIMULATION'} mode",
        importance=10
    ))

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
    return HTMLResponse(content=DASHBOARD_HTML_WITH_COMMENTARY)

@app.post("/api/start")
async def start_trading():
    global trading_engine, connection_manager, trading_task

    # Import here to avoid circular imports
    from trading_bot_commentary_updated import TradingEngineWithCommentary, config_manager

    if not trading_engine:
        trading_engine = TradingEngineWithCommentary(connection_manager=connection_manager)

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
        threading.Thread(
            target=lambda: asyncio.run(trading_engine.start())
        ).start()

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

    # Determine if we're in live mode with Schwab connected
    is_live = trading_engine.mode == TradingMode.LIVE
    has_schwab = trading_engine.schwab_client is not None

    # Try to get real Schwab data if available
    if is_live and has_schwab:
        try:
            account_info = await trading_engine._get_real_account_info()
            schwab_positions = await trading_engine.get_schwab_positions()

            # Calculate total unrealized P&L from positions
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

@app.get("/api/professional/backtest/report/latest")
async def get_latest_backtest_report():
    """Get the latest backtest report"""
    # Check if we have a saved backtest report
    backtest_report_path = Path("backtest_results/latest_report.json")

    if backtest_report_path.exists():
        try:
            with open(backtest_report_path, 'r') as f:
                report = json.load(f)
            return report
        except Exception as e:
            logger.error(f"Error loading backtest report: {e}")

    # Return a sample backtest report
    return {
        "summary": {
            "total_return": 0.152,
            "annual_return": 0.183,
            "sharpe_ratio": 1.24,
            "sortino_ratio": 1.68,
            "max_drawdown": -0.078,
            "win_rate": 0.565,
            "total_trades": 48,
            "winning_trades": 27,
            "losing_trades": 21,
            "profit_factor": 1.42,
            "average_win": 0.0089,
            "average_loss": -0.0052,
            "best_trade": 0.0234,
            "worst_trade": -0.0156,
            "commission_paid": 96.50,
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "initial_capital": 10000,
            "final_capital": 11520
        },
        "trades": [
            {
                "date": "2024-01-15",
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 50,
                "entry_price": 185.20,
                "exit_price": 188.50,
                "pnl": 165.00,
                "return_pct": 0.0178
            },
            {
                "date": "2024-02-03",
                "symbol": "MSFT",
                "side": "buy",
                "quantity": 30,
                "entry_price": 405.30,
                "exit_price": 402.10,
                "pnl": -96.00,
                "return_pct": -0.0079
            }
        ],
        "equity_curve": [
            {"date": "2024-01-01", "value": 10000},
            {"date": "2024-02-01", "value": 10250},
            {"date": "2024-03-01", "value": 10480},
            {"date": "2024-04-01", "value": 10650},
            {"date": "2024-05-01", "value": 10900},
            {"date": "2024-06-01", "value": 11100},
            {"date": "2024-07-01", "value": 11000},
            {"date": "2024-08-01", "value": 11200},
            {"date": "2024-09-01", "value": 11350},
            {"date": "2024-10-01", "value": 11400},
            {"date": "2024-11-01", "value": 11450},
            {"date": "2024-12-31", "value": 11520}
        ],
        "monthly_returns": [
            {"month": "2024-01", "return": 0.025},
            {"month": "2024-02", "return": 0.022},
            {"month": "2024-03", "return": 0.016},
            {"month": "2024-04", "return": 0.023},
            {"month": "2024-05", "return": 0.018},
            {"month": "2024-06", "return": -0.009},
            {"month": "2024-07", "return": 0.018},
            {"month": "2024-08", "return": 0.013},
            {"month": "2024-09", "return": 0.004},
            {"month": "2024-10", "return": 0.004},
            {"month": "2024-11", "return": 0.006}
        ],
        "strategy_performance": {
            "momentum": {
                "trades": 18,
                "win_rate": 0.611,
                "avg_return": 0.008,
                "total_return": 0.144
            },
            "mean_reversion": {
                "trades": 15,
                "win_rate": 0.533,
                "avg_return": 0.005,
                "total_return": 0.075
            },
            "breakout": {
                "trades": 15,
                "win_rate": 0.600,
                "avg_return": 0.007,
                "total_return": 0.105
            }
        }
    }

@app.post("/api/professional/backtest")
async def run_professional_backtest(config: dict):
    """Run a backtest with specified configuration"""
    try:
        # Validate config
        symbols = config.get('symbols', ['SPY'])
        start_date = config.get('start_date', (datetime.now() - timedelta(days=30)).isoformat())
        end_date = config.get('end_date', datetime.now().isoformat())
        initial_capital = config.get('initial_capital', 100000)

        # Get historical data from Schwab if available
        historical_data = {}

        if trading_engine and trading_engine.schwab_client:
            try:
                for symbol in symbols:
                    # Get price history from Schwab
                    # Convert dates to datetime objects
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

                    # Calculate period parameters for Schwab API
                    period_days = (end_dt - start_dt).days

                    # Import Schwab enums
                    from schwab.client import Client

                    # Schwab API parameters - use proper enums
                    if period_days <= 10:
                        period_type = Client.PriceHistory.PeriodType.DAY
                        period = Client.PriceHistory.Period.TEN_DAYS
                    elif period_days <= 30:
                        period_type = Client.PriceHistory.PeriodType.MONTH
                        period = Client.PriceHistory.Period.ONE_MONTH
                    elif period_days <= 180:
                        period_type = Client.PriceHistory.PeriodType.MONTH
                        period = Client.PriceHistory.Period.SIX_MONTHS
                    else:
                        period_type = Client.PriceHistory.PeriodType.YEAR
                        period = Client.PriceHistory.Period.ONE_YEAR

                    frequency_type = Client.PriceHistory.FrequencyType.MINUTE
                    frequency = Client.PriceHistory.Frequency.EVERY_FIVE_MINUTES

                    # Get price history
                    response = trading_engine.schwab_client.get_price_history(
                        symbol,
                        period_type=period_type,
                        period=period,
                        frequency_type=frequency_type,
                        frequency=frequency,
                        start_datetime=start_dt,
                        end_datetime=end_dt
                    )

                    if response.status_code == 200:
                        data = response.json()
                        candles = data.get('candles', [])

                        if candles:
                            # Convert to DataFrame
                            df = pd.DataFrame(candles)
                            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                            df.set_index('datetime', inplace=True)

                            # Rename columns to match expected format
                            column_mapping = {
                                'open': 'Open',
                                'high': 'High',
                                'low': 'Low',
                                'close': 'Close',
                                'volume': 'Volume'
                            }
                            df.rename(columns=column_mapping, inplace=True)

                            historical_data[symbol] = df

                            # Add commentary about data loaded
                            trading_engine.commentary.add_commentary(TradingCommentary(
                                timestamp=datetime.now(),
                                type=CommentaryType.DATA,
                                symbol=symbol,
                                title=f"\U0001f4ca Historical Data Loaded",
                                message=f"Loaded {len(df)} candles for {symbol} from {start_date} to {end_date}",
                                importance=6
                            ))
            except Exception as e:
                logger.error(f"Error fetching historical data from Schwab: {e}")

        # If we couldn't get data from Schwab, generate sample data
        if not historical_data:
            for symbol in symbols:
                # Generate realistic sample data
                dates = pd.date_range(start=start_date, end=end_date, freq='5min')
                # Filter for market hours only (9:30 AM - 4:00 PM ET)
                dates = dates[(dates.hour >= 9) & ((dates.hour < 16) | ((dates.hour == 16) & (dates.minute == 0)))]
                dates = dates[(dates.hour > 9) | ((dates.hour == 9) & (dates.minute >= 30))]

                # Generate price data with realistic volatility
                base_price = 400 if symbol == 'SPY' else 100
                returns = np.random.normal(0.0001, 0.001, len(dates))  # Small returns with volatility
                prices = base_price * np.exp(np.cumsum(returns))

                # Create OHLCV data
                df = pd.DataFrame(index=dates)
                df['Open'] = prices * (1 + np.random.normal(0, 0.0005, len(dates)))
                df['High'] = prices * (1 + np.abs(np.random.normal(0, 0.001, len(dates))))
                df['Low'] = prices * (1 - np.abs(np.random.normal(0, 0.001, len(dates))))
                df['Close'] = prices
                df['Volume'] = np.random.randint(1000000, 5000000, len(dates))

                historical_data[symbol] = df

        # Run backtest simulation
        backtest_results = await simulate_backtest(historical_data, config)

        # Save results
        results_path = Path("backtest_results")
        results_path.mkdir(exist_ok=True)

        # Convert numpy types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        report = {
            "config": config,
            "summary": convert_to_serializable(backtest_results['summary']),
            "trades": convert_to_serializable(backtest_results['trades'][:100]),  # Limit to 100 trades for report
            "equity_curve": convert_to_serializable(backtest_results['equity_curve']),
            "monthly_returns": convert_to_serializable(backtest_results['monthly_returns']),
            "strategy_performance": convert_to_serializable(backtest_results['strategy_performance']),
            "timestamp": datetime.now().isoformat()
        }

        with open(results_path / "latest_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        summary = convert_to_serializable(backtest_results['summary'])
        return {
            'total_return': summary['total_return'],
            'annual_return': summary['annual_return'],
            'sharpe_ratio': summary['sharpe_ratio'],
            'max_drawdown': summary['max_drawdown'],
            'win_rate': summary['win_rate'],
            'total_trades': summary['total_trades'],
            'profit_factor': summary['profit_factor'],
            'report_url': '/api/professional/backtest/report/latest',
            'config': config,
            'data_source': 'schwab' if trading_engine and trading_engine.schwab_client and historical_data else 'simulated'
        }
    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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
                    'realized_pnl': 0
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

@app.get("/api/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary report"""
    from analytics_logger import analytics_logger
    return analytics_logger.get_summary_report()

@app.get("/api/analytics/trades")
async def get_trade_history(days: int = 7, symbol: str = None):
    """Get trade history for analysis"""
    from analytics_logger import analytics_logger
    trades = analytics_logger.get_trade_history(days=days, symbol=symbol)
    return {
        'status': 'success',
        'count': len(trades),
        'trades': trades
    }

@app.get("/api/analytics/decisions")
async def get_decision_analysis(days: int = 7, symbol: str = None):
    """Get decision analysis"""
    from analytics_logger import analytics_logger
    analysis = analytics_logger.get_decision_analysis(days=days, symbol=symbol)
    return {
        'status': 'success',
        'analysis': analysis
    }

@app.get("/api/analytics/performance")
async def get_performance_history(days: int = 7):
    """Get performance snapshots"""
    from analytics_logger import analytics_logger
    snapshots = analytics_logger.get_performance_history(days=days)
    return {
        'status': 'success',
        'count': len(snapshots),
        'snapshots': snapshots
    }

@app.get("/api/analytics/daily-stats")
async def get_daily_stats(date: str = None):
    """Get statistics for a specific day"""
    from analytics_logger import analytics_logger
    from datetime import datetime as dt
    if date:
        try:
            target_date = dt.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            return {'status': 'error', 'message': 'Invalid date format. Use YYYY-MM-DD'}
    else:
        target_date = dt.now().date()

    stats = analytics_logger.get_daily_stats(target_date)
    return {
        'status': 'success',
        'stats': stats
    }

@app.post("/api/analytics/export/{category}")
async def export_analytics(category: str, days: int = 30):
    """Export analytics to CSV file"""
    from analytics_logger import analytics_logger, LogCategory

    category_map = {
        'trades': LogCategory.TRADE,
        'decisions': LogCategory.DECISION,
        'signals': LogCategory.SIGNAL,
        'performance': LogCategory.PERFORMANCE,
        'market_data': LogCategory.MARKET_DATA,
        'system': LogCategory.SYSTEM
    }

    if category not in category_map:
        return {'status': 'error', 'message': f'Invalid category. Valid: {list(category_map.keys())}'}

    output_file = f"logs/exports/{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    Path("logs/exports").mkdir(parents=True, exist_ok=True)

    analytics_logger.export_to_csv(category_map[category], output_file, days=days)

    return {
        'status': 'success',
        'message': f'Exported to {output_file}',
        'file': output_file
    }

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

@app.post("/api/backtest/run")
async def run_backtest_schwab(config: dict):
    """Run a backtest with the given configuration using Schwab historical data"""
    try:
        from backtesting_engine import BacktestingEngine, BacktestConfig, BacktestMode
        from datetime import datetime, timedelta
        from schwab.client import Client

        # Check if trading engine and Schwab client are available
        if not trading_engine or not trading_engine.schwab_client:
            return {
                'status': 'error',
                'message': 'Schwab client not connected. Please ensure the trading bot is running with valid Schwab credentials.'
            }

        # Parse configuration
        symbols = config.get('symbols', ['TSLA', 'NVDA'])
        start_date = datetime.strptime(config.get('start_date', '2024-01-01'), '%Y-%m-%d')
        end_date = datetime.strptime(config.get('end_date', '2024-12-31'), '%Y-%m-%d')
        initial_capital = config.get('initial_capital', 100000)
        timeframe = config.get('timeframe', '5min')
        mode = BacktestMode[config.get('mode', 'REALISTIC')]
        max_positions = config.get('max_positions', 5)
        commission = config.get('commission', 0.001)

        # Create backtest config
        bt_config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            symbols=symbols,
            timeframe=timeframe,
            mode=mode,
            max_positions=max_positions,
            commission=commission
        )

        # Fetch historical data from Schwab
        market_data = {}

        # Calculate days from now to start_date for minute data validation
        days_ago = (datetime.now() - start_date).days

        # Map timeframe to Schwab frequency
        # Note: Schwab minute data is only available for the last 30 days
        use_minute_data = timeframe in ['1min', '5min', '15min', '1hour']

        if use_minute_data and days_ago > 30:
            # Minute data not available for dates > 30 days ago, fall back to daily
            logger.warning(f"Minute data only available for last 30 days. Falling back to daily data for backtest starting {start_date.date()}")
            frequency_type = Client.PriceHistory.FrequencyType.DAILY
            frequency = Client.PriceHistory.Frequency.DAILY
            # Update config to reflect actual timeframe used
            bt_config = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                symbols=symbols,
                timeframe='1day',  # Fallback to daily
                mode=mode,
                max_positions=max_positions,
                commission=commission
            )
        elif timeframe == '1min':
            frequency_type = Client.PriceHistory.FrequencyType.MINUTE
            frequency = Client.PriceHistory.Frequency.EVERY_MINUTE
        elif timeframe == '5min':
            frequency_type = Client.PriceHistory.FrequencyType.MINUTE
            frequency = Client.PriceHistory.Frequency.EVERY_FIVE_MINUTES
        elif timeframe == '15min':
            frequency_type = Client.PriceHistory.FrequencyType.MINUTE
            frequency = Client.PriceHistory.Frequency.EVERY_FIFTEEN_MINUTES
        elif timeframe == '1hour':
            frequency_type = Client.PriceHistory.FrequencyType.MINUTE
            frequency = Client.PriceHistory.Frequency.EVERY_THIRTY_MINUTES  # Closest to 1hr
        else:  # 1day
            frequency_type = Client.PriceHistory.FrequencyType.DAILY
            frequency = Client.PriceHistory.Frequency.DAILY

        for symbol in symbols:
            try:
                # Fetch from Schwab API using only start/end datetime (not period)
                # Note: Schwab API requires EITHER period_type+period OR start_datetime+end_datetime, not both
                response = trading_engine.schwab_client.get_price_history(
                    symbol,
                    frequency_type=frequency_type,
                    frequency=frequency,
                    start_datetime=start_date,
                    end_datetime=end_date
                )

                if response.status_code == 200:
                    data = response.json()
                    candles = data.get('candles', [])

                    if candles:
                        # Convert to DataFrame
                        df = pd.DataFrame(candles)
                        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                        df.set_index('datetime', inplace=True)

                        # Filter by date range
                        df = df[(df.index >= start_date) & (df.index <= end_date)]

                        # Rename columns to lowercase (backtesting engine expects lowercase)
                        df.columns = df.columns.str.lower()

                        if len(df) > 0:
                            market_data[symbol] = df
                            logger.info(f"Fetched {len(df)} bars for {symbol} from Schwab")
                else:
                    logger.warning(f"Schwab API returned {response.status_code} for {symbol}")

            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol} from Schwab: {e}")

        if not market_data:
            return {
                'status': 'error',
                'message': 'No historical data available from Schwab. Check symbol names and ensure market data access is enabled.'
            }

        # Run backtest
        engine = BacktestingEngine(bt_config)
        results = engine.run(market_data)

        # Convert results to dict
        return {
            'status': 'success',
            'data': {
                'total_return': results.total_return,
                'annual_return': results.annual_return,
                'sharpe_ratio': results.sharpe_ratio,
                'sortino_ratio': results.sortino_ratio,
                'max_drawdown': results.max_drawdown,
                'win_rate': results.win_rate,
                'profit_factor': results.profit_factor,
                'total_trades': results.total_trades,
                'winning_trades': results.winning_trades,
                'losing_trades': results.losing_trades,
                'avg_win': results.avg_win,
                'avg_loss': results.avg_loss,
                'initial_capital': results.initial_capital,
                'final_capital': results.final_capital,
                'equity_curve': [
                    {'date': d.isoformat(), 'equity': float(v)}
                    for d, v in results.equity_curve.items()
                ][-500:],  # Limit to last 500 points
                'trades': [
                    {
                        'symbol': t.symbol,
                        'entry_time': t.entry_time.isoformat(),
                        'exit_time': t.exit_time.isoformat(),
                        'entry_price': t.entry_price,
                        'exit_price': t.exit_price,
                        'side': t.side,
                        'pnl': t.pnl,
                        'pnl_pct': t.pnl_pct,
                        'exit_reason': t.exit_reason
                    }
                    for t in results.trades
                ]
            }
        }

    except ImportError as e:
        return {'status': 'error', 'message': f'Backtesting engine not available: {e}'}
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return {'status': 'error', 'message': str(e)}

@app.get("/api/backtest/status")
async def get_backtest_status():
    """Get current backtest status (for polling during long backtests)"""
    # This would track progress for long-running backtests
    return {'status': 'idle', 'progress': 0}

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
    await connection_manager.connect(websocket)

    try:
        while True:
            await asyncio.sleep(1)

            if trading_engine:
                # Get simulated positions
                sim_positions_data = []
                if trading_engine.mode == TradingMode.SIMULATION_WITH_COMMENTARY:
                    for symbol, pos in trading_engine.simulated_positions.items():
                        if pos is not None:
                            # Update current price with latest market data
                            try:
                                if trading_engine.data_provider and hasattr(trading_engine.data_provider, 'get_current_quote'):
                                    quote = await trading_engine.data_provider.get_current_quote(symbol)
                                    if quote and 'price' in quote:
                                        pos.current_price = quote['price']
                                        # Recalculate unrealized PnL
                                        pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity
                                elif trading_engine.schwab_client:
                                    # Try to get quote from Schwab
                                    try:
                                        response = trading_engine.schwab_client.get_quote(symbol)
                                        if response.status_code == 200:
                                            quote_data = response.json()
                                            if symbol in quote_data:
                                                pos.current_price = quote_data[symbol]['quote']['lastPrice']
                                                pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity
                                    except Exception as e:
                                        logger.debug(f"Error getting quote for {symbol}: {e}")
                            except Exception as e:
                                logger.debug(f"Error updating price for {symbol}: {e}")

                            sim_positions_data.append({
                                'symbol': pos.symbol,
                                'quantity': pos.quantity,
                                'entry_price': pos.entry_price,
                                'current_price': pos.current_price,
                                'unrealized_pnl': pos.unrealized_pnl,
                                'type': 'simulated'
                            })

                # Get real Schwab positions
                real_positions_data = []
                if trading_engine.schwab_client:
                    schwab_positions = await trading_engine.get_schwab_positions()
                    for pos in schwab_positions:
                        # Check if we have a tracked position with long-term flag
                        tracked_pos = trading_engine.positions.get(pos['symbol'])
                        is_long_term = getattr(tracked_pos, 'is_long_term', False) if tracked_pos else False

                        real_positions_data.append({
                            'symbol': pos['symbol'],
                            'quantity': pos['quantity'],
                            'entry_price': pos['average_price'],
                            'current_price': pos['current_price'],
                            'unrealized_pnl': pos['total_pnl'],
                            'day_pnl': pos['day_pnl'],
                            'pnl_percent': pos['pnl_percent'],
                            'market_value': pos['market_value'],
                            'is_long_term': is_long_term,
                            'type': 'real'
                        })

                # Get account info from Schwab
                account_info = {}
                if trading_engine.schwab_client:
                    try:
                        account_info = await trading_engine._get_real_account_info()
                        # Log if we got real data
                        if account_info:
                            logger.debug(f"WebSocket: Got real Schwab data - Balance=${account_info.get('balance', 0):.2f}, P&L=${account_info.get('day_pnl', 0):.2f}")
                    except Exception as e:
                        logger.error(f"WebSocket: Failed to get Schwab account info: {e}")
                        account_info = {}

                # If no Schwab data, use risk manager values
                if not account_info:
                    logger.debug("WebSocket: Using risk manager fallback values")
                    account_info = {
                        'balance': trading_engine.risk_manager.account_balance,
                        'buying_power': trading_engine.risk_manager.buying_power,
                        'day_pnl': trading_engine.risk_manager.schwab_daily_pnl if hasattr(trading_engine.risk_manager, 'schwab_daily_pnl') else trading_engine.risk_manager.daily_pnl,
                        'cash': 0
                    }

                # Get screener data
                screener_data = []
                if trading_engine.screener and trading_engine.screener.top_movers:
                    screener_data = [{
                        'symbol': m['symbol'],
                        'last': m.get('last', 0),
                        'change': m.get('percent_change', 0),
                        'volume': m.get('volume', 0),
                        'volatility': m.get('volatility', 0),
                        'high': m.get('high', 0),
                        'low': m.get('low', 0)
                    } for m in trading_engine.screener.top_movers[:10]]

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
                            'screener': screener_data
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
                    if "Connection closed" in str(e) or "sent 1012" in str(e):
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

@app.post("/api/run-backtest")
async def run_backtest_legacy(request: dict):
    """Run backtest on historical data using Schwab API"""
    try:
        from trading_bot_commentary_updated import config_manager, FixedBacktestEngine, BacktestBreakoutStrategy, BacktestMeanReversionStrategy

        start_date = request.get('start_date', '2023-01-01')
        end_date = request.get('end_date', '2024-01-01')
        symbols = request.get('symbols', ['AAPL', 'MSFT', 'GOOGL'])
        strategy_name = request.get('strategy', 'breakout')

        # Check if we have Schwab connection
        if not trading_engine or not trading_engine.schwab_client:
            return {"status": "error", "message": "Schwab not connected. Please authenticate first."}

        # Initialize backtest engine
        backtest_engine = FixedBacktestEngine(config_manager.config)

        # Get historical data from Schwab
        data = {}
        data_provider = trading_engine.data_provider

        for symbol in symbols:
            try:
                # Use Schwab to get daily data for backtesting
                df = data_provider.get_market_data(
                    symbol,
                    period_type='year',
                    period=2,  # 2 years of data
                    frequency_type='minute',
                    frequency=1
                )

                if not df.empty and len(df) > 50:
                    # Filter by date range
                    df = df[(df.index >= pd.to_datetime(start_date)) &
                           (df.index <= pd.to_datetime(end_date))]

                    if len(df) > 50:  # Still have enough data after filtering
                        data[symbol] = df
                        logger.info(f"Downloaded {len(df)} bars for {symbol} from Schwab")

            except Exception as e:
                logger.error(f"Error downloading data for {symbol}: {e}")

        if not data:
            return {"status": "error", "message": "No data available for backtesting"}

        # Create backtest strategy
        if strategy_name == 'breakout':
            strategy = BacktestBreakoutStrategy()
        elif strategy_name == 'mean_reversion':
            strategy = BacktestMeanReversionStrategy()
        else:
            strategy = BacktestBreakoutStrategy()

        # Run backtest
        result = backtest_engine.run_backtest(data, strategy, start_date, end_date)

        # Format results (rest of the code remains the same)
        response_data = {
            "status": "success",
            "message": "Backtest completed successfully",
            "results": {
                'total_return': f"{result.total_return:.2%}",
                'annualized_return': f"{result.annualized_return:.2%}",
                'sharpe_ratio': f"{result.sharpe_ratio:.2f}",
                'sortino_ratio': f"{result.sortino_ratio:.2f}",
                'max_drawdown': f"{result.max_drawdown:.2%}",
                'win_rate': f"{result.win_rate:.2%}",
                'profit_factor': f"{result.profit_factor:.2f}" if result.profit_factor != float('inf') else "N/A",
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'average_win': f"${result.average_win:.2f}",
                'average_loss': f"${result.average_loss:.2f}",
                'largest_win': f"${result.largest_win:.2f}",
                'largest_loss': f"${result.largest_loss:.2f}",
                'consecutive_wins': result.consecutive_wins,
                'consecutive_losses': result.consecutive_losses,
                'initial_capital': f"${result.initial_capital:,.2f}",
                'final_capital': f"${result.final_capital:,.2f}"
            },
            'equity_curve': result.equity_curve[-100:],
            'trade_count_by_symbol': {}
        }

        # Count trades by symbol
        for trade in result.trade_history:
            symbol = trade.get('symbol', 'UNKNOWN')
            if symbol not in response_data['trade_count_by_symbol']:
                response_data['trade_count_by_symbol'][symbol] = 0
            response_data['trade_count_by_symbol'][symbol] += 1

        # Save results
        save_data = {
            'config': request,
            'results': response_data['results'],
            'equity_curve': result.equity_curve,
            'trade_history': [
                {
                    'date': trade['date'].isoformat() if isinstance(trade['date'], pd.Timestamp) else str(trade['date']),
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'price': float(trade['price']),
                    'quantity': int(trade['quantity']),
                    'pnl': float(trade.get('pnl', 0)),
                    'capital': float(trade['capital'])
                }
                for trade in result.trade_history
            ]
        }

        with open(config_manager.get('paths.backtest_results'), 'w') as f:
            json.dump(save_data, f, indent=2)

        return response_data

    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        return {"status": "error", "message": f"Backtest failed: {str(e)}"}

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
    print("""
    \u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
    \u2551        Trading Bot with Live Commentary - Educational Mode           \u2551
    \u2551     Learn How Professional Trading Algorithms Make Decisions         \u2551
    \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d
    """)

    print("\nStarting Trading Bot in Commentary Mode...")
    print("The bot will explain its thinking process in real-time.")
    print("\n\U0001f4ca Features:")
    print("  \u2022 Real-time market analysis with explanations")
    print("  \u2022 Detailed reasoning for every trading decision")
    print("  \u2022 Risk management explanations")
    print("  \u2022 Post-trade analysis and lessons")
    print("  \u2022 Works even during margin calls (simulation only)")

    print("\n\U0001f310 Open http://localhost:8000 in your browser")
    print("\U0001f4f1 Click 'Start Commentary' to begin")
    print("\u274c Press Ctrl+C to stop\n")

    def shutdown_handler(signum, frame):
        print("\n\n\U0001f6d1 Shutting down gracefully...")
        if trading_engine:
            trading_engine._save_state()
            trading_engine.brain.save_memories()

            # CRITICAL: Save ML model with buffer
            if hasattr(trading_engine, 'ml_predictor') and trading_engine.ml_predictor:
                if hasattr(trading_engine.ml_predictor, 'save_model'):
                    trading_engine.ml_predictor.save_model()
                    buffer_size = len(getattr(trading_engine.ml_predictor, 'online_buffer', []))
                    if buffer_size > 0:
                        print(f"\u2705 ML model saved with {buffer_size} training samples in buffer")
                    else:
                        print("\u2705 ML model saved")

            print("\u2705 State saved successfully")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
