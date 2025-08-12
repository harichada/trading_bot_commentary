#!/usr/bin/env python3
"""
Run the trading bot with professional features
This integrates all the new components with your existing bot
"""

# Import safety module first to prevent segmentation faults
import safe_imports

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger('ProfessionalBot')

# Import your existing bot
from trading_bot_commentary_updated import (
    TradingEngineWithCommentary, 
    Config, 
    TradingMode,
    TradingCommentary,
    CommentaryType,
    app,
    connection_manager
)

# Import professional components
from strategy_system import StrategyManager
from ml_model_manager_safe import ModelManager
from risk_management import RiskManager, PositionSizingMethod
from paper_trading import PaperTradingEngine
from performance_analytics import PerformanceAnalyzer
from advanced_orders import OrderManager

# Import FastAPI components
from fastapi import HTTPException
from typing import Dict
import uvicorn

# Global instances for professional features
strategy_manager = None
model_manager = None
risk_manager = None
paper_engine = None
performance_analyzer = None

class ProfessionalConfig:
    """Professional trading configuration"""
    
    @staticmethod
    def load():
        """Load or create professional configuration"""
        config_file = Path("professional_config.json")
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Create default configuration
            default_config = {
                "professional_mode": True,
                "paper_trading": True,
                "starting_capital": 100000,
                "risk_management": {
                    "position_sizing_method": "VOLATILITY_BASED",
                    "max_portfolio_risk": 0.02,
                    "max_position_size": 0.10,
                    "max_daily_loss": 0.02,
                    "use_circuit_breakers": True
                },
                "strategies": {
                    "enabled": ["ma_cross", "rsi_momentum"],
                    "use_consensus": True,
                    "min_consensus_strategies": 2
                },
                "ml_models": {
                    "enabled": ["random_forest", "gradient_boosting"],
                    "use_ensemble": True
                },
                "advanced_orders": {
                    "use_bracket_orders": True,
                    "use_trailing_stops": True,
                    "default_stop_loss_pct": 2.0,
                    "default_take_profit_pct": 5.0
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config

# Override the original trading engine with professional features
class ProfessionalTradingEngine(TradingEngineWithCommentary):
    """Enhanced trading engine with professional features"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load professional config
        self.prof_config = ProfessionalConfig.load()
        
        # Initialize professional components
        self._init_professional_components()
    
    def _init_professional_components(self):
        """Initialize all professional components"""
        global strategy_manager, model_manager, risk_manager, paper_engine, performance_analyzer
        
        # Strategy Manager
        strategy_manager = StrategyManager()
        self.strategy_manager = strategy_manager
        
        # ML Model Manager
        model_manager = ModelManager()
        self.model_manager = model_manager
        
        # Risk Manager
        initial_capital = self.prof_config['starting_capital']
        risk_manager = RiskManager(initial_capital=initial_capital)
        self.risk_manager = risk_manager
        
        # Performance Analyzer
        performance_analyzer = PerformanceAnalyzer()
        self.performance_analyzer = performance_analyzer
        
        # Paper Trading Engine (if enabled)
        if self.prof_config['paper_trading']:
            paper_engine = PaperTradingEngine(initial_balance=initial_capital)
            paper_engine.load_state()  # Load previous state if exists
            self.paper_engine = paper_engine
            self.mode = TradingMode.SIMULATION_WITH_COMMENTARY
        
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.DECISION,
            symbol=None,
            title='🚀 Professional Mode Activated',
            message=f"Strategies: {', '.join(self.prof_config['strategies']['enabled'])}",
            importance=10
        ))
    
    async def generate_trading_signals(self, market_data):
        """Generate signals using professional strategy system"""
        all_signals = []
        
        for symbol, data in market_data.items():
            # Use strategy manager
            strategy_signals = self.strategy_manager.analyze_all(data, self.positions)
            
            # Apply consensus if configured
            if self.prof_config['strategies']['use_consensus']:
                consensus = self.strategy_manager.get_consensus_signal(strategy_signals)
                if consensus:
                    # Convert to your signal format
                    trading_signal = {
                        'symbol': consensus.symbol,
                        'action': 'BUY' if consensus.signal_type.value == 'BUY' else 'SELL',
                        'confidence': consensus.strength,
                        'strategy': consensus.strategy_name,
                        'entry_price': consensus.entry_price,
                        'stop_loss': consensus.stop_loss,
                        'take_profit': consensus.take_profit
                    }
                    all_signals.append(trading_signal)
            else:
                # Use all signals
                for sig in strategy_signals:
                    trading_signal = {
                        'symbol': sig.symbol,
                        'action': 'BUY' if sig.signal_type.value == 'BUY' else 'SELL',
                        'confidence': sig.strength,
                        'strategy': sig.strategy_name,
                        'entry_price': sig.entry_price,
                        'stop_loss': sig.stop_loss,
                        'take_profit': sig.take_profit
                    }
                    all_signals.append(trading_signal)
        
        # Add commentary about signals
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.SIGNAL_GENERATION,
            symbol=None,
            title=f'📊 Generated {len(all_signals)} Signals',
            message=f"Active strategies: {len(self.strategy_manager.strategies)}",
            data={'signals': len(all_signals)},
            importance=8
        ))
        
        return all_signals
    
    def calculate_position_size(self, signal):
        """Calculate position size using risk manager"""
        method = PositionSizingMethod[
            self.prof_config['risk_management']['position_sizing_method']
        ]
        
        signal_data = {
            'symbol': signal['symbol'],
            'price': signal['entry_price'],
            'volatility': 0.02,  # Would calculate from market data
            'stop_loss_distance': abs(signal['entry_price'] - signal.get('stop_loss', signal['entry_price'] * 0.98))
        }
        
        size = self.risk_manager.calculate_position_size(method, signal_data)
        
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.RISK_ASSESSMENT,
            symbol=None,
            title='📏 Position Size Calculated',
            message=f"{signal['symbol']}: {size:.0f} shares using {method.value}",
            importance=7
        ))
        
        return size
    
    async def execute_signal(self, signal):
        """Execute signal with professional features"""
        if self.prof_config['paper_trading']:
            # Use paper trading engine
            order = {
                'symbol': signal['symbol'],
                'side': 'BUY' if signal['action'] == 'BUY' else 'SELL',
                'quantity': self.calculate_position_size(signal),
                'order_type': 'MARKET'
            }
            
            # Update paper trading market data
            if signal['symbol'] in self.market_data:
                self.paper_engine.update_market_data(
                    signal['symbol'], 
                    self.market_data[signal['symbol']]
                )
            
            # Place order
            from advanced_orders import Order, OrderSide, OrderType
            paper_order = Order(
                symbol=order['symbol'],
                side=OrderSide[order['side']],
                quantity=order['quantity'],
                order_type=OrderType[order['order_type']]
            )
            
            order_id = self.paper_engine.place_order(paper_order)
            
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=None,
                title='📝 Paper Order Placed',
                message=f"{order['side']} {order['quantity']} {order['symbol']}",
                data={'order_id': order_id},
                importance=9
            ))
        else:
            # Use real execution
            await super().execute_signal(signal)
    
    async def update_performance(self):
        """Update performance tracking"""
        if self.paper_engine:
            # Get paper trading results
            summary = self.paper_engine.get_account_summary()
            
            # Update performance analyzer
            for trade in self.paper_engine.account.trades[-10:]:  # Last 10 trades
                if trade not in self.performance_analyzer.trades:
                    self.performance_analyzer.add_trade(trade)
            
            # Update equity
            self.performance_analyzer.update_equity(
                datetime.now(), 
                summary['equity']
            )

# API Endpoints for Professional Features
@app.get("/api/professional/strategies")
async def get_strategies():
    """Get available strategies"""
    if not strategy_manager:
        raise HTTPException(status_code=500, detail="Strategy manager not initialized")
    
    strategies = []
    for key, config in strategy_manager.strategy_configs.items():
        strategies.append({
            'key': key,
            'name': config.name,
            'enabled': config.enabled,
            'weight': config.weight,
            'parameters': config.parameters
        })
    
    return {'strategies': strategies}

@app.post("/api/professional/strategies/{strategy_key}/toggle")
async def toggle_strategy(strategy_key: str):
    """Enable/disable a strategy"""
    if not strategy_manager:
        raise HTTPException(status_code=500, detail="Strategy manager not initialized")
    
    if strategy_key in strategy_manager.strategy_configs:
        current = strategy_manager.strategy_configs[strategy_key].enabled
        if current:
            strategy_manager.disable_strategy(strategy_key)
        else:
            strategy_manager.enable_strategy(strategy_key)
        
        return {'success': True, 'enabled': not current}
    
    raise HTTPException(status_code=404, detail="Strategy not found")

@app.get("/api/professional/performance")
async def get_performance():
    """Get performance metrics"""
    if not performance_analyzer:
        raise HTTPException(status_code=500, detail="Performance analyzer not initialized")
    
    metrics = performance_analyzer.calculate_metrics()
    return {
        'total_return': metrics.total_return,
        'sharpe_ratio': metrics.sharpe_ratio,
        'win_rate': metrics.win_rate,
        'total_trades': metrics.total_trades,
        'max_drawdown': metrics.max_drawdown
    }

@app.get("/api/professional/risk")
async def get_risk_metrics():
    """Get current risk metrics"""
    if not risk_manager:
        raise HTTPException(status_code=500, detail="Risk manager not initialized")
    
    metrics = risk_manager.calculate_risk_metrics()
    return {
        'var_95': metrics.var_95,
        'current_drawdown': metrics.current_drawdown,
        'leverage': metrics.leverage,
        'positions_count': len(risk_manager.positions)
    }

@app.get("/api/professional/paper/account")
async def get_paper_account():
    """Get paper trading account info"""
    if not paper_engine:
        raise HTTPException(status_code=500, detail="Paper trading not enabled")
    
    return paper_engine.get_account_summary()

@app.get("/api/professional/paper/positions")
async def get_paper_positions():
    """Get paper trading positions"""
    if not paper_engine:
        raise HTTPException(status_code=500, detail="Paper trading not enabled")
    
    return paper_engine.get_positions()

@app.post("/api/professional/orders/advanced")
async def place_advanced_order(order_data: Dict):
    """Place an advanced order"""
    if not paper_engine:
        raise HTTPException(status_code=500, detail="Paper trading not enabled")
    
    try:
        from advanced_orders import create_bracket_order, create_trailing_stop, OrderSide
        
        order_type = order_data.get('type', 'market')
        
        if order_type == 'bracket':
            order = create_bracket_order(
                symbol=order_data['symbol'],
                side=OrderSide[order_data['side']],
                quantity=order_data['quantity'],
                entry_price=order_data.get('entry_price', 0),
                take_profit=order_data.get('take_profit', 0),
                stop_loss=order_data.get('stop_loss', 0)
            )
        elif order_type == 'trailing_stop':
            order = create_trailing_stop(
                symbol=order_data['symbol'],
                quantity=order_data['quantity'],
                trail_amount=order_data.get('trail_amount', 2.0),
                trail_type=order_data.get('trail_type', 'percentage')
            )
        else:
            # Default to market order
            from advanced_orders import Order, OrderType
            order = Order(
                symbol=order_data['symbol'],
                side=OrderSide[order_data['side']],
                quantity=order_data['quantity'],
                order_type=OrderType.MARKET
            )
        
        # Update market data for paper trading
        if trading_engine and order_data['symbol'] in trading_engine.market_data:
            paper_engine.update_market_data(
                order_data['symbol'], 
                trading_engine.market_data[order_data['symbol']]
            )
        
        order_id = paper_engine.place_order(order)
        
        return {'success': True, 'order_id': order_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/professional/analysis/multi-timeframe/{symbol}")
async def get_multi_timeframe_analysis(symbol: str):
    """Run multi-timeframe analysis for a symbol"""
    try:
        from multi_timeframe_analysis import MultiTimeframeAnalyzer, Timeframe
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Initialize analyzer
        analyzer = MultiTimeframeAnalyzer()
        
        # Generate sample data for demonstration
        dates = pd.date_range(end=datetime.now(), periods=500, freq='5min')
        prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
        base_data = pd.DataFrame({
            'open': prices + np.random.randn(500) * 0.1,
            'high': prices + np.random.randn(500) * 0.2 + 0.5,
            'low': prices + np.random.randn(500) * 0.2 - 0.5,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 500)
        }, index=dates)
        
        # Prepare data for different timeframes
        timeframe_data = {}
        
        # 5-minute data (base)
        timeframe_data[Timeframe.M5] = base_data.tail(100)
        
        # 15-minute data
        timeframe_data[Timeframe.M15] = base_data.resample('15T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # 1-hour data
        timeframe_data[Timeframe.H1] = base_data.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # 4-hour data
        timeframe_data[Timeframe.H4] = base_data.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Run analysis
        analyses = analyzer.analyze(symbol, timeframe_data)
        
        # Generate multi-timeframe signal
        signal = analyzer.generate_multi_timeframe_signal(symbol, analyses, Timeframe.M15)
        
        # Format response
        response = {
            'symbol': symbol,
            'timeframes': {},
            'recommendation': None
        }
        
        # Add timeframe analysis results
        for tf, analysis in analyses.items():
            response['timeframes'][tf.value] = {
                'trend': analysis.trend.value,
                'strength': analysis.trend_strength,
                'momentum': analysis.momentum
            }
        
        # Add signal recommendation if available
        if signal:
            response['recommendation'] = {
                'signal': signal.signal_type.value,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit[0] if signal.take_profit else signal.entry_price * 1.02
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in multi-timeframe analysis: {e}")
        # Return fallback response
        return {
            'symbol': symbol,
            'error': str(e),
            'timeframes': {
                '5m': {'trend': 'UNKNOWN', 'strength': 0},
                '15m': {'trend': 'UNKNOWN', 'strength': 0},
                '1h': {'trend': 'UNKNOWN', 'strength': 0},
                '4h': {'trend': 'UNKNOWN', 'strength': 0}
            },
            'recommendation': None
        }

@app.post("/api/professional/backtest")
async def run_backtest(config: Dict):
    """Run a backtest with specified configuration"""
    try:
        # Use stub implementation to prevent crashes
        from backtest_stub import run_backtest_stub
        
        logger.info("Using stub backtest implementation to prevent crashes")
        results = await run_backtest_stub(config)
        return results
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        # Return mock results even on error
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.5,
            'total_trades': 0,
            'profit_factor': 1.0,
            'error': str(e),
            'message': 'Backtesting temporarily disabled due to stability issues'
        }

# Direct route to enhanced dashboard
@app.get("/enhanced")
async def get_enhanced_dashboard():
    """Serve the enhanced dashboard directly"""
    from fastapi.responses import HTMLResponse
    
    enhanced_dashboard = Path("professional_dashboard_enhanced.html")
    if enhanced_dashboard.exists():
        with open(enhanced_dashboard, 'r') as f:
            content = f.read()
            return HTMLResponse(
                content=content,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
    else:
        return {"error": "Enhanced dashboard not found"}

# Get professional config endpoint
@app.get("/api/professional/config")
async def get_professional_config():
    """Get professional trading configuration"""
    config = ProfessionalConfig.load()
    return config

# Test endpoint to verify dashboard version
@app.get("/api/dashboard-version")
async def get_dashboard_version():
    """Check which dashboard version is being served"""
    enhanced_dashboard = Path("professional_dashboard_enhanced.html")
    if enhanced_dashboard.exists():
        return {"version": "enhanced_v2", "file": str(enhanced_dashboard)}
    
    dashboard_file = Path("professional_dashboard.html")
    if dashboard_file.exists():
        return {"version": "professional_v1", "file": str(dashboard_file)}
    
    return {"version": "original", "file": "embedded"}

# Override the dashboard to show professional features
@app.get("/")
async def get_dashboard():
    """Serve the professional dashboard"""
    from fastapi.responses import HTMLResponse
    
    # Try enhanced dashboard first
    enhanced_dashboard = Path("professional_dashboard_enhanced.html")
    if enhanced_dashboard.exists():
        print(f"Serving enhanced dashboard from: {enhanced_dashboard}")
        with open(enhanced_dashboard, 'r') as f:
            content = f.read()
            # Add no-cache headers
            return HTMLResponse(
                content=content,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
    
    # Then try original professional dashboard
    dashboard_file = Path("professional_dashboard.html")
    if dashboard_file.exists():
        print(f"Serving original professional dashboard from: {dashboard_file}")
        with open(dashboard_file, 'r') as f:
            return HTMLResponse(content=f.read())
    else:
        # Fallback to original dashboard
        print("Serving fallback dashboard")
        from trading_bot_commentary_updated import DASHBOARD_HTML_WITH_COMMENTARY
        return HTMLResponse(content=DASHBOARD_HTML_WITH_COMMENTARY)

async def main():
    """Main entry point"""
    print("\n" + "="*50)
    print("PROFESSIONAL TRADING BOT")
    print("="*50)
    
    # Load configuration
    config = ProfessionalConfig.load()
    print(f"Mode: {'Paper Trading' if config['paper_trading'] else 'Live Trading'}")
    print(f"Strategies: {', '.join(config['strategies']['enabled'])}")
    print(f"Risk Management: {config['risk_management']['position_sizing_method']}")
    
    # Create professional trading engine
    global trading_engine
    trading_engine = ProfessionalTradingEngine(connection_manager=connection_manager)
    
    print("\nStarting server...")
    print("Dashboard: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    
    # Run the FastAPI server
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    # Run the professional bot
    asyncio.run(main())