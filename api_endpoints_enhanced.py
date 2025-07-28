"""
Enhanced API endpoints for professional features
Add these to run_professional_bot.py to support all dashboard features
"""

from fastapi import HTTPException
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

# Add these endpoints to run_professional_bot.py

@app.get("/api/professional/config")
async def get_professional_config():
    """Get professional trading configuration"""
    config = ProfessionalConfig.load()
    return config

@app.get("/api/professional/models")
async def get_ml_models():
    """Get available ML models with their status"""
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    model_info = model_manager.get_model_info()
    models = []
    
    for _, row in model_info.iterrows():
        models.append({
            'key': row['key'],
            'name': row['name'],
            'algorithm': row['algorithm'],
            'trained': row['trained'],
            'active': row['active'],
            'performance': row['performance'] if row['performance'] else {}
        })
    
    return {'models': models}

@app.post("/api/professional/models/{model_key}/select")
async def select_model(model_key: str):
    """Select an ML model as active"""
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    try:
        model_manager.set_active_model(model_key)
        return {'success': True, 'active_model': model_key}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/professional/risk/settings")
async def update_risk_settings(settings: Dict):
    """Update risk management settings"""
    if not risk_manager:
        raise HTTPException(status_code=500, detail="Risk manager not initialized")
    
    # Update risk manager settings
    if 'position_sizing_method' in settings:
        risk_manager.default_method = PositionSizingMethod[settings['position_sizing_method']]
    
    if 'max_position_size' in settings:
        risk_manager.risk_limits.max_position_size = settings['max_position_size']
    
    if 'max_portfolio_risk' in settings:
        risk_manager.risk_limits.max_portfolio_risk = settings['max_portfolio_risk']
    
    if 'max_daily_loss' in settings:
        risk_manager.risk_limits.max_daily_loss = settings['max_daily_loss']
    
    # Save to config
    prof_config = ProfessionalConfig.load()
    prof_config['risk_management'].update(settings)
    
    with open('professional_config.json', 'w') as f:
        json.dump(prof_config, f, indent=2)
    
    return {'success': True, 'settings': settings}

@app.get("/api/professional/analysis/multi-timeframe/{symbol}")
async def get_multi_timeframe_analysis(symbol: str):
    """Run multi-timeframe analysis for a symbol"""
    # This would need to be implemented with real market data
    # For now, return mock data
    return {
        'symbol': symbol,
        'timeframes': {
            '5m': {'trend': 'BULLISH', 'strength': 0.7},
            '15m': {'trend': 'BULLISH', 'strength': 0.6},
            '1h': {'trend': 'NEUTRAL', 'strength': 0.5},
            '4h': {'trend': 'BEARISH', 'strength': 0.4}
        },
        'recommendation': {
            'signal': 'BUY',
            'confidence': 0.65,
            'entry_price': 450.50,
            'stop_loss': 445.00,
            'take_profit': 460.00
        }
    }

@app.post("/api/professional/backtest")
async def run_backtest(config: Dict):
    """Run a backtest with specified configuration"""
    try:
        from backtesting_engine import BacktestingEngine, BacktestConfig
        
        # Create backtest config
        backtest_config = BacktestConfig(
            start_date=datetime.fromisoformat(config['start_date']),
            end_date=datetime.fromisoformat(config['end_date']),
            initial_capital=config['initial_capital'],
            symbols=config['symbols'],
            commission=config.get('commission', 0.001)
        )
        
        # This would need real market data
        # For now, return mock results
        return {
            'total_return': 0.15,
            'annual_return': 0.18,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'win_rate': 0.55,
            'total_trades': 45,
            'profit_factor': 1.4,
            'report_url': '/api/professional/backtest/report/latest'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/professional/orders/advanced")
async def place_advanced_order(order_data: Dict):
    """Place an advanced order"""
    if not paper_engine:
        raise HTTPException(status_code=500, detail="Paper trading not enabled")
    
    try:
        from advanced_orders import create_bracket_order, create_trailing_stop
        
        order_type = order_data['type']
        
        if order_type == 'bracket':
            order = create_bracket_order(
                symbol=order_data['symbol'],
                side=OrderSide[order_data['side']],
                quantity=order_data['quantity'],
                entry_price=order_data['entry_price'],
                take_profit=order_data['take_profit'],
                stop_loss=order_data['stop_loss']
            )
        elif order_type == 'trailing_stop':
            order = create_trailing_stop(
                symbol=order_data['symbol'],
                quantity=order_data['quantity'],
                trail_amount=order_data['trail_amount'],
                trail_type=order_data['trail_type']
            )
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
        
        order_id = paper_engine.place_order(order)
        
        return {'success': True, 'order_id': order_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/professional/positions/{symbol}/close")
async def close_position(symbol: str):
    """Close a position"""
    if not paper_engine:
        raise HTTPException(status_code=500, detail="Paper trading not enabled")
    
    try:
        positions = paper_engine.get_positions()
        if symbol not in positions:
            raise HTTPException(status_code=404, detail="Position not found")
        
        position = positions[symbol]
        
        # Create a market order to close the position
        from advanced_orders import Order, OrderSide, OrderType
        close_order = Order(
            symbol=symbol,
            side=OrderSide.SELL if position['side'] == 'LONG' else OrderSide.BUY,
            quantity=position['quantity'],
            order_type=OrderType.MARKET
        )
        
        order_id = paper_engine.place_order(close_order)
        
        return {'success': True, 'order_id': order_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Strategy configuration endpoint
@app.get("/api/professional/strategies/{strategy_key}/config")
async def get_strategy_config(strategy_key: str):
    """Get configuration for a specific strategy"""
    if not strategy_manager:
        raise HTTPException(status_code=500, detail="Strategy manager not initialized")
    
    if strategy_key not in strategy_manager.strategy_configs:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    config = strategy_manager.strategy_configs[strategy_key]
    return {
        'key': strategy_key,
        'name': config.name,
        'parameters': config.parameters,
        'enabled': config.enabled,
        'weight': config.weight
    }

@app.post("/api/professional/strategies/{strategy_key}/config")
async def update_strategy_config(strategy_key: str, config_update: Dict):
    """Update strategy configuration"""
    if not strategy_manager:
        raise HTTPException(status_code=500, detail="Strategy manager not initialized")
    
    if strategy_key not in strategy_manager.strategy_configs:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    config = strategy_manager.strategy_configs[strategy_key]
    
    if 'parameters' in config_update:
        config.parameters.update(config_update['parameters'])
    
    if 'weight' in config_update:
        config.weight = config_update['weight']
    
    return {'success': True, 'config': {
        'key': strategy_key,
        'parameters': config.parameters,
        'weight': config.weight
    }}