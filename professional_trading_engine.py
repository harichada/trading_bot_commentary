#!/usr/bin/env python3
"""
Professional Trading Engine Integration
Brings together strategies, ML models, and advanced orders
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import pandas as pd
import numpy as np

# Import our new components
from strategy_system import StrategyManager, StrategySignal, SignalType
from ml_model_manager import ModelManager
from advanced_orders import (
    OrderManager, Order, OrderType, OrderSide, OrderStatus,
    create_bracket_order, create_trailing_stop, create_dca_order
)

# Import existing components from the main bot
import sys
sys.path.append(str(Path(__file__).parent))

logger = logging.getLogger('ProfessionalTradingEngine')

class ProfessionalTradingEngine:
    """Enhanced trading engine with professional features"""
    
    def __init__(self, config_path: str = "professional_config.json"):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.strategy_manager = StrategyManager()
        self.model_manager = ModelManager()
        self.order_manager = OrderManager(self._execute_order_callback)
        
        # State management
        self.positions: Dict[str, Dict] = {}
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.is_running = False
        
        # Performance tracking
        self.trades_history = []
        self.performance_metrics = {}
        
        # Risk management
        self.risk_limits = self.config.get('risk_limits', {
            'max_positions': 5,
            'max_position_size': 0.1,  # 10% of portfolio
            'max_daily_loss': 0.02,    # 2% daily loss limit
            'max_drawdown': 0.05       # 5% max drawdown
        })
        
        # Trading mode
        self.paper_trading = self.config.get('paper_trading', True)
        self.starting_capital = self.config.get('starting_capital', 100000)
        self.current_capital = self.starting_capital
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            default_config = {
                'symbols': ['SPY', 'QQQ', 'IWM'],
                'timeframe': '5min',
                'paper_trading': True,
                'starting_capital': 100000,
                'risk_limits': {
                    'max_positions': 5,
                    'max_position_size': 0.1,
                    'max_daily_loss': 0.02,
                    'max_drawdown': 0.05
                },
                'strategies': {
                    'enabled': ['ma_cross', 'rsi_momentum', 'bollinger_bands'],
                    'use_consensus': True,
                    'min_consensus_strategies': 2
                },
                'ml_models': {
                    'enabled': ['random_forest', 'xgboost'],
                    'use_ensemble': True,
                    'retrain_interval_days': 7
                },
                'order_types': {
                    'use_bracket_orders': True,
                    'use_trailing_stops': True,
                    'default_stop_loss': 0.02,
                    'default_take_profit': 0.05
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
    
    async def start(self):
        """Start the trading engine"""
        self.is_running = True
        logger.info("Professional Trading Engine started")
        
        # Load ML models if they exist
        self._load_models()
        
        # Main trading loop
        while self.is_running:
            try:
                # Update market data
                await self._update_market_data()
                
                # Check risk limits
                if not self._check_risk_limits():
                    logger.warning("Risk limits breached, pausing trading")
                    await asyncio.sleep(60)
                    continue
                
                # Run strategies
                signals = await self._run_strategies()
                
                # Filter signals with ML models
                if self.config['ml_models']['enabled']:
                    signals = await self._filter_signals_with_ml(signals)
                
                # Execute signals
                for signal in signals:
                    await self._process_signal(signal)
                
                # Update trailing stops
                self._update_trailing_stops()
                
                # Check for model retraining
                if self._should_retrain_models():
                    await self._retrain_models()
                
                await asyncio.sleep(60)  # 1 minute loop
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _update_market_data(self):
        """Update market data for all symbols"""
        for symbol in self.config['symbols']:
            # In real implementation, fetch from data provider
            # For now, create sample data
            if symbol not in self.market_data:
                self.market_data[symbol] = self._generate_sample_data(symbol)
            else:
                # Add new candle
                new_candle = self._generate_new_candle(self.market_data[symbol])
                self.market_data[symbol] = pd.concat([
                    self.market_data[symbol], 
                    new_candle
                ]).tail(1000)  # Keep last 1000 candles
    
    async def _run_strategies(self) -> List[StrategySignal]:
        """Run all enabled strategies"""
        all_signals = []
        
        for symbol, data in self.market_data.items():
            # Run strategies
            signals = self.strategy_manager.analyze_all(data, self.positions)
            
            # Get consensus if enabled
            if self.config['strategies']['use_consensus']:
                consensus = self.strategy_manager.get_consensus_signal(signals)
                if consensus:
                    all_signals.append(consensus)
            else:
                all_signals.extend(signals)
        
        return all_signals
    
    async def _filter_signals_with_ml(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """Filter signals using ML models"""
        filtered_signals = []
        
        for signal in signals:
            # Prepare features for ML model
            features = self._prepare_ml_features(signal.symbol)
            
            if features is not None:
                # Get ML prediction
                if self.config['ml_models']['use_ensemble']:
                    prediction = self.model_manager.predict(features, 'ensemble')
                    proba = self.model_manager.predict_proba(features, 'ensemble')
                else:
                    # Use first enabled model
                    model_key = self.config['ml_models']['enabled'][0]
                    prediction = self.model_manager.predict(features, model_key)
                    proba = self.model_manager.predict_proba(features, model_key)
                
                # Filter based on ML confidence
                ml_confidence = proba[0][1] if signal.signal_type == SignalType.BUY else proba[0][0]
                
                if ml_confidence > 0.6:  # Confidence threshold
                    signal.strength *= ml_confidence  # Adjust signal strength
                    signal.metadata['ml_confidence'] = ml_confidence
                    filtered_signals.append(signal)
                else:
                    logger.info(f"Signal filtered by ML: {signal.symbol} {signal.signal_type} "
                              f"(ML confidence: {ml_confidence:.2f})")
        
        return filtered_signals
    
    async def _process_signal(self, signal: StrategySignal):
        """Process a trading signal"""
        # Check position limits
        if len(self.positions) >= self.risk_limits['max_positions']:
            logger.warning(f"Max positions reached, skipping signal for {signal.symbol}")
            return
        
        # Calculate position size
        position_size = self._calculate_position_size(signal)
        
        if position_size == 0:
            return
        
        # Create order based on configuration
        if self.config['order_types']['use_bracket_orders']:
            order = create_bracket_order(
                symbol=signal.symbol,
                side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
                quantity=position_size,
                entry_price=signal.entry_price,
                take_profit=signal.take_profit or signal.entry_price * 1.05,
                stop_loss=signal.stop_loss or signal.entry_price * 0.98
            )
        elif self.config['order_types']['use_trailing_stops']:
            # Create market order with trailing stop
            entry_order = Order(
                symbol=signal.symbol,
                side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
                quantity=position_size,
                order_type=OrderType.MARKET
            )
            
            order_id = self.order_manager.place_order(entry_order)
            
            # Add trailing stop
            trailing_stop = create_trailing_stop(
                symbol=signal.symbol,
                side=OrderSide.SELL if signal.signal_type == SignalType.BUY else OrderSide.BUY,
                quantity=position_size,
                trail_percent=0.02  # 2% trailing stop
            )
            
            self.order_manager.place_order(trailing_stop)
            return
        else:
            # Simple market order
            order = Order(
                symbol=signal.symbol,
                side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
                quantity=position_size,
                order_type=OrderType.MARKET
            )
        
        # Place order
        order_id = self.order_manager.place_order(order)
        
        # Update position tracking
        self.positions[signal.symbol] = {
            'quantity': position_size,
            'entry_price': signal.entry_price,
            'entry_time': datetime.now(),
            'signal': signal,
            'order_id': order_id
        }
    
    def _calculate_position_size(self, signal: StrategySignal) -> float:
        """Calculate position size based on risk management"""
        # Kelly Criterion or fixed fractional
        max_position_value = self.current_capital * self.risk_limits['max_position_size']
        
        # Adjust based on signal strength
        position_value = max_position_value * signal.strength
        
        # Calculate shares
        shares = position_value / signal.entry_price
        
        return round(shares, 2)
    
    def _check_risk_limits(self) -> bool:
        """Check if we're within risk limits"""
        # Calculate daily P&L
        daily_pnl = sum(pos.get('pnl', 0) for pos in self.positions.values())
        daily_return = daily_pnl / self.starting_capital
        
        if daily_return < -self.risk_limits['max_daily_loss']:
            logger.warning(f"Daily loss limit hit: {daily_return:.2%}")
            return False
        
        # Check drawdown
        if self.current_capital < self.starting_capital * (1 - self.risk_limits['max_drawdown']):
            logger.warning(f"Max drawdown hit: {(1 - self.current_capital/self.starting_capital):.2%}")
            return False
        
        return True
    
    def _update_trailing_stops(self):
        """Update all trailing stop orders"""
        for symbol, data in self.market_data.items():
            if len(data) > 0:
                current_price = data.iloc[-1]['close']
                self.order_manager.update_market_data(symbol, current_price)
    
    def _should_retrain_models(self) -> bool:
        """Check if models need retraining"""
        # In real implementation, check last training date
        return False
    
    async def _retrain_models(self):
        """Retrain ML models with recent data"""
        logger.info("Retraining ML models...")
        
        # Prepare training data
        X, y = self._prepare_training_data()
        
        if X is not None and len(X) > 1000:
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train all models
            self.model_manager.train_all_models(
                X_train, y_train, 
                validation_data=(X_val, y_val)
            )
            
            # Create ensemble
            self.model_manager.create_ensemble(
                self.config['ml_models']['enabled']
            )
            
            # Save models
            self.model_manager.save_all_models()
            
            logger.info("Model retraining completed")
    
    def _prepare_ml_features(self, symbol: str) -> Optional[pd.DataFrame]:
        """Prepare features for ML prediction"""
        if symbol not in self.market_data:
            return None
        
        data = self.market_data[symbol]
        if len(data) < 100:
            return None
        
        # Calculate technical indicators
        features = pd.DataFrame(index=[data.index[-1]])
        
        # Price features
        features['returns_1'] = data['close'].pct_change(1).iloc[-1]
        features['returns_5'] = data['close'].pct_change(5).iloc[-1]
        features['returns_20'] = data['close'].pct_change(20).iloc[-1]
        
        # Moving averages
        features['sma_ratio'] = data['close'].iloc[-1] / data['close'].rolling(20).mean().iloc[-1]
        features['ema_ratio'] = data['close'].iloc[-1] / data['close'].ewm(span=20).mean().iloc[-1]
        
        # Volatility
        features['volatility'] = data['close'].pct_change().rolling(20).std().iloc[-1]
        
        # Volume
        features['volume_ratio'] = data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]
        
        # RSI
        features['rsi'] = ta.momentum.RSIIndicator(close=data['close']).rsi().iloc[-1]
        
        # MACD
        macd = ta.trend.MACD(close=data['close'])
        features['macd_signal'] = (macd.macd() - macd.macd_signal()).iloc[-1]
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=data['close'])
        features['bb_position'] = (data['close'].iloc[-1] - bb.bollinger_lband().iloc[-1]) / (
            bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]
        )
        
        return features.fillna(0)
    
    def _prepare_training_data(self) -> tuple:
        """Prepare data for model training"""
        # In real implementation, load historical data and create features
        # For now, return None
        return None, None
    
    def _load_models(self):
        """Load saved ML models"""
        model_dir = Path("models")
        if model_dir.exists():
            for model_key in self.config['ml_models']['enabled']:
                model_path = model_dir / f"{model_key}_model.pkl"
                if model_path.exists():
                    try:
                        self.model_manager.load_model(model_key, str(model_path))
                        logger.info(f"Loaded model: {model_key}")
                    except Exception as e:
                        logger.error(f"Error loading model {model_key}: {e}")
            
            # Create ensemble if models are loaded
            if self.config['ml_models']['use_ensemble']:
                try:
                    self.model_manager.create_ensemble(
                        self.config['ml_models']['enabled']
                    )
                    self.model_manager.set_active_model('ensemble')
                except Exception as e:
                    logger.error(f"Error creating ensemble: {e}")
    
    def _execute_order_callback(self, order: Order) -> Dict[str, Any]:
        """Callback for order execution"""
        if self.paper_trading:
            # Simulate execution
            return {
                'success': True,
                'fill_price': order.price or self._get_current_price(order.symbol),
                'filled_quantity': order.quantity
            }
        else:
            # Real broker execution would go here
            pass
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        if symbol in self.market_data and len(self.market_data[symbol]) > 0:
            return self.market_data[symbol].iloc[-1]['close']
        return 100.0
    
    def _generate_sample_data(self, symbol: str) -> pd.DataFrame:
        """Generate sample market data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='5min')
        
        # Random walk
        price = 100
        prices = []
        volumes = []
        
        for _ in range(1000):
            price *= (1 + np.random.normal(0, 0.001))
            prices.append(price)
            volumes.append(np.random.randint(1000000, 5000000))
        
        data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        data.index.name = symbol
        return data
    
    def _generate_new_candle(self, existing_data: pd.DataFrame) -> pd.DataFrame:
        """Generate a new candle for testing"""
        last_close = existing_data.iloc[-1]['close']
        new_close = last_close * (1 + np.random.normal(0, 0.001))
        
        new_candle = pd.DataFrame({
            'open': [last_close],
            'high': [new_close * 1.001],
            'low': [new_close * 0.999],
            'close': [new_close],
            'volume': [np.random.randint(1000000, 5000000)]
        }, index=[existing_data.index[-1] + pd.Timedelta(minutes=5)])
        
        new_candle.index.name = existing_data.index.name
        return new_candle
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not self.trades_history:
            return {}
        
        trades_df = pd.DataFrame(self.trades_history)
        
        # Calculate metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if winning_trades < total_trades else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Sharpe ratio (simplified)
        returns = trades_df['pnl'] / self.starting_capital
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'current_capital': self.current_capital,
            'total_return': (self.current_capital - self.starting_capital) / self.starting_capital
        }

# API endpoints for configuration
def create_api_endpoints(app, engine: ProfessionalTradingEngine):
    """Create FastAPI endpoints for the professional trading engine"""
    
    @app.get("/api/strategies")
    async def get_strategies():
        """Get all available strategies"""
        return {
            'strategies': [
                {
                    'key': key,
                    'config': config.__dict__
                }
                for key, config in engine.strategy_manager.strategy_configs.items()
            ]
        }
    
    @app.post("/api/strategies/{strategy_key}/toggle")
    async def toggle_strategy(strategy_key: str):
        """Enable/disable a strategy"""
        if strategy_key in engine.strategy_manager.strategy_configs:
            current = engine.strategy_manager.strategy_configs[strategy_key].enabled
            if current:
                engine.strategy_manager.disable_strategy(strategy_key)
            else:
                engine.strategy_manager.enable_strategy(strategy_key)
            return {'success': True, 'enabled': not current}
        return {'success': False, 'error': 'Strategy not found'}
    
    @app.post("/api/strategies/{strategy_key}/config")
    async def update_strategy_config(strategy_key: str, config: dict):
        """Update strategy configuration"""
        if strategy_key in engine.strategy_manager.strategy_configs:
            current_config = engine.strategy_manager.strategy_configs[strategy_key]
            
            # Update parameters
            if 'parameters' in config:
                current_config.parameters.update(config['parameters'])
            if 'risk_parameters' in config:
                current_config.risk_parameters.update(config['risk_parameters'])
            if 'weight' in config:
                current_config.weight = config['weight']
            
            engine.strategy_manager.update_strategy_config(strategy_key, current_config)
            return {'success': True}
        return {'success': False, 'error': 'Strategy not found'}
    
    @app.get("/api/models")
    async def get_models():
        """Get all ML models"""
        return {
            'models': engine.model_manager.get_model_info().to_dict('records')
        }
    
    @app.post("/api/models/{model_key}/activate")
    async def activate_model(model_key: str):
        """Set active ML model"""
        try:
            engine.model_manager.set_active_model(model_key)
            return {'success': True}
        except ValueError:
            return {'success': False, 'error': 'Model not found'}
    
    @app.get("/api/performance")
    async def get_performance():
        """Get performance metrics"""
        return engine.get_performance_metrics()
    
    @app.get("/api/positions")
    async def get_positions():
        """Get current positions"""
        return {'positions': engine.positions}
    
    @app.get("/api/orders")
    async def get_orders():
        """Get active orders"""
        return {
            'orders': [
                order.__dict__ for order in engine.order_manager.get_active_orders()
            ]
        }
    
    @app.post("/api/orders/dca")
    async def create_dca_order(request: dict):
        """Create a DCA order"""
        order = create_dca_order(
            symbol=request['symbol'],
            side=OrderSide[request['side']],
            total_amount=request['total_amount'],
            num_orders=request.get('num_orders', 10),
            interval_hours=request.get('interval_hours', 1)
        )
        
        order_id = engine.order_manager.place_order(order)
        return {'success': True, 'order_id': order_id}