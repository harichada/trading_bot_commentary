# Professional Trading Bot - Run and Test Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Testing Individual Components](#testing-individual-components)
3. [Integration with Main Bot](#integration-with-main-bot)
4. [Running Different Modes](#running-different-modes)
5. [Testing Strategies](#testing-strategies)
6. [Performance Analysis](#performance-analysis)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install manually
pip install numpy pandas scipy ta scikit-learn xgboost lightgbm matplotlib seaborn
pip install fastapi uvicorn websockets requests textblob pyyaml rich
pip install schwab-py torch  # torch is optional for neural networks
```

### 2. Create Configuration File

Create `professional_config.json`:

```json
{
  "symbols": ["SPY", "QQQ", "IWM"],
  "timeframe": "5min",
  "paper_trading": true,
  "starting_capital": 100000,
  "risk_limits": {
    "max_positions": 5,
    "max_position_size": 0.1,
    "max_daily_loss": 0.02,
    "max_drawdown": 0.05
  },
  "strategies": {
    "enabled": ["ma_cross", "rsi_momentum", "bollinger_bands"],
    "use_consensus": true,
    "min_consensus_strategies": 2
  },
  "ml_models": {
    "enabled": ["random_forest", "xgboost"],
    "use_ensemble": true,
    "retrain_interval_days": 7
  },
  "order_types": {
    "use_bracket_orders": true,
    "use_trailing_stops": true,
    "default_stop_loss": 0.02,
    "default_take_profit": 0.05
  }
}
```

### 3. Update Main Configuration

Edit your `Config().yaml` to enable professional mode:

```yaml
# Professional Trading Features
professional_mode: true
use_advanced_orders: true
use_multi_timeframe: true
use_ml_models: true

# Risk Management
risk_management:
  position_sizing_method: "VOLATILITY_BASED"  # or "KELLY_CRITERION", "ATR_BASED"
  max_portfolio_risk: 0.02
  use_circuit_breakers: true
```

## Testing Individual Components

### Test 1: Strategy System

```python
#!/usr/bin/env python3
"""Test the strategy system"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategy_system import StrategyManager, StrategyConfig

# Create sample data
dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
data = pd.DataFrame({
    'open': np.random.randn(100).cumsum() + 100,
    'high': np.random.randn(100).cumsum() + 101,
    'low': np.random.randn(100).cumsum() + 99,
    'close': np.random.randn(100).cumsum() + 100,
    'volume': np.random.randint(1000000, 5000000, 100)
}, index=dates)

# Initialize strategy manager
manager = StrategyManager()

# Test individual strategies
print("Testing Moving Average Cross Strategy...")
signals = manager.strategies['ma_cross'].analyze(data, {})
for signal in signals:
    print(f"Signal: {signal.signal_type} at {signal.entry_price:.2f}")

# Test consensus signals
print("\nTesting Consensus Signals...")
all_signals = manager.analyze_all(data, {})
consensus = manager.get_consensus_signal(all_signals)
if consensus:
    print(f"Consensus: {consensus.signal_type} with strength {consensus.strength:.2f}")

# Save configuration
manager.save_configs("strategy_configs.json")
```

### Test 2: ML Models

```python
#!/usr/bin/env python3
"""Test ML model manager"""

from ml_model_manager import ModelManager
import pandas as pd
import numpy as np

# Create sample training data
n_samples = 1000
X = pd.DataFrame({
    'returns_1': np.random.randn(n_samples),
    'returns_5': np.random.randn(n_samples),
    'returns_20': np.random.randn(n_samples),
    'rsi': np.random.uniform(20, 80, n_samples),
    'volume_ratio': np.random.uniform(0.5, 2, n_samples)
})

# Create target (1 for profitable, 0 for not)
y = pd.Series(np.random.randint(0, 2, n_samples))

# Initialize model manager
manager = ModelManager()

# Train all models
print("Training models...")
manager.train_all_models(X, y)

# Compare performance
print("\nModel Comparison:")
comparison = manager.compare_models(X, y)
print(comparison)

# Create ensemble
print("\nCreating ensemble...")
ensemble = manager.create_ensemble(['random_forest', 'xgboost'])
manager.set_active_model('ensemble')

# Make predictions
predictions = manager.predict(X.head(10))
print(f"\nSample predictions: {predictions}")

# Save models
manager.save_all_models()
```

### Test 3: Paper Trading

```python
#!/usr/bin/env python3
"""Test paper trading system"""

from paper_trading import PaperTradingEngine, ExecutionModel
from advanced_orders import Order, OrderType, OrderSide
import pandas as pd

# Initialize paper trading
engine = PaperTradingEngine(
    initial_balance=100000,
    execution_model=ExecutionModel.REALISTIC
)

# Load saved state (if exists)
engine.load_state()

# Create sample market data
data = pd.DataFrame({
    'open': [100, 101, 102],
    'high': [101, 102, 103],
    'low': [99, 100, 101],
    'close': [101, 102, 102.5],
    'volume': [1000000, 1200000, 900000]
})

# Update market data
engine.update_market_data('SPY', data)

# Place a market order
order = Order(
    symbol='SPY',
    side=OrderSide.BUY,
    quantity=100,
    order_type=OrderType.MARKET
)
order_id = engine.place_order(order)
print(f"Order placed: {order_id}")

# Check account
summary = engine.get_account_summary()
print(f"\nAccount Summary:")
print(f"Balance: ${summary['balance']:,.2f}")
print(f"Equity: ${summary['equity']:,.2f}")
print(f"Positions: {summary['positions']}")

# Save state
engine.save_state()
```

### Test 4: Backtesting

```python
#!/usr/bin/env python3
"""Run a backtest"""

from backtesting_engine import BacktestingEngine, BacktestConfig, BacktestReport
from strategy_system import StrategyManager
from datetime import datetime, timedelta
import pandas as pd

# Configure backtest
config = BacktestConfig(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005,
    symbols=['SPY', 'QQQ']
)

# Initialize backtesting engine
engine = BacktestingEngine(config)

# Load historical data (you'll need to provide this)
# For testing, create synthetic data
market_data = {}
for symbol in config.symbols:
    dates = pd.date_range(start=config.start_date, end=config.end_date, freq='5min')
    prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.0001)
    
    market_data[symbol] = pd.DataFrame({
        'open': prices * (1 + np.random.randn(len(dates)) * 0.001),
        'high': prices * (1 + abs(np.random.randn(len(dates)) * 0.002)),
        'low': prices * (1 - abs(np.random.randn(len(dates)) * 0.002)),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)

# Run backtest
print("Running backtest...")
results = engine.run(market_data)

# Generate report
BacktestReport.generate_html_report(results, "backtest_report.html")
BacktestReport.generate_json_report(results, "backtest_results.json")

print(f"\nBacktest Results:")
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Win Rate: {results.win_rate:.2%}")
print(f"\nReport saved to backtest_report.html")
```

## Integration with Main Bot

### Step 1: Create Integration Module

Create `professional_integration.py`:

```python
#!/usr/bin/env python3
"""Integration module for professional features"""

from trading_bot_commentary_updated import TradingEngineWithCommentary
from strategy_system import StrategyManager
from ml_model_manager import ModelManager
from risk_management import RiskManager, PositionSizingMethod
from paper_trading import PaperTradingEngine
from performance_analytics import PerformanceAnalyzer
import asyncio

class ProfessionalTradingBot(TradingEngineWithCommentary):
    """Enhanced trading bot with professional features"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize professional components
        self.strategy_manager = StrategyManager()
        self.model_manager = ModelManager()
        self.risk_manager = RiskManager(
            initial_capital=self.account_balance,
            risk_limits=self.config.get('risk_limits', {})
        )
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Paper trading mode
        if self.config.get('paper_trading', True):
            self.paper_engine = PaperTradingEngine()
            self.paper_engine.load_state()
    
    async def generate_trading_signals(self, market_data):
        """Override to use professional strategy system"""
        all_signals = []
        
        for symbol, data in market_data.items():
            # Multi-timeframe analysis if enabled
            if self.config.get('use_multi_timeframe', False):
                # Would need to implement timeframe data collection
                pass
            
            # Get strategy signals
            strategy_signals = self.strategy_manager.analyze_all(data, self.positions)
            
            # Get consensus if configured
            if self.config['strategies'].get('use_consensus', False):
                consensus = self.strategy_manager.get_consensus_signal(strategy_signals)
                if consensus:
                    all_signals.append(consensus)
            else:
                all_signals.extend(strategy_signals)
        
        # Filter with ML if enabled
        if self.config['ml_models'].get('enabled'):
            all_signals = self._filter_signals_with_ml(all_signals)
        
        return all_signals
    
    def calculate_position_size(self, signal):
        """Use professional position sizing"""
        method = PositionSizingMethod[
            self.config['risk_management'].get('position_sizing_method', 'FIXED_PERCENTAGE')
        ]
        
        signal_data = {
            'symbol': signal.symbol,
            'price': signal.entry_price,
            'volatility': self._calculate_volatility(signal.symbol),
            'atr': self._calculate_atr(signal.symbol),
            'stop_loss_distance': abs(signal.entry_price - signal.stop_loss) if signal.stop_loss else signal.entry_price * 0.02
        }
        
        return self.risk_manager.calculate_position_size(method, signal_data)

# Run the professional bot
async def main():
    bot = ProfessionalTradingBot()
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 2: Run with Professional Dashboard

```python
# Add to your main trading bot file
from professional_dashboard import PROFESSIONAL_DASHBOARD_HTML

@app.get("/")
async def get_dashboard():
    if Config().get('professional_mode', False):
        return HTMLResponse(content=PROFESSIONAL_DASHBOARD_HTML)
    else:
        return HTMLResponse(content=DASHBOARD_HTML_WITH_COMMENTARY)

# Add professional API endpoints
from professional_trading_engine import create_api_endpoints
create_api_endpoints(app, trading_engine)
```

## Running Different Modes

### 1. Paper Trading Mode

```bash
# Start in paper trading mode
python trading_bot_commentary_updated.py

# Access dashboard at http://localhost:8000
# All trades will be simulated
```

### 2. Backtesting Mode

```python
# Create backtest script
python run_backtest.py --symbols SPY,QQQ --days 30 --strategy ma_cross
```

### 3. Live Trading Mode

```bash
# Update Config().yaml
# Set paper_trading: false
# Ensure Schwab credentials are configured

python trading_bot_commentary_updated.py
```

### 4. Strategy Testing Mode

```python
# Test individual strategies
python test_strategies.py --strategy rsi_momentum --symbol SPY
```

## Testing Strategies

### 1. Test Strategy Combinations

```python
# Enable different strategy combinations
strategies = ['ma_cross', 'rsi_momentum', 'bollinger_bands']

for combo in itertools.combinations(strategies, 2):
    manager.strategy_configs = {s: manager.strategy_configs[s] for s in combo}
    # Run backtest and compare results
```

### 2. Optimize Strategy Parameters

```python
# Grid search for optimal parameters
param_grid = {
    'fast_period': [10, 15, 20],
    'slow_period': [30, 40, 50]
}

best_sharpe = 0
best_params = {}

for fast in param_grid['fast_period']:
    for slow in param_grid['slow_period']:
        # Update strategy parameters
        config.parameters['fast_period'] = fast
        config.parameters['slow_period'] = slow
        
        # Run backtest
        results = engine.run(market_data)
        
        if results.sharpe_ratio > best_sharpe:
            best_sharpe = results.sharpe_ratio
            best_params = {'fast': fast, 'slow': slow}
```

### 3. Test Risk Management

```python
# Test different position sizing methods
methods = [
    PositionSizingMethod.FIXED_PERCENTAGE,
    PositionSizingMethod.KELLY_CRITERION,
    PositionSizingMethod.VOLATILITY_BASED
]

for method in methods:
    risk_manager = RiskManager(initial_capital=100000)
    # Run backtest with each method
    print(f"{method.value}: Sharpe = {results.sharpe_ratio:.2f}")
```

## Performance Analysis

### 1. Generate Performance Reports

```python
# After running live or paper trading
analyzer = trading_engine.performance_analyzer

# Generate HTML report
analyzer.generate_report("performance_report.html")

# Get specific metrics
metrics = analyzer.calculate_metrics()
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Win Rate: {metrics.win_rate:.2%}")

# Analyze by strategy
strategy_performance = analyzer.analyze_by_strategy()
for strategy, perf in strategy_performance.items():
    print(f"{strategy}: {perf.metrics.total_return:.2%}")
```

### 2. Real-time Monitoring

Access the dashboard at `http://localhost:8000` to see:
- Live performance metrics
- Strategy performance breakdown
- Position management
- Risk metrics
- ML model performance

### 3. Export Results

```python
# Export to CSV
trades_df = pd.DataFrame(analyzer.trades)
trades_df.to_csv('trades_history.csv')

# Export to JSON
with open('performance_metrics.json', 'w') as f:
    json.dump(metrics.to_dict(), f, indent=2)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all files are in the same directory
   ls *.py | grep -E "(strategy|ml_model|risk|paper|performance)"
   ```

2. **No Trading Signals**
   - Check strategy parameters
   - Ensure sufficient historical data
   - Verify market data updates

3. **ML Model Errors**
   - Train models before use
   - Check feature engineering
   - Verify data quality

4. **Performance Issues**
   - Reduce number of active strategies
   - Increase data update intervals
   - Use faster ML models (Random Forest vs Neural Network)

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug prints in strategies
def analyze(self, market_data, positions):
    logger.debug(f"Analyzing {len(market_data)} bars")
    # ... strategy logic
```

### Testing Checklist

- [ ] All dependencies installed
- [ ] Configuration files created
- [ ] Paper trading working
- [ ] Strategies generating signals
- [ ] ML models trained
- [ ] Risk limits enforced
- [ ] Performance tracking active
- [ ] Dashboard accessible

## Next Steps

1. **Collect Historical Data**: Get real market data for better testing
2. **Train ML Models**: Use your historical trades to train models
3. **Optimize Strategies**: Run parameter optimization
4. **Set Risk Limits**: Configure appropriate risk parameters
5. **Monitor Performance**: Track results and adjust

Remember to always start with paper trading before going live!