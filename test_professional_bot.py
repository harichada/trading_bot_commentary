#!/usr/bin/env python3
"""
Quick test script for the professional trading bot
Run this to see all components working together
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

# Import all professional components
from strategy_system import StrategyManager, StrategyConfig
from ml_model_manager_safe import ModelManager
from risk_management import RiskManager, RiskLimits, PositionSizingMethod
from paper_trading import PaperTradingEngine, ExecutionModel
from advanced_orders import Order, OrderType, OrderSide, create_bracket_order, create_trailing_stop
from backtesting_engine import BacktestingEngine, BacktestConfig, BacktestReport
from multi_timeframe_analysis import MultiTimeframeAnalyzer, Timeframe
from performance_analytics import PerformanceAnalyzer

def generate_sample_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Generate realistic sample market data"""
    # Create datetime index
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # Generate realistic price movement
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.01, len(dates))
    price = 100
    prices = []
    
    for ret in returns:
        price *= (1 + ret)
        prices.append(price)
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    data['open'] = data['close'].shift(1).fillna(prices[0])
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.002, len(dates)))
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.002, len(dates)))
    data['volume'] = np.random.randint(1000000, 5000000, len(dates))
    
    return data

def test_strategies():
    """Test the strategy system"""
    print("\n" + "="*50)
    print("TESTING STRATEGY SYSTEM")
    print("="*50)
    
    # Generate sample data
    data = generate_sample_data('SPY', days=5)
    print(f"Generated {len(data)} bars of sample data")
    
    # Initialize strategy manager
    manager = StrategyManager()
    
    # List available strategies
    print("\nAvailable strategies:")
    for key, config in manager.strategy_configs.items():
        print(f"  - {key}: {config.name} (enabled: {config.enabled})")
    
    # Test signal generation
    print("\nGenerating signals...")
    signals = manager.analyze_all(data, {})
    print(f"Generated {len(signals)} signals")
    
    for signal in signals[:3]:  # Show first 3 signals
        print(f"  {signal.strategy_name}: {signal.signal_type.value} at ${signal.entry_price:.2f} "
              f"(strength: {signal.strength:.2f})")
    
    # Test consensus
    consensus = manager.get_consensus_signal(signals)
    if consensus:
        print(f"\nConsensus signal: {consensus.signal_type.value} with strength {consensus.strength:.2f}")
    
    return manager

def test_ml_models():
    """Test ML model manager"""
    print("\n" + "="*50)
    print("TESTING ML MODELS")
    print("="*50)
    
    # Generate training data
    n_samples = 500
    X = pd.DataFrame({
        'returns_1': np.random.randn(n_samples),
        'returns_5': np.random.randn(n_samples),
        'returns_20': np.random.randn(n_samples),
        'sma_ratio': np.random.uniform(0.95, 1.05, n_samples),
        'rsi': np.random.uniform(20, 80, n_samples),
        'volume_ratio': np.random.uniform(0.5, 2, n_samples),
        'volatility': np.random.uniform(0.01, 0.03, n_samples),
        'bb_position': np.random.uniform(0, 1, n_samples),
        'macd_signal': np.random.randn(n_samples) * 0.01
    })
    
    # Create realistic target
    y = pd.Series((X['returns_5'] > 0.002).astype(int))
    
    # Initialize model manager
    manager = ModelManager()
    
    print("Training Random Forest model...")
    manager.train_model('random_forest', X, y)
    
    print("Training Gradient Boosting model...")
    manager.train_model('gradient_boosting', X, y)
    
    # Compare models
    print("\nModel comparison:")
    comparison = manager.compare_models(X, y)
    print(comparison[['model', 'accuracy', 'f1_score']])
    
    # Create ensemble
    print("\nCreating ensemble model...")
    ensemble = manager.create_ensemble(['random_forest', 'gradient_boosting'])
    manager.set_active_model('ensemble')
    
    # Make predictions
    test_data = X.head(5)
    predictions = manager.predict(test_data)
    print(f"Sample predictions: {predictions}")
    
    return manager

def test_risk_management():
    """Test risk management system"""
    print("\n" + "="*50)
    print("TESTING RISK MANAGEMENT")
    print("="*50)
    
    # Initialize risk manager
    risk_limits = RiskLimits(
        max_portfolio_risk=0.02,
        max_position_risk=0.01,
        max_positions=5
    )
    
    risk_manager = RiskManager(
        initial_capital=100000,
        risk_limits=risk_limits
    )
    
    # Test different position sizing methods
    signal_data = {
        'symbol': 'SPY',
        'price': 450.0,
        'volatility': 0.015,
        'atr': 5.0,
        'stop_loss_distance': 4.5,
        'win_rate': 0.55,
        'avg_win': 0.03,
        'avg_loss': 0.02
    }
    
    print("Position sizing recommendations:")
    recommendations = risk_manager.get_position_sizing_recommendation(signal_data)
    
    for method, data in recommendations['recommendations'].items():
        print(f"  {method}: {data['size']:.0f} shares (${data['value']:,.2f})")
    
    # Test risk metrics
    print("\nCurrent risk metrics:")
    metrics = risk_manager.calculate_risk_metrics()
    print(f"  VaR (95%): {metrics.var_95:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Current Drawdown: {metrics.current_drawdown:.2%}")
    
    return risk_manager

def test_paper_trading():
    """Test paper trading system"""
    print("\n" + "="*50)
    print("TESTING PAPER TRADING")
    print("="*50)
    
    # Initialize paper trading engine
    engine = PaperTradingEngine(
        initial_balance=100000,
        execution_model=ExecutionModel.REALISTIC
    )
    
    # Generate and update market data
    data = generate_sample_data('SPY', days=1)
    engine.market_simulator.update_market_data('SPY', data)
    
    print(f"Initial balance: ${engine.account.current_balance:,.2f}")
    
    # Place a market order
    print("\nPlacing market buy order...")
    order = Order(
        symbol='SPY',
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET
    )
    order_id = engine.place_order(order)
    print(f"Order placed: {order_id}")
    
    # Check positions
    positions = engine.get_positions()
    if positions:
        print("\nCurrent positions:")
        for symbol, pos in positions.items():
            print(f"  {symbol}: {pos['quantity']} shares @ ${pos['entry_price']:.2f}")
    
    # Place a bracket order
    print("\nPlacing bracket order...")
    bracket = create_bracket_order(
        symbol='SPY',
        side=OrderSide.BUY,
        quantity=50,
        entry_price=data.iloc[-1]['close'],
        take_profit=data.iloc[-1]['close'] * 1.02,
        stop_loss=data.iloc[-1]['close'] * 0.98
    )
    bracket_id = engine.place_order(bracket)
    print(f"Bracket order placed: {bracket_id}")
    
    # Get account summary
    summary = engine.get_account_summary()
    print(f"\nAccount summary:")
    print(f"  Equity: ${summary['equity']:,.2f}")
    print(f"  Positions: {summary['positions']}")
    print(f"  Unrealized P&L: ${summary['unrealized_pnl']:,.2f}")
    
    return engine

def test_backtesting():
    """Test backtesting engine"""
    print("\n" + "="*50)
    print("TESTING BACKTESTING ENGINE")
    print("="*50)
    
    # Configure backtest
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        symbols=['SPY', 'QQQ'],
        max_positions=3
    )
    
    # Initialize backtesting engine
    engine = BacktestingEngine(config)
    
    # Generate sample data for multiple symbols
    market_data = {}
    for symbol in config.symbols:
        market_data[symbol] = generate_sample_data(symbol, days=31)
    
    print(f"Running backtest from {config.start_date.date()} to {config.end_date.date()}")
    print(f"Symbols: {', '.join(config.symbols)}")
    
    # Run backtest
    results = engine.run(market_data)
    
    print(f"\nBacktest Results:")
    print(f"  Total Return: {results.total_return:.2%}")
    print(f"  Annual Return: {results.annual_return:.2%}")
    print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {results.max_drawdown:.2%}")
    print(f"  Win Rate: {results.win_rate:.2%}")
    print(f"  Total Trades: {results.total_trades}")
    
    # Generate report
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "backtest_report.html"
    BacktestReport.generate_html_report(results, str(report_path))
    print(f"\nDetailed report saved to: {report_path}")
    
    return results

def test_multi_timeframe():
    """Test multi-timeframe analysis"""
    print("\n" + "="*50)
    print("TESTING MULTI-TIMEFRAME ANALYSIS")
    print("="*50)
    
    # Generate data for multiple timeframes
    base_data = generate_sample_data('SPY', days=5)
    
    # Create analyzer
    analyzer = MultiTimeframeAnalyzer()
    
    # Prepare data for different timeframes
    timeframe_data = {
        Timeframe.M5: base_data,
        Timeframe.M15: base_data.resample('15T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }),
        Timeframe.H1: base_data.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
    }
    
    # Analyze
    print("Analyzing multiple timeframes...")
    analyses = analyzer.analyze('SPY', timeframe_data)
    
    for timeframe, analysis in analyses.items():
        print(f"\n{timeframe.value}:")
        print(f"  Trend: {analysis.trend} (strength: {analysis.trend_strength:.2f})")
        print(f"  Momentum: {analysis.momentum:.2f}")
        print(f"  Key levels: {[f'${l:.2f}' for l in analysis.key_levels[:3]]}")
    
    # Generate multi-timeframe signal
    signal = analyzer.generate_multi_timeframe_signal('SPY', analyses, Timeframe.M15)
    if signal:
        print(f"\nMulti-timeframe signal:")
        print(f"  Type: {signal.signal_type}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Entry: ${signal.entry_price:.2f}")
        print(f"  Stop Loss: ${signal.stop_loss:.2f}")
        print(f"  Targets: {[f'${t:.2f}' for t in signal.take_profit]}")
    
    return analyzer

def test_performance_analytics():
    """Test performance analytics"""
    print("\n" + "="*50)
    print("TESTING PERFORMANCE ANALYTICS")
    print("="*50)
    
    # Create analyzer
    analyzer = PerformanceAnalyzer()
    
    # Add sample trades
    print("Adding sample trades...")
    base_time = datetime.now() - timedelta(days=30)
    
    trades = [
        {'symbol': 'SPY', 'pnl': 150, 'entry_time': base_time, 'exit_time': base_time + timedelta(hours=2), 'strategy': 'ma_cross'},
        {'symbol': 'QQQ', 'pnl': -50, 'entry_time': base_time + timedelta(days=1), 'exit_time': base_time + timedelta(days=1, hours=3), 'strategy': 'rsi_momentum'},
        {'symbol': 'SPY', 'pnl': 200, 'entry_time': base_time + timedelta(days=2), 'exit_time': base_time + timedelta(days=2, hours=1), 'strategy': 'ma_cross'},
        {'symbol': 'IWM', 'pnl': -30, 'entry_time': base_time + timedelta(days=3), 'exit_time': base_time + timedelta(days=3, hours=4), 'strategy': 'bollinger_bands'},
        {'symbol': 'SPY', 'pnl': 100, 'entry_time': base_time + timedelta(days=4), 'exit_time': base_time + timedelta(days=4, hours=2), 'strategy': 'ma_cross'},
    ]
    
    for trade in trades:
        analyzer.add_trade(trade)
    
    # Update equity curve
    equity = 100000
    for i, trade in enumerate(trades):
        equity += trade['pnl']
        analyzer.update_equity(trade['exit_time'], equity)
    
    # Calculate metrics
    metrics = analyzer.calculate_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  Win Rate: {metrics.win_rate:.2%}")
    print(f"  Avg Win: ${metrics.avg_win:.2f}")
    print(f"  Avg Loss: ${abs(metrics.avg_loss):.2f}")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    
    # Analyze by strategy
    print("\nPerformance by Strategy:")
    strategy_perf = analyzer.analyze_by_strategy()
    for strategy, perf in strategy_perf.items():
        print(f"  {strategy}: {perf.metrics.win_rate:.0%} win rate, "
              f"${sum(t['pnl'] for t in trades if t.get('strategy') == strategy):.2f} P&L")
    
    # Generate report
    report_path = Path("test_results") / "performance_report.html"
    analyzer.generate_report(str(report_path))
    print(f"\nPerformance report saved to: {report_path}")
    
    return analyzer

def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("PROFESSIONAL TRADING BOT TEST SUITE")
    print("="*50)
    print("This will test all major components of the system")
    
    try:
        # Test each component
        strategy_manager = test_strategies()
        ml_manager = test_ml_models()
        risk_manager = test_risk_management()
        paper_engine = test_paper_trading()
        backtest_results = test_backtesting()
        mtf_analyzer = test_multi_timeframe()
        perf_analyzer = test_performance_analytics()
        
        print("\n" + "="*50)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        print("\nNext steps:")
        print("1. Review the generated reports in 'test_results' directory")
        print("2. Modify strategy parameters in the code")
        print("3. Connect real market data for live testing")
        print("4. Run paper trading before going live")
        
        # Save test configuration
        test_config = {
            'test_run': datetime.now().isoformat(),
            'components_tested': [
                'strategy_system',
                'ml_models',
                'risk_management',
                'paper_trading',
                'backtesting',
                'multi_timeframe',
                'performance_analytics'
            ],
            'status': 'success'
        }
        
        with open('test_results/test_summary.json', 'w') as f:
            json.dump(test_config, f, indent=2)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()