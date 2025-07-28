"""
Stub implementation for backtesting to prevent crashes
Returns mock results instead of running actual backtest
"""

from datetime import datetime
import logging

logger = logging.getLogger('BacktestStub')

async def run_backtest_stub(config):
    """
    Return mock backtest results without actually running the backtest
    This prevents any crashes from the backtesting engine
    """
    logger.info(f"Running stub backtest for {config.get('symbols', [])}")
    
    # Calculate some basic mock values based on config
    days = (datetime.fromisoformat(config['end_date']) - datetime.fromisoformat(config['start_date'])).days
    
    # Return realistic-looking mock results
    return {
        'total_return': 0.0823,  # 8.23%
        'annual_return': 0.0823 * (365 / max(days, 1)),
        'sharpe_ratio': 1.35,
        'max_drawdown': -0.0456,  # -4.56%
        'win_rate': 0.5421,  # 54.21%
        'total_trades': int(days * 2.5),  # Approximate number of trades
        'profit_factor': 1.42,
        'best_trade': {'symbol': config['symbols'][0] if config['symbols'] else 'SPY', 'pnl': 523.45},
        'worst_trade': {'symbol': config['symbols'][0] if config['symbols'] else 'SPY', 'pnl': -234.12},
        'avg_trade_duration': 3.7,  # hours
        'report_url': '/api/professional/backtest/report/latest',
        'message': 'Note: This is a mock backtest result. Full backtesting temporarily disabled to prevent crashes.'
    }