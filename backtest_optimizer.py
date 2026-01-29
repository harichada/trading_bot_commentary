#!/usr/bin/env python3
"""
Backtest Parameter Optimizer
Runs grid search over multiple parameter combinations to find optimal settings.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from itertools import combinations, product
import json
import logging
from pathlib import Path

from backtesting_engine import BacktestingEngine, BacktestConfig, BacktestMode

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Result from a single backtest run"""
    params: Dict[str, Any]
    total_return: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    max_drawdown: float
    final_capital: float

def run_optimization(
    market_data: Dict[str, pd.DataFrame],
    param_grid: Dict[str, List[Any]],
    initial_capital: float = 100000,
    start_date: datetime = None,
    end_date: datetime = None
) -> List[OptimizationResult]:
    """
    Run grid search optimization over parameter combinations.

    Args:
        market_data: Dict of symbol -> DataFrame with OHLCV data
        param_grid: Dict of parameter name -> list of values to test
        initial_capital: Starting capital
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        List of OptimizationResult sorted by total_return descending
    """
    results = []

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    total_combinations = 1
    for v in param_values:
        total_combinations *= len(v)

    print(f"Running {total_combinations} parameter combinations...")
    print(f"Parameters: {param_names}")
    print("-" * 60)

    for i, combo in enumerate(product(*param_values)):
        params = dict(zip(param_names, combo))

        # Extract strategy list if present
        strategies = params.pop('strategies', None)

        # Build config
        config = BacktestConfig(
            start_date=start_date or datetime(2026, 1, 1),
            end_date=end_date or datetime(2026, 1, 29),
            initial_capital=initial_capital,
            symbols=list(market_data.keys()),
            timeframe='5min',
            mode=BacktestMode.REALISTIC,
            max_positions=params.get('max_positions', 5),
            max_position_size=params.get('max_position_size', 0.1),
            stop_loss_default=params.get('stop_loss', 0.02),
            take_profit_default=params.get('take_profit', 0.05),
            use_trailing_stops=params.get('use_trailing_stops', True),
            use_consensus=params.get('use_consensus', True),
            commission=0.001
        )

        try:
            # Create engine with strategy filter
            engine = BacktestingEngine(config, enabled_strategies=strategies)

            # Run backtest
            bt_results = engine.run(market_data)

            # Store result
            result = OptimizationResult(
                params={**params, 'strategies': strategies},
                total_return=bt_results.total_return,
                sharpe_ratio=bt_results.sharpe_ratio,
                win_rate=bt_results.win_rate,
                profit_factor=bt_results.profit_factor,
                total_trades=bt_results.total_trades,
                max_drawdown=bt_results.max_drawdown,
                final_capital=bt_results.final_capital
            )
            results.append(result)

            # Progress update
            if (i + 1) % 10 == 0 or i == total_combinations - 1:
                print(f"[{i+1}/{total_combinations}] Best so far: {max(r.total_return for r in results)*100:.2f}%")

        except Exception as e:
            logger.warning(f"Failed with params {params}: {e}")
            continue

    # Sort by total return descending
    results.sort(key=lambda x: x.total_return, reverse=True)

    return results


def print_results(results: List[OptimizationResult], top_n: int = 20):
    """Print top N results"""
    print("\n" + "=" * 80)
    print(f"TOP {top_n} PARAMETER COMBINATIONS")
    print("=" * 80)

    for i, r in enumerate(results[:top_n]):
        print(f"\n#{i+1} | Return: {r.total_return*100:+.2f}% | Sharpe: {r.sharpe_ratio:.2f} | "
              f"Win: {r.win_rate*100:.1f}% | PF: {r.profit_factor:.2f} | Trades: {r.total_trades}")
        print(f"    Params: {r.params}")

    # Summary stats
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    profitable = [r for r in results if r.total_return > 0]
    print(f"Profitable combinations: {len(profitable)}/{len(results)} ({100*len(profitable)/len(results):.1f}%)")
    if profitable:
        print(f"Best return: {max(r.total_return for r in profitable)*100:.2f}%")
        print(f"Best Sharpe: {max(r.sharpe_ratio for r in profitable):.2f}")
        print(f"Best win rate: {max(r.win_rate for r in profitable)*100:.1f}%")


def save_results(results: List[OptimizationResult], filename: str = "optimization_results.json"):
    """Save results to JSON file"""
    data = [{
        'params': r.params,
        'total_return': r.total_return,
        'sharpe_ratio': r.sharpe_ratio,
        'win_rate': r.win_rate,
        'profit_factor': r.profit_factor,
        'total_trades': r.total_trades,
        'max_drawdown': r.max_drawdown,
        'final_capital': r.final_capital
    } for r in results]

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    # This script needs market data to be passed in
    # It will be called from the main trading bot or with pre-fetched data

    print("Backtest Optimizer")
    print("=" * 60)
    print("This script optimizes backtest parameters.")
    print("Run it with market data from the trading bot.")
    print()
    print("Example usage:")
    print("  from backtest_optimizer import run_optimization, print_results")
    print("  results = run_optimization(market_data, param_grid)")
    print("  print_results(results)")
