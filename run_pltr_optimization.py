#!/usr/bin/env python3
"""
PLTR Parameter Optimization Script
Finds optimal backtest parameters for PLTR Jan 1-29, 2026
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations, product
from dataclasses import dataclass
from typing import List, Dict, Any
import logging

# Suppress verbose logging
logging.basicConfig(level=logging.WARNING)

from backtesting_engine import BacktestingEngine, BacktestConfig, BacktestMode

def fetch_pltr_data():
    """Fetch PLTR data from Schwab API"""
    try:
        # Try to load from existing backtest report first
        report_path = 'backtest_reports/backtest_20260129_082004_PLTR.json'
        if os.path.exists(report_path):
            print("Loading PLTR data from existing report...")
            with open(report_path) as f:
                data = json.load(f)
            # We can't reconstruct full OHLCV from the report
            # Need to fetch fresh data
    except:
        pass

    # Need Schwab client - try to initialize
    try:
        from schwab.auth import easy_client
        from schwab.client import Client

        print("Connecting to Schwab API...")
        client = easy_client(
            token_path='token_1.json',
            api_key=os.environ.get('SCHWAB_API_KEY', ''),
            app_secret=os.environ.get('SCHWAB_APP_SECRET', ''),
            callback_url='https://127.0.0.1:8182'
        )

        print("Fetching PLTR data (Jan 1-29, 2026)...")
        start_date = datetime(2026, 1, 1)
        end_date = datetime(2026, 1, 29)

        response = client.get_price_history(
            'PLTR',
            period_type=Client.PriceHistory.PeriodType.DAY,
            frequency_type=Client.PriceHistory.FrequencyType.MINUTE,
            frequency=Client.PriceHistory.Frequency.EVERY_FIVE_MINUTES,
            start_datetime=start_date,
            end_datetime=end_date
        )

        if response.status_code == 200:
            data = response.json()
            candles = data.get('candles', [])
            if candles:
                df = pd.DataFrame(candles)
                df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                df.set_index('datetime', inplace=True)
                df.columns = df.columns.str.lower()
                print(f"Fetched {len(df)} bars for PLTR")
                return {'PLTR': df}

        print(f"API returned status {response.status_code}")
        return None

    except Exception as e:
        print(f"Failed to fetch data: {e}")
        print("\nTo run optimization, ensure:")
        print("1. The trading bot is running (python trading_bot_commentary_updated.py)")
        print("2. Use the web UI at http://localhost:8000 -> Backtest tab")
        print("3. Or call POST /api/backtest/optimize with curl")
        return None


def run_optimization(market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
    """Run parameter optimization"""

    # All available strategies
    all_strategies = [
        'ma_cross', 'rsi_momentum', 'bollinger_bands',
        'macd', 'momentum_breakout', 'simple_price_action'
    ]

    # Generate strategy combinations (2 to 6 strategies)
    strategy_combos = []
    for r in range(2, len(all_strategies) + 1):
        for combo in combinations(all_strategies, r):
            strategy_combos.append(list(combo))

    print(f"Strategy combinations: {len(strategy_combos)}")

    # Parameter grid
    param_grid = {
        'strategies': strategy_combos,
        'use_consensus': [True, False],
        'stop_loss': [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05],
        'take_profit': [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
        'max_position_size': [0.05, 0.10, 0.15, 0.20, 0.25]
    }

    # Calculate total
    total = len(strategy_combos) * len(param_grid['use_consensus']) * \
            len(param_grid['stop_loss']) * len(param_grid['take_profit']) * \
            len(param_grid['max_position_size'])

    print(f"Total combinations to test: {total}")
    print("-" * 60)

    results = []
    best_return = -999

    # Run grid search
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    for i, combo in enumerate(product(*param_values)):
        params = dict(zip(param_names, combo))
        strategies = params.pop('strategies')

        config = BacktestConfig(
            start_date=datetime(2026, 1, 1),
            end_date=datetime(2026, 1, 29),
            initial_capital=100000,
            symbols=['PLTR'],
            timeframe='5min',
            mode=BacktestMode.REALISTIC,
            stop_loss_default=params['stop_loss'],
            take_profit_default=params['take_profit'],
            max_position_size=params['max_position_size'],
            use_consensus=params['use_consensus'],
            commission=0.001
        )

        try:
            engine = BacktestingEngine(config, enabled_strategies=strategies)
            bt_results = engine.run(market_data)

            result = {
                'params': {**params, 'strategies': strategies},
                'total_return': float(bt_results.total_return),
                'sharpe_ratio': float(bt_results.sharpe_ratio),
                'win_rate': float(bt_results.win_rate),
                'profit_factor': float(bt_results.profit_factor),
                'total_trades': int(bt_results.total_trades),
                'max_drawdown': float(bt_results.max_drawdown),
                'final_capital': float(bt_results.final_capital)
            }
            results.append(result)

            # Track best
            if bt_results.total_return > best_return:
                best_return = bt_results.total_return
                print(f"[{i+1}/{total}] NEW BEST: {best_return*100:+.2f}% | "
                      f"SL={params['stop_loss']*100:.1f}% TP={params['take_profit']*100:.1f}% "
                      f"Consensus={params['use_consensus']} Strats={len(strategies)}")

            # Progress every 500
            if (i + 1) % 500 == 0:
                print(f"[{i+1}/{total}] Progress: {100*(i+1)/total:.1f}% | Best: {best_return*100:+.2f}%")

        except Exception as e:
            pass  # Skip failed runs

    # Sort by total return
    results.sort(key=lambda x: x['total_return'], reverse=True)

    return results


def print_top_results(results: List[Dict], top_n: int = 25):
    """Print top results"""
    print("\n" + "=" * 80)
    print(f"TOP {top_n} PARAMETER COMBINATIONS")
    print("=" * 80)

    for i, r in enumerate(results[:top_n]):
        params = r['params']
        print(f"\n#{i+1} | Return: {r['total_return']*100:+.2f}% | "
              f"Sharpe: {r['sharpe_ratio']:.2f} | Win: {r['win_rate']*100:.1f}% | "
              f"PF: {r['profit_factor']:.2f} | Trades: {r['total_trades']}")
        print(f"    Stop Loss: {params['stop_loss']*100:.1f}% | "
              f"Take Profit: {params['take_profit']*100:.1f}% | "
              f"Position Size: {params['max_position_size']*100:.0f}%")
        print(f"    Consensus: {params['use_consensus']} | "
              f"Strategies: {', '.join(params['strategies'])}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    profitable = [r for r in results if r['total_return'] > 0]
    print(f"Total combinations tested: {len(results)}")
    print(f"Profitable combinations: {len(profitable)} ({100*len(profitable)/len(results):.1f}%)")

    if profitable:
        print(f"\nBest overall return: {results[0]['total_return']*100:+.2f}%")
        best_sharpe = max(results, key=lambda x: x['sharpe_ratio'])
        print(f"Best Sharpe ratio: {best_sharpe['sharpe_ratio']:.2f} "
              f"(return: {best_sharpe['total_return']*100:+.2f}%)")
        best_winrate = max(results, key=lambda x: x['win_rate'])
        print(f"Best win rate: {best_winrate['win_rate']*100:.1f}% "
              f"(return: {best_winrate['total_return']*100:+.2f}%)")

    # Analyze patterns in top results
    print("\n" + "=" * 80)
    print("PATTERN ANALYSIS (Top 25)")
    print("=" * 80)

    top_25 = results[:25]

    # Consensus usage
    consensus_true = len([r for r in top_25 if r['params']['use_consensus']])
    print(f"Consensus=True: {consensus_true}/25 ({100*consensus_true/25:.0f}%)")

    # Average stop loss
    avg_sl = np.mean([r['params']['stop_loss'] for r in top_25])
    print(f"Avg Stop Loss: {avg_sl*100:.2f}%")

    # Average take profit
    avg_tp = np.mean([r['params']['take_profit'] for r in top_25])
    print(f"Avg Take Profit: {avg_tp*100:.2f}%")

    # Strategy frequency
    strat_counts = {}
    for r in top_25:
        for s in r['params']['strategies']:
            strat_counts[s] = strat_counts.get(s, 0) + 1
    print("\nStrategy frequency in top 25:")
    for s, c in sorted(strat_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}/25 ({100*c/25:.0f}%)")


def save_results(results: List[Dict], filename: str = "pltr_optimization_results.json"):
    """Save results to file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    print("=" * 60)
    print("PLTR Parameter Optimization")
    print("Date Range: Jan 1 - Jan 29, 2026")
    print("=" * 60)

    # Try to fetch data
    market_data = fetch_pltr_data()

    if market_data is None:
        print("\nCannot fetch PLTR data directly.")
        print("\nAlternative: Run via the web UI:")
        print("1. Start the bot: python trading_bot_commentary_updated.py")
        print("2. Open http://localhost:8000")
        print("3. Go to Backtest tab")
        print("4. Use the API: curl -X POST http://localhost:8000/api/backtest/optimize \\")
        print('   -H "Content-Type: application/json" \\')
        print('   -d \'{"symbols":["PLTR"],"start_date":"2026-01-01","end_date":"2026-01-29"}\'')
        sys.exit(1)

    # Run optimization
    results = run_optimization(market_data)

    # Print results
    print_top_results(results)

    # Save results
    save_results(results)

    # Print recommended settings
    if results and results[0]['total_return'] > 0:
        best = results[0]
        print("\n" + "=" * 60)
        print("RECOMMENDED SETTINGS")
        print("=" * 60)
        print(f"Stop Loss: {best['params']['stop_loss']*100:.1f}%")
        print(f"Take Profit: {best['params']['take_profit']*100:.1f}%")
        print(f"Position Size: {best['params']['max_position_size']*100:.0f}%")
        print(f"Use Consensus: {best['params']['use_consensus']}")
        print(f"Strategies: {', '.join(best['params']['strategies'])}")
        print(f"\nExpected Return: {best['total_return']*100:+.2f}%")
        print(f"Expected Win Rate: {best['win_rate']*100:.1f}%")
