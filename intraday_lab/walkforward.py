"""Walk-Forward Optimization for intraday strategies.

Prevents overfitting by optimizing on in-sample data, then testing on
unseen out-of-sample data, stepping forward through time.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable

from .backtester import IntradayBacktester

logger = logging.getLogger('IntradayLab')


class IntradayWalkForward:
    """Walk-forward optimizer for a single intraday strategy."""

    def __init__(self, db_url: str, capital: float = 100000):
        self.db_url = db_url
        self.capital = capital

    def run(self, strategy_id: str, param_grid: Dict[str, List],
            start_date: str, end_date: str,
            in_sample_days: int = 60, out_sample_days: int = 15,
            step_days: int = 15,
            symbols: Optional[List[str]] = None,
            max_symbols_per_day: int = 20,
            progress_callback: Optional[Callable] = None) -> Dict:
        """Run walk-forward optimization.

        For each window:
        1. Optimize params on in-sample period
        2. Test best params on out-of-sample period
        3. Step forward and repeat
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        windows = []
        cursor = start_dt
        window_num = 0

        while cursor + timedelta(days=in_sample_days + out_sample_days) <= end_dt:
            window_num += 1
            is_start = cursor
            is_end = cursor + timedelta(days=in_sample_days)
            oos_start = is_end + timedelta(days=1)
            oos_end = oos_start + timedelta(days=out_sample_days)

            if progress_callback:
                progress_callback(
                    window_num * 10,  # rough progress
                    f'Window {window_num}: IS {is_start.date()}→{is_end.date()}, '
                    f'OOS {oos_start.date()}→{oos_end.date()}')

            # Step 1: Optimize on in-sample
            bt = IntradayBacktester(self.db_url, self.capital)
            is_results = bt.sweep(
                strategy_id, str(is_start.date()), str(is_end.date()),
                param_grid, symbols=symbols,
                max_symbols_per_day=max_symbols_per_day,
            )

            if not is_results:
                cursor += timedelta(days=step_days)
                continue

            best_is = is_results[0]
            best_params = best_is.get('params', {})
            is_pnl = best_is.get('net_pnl', 0)
            is_sharpe = best_is.get('sharpe', 0)
            is_trades = best_is.get('total_trades', 0)

            # Step 2: Test best params on out-of-sample
            bt_oos = IntradayBacktester(self.db_url, self.capital)
            oos_result = bt_oos.run(
                strategy_id, str(oos_start.date()), str(oos_end.date()),
                symbols=symbols, config=best_params,
                max_symbols_per_day=max_symbols_per_day,
            )

            oos_pnl = oos_result.get('net_pnl', 0)
            oos_sharpe = oos_result.get('sharpe', 0)
            oos_trades = oos_result.get('total_trades', 0)

            # Efficiency: OOS performance / IS performance
            efficiency = 0
            if is_pnl > 0:
                efficiency = oos_pnl / is_pnl * 100

            windows.append({
                'window': window_num,
                'is_start': str(is_start.date()),
                'is_end': str(is_end.date()),
                'oos_start': str(oos_start.date()),
                'oos_end': str(oos_end.date()),
                'best_params': best_params,
                'is_pnl': round(is_pnl, 2),
                'is_sharpe': round(is_sharpe, 2),
                'is_trades': is_trades,
                'oos_pnl': round(oos_pnl, 2),
                'oos_sharpe': round(oos_sharpe, 2),
                'oos_trades': oos_trades,
                'efficiency': round(efficiency, 1),
            })

            cursor += timedelta(days=step_days)

        # Summary
        total_oos_pnl = sum(w['oos_pnl'] for w in windows)
        total_oos_trades = sum(w['oos_trades'] for w in windows)
        avg_efficiency = (sum(w['efficiency'] for w in windows) / len(windows)
                          if windows else 0)
        oos_sharpes = [w['oos_sharpe'] for w in windows if w['oos_trades'] > 0]
        avg_oos_sharpe = sum(oos_sharpes) / len(oos_sharpes) if oos_sharpes else 0

        # Stability: % of OOS windows that are profitable
        profitable_windows = sum(1 for w in windows if w['oos_pnl'] > 0)
        stability = profitable_windows / len(windows) * 100 if windows else 0

        # Status determination
        if avg_oos_sharpe >= 2.0 and stability >= 60:
            status = 'ready'
        elif avg_oos_sharpe >= 1.0 and stability >= 50:
            status = 'marginal'
        else:
            status = 'reject'

        return {
            'strategy_id': strategy_id,
            'windows': windows,
            'total_oos_pnl': round(total_oos_pnl, 2),
            'total_oos_trades': total_oos_trades,
            'avg_efficiency': round(avg_efficiency, 1),
            'avg_oos_sharpe': round(avg_oos_sharpe, 2),
            'stability': round(stability, 1),
            'status': status,
            'window_count': len(windows),
        }
