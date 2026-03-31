"""Individual strategy backtester with parameter grid optimization.

Tests a single intraday strategy across a date range with configurable
parameters. Supports parameter sweep for optimization.
"""

import itertools
import logging
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Callable

from .engine import ReplayEngine, LabTrade
from .adapter import create_strategy, get_strategy_params

logger = logging.getLogger('IntradayLab')


class IntradayBacktester:
    """Backtest a single intraday strategy with optional parameter grid."""

    def __init__(self, db_url: str, capital: float = 100000, risk_pct: float = 0.01,
                 max_positions: int = 3):
        self.db_url = db_url
        self.capital = capital
        self.risk_pct = risk_pct
        self.max_positions = max_positions

    def run(self, strategy_id: str, start_date: str, end_date: str,
            symbols: Optional[List[str]] = None,
            config: Optional[Dict] = None,
            start_time: str = '10:00', end_time: str = '15:50',
            max_symbols_per_day: int = 20,
            progress_callback: Optional[Callable] = None) -> Dict:
        """Run backtest for a single strategy with given config."""

        engine = ReplayEngine(self.db_url, self.capital, self.risk_pct, self.max_positions)
        strategy = create_strategy(strategy_id, config)
        if not strategy:
            return {'error': f'Strategy {strategy_id} not found'}

        dates = engine.loader.get_trading_dates(start_date, end_date)
        if not dates:
            return {'error': 'No trading dates found in range'}

        t0 = time.time()
        for i, date in enumerate(dates):
            # Get symbols for this day
            if symbols:
                day_symbols = symbols
            else:
                day_symbols = engine.loader.get_symbols_for_date(
                    date, min_bars=60, min_volume=100000)
                day_symbols = day_symbols[:max_symbols_per_day]

            if day_symbols:
                engine.run_day(strategy, day_symbols, date, start_time, end_time)

            if progress_callback and (i + 1) % 10 == 0:
                pct = (i + 1) / len(dates) * 100
                progress_callback(pct, f'Day {i+1}/{len(dates)} ({date})')

        elapsed = time.time() - t0
        metrics = engine.get_metrics()
        metrics['strategy_id'] = strategy_id
        metrics['strategy_name'] = getattr(strategy, 'name', strategy_id)
        metrics['start_date'] = start_date
        metrics['end_date'] = end_date
        metrics['elapsed_seconds'] = round(elapsed, 1)
        metrics['trading_days'] = len(dates)
        metrics['config'] = config or get_strategy_params(strategy_id)
        metrics['trades'] = [asdict(t) for t in engine.trades]

        engine.loader.close()
        return metrics

    def sweep(self, strategy_id: str, start_date: str, end_date: str,
              param_grid: Dict[str, List],
              symbols: Optional[List[str]] = None,
              start_time: str = '10:00', end_time: str = '15:50',
              max_symbols_per_day: int = 20,
              progress_callback: Optional[Callable] = None) -> List[Dict]:
        """Run parameter sweep — test all combinations in the grid.

        Args:
            param_grid: e.g. {'rsi_period': [7, 14], 'rsi_oversold': [20, 25, 30]}

        Returns:
            List of results sorted by net_pnl descending.
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))

        results = []
        total = len(combos)
        t0 = time.time()

        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))

            result = self.run(
                strategy_id, start_date, end_date,
                symbols=symbols, config=params,
                start_time=start_time, end_time=end_time,
                max_symbols_per_day=max_symbols_per_day,
            )

            if result and 'error' not in result:
                # Strip trades to save memory (keep only metrics)
                summary = {k: v for k, v in result.items() if k != 'trades'}
                summary['params'] = params
                results.append(summary)

            if progress_callback:
                pct = (i + 1) / total * 100
                elapsed = time.time() - t0
                eta = (total - i - 1) * (elapsed / (i + 1)) if i > 0 else 0
                best = max((r['net_pnl'] for r in results), default=0)
                progress_callback(pct,
                    f'Combo {i+1}/{total} | best=${best:,.0f} | ETA {eta:.0f}s')

        results.sort(key=lambda r: r.get('net_pnl', 0), reverse=True)
        return results
