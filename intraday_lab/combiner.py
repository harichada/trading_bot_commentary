"""Multi-strategy combination tester.

Runs multiple intraday strategies simultaneously to test how they interact
when sharing capital and position limits.
"""

import logging
from dataclasses import asdict
from typing import Dict, List, Optional

from .engine import ReplayEngine, LabTrade
from .adapter import create_strategy

logger = logging.getLogger('IntradayLab')


class StrategyCombiner:
    """Test multiple strategies running together."""

    def __init__(self, db_url: str, capital: float = 100000, risk_pct: float = 0.01,
                 max_positions: int = 5):
        self.db_url = db_url
        self.capital = capital
        self.risk_pct = risk_pct
        self.max_positions = max_positions

    def run(self, strategy_configs: List[Dict], start_date: str, end_date: str,
            symbols: Optional[List[str]] = None,
            max_symbols_per_day: int = 20) -> Dict:
        """Run multiple strategies simultaneously.

        Args:
            strategy_configs: list of {'strategy_id': str, 'config': dict, 'weight': float}
                weight determines capital allocation (e.g. 0.33 for equal split of 3)
        """
        engine = ReplayEngine(self.db_url, self.capital, self.risk_pct, self.max_positions)

        # Create all strategies
        strategies = []
        for sc in strategy_configs:
            sid = sc['strategy_id']
            cfg = sc.get('config')
            strat = create_strategy(sid, cfg)
            if strat:
                strategies.append({'strategy': strat, 'id': sid, 'weight': sc.get('weight', 1.0)})

        if not strategies:
            return {'error': 'No valid strategies'}

        dates = engine.loader.get_trading_dates(start_date, end_date)

        # Per-strategy trade tracking
        strategy_trades: Dict[str, List[LabTrade]] = {s['id']: [] for s in strategies}

        for date in dates:
            if symbols:
                day_symbols = symbols
            else:
                day_symbols = engine.loader.get_symbols_for_date(
                    date, min_bars=60, min_volume=100000)
                day_symbols = day_symbols[:max_symbols_per_day]

            if not day_symbols:
                continue

            # Run each strategy, collecting trades
            for s in strategies:
                # Each strategy gets its own engine run but shares position limits
                day_trades = engine.run_day(s['strategy'], day_symbols, date)
                for t in day_trades:
                    strategy_trades[s['id']].append(t)

        # Overall metrics
        overall = engine.get_metrics()
        overall['strategies'] = []

        # Per-strategy breakdown
        for s in strategies:
            strades = strategy_trades[s['id']]
            if strades:
                wins = [t for t in strades if t.pnl > 0]
                losses = [t for t in strades if t.pnl <= 0]
                gp = sum(t.pnl for t in wins)
                gl = abs(sum(t.pnl for t in losses))
                overall['strategies'].append({
                    'strategy_id': s['id'],
                    'trades': len(strades),
                    'wins': len(wins),
                    'win_rate': len(wins) / len(strades) if strades else 0,
                    'net_pnl': round(sum(t.pnl for t in strades), 2),
                    'profit_factor': round(gp / gl, 2) if gl > 0 else 999,
                })

        # Correlation: check overlap in trade symbols/times
        all_entries = {}
        for sid, trades in strategy_trades.items():
            for t in trades:
                key = f"{t.symbol}_{t.entry_time}"
                all_entries.setdefault(key, []).append(sid)

        overlapping = sum(1 for v in all_entries.values() if len(v) > 1)
        overall['trade_overlap'] = overlapping
        overall['trade_overlap_pct'] = round(overlapping / len(all_entries) * 100, 1) if all_entries else 0
        overall['combined_trades'] = [asdict(t) for t in engine.trades]

        engine.loader.close()
        return overall
