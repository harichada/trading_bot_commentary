"""
Performance Tracker - R-Multiple Based Performance Analysis
Tracks expectancy, win rate by setup, and generates weekly reports

"You can't improve what you don't measure. Measure in R."
"""

import numpy as np
import json
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from collections import defaultdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a completed trade"""
    trade_id: str
    symbol: str
    direction: str  # 'long' or 'short'
    strategy: str
    setup_type: str  # 'breakout', 'pullback', 'reversal', etc.

    # Prices
    entry_price: float
    exit_price: float
    stop_loss: float

    # R-multiple tracking
    risk_per_share: float  # 1R
    r_result: float  # Final R-multiple result
    max_favorable_r: float  # Best R reached during trade (MFE)
    max_adverse_r: float  # Worst R during trade (MAE)

    # Context
    regime: str  # Market regime at entry
    session: str  # Trading session at entry
    quality_score: float  # Setup quality at entry
    confluence_score: int

    # Timing
    entry_time: datetime
    exit_time: datetime
    bars_held: int
    exit_reason: str

    # Size
    position_size: float
    dollar_risk: float
    dollar_pnl: float

    # Metadata
    notes: str = ""


@dataclass
class DailyPerformance:
    """Daily performance summary"""
    date: date
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_r: float = 0.0
    best_trade_r: float = 0.0
    worst_trade_r: float = 0.0
    average_winner_r: float = 0.0
    average_loser_r: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0


@dataclass
class WeeklyReport:
    """Weekly performance report"""
    week_start: date
    week_end: date
    trading_days: int
    total_trades: int
    wins: int
    losses: int
    total_r: float
    win_rate: float
    expectancy: float
    profit_factor: float
    avg_r_per_trade: float
    avg_r_per_day: float
    best_day_r: float
    worst_day_r: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    best_strategy: str
    worst_strategy: str
    best_symbol: str
    worst_symbol: str
    notes: str = ""


class PerformanceTracker:
    """
    Professional performance tracking system.

    Philosophy:
    - Track everything in R-multiples (risk-adjusted returns)
    - Expectancy is king: (Win% × Avg Win) - (Loss% × Avg Loss)
    - Know which setups work and which don't
    - Weekly reviews are essential for improvement

    "Pros know their numbers. Amateurs guess."
    """

    def __init__(self, config: Optional[Dict] = None,
                 data_file: str = "performance_data.json"):
        self.config = config or {}
        self.data_file = Path(data_file)

        # Performance thresholds
        self.min_expectancy_threshold = self.config.get('min_expectancy', 0.20)  # 0.2R per trade
        self.min_trades_for_stats = self.config.get('min_trades_for_stats', 10)

        # Trade records
        self.trades: List[TradeRecord] = []
        self.daily_performance: Dict[date, DailyPerformance] = {}

        # Strategy tracking
        self.strategy_stats: Dict[str, Dict] = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total_r': 0.0, 'trades': []
        })

        # Symbol tracking
        self.symbol_stats: Dict[str, Dict] = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total_r': 0.0, 'trades': []
        })

        # Setup type tracking
        self.setup_stats: Dict[str, Dict] = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total_r': 0.0, 'trades': []
        })

        # Regime tracking
        self.regime_stats: Dict[str, Dict] = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total_r': 0.0, 'trades': []
        })

        # Session tracking
        self.session_stats: Dict[str, Dict] = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total_r': 0.0, 'trades': []
        })

        # Streak tracking
        self.current_streak: int = 0  # Positive = wins, negative = losses
        self.max_win_streak: int = 0
        self.max_loss_streak: int = 0

        # Load existing data
        self._load_data()

    def record_trade(self, trade: TradeRecord):
        """Record a completed trade"""
        self.trades.append(trade)

        won = trade.r_result > 0

        # Update strategy stats
        strategy_stat = self.strategy_stats[trade.strategy]
        strategy_stat['total_r'] += trade.r_result
        strategy_stat['trades'].append(trade.trade_id)
        if won:
            strategy_stat['wins'] += 1
        else:
            strategy_stat['losses'] += 1

        # Update symbol stats
        symbol_stat = self.symbol_stats[trade.symbol]
        symbol_stat['total_r'] += trade.r_result
        symbol_stat['trades'].append(trade.trade_id)
        if won:
            symbol_stat['wins'] += 1
        else:
            symbol_stat['losses'] += 1

        # Update setup stats
        setup_stat = self.setup_stats[trade.setup_type]
        setup_stat['total_r'] += trade.r_result
        setup_stat['trades'].append(trade.trade_id)
        if won:
            setup_stat['wins'] += 1
        else:
            setup_stat['losses'] += 1

        # Update regime stats
        regime_stat = self.regime_stats[trade.regime]
        regime_stat['total_r'] += trade.r_result
        regime_stat['trades'].append(trade.trade_id)
        if won:
            regime_stat['wins'] += 1
        else:
            regime_stat['losses'] += 1

        # Update session stats
        session_stat = self.session_stats[trade.session]
        session_stat['total_r'] += trade.r_result
        session_stat['trades'].append(trade.trade_id)
        if won:
            session_stat['wins'] += 1
        else:
            session_stat['losses'] += 1

        # Update daily performance
        trade_date = trade.exit_time.date()
        if trade_date not in self.daily_performance:
            self.daily_performance[trade_date] = DailyPerformance(date=trade_date)

        daily = self.daily_performance[trade_date]
        daily.trades += 1
        daily.total_r += trade.r_result
        if won:
            daily.wins += 1
            daily.best_trade_r = max(daily.best_trade_r, trade.r_result)
        else:
            daily.losses += 1
            daily.worst_trade_r = min(daily.worst_trade_r, trade.r_result)

        # Update streaks
        if won:
            if self.current_streak >= 0:
                self.current_streak += 1
            else:
                self.current_streak = 1
            self.max_win_streak = max(self.max_win_streak, self.current_streak)
        else:
            if self.current_streak <= 0:
                self.current_streak -= 1
            else:
                self.current_streak = -1
            self.max_loss_streak = max(self.max_loss_streak, abs(self.current_streak))

        # Save data
        self._save_data()

        logger.info(f"Trade recorded: {trade.symbol} {trade.strategy} = {trade.r_result:+.2f}R "
                   f"(Streak: {self.current_streak:+d})")

    def get_expectancy(self, trades: Optional[List[TradeRecord]] = None) -> float:
        """
        Calculate expectancy (expected R per trade).

        Expectancy = (Win% × Avg Win R) - (Loss% × Avg Loss R)

        A positive expectancy means you have an edge.
        """
        if trades is None:
            trades = self.trades

        if len(trades) < self.min_trades_for_stats:
            return 0.0

        wins = [t.r_result for t in trades if t.r_result > 0]
        losses = [t.r_result for t in trades if t.r_result <= 0]

        if not wins and not losses:
            return 0.0

        win_rate = len(wins) / len(trades)
        loss_rate = len(losses) / len(trades)

        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0

        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        return expectancy

    def get_profit_factor(self, trades: Optional[List[TradeRecord]] = None) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        PF > 1.5 is good, PF > 2.0 is excellent.
        """
        if trades is None:
            trades = self.trades

        gross_profit = sum(t.r_result for t in trades if t.r_result > 0)
        gross_loss = abs(sum(t.r_result for t in trades if t.r_result < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def get_win_rate(self, trades: Optional[List[TradeRecord]] = None) -> float:
        """Calculate win rate"""
        if trades is None:
            trades = self.trades

        if not trades:
            return 0.0

        wins = sum(1 for t in trades if t.r_result > 0)
        return wins / len(trades)

    def get_strategy_performance(self, strategy: str) -> Dict:
        """Get performance stats for a specific strategy"""
        stats = self.strategy_stats.get(strategy, {})
        trades = [t for t in self.trades if t.strategy == strategy]

        if len(trades) < self.min_trades_for_stats:
            return {
                'strategy': strategy,
                'trades': len(trades),
                'message': f'Insufficient trades ({len(trades)}/{self.min_trades_for_stats})'
            }

        return {
            'strategy': strategy,
            'trades': len(trades),
            'wins': stats.get('wins', 0),
            'losses': stats.get('losses', 0),
            'total_r': f"{stats.get('total_r', 0):+.2f}R",
            'win_rate': f"{self.get_win_rate(trades):.1%}",
            'expectancy': f"{self.get_expectancy(trades):+.3f}R",
            'profit_factor': f"{self.get_profit_factor(trades):.2f}",
            'avg_r_per_trade': f"{np.mean([t.r_result for t in trades]):+.2f}R",
            'is_profitable': self.get_expectancy(trades) >= self.min_expectancy_threshold
        }

    def get_symbol_performance(self, symbol: str) -> Dict:
        """Get performance stats for a specific symbol"""
        stats = self.symbol_stats.get(symbol, {})
        trades = [t for t in self.trades if t.symbol == symbol]

        if len(trades) < 3:  # Lower threshold for symbols
            return {
                'symbol': symbol,
                'trades': len(trades),
                'message': 'Insufficient trades'
            }

        return {
            'symbol': symbol,
            'trades': len(trades),
            'wins': stats.get('wins', 0),
            'losses': stats.get('losses', 0),
            'total_r': f"{stats.get('total_r', 0):+.2f}R",
            'win_rate': f"{self.get_win_rate(trades):.1%}",
            'avg_r_per_trade': f"{np.mean([t.r_result for t in trades]):+.2f}R",
        }

    def should_use_strategy(self, strategy: str) -> Tuple[bool, str]:
        """
        Determine if a strategy should be used based on performance.

        Returns:
            (should_use, reason)
        """
        trades = [t for t in self.trades if t.strategy == strategy]

        if len(trades) < self.min_trades_for_stats:
            return True, f"Insufficient data ({len(trades)} trades)"

        expectancy = self.get_expectancy(trades)
        win_rate = self.get_win_rate(trades)

        if expectancy < 0:
            return False, f"Negative expectancy ({expectancy:+.3f}R)"

        if expectancy < self.min_expectancy_threshold:
            return False, f"Below minimum expectancy ({expectancy:+.3f}R < {self.min_expectancy_threshold}R)"

        if win_rate < 0.30:
            return False, f"Win rate too low ({win_rate:.1%})"

        return True, f"Strategy is profitable (Exp: {expectancy:+.3f}R, WR: {win_rate:.1%})"

    def should_trade_symbol(self, symbol: str) -> Tuple[bool, str]:
        """Determine if a symbol should be traded based on history"""
        trades = [t for t in self.trades if t.symbol == symbol]

        if len(trades) < 5:
            return True, "Insufficient data"

        total_r = sum(t.r_result for t in trades)
        win_rate = self.get_win_rate(trades)

        if total_r < -5:  # Lost more than 5R on this symbol
            return False, f"Historical loser ({total_r:+.2f}R)"

        if win_rate < 0.25 and len(trades) >= 10:
            return False, f"Very low win rate ({win_rate:.1%})"

        return True, f"Symbol OK ({total_r:+.2f}R, {win_rate:.1%} WR)"

    def generate_weekly_report(self, week_end: Optional[date] = None) -> WeeklyReport:
        """Generate weekly performance report"""
        if week_end is None:
            week_end = date.today()

        # Get start of week (Monday)
        week_start = week_end - timedelta(days=week_end.weekday())

        # Filter trades for this week
        week_trades = [
            t for t in self.trades
            if week_start <= t.exit_time.date() <= week_end
        ]

        if not week_trades:
            return WeeklyReport(
                week_start=week_start,
                week_end=week_end,
                trading_days=0,
                total_trades=0,
                wins=0,
                losses=0,
                total_r=0.0,
                win_rate=0.0,
                expectancy=0.0,
                profit_factor=0.0,
                avg_r_per_trade=0.0,
                avg_r_per_day=0.0,
                best_day_r=0.0,
                worst_day_r=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                best_strategy="N/A",
                worst_strategy="N/A",
                best_symbol="N/A",
                worst_symbol="N/A",
                notes="No trades this week"
            )

        # Calculate stats
        wins = sum(1 for t in week_trades if t.r_result > 0)
        losses = len(week_trades) - wins
        total_r = sum(t.r_result for t in week_trades)

        # Daily breakdown
        trading_dates = set(t.exit_time.date() for t in week_trades)
        trading_days = len(trading_dates)

        # Best/worst day
        daily_r = defaultdict(float)
        for t in week_trades:
            daily_r[t.exit_time.date()] += t.r_result

        best_day_r = max(daily_r.values())
        worst_day_r = min(daily_r.values())

        # Strategy performance this week
        strategy_r = defaultdict(float)
        for t in week_trades:
            strategy_r[t.strategy] += t.r_result

        best_strategy = max(strategy_r, key=strategy_r.get) if strategy_r else "N/A"
        worst_strategy = min(strategy_r, key=strategy_r.get) if strategy_r else "N/A"

        # Symbol performance this week
        symbol_r = defaultdict(float)
        for t in week_trades:
            symbol_r[t.symbol] += t.r_result

        best_symbol = max(symbol_r, key=symbol_r.get) if symbol_r else "N/A"
        worst_symbol = min(symbol_r, key=symbol_r.get) if symbol_r else "N/A"

        # Consecutive wins/losses this week
        sorted_trades = sorted(week_trades, key=lambda t: t.exit_time)
        max_win = max_loss = current = 0
        for t in sorted_trades:
            if t.r_result > 0:
                current = current + 1 if current > 0 else 1
                max_win = max(max_win, current)
            else:
                current = current - 1 if current < 0 else -1
                max_loss = max(max_loss, abs(current))

        return WeeklyReport(
            week_start=week_start,
            week_end=week_end,
            trading_days=trading_days,
            total_trades=len(week_trades),
            wins=wins,
            losses=losses,
            total_r=total_r,
            win_rate=wins / len(week_trades) if week_trades else 0,
            expectancy=self.get_expectancy(week_trades),
            profit_factor=self.get_profit_factor(week_trades),
            avg_r_per_trade=total_r / len(week_trades) if week_trades else 0,
            avg_r_per_day=total_r / trading_days if trading_days else 0,
            best_day_r=best_day_r,
            worst_day_r=worst_day_r,
            max_consecutive_wins=max_win,
            max_consecutive_losses=max_loss,
            best_strategy=f"{best_strategy} ({strategy_r[best_strategy]:+.2f}R)" if best_strategy != "N/A" else "N/A",
            worst_strategy=f"{worst_strategy} ({strategy_r[worst_strategy]:+.2f}R)" if worst_strategy != "N/A" else "N/A",
            best_symbol=f"{best_symbol} ({symbol_r[best_symbol]:+.2f}R)" if best_symbol != "N/A" else "N/A",
            worst_symbol=f"{worst_symbol} ({symbol_r[worst_symbol]:+.2f}R)" if worst_symbol != "N/A" else "N/A",
        )

    def get_overall_stats(self) -> Dict:
        """Get overall performance statistics"""
        if not self.trades:
            return {'message': 'No trades recorded'}

        wins = sum(1 for t in self.trades if t.r_result > 0)
        losses = len(self.trades) - wins
        total_r = sum(t.r_result for t in self.trades)

        winners = [t.r_result for t in self.trades if t.r_result > 0]
        losers = [t.r_result for t in self.trades if t.r_result <= 0]

        return {
            'total_trades': len(self.trades),
            'wins': wins,
            'losses': losses,
            'win_rate': f"{self.get_win_rate():.1%}",
            'total_r': f"{total_r:+.2f}R",
            'expectancy': f"{self.get_expectancy():+.3f}R",
            'profit_factor': f"{self.get_profit_factor():.2f}",
            'avg_winner': f"{np.mean(winners):+.2f}R" if winners else "N/A",
            'avg_loser': f"{np.mean(losers):+.2f}R" if losers else "N/A",
            'largest_winner': f"{max(winners):+.2f}R" if winners else "N/A",
            'largest_loser': f"{min(losers):+.2f}R" if losers else "N/A",
            'max_win_streak': self.max_win_streak,
            'max_loss_streak': self.max_loss_streak,
            'current_streak': self.current_streak,
            'trading_days': len(self.daily_performance),
        }

    def format_weekly_report(self, report: WeeklyReport) -> str:
        """Format weekly report as readable text"""
        return f"""
================================================================================
                    WEEKLY PERFORMANCE REPORT
                    {report.week_start} to {report.week_end}
================================================================================

SUMMARY
-------
Trading Days:        {report.trading_days}
Total Trades:        {report.total_trades}
Wins/Losses:         {report.wins}/{report.losses}
Win Rate:            {report.win_rate:.1%}

P&L (R-Multiples)
-----------------
Total R:             {report.total_r:+.2f}R
Expectancy:          {report.expectancy:+.3f}R per trade
Profit Factor:       {report.profit_factor:.2f}
Avg R/Trade:         {report.avg_r_per_trade:+.2f}R
Avg R/Day:           {report.avg_r_per_day:+.2f}R

DAILY BREAKDOWN
---------------
Best Day:            {report.best_day_r:+.2f}R
Worst Day:           {report.worst_day_r:+.2f}R

STREAKS
-------
Max Consecutive Wins:    {report.max_consecutive_wins}
Max Consecutive Losses:  {report.max_consecutive_losses}

ANALYSIS
--------
Best Strategy:       {report.best_strategy}
Worst Strategy:      {report.worst_strategy}
Best Symbol:         {report.best_symbol}
Worst Symbol:        {report.worst_symbol}

{report.notes if report.notes else ''}
================================================================================
"""

    def _save_data(self):
        """Save performance data to file"""
        try:
            data = {
                'trades': [asdict(t) for t in self.trades[-500:]],  # Keep last 500
                'max_win_streak': self.max_win_streak,
                'max_loss_streak': self.max_loss_streak,
                'current_streak': self.current_streak,
                'last_updated': datetime.now().isoformat()
            }

            # Convert datetime objects to strings
            for trade in data['trades']:
                trade['entry_time'] = trade['entry_time'].isoformat() if isinstance(trade['entry_time'], datetime) else trade['entry_time']
                trade['exit_time'] = trade['exit_time'].isoformat() if isinstance(trade['exit_time'], datetime) else trade['exit_time']

            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")

    def _load_data(self):
        """Load performance data from file"""
        if not self.data_file.exists():
            return

        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)

            self.max_win_streak = data.get('max_win_streak', 0)
            self.max_loss_streak = data.get('max_loss_streak', 0)
            self.current_streak = data.get('current_streak', 0)

            # Reconstruct trades
            for trade_dict in data.get('trades', []):
                try:
                    trade_dict['entry_time'] = datetime.fromisoformat(trade_dict['entry_time'])
                    trade_dict['exit_time'] = datetime.fromisoformat(trade_dict['exit_time'])
                    trade = TradeRecord(**trade_dict)
                    self.trades.append(trade)

                    # Rebuild stats
                    won = trade.r_result > 0
                    self.strategy_stats[trade.strategy]['total_r'] += trade.r_result
                    self.strategy_stats[trade.strategy]['trades'].append(trade.trade_id)
                    if won:
                        self.strategy_stats[trade.strategy]['wins'] += 1
                    else:
                        self.strategy_stats[trade.strategy]['losses'] += 1

                except Exception as e:
                    logger.warning(f"Failed to load trade: {e}")

            logger.info(f"Loaded {len(self.trades)} trades from {self.data_file}")

        except Exception as e:
            logger.error(f"Failed to load performance data: {e}")


# Singleton instance
_performance_tracker: Optional[PerformanceTracker] = None

def get_performance_tracker(config: Optional[Dict] = None) -> PerformanceTracker:
    """Get or create singleton PerformanceTracker instance"""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker(config)
    return _performance_tracker
