#!/usr/bin/env python3
"""Intraday Strategy Backtester v2 — multi-strategy, bar-by-bar 1-minute simulation.

Replays 1-min bars from PostgreSQL through TickIndicatorEngine, calls actual
strategy scan_for_setups() methods, manages position lifecycle with stop/target/
trailing stop/strategy exits, tracks per-strategy and per-symbol P&L.

Unlike v1 which runs one strategy at a time, v2 runs all 13 intraday strategies
simultaneously with a shared position pool and combined risk limits.

Usage:
    python backtest_intraday_v2.py                              # All strategies
    python backtest_intraday_v2.py --strategy vwap_mean_reversion
    python backtest_intraday_v2.py --start 2024-01-01 --end 2026-03-21
    python backtest_intraday_v2.py --output results.json
    python backtest_intraday_v2.py --strategy gap_bounce --strategy gap_continuation
"""

import argparse
import json
import logging
import math
import os
import statistics
import sys
import time as time_mod
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras

# ═══════════════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════════════

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gap_fade_strategies.base import ExitSignal
from gap_fade_strategies.indicators import FiveMinBar, TickIndicatorEngine
from gap_fade_strategies.intraday_base import IntradaySetup, IntradayStrategy
from gap_fade_strategies.intraday_registry import IntradayStrategyRegistry

# Force-import all strategy modules so @register decorators fire
import gap_fade_strategies  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('bt_intraday_v2')

DB_URL = os.environ.get(
    'DATABASE_URL',
    'postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev',
)

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

UNIVERSE = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AMD', 'INTC', 'MU',
    'AVGO', 'NFLX', 'CRM', 'ORCL', 'ADBE', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'KLAC',
    'MRVL', 'SNPS', 'CDNS', 'NOW', 'UBER', 'SHOP', 'SQ', 'COIN', 'PLTR', 'PYPL',
    'BAC', 'JPM', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'AXP', 'BRK.B',
    'XOM', 'CVX', 'COP', 'SLB', 'HAL', 'FCX', 'NEM', 'GOLD', 'SPY', 'QQQ',
]

DEFAULT_CAPITAL = 12_500.0
RISK_PCT_PER_TRADE = 0.005       # 0.5% of equity = $62.50
SLIPPAGE_PCT = 0.0005            # 0.05% per side
MAX_CONCURRENT_POSITIONS = 5
MAX_TRADES_PER_DAY = 20
TIME_EXIT_HOUR = 15
TIME_EXIT_MIN = 50               # 3:50 PM forced close
MIN_BARS_PER_DAY = 30            # skip symbol-days with < 30 bars

# Backtest-tuned config overrides (from v1, proven on 5-min bar simulation)
BACKTEST_CONFIG_OVERRIDES: Dict[str, Dict] = {
    'orb_breakout': {
        'volume_confirm_ratio': 0.7,
        'min_or_range_pct': 0.005,
        'max_or_range_pct': 0.02,
        'stop_buffer_pct': 0.004,
        'target_rr': 1.5,
        'breakout_buffer_pct': 0.002,
        'max_entries_per_session': 1,
        'min_risk_reward': 1.2,
        'min_confidence': 0.60,
    },
    'momentum_surge': {
        'volume_surge_ratio': 1.0,
        'near_high_pct': 0.005,
        'near_low_pct': 0.005,
        'atr_stop_mult': 2.0,
        'target_rr': 1.5,
        'rsi_min_long': 55,
        'rsi_max_long': 68,
        'rsi_min_short': 32,
        'rsi_max_short': 45,
        'max_entries_per_session': 1,
        'min_risk_reward': 1.2,
        'min_confidence': 0.55,
    },
    'pullback_entry': {
        'max_volume_surge_pullback': 1.5,
        'pullback_proximity_pct': 0.012,
        'rsi_pullback_min': 38,
        'rsi_pullback_max': 53,
        'target_rr': 1.5,
        'max_entries_per_session': 1,
        'min_risk_reward': 1.2,
        'min_confidence': 0.55,
    },
    'range_trade': {
        'max_volume_surge': 1.5,
        'boundary_proximity_pct': 0.004,
        'stop_buffer_pct': 0.008,
        'min_bars_for_range': 10,
        'min_range_pct': 0.008,
        'min_risk_reward': 1.0,
        'min_confidence': 0.55,
    },
    'vwap_mean_reversion': {
        'vwap_deviation_pct': 0.012,
        'rsi_overbought': 62,
        'rsi_oversold': 38,
        'min_volume_surge': 0.8,
        'stop_atr_mult': 2.0,
        'min_stop_pct': 0.008,
        'target_near_vwap_pct': 0.003,
        'max_hold_minutes': 60,
        'max_entries_per_day': 3,
        'min_risk_reward': 0.8,
        'min_confidence': 0.35,
    },
    'opening_trend': {
        'min_trend_pct': 0.002,
        'pullback_pct': 0.002,
        'stop_atr_mult': 1.5,
        'min_stop_pct': 0.006,
        'reward_ratio': 2.0,
        'max_hold_minutes': 120,
        'max_entries_per_day': 2,
        'min_risk_reward': 1.2,
        'min_confidence': 0.40,
    },
    'connors_rsi2': {
        'rsi_entry_long': 28,
        'rsi_entry_short': 72,
        'rsi_exit_long': 48,
        'rsi_exit_short': 52,
        'trend_filter_period': 50,
        'require_trend_filter': False,
        'stop_atr_mult': 1.5,
        'min_stop_pct': 0.005,
        'max_stop_pct': 0.015,
        'target_atr_mult': 1.3,
        'min_target_pct': 0.005,
        'max_hold_minutes': 45,
        'max_entries_per_day': 3,
        'min_risk_reward': 0.8,
        'min_confidence': 0.35,
        'min_volume_ratio': 0.3,
    },
    'vwap_bounce': {
        'vwap_proximity_pct': 0.003,
        'vwap_bounce_pct': 0.001,
        'require_ema_trend': False,
        'stop_atr_mult': 1.2,
        'min_stop_pct': 0.004,
        'max_stop_pct': 0.012,
        'target_rr': 1.5,
        'min_target_pct': 0.004,
        'enable_reclaim': True,
        'reclaim_volume_surge': 1.0,
        'max_hold_minutes': 60,
        'max_entries_per_day': 3,
        'min_risk_reward': 1.0,
        'min_confidence': 0.35,
        'min_volume_ratio': 0.3,
    },
    'first_hour_breakout': {
        'min_range_pct': 0.004,
        'max_range_pct': 0.035,
        'breakout_buffer_pct': 0.001,
        'min_volume_surge': 0.8,
        'target_range_mult': 1.0,
        'use_midpoint_stop': True,
        'wide_range_threshold': 0.015,
        'min_stop_pct': 0.008,
        'max_hold_minutes': 120,
        'max_entries_per_day': 2,
        'min_risk_reward': 1.0,
        'min_confidence': 0.40,
    },
    'catalyst_momentum': {
        'min_gap_pct': 0.008,
        'min_rvol': 1.5,
        'vwap_zone_pct': 0.005,
        'ema_pullback_zone': 0.004,
        'rsi_long_min': 35,
        'rsi_long_max': 68,
        'rsi_short_min': 32,
        'rsi_short_max': 65,
        'stop_below_pullback_buffer': 0.004,
        'min_stop_pct': 0.005,
        'max_stop_pct': 0.012,
        'initial_target_rr': 0.7,
        'trail_after_r': 0.3,
        'avoid_midday': True,
        'max_entries_per_day': 2,
        'max_hold_minutes': 60,
        'min_risk_reward': 0.5,
        'min_confidence': 0.35,
    },
    'gap_continuation': {
        'min_gap_down_pct': 0.04,
        'earliest_entry_bar': 6,
        'stop_pct': 0.025,
        'min_stop_pct': 0.015,
        'target_pct': 0.015,
        'exit_hour': 15,
        'exit_min': 0,
        'max_entries_per_day': 1,
        'min_risk_reward': 0.3,
        'min_confidence': 0.35,
    },
    'gap_bounce': {
        'min_gap_down_pct': 0.04,
        'preferred_gap_min': 0.06,
        'max_daily_rsi': 40,
        'earliest_entry_bar': 6,
        'stop_pct': 0.50,
        'min_stop_pct': 0.40,
        'target_pct': 0.50,
        'exit_hour': 15,
        'exit_min': 0,
        'max_entries_per_day': 1,
        'min_risk_reward': 0.01,
        'min_confidence': 0.35,
    },
    'micro_scalp': {
        'require_ema_stack': True,
        'min_ema_spread': 0.0015,
        'ema9_pullback_zone': 0.003,
        'require_above_vwap_long': True,
        'rsi_long_min': 40,
        'rsi_long_max': 68,
        'rsi_short_min': 32,
        'rsi_short_max': 60,
        'min_rvol': 1.0,
        'min_trend_bars': 3,
        'stop_beyond_ema20': 0.002,
        'min_stop_pct': 0.003,
        'max_stop_pct': 0.006,
        'target_rr': 0.7,
        'min_target_pct': 0.002,
        'max_hold_minutes': 25,
        'max_entries_per_day': 4,
        'min_risk_reward': 0.5,
        'min_confidence': 0.40,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """An open backtest position."""
    symbol: str
    direction: str              # 'long' or 'short'
    entry_price: float          # actual fill (after slippage)
    stop_price: float
    target_price: float
    shares: int
    entry_time: datetime
    strategy_id: str
    signal_price: float         # price at signal time (before slippage)
    setup: Optional[IntradaySetup] = field(default=None, repr=False)


@dataclass
class Trade:
    """A completed backtest trade."""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    entry_time: str
    exit_time: str
    exit_reason: str
    strategy_id: str
    holding_minutes: int = 0
    slippage_cost: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Strategy Factory
# ═══════════════════════════════════════════════════════════════════════

def create_strategy(strategy_id: str, config: Optional[Dict] = None) -> IntradayStrategy:
    """Instantiate a strategy with backtest-tuned config overrides."""
    cls = IntradayStrategyRegistry.get_strategy_class(strategy_id)
    instance = cls()
    merged = dict(instance.get_default_config())
    merged.update(BACKTEST_CONFIG_OVERRIDES.get(strategy_id, {}))
    if config:
        merged.update(config)
    return cls(merged)


def create_all_strategies(
    only: Optional[List[str]] = None,
) -> Dict[str, IntradayStrategy]:
    """Create instances of all (or selected) strategies."""
    all_ids = IntradayStrategyRegistry.get_all_ids()
    selected = only if only else all_ids
    result: Dict[str, IntradayStrategy] = {}
    for sid in selected:
        if sid not in all_ids:
            log.warning("Unknown strategy '%s', skipping. Available: %s", sid, all_ids)
            continue
        result[sid] = create_strategy(sid)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def get_trading_dates(conn, start_date: str, end_date: str) -> List[str]:
    """Get distinct trading dates with minute data in the range."""
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT ts::date::text AS d
        FROM minute_bars
        WHERE ts >= %s::date AND ts < (%s::date + 1)
        ORDER BY d
    """, (start_date, end_date))
    return [row[0] for row in cur.fetchall()]


def load_day_bars(
    conn, symbols: List[str], date_str: str,
) -> Dict[str, List[dict]]:
    """Load all 1-min bars for given symbols on a single date.

    Returns {symbol: [bar_dict, ...]} sorted by timestamp.
    Timestamps in the DB are naive ET (per project convention).
    Filters to RTH: 9:30-15:59.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT symbol, ts, open, high, low, close, volume
        FROM minute_bars
        WHERE symbol = ANY(%s)
          AND ts::date = %s::date
          AND ts::time >= '09:30'
          AND ts::time <= '15:59'
        ORDER BY symbol, ts
    """, (symbols, date_str))

    result: Dict[str, List[dict]] = defaultdict(list)
    for row in cur.fetchall():
        result[row[0]].append({
            'ts': row[1],   # naive datetime (ET)
            'open': float(row[2]),
            'high': float(row[3]),
            'low': float(row[4]),
            'close': float(row[5]),
            'volume': int(row[6] or 0),
        })
    return dict(result)


def load_prev_close(conn, date_str: str, symbols: List[str]) -> Dict[str, float]:
    """Load previous trading day close for each symbol (for gap detection)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT symbol, close
        FROM daily_bars
        WHERE date = (SELECT MAX(date) FROM daily_bars WHERE date < %s)
          AND symbol = ANY(%s)
    """, (date_str, symbols))
    return {row[0]: float(row[1]) for row in cur.fetchall()}


def load_daily_rsi(conn, date_str: str, symbols: List[str]) -> Dict[str, float]:
    """Load pre-computed daily RSI(14) using the last 14 daily changes."""
    cur = conn.cursor()
    cur.execute("""
        WITH recent AS (
            SELECT symbol, date, close,
                   close - LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS change
            FROM daily_bars
            WHERE symbol = ANY(%s)
              AND date <= (SELECT MAX(date) FROM daily_bars WHERE date < %s)
            ORDER BY date DESC
        ),
        last_15 AS (
            SELECT symbol, date, change,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
            FROM recent
            WHERE change IS NOT NULL
        ),
        rsi_calc AS (
            SELECT symbol,
                   AVG(CASE WHEN change > 0 THEN change ELSE 0 END) AS avg_gain,
                   AVG(CASE WHEN change < 0 THEN ABS(change) ELSE 0 END) AS avg_loss
            FROM last_15
            WHERE rn <= 14
            GROUP BY symbol
            HAVING COUNT(*) >= 10
        )
        SELECT symbol,
               CASE WHEN avg_loss > 0
                    THEN 100 - (100 / (1 + avg_gain / avg_loss))
                    ELSE 100 END AS rsi
        FROM rsi_calc
    """, (symbols, date_str))
    return {row[0]: float(row[1]) for row in cur.fetchall()}


# ═══════════════════════════════════════════════════════════════════════
# 5-Min Aggregation
# ═══════════════════════════════════════════════════════════════════════

def aggregate_to_5min(bars_1min: List[dict]) -> List[dict]:
    """Aggregate 1-minute bars into 5-minute OHLCV bars.

    bar_time uses 'H:MM' format (e.g. '9:30', '10:25') to match
    what the TickIndicatorEngine and strategies expect.
    """
    if not bars_1min:
        return []

    five_min: List[dict] = []
    current: Optional[dict] = None
    current_key: Optional[Tuple[int, int]] = None

    for bar in bars_1min:
        ts = bar['ts']
        key = (ts.hour, (ts.minute // 5) * 5)

        if key != current_key:
            if current is not None:
                five_min.append(current)
            bar_time = f'{key[0]}:{key[1]:02d}'
            current = {
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close'],
                'volume': bar['volume'],
                'bar_time': bar_time,
                'ts': ts,
            }
            current_key = key
        else:
            current['high'] = max(current['high'], bar['high'])
            current['low'] = min(current['low'], bar['low'])
            current['close'] = bar['close']
            current['volume'] += bar['volume']
            current['ts'] = ts

    if current is not None:
        five_min.append(current)

    return five_min


# ═══════════════════════════════════════════════════════════════════════
# Multi-Strategy Backtester
# ═══════════════════════════════════════════════════════════════════════

class IntradayBacktesterV2:
    """Bar-by-bar multi-strategy intraday backtester.

    Runs all enabled strategies simultaneously. Each strategy scans every
    symbol on every 5-min bar completion, but the position pool and risk
    limits are shared.
    """

    def __init__(
        self,
        strategies: Dict[str, IntradayStrategy],
        capital: float = DEFAULT_CAPITAL,
        risk_pct: float = RISK_PCT_PER_TRADE,
        max_positions: int = MAX_CONCURRENT_POSITIONS,
        max_trades_per_day: int = MAX_TRADES_PER_DAY,
        slippage_pct: float = SLIPPAGE_PCT,
    ):
        self.strategies = strategies
        self.initial_capital = capital
        self.equity = capital
        self.risk_pct = risk_pct
        self.max_positions = max_positions
        self.max_trades_per_day = max_trades_per_day
        self.slippage_pct = slippage_pct

        # Shared indicator engine covering all strategies' needs
        all_indicators: set = set()
        all_ema_periods: set = set()
        for strat in strategies.values():
            all_indicators.update(strat.get_required_indicators())
            all_ema_periods.update(strat.get_ema_periods())
        # Always include fundamentals
        all_indicators.update(['day_high', 'day_low', 'bar_history'])
        all_ema_periods.update([9, 20, 50])

        self.engine = TickIndicatorEngine(
            indicators=list(all_indicators),
            ema_periods=sorted(all_ema_periods),
            rsi_period=14,
            atr_period=14,
            or_minutes=15,
        )

        # State
        self.positions: Dict[str, Position] = {}      # symbol -> Position
        self.trades: List[Trade] = []
        self.daily_pnl: Dict[str, float] = {}         # date -> daily pnl
        self._day_trade_count: int = 0

    # ── Public ────────────────────────────────────────────────────────

    def run(
        self,
        conn,
        symbols: List[str],
        start_date: str,
        end_date: str,
        progress_every: int = 20,
    ) -> Dict:
        """Run backtest. Returns comprehensive results dict."""
        t0 = time_mod.time()
        dates = get_trading_dates(conn, start_date, end_date)
        strat_names = list(self.strategies.keys())
        log.info(
            "Backtest v2: %d strategies, %d symbols, %d dates (%s to %s)",
            len(strat_names), len(symbols), len(dates), start_date, end_date,
        )
        log.info("  Strategies: %s", ', '.join(strat_names))

        for i, date_str in enumerate(dates):
            self._simulate_day(conn, symbols, date_str)
            if (i + 1) % progress_every == 0 or (i + 1) == len(dates):
                wins = sum(1 for t in self.trades if t.pnl > 0)
                total = len(self.trades)
                wr = wins / total * 100 if total else 0.0
                log.info(
                    "  Day %d/%d (%s): %d trades, WR=%.1f%%, equity=$%.0f",
                    i + 1, len(dates), date_str, total, wr, self.equity,
                )

        elapsed = time_mod.time() - t0
        results = self._compute_results(symbols)
        results['elapsed_sec'] = round(elapsed, 1)
        results['start_date'] = start_date
        results['end_date'] = end_date
        results['num_symbols'] = len(symbols)
        results['num_dates'] = len(dates)

        self._print_summary(results)
        return results

    # ── Day Simulation ────────────────────────────────────────────────

    def _simulate_day(self, conn, symbols: List[str], date_str: str):
        """Simulate one trading day across all strategies."""
        # Reset daily state
        self.engine.reset()
        self._day_trade_count = 0
        for strat in self.strategies.values():
            strat.on_day_start()

        # Load all 1-min bars for this date
        all_bars = load_day_bars(conn, symbols, date_str)
        if not all_bars:
            return

        active_symbols = [s for s, bars in all_bars.items() if len(bars) >= MIN_BARS_PER_DAY]
        if not active_symbols:
            return

        # Feed prev_close and daily_rsi to strategies that need them
        prev_closes = load_prev_close(conn, date_str, active_symbols)
        daily_rsis = load_daily_rsi(conn, date_str, active_symbols)
        for strat in self.strategies.values():
            if hasattr(strat, '_prev_close'):
                strat._prev_close = dict(prev_closes)
            if hasattr(strat, '_daily_rsi'):
                strat._daily_rsi = dict(daily_rsis)

        # Aggregate to 5-min bars per symbol
        fivem_by_sym: Dict[str, List[dict]] = {}
        for sym in active_symbols:
            fivem_by_sym[sym] = aggregate_to_5min(all_bars[sym])

        # Build index: 1-min bars by (symbol, 5min_slot)
        bars_1min_idx: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
        for sym in active_symbols:
            for bar in all_bars[sym]:
                ts = bar['ts']
                slot = f'{ts.hour}:{(ts.minute // 5) * 5:02d}'
                bars_1min_idx[sym][slot].append(bar)

        # Collect all unique 5-min time slots, sorted chronologically
        all_slots: set = set()
        for bars5 in fivem_by_sym.values():
            for b in bars5:
                all_slots.add(b['bar_time'])
        sorted_slots = sorted(
            all_slots,
            key=lambda x: (int(x.split(':')[0]), int(x.split(':')[1])),
        )

        day_pnl = 0.0

        for slot in sorted_slots:
            h, m = int(slot.split(':')[0]), int(slot.split(':')[1])
            past_time_exit = (
                h > TIME_EXIT_HOUR
                or (h == TIME_EXIT_HOUR and m >= TIME_EXIT_MIN)
            )

            for sym in active_symbols:
                # Find the 5-min bar for this slot
                bar5 = None
                for b in fivem_by_sym.get(sym, []):
                    if b['bar_time'] == slot:
                        bar5 = b
                        break
                if bar5 is None:
                    continue

                # 1) Check 1-min bars for stop/target exits (no strategy exits)
                one_min_bars = bars_1min_idx[sym].get(slot, [])
                for bar1 in one_min_bars:
                    if sym in self.positions:
                        pos = self.positions[sym]
                        exit_reason = self._check_stop_target(pos, bar1)
                        if exit_reason:
                            trade = self._close_position(pos, bar1, exit_reason)
                            day_pnl += trade.pnl

                    # Time exit
                    if past_time_exit and sym in self.positions:
                        pos = self.positions[sym]
                        trade = self._close_position(
                            pos, bar1, 'time_exit_3:50pm',
                        )
                        day_pnl += trade.pnl

                # 2) Feed 5-min bar to indicator engine
                self.engine.seed_bars(sym, [bar5], today=date_str)

                # Fix OR for days where data starts late
                si = self.engine._symbols.get(sym)
                if si and len(si.completed_bars) >= 3:
                    if si.or_high <= 0 or si.or_low >= float('inf'):
                        or_bars = si.completed_bars[:3]
                        si.or_high = max(b.high for b in or_bars)
                        si.or_low = min(b.low for b in or_bars)
                        si.or_complete = True

                # 3) Strategy-level exits on 5-min bar completion
                if sym in self.positions:
                    pos = self.positions[sym]
                    exit_reason = self._check_strategy_exit(pos, sym, bar5)
                    if exit_reason:
                        trade = self._close_position(pos, bar5, exit_reason)
                        day_pnl += trade.pnl

                # 4) Scan all strategies for new entries (NEXT bar entry model)
                #    Signal is generated here; entry would be at next bar open.
                #    We approximate by entering at this bar's close + slippage.
                if past_time_exit:
                    continue
                if sym in self.positions:
                    continue
                if len(self.positions) >= self.max_positions:
                    continue
                if self._day_trade_count >= self.max_trades_per_day:
                    continue

                tick_data = self.engine.get_data(sym)
                if not tick_data or tick_data.get('bar_count', 0) < 3:
                    continue

                # Fix volume_surge_ratio (seed_bars doesn't set current_bar)
                if si and si.completed_bars and si.volume_avg_20bar > 0:
                    tick_data['volume_surge_ratio'] = (
                        si.completed_bars[-1].volume / si.volume_avg_20bar
                    )
                elif si and si.completed_bars:
                    tick_data['volume_surge_ratio'] = 1.0

                snapshot = {'price': bar5['close']}
                best_setup = self._pick_best_setup(sym, tick_data, snapshot, bar5)
                if best_setup is not None:
                    self._open_position(best_setup, bar5)

        # Force close any remaining positions at EOD
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            sym_bars = all_bars.get(sym, [])
            if sym_bars:
                trade = self._close_position(pos, sym_bars[-1], 'eod_close')
                day_pnl += trade.pnl

        self.daily_pnl[date_str] = day_pnl
        for strat in self.strategies.values():
            strat.on_day_end()

    # ── Setup Selection ───────────────────────────────────────────────

    def _pick_best_setup(
        self,
        symbol: str,
        tick_data: Dict,
        snapshot: Dict,
        bar5: dict,
    ) -> Optional[IntradaySetup]:
        """Scan all strategies, return the highest-confidence valid setup."""
        candidates: List[IntradaySetup] = []

        for sid, strat in self.strategies.items():
            try:
                setup = strat.scan_for_setups(symbol, tick_data, snapshot, bar5['ts'])
            except Exception:
                continue

            if setup is None:
                continue

            try:
                valid, reason = strat.validate_setup(setup, tick_data, bar5['ts'])
            except Exception:
                continue

            if valid:
                candidates.append(setup)

        if not candidates:
            return None

        # Pick highest confidence; break ties by risk_reward
        candidates.sort(key=lambda s: (s.confidence, s.risk_reward), reverse=True)
        return candidates[0]

    # ── Position Management ───────────────────────────────────────────

    def _open_position(self, setup: IntradaySetup, bar5: dict):
        """Open a position from a setup signal.

        Entry at NEXT bar's open approximated by this bar's close + slippage.
        """
        signal_price = setup.entry_price
        if signal_price <= 0:
            return

        # Apply slippage to entry
        if setup.direction == 'long':
            entry_price = signal_price * (1 + self.slippage_pct)
        else:
            entry_price = signal_price * (1 - self.slippage_pct)

        # Widen stops: ensure minimum distance of 0.8% or 2x ATR
        tick_data = self.engine.get_data(setup.symbol)
        atr = tick_data.get('atr', 0) if tick_data else 0
        min_stop_dist = max(entry_price * 0.008, atr * 2.0)
        actual_dist = abs(entry_price - setup.stop_price)
        if actual_dist < min_stop_dist and min_stop_dist > 0:
            if setup.direction == 'long':
                setup.stop_price = round(entry_price - min_stop_dist, 2)
            else:
                setup.stop_price = round(entry_price + min_stop_dist, 2)
            risk = abs(entry_price - setup.stop_price)
            target_rr = max(setup.risk_reward, 1.5)
            if setup.direction == 'long':
                setup.target_price = round(entry_price + risk * target_rr, 2)
            else:
                setup.target_price = round(entry_price - risk * target_rr, 2)

        # Position sizing: risk_pct of equity per trade
        risk_per_share = abs(entry_price - setup.stop_price)
        if risk_per_share <= 0:
            return
        risk_dollars = self.equity * self.risk_pct
        shares = int(risk_dollars / risk_per_share)
        if shares <= 0:
            return

        # Cap position size at 10% of equity
        max_shares = int(self.equity * 0.10 / entry_price)
        shares = min(shares, max(max_shares, 1))

        pos = Position(
            symbol=setup.symbol,
            direction=setup.direction,
            entry_price=round(entry_price, 4),
            stop_price=setup.stop_price,
            target_price=setup.target_price,
            shares=shares,
            entry_time=bar5['ts'],
            strategy_id=setup.strategy_id,
            signal_price=signal_price,
            setup=setup,
        )
        self.positions[setup.symbol] = pos
        self._day_trade_count += 1

        # Notify strategy of entry
        strat = self.strategies.get(setup.strategy_id)
        if strat and hasattr(strat, 'record_entry'):
            try:
                strat.record_entry(setup.symbol, bar5['ts'])
            except TypeError:
                strat.record_entry(setup.symbol)

    def _close_position(self, pos: Position, bar: dict, reason: str) -> Trade:
        """Close position at bar's close + slippage. Returns Trade."""
        raw_exit = bar['close']

        # Apply slippage to exit
        if pos.direction == 'long':
            exit_price = raw_exit * (1 - self.slippage_pct)
        else:
            exit_price = raw_exit * (1 + self.slippage_pct)

        if pos.direction == 'long':
            pnl_per_share = exit_price - pos.entry_price
        else:
            pnl_per_share = pos.entry_price - exit_price

        gross_pnl = pnl_per_share * pos.shares
        # Slippage already baked into entry/exit prices; no separate commission
        net_pnl = gross_pnl
        pnl_pct = pnl_per_share / pos.entry_price if pos.entry_price > 0 else 0.0

        entry_slippage = abs(pos.entry_price - pos.signal_price) * pos.shares
        exit_slippage = abs(exit_price - raw_exit) * pos.shares
        total_slippage = entry_slippage + exit_slippage

        holding_min = 0
        ts_exit = bar['ts']
        if isinstance(ts_exit, datetime) and isinstance(pos.entry_time, datetime):
            holding_min = int((ts_exit - pos.entry_time).total_seconds() / 60)

        trade = Trade(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=round(exit_price, 4),
            shares=pos.shares,
            pnl=round(net_pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            entry_time=str(pos.entry_time),
            exit_time=str(ts_exit),
            exit_reason=reason,
            strategy_id=pos.strategy_id,
            holding_minutes=holding_min,
            slippage_cost=round(total_slippage, 2),
        )
        self.trades.append(trade)
        self.equity += net_pnl
        del self.positions[pos.symbol]
        return trade

    # ── Exit Checks ───────────────────────────────────────────────────

    def _check_stop_target(self, pos: Position, bar1: dict) -> Optional[str]:
        """Check stop-loss and target on a 1-min bar. Returns reason or None.

        Order: check favourable direction first (target for longs = high first,
        target for shorts = low first) to avoid anti-short bias.
        """
        if pos.direction == 'long':
            # Target is up, stop is down
            if bar1['high'] >= pos.target_price:
                return 'target_hit'
            if bar1['low'] <= pos.stop_price:
                return 'stop_loss'
        else:
            # Target is down, stop is up
            if bar1['low'] <= pos.target_price:
                return 'target_hit'
            if bar1['high'] >= pos.stop_price:
                return 'stop_loss'
        return None

    def _check_strategy_exit(
        self, pos: Position, symbol: str, bar5: dict,
    ) -> Optional[str]:
        """Check strategy-level exit and trailing stop on a 5-min bar."""
        strat = self.strategies.get(pos.strategy_id)
        if strat is None:
            return None

        tick_data = self.engine.get_data(symbol)
        position_dict = {
            'direction': pos.direction,
            'entry_price': pos.entry_price,
            'stop_price': pos.stop_price,
            'entry_time': pos.entry_time.strftime('%Y-%m-%d %H:%M')
            if isinstance(pos.entry_time, datetime) else str(pos.entry_time),
        }
        price = bar5['close']

        # Strategy-specific exit
        try:
            exit_signal = strat.evaluate_exit(position_dict, price, tick_data, bar5['ts'])
            if exit_signal and getattr(exit_signal, 'action', '') == 'close':
                return f'strategy_exit:{getattr(exit_signal, "reason", "")}'
        except Exception:
            pass

        # Trailing stop update
        try:
            new_stop = strat.update_trailing_stop(
                position_dict, price, tick_data, bar5['ts'],
            )
            if new_stop is not None:
                if pos.direction == 'long' and new_stop > pos.stop_price:
                    pos.stop_price = new_stop
                elif pos.direction == 'short' and new_stop < pos.stop_price:
                    pos.stop_price = new_stop
        except Exception:
            pass

        return None

    # ── Results Computation ───────────────────────────────────────────

    def _compute_results(self, symbols: List[str]) -> Dict:
        """Compute comprehensive results with per-strategy and per-symbol breakdowns."""
        overall = self._compute_metrics(self.trades, 'combined')

        # Per-strategy breakdown
        by_strategy: Dict[str, List[Trade]] = defaultdict(list)
        for t in self.trades:
            by_strategy[t.strategy_id].append(t)
        strategy_results = {}
        for sid in sorted(by_strategy.keys()):
            strategy_results[sid] = self._compute_metrics(by_strategy[sid], sid)

        # Per-symbol breakdown
        by_symbol: Dict[str, List[Trade]] = defaultdict(list)
        for t in self.trades:
            by_symbol[t.symbol].append(t)
        symbol_results = {}
        for sym in sorted(by_symbol.keys()):
            symbol_results[sym] = self._compute_metrics(by_symbol[sym], sym)

        # Monthly P&L
        monthly_pnl: Dict[str, float] = defaultdict(float)
        monthly_trades: Dict[str, int] = defaultdict(int)
        for t in self.trades:
            month = t.entry_time[:7]
            monthly_pnl[month] += t.pnl
            monthly_trades[month] += 1

        return {
            'overall': overall,
            'by_strategy': strategy_results,
            'by_symbol': symbol_results,
            'monthly_pnl': dict(monthly_pnl),
            'monthly_trades': dict(monthly_trades),
            'daily_pnl': self.daily_pnl,
            'initial_capital': self.initial_capital,
            'final_equity': round(self.equity, 2),
            'total_slippage': round(sum(t.slippage_cost for t in self.trades), 2),
            'all_trades': [asdict(t) for t in self.trades],
        }

    def _compute_metrics(self, trades: List[Trade], label: str) -> Dict:
        """Compute metrics for a subset of trades."""
        if not trades:
            return {
                'label': label, 'total_trades': 0, 'wins': 0, 'losses': 0,
                'win_rate': 0.0, 'total_pnl': 0.0, 'profit_factor': 0.0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'max_win': 0.0,
                'max_loss': 0.0, 'avg_holding_min': 0.0,
                'max_drawdown_pct': 0.0, 'max_consecutive_losses': 0,
                'sharpe': 0.0, 'exit_reasons': {},
                'long_trades': 0, 'long_wins': 0,
                'short_trades': 0, 'short_wins': 0,
            }

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total_win_pnl = sum(t.pnl for t in wins)
        total_loss_pnl = abs(sum(t.pnl for t in losses))
        pf = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float('inf')

        # Max consecutive losses
        max_consec = 0
        cur_consec = 0
        for t in trades:
            if t.pnl <= 0:
                cur_consec += 1
                max_consec = max(max_consec, cur_consec)
            else:
                cur_consec = 0

        # Max drawdown on equity curve
        eq = [self.initial_capital]
        for t in trades:
            eq.append(eq[-1] + t.pnl)
        peak = eq[0]
        max_dd = 0.0
        for e in eq:
            peak = max(peak, e)
            dd = (peak - e) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        # Sharpe ratio (annualized from daily returns)
        daily_returns = list(self.daily_pnl.values())
        sharpe = 0.0
        if len(daily_returns) > 1:
            mean_r = statistics.mean(daily_returns)
            std_r = statistics.stdev(daily_returns)
            if std_r > 0:
                sharpe = mean_r / std_r * math.sqrt(252)

        exit_reasons = Counter(
            t.exit_reason.split(':')[0].strip() for t in trades
        )

        long_trades = [t for t in trades if t.direction == 'long']
        short_trades = [t for t in trades if t.direction == 'short']

        return {
            'label': label,
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
            'total_pnl': round(sum(t.pnl for t in trades), 2),
            'profit_factor': round(pf, 2),
            'avg_win': round(total_win_pnl / len(wins), 2) if wins else 0.0,
            'avg_loss': round(-total_loss_pnl / len(losses), 2) if losses else 0.0,
            'max_win': round(max(t.pnl for t in trades), 2),
            'max_loss': round(min(t.pnl for t in trades), 2),
            'avg_holding_min': round(
                sum(t.holding_minutes for t in trades) / len(trades), 1,
            ),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'max_consecutive_losses': max_consec,
            'sharpe': round(sharpe, 2),
            'exit_reasons': dict(exit_reasons),
            'long_trades': len(long_trades),
            'long_wins': sum(1 for t in long_trades if t.pnl > 0),
            'short_trades': len(short_trades),
            'short_wins': sum(1 for t in short_trades if t.pnl > 0),
        }

    # ── Print Summary ─────────────────────────────────────────────────

    def _print_summary(self, results: Dict):
        """Print a formatted summary to stdout."""
        o = results['overall']
        print()
        print('=' * 78)
        print('  INTRADAY BACKTEST v2 — RESULTS')
        print('=' * 78)
        print(f"  Period:   {results.get('start_date')} to {results.get('end_date')}")
        print(f"  Symbols:  {results.get('num_symbols')}    "
              f"Days: {results.get('num_dates')}    "
              f"Time: {results.get('elapsed_sec', 0):.1f}s")
        print(f"  Capital:  ${results['initial_capital']:,.0f} -> "
              f"${results['final_equity']:,.0f}    "
              f"Slippage: ${results['total_slippage']:,.0f}")
        print('-' * 78)
        print(f"  Trades: {o['total_trades']}   "
              f"Wins: {o['wins']}   "
              f"Losses: {o['losses']}   "
              f"WR: {o['win_rate']:.1f}%")
        print(f"  P&L: ${o['total_pnl']:,.2f}   "
              f"PF: {o['profit_factor']:.2f}   "
              f"Sharpe: {o['sharpe']:.2f}   "
              f"MaxDD: {o['max_drawdown_pct']:.1f}%")
        print(f"  Avg Win: ${o['avg_win']:,.2f}   "
              f"Avg Loss: ${o['avg_loss']:,.2f}   "
              f"Avg Hold: {o['avg_holding_min']:.0f}m")
        print(f"  Long: {o['long_trades']} ({o['long_wins']}W)   "
              f"Short: {o['short_trades']} ({o['short_wins']}W)")

        # Per-strategy breakdown
        by_strat = results.get('by_strategy', {})
        if by_strat:
            print()
            print('-' * 78)
            header = (
                f"  {'Strategy':<25s} {'Trades':>6s} {'WR%':>6s} "
                f"{'PF':>6s} {'P&L':>10s} {'Sharpe':>7s} {'AvgHold':>7s}"
            )
            print(header)
            print('-' * 78)
            for sid in sorted(by_strat.keys()):
                m = by_strat[sid]
                print(
                    f"  {sid:<25s} {m['total_trades']:>6d} "
                    f"{m['win_rate']:>5.1f}% {m['profit_factor']:>6.2f} "
                    f"${m['total_pnl']:>9,.0f} {m['sharpe']:>7.2f} "
                    f"{m['avg_holding_min']:>6.0f}m"
                )

        # Monthly P&L
        monthly = results.get('monthly_pnl', {})
        if monthly:
            print()
            print('-' * 78)
            print('  Monthly P&L:')
            for month in sorted(monthly.keys()):
                trades_count = results.get('monthly_trades', {}).get(month, 0)
                pnl = monthly[month]
                bar = '+' * max(0, int(pnl / 50)) if pnl > 0 else '-' * min(40, max(0, int(-pnl / 50)))
                print(f"    {month}: ${pnl:>9,.0f}  ({trades_count:>3d} trades)  {bar}")

        # Top/bottom symbols
        by_sym = results.get('by_symbol', {})
        if by_sym:
            sorted_syms = sorted(by_sym.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
            print()
            print('-' * 78)
            print('  Top 5 Symbols:')
            for sym, m in sorted_syms[:5]:
                print(f"    {sym:<8s}: ${m['total_pnl']:>8,.0f}  "
                      f"({m['total_trades']} trades, WR={m['win_rate']:.0f}%)")
            print('  Bottom 5 Symbols:')
            for sym, m in sorted_syms[-5:]:
                print(f"    {sym:<8s}: ${m['total_pnl']:>8,.0f}  "
                      f"({m['total_trades']} trades, WR={m['win_rate']:.0f}%)")

        # Exit reason breakdown
        if o.get('exit_reasons'):
            print()
            print('-' * 78)
            print('  Exit Reasons:')
            for reason, count in sorted(o['exit_reasons'].items(), key=lambda x: -x[1]):
                print(f"    {reason:<30s}: {count:>5d}")

        print('=' * 78)
        print()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Intraday Strategy Backtester v2 — multi-strategy simulation',
    )
    parser.add_argument(
        '--strategy', action='append', dest='strategies', default=None,
        help='Strategy ID to run (can repeat). Default: all 13.',
    )
    parser.add_argument(
        '--start', default='2024-01-01',
        help='Start date YYYY-MM-DD (default: 2024-01-01)',
    )
    parser.add_argument(
        '--end', default='2026-03-21',
        help='End date YYYY-MM-DD (default: 2026-03-21)',
    )
    parser.add_argument(
        '--capital', type=float, default=DEFAULT_CAPITAL,
        help=f'Starting capital (default: {DEFAULT_CAPITAL})',
    )
    parser.add_argument(
        '--max-positions', type=int, default=MAX_CONCURRENT_POSITIONS,
        help=f'Max concurrent positions (default: {MAX_CONCURRENT_POSITIONS})',
    )
    parser.add_argument(
        '--max-daily-trades', type=int, default=MAX_TRADES_PER_DAY,
        help=f'Max trades per day (default: {MAX_TRADES_PER_DAY})',
    )
    parser.add_argument(
        '--slippage', type=float, default=SLIPPAGE_PCT,
        help=f'Slippage per side as decimal (default: {SLIPPAGE_PCT})',
    )
    parser.add_argument(
        '--risk', type=float, default=RISK_PCT_PER_TRADE,
        help=f'Risk per trade as pct of equity (default: {RISK_PCT_PER_TRADE})',
    )
    parser.add_argument(
        '--output', '-o', default=None,
        help='Path to write JSON results (default: print only)',
    )
    parser.add_argument(
        '--symbols', nargs='+', default=None,
        help='Override symbol universe (default: top-50 large caps)',
    )
    parser.add_argument(
        '--list-strategies', action='store_true',
        help='List available strategies and exit',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_strategies:
        all_ids = IntradayStrategyRegistry.get_all_ids()
        print(f"\nAvailable strategies ({len(all_ids)}):")
        for sid in sorted(all_ids):
            strat = IntradayStrategyRegistry.create_strategy(sid)
            print(f"  {sid:<25s} — {strat.name}")
        return

    if args.symbols and len(args.symbols) == 1 and args.symbols[0] == 'ALL':
        # Load full universe from DB: all symbols with >100 days of minute bars
        _conn = psycopg2.connect(DB_URL)
        _cur = _conn.cursor()
        _cur.execute("SELECT symbol FROM minute_bars GROUP BY symbol HAVING COUNT(DISTINCT ts::date) > 100 ORDER BY symbol")
        symbols = [r[0] for r in _cur.fetchall()]
        _conn.close()
        log.info("Loaded %d symbols from DB (full universe)", len(symbols))
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = UNIVERSE
    strategies = create_all_strategies(only=args.strategies)
    if not strategies:
        log.error("No valid strategies selected. Use --list-strategies to see options.")
        sys.exit(1)

    conn = psycopg2.connect(DB_URL)
    try:
        bt = IntradayBacktesterV2(
            strategies=strategies,
            capital=args.capital,
            risk_pct=args.risk,
            max_positions=args.max_positions,
            max_trades_per_day=args.max_daily_trades,
            slippage_pct=args.slippage,
        )
        results = bt.run(conn, symbols, args.start, args.end)

        if args.output:
            # Strip all_trades for large files; keep only last 500
            output = dict(results)
            if len(output.get('all_trades', [])) > 500:
                output['all_trades_truncated'] = True
                output['all_trades'] = output['all_trades'][-500:]
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2, default=str)
            log.info("Results written to %s", args.output)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
