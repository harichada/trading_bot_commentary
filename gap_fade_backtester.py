"""
Vectorbt-powered gap-fade backtester module.

Bulk-loads all daily bars into RAM, uses numpy for gap scanning, and
optionally leverages vectorbt for portfolio analytics.  Drop-in replacement
for GapFadeBacktester — same constructor, same run() signature, same result dict.

Performance: the main speedup comes from BatchDataProvider which eliminates
per-candidate SQLite queries (e.g. Minervini: 3ms → 0.05ms per candidate).
"""
from __future__ import annotations

import asyncio
import logging
import math
import random
import sqlite3
import time as _time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from itertools import groupby
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- Graceful vectorbt import ---
try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False

# --- Imports from gap_fade_app (all are dataclasses / pure functions) ---
from gap_fade_app import (
    GapFadeConfig,
    GapCandidate,
    TradeRecord,
    StopOutRecord,
    MarketRegime,
    DailyStats,
    PriceDB,
    get_price_db,
    UNIVERSE,
    compute_adaptive_stop_pct,
    _direction_pnl,
    _stop_hit,
    _target_hit,
    fetch_market_regime_backtest,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DataLoader — bulk-load all daily bars into RAM
# ---------------------------------------------------------------------------

class DataLoader:
    """Bulk-loads all daily bars for a symbol universe into a wide DataFrame."""

    def __init__(self, db_path: str = None):
        self._db_path = db_path or PriceDB.DB_PATH

    def load_universe(self, symbols: List[str], start: str, end: str,
                      warmup_days: int = 252) -> pd.DataFrame:
        """Load daily bars for all symbols in one bulk SQL query.

        Returns a wide DataFrame with DatetimeIndex and MultiIndex columns
        (field, symbol) where field ∈ {open, high, low, close, volume}.
        """
        adj_start = (datetime.strptime(start, '%Y-%m-%d')
                     - timedelta(days=int(warmup_days * 1.5))).strftime('%Y-%m-%d')

        conn = sqlite3.connect(self._db_path)
        try:
            # For smaller universes, use IN-clause for targeted SQL.
            # For very large universes (>2000), full scan + Python filter is faster.
            if len(symbols) <= 2000:
                rows = []
                chunk_size = 500
                for i in range(0, len(symbols), chunk_size):
                    chunk = symbols[i:i + chunk_size]
                    placeholders = ','.join('?' * len(chunk))
                    sql = (
                        f'SELECT symbol, date, open, high, low, close, volume '
                        f'FROM daily_bars WHERE symbol IN ({placeholders}) '
                        f'AND date >= ? AND date <= ? ORDER BY symbol, date'
                    )
                    cur = conn.execute(sql, chunk + [adj_start, end])
                    rows.extend(cur.fetchall())
            else:
                sql = (
                    'SELECT symbol, date, open, high, low, close, volume '
                    'FROM daily_bars WHERE date >= ? AND date <= ? '
                    'ORDER BY symbol, date'
                )
                cur = conn.execute(sql, (adj_start, end))
                rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return pd.DataFrame()

        # Build long-form DataFrame
        df = pd.DataFrame(rows, columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'])

        # For full-scan path, filter to requested symbols
        if len(symbols) > 2000:
            sym_set = set(symbols)
            df = df[df['symbol'].isin(sym_set)]
        if df.empty:
            return pd.DataFrame()

        for col in ('open', 'high', 'low', 'close', 'volume'):
            df[col] = df[col].astype(np.float64)

        # Pivot to wide format: index=date, columns=MultiIndex(field, symbol)
        df = df.set_index(['date', 'symbol'])
        wide = df.unstack(level='symbol')  # columns become (field, symbol)
        wide.sort_index(inplace=True)
        return wide

    def load_spy(self, start: str, end: str) -> pd.DataFrame:
        """Load SPY bars for regime filter and relative strength.

        Returns DataFrame with DatetimeIndex, columns=[open, high, low, close, volume].
        """
        conn = sqlite3.connect(self._db_path)
        try:
            adj_start = (datetime.strptime(start, '%Y-%m-%d') - timedelta(days=10)).strftime('%Y-%m-%d')
            cur = conn.execute(
                'SELECT date, open, high, low, close, volume FROM daily_bars '
                'WHERE symbol = ? AND date >= ? AND date <= ? ORDER BY date',
                ('SPY', adj_start, end)
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        for col in ('open', 'high', 'low', 'close', 'volume'):
            df[col] = df[col].astype(np.float64)
        df.set_index('date', inplace=True)
        return df

    @staticmethod
    def get_symbol_bars(symbol: str, all_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Zero-copy slice for one symbol from the wide DataFrame.

        Returns DataFrame with columns [open, high, low, close, volume]
        or None if symbol not present.
        """
        if all_data.empty:
            return None
        try:
            sliced = all_data.xs(symbol, axis=1, level=1, drop_level=True)
        except KeyError:
            return None
        # Drop rows where all values are NaN (dates when symbol had no data)
        sliced = sliced.dropna(how='all')
        if sliced.empty:
            return None
        sliced.index.name = symbol
        return sliced


# ---------------------------------------------------------------------------
# BatchDataProvider — patches PriceDB.get_bars for RAM reads
# ---------------------------------------------------------------------------

class BatchDataProvider:
    """Provides PriceDB.get_bars()-compatible interface from pre-loaded DataFrame."""

    def __init__(self, all_data: pd.DataFrame):
        self._data = all_data

    def get_bars(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Return bars from pre-loaded DataFrame, matching PriceDB.get_bars format.

        Returns DataFrame with DatetimeIndex (named as symbol),
        columns=[open, high, low, close, volume], or None.
        """
        sliced = DataLoader.get_symbol_bars(symbol, self._data)
        if sliced is None:
            return None

        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        mask = (sliced.index >= start_dt) & (sliced.index <= end_dt)
        result = sliced.loc[mask]
        if result.empty:
            return None
        return result


# ---------------------------------------------------------------------------
# VectorizedGapScanner — numpy gap detection across all symbols
# ---------------------------------------------------------------------------

class VectorizedGapScanner:
    """Detects gaps using vectorized numpy operations on in-memory DataFrame."""

    def __init__(self, config: GapFadeConfig):
        self.config = config

    def scan(self, data: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        """Vectorized gap detection across all symbols.

        Args:
            data: Wide DataFrame from DataLoader.load_universe()
            start, end: Date range (YYYY-MM-DD strings)

        Returns:
            DataFrame in long format with columns:
            date, symbol, open, high, low, close, prev_close, gap_pct,
            volume, avg_vol, vol_ratio, direction
        """
        if data.empty:
            return pd.DataFrame()

        cfg = self.config
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        closes = data['close']
        opens = data['open']
        highs = data['high']
        lows = data['low']
        volumes = data['volume']

        # Previous close (shift by 1 trading day)
        prev_close = closes.shift(1)

        # Gap % = (open - prev_close) / prev_close
        gap_pct = (opens - prev_close) / prev_close

        # 20-day rolling average volume (shifted so it's as-of yesterday)
        avg_vol = volumes.rolling(20, min_periods=5).mean().shift(1)

        # Volume ratio: previous day's volume / average volume
        prev_volume = volumes.shift(1)
        vol_ratio = prev_volume / avg_vol

        # --- Gap-up mask ---
        gap_up_mask = (
            (gap_pct >= cfg.gap_threshold)
            & (opens >= cfg.min_price)
            & (avg_vol >= cfg.min_avg_volume)
            & (prev_close > 0)
        )
        if cfg.max_gap_pct > 0:
            gap_up_mask &= (gap_pct <= cfg.max_gap_pct)

        # --- Gap-down mask (if enabled) ---
        if cfg.trade_gap_downs:
            gap_down_mask = (
                (gap_pct <= -cfg.gap_down_threshold)
                & (gap_pct >= -cfg.gap_down_max_pct)
                & (opens >= cfg.min_price)
                & (avg_vol >= cfg.min_avg_volume)
                & (prev_close > 0)
            )
            combined_mask = gap_up_mask | gap_down_mask
        else:
            combined_mask = gap_up_mask

        # Filter to date range — broadcast 1D index mask across all columns
        date_mask = pd.Series((data.index >= start_dt) & (data.index <= end_dt),
                              index=data.index)
        # Use .values to broadcast across columns (avoid shape mismatch)
        for sym in combined_mask.columns:
            combined_mask[sym] = combined_mask[sym] & date_mask.values

        # Stack to long format
        gap_results = []

        symbols = opens.columns.tolist()
        for sym in symbols:
            sym_mask = combined_mask[sym]
            if not sym_mask.any():
                continue

            dates_hit = data.index[sym_mask]
            for dt in dates_hit:
                gp = float(gap_pct.loc[dt, sym])
                direction = 'short' if gp > 0 else 'long'
                av = avg_vol.loc[dt, sym]
                vr = vol_ratio.loc[dt, sym]
                gap_results.append({
                    'date': dt.strftime('%Y-%m-%d'),
                    'symbol': sym,
                    'open': float(opens.loc[dt, sym]),
                    'high': float(highs.loc[dt, sym]),
                    'low': float(lows.loc[dt, sym]),
                    'close': float(closes.loc[dt, sym]),
                    'prev_close': float(prev_close.loc[dt, sym]),
                    'gap_pct': round(gp, 4),
                    'volume': int(volumes.loc[dt, sym]),
                    'avg_vol': int(av) if not np.isnan(av) else 0,
                    'vol_ratio': round(float(vr), 2) if not np.isnan(vr) else 1.0,
                    'direction': direction,
                })

        if not gap_results:
            return pd.DataFrame()

        gaps_df = pd.DataFrame(gap_results)
        gaps_df.sort_values('date', inplace=True)
        gaps_df.reset_index(drop=True, inplace=True)
        return gaps_df

    def apply_vol_filter(self, gaps_df: pd.DataFrame) -> pd.DataFrame:
        """Direction-aware vol_ratio filter."""
        if gaps_df.empty:
            return gaps_df

        cfg = self.config

        def _vol_ok(row):
            if row['direction'] == 'long':
                if not cfg.trade_gap_downs:
                    return False
                return row['vol_ratio'] <= cfg.gap_down_vol_ratio_max
            return row['vol_ratio'] <= cfg.vol_ratio_max

        mask = gaps_df.apply(_vol_ok, axis=1)
        return gaps_df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# SimulationState — mutable state for the simulation loop
# ---------------------------------------------------------------------------

@dataclass
class SimulationState:
    """Mutable state for the backtest simulation loop."""
    config: GapFadeConfig
    equity: float = 0.0
    peak_equity: float = 0.0
    trade_log: List[TradeRecord] = field(default_factory=list)
    all_trade_log: List[TradeRecord] = field(default_factory=list)
    daily_stats: DailyStats = field(default_factory=DailyStats)
    stopped_today: Dict[str, StopOutRecord] = field(default_factory=dict)
    effective_max_positions: int = 5
    positions: Dict[str, Any] = field(default_factory=dict)
    equity_curve: List[float] = field(default_factory=list)
    bt_log: List[dict] = field(default_factory=list)

    # Drawdown circuit breaker tracking
    dd_days_tier1: int = 0
    dd_days_tier2: int = 0
    dd_trades_skipped: int = 0       # gaps skipped by hard stop
    dd_hard_stopped: bool = False
    dd_hard_stop_date: str = ''

    def __post_init__(self):
        if self.equity == 0.0:
            self.equity = self.config.initial_capital
        if self.peak_equity == 0.0:
            self.peak_equity = self.config.initial_capital

    def record_trade(self, trade: TradeRecord):
        """Record a completed trade and update equity."""
        self.all_trade_log.append(trade)
        self.trade_log.append(trade)
        self.equity += trade.pnl
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Update daily stats
        self.daily_stats.trades += 1
        self.daily_stats.pnl += trade.pnl
        if trade.pnl > 0:
            self.daily_stats.wins += 1
            self.daily_stats.consecutive_losses = 0
        else:
            self.daily_stats.losses += 1
            self.daily_stats.consecutive_losses += 1

    def compute_kelly_size(self) -> float:
        """Compute optimal position size using Kelly criterion (study data)."""
        p = 0.709  # STUDY_WIN_RATE
        q = 1 - p
        b = abs(2.5 / -3.0)  # STUDY_AVG_WIN / STUDY_AVG_LOSS
        kelly = (p * b - q) / b if b > 0 else 0
        return max(0, kelly * self.config.kelly_fraction)

    def current_drawdown(self) -> float:
        """DD as fraction of initial capital (matches dashboard Max DD metric)."""
        if self.config.initial_capital <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.config.initial_capital

    def dd_tier(self) -> int:
        """0=normal, 1=reduced, 2=minimal, 3=hard_stopped."""
        if not self.config.dd_circuit_breaker:
            return 0
        if self.dd_hard_stopped:
            return 3
        dd = self.current_drawdown()
        if self.config.dd_hard_stop > 0 and dd >= self.config.dd_hard_stop:
            self.dd_hard_stopped = True
            return 3
        if dd >= self.config.dd_tier2_threshold:
            return 2
        if dd >= self.config.dd_tier1_threshold:
            return 1
        return 0

    def effective_max_pos(self, n_candidates: int) -> int:
        """If fewer candidates than thin_day_threshold, trade all."""
        if n_candidates < self.config.thin_day_threshold:
            return n_candidates
        return self.config.max_positions

    def reset_daily(self, date_str: str, n_candidates: int):
        """Reset daily state for a new trading day."""
        self.daily_stats = DailyStats(date=date_str, peak_equity=self.equity)
        self.trade_log = []
        self.stopped_today = {}
        self.positions = {}
        self.effective_max_positions = self.effective_max_pos(n_candidates)
        self.equity_curve.append(self.equity)


# ---------------------------------------------------------------------------
# simulate_gap_day — core single-day simulation logic
# ---------------------------------------------------------------------------

def simulate_gap_day(gap: dict, state: SimulationState, config: GapFadeConfig,
                     strategy=None, dd_scale: float = 1.0) -> Tuple[List[TradeRecord], List[dict]]:
    """Simulate one gap day. Returns (trades, log_entries).

    Identical logic to GapFadeBacktester._simulate_day_daily() but:
    - Synchronous (no async/await)
    - Uses SimulationState instead of GapFadeEngine
    - No WebSocket broadcasts
    """
    sym = gap['symbol']
    prev_close = gap['prev_close']
    day_open = gap['open']
    day_high = gap['high']
    day_low = gap['low']
    day_close = gap['close']
    day_trades: List[TradeRecord] = []
    log: List[dict] = []
    slip = config.slippage_pct
    direction = gap.get('direction', 'short')

    candidate = GapCandidate(
        symbol=sym, gap_pct=gap['gap_pct'], prev_close=prev_close,
        premarket_price=day_open, avg_vol_20d=gap.get('avg_vol', 0),
        vol_ratio=gap['vol_ratio'], shortable=True, easy_to_borrow=True,
        direction=direction,
    )

    # Bounce entry: if configured, wait for bounce above open before shorting
    bounce = config.bounce_entry_pct
    if direction == 'short' and bounce > 0 and day_high >= day_open * (1 + bounce):
        entry_price = day_open * (1 + bounce) * (1 - slip)
    elif direction == 'short' and bounce > 0:
        log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — no bounce to {bounce:.1%} above open'})
        return day_trades, log
    else:
        if direction == 'long':
            entry_price = day_open * (1 + slip)
        else:
            entry_price = day_open * (1 - slip)

    # Entry time estimates
    _ew = strategy.get_entry_window() if strategy else None
    _entry_min = _ew[1] if _ew else 31
    entry_time_str = f'{gap["date"]} 09:{_entry_min:02d}'
    exit_stop_str = f'{gap["date"]} 10:30'
    exit_partial_str = f'{gap["date"]} 12:00'
    exit_full_str = f'{gap["date"]} 14:00'
    exit_close_str = f'{gap["date"]} 15:55'
    _base_min = _entry_min
    hold_stop = 60 + (31 - _base_min)
    hold_partial = 150 + (31 - _base_min)
    hold_full = 270 + (31 - _base_min)
    hold_close = 385 + (31 - _base_min)

    # --- should_enter checks (replaces GapFadeEngine.should_enter) ---
    if candidate.catalyst in ('earnings', 'ma'):
        log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — catalyst-driven gap ({candidate.catalyst})'})
        return day_trades, log

    if sym in state.positions:
        log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — already in position'})
        return day_trades, log

    if sym in state.stopped_today:
        rec = state.stopped_today[sym]
        if rec.reentry_count >= config.reentry_max_per_symbol:
            log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — re-entry cap reached'})
            return day_trades, log

    eff_max = state.effective_max_positions
    if len(state.positions) >= eff_max:
        log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — max positions ({eff_max}) reached'})
        return day_trades, log

    vol_limit = config.gap_down_vol_ratio_max if candidate.direction == 'long' else config.vol_ratio_max
    if candidate.vol_ratio > vol_limit:
        log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — vol ratio {candidate.vol_ratio:.1f} > {vol_limit}'})
        return day_trades, log

    # --- Compute stop and targets ---
    eff_stop_pct = compute_adaptive_stop_pct(config, gap['gap_pct'])
    if direction == 'long':
        stop_price = entry_price * (1 - eff_stop_pct)
        half_target = (entry_price + prev_close) / 2
        full_target = prev_close
    else:
        stop_price = entry_price * (1 + eff_stop_pct)
        half_target = (entry_price + prev_close) / 2
        full_target = prev_close

    # Strategy plugin: override stop and targets
    if strategy:
        _est_hod = max(day_open, entry_price) * 1.01
        _bt_tick_data = {'day_high': _est_hod, 'vwap': 0, 'or_high': 0, 'or_low': 0, 'or_complete': False}
        strat_stop = strategy.compute_stop_price(entry_price, gap, _bt_tick_data)
        if strat_stop is not None:
            stop_price = strat_stop
        strat_targets = strategy.compute_targets(entry_price, gap, _bt_tick_data)
        if strat_targets is not None:
            half_target, full_target = strat_targets

    # --- Position sizing ---
    risk_per_share = abs(stop_price - entry_price)
    kelly_risk = state.compute_kelly_size()
    risk_frac = min(config.risk_pct, kelly_risk) if kelly_risk > 0 else config.risk_pct
    dollar_risk = state.equity * risk_frac
    shares = int(dollar_risk / risk_per_share) if risk_per_share > 0 else 0

    per_slot_equity = state.equity / max(1, eff_max)
    max_shares = int(per_slot_equity / entry_price) if entry_price > 0 else 0
    shares = min(shares, max_shares)

    if entry_price > 0 and config.max_notional > 0:
        notional_limit = min(config.max_notional, per_slot_equity)
        max_shares_notional = int(notional_limit / entry_price)
        shares = min(shares, max_shares_notional)

    avg_vol = gap.get('avg_vol', 0)
    if avg_vol > 0 and config.max_pct_adv > 0:
        max_shares_liq = int(avg_vol * config.max_pct_adv)
        if shares > max_shares_liq:
            shares = max_shares_liq

    # Drawdown circuit breaker: reduce position size (after all caps)
    if dd_scale < 1.0:
        shares = int(shares * dd_scale)

    if shares <= 0:
        log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: position size = 0'})
        return day_trades, log

    # Mark position as open
    state.positions[sym] = True

    # Borrow fee (shorts only)
    notional = shares * entry_price
    borrow_cost = notional * (config.borrow_rate_annual / 252) if direction == 'short' else 0

    stop_dist = abs(stop_price - entry_price) / entry_price
    target_dist = abs(entry_price - full_target) / entry_price
    side_label = 'LONG' if direction == 'long' else 'SHORT'
    cost_note = f' | borrow ${borrow_cost:.0f}' if borrow_cost > 0.5 else ''
    liq_note = f' | liq-capped' if avg_vol > 0 and shares == int(avg_vol * config.max_pct_adv) else ''
    log.append({
        'level': 'entry',
        'msg': (f'{sym} {gap["date"]}: {side_label} {shares}sh @ ${entry_price:.2f} (slip {slip:.2%}) '
                f'| gap {gap["gap_pct"]:.1%} vol {gap["vol_ratio"]:.2f}x '
                f'| stop ${stop_price:.2f} ({stop_dist:.1%}) '
                f'| half ${half_target:.2f} full ${full_target:.2f} ({target_dist:.1%})'
                f'{cost_note}{liq_note}'),
        'data': {'symbol': sym, 'shares': shares, 'entry': entry_price,
                 'stop': stop_price, 'gap_pct': gap['gap_pct'], 'vol_ratio': gap['vol_ratio'],
                 'direction': direction}
    })

    remaining = shares
    total_pnl = 0.0

    def _exit_slip(px):
        return px * (1 + slip) if direction == 'short' else px * (1 - slip)

    def _make_bt_trade(exit_px, reason, n_shares, exit_time, hold_min, extra_cost=0):
        fp = _exit_slip(exit_px)
        pnl = _direction_pnl(direction, entry_price, fp, n_shares) - extra_cost
        pnl_pct = pnl / (entry_price * n_shares) if entry_price > 0 and n_shares > 0 else 0
        return TradeRecord(
            symbol=sym, entry_price=entry_price, exit_price=fp,
            shares=n_shares, pnl=pnl, pnl_pct=pnl_pct,
            entry_time=entry_time_str, exit_time=exit_time,
            exit_reason=reason, holding_minutes=hold_min,
            side=direction,
        ), pnl

    # --- Exit logic (same order as original) ---
    stopped = _stop_hit(direction, day_high, day_low, stop_price)
    partial_hit = _target_hit(direction, day_low if direction == 'short' else day_high, half_target)
    full_hit = _target_hit(direction, day_close, full_target)

    if stopped and partial_hit and config.adverse_fill:
        # AMBIGUOUS BAR: both stop and target reachable
        if direction == 'short':
            bullish_close = day_close > entry_price
            bearish_close = day_close < day_open
        else:
            bullish_close = day_close > day_open
            bearish_close = day_close < entry_price

        if bearish_close if direction == 'short' else bullish_close:
            assume_stop = random.random() < (config.adverse_fill_pct * 0.5)
        elif bullish_close if direction == 'short' else bearish_close:
            assume_stop = random.random() < min(1.0, config.adverse_fill_pct * 1.5)
        else:
            assume_stop = random.random() < config.adverse_fill_pct

        if assume_stop:
            trade, pnl = _make_bt_trade(stop_price, 'stop_adverse', remaining, exit_stop_str, hold_stop, borrow_cost)
            day_trades.append(trade)
            state.record_trade(trade)
            total_pnl += pnl
            remaining = 0
            if config.reentry_enabled:
                state.stopped_today[sym] = StopOutRecord(
                    symbol=sym, stop_time=datetime.strptime(f'{gap["date"]} 09:35', '%Y-%m-%d %H:%M'),
                    original_entry=entry_price, prev_close=prev_close,
                    gap_pct=gap['gap_pct'], avg_vol_20d=gap.get('avg_vol', 0),
                    direction=direction,
                )
            log.append({
                'level': 'stop',
                'msg': (f'{sym} {gap["date"]}: STOP (adverse) {shares}sh @ ${trade.exit_price:.2f} '
                        f'| P&L ${pnl:.0f} ({trade.pnl_pct:+.1%}) '
                        f'| ambiguous bar — resolved as stop first'),
                'data': {'pnl': pnl, 'reason': 'stop_adverse'}
            })
        else:
            stopped = False

    elif stopped:
        trade, pnl = _make_bt_trade(stop_price, 'stop', remaining, exit_stop_str, hold_stop, borrow_cost)
        day_trades.append(trade)
        state.record_trade(trade)
        total_pnl += pnl
        remaining = 0
        if config.reentry_enabled:
            state.stopped_today[sym] = StopOutRecord(
                symbol=sym, stop_time=datetime.strptime(f'{gap["date"]} 09:35', '%Y-%m-%d %H:%M'),
                original_entry=entry_price, prev_close=prev_close,
                gap_pct=gap['gap_pct'], avg_vol_20d=gap.get('avg_vol', 0),
                direction=direction,
            )
        log.append({
            'level': 'stop',
            'msg': (f'{sym} {gap["date"]}: STOP {shares}sh @ ${trade.exit_price:.2f} '
                    f'| P&L ${pnl:.0f} ({trade.pnl_pct:+.1%}) | high ${day_high:.2f}'),
            'data': {'pnl': pnl, 'reason': 'stop'}
        })

    elif partial_hit:
        cover_shares = max(1, int(shares * config.partial_cover_frac))
        if cover_shares > 0:
            partial_borrow = borrow_cost * cover_shares / shares if shares > 0 else 0
            trade, pnl1 = _make_bt_trade(half_target, 'partial', cover_shares, exit_partial_str, hold_partial, partial_borrow)
            day_trades.append(trade)
            state.record_trade(trade)
            total_pnl += pnl1
            remaining -= cover_shares
            log.append({
                'level': 'exit',
                'msg': (f'{sym} {gap["date"]}: PARTIAL {cover_shares}sh @ ${trade.exit_price:.2f} '
                        f'| P&L ${pnl1:+.0f} ({trade.pnl_pct:+.1%})'),
                'data': {'pnl': pnl1, 'reason': 'partial'}
            })

        remaining_borrow = borrow_cost * remaining / shares if shares > 0 else 0
        if remaining > 0 and full_hit:
            trade, pnl2 = _make_bt_trade(full_target, 'full_target', remaining, exit_full_str, hold_full, remaining_borrow)
            day_trades.append(trade)
            state.record_trade(trade)
            total_pnl += pnl2
            remaining = 0
            log.append({
                'level': 'exit',
                'msg': (f'{sym} {gap["date"]}: FULL TARGET {trade.shares}sh @ ${trade.exit_price:.2f} '
                        f'| P&L ${pnl2:+.0f} ({trade.pnl_pct:+.1%})'),
                'data': {'pnl': pnl2, 'reason': 'full_target'}
            })
        elif remaining > 0:
            trade, pnl2 = _make_bt_trade(day_close, 'time_exit', remaining, exit_close_str, hold_close, remaining_borrow)
            day_trades.append(trade)
            state.record_trade(trade)
            total_pnl += pnl2
            remaining = 0
            log.append({
                'level': 'exit' if pnl2 >= 0 else 'stop',
                'msg': (f'{sym} {gap["date"]}: TIME EXIT {trade.shares}sh @ ${trade.exit_price:.2f} '
                        f'| P&L ${pnl2:+.0f} ({trade.pnl_pct:+.1%})'),
                'data': {'pnl': pnl2, 'reason': 'time_exit'}
            })

    else:
        trade, pnl = _make_bt_trade(day_close, 'time_exit', remaining, exit_close_str, hold_close, borrow_cost)
        day_trades.append(trade)
        state.record_trade(trade)
        total_pnl += pnl
        remaining = 0
        log.append({
            'level': 'exit' if pnl >= 0 else 'stop',
            'msg': (f'{sym} {gap["date"]}: EXIT @ CLOSE ${trade.exit_price:.2f} '
                    f'| P&L ${pnl:+.0f} ({trade.pnl_pct:+.1%}) | H ${day_high:.2f} L ${day_low:.2f}'),
            'data': {'pnl': pnl, 'reason': 'time_exit'}
        })

    # --- Re-entry after stop-out ---
    re_eligible = (day_close < entry_price) if direction == 'short' else (day_close > entry_price)
    if config.reentry_enabled and stopped and remaining == 0 and re_eligible:
        prev_rec = state.stopped_today.get(sym)
        reentry_count = prev_rec.reentry_count if prev_rec else 0
        if reentry_count < config.reentry_max_per_symbol:
            re_entry = (stop_price + day_close) / 2
            re_stop_pct = config.reentry_stop_pct
            if direction == 'long':
                re_stop = re_entry * (1 - re_stop_pct)
            else:
                re_stop = re_entry * (1 + re_stop_pct)

            re_favorable = (day_close < re_entry) if direction == 'short' else (day_close > re_entry)
            if re_favorable:
                # Check if re-entry stop was hit
                re_stopped = _stop_hit(direction, day_high, day_low, re_stop)
                if re_stopped:
                    re_exit = _exit_slip(re_stop)
                    re_exit_reason = 'reentry_stop'
                    re_hold = 120  # assume ~2hr hold if stopped
                    re_exit_time = f'{gap["date"]} 14:00'
                else:
                    re_exit = _exit_slip(day_close)
                    re_exit_reason = 'reentry_time'
                    re_hold = 240
                    re_exit_time = exit_close_str
                re_risk_per_share = abs(re_stop - re_entry)
                re_kelly = state.compute_kelly_size()
                re_risk_frac = min(config.risk_pct, re_kelly) if re_kelly > 0 else config.risk_pct
                re_risk_budget = state.equity * re_risk_frac
                re_shares = int(re_risk_budget / re_risk_per_share) if re_risk_per_share > 0 else 0

                re_eff_max = state.effective_max_positions
                re_per_slot = state.equity / max(1, re_eff_max)
                re_max_shares = int(re_per_slot / re_entry) if re_entry > 0 else 0
                re_shares = min(re_shares, re_max_shares)
                if re_entry > 0 and config.max_notional > 0:
                    re_notional_limit = min(config.max_notional, re_per_slot)
                    re_shares = min(re_shares, int(re_notional_limit / re_entry))

                if re_shares > 0:
                    re_pnl = _direction_pnl(direction, re_entry, re_exit, re_shares)
                    re_borrow = re_shares * re_entry * (config.borrow_rate_annual / 252) if direction == 'short' else 0
                    re_pnl -= re_borrow
                    re_pnl_pct = re_pnl / (re_entry * re_shares) if re_entry > 0 else 0
                    re_trade = TradeRecord(
                        symbol=sym, entry_price=re_entry, exit_price=re_exit,
                        shares=re_shares, pnl=re_pnl, pnl_pct=re_pnl_pct,
                        entry_time=f'{gap["date"]} 12:00', exit_time=re_exit_time,
                        exit_reason=re_exit_reason, holding_minutes=re_hold,
                        side=direction,
                    )
                    day_trades.append(re_trade)
                    state.record_trade(re_trade)
                    total_pnl += re_pnl
                    state.stopped_today[sym] = StopOutRecord(
                        symbol=sym, stop_time=datetime.strptime(f'{gap["date"]} 10:30', '%Y-%m-%d %H:%M'),
                        original_entry=entry_price, prev_close=prev_close,
                        gap_pct=gap['gap_pct'], avg_vol_20d=gap.get('avg_vol', 0),
                        reentry_count=reentry_count + 1,
                        direction=direction,
                    )
                    log.append({
                        'level': 'entry',
                        'msg': (f'{sym} {gap["date"]}: RE-ENTRY {re_shares}sh @ ${re_entry:.2f} '
                                f'| stop ${re_stop:.2f} (+{re_stop_pct:.1%}) '
                                f'| exit @ close ${re_exit:.2f} '
                                f'| P&L ${re_pnl:+.0f} ({re_pnl_pct:+.1%})'),
                        'data': {'pnl': re_pnl, 'reason': 'reentry'}
                    })

    # Remove from positions
    state.positions.pop(sym, None)

    pnl_sign = '+' if total_pnl >= 0 else ''
    log.append({
        'level': 'summary',
        'msg': (f'{sym} {gap["date"]}: {len(day_trades)} fills, '
                f'net {pnl_sign}${total_pnl:.0f} | equity ${state.equity:,.0f}'),
        'data': {'total_pnl': total_pnl, 'equity': state.equity}
    })

    return day_trades, log


# ---------------------------------------------------------------------------
# MetricsAdapter — converts simulation results to dashboard-compatible dict
# ---------------------------------------------------------------------------

class MetricsAdapter:
    """Converts SimulationState results to the exact dict format the dashboard expects."""

    @staticmethod
    def build_result(state: SimulationState, config: GapFadeConfig,
                     meta: dict) -> dict:
        """Build the full result dict matching GapFadeBacktester.run() output.

        meta: dict with keys like raw_gaps, gap_days_found, bars_missing,
              gap_days_traded, regime_skipped, regime_halved, symbols_scanned,
              start_date, end_date, strategy_id, strategy_name, bt_log,
              daily_summary, warnings.
        """
        trades = state.all_trade_log
        if not trades:
            result = {'total_trades': 0}
            result.update(meta)
            return result

        pnls = [t.pnl for t in trades]
        pnl_pcts = [t.pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = float(np.mean(wins)) if wins else 0
        avg_loss = float(np.mean(losses)) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 9999.99

        # Max drawdown from equity curve (not cumsum of trade P&L)
        eq = state.equity_curve
        if len(eq) > 1:
            eq_arr = np.array(eq)
            peak_eq = np.maximum.accumulate(eq_arr)
            dd_arr = (peak_eq - eq_arr) / np.where(peak_eq > 0, peak_eq, 1)
            max_dd_pct = float(np.max(dd_arr))
            max_dd = max_dd_pct * config.initial_capital  # approximate $ DD for backward compat
        else:
            max_dd = 0
            max_dd_pct = 0

        # Sharpe from daily equity curve returns (not per-trade returns)
        eq = state.equity_curve
        if len(eq) > 2:
            eq_arr = np.array(eq)
            daily_rets = np.diff(eq_arr) / eq_arr[:-1]
            daily_rets = daily_rets[np.isfinite(daily_rets)]
            if len(daily_rets) > 1 and np.std(daily_rets) > 0:
                sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252))
            else:
                sharpe = 0
        else:
            sharpe = 0

        avg_holding = float(np.mean([t.holding_minutes for t in trades])) if trades else 0

        result = {
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(win_rate, 4),
            'total_pnl': round(total_pnl, 2),
            'return_pct': round(total_pnl / config.initial_capital * 100, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown': round(max_dd, 2),
            'max_drawdown_pct': round(max_dd_pct * 100, 2),
            'sharpe': round(sharpe, 2),
            'avg_holding_min': round(avg_holding, 1),
            'final_equity': round(state.equity, 2),
        }

        # Funnel and metadata
        result['raw_gaps'] = meta.get('raw_gaps', 0)
        result['gap_days_found'] = meta.get('gap_days_found', 0)
        result['bars_missing'] = meta.get('bars_missing', 0)
        result['gap_days_traded'] = meta.get('gap_days_traded', 0)
        result['regime_skipped'] = meta.get('regime_skipped', 0)
        result['regime_halved'] = meta.get('regime_halved', 0)
        result['symbols_scanned'] = meta.get('symbols_scanned', 0)
        result['start_date'] = meta.get('start_date', '')
        result['end_date'] = meta.get('end_date', '')
        result['config'] = asdict(config)
        result['trades'] = [asdict(t) for t in trades[-200:]]
        result['all_trades'] = [asdict(t) for t in trades]
        result['daily_summary'] = meta.get('daily_summary', [])
        result['bt_log'] = meta.get('bt_log', [])[-500:]

        result['realism'] = {
            'slippage_pct': config.slippage_pct,
            'borrow_rate_annual': config.borrow_rate_annual,
            'max_pct_adv': config.max_pct_adv,
            'adverse_fill': config.adverse_fill,
            'adverse_fill_pct': config.adverse_fill_pct,
        }

        if config.dd_circuit_breaker:
            result['circuit_breaker'] = {
                'enabled': True,
                'tier1_threshold': config.dd_tier1_threshold,
                'tier1_scale': config.dd_tier1_scale,
                'tier2_threshold': config.dd_tier2_threshold,
                'tier2_scale': config.dd_tier2_scale,
                'tier2_max_positions': config.dd_tier2_max_positions,
                'hard_stop': config.dd_hard_stop,
                'days_in_tier1': state.dd_days_tier1,
                'days_in_tier2': state.dd_days_tier2,
                'trades_skipped_hard_stop': state.dd_trades_skipped,
                'hard_stopped': state.dd_hard_stopped,
                'hard_stop_date': state.dd_hard_stop_date,
                'final_drawdown': round(state.current_drawdown() * 100, 1),
            }

        result['strategy'] = {
            'id': meta.get('strategy_id', 'classic_gap_fade'),
            'name': meta.get('strategy_name', 'Classic Gap Fade (engine defaults)'),
        }

        result['warnings'] = meta.get('warnings', [])

        return result

    @staticmethod
    def add_vbt_metrics(result: dict, state: SimulationState, config: GapFadeConfig):
        """If vectorbt available, add extra analytics: sortino, calmar, omega, VaR, CVaR."""
        if not HAS_VBT:
            return

        trades = state.all_trade_log
        if not trades:
            return

        try:
            # Build returns series from equity curve
            eq = [config.initial_capital] + state.equity_curve
            if len(eq) < 3:
                return

            eq_series = pd.Series(eq, dtype=float)
            returns = eq_series.pct_change().dropna()
            if returns.empty:
                return

            pf = vbt.Portfolio.from_returns(returns, init_cash=config.initial_capital)
            stats = pf.stats()

            result['vbt_metrics'] = {
                'sortino': round(float(stats.get('Sortino Ratio', 0)), 2),
                'calmar': round(float(stats.get('Calmar Ratio', 0)), 2),
                'omega': round(float(stats.get('Omega Ratio', 0)), 2),
            }

            # VaR and CVaR from returns
            returns_arr = returns.values
            if len(returns_arr) > 10:
                sorted_ret = np.sort(returns_arr)
                idx_5 = int(len(sorted_ret) * 0.05)
                var_5 = float(sorted_ret[max(0, idx_5)])
                cvar_5 = float(np.mean(sorted_ret[:max(1, idx_5 + 1)]))
                result['vbt_metrics']['var_5pct'] = round(var_5 * 100, 2)
                result['vbt_metrics']['cvar_5pct'] = round(cvar_5 * 100, 2)
        except Exception as e:
            logger.warning(f"vbt metrics failed: {e}")


# ---------------------------------------------------------------------------
# VbtGapFadeBacktester — drop-in replacement
# ---------------------------------------------------------------------------

class VbtGapFadeBacktester:
    """Vectorbt-powered backtester. Same interface as GapFadeBacktester."""

    def __init__(self, config: GapFadeConfig = None, strategy_id: str = ''):
        self.config = config or GapFadeConfig()
        self.progress = 0.0
        self.status = 'idle'
        self._cancel = False
        self.result = None
        self.strategy = None
        self.strategy_id = strategy_id
        if strategy_id:
            self._load_strategy(strategy_id)

    def _load_strategy(self, strategy_id: str, strategy_config: dict = None):
        """Load a strategy for backtest use."""
        try:
            from gap_fade_strategies import GapFadeStrategyRegistry
            if GapFadeStrategyRegistry.has_strategy(strategy_id):
                self.strategy = GapFadeStrategyRegistry.create_strategy(strategy_id, strategy_config)
                self.strategy_id = strategy_id
                logger.info(f"Vbt backtest strategy: {self.strategy.name} ({strategy_id})")
            else:
                logger.warning(f"Vbt backtest strategy '{strategy_id}' not found, using engine defaults")
                self.strategy = None
                self.strategy_id = ''
        except Exception as e:
            logger.warning(f"Failed to load vbt backtest strategy '{strategy_id}': {e}")
            self.strategy = None
            self.strategy_id = ''

    async def run(self, symbol: str = None, symbols: List[str] = None,
                  start_date: str = None, end_date: str = None,
                  config: GapFadeConfig = None,
                  progress_callback=None, **kwargs) -> dict:
        """Run backtest with bulk data loading and vectorized gap scanning.

        Same signature and result dict as GapFadeBacktester.run().
        """
        if config:
            self.config = config

        strategy_id = kwargs.get('strategy_id', self.strategy_id or '')
        strategy_config = kwargs.get('strategy_config')
        if strategy_id:
            self._load_strategy(strategy_id, strategy_config)

        self.status = 'running'
        self._cancel = False
        self.progress = 0
        bt_log: List[dict] = []

        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=self.config.backtest_years * 365)).strftime('%Y-%m-%d')

        syms = symbols or ([symbol] if symbol else UNIVERSE)
        logger.info(f"VBT BACKTEST START: strategy={strategy_id or 'engine'}, "
                     f"symbols={len(syms)}, range={start_date} to {end_date}")

        async def _progress(pct, msg):
            self.progress = pct
            if progress_callback:
                await progress_callback(pct, msg)

        # Broadcast helper (import here to avoid circular import at module level)
        try:
            from gap_fade_app import broadcast
        except ImportError:
            async def broadcast(msg):
                pass

        async def _log(level: str, msg: str, data: dict = None):
            entry = {'level': level, 'msg': msg, 'data': data or {}}
            bt_log.append(entry)
            await broadcast({'type': 'bt_log', 'entry': entry})

        # Seed random for reproducible adverse fill resolution
        random.seed(42)

        # ---- Phase 1 (0-15%): Bulk data load ----
        await _log('info', f'Loading bulk data for {len(syms)} symbols...')
        await _progress(2, f"Loading bulk data for {len(syms)} symbols...")

        loader = DataLoader()
        t0 = _time.time()
        all_data = await asyncio.to_thread(loader.load_universe, syms, start_date, end_date)
        elapsed_load = _time.time() - t0

        if all_data.empty:
            self.status = 'done'
            no_gaps = {'error': 'No data found in DB', 'total_trades': 0, 'bt_log': bt_log}
            self.result = no_gaps
            await broadcast({'type': 'backtest_complete', 'result': no_gaps})
            return no_gaps

        n_syms_loaded = len(all_data.columns.get_level_values(1).unique())
        await _log('info', f'Loaded {n_syms_loaded} symbols in {elapsed_load:.1f}s '
                   f'({all_data.shape[0]} dates × {all_data.shape[1]} columns)')
        await _progress(15, f"Data loaded. Scanning for gaps...")

        # ---- Phase 2 (15-25%): Vectorized gap scan ----
        scanner = VectorizedGapScanner(self.config)
        t1 = _time.time()
        gaps_df = await asyncio.to_thread(scanner.scan, all_data, start_date, end_date)
        elapsed_scan = _time.time() - t1

        raw_gap_count = len(gaps_df)
        await _log('info', f'Vectorized scan found {raw_gap_count} raw gaps in {elapsed_scan:.2f}s')

        # Vol filter
        gaps_df = scanner.apply_vol_filter(gaps_df)
        await _log('scan', f'After vol filter: {len(gaps_df)} gaps (from {raw_gap_count} raw)')
        await _progress(25, f"Found {len(gaps_df)} gaps. Filtering...")

        if gaps_df.empty:
            self.status = 'done'
            no_gaps = {'error': 'No qualifying gap days found', 'total_trades': 0, 'bt_log': bt_log}
            self.result = no_gaps
            await broadcast({'type': 'backtest_complete', 'result': no_gaps})
            return no_gaps

        # ---- Phase 3 (25-35%): Strategy filter with BatchDataProvider ----
        all_gap_days = gaps_df.to_dict('records')
        batch_provider = BatchDataProvider(all_data)

        if self.strategy:
            await _log('info', f'Applying {self.strategy.name} strategy filter...')
            # Monkey-patch PriceDB.get_bars temporarily
            db = get_price_db()
            original_get_bars = db.get_bars
            db.get_bars = batch_provider.get_bars
            try:
                filtered_gaps = []
                for i, g in enumerate(all_gap_days):
                    if self._cancel:
                        break
                    try:
                        ok, reason = self.strategy.filter_candidate(g)
                    except Exception as e:
                        logger.error(f"Strategy filter error for {g.get('symbol', '?')} {g.get('date', '?')}: {e}")
                        ok = True
                    if ok:
                        try:
                            new_score = self.strategy.score_candidate(g)
                        except Exception as e:
                            logger.error(f"Strategy score error for {g.get('symbol', '?')}: {e}")
                            new_score = None
                        if new_score is not None:
                            g['score'] = new_score
                        filtered_gaps.append(g)
                    if i % 200 == 199:
                        pct = 25 + (i / len(all_gap_days)) * 10
                        await _progress(pct, f"Strategy filter: {i+1}/{len(all_gap_days)}")
                        await asyncio.sleep(0)
                filtered_out = len(all_gap_days) - len(filtered_gaps)
                await _log('strategy', f'{self.strategy.name}: filtered {filtered_out} candidates '
                           f'({len(filtered_gaps)} remain)')
                all_gap_days = filtered_gaps
            finally:
                db.get_bars = original_get_bars
        await _progress(35, f"Strategy filter done. Building schedule...")

        # ---- Phase 4 (35-40%): Build gap schedule + regime filter ----
        gaps_by_date: Dict[str, List[dict]] = defaultdict(list)
        for gap in all_gap_days:
            gaps_by_date[gap['date']].append(gap)

        for d in gaps_by_date:
            gaps_by_date[d].sort(key=lambda g: abs(g['gap_pct']), reverse=True)

        # Pre-cache SPY data for market regime filter
        spy_data: Dict[str, dict] = {}
        if self.config.regime_filter:
            try:
                spy_df = loader.load_spy(start_date, end_date)
                if not spy_df.empty:
                    spy_close = spy_df['close']
                    spy_open = spy_df['open']
                    spy_prev = spy_close.shift(1)
                    for dt in spy_df.index:
                        date_str = dt.strftime('%Y-%m-%d')
                        pc = spy_prev.get(dt)
                        if pc and not np.isnan(pc) and pc > 0:
                            spy_data[date_str] = {'prev_close': float(pc), 'open': float(spy_open.loc[dt])}
                    await _log('info', f'Market regime filter: loaded {len(spy_data)} SPY trading days')
            except Exception as e:
                await _log('warn', f'Market regime filter: failed — {e}')

        await _progress(40, f"Simulating {len(all_gap_days)} gaps...")

        # ---- Phase 5 (40-95%): Simulation loop ----
        state = SimulationState(config=self.config)
        sorted_dates = sorted(gaps_by_date.keys())
        total_gaps = len(all_gap_days)
        trades_by_day = []
        gi = 0
        regime_skipped = 0
        regime_halved = 0

        def _simulate_all():
            nonlocal gi, regime_skipped, regime_halved
            for date_str in sorted_dates:
                if self._cancel:
                    break

                day_gaps = gaps_by_date[date_str]
                day_gap_count = len(day_gaps)

                # --- Drawdown circuit breaker ---
                tier = state.dd_tier()
                if tier == 3:  # Hard stop
                    state.dd_trades_skipped += day_gap_count
                    if not state.dd_hard_stop_date:
                        state.dd_hard_stop_date = date_str
                    bt_log.append({'level': 'halt', 'msg': f'{date_str}: HARD STOP — DD {state.current_drawdown():.1%}'})
                    gi += day_gap_count
                    state.equity_curve.append(state.equity)
                    continue

                # Thin day logic
                eff_max = state.effective_max_pos(len(day_gaps))
                state.effective_max_positions = eff_max

                # Market regime filter
                regime = fetch_market_regime_backtest(date_str, spy_data, self.config)
                if regime and regime.position_reduction == -999:
                    regime_skipped += 1
                    gi += day_gap_count
                    state.equity_curve.append(state.equity)
                    continue
                if regime and regime.position_reduction == -1:
                    regime_halved += 1
                    eff_max = max(1, eff_max // 2)
                    state.effective_max_positions = eff_max

                # --- Apply DD tier caps ---
                dd_scale = 1.0
                if tier == 2:
                    state.dd_days_tier2 += 1
                    eff_max = min(eff_max, self.config.dd_tier2_max_positions)
                    state.effective_max_positions = eff_max
                    dd_scale = self.config.dd_tier2_scale
                elif tier == 1:
                    state.dd_days_tier1 += 1
                    dd_scale = self.config.dd_tier1_scale

                simulated_gaps = day_gaps[:eff_max]
                skipped_count = day_gap_count - len(simulated_gaps)

                state.reset_daily(date_str, len(day_gaps))

                for gap in simulated_gaps:
                    if self._cancel:
                        break
                    gi += 1
                    day_trades_result, day_log = simulate_gap_day(gap, state, self.config, self.strategy, dd_scale=dd_scale)
                    for entry in day_log:
                        bt_log.append(entry)
                    trades_by_day.append({
                        'date': gap['date'],
                        'symbol': gap['symbol'],
                        'gap_pct': gap['gap_pct'],
                        'vol_ratio': gap['vol_ratio'],
                        'trades': len(day_trades_result),
                        'pnl': sum(t.pnl for t in day_trades_result),
                    })

                gi += skipped_count

        # Run simulation in thread to not block event loop
        await asyncio.to_thread(_simulate_all)

        # Update progress during simulation is hard from thread, so just set final
        await _progress(95, "Computing metrics...")
        logger.info("[VBT-DEBUG] Phase 6: simulation done, building metrics...")

        # ---- Phase 6 (95-100%): Metrics + vbt Portfolio construction ----
        # Final equity curve entry
        state.equity_curve.append(state.equity)

        # Build warnings
        warnings = []
        if len(syms) > 50:
            warnings.append(
                'Survivorship bias: backtest uses currently-listed symbols only. '
                'Delisted/bankrupt stocks that gapped up and never recovered are excluded, '
                'which may overstate win rate.')
        if self.config.slippage_pct == 0:
            warnings.append('No slippage applied — results assume perfect fills at exact prices.')
        if self.config.borrow_rate_annual == 0:
            warnings.append('No short borrow fees — real borrow costs can be 2-100%+ annualized.')

        meta = {
            'raw_gaps': raw_gap_count,
            'gap_days_found': len(all_gap_days),
            'bars_missing': 0,
            'gap_days_traded': len([d for d in trades_by_day if d['trades'] > 0]),
            'regime_skipped': regime_skipped,
            'regime_halved': regime_halved,
            'symbols_scanned': len(syms),
            'start_date': start_date,
            'end_date': end_date,
            'strategy_id': self.strategy_id or 'classic_gap_fade',
            'strategy_name': self.strategy.name if self.strategy else 'Classic Gap Fade (engine defaults)',
            'bt_log': bt_log,
            'daily_summary': trades_by_day,
            'warnings': warnings,
        }

        logger.info("[VBT-DEBUG] Building result dict...")
        result = MetricsAdapter.build_result(state, self.config, meta)
        MetricsAdapter.add_vbt_metrics(result, state, self.config)
        logger.info(f"[VBT-DEBUG] Result built: {result.get('total_trades', 0)} trades")

        # Sanitize numpy types + non-JSON-safe floats for ws.send_json()
        def _sanitize(obj):
            # Convert numpy scalars to Python native types first
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                val = float(obj)
                if math.isinf(val):
                    return 9999.99 if val > 0 else -9999.99
                if math.isnan(val):
                    return 0.0
                return val
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, float):
                if math.isinf(obj):
                    return 9999.99 if obj > 0 else -9999.99
                if math.isnan(obj):
                    return 0.0
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return [_sanitize(v) for v in obj.tolist()]
            return obj

        result = _sanitize(result)
        logger.info("[VBT-DEBUG] Result sanitized, verifying JSON...")

        # Verify JSON serialization before broadcasting
        import json as _json
        try:
            _json_payload = _json.dumps(result)
            logger.info(f"[VBT-DEBUG] JSON OK: {len(_json_payload):,} bytes")
        except Exception as e:
            logger.error(f"[VBT-DEBUG] JSON serialization FAILED: {e}")
            # Find the problematic value
            def _find_bad(obj, path=''):
                try:
                    _json.dumps(obj)
                except (TypeError, ValueError):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            _find_bad(v, f'{path}.{k}')
                    elif isinstance(obj, (list, tuple)):
                        for i, v in enumerate(obj):
                            _find_bad(v, f'{path}[{i}]')
                    else:
                        logger.error(f"[VBT-DEBUG] Bad value at {path}: {type(obj).__name__} = {repr(obj)[:100]}")
            _find_bad(result)

        self.progress = 100
        self.status = 'done'
        self.result = result

        elapsed_total = _time.time() - t0
        logger.info(f"VBT BACKTEST DONE: {result.get('total_trades', 0)} trades in {elapsed_total:.1f}s "
                     f"(load={elapsed_load:.1f}s, scan={elapsed_scan:.2f}s)")

        try:
            await _log('info', f'Backtest complete: {result.get("total_trades", 0)} trades, '
                       f'{result.get("win_rate", 0):.1%} win rate, '
                       f'${result.get("total_pnl", 0):,.0f} P&L ({result.get("return_pct", 0):.1f}%)')
        except Exception as e:
            logger.error(f"[VBT-DEBUG] _log failed: {e}")

        try:
            logger.info("[VBT-DEBUG] Broadcasting backtest_complete...")
            await broadcast({'type': 'backtest_complete', 'result': result})
            logger.info("[VBT-DEBUG] Broadcast complete!")
        except Exception as e:
            logger.error(f"[VBT-DEBUG] broadcast failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return result

    def cancel(self):
        """Cancel running backtest."""
        self._cancel = True
        self.status = 'cancelled'
