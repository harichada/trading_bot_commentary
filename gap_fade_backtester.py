"""
Vectorbt-powered backtester module for Rudra Trading Engine.

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
import os
import random
import psycopg2
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
    LEVERAGED_ETFS,
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

    def __init__(self, db_url: str = None):
        self._db_url = db_url or os.environ.get('DATABASE_URL', PriceDB.DEFAULT_DB_URL)

    def load_universe(self, symbols: List[str], start: str, end: str,
                      warmup_days: int = 252) -> pd.DataFrame:
        """Load daily bars for all symbols in one bulk SQL query.

        Returns a wide DataFrame with DatetimeIndex and MultiIndex columns
        (field, symbol) where field ∈ {open, high, low, close, volume}.
        """
        adj_start = (datetime.strptime(start, '%Y-%m-%d')
                     - timedelta(days=int(warmup_days * 1.5))).strftime('%Y-%m-%d')

        conn = psycopg2.connect(self._db_url)
        try:
            cur = conn.cursor()
            if len(symbols) <= 2000:
                rows = []
                chunk_size = 500
                for i in range(0, len(symbols), chunk_size):
                    chunk = symbols[i:i + chunk_size]
                    placeholders = ','.join(['%s'] * len(chunk))
                    sql = (
                        f'SELECT symbol, date, open, high, low, close, volume '
                        f'FROM daily_bars WHERE symbol IN ({placeholders}) '
                        f'AND date >= %s AND date <= %s ORDER BY symbol, date'
                    )
                    cur.execute(sql, chunk + [adj_start, end])
                    rows.extend(cur.fetchall())
            else:
                sql = (
                    'SELECT symbol, date, open, high, low, close, volume '
                    'FROM daily_bars WHERE date >= %s AND date <= %s '
                    'ORDER BY symbol, date'
                )
                cur.execute(sql, (adj_start, end))
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
        conn = psycopg2.connect(self._db_url)
        try:
            adj_start = (datetime.strptime(start, '%Y-%m-%d') - timedelta(days=10)).strftime('%Y-%m-%d')
            cur = conn.cursor()
            cur.execute(
                'SELECT date, open, high, low, close, volume FROM daily_bars '
                'WHERE symbol = %s AND date >= %s AND date <= %s ORDER BY date',
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

    # --- ORB confirmation: simulate opening range breakout ---
    if config.orb_enabled:
        # Estimate OR range: use gap_pct-proportional range, clamped
        or_range_pct = abs(gap['gap_pct']) * config.orb_atr_fraction
        or_range_pct = max(config.orb_min_range_pct, min(config.orb_max_range_pct, or_range_pct))

        if direction == 'short':
            # For gap-up short: entry triggers on break BELOW opening range low
            or_low = day_open * (1 - or_range_pct)
            or_high = day_open * (1 + or_range_pct)
            if day_low > or_low:
                # Price never broke below OR floor — gap held, skip
                log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — no OR breakdown (low ${day_low:.2f} > OR_low ${or_low:.2f})'})
                return day_trades, log
            # Enter at breakdown price (honest: this is where a live ORB order fills)
            entry_price = or_low * (1 - slip)
        else:
            # For gap-down long: entry triggers on break ABOVE opening range high
            or_high = day_open * (1 + or_range_pct)
            or_low = day_open * (1 - or_range_pct)
            if day_high < or_high:
                # Price never broke above OR ceiling — gap held, skip
                log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — no OR breakout (high ${day_high:.2f} < OR_high ${or_high:.2f})'})
                return day_trades, log
            # Enter at breakout price (honest: this is where a live ORB order fills)
            entry_price = or_high * (1 + slip)
    else:
        # Legacy: blind entry at open
        or_low = or_high = None
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

    # Entry time estimates — ORB entry is delayed (~10:00 after OR forms)
    _ew = strategy.get_entry_window() if strategy else None
    if config.orb_enabled:
        _entry_min = 0  # 10:00 AM (after 30-min OR)
        entry_time_str = f'{gap["date"]} 10:00'
    else:
        _entry_min = _ew[1] if _ew else 31
        entry_time_str = f'{gap["date"]} 09:{_entry_min:02d}'
    exit_stop_str = f'{gap["date"]} 10:30'
    exit_partial_str = f'{gap["date"]} 12:00'
    exit_full_str = f'{gap["date"]} 14:00'
    exit_close_str = f'{gap["date"]} 15:55'
    if config.orb_enabled:
        hold_stop = 30       # 10:00 → 10:30
        hold_partial = 120   # 10:00 → 12:00
        hold_full = 240      # 10:00 → 14:00
        hold_close = 355     # 10:00 → 15:55
    else:
        _base_min = _entry_min
        hold_stop = 60 + (31 - _base_min)
        hold_partial = 150 + (31 - _base_min)
        hold_full = 270 + (31 - _base_min)
        hold_close = 385 + (31 - _base_min)

    # --- should_enter checks (replaces GapFadeEngine.should_enter) ---
    if config.exclude_leveraged and sym in LEVERAGED_ETFS:
        log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — leveraged/inverse ETF'})
        return day_trades, log

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
    if config.orb_enabled and config.orb_dynamic_stop and or_high is not None:
        # Dynamic stop at opposite side of opening range
        if direction == 'short':
            stop_price = or_high * (1 + slip)  # stop above OR high
        else:
            stop_price = or_low * (1 - slip)   # stop below OR low
    else:
        eff_stop_pct = compute_adaptive_stop_pct(config, gap['gap_pct'])
        if direction == 'long':
            stop_price = entry_price * (1 - eff_stop_pct)
        else:
            stop_price = entry_price * (1 + eff_stop_pct)

    if direction == 'long':
        half_target = (entry_price + prev_close) / 2
        full_target = prev_close
    else:
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
                # Exit at close (matches original _simulate_day_daily)
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
                    re_pnl_pct = re_pnl / (re_entry * re_shares) if re_entry > 0 else 0
                    re_borrow = re_shares * re_entry * (config.borrow_rate_annual / 252) if direction == 'short' else 0
                    re_pnl -= re_borrow
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
# simulate_gap_day_orb_1min — ORB simulation with real 1-minute bars
# ---------------------------------------------------------------------------

def simulate_gap_day_orb_1min(gap: dict, min_bars: pd.DataFrame,
                               state: SimulationState, config: GapFadeConfig,
                               dd_scale: float = 1.0) -> Tuple[List[TradeRecord], List[dict]]:
    """Simulate one gap day with real ORB confirmation from 1-min bars.

    Uses actual 1-min OHLCV to:
    1. Build opening range from first N minutes (orb_atr_period used as OR minutes)
    2. Wait for price to break through OR level
    3. Enter at breakdown price with stop at opposite OR extreme
    4. Walk forward through remaining bars for exits
    """
    sym = gap['symbol']
    prev_close = gap['prev_close']
    direction = gap.get('direction', 'short')
    day_trades: List[TradeRecord] = []
    log: List[dict] = []
    slip = config.slippage_pct

    # --- Pre-entry checks ---
    candidate = GapCandidate(
        symbol=sym, gap_pct=gap['gap_pct'], prev_close=prev_close,
        premarket_price=gap['open'], avg_vol_20d=gap.get('avg_vol', 0),
        vol_ratio=gap['vol_ratio'], shortable=True, easy_to_borrow=True,
        direction=direction,
    )

    if config.exclude_leveraged and sym in LEVERAGED_ETFS:
        log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — leveraged/inverse ETF'})
        return day_trades, log

    if candidate.catalyst in ('earnings', 'ma'):
        log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — catalyst-driven gap ({candidate.catalyst})'})
        return day_trades, log

    if sym in state.positions:
        return day_trades, log

    if sym in state.stopped_today:
        rec = state.stopped_today[sym]
        if rec.reentry_count >= config.reentry_max_per_symbol:
            return day_trades, log

    eff_max = state.effective_max_positions
    if len(state.positions) >= eff_max:
        return day_trades, log

    vol_limit = config.gap_down_vol_ratio_max if direction == 'long' else config.vol_ratio_max
    if candidate.vol_ratio > vol_limit:
        return day_trades, log

    # --- Parse 1-min bars ---
    if min_bars is None or len(min_bars) < 10:
        log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — insufficient 1-min data ({len(min_bars) if min_bars is not None else 0} bars)'})
        return day_trades, log

    opens = min_bars['open'].values.astype(float)
    highs = min_bars['high'].values.astype(float)
    lows = min_bars['low'].values.astype(float)
    closes = min_bars['close'].values.astype(float)
    times = min_bars.index

    # Convert times to hours/minutes for comparison
    def _hm(t):
        if hasattr(t, 'hour'):
            return t.hour, t.minute
        t = t.to_pydatetime()
        return t.hour, t.minute

    # --- Build Opening Range from first 30 min (9:30-10:00) ---
    or_minutes = 30  # fixed 30-min opening range
    or_end_h, or_end_m = 10, 0  # OR complete at 10:00

    or_high = -1e9
    or_low = 1e9
    or_end_bar = -1

    for bi in range(len(times)):
        h, m = _hm(times[bi])
        if h < 9 or (h == 9 and m < 30):
            continue  # pre-market bars
        if (h, m) >= (or_end_h, or_end_m):
            or_end_bar = bi
            break
        or_high = max(or_high, highs[bi])
        or_low = min(or_low, lows[bi])

    if or_end_bar < 0 or or_high <= 0 or or_low >= 1e9:
        log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — could not build opening range'})
        return day_trades, log

    or_range_pct = (or_high - or_low) / gap['open'] if gap['open'] > 0 else 0
    log.append({'level': 'info', 'msg': f'{sym} {gap["date"]}: OR formed — high ${or_high:.2f} low ${or_low:.2f} range {or_range_pct:.2%}'})

    # --- Scan for ORB entry after OR completes ---
    entry_bar = -1
    entry_price = 0.0

    for bi in range(or_end_bar, len(times)):
        h, m = _hm(times[bi])
        # Entry cutoff
        if h > config.entry_cutoff_hour or (h == config.entry_cutoff_hour and m >= config.entry_cutoff_min):
            break

        if direction == 'short':
            # Short entry: price breaks below OR low
            if lows[bi] <= or_low:
                entry_bar = bi
                entry_price = or_low * (1 - slip)  # fill at OR low with slippage
                break
        else:
            # Long entry: price breaks above OR high
            if highs[bi] >= or_high:
                entry_bar = bi
                entry_price = or_high * (1 + slip)
                break

    if entry_bar < 0:
        log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: SKIP — no OR breakdown by {config.entry_cutoff_hour}:{config.entry_cutoff_min:02d}'})
        return day_trades, log

    entry_time = times[entry_bar]
    if hasattr(entry_time, 'to_pydatetime'):
        entry_time = entry_time.to_pydatetime()
    entry_time_str = entry_time.strftime('%Y-%m-%d %H:%M')

    # --- Compute stop and targets ---
    if config.orb_dynamic_stop:
        # Dynamic stop at opposite side of opening range
        if direction == 'short':
            stop_price = or_high * (1 + slip)
        else:
            stop_price = or_low * (1 - slip)
    else:
        eff_stop_pct = compute_adaptive_stop_pct(config, gap['gap_pct'])
        if direction == 'short':
            stop_price = entry_price * (1 + eff_stop_pct)
        else:
            stop_price = entry_price * (1 - eff_stop_pct)

    if direction == 'long':
        half_target = (entry_price + prev_close) / 2
        full_target = prev_close
    else:
        half_target = (entry_price + prev_close) / 2
        full_target = prev_close

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
        shares = min(shares, int(notional_limit / entry_price))

    avg_vol = gap.get('avg_vol', 0)
    if avg_vol > 0 and config.max_pct_adv > 0:
        shares = min(shares, int(avg_vol * config.max_pct_adv))

    if dd_scale < 1.0:
        shares = int(shares * dd_scale)

    if shares <= 0:
        log.append({'level': 'skip', 'msg': f'{sym} {gap["date"]}: position size = 0'})
        return day_trades, log

    state.positions[sym] = True
    notional = shares * entry_price
    borrow_cost = notional * (config.borrow_rate_annual / 252) if direction == 'short' else 0

    side_label = 'LONG' if direction == 'long' else 'SHORT'
    stop_dist = abs(stop_price - entry_price) / entry_price
    target_dist = abs(entry_price - full_target) / entry_price
    log.append({
        'level': 'entry',
        'msg': (f'{sym} {entry_time_str}: ORB {side_label} {shares}sh @ ${entry_price:.2f} '
                f'| gap {gap["gap_pct"]:.1%} vol {gap["vol_ratio"]:.2f}x '
                f'| OR [{or_low:.2f}-{or_high:.2f}] '
                f'| stop ${stop_price:.2f} ({stop_dist:.1%}) '
                f'| target ${full_target:.2f} ({target_dist:.1%})'),
        'data': {'symbol': sym, 'shares': shares, 'entry': entry_price,
                 'stop': stop_price, 'direction': direction}
    })

    # --- Walk forward through remaining 1-min bars for exits ---
    remaining = shares
    total_pnl = 0.0
    partial_done = False

    def _exit_slip(px):
        return px * (1 + slip) if direction == 'short' else px * (1 - slip)

    for bi in range(entry_bar + 1, len(times)):
        if remaining <= 0:
            break

        bar_high = float(highs[bi])
        bar_low = float(lows[bi])
        bar_close = float(closes[bi])
        bar_time = times[bi]
        if hasattr(bar_time, 'to_pydatetime'):
            bar_time = bar_time.to_pydatetime()
        bar_h, bar_m = bar_time.hour, bar_time.minute

        # Check stop hit
        stopped = _stop_hit(direction, bar_high, bar_low, stop_price)
        if stopped:
            fp = _exit_slip(stop_price)
            pnl = _direction_pnl(direction, entry_price, fp, remaining) - borrow_cost
            pnl_pct = pnl / (entry_price * remaining) if entry_price > 0 else 0
            hold_min = int((bar_time - entry_time).total_seconds() / 60)
            trade = TradeRecord(
                symbol=sym, entry_price=entry_price, exit_price=fp,
                shares=remaining, pnl=pnl, pnl_pct=pnl_pct,
                entry_time=entry_time_str, exit_time=bar_time.strftime('%Y-%m-%d %H:%M'),
                exit_reason='stop', holding_minutes=hold_min, side=direction,
            )
            day_trades.append(trade)
            state.record_trade(trade)
            total_pnl += pnl
            remaining = 0
            log.append({'level': 'stop', 'msg': f'{sym} {bar_time.strftime("%H:%M")}: STOP {shares}sh @ ${fp:.2f} | P&L ${pnl:.0f} ({pnl_pct:+.1%})',
                        'data': {'pnl': pnl, 'reason': 'stop'}})
            break

        # Check partial target
        partial_hit = _target_hit(direction, bar_low if direction == 'short' else bar_high, half_target)
        if partial_hit and not partial_done:
            cover_shares = max(1, int(shares * config.partial_cover_frac))
            partial_borrow = borrow_cost * cover_shares / shares if shares > 0 else 0
            fp = _exit_slip(half_target)
            pnl = _direction_pnl(direction, entry_price, fp, cover_shares) - partial_borrow
            pnl_pct = pnl / (entry_price * cover_shares) if entry_price > 0 else 0
            hold_min = int((bar_time - entry_time).total_seconds() / 60)
            trade = TradeRecord(
                symbol=sym, entry_price=entry_price, exit_price=fp,
                shares=cover_shares, pnl=pnl, pnl_pct=pnl_pct,
                entry_time=entry_time_str, exit_time=bar_time.strftime('%Y-%m-%d %H:%M'),
                exit_reason='partial', holding_minutes=hold_min, side=direction,
            )
            day_trades.append(trade)
            state.record_trade(trade)
            total_pnl += pnl
            remaining -= cover_shares
            partial_done = True
            log.append({'level': 'exit', 'msg': f'{sym} {bar_time.strftime("%H:%M")}: PARTIAL {cover_shares}sh @ ${fp:.2f} | P&L ${pnl:+.0f}',
                        'data': {'pnl': pnl, 'reason': 'partial'}})

        # Check full target
        full_hit = _target_hit(direction, bar_close, full_target)
        if full_hit and remaining > 0:
            remaining_borrow = borrow_cost * remaining / shares if shares > 0 else 0
            fp = _exit_slip(full_target)
            pnl = _direction_pnl(direction, entry_price, fp, remaining) - remaining_borrow
            pnl_pct = pnl / (entry_price * remaining) if entry_price > 0 else 0
            hold_min = int((bar_time - entry_time).total_seconds() / 60)
            trade = TradeRecord(
                symbol=sym, entry_price=entry_price, exit_price=fp,
                shares=remaining, pnl=pnl, pnl_pct=pnl_pct,
                entry_time=entry_time_str, exit_time=bar_time.strftime('%Y-%m-%d %H:%M'),
                exit_reason='full_target', holding_minutes=hold_min, side=direction,
            )
            day_trades.append(trade)
            state.record_trade(trade)
            total_pnl += pnl
            remaining = 0
            log.append({'level': 'exit', 'msg': f'{sym} {bar_time.strftime("%H:%M")}: FULL TARGET {trade.shares}sh @ ${fp:.2f} | P&L ${pnl:+.0f}',
                        'data': {'pnl': pnl, 'reason': 'full_target'}})
            break

        # Time exit at 15:55
        if bar_h >= 15 and bar_m >= 55 and remaining > 0:
            remaining_borrow = borrow_cost * remaining / shares if shares > 0 else 0
            fp = _exit_slip(bar_close)
            pnl = _direction_pnl(direction, entry_price, fp, remaining) - remaining_borrow
            pnl_pct = pnl / (entry_price * remaining) if entry_price > 0 else 0
            hold_min = int((bar_time - entry_time).total_seconds() / 60)
            trade = TradeRecord(
                symbol=sym, entry_price=entry_price, exit_price=fp,
                shares=remaining, pnl=pnl, pnl_pct=pnl_pct,
                entry_time=entry_time_str, exit_time=bar_time.strftime('%Y-%m-%d %H:%M'),
                exit_reason='time_exit', holding_minutes=hold_min, side=direction,
            )
            day_trades.append(trade)
            state.record_trade(trade)
            total_pnl += pnl
            remaining = 0
            log.append({'level': 'exit', 'msg': f'{sym} {bar_time.strftime("%H:%M")}: TIME EXIT {trade.shares}sh @ ${fp:.2f} | P&L ${pnl:+.0f}',
                        'data': {'pnl': pnl, 'reason': 'time_exit'}})
            break

    # If still holding (shouldn't happen with time exit, but safety)
    if remaining > 0:
        last_close = float(closes[-1])
        fp = _exit_slip(last_close)
        pnl = _direction_pnl(direction, entry_price, fp, remaining) - borrow_cost
        pnl_pct = pnl / (entry_price * remaining) if entry_price > 0 else 0
        trade = TradeRecord(
            symbol=sym, entry_price=entry_price, exit_price=fp,
            shares=remaining, pnl=pnl, pnl_pct=pnl_pct,
            entry_time=entry_time_str, exit_time=f'{gap["date"]} 16:00',
            exit_reason='time_exit', holding_minutes=360, side=direction,
        )
        day_trades.append(trade)
        state.record_trade(trade)
        total_pnl += pnl
        remaining = 0

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

        # ---- Phase 1 (0-25%): SQL gap scan (identical to GapFadeBacktester) ----
        await _log('info', f'Scanning {len(syms)} symbols via SQL...')
        await _progress(2, f"SQL gap scan for {len(syms)} symbols...")

        from gap_fade_app import GapScanner
        scanner_sql = GapScanner(self.config, syms)
        t0 = _time.time()
        all_gap_days_raw = await asyncio.to_thread(
            scanner_sql.scan_historical_batch,
            syms, start_date, end_date,
        )
        elapsed_scan = _time.time() - t0
        raw_gap_count = len(all_gap_days_raw)
        await _log('info', f'SQL scan found {raw_gap_count} raw gaps in {elapsed_scan:.1f}s')
        await _progress(15, f"Found {raw_gap_count} raw gaps. Filtering...")

        # Vol filter (direction-aware, same as original backtester)
        def _vol_ok(g):
            if g.get('direction') == 'long':
                if not self.config.trade_gap_downs:
                    return False
                return g['vol_ratio'] <= self.config.gap_down_vol_ratio_max
            return g['vol_ratio'] <= self.config.vol_ratio_max

        all_gap_days = [g for g in all_gap_days_raw if _vol_ok(g)]
        after_vol = len(all_gap_days)

        # Leveraged ETF filter
        if self.config.exclude_leveraged:
            all_gap_days = [g for g in all_gap_days if g['symbol'] not in LEVERAGED_ETFS]
            lev_filtered = after_vol - len(all_gap_days)
            if lev_filtered:
                await _log('scan', f'Excluded {lev_filtered} leveraged/inverse ETF gaps')

        await _log('scan', f'After filters: {len(all_gap_days)} gaps (from {raw_gap_count} raw)')
        await _progress(25, f"Found {len(all_gap_days)} gaps. Filtering...")

        if not all_gap_days:
            self.status = 'done'
            no_gaps = {'error': 'No qualifying gap days found', 'total_trades': 0, 'bt_log': bt_log}
            self.result = no_gaps
            await broadcast({'type': 'backtest_complete', 'result': no_gaps})
            return no_gaps

        # ---- Phase 2 (25-35%): Bulk data load for strategy filter ----
        # Load wide DataFrame only if strategy needs it (for BatchDataProvider)
        loader = DataLoader()
        all_data = None
        batch_provider = None

        if self.strategy:
            await _log('info', f'Applying {self.strategy.name} strategy filter...')
            # Load bulk data for BatchDataProvider (strategy needs get_bars)
            if all_data is None:
                await _progress(26, f"Loading bulk data for strategy filter...")
                all_data = await asyncio.to_thread(loader.load_universe, syms, start_date, end_date)
                batch_provider = BatchDataProvider(all_data) if not all_data.empty else None
            # Monkey-patch PriceDB.get_bars temporarily
            db = get_price_db()
            original_get_bars = db.get_bars
            if batch_provider:
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

        # ---- Phase 4b (40-50%): Fetch 1-min bars for ORB if enabled ----
        min_bar_cache: Dict[str, Dict[str, pd.DataFrame]] = {}  # {date: {symbol: df}}
        if self.config.orb_enabled:
            from gap_fade_app import fetch_alpaca_bars
            # Collect unique (symbol, date) pairs
            orb_pairs = [(g['symbol'], g['date']) for g in all_gap_days]
            unique_dates = sorted(set(d for _, d in orb_pairs))
            symbols_by_date = defaultdict(list)
            for sym, dt in orb_pairs:
                symbols_by_date[dt].append(sym)

            total_fetches = len(orb_pairs)
            fetched = 0
            await _log('info', f'ORB mode: fetching 1-min bars for {total_fetches} gap events across {len(unique_dates)} days...')

            for dt in unique_dates:
                if self._cancel:
                    break
                dt_syms = symbols_by_date[dt]
                min_bar_cache[dt] = {}
                for sym in dt_syms:
                    if self._cancel:
                        break
                    try:
                        df = await asyncio.to_thread(fetch_alpaca_bars, sym, dt, dt, '1Min', 'sip')
                        if df is not None and len(df) >= 10:
                            min_bar_cache[dt][sym] = df
                    except Exception as e:
                        logger.debug(f"1-min fetch failed for {sym} {dt}: {e}")
                    fetched += 1
                    if fetched % 20 == 0:
                        pct = 40 + (fetched / total_fetches) * 10
                        await _progress(pct, f"Fetching 1-min bars: {fetched}/{total_fetches}")
                        await asyncio.sleep(0)

            cached_count = sum(len(v) for v in min_bar_cache.values())
            await _log('info', f'ORB: fetched 1-min bars for {cached_count}/{total_fetches} gap events')

        await _progress(50, f"Simulating {len(all_gap_days)} gaps...")

        # ---- Phase 5 (50-95%): Simulation loop ----
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
                    # Use 1-min ORB simulation when enabled and data available
                    if self.config.orb_enabled and date_str in min_bar_cache and gap['symbol'] in min_bar_cache.get(date_str, {}):
                        day_trades_result, day_log = simulate_gap_day_orb_1min(
                            gap, min_bar_cache[date_str][gap['symbol']],
                            state, self.config, dd_scale=dd_scale)
                    else:
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
                     f"(scan={elapsed_scan:.1f}s)")

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


# ---------------------------------------------------------------------------
# IntradayBacktester — backtest intraday strategies on daily OHLCV data
# ---------------------------------------------------------------------------

@dataclass
class IntradayBarSim:
    """Simulated 5-min bar derived from daily OHLCV."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def synthesize_intraday_bars(date_str: str, day_open: float, day_high: float,
                              day_low: float, day_close: float,
                              day_volume: float, prev_close: float,
                              n_bars: int = 78) -> List[IntradayBarSim]:
    """Synthesize plausible 5-min bars from daily OHLCV data.

    Creates a realistic intraday price path:
    - First 3 bars (9:30-9:45): opening range from open to ~15-min extremes
    - Middle bars: random walk constrained by day high/low
    - Final bars: converge toward close

    Returns list of IntradayBarSim (78 bars = 6.5 hours, 9:30 to 16:00).
    """
    from zoneinfo import ZoneInfo
    ET = ZoneInfo('US/Eastern')
    base_dt = datetime.strptime(date_str, '%Y-%m-%d').replace(
        hour=9, minute=30, tzinfo=ET)

    if n_bars <= 0 or day_open <= 0:
        return []

    bars: List[IntradayBarSim] = []
    bar_vol = day_volume / n_bars if day_volume > 0 else 1000

    # Generate price path with n_bars+1 price points
    prices = [day_open]
    rng = np.random.RandomState(hash(date_str) % (2**31))

    # Phase 1: Opening range (bars 0-2, 9:30-9:45)
    # Determine OR extremes (fraction of day range)
    day_range = day_high - day_low
    or_fraction = 0.4 + rng.random() * 0.3  # OR captures 40-70% of day range
    or_range = day_range * or_fraction

    # OR high/low centered around open
    or_mid = day_open
    or_high = min(day_high, or_mid + or_range * 0.5)
    or_low = max(day_low, or_mid - or_range * 0.5)

    # Generate OR path
    or_prices = [day_open]
    for i in range(3):
        if i == 0:
            # First bar: volatile, move toward one extreme
            target = or_low if rng.random() < 0.5 else or_high
            p = day_open + (target - day_open) * (0.3 + rng.random() * 0.4)
        elif i == 1:
            # Second bar: move toward the other extreme
            if or_prices[-1] < or_mid:
                p = or_mid + (or_high - or_mid) * rng.random() * 0.7
            else:
                p = or_mid - (or_mid - or_low) * rng.random() * 0.7
        else:
            # Third bar: settle near or_mid
            p = or_mid + (rng.random() - 0.5) * or_range * 0.3
        or_prices.append(np.clip(p, day_low, day_high))

    prices = or_prices  # 4 points (open + 3 closes)

    # Phase 2: Mid-session (bars 3 to n_bars-10)
    mid_end = max(4, n_bars - 10)
    current = prices[-1]
    drift = (day_close - current) / max(1, mid_end - 3)  # gentle drift toward close
    volatility = day_range / n_bars * 1.5

    for i in range(3, mid_end):
        noise = rng.normal(0, volatility)
        current = current + drift + noise
        current = np.clip(current, day_low, day_high)
        prices.append(float(current))

    # Phase 3: Convergence to close (last ~10 bars)
    remaining_bars = n_bars - len(prices) + 1
    current = prices[-1]
    for i in range(remaining_bars):
        progress = (i + 1) / remaining_bars
        target = day_close
        # Increase pull toward close as we approach end
        current = current + (target - current) * (0.1 + 0.3 * progress)
        noise = rng.normal(0, volatility * (1 - progress * 0.7))
        current = np.clip(current + noise, day_low, day_high)
        prices.append(float(current))

    # Force last price to be close
    prices[-1] = day_close

    # Build bars from consecutive price points
    for i in range(min(n_bars, len(prices) - 1)):
        bar_open = prices[i]
        bar_close = prices[i + 1]
        bar_high = max(bar_open, bar_close) + abs(rng.normal(0, volatility * 0.3))
        bar_low = min(bar_open, bar_close) - abs(rng.normal(0, volatility * 0.3))
        bar_high = min(bar_high, day_high)
        bar_low = max(bar_low, day_low)

        # Volume: higher at open and close
        time_frac = i / n_bars
        vol_mult = 1.0
        if time_frac < 0.1:
            vol_mult = 2.0 + rng.random()  # Opening volume surge
        elif time_frac > 0.85:
            vol_mult = 1.5 + rng.random() * 0.5  # Closing volume
        else:
            vol_mult = 0.5 + rng.random() * 0.8

        ts = base_dt + timedelta(minutes=5 * i)
        bars.append(IntradayBarSim(
            timestamp=ts,
            open=round(bar_open, 2),
            high=round(bar_high, 2),
            low=round(bar_low, 2),
            close=round(bar_close, 2),
            volume=round(bar_vol * vol_mult),
        ))

    return bars


@dataclass
class IntradayPosition:
    """Tracks an open intraday position during backtest."""
    symbol: str
    strategy_id: str
    direction: str
    entry_price: float
    stop_price: float
    target_price: float
    shares: int
    entry_time: datetime
    setup_type: str = ''


class IntradayBacktester:
    """Backtests intraday strategies (ORB, Momentum, Pullback, Range) on daily bars.

    Synthesizes plausible 5-min bar paths from daily OHLCV, feeds them through
    the indicator engine and each strategy's scan_for_setups(), then simulates
    entry/exit with proper stop/target logic.
    """

    def __init__(self, config: GapFadeConfig = None,
                 strategy_ids: List[str] = None):
        self.config = config or GapFadeConfig()
        self.strategy_ids = strategy_ids or ['orb_breakout', 'momentum_surge',
                                              'pullback_entry', 'range_trade']
        self.progress = 0.0
        self.status = 'idle'
        self._cancel = False
        self.result = None

    def _load_strategies(self) -> Dict[str, Any]:
        """Load intraday strategy instances."""
        from gap_fade_strategies import IntradayStrategyRegistry
        strategies = {}
        for sid in self.strategy_ids:
            if IntradayStrategyRegistry.has_strategy(sid):
                strategies[sid] = IntradayStrategyRegistry.create_strategy(sid)
        return strategies

    def _build_tick_data(self, bars: List[IntradayBarSim], bar_idx: int,
                         prev_close: float) -> Dict:
        """Build tick_data dict from simulated bars up to bar_idx.

        Computes EMAs, RSI, VWAP, ATR, volume stats from the bars seen so far.
        """
        if bar_idx < 0 or not bars:
            return {}

        visible_bars = bars[:bar_idx + 1]
        current = visible_bars[-1]

        # VWAP: cumulative (price * volume) / cumulative volume
        cum_pv = sum(b.close * b.volume for b in visible_bars)
        cum_vol = sum(b.volume for b in visible_bars)
        vwap = cum_pv / cum_vol if cum_vol > 0 else current.close

        # Day high/low
        day_high = max(b.high for b in visible_bars)
        day_low = min(b.low for b in visible_bars)

        # Opening range (first 3 bars = 15 min)
        or_bars = visible_bars[:min(3, len(visible_bars))]
        or_high = max(b.high for b in or_bars)
        or_low = min(b.low for b in or_bars)
        or_complete = len(visible_bars) >= 3

        # Simple EMAs (exponential smoothing over close prices)
        closes = [b.close for b in visible_bars]
        n = len(closes)

        def _ema(period):
            if n == 0:
                return current.close
            if n < period:
                return sum(closes) / n
            mult = 2.0 / (period + 1)
            ema_val = closes[0]
            for c in closes[1:]:
                ema_val = c * mult + ema_val * (1 - mult)
            return ema_val

        ema9 = _ema(9)
        ema20 = _ema(20)
        ema50 = _ema(50)

        # RSI (Wilder's smoothed, 14-period)
        rsi = 50.0
        rsi_initialized = False
        rsi_history = []
        if n >= 2:
            changes = [closes[i] - closes[i-1] for i in range(1, n)]
            gains = [max(0, c) for c in changes]
            losses = [max(0, -c) for c in changes]

            if len(changes) >= 14:
                avg_gain = sum(gains[:14]) / 14
                avg_loss = sum(losses[:14]) / 14
                for i in range(14, len(changes)):
                    avg_gain = (avg_gain * 13 + gains[i]) / 14
                    avg_loss = (avg_loss * 13 + losses[i]) / 14
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100.0 if avg_gain > 0 else 50.0
                rsi_initialized = True
            elif len(changes) >= 5:
                avg_gain = sum(gains) / len(gains)
                avg_loss = sum(losses) / len(losses)
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                rsi_initialized = True

        # ATR (14-period)
        atr = 0.0
        if n >= 2:
            trs = []
            for i in range(1, n):
                prev_c = visible_bars[i-1].close
                tr = max(
                    visible_bars[i].high - visible_bars[i].low,
                    abs(visible_bars[i].high - prev_c),
                    abs(visible_bars[i].low - prev_c)
                )
                trs.append(tr)
            if trs:
                atr = sum(trs[-min(14, len(trs)):]) / min(14, len(trs))

        # Volume surge
        volumes = [b.volume for b in visible_bars]
        vol_avg_20 = sum(volumes[-min(20, n):]) / min(20, n) if n > 0 else 1
        volume_surge = current.volume / vol_avg_20 if vol_avg_20 > 0 else 1.0

        return {
            'ema9': ema9, 'ema20': ema20, 'ema50': ema50,
            'rsi': rsi, 'rsi_initialized': rsi_initialized,
            'rsi_history': rsi_history,
            'atr': atr, 'atr_initialized': n >= 15,
            'vwap': vwap,
            'day_high': day_high, 'day_low': day_low,
            'or_high': or_high, 'or_low': or_low, 'or_complete': or_complete,
            'volume_surge_ratio': volume_surge,
            'volume_avg_20bar': vol_avg_20,
            'bar_count': n,
            'bar_history': [],
        }

    async def run(self, symbols: List[str] = None,
                  start_date: str = None, end_date: str = None,
                  config: GapFadeConfig = None,
                  progress_callback=None, **kwargs) -> dict:
        """Run intraday strategy backtest.

        Synthesizes 5-min bars from daily data, runs each strategy's
        scan_for_setups() per bar, simulates positions.

        Returns result dict with performance metrics.
        """
        if config:
            self.config = config

        self.status = 'running'
        self._cancel = False
        self.progress = 0

        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        syms = symbols or UNIVERSE[:50]  # Default to top 50 symbols
        strategy_ids = kwargs.get('strategy_ids', self.strategy_ids)

        async def _progress(pct, msg):
            self.progress = pct
            if progress_callback:
                await progress_callback(pct, msg)

        # Load strategies
        strategies = self._load_strategies()
        if not strategies:
            self.status = 'done'
            self.result = {'error': 'No strategies found', 'total_trades': 0}
            return self.result

        await _progress(5, f"Loading data for {len(syms)} symbols...")

        # Load daily data
        loader = DataLoader()
        all_data = await asyncio.to_thread(
            loader.load_universe, syms, start_date, end_date, warmup_days=30)

        if all_data.empty:
            self.status = 'done'
            self.result = {'error': 'No data found', 'total_trades': 0}
            return self.result

        # Load SPY for market condition detection
        spy_df = await asyncio.to_thread(loader.load_spy, start_date, end_date)

        await _progress(15, "Data loaded. Building simulation schedule...")

        # Get trading dates in range
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        date_mask = (all_data.index >= start_dt) & (all_data.index <= end_dt)
        trading_dates = all_data.index[date_mask].unique()

        if len(trading_dates) == 0:
            self.status = 'done'
            self.result = {'error': 'No trading dates in range', 'total_trades': 0}
            return self.result

        logger.info(f"INTRADAY BACKTEST: {len(strategy_ids)} strategies, "
                    f"{len(syms)} symbols, {len(trading_dates)} days")

        # Initialize market condition detector
        from gap_fade_strategies import MarketConditionDetector, StrategySelector
        selector = StrategySelector(strategies)

        # Simulation state
        initial_capital = self.config.initial_capital
        equity = initial_capital
        peak_equity = initial_capital
        equity_curve = [initial_capital]
        all_trades: List[TradeRecord] = []
        bt_log: List[dict] = []
        trades_by_strategy: Dict[str, List[TradeRecord]] = defaultdict(list)
        daily_pnl: List[float] = []
        setups_found = 0
        setups_entered = 0
        risk_pct = self.config.intraday_risk_pct or 0.01
        max_entries = self.config.intraday_max_entries or 3
        daily_loss_limit = self.config.intraday_daily_loss_limit or 0.02
        slippage = self.config.slippage_pct

        await _progress(20, f"Simulating {len(trading_dates)} trading days...")

        def _simulate():
            nonlocal equity, peak_equity, setups_found, setups_entered

            for di, trade_date in enumerate(trading_dates):
                if self._cancel:
                    break

                date_str = trade_date.strftime('%Y-%m-%d')
                day_entries = 0
                day_pnl = 0.0
                positions: Dict[str, IntradayPosition] = {}

                # Reset strategies for the day
                for strat in strategies.values():
                    strat.on_day_start()

                # Get SPY data for market condition
                spy_tick_data = None
                if not spy_df.empty and trade_date in spy_df.index:
                    spy_row = spy_df.loc[trade_date]
                    # Build minimal SPY tick data for condition detection
                    spy_o = float(spy_row['open'])
                    spy_h = float(spy_row['high'])
                    spy_l = float(spy_row['low'])
                    spy_c = float(spy_row['close'])
                    spy_range = spy_h - spy_l
                    spy_mid = (spy_h + spy_l) / 2
                    spy_tick_data = {
                        'ema9': spy_c * 1.001 if spy_c > spy_o else spy_c * 0.999,
                        'ema20': spy_mid,
                        'ema50': spy_mid * 0.998,
                        'rsi': 60 if spy_c > spy_o else 40,
                        'rsi_initialized': True,
                        'rsi_history': [50, 52, 48, 51, 49],
                        'day_high': spy_h,
                        'day_low': spy_l,
                        'vwap': spy_mid,
                        'bar_history': [],
                    }

                # Determine active strategies via selector
                if spy_tick_data:
                    from zoneinfo import ZoneInfo
                    ET = ZoneInfo('US/Eastern')
                    mid_ts = datetime.strptime(date_str, '%Y-%m-%d').replace(
                        hour=11, minute=0, tzinfo=ET)
                    selector.update(spy_tick_data, mid_ts, force=True)
                    active_strategies = selector.active_strategies
                    size_mult = selector.size_multiplier
                else:
                    active_strategies = strategies
                    size_mult = 1.0

                if not active_strategies:
                    equity_curve.append(equity)
                    daily_pnl.append(0.0)
                    continue

                # Iterate over symbols for this day
                for sym in syms:
                    if self._cancel:
                        break
                    if day_entries >= max_entries:
                        break
                    if abs(day_pnl) > equity * daily_loss_limit:
                        break

                    sym_bars_df = DataLoader.get_symbol_bars(sym, all_data)
                    if sym_bars_df is None or trade_date not in sym_bars_df.index:
                        continue

                    row = sym_bars_df.loc[trade_date]
                    day_open = float(row['open'])
                    day_high = float(row['high'])
                    day_low = float(row['low'])
                    day_close = float(row['close'])
                    day_volume = float(row['volume'])

                    if day_open <= 0 or day_high <= 0:
                        continue

                    # Get previous close for context
                    date_idx = sym_bars_df.index.get_loc(trade_date)
                    if date_idx == 0:
                        prev_close = day_open
                    else:
                        prev_close = float(sym_bars_df.iloc[date_idx - 1]['close'])

                    # Synthesize intraday bars
                    intraday_bars = synthesize_intraday_bars(
                        date_str, day_open, day_high, day_low,
                        day_close, day_volume, prev_close)

                    if not intraday_bars:
                        continue

                    # Skip first 3 bars (opening range build-up)
                    # Scan bars 3+ for setups
                    for bi in range(3, len(intraday_bars)):
                        if day_entries >= max_entries:
                            break
                        if abs(day_pnl) > equity * daily_loss_limit:
                            break

                        bar = intraday_bars[bi]
                        tick_data = self._build_tick_data(
                            intraday_bars, bi, prev_close)
                        snapshot = {'price': bar.close}

                        # Check if already in position for this symbol
                        if sym in positions:
                            pos = positions[sym]
                            # Check stop hit
                            if pos.direction == 'long':
                                if bar.low <= pos.stop_price:
                                    exit_price = pos.stop_price * (1 - slippage)
                                    pnl = (exit_price - pos.entry_price) * pos.shares
                                    trade = TradeRecord(
                                        symbol=sym, entry_price=pos.entry_price,
                                        exit_price=exit_price, shares=pos.shares,
                                        pnl=pnl, pnl_pct=pnl / (pos.entry_price * pos.shares),
                                        entry_time=pos.entry_time.strftime('%Y-%m-%d %H:%M'),
                                        exit_time=bar.timestamp.strftime('%Y-%m-%d %H:%M'),
                                        exit_reason='stop', holding_minutes=(bi - 3) * 5,
                                        side='long',
                                    )
                                    all_trades.append(trade)
                                    trades_by_strategy[pos.strategy_id].append(trade)
                                    equity += pnl
                                    day_pnl += pnl
                                    del positions[sym]
                                    continue

                                # Check target hit
                                if bar.high >= pos.target_price:
                                    exit_price = pos.target_price * (1 - slippage)
                                    pnl = (exit_price - pos.entry_price) * pos.shares
                                    trade = TradeRecord(
                                        symbol=sym, entry_price=pos.entry_price,
                                        exit_price=exit_price, shares=pos.shares,
                                        pnl=pnl, pnl_pct=pnl / (pos.entry_price * pos.shares),
                                        entry_time=pos.entry_time.strftime('%Y-%m-%d %H:%M'),
                                        exit_time=bar.timestamp.strftime('%Y-%m-%d %H:%M'),
                                        exit_reason='target', holding_minutes=(bi - 3) * 5,
                                        side='long',
                                    )
                                    all_trades.append(trade)
                                    trades_by_strategy[pos.strategy_id].append(trade)
                                    equity += pnl
                                    day_pnl += pnl
                                    del positions[sym]
                                    continue

                                # Update trailing stop via strategy
                                strat = active_strategies.get(pos.strategy_id) or strategies.get(pos.strategy_id)
                                if strat:
                                    pos_dict = {
                                        'direction': pos.direction,
                                        'entry_price': pos.entry_price,
                                        'stop_price': pos.stop_price,
                                    }
                                    new_stop = strat.update_trailing_stop(
                                        pos_dict, bar.close, tick_data, bar.timestamp)
                                    if new_stop is not None and new_stop > pos.stop_price:
                                        pos.stop_price = new_stop

                            else:  # short
                                if bar.high >= pos.stop_price:
                                    exit_price = pos.stop_price * (1 + slippage)
                                    pnl = (pos.entry_price - exit_price) * pos.shares
                                    trade = TradeRecord(
                                        symbol=sym, entry_price=pos.entry_price,
                                        exit_price=exit_price, shares=pos.shares,
                                        pnl=pnl, pnl_pct=pnl / (pos.entry_price * pos.shares),
                                        entry_time=pos.entry_time.strftime('%Y-%m-%d %H:%M'),
                                        exit_time=bar.timestamp.strftime('%Y-%m-%d %H:%M'),
                                        exit_reason='stop', holding_minutes=(bi - 3) * 5,
                                        side='short',
                                    )
                                    all_trades.append(trade)
                                    trades_by_strategy[pos.strategy_id].append(trade)
                                    equity += pnl
                                    day_pnl += pnl
                                    del positions[sym]
                                    continue

                                if bar.low <= pos.target_price:
                                    exit_price = pos.target_price * (1 + slippage)
                                    pnl = (pos.entry_price - exit_price) * pos.shares
                                    trade = TradeRecord(
                                        symbol=sym, entry_price=pos.entry_price,
                                        exit_price=exit_price, shares=pos.shares,
                                        pnl=pnl, pnl_pct=pnl / (pos.entry_price * pos.shares),
                                        entry_time=pos.entry_time.strftime('%Y-%m-%d %H:%M'),
                                        exit_time=bar.timestamp.strftime('%Y-%m-%d %H:%M'),
                                        exit_reason='target', holding_minutes=(bi - 3) * 5,
                                        side='short',
                                    )
                                    all_trades.append(trade)
                                    trades_by_strategy[pos.strategy_id].append(trade)
                                    equity += pnl
                                    day_pnl += pnl
                                    del positions[sym]
                                    continue

                                strat = active_strategies.get(pos.strategy_id) or strategies.get(pos.strategy_id)
                                if strat:
                                    pos_dict = {
                                        'direction': pos.direction,
                                        'entry_price': pos.entry_price,
                                        'stop_price': pos.stop_price,
                                    }
                                    new_stop = strat.update_trailing_stop(
                                        pos_dict, bar.close, tick_data, bar.timestamp)
                                    if new_stop is not None and new_stop < pos.stop_price:
                                        pos.stop_price = new_stop

                            continue  # Still in position, skip scanning

                        # Scan for setups with active strategies
                        for sid, strat in active_strategies.items():
                            if day_entries >= max_entries:
                                break

                            setup = strat.scan_for_setups(
                                sym, tick_data, snapshot, bar.timestamp)
                            if setup is None:
                                continue

                            setups_found += 1

                            # Validate R:R and confidence
                            valid, reason = strat.validate_setup(
                                setup, tick_data, bar.timestamp)
                            if not valid:
                                continue

                            # Position sizing
                            risk_per_share = abs(setup.entry_price - setup.stop_price)
                            if risk_per_share <= 0:
                                continue

                            dollar_risk = equity * risk_pct * size_mult
                            shares = int(dollar_risk / risk_per_share)
                            if shares <= 0:
                                continue

                            # Entry slippage
                            if setup.direction == 'long':
                                entry_price = setup.entry_price * (1 + slippage)
                            else:
                                entry_price = setup.entry_price * (1 - slippage)

                            positions[sym] = IntradayPosition(
                                symbol=sym,
                                strategy_id=sid,
                                direction=setup.direction,
                                entry_price=entry_price,
                                stop_price=setup.stop_price,
                                target_price=setup.target_price,
                                shares=shares,
                                entry_time=bar.timestamp,
                                setup_type=setup.setup_type,
                            )
                            day_entries += 1
                            setups_entered += 1
                            break  # One setup per symbol per scan

                    # EOD: close any remaining positions
                    if sym in positions:
                        pos = positions[sym]
                        exit_price = day_close
                        if pos.direction == 'long':
                            exit_price *= (1 - slippage)
                            pnl = (exit_price - pos.entry_price) * pos.shares
                        else:
                            exit_price *= (1 + slippage)
                            pnl = (pos.entry_price - exit_price) * pos.shares

                        trade = TradeRecord(
                            symbol=sym, entry_price=pos.entry_price,
                            exit_price=exit_price, shares=pos.shares,
                            pnl=pnl, pnl_pct=pnl / (pos.entry_price * pos.shares),
                            entry_time=pos.entry_time.strftime('%Y-%m-%d %H:%M'),
                            exit_time=f'{date_str} 15:55',
                            exit_reason='eod', holding_minutes=375,
                            side=pos.direction,
                        )
                        all_trades.append(trade)
                        trades_by_strategy[pos.strategy_id].append(trade)
                        equity += pnl
                        day_pnl += pnl
                        del positions[sym]

                if equity > peak_equity:
                    peak_equity = equity
                equity_curve.append(equity)
                daily_pnl.append(day_pnl)

        # Run simulation in thread
        await _progress(25, "Running simulation...")
        await asyncio.to_thread(_simulate)
        await _progress(90, "Computing metrics...")

        # Build metrics
        if not all_trades:
            self.status = 'done'
            self.result = {
                'total_trades': 0,
                'setups_found': setups_found,
                'start_date': start_date,
                'end_date': end_date,
                'strategies': list(strategy_ids),
                'symbols_count': len(syms),
                'trading_days': len(trading_dates),
            }
            return self.result

        pnls = [t.pnl for t in all_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_pnl = sum(pnls)
        win_rate = len(wins) / len(all_trades)
        avg_win = float(np.mean(wins)) if wins else 0
        avg_loss = float(np.mean(losses)) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 9999.99

        # Max drawdown
        eq_arr = np.array(equity_curve)
        peak_arr = np.maximum.accumulate(eq_arr)
        dd_arr = (peak_arr - eq_arr) / np.where(peak_arr > 0, peak_arr, 1)
        max_dd_pct = float(np.max(dd_arr)) if len(dd_arr) > 0 else 0

        # Sharpe
        if len(equity_curve) > 2:
            eq_np = np.array(equity_curve)
            daily_rets = np.diff(eq_np) / eq_np[:-1]
            daily_rets = daily_rets[np.isfinite(daily_rets)]
            if len(daily_rets) > 1 and np.std(daily_rets) > 0:
                sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252))
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Per-strategy breakdown
        strategy_metrics = {}
        for sid, strades in trades_by_strategy.items():
            if not strades:
                continue
            s_pnls = [t.pnl for t in strades]
            s_wins = [p for p in s_pnls if p > 0]
            s_losses = [p for p in s_pnls if p <= 0]
            strategy_metrics[sid] = {
                'trades': len(strades),
                'wins': len(s_wins),
                'losses': len(s_losses),
                'win_rate': round(len(s_wins) / len(strades), 4),
                'total_pnl': round(sum(s_pnls), 2),
                'avg_win': round(float(np.mean(s_wins)), 2) if s_wins else 0,
                'avg_loss': round(float(np.mean(s_losses)), 2) if s_losses else 0,
                'profit_factor': round(abs(sum(s_wins) / sum(s_losses)), 2) if s_losses and sum(s_losses) != 0 else 9999.99,
            }

        # Exit reason breakdown
        exit_reasons = defaultdict(int)
        for t in all_trades:
            exit_reasons[t.exit_reason] += 1

        result = {
            'total_trades': len(all_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(win_rate, 4),
            'total_pnl': round(total_pnl, 2),
            'return_pct': round(total_pnl / initial_capital * 100, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown_pct': round(max_dd_pct * 100, 2),
            'sharpe': round(sharpe, 2),
            'final_equity': round(equity, 2),

            'setups_found': setups_found,
            'setups_entered': setups_entered,
            'trading_days': len(trading_dates),
            'symbols_count': len(syms),
            'start_date': start_date,
            'end_date': end_date,

            'strategy_breakdown': strategy_metrics,
            'exit_reasons': dict(exit_reasons),
            'strategies': list(strategy_ids),

            'trades': [asdict(t) for t in all_trades[-200:]],
            'all_trades': [asdict(t) for t in all_trades],
            'equity_curve': [round(e, 2) for e in equity_curve[-500:]],

            'config': {
                'risk_pct': risk_pct,
                'max_entries': max_entries,
                'daily_loss_limit': daily_loss_limit,
                'slippage_pct': slippage,
                'size_multiplier_source': 'strategy_selector',
            },
        }

        self.progress = 100
        self.status = 'done'
        self.result = result

        logger.info(f"INTRADAY BACKTEST DONE: {result['total_trades']} trades, "
                    f"{result['win_rate']:.1%} win rate, "
                    f"${result['total_pnl']:,.0f} P&L ({result['return_pct']:.1f}%)")

        return result

    def cancel(self):
        """Cancel running backtest."""
        self._cancel = True
        self.status = 'cancelled'


# ---------------------------------------------------------------------------
# Intraday Walk-Forward Optimization
# ---------------------------------------------------------------------------

DEFAULT_INTRADAY_WF_GRIDS = {
    'orb_breakout': {
        'breakout_buffer_pct': [0.0005, 0.001, 0.002],
        'volume_confirm_ratio': [1.2, 1.5, 2.0],
        'stop_buffer_pct': [0.001, 0.002, 0.003],
        'target_rr': [1.5, 2.0, 2.5],
    },  # 81 combos
    'momentum_surge': {
        'volume_surge_ratio': [1.5, 2.0, 3.0],
        'rsi_min_long': [45, 50, 55],
        'atr_stop_mult': [1.0, 1.5, 2.0],
        'target_rr': [1.5, 2.0, 3.0],
    },  # 81 combos
    'pullback_entry': {
        'pullback_proximity_pct': [0.005, 0.01, 0.015],
        'rsi_pullback_min': [35, 40, 45],
        'rsi_pullback_max': [50, 55, 60],
        'target_rr': [1.5, 2.0, 2.5],
    },  # 81 combos
    'range_trade': {
        'boundary_proximity_pct': [0.002, 0.003, 0.005],
        'stop_buffer_pct': [0.003, 0.005, 0.008],
        'rsi_support_max': [35, 40, 45],
        'target_rr': [1.5, 2.0, 2.5],
    },  # 81 combos
}


@dataclass
class IntradayDayCache:
    """Pre-computed intraday data for one symbol on one day."""
    date_str: str
    symbol: str
    bars: List[IntradayBarSim]
    tick_data_by_bar: List[Dict]    # tick_data for bars 3..77
    market_condition: str           # trending_up/down/choppy/sideways
    day_data: dict                  # raw OHLCV


# Market condition routing (mirrors StrategySelector.DEFAULT_STRATEGY_MAP)
_CONDITION_STRATEGY_MAP = {
    'trending_up': ['orb_breakout', 'momentum_surge', 'pullback_entry'],
    'trending_down': ['orb_breakout', 'momentum_surge', 'pullback_entry'],
    'choppy': ['orb_breakout'],
    'sideways': ['range_trade'],
}


def _classify_market_condition(spy_row: dict) -> str:
    """Classify market condition from SPY daily bar."""
    if not spy_row:
        return 'sideways'
    o = spy_row.get('open', 0)
    h = spy_row.get('high', 0)
    l = spy_row.get('low', 0)
    c = spy_row.get('close', 0)
    if o <= 0 or c <= 0:
        return 'sideways'
    change_pct = (c - o) / o
    day_range = (h - l) / o if o > 0 else 0
    body_pct = abs(c - o) / o
    # Trending: strong directional move
    if change_pct > 0.005 and body_pct > day_range * 0.4:
        return 'trending_up'
    if change_pct < -0.005 and body_pct > day_range * 0.4:
        return 'trending_down'
    # Choppy: wide range but small body
    if day_range > 0.015 and body_pct < day_range * 0.3:
        return 'choppy'
    return 'sideways'


def _precompute_intraday_data(
    symbols: List[str],
    all_data: pd.DataFrame,
    spy_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    cancel_check: Callable = None,
) -> List[IntradayDayCache]:
    """Synthesize bars and build tick_data once for all symbols/days.

    This is the expensive step. Runs once per fold window, then reused
    across all grid combos.
    """
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    date_mask = (all_data.index >= start_dt) & (all_data.index <= end_dt)
    trading_dates = sorted(all_data.index[date_mask].unique())

    caches: List[IntradayDayCache] = []

    # Build a quick helper for _build_tick_data
    bt = IntradayBacktester()

    for trade_date in trading_dates:
        if cancel_check and cancel_check():
            break
        date_str = trade_date.strftime('%Y-%m-%d')

        # SPY market condition
        spy_row = {}
        if not spy_df.empty and trade_date in spy_df.index:
            sr = spy_df.loc[trade_date]
            spy_row = {
                'open': float(sr['open']),
                'high': float(sr['high']),
                'low': float(sr['low']),
                'close': float(sr['close']),
            }
        condition = _classify_market_condition(spy_row)

        for sym in symbols:
            sym_bars_df = DataLoader.get_symbol_bars(sym, all_data)
            if sym_bars_df is None or trade_date not in sym_bars_df.index:
                continue

            row = sym_bars_df.loc[trade_date]
            day_open = float(row['open'])
            day_high = float(row['high'])
            day_low = float(row['low'])
            day_close = float(row['close'])
            day_volume = float(row['volume'])

            if day_open <= 0 or day_high <= 0:
                continue

            # Previous close
            date_idx = sym_bars_df.index.get_loc(trade_date)
            prev_close = float(sym_bars_df.iloc[date_idx - 1]['close']) if date_idx > 0 else day_open

            # Synthesize bars
            bars = synthesize_intraday_bars(
                date_str, day_open, day_high, day_low,
                day_close, day_volume, prev_close)
            if not bars:
                continue

            # Pre-compute tick_data for bars 3..77
            tick_data_list = []
            for bi in range(3, len(bars)):
                td = bt._build_tick_data(bars, bi, prev_close)
                tick_data_list.append(td)

            caches.append(IntradayDayCache(
                date_str=date_str,
                symbol=sym,
                bars=bars,
                tick_data_by_bar=tick_data_list,
                market_condition=condition,
                day_data={
                    'open': day_open, 'high': day_high,
                    'low': day_low, 'close': day_close,
                    'volume': day_volume, 'prev_close': prev_close,
                },
            ))

    return caches


def _simulate_intraday_strategy_fast(
    strategy_id: str,
    day_caches: List[IntradayDayCache],
    config_overrides: Dict,
    base_config: GapFadeConfig = None,
) -> dict:
    """Fast sync simulation of one strategy with given config overrides.

    Uses pre-computed bars and tick_data. Only re-runs strategy logic.
    Returns metrics dict.
    """
    from gap_fade_strategies import IntradayStrategyRegistry

    config = base_config or GapFadeConfig()
    risk_pct = config.intraday_risk_pct or 0.01
    slippage = config.slippage_pct
    initial_capital = config.initial_capital
    max_entries = config.intraday_max_entries or 3
    daily_loss_limit = config.intraday_daily_loss_limit or 0.02

    # Instantiate strategy with merged config
    strategy = IntradayStrategyRegistry.create_strategy(strategy_id, config=config_overrides)

    equity = initial_capital
    peak_equity = initial_capital
    equity_curve = [initial_capital]
    trades = []
    trades_by_condition = defaultdict(list)

    # Group caches by date for day-level simulation
    from itertools import groupby as _gb
    caches_sorted = sorted(day_caches, key=lambda c: c.date_str)

    for date_str, day_group in _gb(caches_sorted, key=lambda c: c.date_str):
        day_caches_list = list(day_group)
        if not day_caches_list:
            continue

        condition = day_caches_list[0].market_condition

        # Check if this strategy should be active in this market condition
        allowed = _CONDITION_STRATEGY_MAP.get(condition, [])
        if strategy_id not in allowed:
            equity_curve.append(equity)
            continue

        strategy.on_day_start()
        day_entries = 0
        day_pnl = 0.0
        positions: Dict[str, IntradayPosition] = {}

        for cache in day_caches_list:
            if day_entries >= max_entries:
                break
            if abs(day_pnl) > equity * daily_loss_limit:
                break

            sym = cache.symbol
            bars = cache.bars
            tick_data_list = cache.tick_data_by_bar

            for ti, td in enumerate(tick_data_list):
                bi = ti + 3  # actual bar index
                if day_entries >= max_entries:
                    break
                if abs(day_pnl) > equity * daily_loss_limit:
                    break

                bar = bars[bi]
                snapshot = {'price': bar.close}

                # Position management
                if sym in positions:
                    pos = positions[sym]
                    exited = False
                    if pos.direction == 'long':
                        if bar.low <= pos.stop_price:
                            exit_price = pos.stop_price * (1 - slippage)
                            pnl = (exit_price - pos.entry_price) * pos.shares
                            trades.append({'pnl': pnl, 'exit_reason': 'stop', 'direction': 'long', 'condition': condition})
                            trades_by_condition[condition].append(pnl)
                            equity += pnl; day_pnl += pnl
                            del positions[sym]; exited = True
                        elif bar.high >= pos.target_price:
                            exit_price = pos.target_price * (1 - slippage)
                            pnl = (exit_price - pos.entry_price) * pos.shares
                            trades.append({'pnl': pnl, 'exit_reason': 'target', 'direction': 'long', 'condition': condition})
                            trades_by_condition[condition].append(pnl)
                            equity += pnl; day_pnl += pnl
                            del positions[sym]; exited = True
                        else:
                            pos_dict = {'direction': 'long', 'entry_price': pos.entry_price, 'stop_price': pos.stop_price}
                            new_stop = strategy.update_trailing_stop(pos_dict, bar.close, td, bar.timestamp)
                            if new_stop is not None and new_stop > pos.stop_price:
                                pos.stop_price = new_stop
                    else:  # short
                        if bar.high >= pos.stop_price:
                            exit_price = pos.stop_price * (1 + slippage)
                            pnl = (pos.entry_price - exit_price) * pos.shares
                            trades.append({'pnl': pnl, 'exit_reason': 'stop', 'direction': 'short', 'condition': condition})
                            trades_by_condition[condition].append(pnl)
                            equity += pnl; day_pnl += pnl
                            del positions[sym]; exited = True
                        elif bar.low <= pos.target_price:
                            exit_price = pos.target_price * (1 + slippage)
                            pnl = (pos.entry_price - exit_price) * pos.shares
                            trades.append({'pnl': pnl, 'exit_reason': 'target', 'direction': 'short', 'condition': condition})
                            trades_by_condition[condition].append(pnl)
                            equity += pnl; day_pnl += pnl
                            del positions[sym]; exited = True
                        else:
                            pos_dict = {'direction': 'short', 'entry_price': pos.entry_price, 'stop_price': pos.stop_price}
                            new_stop = strategy.update_trailing_stop(pos_dict, bar.close, td, bar.timestamp)
                            if new_stop is not None and new_stop < pos.stop_price:
                                pos.stop_price = new_stop
                    if exited:
                        continue
                    continue  # still in position

                # Scan for setup
                setup = strategy.scan_for_setups(sym, td, snapshot, bar.timestamp)
                if setup is None:
                    continue

                valid, _ = strategy.validate_setup(setup, td, bar.timestamp)
                if not valid:
                    continue

                risk_per_share = abs(setup.entry_price - setup.stop_price)
                if risk_per_share <= 0:
                    continue

                dollar_risk = equity * risk_pct
                shares = int(dollar_risk / risk_per_share)
                if shares <= 0:
                    continue

                entry_price = setup.entry_price * (1 + slippage) if setup.direction == 'long' else setup.entry_price * (1 - slippage)
                positions[sym] = IntradayPosition(
                    symbol=sym, strategy_id=strategy_id,
                    direction=setup.direction, entry_price=entry_price,
                    stop_price=setup.stop_price, target_price=setup.target_price,
                    shares=shares, entry_time=bar.timestamp,
                )
                day_entries += 1
                break  # one setup per symbol per scan

            # EOD close
            if sym in positions:
                pos = positions[sym]
                exit_price = cache.day_data['close']
                if pos.direction == 'long':
                    exit_price *= (1 - slippage)
                    pnl = (exit_price - pos.entry_price) * pos.shares
                else:
                    exit_price *= (1 + slippage)
                    pnl = (pos.entry_price - exit_price) * pos.shares
                trades.append({'pnl': pnl, 'exit_reason': 'eod', 'direction': pos.direction, 'condition': condition})
                trades_by_condition[condition].append(pnl)
                equity += pnl; day_pnl += pnl
                del positions[sym]

        if equity > peak_equity:
            peak_equity = equity
        equity_curve.append(equity)

    # Compute metrics
    num_trades = len(trades)
    if num_trades == 0:
        return {
            'return_pct': 0, 'num_trades': 0, 'win_rate': 0,
            'profit_factor': 0, 'max_drawdown_pct': 0, 'sharpe': 0,
            'trades_by_condition': {},
        }

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / num_trades
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 9999.99

    eq_arr = np.array(equity_curve)
    peak_arr = np.maximum.accumulate(eq_arr)
    dd_arr = (peak_arr - eq_arr) / np.where(peak_arr > 0, peak_arr, 1)
    max_dd = float(np.max(dd_arr)) * 100 if len(dd_arr) > 0 else 0

    sharpe = 0
    if len(equity_curve) > 2:
        eq_np = np.array(equity_curve)
        daily_rets = np.diff(eq_np) / eq_np[:-1]
        daily_rets = daily_rets[np.isfinite(daily_rets)]
        if len(daily_rets) > 1 and np.std(daily_rets) > 0:
            sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252))

    ret_pct = (equity - initial_capital) / initial_capital * 100

    # Per-condition breakdown
    cond_breakdown = {}
    for cond, cond_pnls in trades_by_condition.items():
        cond_wins = [p for p in cond_pnls if p > 0]
        cond_breakdown[cond] = {
            'trades': len(cond_pnls),
            'win_rate': round(len(cond_wins) / len(cond_pnls), 4) if cond_pnls else 0,
            'avg_pnl': round(float(np.mean(cond_pnls)), 2) if cond_pnls else 0,
            'total_pnl': round(sum(cond_pnls), 2),
        }

    return {
        'return_pct': round(ret_pct, 2),
        'num_trades': num_trades,
        'win_rate': round(win_rate, 4),
        'profit_factor': round(pf, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'sharpe': round(sharpe, 2),
        'trades_by_condition': cond_breakdown,
    }


# Deployment gate criteria
INTRADAY_GATE_CRITERIA = {
    'min_oos_trades': 30,
    'min_oos_win_rate': 0.45,
    'min_oos_profit_factor': 1.1,
    'max_oos_drawdown': -15.0,
    'min_overfit_ratio': 0.4,
    'min_param_stability': 0.5,
}


def _check_deployment_gate(results: dict) -> dict:
    """Check if strategy passes deployment criteria."""
    agg = results.get('aggregate_oos', {})
    stability = results.get('param_stability', {})

    criteria_results = {}
    failures = 0

    # OOS trades
    actual = agg.get('num_trades', 0)
    passed = actual >= INTRADAY_GATE_CRITERIA['min_oos_trades']
    criteria_results['min_oos_trades'] = {
        'required': INTRADAY_GATE_CRITERIA['min_oos_trades'],
        'actual': actual, 'passed': passed,
    }
    if not passed:
        failures += 1

    # OOS win rate
    actual = agg.get('win_rate', 0)
    passed = actual >= INTRADAY_GATE_CRITERIA['min_oos_win_rate']
    criteria_results['min_oos_win_rate'] = {
        'required': INTRADAY_GATE_CRITERIA['min_oos_win_rate'],
        'actual': round(actual, 4), 'passed': passed,
    }
    if not passed:
        failures += 1

    # OOS profit factor
    actual = agg.get('profit_factor', 0)
    passed = actual >= INTRADAY_GATE_CRITERIA['min_oos_profit_factor']
    criteria_results['min_oos_profit_factor'] = {
        'required': INTRADAY_GATE_CRITERIA['min_oos_profit_factor'],
        'actual': round(actual, 2), 'passed': passed,
    }
    if not passed:
        failures += 1

    # OOS max drawdown
    actual_dd = -abs(agg.get('max_drawdown_pct', 0))
    passed = actual_dd >= INTRADAY_GATE_CRITERIA['max_oos_drawdown']
    criteria_results['max_oos_drawdown'] = {
        'required': INTRADAY_GATE_CRITERIA['max_oos_drawdown'],
        'actual': round(actual_dd, 2), 'passed': passed,
    }
    if not passed:
        failures += 1

    # Overfit ratio
    actual = agg.get('avg_overfit_ratio', 0)
    passed = actual >= INTRADAY_GATE_CRITERIA['min_overfit_ratio']
    criteria_results['min_overfit_ratio'] = {
        'required': INTRADAY_GATE_CRITERIA['min_overfit_ratio'],
        'actual': round(actual, 2), 'passed': passed,
    }
    if not passed:
        failures += 1

    # Parameter stability
    if stability:
        agreements = [v.get('agreement', 0) for v in stability.values()]
        avg_stability = sum(agreements) / len(agreements) if agreements else 0
    else:
        avg_stability = 0
    passed = avg_stability >= INTRADAY_GATE_CRITERIA['min_param_stability']
    criteria_results['min_param_stability'] = {
        'required': INTRADAY_GATE_CRITERIA['min_param_stability'],
        'actual': round(avg_stability, 2), 'passed': passed,
    }
    if not passed:
        failures += 1

    if failures == 0:
        recommendation = 'DEPLOY'
        size_mult = 1.0
    elif failures <= 2:
        recommendation = 'PAPER_TRADE_FIRST'
        size_mult = 0.5
    else:
        recommendation = 'DO_NOT_DEPLOY'
        size_mult = 0.0

    return {
        'passed': failures == 0,
        'failures': failures,
        'criteria': criteria_results,
        'recommendation': recommendation,
        'suggested_size_mult': size_mult,
    }


def run_intraday_walk_forward(
    strategy_id: str,
    symbols: List[str],
    full_start: str,
    full_end: str,
    train_days: int = 504,
    test_days: int = 252,
    step_days: int = 126,
    base_config: GapFadeConfig = None,
    param_grid: dict = None,
    progress_callback: Callable = None,
    cancel_check: Callable = None,
) -> dict:
    """Walk-forward optimization for a single intraday strategy.

    Slides train/test windows, optimizes strategy params in-sample,
    validates out-of-sample. Returns per-strategy results with
    param stability, market condition breakdown, and deployment gate.
    """
    from itertools import product as _product

    config = base_config or GapFadeConfig()
    grid = param_grid or DEFAULT_INTRADAY_WF_GRIDS.get(strategy_id, {})

    if not grid:
        return {'status': 'error', 'error': f'No parameter grid for strategy {strategy_id}'}

    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    _progress(1, f'Loading data for {strategy_id}...')

    # Load daily data
    loader = DataLoader()
    all_data = loader.load_universe(symbols, full_start, full_end, warmup_days=30)
    if all_data.empty:
        return {'status': 'error', 'error': 'No data found in date range'}

    spy_df = loader.load_spy(full_start, full_end)

    _progress(5, 'Building trading date schedule...')

    # Get sorted trading dates
    start_dt = pd.Timestamp(full_start)
    end_dt = pd.Timestamp(full_end)
    date_mask = (all_data.index >= start_dt) & (all_data.index <= end_dt)
    trading_dates = sorted(all_data.index[date_mask].unique())
    total_dates = len(trading_dates)

    if total_dates < train_days + test_days:
        return {
            'status': 'error',
            'error': f'Not enough data: {total_dates} trading days, need {train_days + test_days}',
        }

    # Generate folds
    folds = []
    fold_num = 0
    i = 0
    while i + train_days + test_days <= total_dates:
        fold_num += 1
        train_start = trading_dates[i].strftime('%Y-%m-%d')
        train_end = trading_dates[i + train_days - 1].strftime('%Y-%m-%d')
        test_start = trading_dates[i + train_days].strftime('%Y-%m-%d')
        test_end_idx = min(i + train_days + test_days - 1, total_dates - 1)
        test_end = trading_dates[test_end_idx].strftime('%Y-%m-%d')
        folds.append({
            'fold': fold_num,
            'train_start': train_start, 'train_end': train_end,
            'test_start': test_start, 'test_end': test_end,
        })
        i += step_days

    if not folds:
        return {'status': 'error', 'error': 'Not enough data to generate folds'}

    # Build param combos
    param_names = sorted(grid.keys())
    param_values = [grid[k] for k in param_names]
    combos = list(_product(*param_values))
    total_combos = len(combos)

    _progress(8, f'{len(folds)} folds x {total_combos} combos = {len(folds) * total_combos} trials')

    fold_results = []
    total_work = len(folds) * 2  # precompute + grid per fold
    work_done = 0

    for fold_info in folds:
        if cancel_check and cancel_check():
            break

        fold_n = fold_info['fold']
        train_start = fold_info['train_start']
        train_end = fold_info['train_end']
        test_start = fold_info['test_start']
        test_end = fold_info['test_end']

        _progress(10 + (work_done / total_work) * 80,
                  f'Fold {fold_n}/{len(folds)}: pre-computing bars for {train_start}..{train_end}')

        # Pre-compute bar cache for train window
        train_caches = _precompute_intraday_data(
            symbols, all_data, spy_df, train_start, train_end, cancel_check)

        if not train_caches:
            fold_results.append({
                'fold': fold_n,
                'train_range': f'{train_start} to {train_end}',
                'test_range': f'{test_start} to {test_end}',
                'best_params': {}, 'skipped': True,
                'is_return': 0, 'is_trades': 0, 'is_win_rate': 0,
                'is_max_dd': 0, 'is_sharpe': 0, 'is_pf': 0,
                'oos_return': 0, 'oos_trades': 0, 'oos_win_rate': 0,
                'oos_max_dd': 0, 'oos_sharpe': 0, 'oos_pf': 0,
                'overfit_ratio': 0, 'condition_breakdown': {},
            })
            work_done += 2
            continue

        # Grid search
        _progress(10 + (work_done / total_work) * 80,
                  f'Fold {fold_n}/{len(folds)}: grid search ({total_combos} combos)...')

        best_fitness = -999
        best_params = {}
        best_is_metrics = {}

        for combo in combos:
            if cancel_check and cancel_check():
                break
            overrides = dict(zip(param_names, combo))
            metrics = _simulate_intraday_strategy_fast(
                strategy_id, train_caches, overrides, config)
            if metrics['num_trades'] < 5:
                continue
            fitness = metrics['return_pct'] - metrics['max_drawdown_pct'] * 0.5
            if fitness > best_fitness:
                best_fitness = fitness
                best_params = overrides.copy()
                best_is_metrics = metrics

        work_done += 1

        if not best_params:
            fold_results.append({
                'fold': fold_n,
                'train_range': f'{train_start} to {train_end}',
                'test_range': f'{test_start} to {test_end}',
                'best_params': {}, 'skipped': True,
                'is_return': 0, 'is_trades': 0, 'is_win_rate': 0,
                'is_max_dd': 0, 'is_sharpe': 0, 'is_pf': 0,
                'oos_return': 0, 'oos_trades': 0, 'oos_win_rate': 0,
                'oos_max_dd': 0, 'oos_sharpe': 0, 'oos_pf': 0,
                'overfit_ratio': 0, 'condition_breakdown': {},
            })
            work_done += 1
            continue

        # OOS validation
        _progress(10 + (work_done / total_work) * 80,
                  f'Fold {fold_n}/{len(folds)}: OOS on {test_start}..{test_end}')

        test_caches = _precompute_intraday_data(
            symbols, all_data, spy_df, test_start, test_end, cancel_check)
        oos_metrics = _simulate_intraday_strategy_fast(
            strategy_id, test_caches, best_params, config)

        is_ret = best_is_metrics.get('return_pct', 0)
        oos_ret = oos_metrics.get('return_pct', 0)
        overfit = oos_ret / is_ret if is_ret != 0 else 0

        fold_results.append({
            'fold': fold_n,
            'train_range': f'{train_start} to {train_end}',
            'test_range': f'{test_start} to {test_end}',
            'best_params': best_params,
            'is_return': round(is_ret, 2),
            'is_trades': best_is_metrics.get('num_trades', 0),
            'is_win_rate': round(best_is_metrics.get('win_rate', 0), 4),
            'is_max_dd': round(-best_is_metrics.get('max_drawdown_pct', 0), 2),
            'is_sharpe': round(best_is_metrics.get('sharpe', 0), 2),
            'is_pf': round(best_is_metrics.get('profit_factor', 0), 2),
            'oos_return': round(oos_ret, 2),
            'oos_trades': oos_metrics.get('num_trades', 0),
            'oos_win_rate': round(oos_metrics.get('win_rate', 0), 4),
            'oos_max_dd': round(-oos_metrics.get('max_drawdown_pct', 0), 2),
            'oos_sharpe': round(oos_metrics.get('sharpe', 0), 2),
            'oos_pf': round(oos_metrics.get('profit_factor', 0), 2),
            'overfit_ratio': round(overfit, 2),
            'condition_breakdown': oos_metrics.get('trades_by_condition', {}),
        })
        work_done += 1

    # Aggregate OOS results
    valid_folds = [f for f in fold_results if not f.get('skipped')]
    if valid_folds:
        agg_oos_pnl = sum(f['oos_return'] for f in valid_folds)
        agg_oos_trades = sum(f['oos_trades'] for f in valid_folds)
        agg_oos_wins = sum(f['oos_trades'] * f['oos_win_rate'] for f in valid_folds)
        agg_win_rate = agg_oos_wins / agg_oos_trades if agg_oos_trades > 0 else 0
        agg_pfs = [f['oos_pf'] for f in valid_folds if f['oos_trades'] > 0]
        agg_pf = sum(agg_pfs) / len(agg_pfs) if agg_pfs else 0
        agg_dds = [abs(f['oos_max_dd']) for f in valid_folds]
        agg_max_dd = max(agg_dds) if agg_dds else 0
        agg_sharpes = [f['oos_sharpe'] for f in valid_folds if f['oos_trades'] > 0]
        agg_sharpe = sum(agg_sharpes) / len(agg_sharpes) if agg_sharpes else 0
        agg_overfit = [f['overfit_ratio'] for f in valid_folds if f['overfit_ratio'] != 0]
        avg_overfit = sum(agg_overfit) / len(agg_overfit) if agg_overfit else 0

        aggregate_oos = {
            'total_return_pct': round(agg_oos_pnl, 2),
            'total_pnl': round(agg_oos_pnl * config.initial_capital / 100, 2),
            'num_trades': agg_oos_trades,
            'win_rate': round(agg_win_rate, 4),
            'profit_factor': round(agg_pf, 2),
            'max_drawdown_pct': round(agg_max_dd, 2),
            'sharpe_ratio': round(agg_sharpe, 2),
            'avg_overfit_ratio': round(avg_overfit, 2),
        }
    else:
        aggregate_oos = {
            'total_return_pct': 0, 'total_pnl': 0, 'num_trades': 0,
            'win_rate': 0, 'profit_factor': 0, 'max_drawdown_pct': 0,
            'sharpe_ratio': 0, 'avg_overfit_ratio': 0,
        }

    # Parameter stability
    param_stability = {}
    for pname in param_names:
        values = [f['best_params'].get(pname) for f in valid_folds if f.get('best_params')]
        if values:
            from collections import Counter
            counts = Counter(values)
            most_common_val = counts.most_common(1)[0][0]
            param_stability[pname] = {
                'values': [str(v) for v in values],
                'most_common': most_common_val,
                'agreement': round(counts[most_common_val] / len(values), 2),
            }

    # Market condition breakdown (aggregate across folds)
    all_condition_data = defaultdict(lambda: {'trades': 0, 'wins': 0, 'total_pnl': 0.0})
    for f in valid_folds:
        for cond, cdata in f.get('condition_breakdown', {}).items():
            all_condition_data[cond]['trades'] += cdata.get('trades', 0)
            all_condition_data[cond]['wins'] += int(cdata.get('win_rate', 0) * cdata.get('trades', 0))
            all_condition_data[cond]['total_pnl'] += cdata.get('total_pnl', 0)

    condition_breakdown = {}
    for cond, cdata in all_condition_data.items():
        t = cdata['trades']
        condition_breakdown[cond] = {
            'trades': t,
            'win_rate': round(cdata['wins'] / t, 4) if t > 0 else 0,
            'avg_pnl': round(cdata['total_pnl'] / t, 2) if t > 0 else 0,
            'total_pnl': round(cdata['total_pnl'], 2),
        }

    # Recommendations
    recommendations = []
    n_folds = len(valid_folds)
    if n_folds > 0:
        for pname, pdata in param_stability.items():
            agreement = pdata['agreement']
            mc = pdata['most_common']
            n_same = int(agreement * n_folds)
            if agreement >= 0.75:
                recommendations.append({
                    'type': 'stable', 'level': 'green',
                    'text': f"{pname}={mc} selected in {n_same}/{n_folds} folds — stable",
                })
            elif agreement < 0.5:
                recommendations.append({
                    'type': 'unstable', 'level': 'red',
                    'text': f"{pname} unstable ({len(set(pdata['values']))} different values) — consider fixing at {mc}",
                })
            else:
                recommendations.append({
                    'type': 'moderate', 'level': 'yellow',
                    'text': f"{pname}={mc} selected in {n_same}/{n_folds} folds — moderate stability",
                })

        avg_is_wr = sum(f['is_win_rate'] for f in valid_folds) / n_folds
        avg_oos_wr = sum(f['oos_win_rate'] for f in valid_folds) / n_folds
        wr_deg = avg_is_wr - avg_oos_wr
        if wr_deg < 0.05:
            recommendations.append({
                'type': 'performance', 'level': 'green',
                'text': f"OOS win rate {avg_oos_wr:.0%} vs IS {avg_is_wr:.0%} — minimal degradation",
            })
        elif wr_deg < 0.10:
            recommendations.append({
                'type': 'performance', 'level': 'yellow',
                'text': f"OOS win rate {avg_oos_wr:.0%} vs IS {avg_is_wr:.0%} — moderate degradation",
            })
        else:
            recommendations.append({
                'type': 'performance', 'level': 'red',
                'text': f"OOS win rate {avg_oos_wr:.0%} vs IS {avg_is_wr:.0%} — significant degradation, possible overfit",
            })

        for cond, cdata in condition_breakdown.items():
            if cdata['trades'] >= 5 and cdata['win_rate'] < 0.40:
                recommendations.append({
                    'type': 'condition', 'level': 'red',
                    'text': f"{strategy_id} underperforms in {cond} markets ({cdata['win_rate']:.0%} win rate) — consider disabling",
                })
            elif cdata['trades'] >= 5 and cdata['win_rate'] >= 0.60:
                recommendations.append({
                    'type': 'condition', 'level': 'green',
                    'text': f"{strategy_id} excels in {cond} markets ({cdata['win_rate']:.0%} win rate)",
                })

    _progress(95, 'Running deployment gate...')

    results = {
        'strategy_id': strategy_id,
        'folds': fold_results,
        'aggregate_oos': aggregate_oos,
        'param_stability': param_stability,
        'condition_breakdown': condition_breakdown,
        'recommendations': recommendations,
        'status': 'done',
        'num_folds': len(folds),
        'num_valid_folds': len(valid_folds),
        'param_grid': grid,
        'config_base': {
            'initial_capital': config.initial_capital,
            'slippage_pct': config.slippage_pct,
            'risk_pct': config.intraday_risk_pct,
        },
    }

    results['deployment_gate'] = _check_deployment_gate(results)

    _progress(100, f'Walk-forward complete for {strategy_id}')
    return results
