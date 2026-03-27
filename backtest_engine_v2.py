"""
backtest_engine_v2.py — Institution-grade walk-forward backtesting engine.

Simulates the gap-fade short strategy bar-by-bar on 1-minute data with:
  - Pluggable entry modes: blind (9:31) or exhaustion (9:45+ OR fade)
  - Optional sweep detection and pyramiding
  - ATR-based slippage model
  - Tiered smart-exit trailing stops with reversal detection
  - Adaptive stop sizing (gap% * fraction, clamped)
  - Walk-forward optimization (train/test rolling windows)
  - Monte Carlo block-bootstrap confidence intervals
  - Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
  - Margin enforcement (PDT 4:1)
  - Order rejection simulation
  - Survivorship-bias discount

Usage:
    python backtest_engine_v2.py --entry blind        # 9:31 blind (original)
    python backtest_engine_v2.py --entry exhaustion    # 9:45+ exhaustion (matches live)
    python backtest_engine_v2.py --sweep               # enable sweep detector
    python backtest_engine_v2.py --pyramid             # enable pyramiding
    python backtest_engine_v2.py --entry exhaustion --sweep --pyramid  # full live match
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import warnings

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("backtest_v2")

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestConfig:
    """All tunable parameters — immutable after creation."""

    # Date range
    start_date: str = "2024-03-20"
    end_date: str = "2026-03-20"

    # Capital & sizing
    initial_capital: float = 25_000.0
    risk_pct: float = 0.02
    max_positions: int = 5  # max CONCURRENT (not per day — positions rotate)
    max_notional: float = 50_000.0

    # Gap detection
    gap_threshold: float = 0.03
    max_gap_pct: float = 0.12
    vol_ratio_max: float = 2.0
    min_price: float = 10.0
    min_avg_volume: int = 200_000
    vol_window: int = 20

    # Exclusions
    excluded_symbols: tuple = (
        'SOXL','SOXS','TQQQ','SQQQ','LABU','LABD','SPXL','SPXS','UPRO','SPXU',
        'UDOW','SDOW','TNA','TZA','FNGU','FNGD','TECL','TECS','FAS','FAZ',
        'ERX','ERY','NUGT','DUST','JNUG','JDST','CURE','DRIP','GUSH','UCO',
        'SCO','BOIL','KOLD','UVXY','SVXY','VIXY','VXX','MSTU','MSTX','MSTZ',
        'CONL','GDXU','GDXD','TSLL','TSDD','NVDL','NVDD','AMDL','AMDY','BITO',
        'BITX','AGQ','ZSL','UGL','GLL','UVIX','ETHE','ETHA','ETHB','TETH',
    )

    # Adaptive stop
    stop_gap_fraction: float = 0.25
    stop_min_pct: float = 0.0164
    stop_max_pct: float = 0.025

    # Partial exit
    partial_cover_frac: float = 0.33

    # Time exits
    time_exit_hour: int = 15
    time_exit_min: int = 0
    eod_exit_hour: int = 15
    eod_exit_min: int = 50

    # Market regime filter
    regime_filter: bool = True
    regime_spy_gap_limit: float = 0.01
    regime_spy_block_pct: float = 0.015
    entry_delay_minutes: int = 1       # minutes after 9:30 to enter (1 = 9:31, 15 = 9:45, 30 = 10:00)

    # Slippage model: slippage = alpha + beta * ATR(14)
    slippage_alpha: float = 0.0003   # 0.03% base (limit order offset)
    slippage_beta: float = 0.02      # 2% of ATR (market impact for gap stocks)

    # Execution realism
    rejection_rate: float = 0.05
    random_seed: int = 42

    # Survivorship-bias haircut (annual %)
    survivorship_discount_annual: float = 0.01

    # Event blocks (placeholders — no external calendar wired)
    fomc_block: bool = True
    earnings_block: bool = True

    # Entry mode: 'blind' (9:31 bar open) or 'exhaustion' (9:45+ OR fade)
    entry_mode: str = "exhaustion"

    # Exhaustion-specific settings
    exhaustion_or_minutes: int = 15         # Opening Range: first 15 minutes (9:30-9:44)
    exhaustion_min_fade_pct: float = 0.0    # minimum fade from open to trigger (0 = any fade)

    # Sweep detector
    sweep_enabled: bool = False
    sweep_recovery_bars: int = 5            # max bars for price to recover after sweep

    # Pyramiding
    pyramid_enabled: bool = False
    pyramid_max_adds: int = 2
    pyramid_min_profit_pct: float = 0.03    # +3% before adding
    pyramid_size_decay: float = 0.5         # each add = 50% of previous

    # Database
    db_url: str = "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev"


# ---------------------------------------------------------------------------
# Data structures for positions / trades
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Mutable state for an open position."""

    symbol: str
    direction: str  # 'short' or 'long'
    entry_price: float
    entry_time: str
    shares: int
    remaining_shares: int
    stop_price: float
    prev_close: float
    gap_pct: float
    atr: float
    slippage_entry: float

    # Smart-exit tracking
    high_water_price: float = 0.0
    high_water_pnl_pct: float = 0.0
    partial_filled: bool = False

    # Pyramiding tracking
    pyramid_count: int = 0
    total_cost_basis: float = 0.0  # sum(entry_price * shares) for avg calc


@dataclass(frozen=True)
class TradeRecord:
    """Immutable record of a completed trade."""

    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    shares: int
    entry_time: str
    exit_time: str
    pnl: float
    pnl_pct: float
    exit_reason: str
    gap_pct: float
    slippage_entry: float
    slippage_exit: float
    regime: str


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _connect(db_url: str) -> psycopg2.extensions.connection:
    conn = psycopg2.connect(db_url)
    conn.set_session(readonly=True, autocommit=True)
    return conn


def load_gap_candidates(
    conn: psycopg2.extensions.connection,
    start: str,
    end: str,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """Load daily-bar gap-up candidates via SQL window functions.

    Returns DataFrame with columns:
        symbol, date, open, high, low, close, volume, prev_close,
        prev_volume, avg_vol, gap_pct, vol_ratio
    """
    adj_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=90)).strftime(
        "%Y-%m-%d"
    )
    sql = """
    WITH w AS (
        SELECT symbol, date, open, high, low, close, volume,
               LAG(close)  OVER (PARTITION BY symbol ORDER BY date) AS prev_close,
               LAG(volume) OVER (PARTITION BY symbol ORDER BY date) AS prev_volume,
               AVG(volume) OVER (
                   PARTITION BY symbol ORDER BY date
                   ROWS BETWEEN %s PRECEDING AND 1 PRECEDING
               ) AS avg_vol
        FROM daily_bars
        WHERE date >= %s AND date <= %s
    )
    SELECT symbol, date, open, high, low, close, volume,
           prev_close, prev_volume, avg_vol,
           (open - prev_close) / prev_close AS gap_pct,
           CASE WHEN avg_vol > 0 THEN volume::float / avg_vol ELSE 0 END AS vol_ratio
    FROM w
    WHERE date >= %s
      AND prev_close > 0
      AND open > 0
      AND (open - prev_close) / prev_close >= %s
      AND (open - prev_close) / prev_close <= %s
      AND open >= %s
      AND avg_vol >= %s
    ORDER BY date, symbol
    """
    params = [
        cfg.vol_window,
        adj_start,
        end,
        start,
        cfg.gap_threshold,
        cfg.max_gap_pct,
        cfg.min_price,
        cfg.min_avg_volume,
    ]
    df = pd.read_sql(sql, conn, params=params)
    # Apply exclusion list
    if cfg.excluded_symbols:
        df = df[~df["symbol"].isin(cfg.excluded_symbols)]
    return df


def load_atr_map(
    conn: psycopg2.extensions.connection,
    symbols: Sequence[str],
    date_str: str,
    lookback: int = 14,
) -> Dict[str, float]:
    """Compute ATR(lookback) for each symbol as of *date_str* (exclusive).

    Uses daily_bars true-range calculation.
    """
    if not symbols:
        return {}
    placeholders = ",".join(["%s"] * len(symbols))
    sql = f"""
    WITH ranked AS (
        SELECT symbol, date, high, low, close,
               LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS prev_c,
               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
        FROM daily_bars
        WHERE symbol IN ({placeholders})
          AND date < %s
        ORDER BY symbol, date
    ),
    tr AS (
        SELECT symbol,
               GREATEST(high - low,
                        ABS(high - prev_c),
                        ABS(low  - prev_c)) AS true_range,
               rn
        FROM ranked
        WHERE prev_c IS NOT NULL AND rn <= %s
    )
    SELECT symbol, AVG(true_range) AS atr
    FROM tr
    GROUP BY symbol
    """
    params = list(symbols) + [date_str, lookback]
    cur = conn.cursor()
    cur.execute(sql, params)
    result: Dict[str, float] = {}
    for sym, atr in cur:
        result[sym] = float(atr) if atr else 0.0
    cur.close()
    return result


def load_minute_bars_for_date(
    conn: psycopg2.extensions.connection,
    symbols: Sequence[str],
    date_str: str,
) -> pd.DataFrame:
    """Load RTH 1-min bars for given symbols on one date.

    Returns DataFrame indexed by (symbol, ts) with open/high/low/close/volume.
    """
    if not symbols:
        return pd.DataFrame()
    placeholders = ",".join(["%s"] * len(symbols))
    sql = f"""
    SELECT symbol, ts, open, high, low, close, volume
    FROM minute_bars
    WHERE symbol IN ({placeholders})
      AND ts >= %s
      AND ts < %s
    ORDER BY symbol, ts
    """
    day_start = f"{date_str} 09:30:00"
    day_end = f"{date_str} 16:00:00"
    params = list(symbols) + [day_start, day_end]
    df = pd.read_sql(sql, conn, params=params)
    return df


def load_spy_gap(
    conn: psycopg2.extensions.connection, date_str: str
) -> Optional[float]:
    """Get SPY gap % for regime filter. Returns None if no data."""
    sql = """
    WITH w AS (
        SELECT date, open,
               LAG(close) OVER (ORDER BY date) AS prev_close
        FROM daily_bars
        WHERE symbol = 'SPY' AND date <= %s
        ORDER BY date DESC
        LIMIT 2
    )
    SELECT (open - prev_close) / prev_close
    FROM w
    WHERE date = %s AND prev_close > 0
    """
    cur = conn.cursor()
    cur.execute(sql, [date_str, date_str])
    row = cur.fetchone()
    cur.close()
    return float(row[0]) if row else None


# ---------------------------------------------------------------------------
# Smart Exit logic (mirrors gap_fade_app.py lines 4501–4580)
# ---------------------------------------------------------------------------

def _compute_smart_exit(
    pos: Position,
    price: float,
    bar_time: datetime,
    cfg: BacktestConfig,
) -> Optional[Dict[str, Any]]:
    """Evaluate smart-exit conditions. Mutates pos (stop_price, hwm).

    Returns exit dict if triggered, else None.
    """
    d = pos.direction
    entry = pos.entry_price

    # --- Stop loss ---
    if d == "short" and price >= pos.stop_price:
        return {
            "reason": "stop",
            "shares": pos.remaining_shares,
            "trigger_price": pos.stop_price,
        }
    if d == "long" and price <= pos.stop_price:
        return {
            "reason": "stop",
            "shares": pos.remaining_shares,
            "trigger_price": pos.stop_price,
        }

    # --- Partial target (half-gap fill) ---
    if not pos.partial_filled and d == "short":
        half_target = entry - (entry - pos.prev_close) * 0.5
        if price <= half_target:
            cover_shares = max(1, int(pos.remaining_shares * cfg.partial_cover_frac))
            pos.partial_filled = True
            return {
                "reason": "partial_target",
                "shares": cover_shares,
                "trigger_price": half_target,
            }

    # --- Full target (prev_close) ---
    if d == "short" and price <= pos.prev_close:
        return {
            "reason": "full_target",
            "shares": pos.remaining_shares,
            "trigger_price": pos.prev_close,
        }

    # --- Smart exit: tiered trailing with reversal detection ---
    if entry > 0:
        if d == "short":
            unrealized_pct = (entry - price) / entry
            if pos.high_water_price <= 0 or price < pos.high_water_price:
                pos.high_water_price = price
            hwm_profit = (
                (entry - pos.high_water_price) / entry
                if pos.high_water_price > 0
                else 0.0
            )
            pos.high_water_pnl_pct = max(pos.high_water_pnl_pct, hwm_profit)
        else:
            unrealized_pct = (price - entry) / entry
            if price > pos.high_water_price:
                pos.high_water_price = price
            hwm_profit = (
                (pos.high_water_price - entry) / entry
                if pos.high_water_price > 0
                else 0.0
            )
            pos.high_water_pnl_pct = max(pos.high_water_pnl_pct, hwm_profit)

        hwm_pct = pos.high_water_pnl_pct

        # Determine trail distance by tier
        if hwm_pct >= 0.05:
            trail_pct = 0.002  # Tier 4
        elif hwm_pct >= 0.03:
            trail_pct = 0.003  # Tier 3
        elif hwm_pct >= 0.01:
            trail_pct = 0.005  # Tier 2
        elif hwm_pct >= 0.001:
            trail_pct = None  # Tier 1: breakeven only
        else:
            trail_pct = None  # Tier 0: hold

        # Apply trailing stop adjustment
        if trail_pct is not None and pos.high_water_price > 0:
            if d == "short":
                trail_stop = pos.high_water_price * (1 + trail_pct)
                if trail_stop < pos.stop_price:
                    pos.stop_price = trail_stop
            else:
                trail_stop = pos.high_water_price * (1 - trail_pct)
                if trail_stop > pos.stop_price:
                    pos.stop_price = trail_stop
        elif hwm_pct >= 0.001:
            # Tier 1: move to breakeven
            if d == "short" and entry < pos.stop_price:
                pos.stop_price = entry
            elif d == "long" and entry > pos.stop_price:
                pos.stop_price = entry

        # Reversal detection: retraced 40%+ from HWM while Tier 2+
        if hwm_pct >= 0.01 and pos.high_water_price > 0:
            if d == "short":
                denom = entry - pos.high_water_price
                retracement = (
                    (price - pos.high_water_price) / denom if denom > 0 else 0.0
                )
            else:
                denom = pos.high_water_price - entry
                retracement = (
                    (pos.high_water_price - price) / denom if denom > 0 else 0.0
                )
            if retracement >= 0.40:
                return {
                    "reason": "reversal_detection",
                    "shares": pos.remaining_shares,
                    "trigger_price": price,
                }

    # --- Time exit ---
    bar_mins = bar_time.hour * 60 + bar_time.minute
    time_exit_mins = cfg.time_exit_hour * 60 + cfg.time_exit_min
    eod_mins = cfg.eod_exit_hour * 60 + cfg.eod_exit_min

    if bar_mins >= time_exit_mins:
        return {
            "reason": "time_exit",
            "shares": pos.remaining_shares,
            "trigger_price": price,
        }

    # --- EOD force close ---
    if bar_mins >= eod_mins:
        return {
            "reason": "eod_force",
            "shares": pos.remaining_shares,
            "trigger_price": price,
        }

    return None


# ---------------------------------------------------------------------------
# Core Backtesting Engine
# ---------------------------------------------------------------------------

class BacktestEngineV2:
    """Bar-by-bar 1-minute gap-fade backtester with walk-forward support."""

    def __init__(self, cfg: BacktestConfig) -> None:
        self.cfg = cfg
        self._rng = np.random.RandomState(cfg.random_seed)
        self._conn: Optional[psycopg2.extensions.connection] = None

    # -- connection management -----------------------------------------------

    def _get_conn(self) -> psycopg2.extensions.connection:
        if self._conn is None or self._conn.closed:
            self._conn = _connect(self.cfg.db_url)
        return self._conn

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    # -- slippage model ------------------------------------------------------

    def _slippage(self, price: float, atr: float, direction: str) -> float:
        """Adverse slippage in dollar terms.

        For shorts: slippage is negative (worse entry = lower price received)
        For longs:  slippage is positive (worse entry = higher price paid)
        We return the *amount* to add to the raw price for a worse fill.
        """
        slip_pct = self.cfg.slippage_alpha + self.cfg.slippage_beta * (
            atr / price if price > 0 else 0.0
        )
        return price * slip_pct

    # -- position sizing -----------------------------------------------------

    def _size_position(
        self,
        equity: float,
        entry_price: float,
        stop_distance_pct: float,
        current_notional: float,
    ) -> int:
        """Kelly-based position sizing with margin + notional cap."""
        if stop_distance_pct <= 0 or entry_price <= 0:
            return 0
        risk_dollars = equity * self.cfg.risk_pct
        shares = int(risk_dollars / (entry_price * stop_distance_pct))
        if shares <= 0:
            return 0

        # Notional cap
        notional = shares * entry_price
        if notional > self.cfg.max_notional:
            shares = int(self.cfg.max_notional / entry_price)

        # Margin enforcement (PDT 4:1)
        buying_power = equity * 4.0
        available = buying_power - current_notional
        if available <= 0:
            return 0
        max_shares_margin = int(available / entry_price)
        shares = min(shares, max_shares_margin)

        return max(shares, 0)

    # -- adaptive stop -------------------------------------------------------

    def _adaptive_stop_pct(self, gap_pct: float) -> float:
        raw = abs(gap_pct) * self.cfg.stop_gap_fraction
        return max(self.cfg.stop_min_pct, min(raw, self.cfg.stop_max_pct))

    # -- single-day simulation -----------------------------------------------

    def _simulate_day(
        self,
        date_str: str,
        candidates: pd.DataFrame,
        equity: float,
        positions: List[Position],
        spy_gap: Optional[float],
    ) -> Tuple[List[TradeRecord], float, List[Position]]:
        """Simulate one trading day bar-by-bar.

        Supports two entry modes controlled by cfg.entry_mode:
          - 'blind':      enter at entry_delay bar open (default 9:31)
          - 'exhaustion':  wait until 9:45+, check price < OR high AND price < gap open

        Optional sweep detection (cfg.sweep_enabled) and pyramiding (cfg.pyramid_enabled).

        Args:
            date_str:   date string 'YYYY-MM-DD'
            candidates: rows from gap scan for this date
            equity:     current equity at start of day
            positions:  open positions carried over (should be empty for intraday)
            spy_gap:    SPY gap % for regime filter

        Returns:
            (trades, equity_after, remaining_positions)
        """
        conn = self._get_conn()
        trades: List[TradeRecord] = []

        # Regime classification
        regime = "normal"
        max_pos = self.cfg.max_positions
        if self.cfg.regime_filter and spy_gap is not None:
            if abs(spy_gap) >= self.cfg.regime_spy_block_pct:
                regime = "blocked"
                return trades, equity, positions
            if abs(spy_gap) >= self.cfg.regime_spy_gap_limit:
                regime = "cautious"
                max_pos = max(1, max_pos // 2)

        if candidates.empty:
            return trades, equity, positions

        symbols = candidates["symbol"].tolist()

        # Load ATR for slippage
        atr_map = load_atr_map(conn, symbols, date_str)

        # Load 1-min bars
        minute_df = load_minute_bars_for_date(conn, symbols, date_str)
        if minute_df.empty:
            if len(symbols) > 0:
                log.debug("No minute bars for %d candidates on %s: %s",
                         len(symbols), date_str, symbols[:5])
            return trades, equity, positions

        # Group bars by symbol and apply split adjustment
        bars_by_symbol: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            sym_bars = minute_df[minute_df["symbol"] == sym].copy()
            if not sym_bars.empty:
                sym_bars = sym_bars.sort_values("ts").reset_index(drop=True)
                cand_row = candidates[candidates["symbol"] == sym]
                if not cand_row.empty:
                    daily_open = float(cand_row.iloc[0]["open"])
                    min_open = float(sym_bars.iloc[0]["open"])
                    if min_open > 0 and daily_open > 0:
                        ratio = daily_open / min_open
                        if abs(ratio - 1.0) > 0.05:
                            for col in ("open", "high", "low", "close"):
                                sym_bars[col] = sym_bars[col] * ratio
                bars_by_symbol[sym] = sym_bars

        current_notional = sum(
            p.remaining_shares * p.entry_price for p in positions
        )
        new_positions: List[Position] = list(positions)

        # ═══════════════════════════════════════════════════════════════
        # ENTRY MODE DISPATCH
        # ═══════════════════════════════════════════════════════════════

        if self.cfg.entry_mode == "blind":
            self._enter_blind(
                candidates, bars_by_symbol, atr_map, equity,
                current_notional, max_pos, new_positions,
            )
        else:
            self._enter_exhaustion(
                candidates, bars_by_symbol, atr_map, equity,
                current_notional, max_pos, new_positions,
            )

        # --- Bar-by-bar exit simulation (with optional pyramiding) ---
        closed_indices: set = set()

        all_times = sorted(minute_df["ts"].unique())

        for bar_ts in all_times:
            bar_time = pd.Timestamp(bar_ts).to_pydatetime()
            if bar_time.hour == 9 and bar_time.minute <= 30:
                continue

            # --- Pyramiding check (before exit logic) ---
            if self.cfg.pyramid_enabled:
                self._check_pyramid(
                    bar_ts, bars_by_symbol, atr_map, equity,
                    current_notional, new_positions, closed_indices,
                )

            for idx, pos in enumerate(new_positions):
                if idx in closed_indices:
                    continue
                sym = pos.symbol
                if sym not in bars_by_symbol:
                    continue

                sym_bars = bars_by_symbol[sym]
                bar_rows = sym_bars[sym_bars["ts"] == bar_ts]
                if bar_rows.empty:
                    continue

                bar = bar_rows.iloc[0]
                bar_high = float(bar["high"])
                bar_low = float(bar["low"])
                bar_close = float(bar["close"])

                # Check stop hit intra-bar (use high for shorts)
                intra_price = bar_high if pos.direction == "short" else bar_low

                exit_signal = _compute_smart_exit(pos, intra_price, bar_time, self.cfg)
                if exit_signal is None:
                    # Also check with bar close for non-stop exits
                    exit_signal = _compute_smart_exit(
                        pos, bar_close, bar_time, self.cfg
                    )

                if exit_signal is not None:
                    reason = exit_signal["reason"]
                    exit_shares = exit_signal["shares"]

                    # Exit at bar close + slippage (adverse)
                    raw_exit = bar_close
                    slip_exit = self._slippage(raw_exit, pos.atr, "short")
                    # Covering short: slippage = higher price
                    exit_price = raw_exit + slip_exit

                    if reason == "stop":
                        # Use stop price (already includes some slippage logic)
                        exit_price = pos.stop_price + slip_exit

                    if pos.direction == "short":
                        pnl = (pos.entry_price - exit_price) * exit_shares
                    else:
                        pnl = (exit_price - pos.entry_price) * exit_shares

                    pnl_pct = (
                        (pos.entry_price - exit_price) / pos.entry_price
                        if pos.direction == "short"
                        else (exit_price - pos.entry_price) / pos.entry_price
                    )

                    trade = TradeRecord(
                        symbol=sym,
                        direction=pos.direction,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        shares=exit_shares,
                        entry_time=pos.entry_time,
                        exit_time=str(bar_time),
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=reason,
                        gap_pct=pos.gap_pct,
                        slippage_entry=pos.slippage_entry,
                        slippage_exit=slip_exit,
                        regime=regime,
                    )
                    trades.append(trade)
                    equity += pnl

                    pos.remaining_shares -= exit_shares
                    if pos.remaining_shares <= 0:
                        closed_indices.add(idx)

        # Force-close anything still open (shouldn't happen if EOD works)
        for idx, pos in enumerate(new_positions):
            if idx in closed_indices or pos.remaining_shares <= 0:
                continue
            log.warning(
                "Force-closing unclosed position %s on %s", pos.symbol, date_str
            )
            # Use last available price
            if pos.symbol in bars_by_symbol and not bars_by_symbol[pos.symbol].empty:
                last_bar = bars_by_symbol[pos.symbol].iloc[-1]
                raw_exit = float(last_bar["close"])
            else:
                raw_exit = pos.entry_price

            slip_exit = self._slippage(raw_exit, pos.atr, "short")
            exit_price = raw_exit + slip_exit

            pnl = (
                (pos.entry_price - exit_price) * pos.remaining_shares
                if pos.direction == "short"
                else (exit_price - pos.entry_price) * pos.remaining_shares
            )
            pnl_pct = (
                (pos.entry_price - exit_price) / pos.entry_price
                if pos.direction == "short"
                else (exit_price - pos.entry_price) / pos.entry_price
            )
            trades.append(
                TradeRecord(
                    symbol=pos.symbol,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    shares=pos.remaining_shares,
                    entry_time=pos.entry_time,
                    exit_time=f"{date_str} 15:59:00",
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason="eod_force_residual",
                    gap_pct=pos.gap_pct,
                    slippage_entry=pos.slippage_entry,
                    slippage_exit=slip_exit,
                    regime=regime,
                )
            )
            equity += pnl

        remaining = [
            p for i, p in enumerate(new_positions) if i not in closed_indices and p.remaining_shares > 0
        ]
        return trades, equity, remaining

    # -- blind entry mode ----------------------------------------------------

    def _enter_blind(
        self,
        candidates: pd.DataFrame,
        bars_by_symbol: Dict[str, pd.DataFrame],
        atr_map: Dict[str, float],
        equity: float,
        current_notional: float,
        max_pos: int,
        new_positions: List[Position],
    ) -> None:
        """Blind entry: enter at entry_delay bar's open (default 9:31).

        Simple mode — for each candidate, place order at the bar open
        that corresponds to entry_delay_minutes after 9:30, set adaptive stop.
        """
        entry_minute = 30 + self.cfg.entry_delay_minutes  # e.g. 31 for 1-min delay

        for _, cand in candidates.iterrows():
            if len(new_positions) >= max_pos:
                break

            sym = cand["symbol"]
            if sym not in bars_by_symbol:
                continue

            sym_bars = bars_by_symbol[sym]
            if len(sym_bars) < 2:
                continue

            # Find the entry bar (entry_delay minutes after 9:30)
            entry_bar = None
            for bi in range(len(sym_bars)):
                bar = sym_bars.iloc[bi]
                ts = bar["ts"]
                if isinstance(ts, datetime):
                    if ts.hour == 9 and ts.minute == entry_minute:
                        entry_bar = bar
                        break
                    # If exact minute not found, take first bar after delay
                    if (ts.hour > 9 or ts.minute >= entry_minute) and entry_bar is None:
                        entry_bar = bar
                        break

            if entry_bar is None:
                continue

            raw_entry = float(entry_bar["open"])
            atr = atr_map.get(sym, raw_entry * 0.02)
            slip = self._slippage(raw_entry, atr, "short")
            entry_price = raw_entry - slip

            if entry_price <= 0:
                continue

            # Order rejection
            if self._rng.random() < self.cfg.rejection_rate:
                continue

            gap_pct = float(cand["gap_pct"])
            prev_close = float(cand["prev_close"])

            # Adaptive stop (gap% * fraction, clamped)
            stop_pct = self._adaptive_stop_pct(gap_pct)
            stop_price = entry_price * (1 + stop_pct)

            shares = self._size_position(
                equity, entry_price, stop_pct, current_notional
            )
            if shares <= 0:
                continue

            notional = shares * entry_price
            current_notional += notional

            pos = Position(
                symbol=sym,
                direction="short",
                entry_price=entry_price,
                entry_time=str(entry_bar["ts"]),
                shares=shares,
                remaining_shares=shares,
                stop_price=stop_price,
                prev_close=prev_close,
                gap_pct=gap_pct,
                atr=atr,
                slippage_entry=slip,
                total_cost_basis=entry_price * shares,
            )
            new_positions.append(pos)

    # -- exhaustion entry mode -----------------------------------------------

    def _enter_exhaustion(
        self,
        candidates: pd.DataFrame,
        bars_by_symbol: Dict[str, pd.DataFrame],
        atr_map: Dict[str, float],
        equity: float,
        current_notional: float,
        max_pos: int,
        new_positions: List[Position],
    ) -> None:
        """Exhaustion entry: matches live bot _check_exhaustion_signal() EXACTLY.

        Phase 1 (bars 0-14, 9:30-9:44): Build Opening Range (track high/low).
        Phase 2 (bars 15+, 9:45+): Check each bar:
          - Is current close < OR high? (momentum fading)
          - Is current close < the gap open price? (gap actually fading)
          - If both YES -> enter at NEXT bar's open
          - Stop = OR high * 1.002 (structural)

        Optional sweep detection: if a bar's high goes ABOVE OR high then
        a subsequent bar closes BELOW OR high within sweep_recovery_bars,
        enter on sweep signal with stop above sweep_high.
        """
        or_minutes = self.cfg.exhaustion_or_minutes  # default 15
        or_end_minute = 30 + or_minutes  # e.g. 45

        entry_signals: Dict[str, dict] = {}

        for _, cand in candidates.iterrows():
            sym = cand["symbol"]
            if sym not in bars_by_symbol:
                continue

            sym_bars = bars_by_symbol[sym]
            if len(sym_bars) < or_minutes + 5:
                continue

            gap_open = float(cand["open"])  # the gap open price

            # Phase 1: Build Opening Range (9:30 to 9:30+or_minutes)
            or_bars = sym_bars[sym_bars["ts"].apply(
                lambda t, om=or_end_minute: t.hour == 9 and 30 <= t.minute < om
                if isinstance(t, datetime) else False
            )]
            if or_bars.empty:
                continue

            or_high = float(or_bars["high"].max())
            or_low = float(or_bars["low"].min())

            # Phase 2: Scan bars from or_end_minute onward
            post_or = sym_bars[sym_bars["ts"].apply(
                lambda t, om=or_end_minute: (t.hour > 9 or (t.hour == 9 and t.minute >= om))
                          and (t.hour < self.cfg.time_exit_hour or
                               (t.hour == self.cfg.time_exit_hour and t.minute < self.cfg.time_exit_min))
                if isinstance(t, datetime) else False
            )].reset_index(drop=True)

            if post_or.empty:
                continue

            # Sweep tracking state
            sweep_high: Optional[float] = None
            sweep_bar_count = 0
            triggered = False

            for bi in range(len(post_or)):
                bar = post_or.iloc[bi]
                bar_close = float(bar["close"])
                bar_high = float(bar["high"])

                # ── SWEEP DETECTION ──
                if self.cfg.sweep_enabled:
                    if bar_high > or_high:
                        # Price swept above OR high
                        if sweep_high is None or bar_high > sweep_high:
                            sweep_high = bar_high
                        sweep_bar_count = 0  # reset recovery counter
                    elif sweep_high is not None:
                        sweep_bar_count += 1
                        # Check if price recovered (closed back below OR high)
                        if bar_close < or_high and sweep_bar_count <= self.cfg.sweep_recovery_bars:
                            # Sweep signal: enter at next bar
                            entry_signals[sym] = {
                                "cand": cand,
                                "entry_bar_idx": bi + 1,
                                "entry_bars": post_or,
                                "or_high": or_high,
                                "or_low": or_low,
                                "stop_ref": sweep_high,  # stop above sweep high
                                "signal_type": "sweep",
                            }
                            triggered = True
                            break
                        if sweep_bar_count > self.cfg.sweep_recovery_bars:
                            sweep_high = None  # reset, too many bars elapsed

                # ── EXHAUSTION SIGNAL CHECK (matches live bot EXACTLY) ──
                # Condition 1: Price below OR high
                below_or_high = bar_close < or_high

                # Condition 2: Price below the gap open (gap is fading)
                price_fading = bar_close < gap_open

                # Optional minimum fade filter
                if self.cfg.exhaustion_min_fade_pct > 0:
                    fade_pct = (gap_open - bar_close) / gap_open if gap_open > 0 else 0.0
                    if fade_pct < self.cfg.exhaustion_min_fade_pct:
                        price_fading = False

                if below_or_high and price_fading:
                    # Enter at NEXT bar's open
                    entry_signals[sym] = {
                        "cand": cand,
                        "entry_bar_idx": bi + 1,
                        "entry_bars": post_or,
                        "or_high": or_high,
                        "or_low": or_low,
                        "stop_ref": or_high,  # structural stop ref
                        "signal_type": "exhaustion",
                    }
                    triggered = True
                    break

        # Phase 3: Execute entries from signals
        for sym, sig in entry_signals.items():
            if len(new_positions) >= max_pos:
                break

            cand = sig["cand"]
            post_or = sig["entry_bars"]
            entry_idx = sig["entry_bar_idx"]
            stop_ref = sig["stop_ref"]

            if entry_idx >= len(post_or):
                continue

            entry_bar = post_or.iloc[entry_idx]
            raw_entry = float(entry_bar["open"])
            atr = atr_map.get(sym, raw_entry * 0.02)
            slip = self._slippage(raw_entry, atr, "short")
            entry_price = raw_entry - slip

            if entry_price <= 0:
                continue

            # Order rejection
            if self._rng.random() < self.cfg.rejection_rate:
                continue

            gap_pct = float(cand["gap_pct"])
            prev_close = float(cand["prev_close"])

            # STRUCTURAL STOP: 0.20% above stop reference
            stop_price = stop_ref * 1.002
            stop_pct = (stop_price - entry_price) / entry_price if entry_price > 0 else 0.03

            # Minimum stop distance
            if stop_pct < 0.005:
                stop_pct = 0.005
                stop_price = entry_price * (1 + stop_pct)

            shares = self._size_position(
                equity, entry_price, stop_pct, current_notional
            )
            if shares <= 0:
                continue

            notional = shares * entry_price
            current_notional += notional

            pos = Position(
                symbol=sym,
                direction="short",
                entry_price=entry_price,
                entry_time=str(entry_bar["ts"]),
                shares=shares,
                remaining_shares=shares,
                stop_price=stop_price,
                prev_close=prev_close,
                gap_pct=gap_pct,
                atr=atr,
                slippage_entry=slip,
                total_cost_basis=entry_price * shares,
            )
            new_positions.append(pos)

    # -- pyramiding ----------------------------------------------------------

    def _check_pyramid(
        self,
        bar_ts: Any,
        bars_by_symbol: Dict[str, pd.DataFrame],
        atr_map: Dict[str, float],
        equity: float,
        current_notional: float,
        positions: List[Position],
        closed_indices: set,
    ) -> None:
        """Check if any open position qualifies for pyramid add.

        Adds shares if position is >= pyramid_min_profit_pct profitable
        and pyramid_count < pyramid_max_adds. Each add is sized at
        original_shares * pyramid_size_decay^(count+1). Stop moves to
        breakeven of combined position.
        """
        for idx, pos in enumerate(positions):
            if idx in closed_indices:
                continue
            if pos.pyramid_count >= self.cfg.pyramid_max_adds:
                continue

            sym = pos.symbol
            if sym not in bars_by_symbol:
                continue

            sym_bars = bars_by_symbol[sym]
            bar_rows = sym_bars[sym_bars["ts"] == bar_ts]
            if bar_rows.empty:
                continue

            bar = bar_rows.iloc[0]
            bar_close = float(bar["close"])

            # Calculate unrealized P&L %
            if pos.direction == "short":
                unrealized_pct = (pos.entry_price - bar_close) / pos.entry_price
            else:
                unrealized_pct = (bar_close - pos.entry_price) / pos.entry_price

            if unrealized_pct < self.cfg.pyramid_min_profit_pct:
                continue

            # Calculate add size with decay
            decay = self.cfg.pyramid_size_decay ** (pos.pyramid_count + 1)
            add_shares = max(1, int(pos.shares * decay))

            # Slippage on add
            atr = atr_map.get(sym, bar_close * 0.02)
            slip = self._slippage(bar_close, atr, pos.direction)
            if pos.direction == "short":
                add_price = bar_close - slip
            else:
                add_price = bar_close + slip

            if add_price <= 0:
                continue

            # Update position: add shares, recalculate avg entry
            old_cost = pos.total_cost_basis if pos.total_cost_basis > 0 else pos.entry_price * pos.remaining_shares
            new_cost = old_cost + add_price * add_shares
            new_total_shares = pos.remaining_shares + add_shares

            avg_entry = new_cost / new_total_shares if new_total_shares > 0 else pos.entry_price

            pos.remaining_shares = new_total_shares
            pos.total_cost_basis = new_cost
            pos.pyramid_count += 1

            # Move stop to breakeven of combined position
            if pos.direction == "short":
                pos.stop_price = avg_entry * 1.002  # tiny buffer above breakeven
            else:
                pos.stop_price = avg_entry * 0.998

            # Update entry_price to weighted average for P&L calcs
            pos.entry_price = avg_entry

    # -- full backtest -------------------------------------------------------

    def run(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        initial_capital: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute full backtest over date range.

        Returns dict with trades, equity_curve, summary stats.
        """
        start = start or self.cfg.start_date
        end = end or self.cfg.end_date
        capital = initial_capital or self.cfg.initial_capital

        log.info(
            "Starting backtest %s -> %s  capital=$%.0f", start, end, capital
        )
        t0 = time.time()

        conn = self._get_conn()

        # Load all gap candidates at once
        all_candidates = load_gap_candidates(conn, start, end, self.cfg)
        if all_candidates.empty:
            log.warning("No gap candidates found in date range")
            return self._empty_result(capital)

        trading_dates = sorted(all_candidates["date"].unique())
        log.info(
            "Found %d gap events across %d trading dates",
            len(all_candidates),
            len(trading_dates),
        )

        equity = capital
        all_trades: List[TradeRecord] = []
        equity_curve: List[Dict[str, Any]] = []
        positions: List[Position] = []
        daily_pnl_list: List[float] = []
        peak_equity = capital
        max_drawdown = 0.0
        margin_rejections = 0

        for date_str in trading_dates:
            day_candidates = all_candidates[all_candidates["date"] == date_str]

            # Vol ratio filter
            if "vol_ratio" in day_candidates.columns:
                day_candidates = day_candidates[
                    day_candidates["vol_ratio"] <= self.cfg.vol_ratio_max
                ]

            if day_candidates.empty:
                equity_curve.append(
                    {"date": date_str, "equity": equity, "trades": 0, "pnl": 0.0}
                )
                daily_pnl_list.append(0.0)
                continue

            # SPY regime
            spy_gap = load_spy_gap(conn, date_str) if self.cfg.regime_filter else None

            equity_before = equity
            day_trades, equity, positions = self._simulate_day(
                date_str, day_candidates, equity, positions, spy_gap
            )

            day_pnl = equity - equity_before
            daily_pnl_list.append(day_pnl)
            all_trades.extend(day_trades)

            # Drawdown tracking
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            max_drawdown = max(max_drawdown, dd)

            equity_curve.append(
                {
                    "date": date_str,
                    "equity": equity,
                    "trades": len(day_trades),
                    "pnl": day_pnl,
                    "drawdown": dd,
                }
            )

        elapsed = time.time() - t0

        # Survivorship-bias discount
        years = len(trading_dates) / 252.0
        surv_factor = (1 - self.cfg.survivorship_discount_annual) ** years
        adj_equity = capital + (equity - capital) * surv_factor

        # Compute summary
        summary = self._compute_summary(
            all_trades, daily_pnl_list, equity_curve, capital, equity, adj_equity,
            max_drawdown, elapsed
        )

        return {
            "config": asdict(self.cfg),
            "trades": [asdict(t) for t in all_trades],
            "equity_curve": equity_curve,
            "summary": summary,
        }

    def _compute_summary(
        self,
        trades: List[TradeRecord],
        daily_pnl: List[float],
        equity_curve: List[Dict],
        initial: float,
        final: float,
        adj_final: float,
        max_dd: float,
        elapsed: float,
    ) -> Dict[str, Any]:
        """Compute summary statistics from trades and equity curve."""
        n = len(trades)
        if n == 0:
            return {"total_trades": 0, "elapsed_seconds": elapsed}

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades)
        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0

        # Daily returns for Sharpe
        daily_returns = np.array(daily_pnl)
        nonzero_days = daily_returns[daily_returns != 0]

        sharpe = 0.0
        if len(nonzero_days) > 1:
            mean_r = np.mean(nonzero_days)
            std_r = np.std(nonzero_days, ddof=1)
            if std_r > 0:
                sharpe = (mean_r / std_r) * np.sqrt(252)

        # Exit reason breakdown
        exit_reasons: Dict[str, int] = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        # Slippage cost
        total_slippage = sum(
            (t.slippage_entry + t.slippage_exit) * t.shares for t in trades
        )

        years = len(equity_curve) / 252.0 if equity_curve else 1.0
        ratio = final / initial if initial > 0 else 0.0
        cagr = (
            (ratio ** (1.0 / years) - 1.0) if years > 0 and ratio > 0 else 0.0
        )

        return {
            "total_trades": n,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / n if n > 0 else 0.0,
            "total_pnl": round(total_pnl, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": (
                round(float(gross_profit / abs(gross_loss)), 3) if abs(gross_loss) > 0 else float("inf")
            ),
            "avg_trade_pnl": round(total_pnl / n, 2),
            "avg_win": round(gross_profit / len(wins), 2) if wins else 0.0,
            "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
            "max_drawdown": round(max_dd, 4),
            "sharpe_ratio": round(sharpe, 3),
            "cagr": round(cagr, 4),
            "final_equity": round(final, 2),
            "adj_equity_surv": round(adj_final, 2),
            "initial_capital": round(initial, 2),
            "total_slippage_cost": round(total_slippage, 2),
            "exit_reasons": exit_reasons,
            "trading_days": len(equity_curve),
            "elapsed_seconds": round(elapsed, 2),
        }

    def _empty_result(self, capital: float) -> Dict[str, Any]:
        return {
            "config": asdict(self.cfg),
            "trades": [],
            "equity_curve": [],
            "summary": {"total_trades": 0, "final_equity": capital},
        }

    # -----------------------------------------------------------------------
    # Walk-Forward Optimization
    # -----------------------------------------------------------------------

    def walk_forward(
        self,
        train_months: int = 12,
        test_months: int = 3,
        roll_months: int = 1,
    ) -> Dict[str, Any]:
        """Rolling walk-forward optimization.

        For each window:
          1) Train: grid-search params on in-sample, maximize Sharpe
          2) Test:  run best params on out-of-sample

        Returns aggregate OOS results and per-window breakdown.
        """
        log.info(
            "Walk-forward: train=%dm, test=%dm, roll=%dm",
            train_months,
            test_months,
            roll_months,
        )

        start_dt = datetime.strptime(self.cfg.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.cfg.end_date, "%Y-%m-%d")

        windows: List[Dict[str, Any]] = []
        oos_trades: List[TradeRecord] = []
        oos_equity_curve: List[Dict] = []
        equity = self.cfg.initial_capital

        cursor = start_dt
        window_num = 0

        while True:
            train_start = cursor
            train_end = train_start + timedelta(days=train_months * 30)
            test_start = train_end
            test_end = test_start + timedelta(days=test_months * 30)

            if test_end > end_dt:
                break

            window_num += 1
            log.info(
                "Window %d: train %s..%s  test %s..%s",
                window_num,
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            )

            # -- In-sample optimization --
            best_params, best_sharpe = self._optimize_params(
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
            )

            # -- Out-of-sample run with best params --
            test_cfg = replace(
                self.cfg,
                start_date=test_start.strftime("%Y-%m-%d"),
                end_date=test_end.strftime("%Y-%m-%d"),
                initial_capital=equity,
                **best_params,
            )
            test_engine = BacktestEngineV2(test_cfg)
            test_result = test_engine.run(initial_capital=equity)
            test_engine.close()

            window_trades = [
                TradeRecord(**t) for t in test_result["trades"]
            ]
            oos_trades.extend(window_trades)
            oos_equity_curve.extend(test_result["equity_curve"])
            equity = test_result["summary"].get("final_equity", equity)

            windows.append(
                {
                    "window": window_num,
                    "train_start": train_start.strftime("%Y-%m-%d"),
                    "train_end": train_end.strftime("%Y-%m-%d"),
                    "test_start": test_start.strftime("%Y-%m-%d"),
                    "test_end": test_end.strftime("%Y-%m-%d"),
                    "best_params": best_params,
                    "is_sharpe": round(best_sharpe, 3),
                    "oos_summary": test_result["summary"],
                }
            )

            cursor += timedelta(days=roll_months * 30)

        # Aggregate OOS summary
        total_pnl = sum(t.pnl for t in oos_trades)
        daily_pnl = [ec["pnl"] for ec in oos_equity_curve]

        agg_summary = self._compute_summary(
            oos_trades,
            daily_pnl,
            oos_equity_curve,
            self.cfg.initial_capital,
            equity,
            equity,
            0.0,
            0.0,
        )

        return {
            "windows": windows,
            "oos_trades": [asdict(t) for t in oos_trades],
            "oos_equity_curve": oos_equity_curve,
            "oos_summary": agg_summary,
        }

    def _optimize_params(
        self, start: str, end: str
    ) -> Tuple[Dict[str, Any], float]:
        """Grid search over key parameters, maximize Sharpe ratio.

        Returns (best_param_dict, best_sharpe).
        """
        # Coarse grid to keep runtime reasonable
        param_grid = {
            "gap_threshold": [0.04, 0.05, 0.07],
            "max_gap_pct": [0.10, 0.12, 0.15],
            "stop_min_pct": [0.012, 0.0164, 0.02],
            "stop_max_pct": [0.02, 0.025, 0.03],
            "risk_pct": [0.015, 0.02, 0.025],
        }

        keys = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in keys]))

        # Sub-sample to cap at 50 combinations (randomly if needed)
        if len(combos) > 50:
            indices = self._rng.choice(len(combos), 50, replace=False)
            combos = [combos[i] for i in indices]

        log.info("Optimizing over %d parameter combinations", len(combos))

        best_sharpe = -999.0
        best_params: Dict[str, Any] = {}

        for combo in combos:
            overrides = dict(zip(keys, combo))
            trial_cfg = replace(
                self.cfg,
                start_date=start,
                end_date=end,
                **overrides,
            )
            engine = BacktestEngineV2(trial_cfg)
            try:
                result = engine.run()
                sharpe = result["summary"].get("sharpe_ratio", -999.0)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = overrides
            except Exception as exc:
                log.warning("Trial failed: %s — %s", overrides, exc)
            finally:
                engine.close()

        if not best_params:
            # Fallback to defaults
            best_params = {k: getattr(self.cfg, k) for k in keys}

        log.info("Best IS Sharpe=%.3f  params=%s", best_sharpe, best_params)
        return best_params, best_sharpe

    # -----------------------------------------------------------------------
    # Monte Carlo Simulation
    # -----------------------------------------------------------------------

    def monte_carlo(
        self,
        trades: List[Dict[str, Any]],
        n_simulations: int = 10_000,
        block_size: int = 5,
    ) -> Dict[str, Any]:
        """Block-bootstrap Monte Carlo simulation.

        Resamples blocks of consecutive trading days, computes distribution
        of P&L, drawdown, and Sharpe.
        """
        if not trades:
            return {"error": "No trades to simulate"}

        log.info(
            "Monte Carlo: %d sims, block_size=%d", n_simulations, block_size
        )

        # Group trades by date
        trades_by_date: Dict[str, List[float]] = {}
        for t in trades:
            d = t["entry_time"][:10]
            trades_by_date.setdefault(d, []).append(t["pnl"])

        dates = sorted(trades_by_date.keys())
        daily_pnl = np.array([sum(trades_by_date[d]) for d in dates])
        n_days = len(daily_pnl)

        if n_days < block_size:
            return {"error": f"Not enough trading days ({n_days}) for block size {block_size}"}

        # Create blocks
        n_blocks = n_days - block_size + 1
        blocks = np.array(
            [daily_pnl[i : i + block_size] for i in range(n_blocks)]
        )

        # Number of blocks needed to cover original length
        blocks_needed = int(np.ceil(n_days / block_size))

        rng = np.random.RandomState(self.cfg.random_seed)

        pnl_dist = np.zeros(n_simulations)
        dd_dist = np.zeros(n_simulations)
        sharpe_dist = np.zeros(n_simulations)

        initial = self.cfg.initial_capital

        for i in range(n_simulations):
            # Sample blocks with replacement
            chosen = rng.randint(0, n_blocks, size=blocks_needed)
            sim_pnl = np.concatenate([blocks[c] for c in chosen])[:n_days]

            # Compute equity curve
            cum_pnl = np.cumsum(sim_pnl)
            equity = initial + cum_pnl
            peak = np.maximum.accumulate(equity)
            drawdowns = (peak - equity) / np.where(peak > 0, peak, 1.0)

            pnl_dist[i] = cum_pnl[-1]
            dd_dist[i] = np.max(drawdowns)

            # Sharpe
            if np.std(sim_pnl) > 0:
                sharpe_dist[i] = (np.mean(sim_pnl) / np.std(sim_pnl)) * np.sqrt(252)
            else:
                sharpe_dist[i] = 0.0

        return {
            "n_simulations": n_simulations,
            "block_size": block_size,
            "original_days": n_days,
            "pnl": {
                "median": round(float(np.median(pnl_dist)), 2),
                "p5": round(float(np.percentile(pnl_dist, 5)), 2),
                "p25": round(float(np.percentile(pnl_dist, 25)), 2),
                "p75": round(float(np.percentile(pnl_dist, 75)), 2),
                "p95": round(float(np.percentile(pnl_dist, 95)), 2),
                "mean": round(float(np.mean(pnl_dist)), 2),
            },
            "max_drawdown": {
                "median": round(float(np.median(dd_dist)), 4),
                "p5": round(float(np.percentile(dd_dist, 5)), 4),
                "p95": round(float(np.percentile(dd_dist, 95)), 4),
            },
            "sharpe": {
                "median": round(float(np.median(sharpe_dist)), 3),
                "p5": round(float(np.percentile(sharpe_dist, 5)), 3),
                "p95": round(float(np.percentile(sharpe_dist, 95)), 3),
            },
            "prob_loss": round(float(np.mean(pnl_dist < 0)), 4),
            "prob_dd_gt_20pct": round(float(np.mean(dd_dist > 0.20)), 4),
        }

    # -----------------------------------------------------------------------
    # Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
    # -----------------------------------------------------------------------

    @staticmethod
    def deflated_sharpe(
        observed_sharpe: float,
        n_observations: int,
        skewness: float,
        kurtosis: float,
        n_tests: int = 20,
    ) -> Dict[str, float]:
        """Compute Deflated Sharpe Ratio.

        DSR adjusts the observed Sharpe for multiple testing, non-normality,
        and sample length.

        Args:
            observed_sharpe: Annualized Sharpe from backtest
            n_observations:  Number of daily returns
            skewness:        Skewness of daily returns
            kurtosis:        Excess kurtosis of daily returns
            n_tests:         Number of strategy variants tested

        Returns:
            dict with dsr, expected_max_sharpe, p_value
        """
        from scipy import stats as sp_stats

        if n_observations < 2:
            return {"dsr": 0.0, "expected_max_sharpe": 0.0, "p_value": 1.0}

        # Work with annualized Sharpe throughout (consistent scale)
        sr = observed_sharpe

        # Expected maximum Sharpe under null (Euler-Mascheroni)
        # E[max(SR)] for n_tests independent trials of SR ~ N(0, se)
        euler_mascheroni = 0.5772156649
        if n_tests > 1:
            log_n = np.log(n_tests)
            e_max_sr = np.sqrt(2 * log_n) - (
                (np.log(np.pi) + euler_mascheroni)
                / (2 * np.sqrt(2 * log_n))
            )
        else:
            e_max_sr = 0.0

        # Standard error of annualized Sharpe accounting for non-normality
        # Bailey & Lopez de Prado eq. (4): SE(SR) incorporates skew/kurtosis
        sr_daily = sr / np.sqrt(252)
        se_daily_sq = (
            (1 - skewness * sr_daily + ((kurtosis - 1) / 4) * sr_daily ** 2)
            / (n_observations - 1)
        )
        # Annualize the SE: SE(SR_annual) = SE(SR_daily) * sqrt(252)
        sr_std = np.sqrt(max(se_daily_sq, 0.0)) * np.sqrt(252)

        if sr_std <= 0:
            return {"dsr": 0.0, "expected_max_sharpe": round(float(e_max_sr), 3), "p_value": 1.0}

        # DSR test statistic: is observed SR significantly above E[max(SR)]?
        z = (sr - e_max_sr) / sr_std
        p_value = 1 - sp_stats.norm.cdf(z)

        e_max_sr_annual = e_max_sr

        return {
            "dsr": round(float(1 - p_value), 4),
            "expected_max_sharpe": round(float(e_max_sr_annual), 3),
            "p_value": round(float(p_value), 4),
            "z_score": round(float(z), 3),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(result: Dict[str, Any]) -> None:
    """Pretty-print backtest results."""
    s = result.get("summary", {})
    if not s:
        print("No results.")
        return

    cfg = result.get("config", {})
    entry_mode = cfg.get("entry_mode", "exhaustion")
    sweep_on = cfg.get("sweep_enabled", False)
    pyramid_on = cfg.get("pyramid_enabled", False)

    mode_label = "blind (9:31 bar open)" if entry_mode == "blind" else "exhaustion (9:45+ OR fade)"

    print("\n" + "=" * 65)
    print("  BACKTEST RESULTS — Gap Fade Short Strategy (1-min bars)")
    print(f"  Entry Mode: {mode_label}")
    print(f"  Sweep: {'enabled' if sweep_on else 'disabled'}")
    print(f"  Pyramid: {'enabled' if pyramid_on else 'disabled'}")
    print("=" * 65)

    fmt = "  {:<30s} {}"
    print(fmt.format("Total Trades:", s.get("total_trades", 0)))
    print(fmt.format("Win Rate:", f"{s.get('win_rate', 0):.1%}"))
    print(fmt.format("Profit Factor:", s.get("profit_factor", 0)))
    print(fmt.format("Total P&L:", f"${s.get('total_pnl', 0):,.2f}"))
    print(fmt.format("Avg Trade:", f"${s.get('avg_trade_pnl', 0):,.2f}"))
    print(fmt.format("Avg Win:", f"${s.get('avg_win', 0):,.2f}"))
    print(fmt.format("Avg Loss:", f"${s.get('avg_loss', 0):,.2f}"))
    print(fmt.format("Max Drawdown:", f"{s.get('max_drawdown', 0):.2%}"))
    print(fmt.format("Sharpe Ratio:", s.get("sharpe_ratio", 0)))
    print(fmt.format("CAGR:", f"{s.get('cagr', 0):.2%}"))
    print(fmt.format("Final Equity:", f"${s.get('final_equity', 0):,.2f}"))
    print(
        fmt.format(
            "Adj Equity (Surv):", f"${s.get('adj_equity_surv', 0):,.2f}"
        )
    )
    print(
        fmt.format("Total Slippage Cost:", f"${s.get('total_slippage_cost', 0):,.2f}")
    )
    print(fmt.format("Trading Days:", s.get("trading_days", 0)))
    print(fmt.format("Elapsed:", f"{s.get('elapsed_seconds', 0):.1f}s"))

    exits = s.get("exit_reasons", {})
    if exits:
        print("\n  Exit Reasons:")
        for reason, count in sorted(exits.items(), key=lambda x: -x[1]):
            print(f"    {reason:<25s} {count}")

    print("=" * 65)


def _print_monte_carlo(mc: Dict[str, Any]) -> None:
    print("\n" + "-" * 55)
    print("  MONTE CARLO SIMULATION")
    print("-" * 55)
    fmt = "  {:<30s} {}"
    pnl = mc.get("pnl", {})
    dd = mc.get("max_drawdown", {})
    sh = mc.get("sharpe", {})
    print(fmt.format("Simulations:", mc.get("n_simulations", 0)))
    print(fmt.format("P&L Median:", f"${pnl.get('median', 0):,.2f}"))
    print(fmt.format("P&L 5th pct:", f"${pnl.get('p5', 0):,.2f}"))
    print(fmt.format("P&L 95th pct:", f"${pnl.get('p95', 0):,.2f}"))
    print(fmt.format("Max DD Median:", f"{dd.get('median', 0):.2%}"))
    print(fmt.format("Max DD 95th pct:", f"{dd.get('p95', 0):.2%}"))
    print(fmt.format("Sharpe Median:", sh.get("median", 0)))
    print(fmt.format("P(Loss):", f"{mc.get('prob_loss', 0):.1%}"))
    print(fmt.format("P(DD > 20%):", f"{mc.get('prob_dd_gt_20pct', 0):.1%}"))
    print("-" * 55)


def _print_deflated_sharpe(dsr: Dict[str, float]) -> None:
    print("\n" + "-" * 55)
    print("  DEFLATED SHARPE RATIO")
    print("-" * 55)
    fmt = "  {:<30s} {}"
    print(fmt.format("DSR:", f"{dsr.get('dsr', 0):.4f}"))
    print(fmt.format("Expected Max SR:", dsr.get("expected_max_sharpe", 0)))
    print(fmt.format("p-value:", dsr.get("p_value", 0)))
    print(fmt.format("z-score:", dsr.get("z_score", 0)))
    print("-" * 55)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gap Fade Walk-Forward Backtester v2")
    parser.add_argument("--start", default="2024-03-20", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-03-20", help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=25_000, help="Initial capital")
    parser.add_argument(
        "--entry", choices=["blind", "exhaustion"], default="exhaustion",
        help="Entry mode: blind (9:31 bar open) or exhaustion (9:45+ OR fade, matches live bot)"
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Enable sweep detector (exhaustion mode)"
    )
    parser.add_argument(
        "--pyramid", action="store_true", help="Enable pyramiding into winners"
    )
    parser.add_argument(
        "--walk-forward", action="store_true", help="Run walk-forward optimization"
    )
    parser.add_argument(
        "--monte-carlo", action="store_true", help="Run Monte Carlo simulation"
    )
    parser.add_argument(
        "--dsr", action="store_true", help="Compute Deflated Sharpe Ratio"
    )
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    parser.add_argument(
        "--output", type=str, default=None, help="Save results to JSON file"
    )
    args = parser.parse_args()

    cfg = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        entry_mode=args.entry,
        sweep_enabled=args.sweep,
        pyramid_enabled=args.pyramid,
    )
    engine = BacktestEngineV2(cfg)

    try:
        # --- Standard backtest ---
        result = engine.run()
        _print_summary(result)

        all_results: Dict[str, Any] = {"backtest": result}

        # --- Walk-forward ---
        if args.walk_forward or args.all:
            wf = engine.walk_forward()
            all_results["walk_forward"] = wf
            print("\n  Walk-Forward Windows:")
            for w in wf["windows"]:
                oos = w["oos_summary"]
                print(
                    f"    Window {w['window']}: "
                    f"IS Sharpe={w['is_sharpe']:.3f}  "
                    f"OOS trades={oos.get('total_trades', 0)}  "
                    f"OOS P&L=${oos.get('total_pnl', 0):,.2f}  "
                    f"OOS Sharpe={oos.get('sharpe_ratio', 0):.3f}"
                )
            print(f"\n  Aggregate OOS: {wf['oos_summary']}")

        # --- Monte Carlo ---
        if args.monte_carlo or args.all:
            mc = engine.monte_carlo(result["trades"])
            all_results["monte_carlo"] = mc
            _print_monte_carlo(mc)

        # --- Deflated Sharpe ---
        if args.dsr or args.all:
            sharpe = result["summary"].get("sharpe_ratio", 0)
            n_obs = result["summary"].get("trading_days", 0)
            # Compute skew/kurtosis from daily equity changes
            daily_pnl = np.array(
                [ec.get("pnl", 0) for ec in result["equity_curve"]]
            )
            nonzero = daily_pnl[daily_pnl != 0]
            if len(nonzero) > 2:
                skew = float(pd.Series(nonzero).skew())
                kurt = float(pd.Series(nonzero).kurtosis())
            else:
                skew, kurt = 0.0, 3.0

            dsr = BacktestEngineV2.deflated_sharpe(
                observed_sharpe=sharpe,
                n_observations=n_obs,
                skewness=skew,
                kurtosis=kurt,
                n_tests=20,
            )
            all_results["deflated_sharpe"] = dsr
            _print_deflated_sharpe(dsr)

        # --- Save ---
        if args.output:
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            log.info("Results saved to %s", args.output)

    finally:
        engine.close()


if __name__ == "__main__":
    main()
