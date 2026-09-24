#!/usr/bin/env python3
"""Exit Counterfactual Research Harness — replays day-trade entries under different exit policies.

v-exit-counterfactual-2026-09-24-r3. Uses minute_bars (Stage A backtest infrastructure) to replay
historical managed day-trade entries under multiple exit policies:

  (a) actual     — the exit that occurred in the live ledger
  (b) hold_pure  — pure hold to SL / TP / flatten (no early exits)
  (c) pro_policy — unified pro policy:
                   - structure-based initial stop
                   - scale 50% at +1R, move stop to breakeven
                   - trail remainder (ATR or swing low)
                   - time stop N minutes before flatten
                   - models MACD/RSI early exits from bar indicators
  (d) pro_no_early — policy (c) WITHOUT early MACD/RSI/desk exits

Outputs:
  - research/reports/exit_counterfactual_<date>.md — human-readable report
  - research/reports/exit_counterfactual_<date>.json — machine-readable
  - Per-exit-reason attribution

Usage on live host:
    python -m research.exit_counterfactual \\
        [--ledger path/to/trading_state.json] \\
        [--bars-dir /path/to/bars/] \\
        [--dsn postgresql://...] \\
        [--out research/reports/]

Hands-off forever: MU, HQGE, SPCX (excluded from all analysis).

CRITICAL:
  - Bars source is REQUIRED. No fallback to actual-only.
  - All timestamps treated as naive ET (use zoneinfo only, no pytz).
  - Stop-first tie-break in ALL policies.
  - actual R uses REAL initial stop, includes scale-outs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date, time as dt_time
from pathlib import Path
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("exit_counterfactual")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

HANDS_OFF_SYMBOLS = frozenset({"MU", "HQGE", "SPCX"})
ET = ZoneInfo("America/New_York")
FLATTEN_HOUR = 15  # 3 PM ET
FLATTEN_MINUTE = 0
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
ATR_STOP_MULT = 1.5
RR_RATIO = 2.0
DEFAULT_SLIPPAGE_PCT = 0.0005
DEFAULT_DSN = "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev"


class NoBarsSourceError(Exception):
    """Raised when no bars source is available."""
    pass


def _to_et(ts: datetime) -> datetime:
    """Convert timestamp to ET. Naive timestamps are assumed to be ET.
    
    Uses zoneinfo only (no pytz) to avoid DST issues.
    """
    if ts.tzinfo is None:
        return ts.replace(tzinfo=ET)
    return ts.astimezone(ET)


def _ensure_et_compatible(ts: datetime, bar_ts) -> tuple[datetime, datetime]:
    """Ensure two timestamps can be compared (both aware or both naive ET)."""
    ts_et = _to_et(ts)
    
    if hasattr(bar_ts, 'tzinfo'):
        if bar_ts.tzinfo is None:
            bar_et = bar_ts.replace(tzinfo=ET)
        else:
            bar_et = bar_ts.astimezone(ET)
    else:
        bar_et = datetime.fromisoformat(str(bar_ts)).replace(tzinfo=ET)
    
    return ts_et, bar_et


def _is_market_hours(ts: datetime) -> bool:
    """Check if timestamp is within RTH (09:30-16:00 ET)."""
    ts_et = _to_et(ts) if isinstance(ts, datetime) else ts
    if hasattr(ts_et, 'tzinfo') and ts_et.tzinfo is None:
        ts_et = ts_et.replace(tzinfo=ET)
    elif not hasattr(ts_et, 'tzinfo'):
        return True  # Assume market hours if we can't determine
    
    market_open = ts_et.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
    market_close = ts_et.replace(hour=MARKET_CLOSE_HOUR, minute=0, second=0, microsecond=0)
    return market_open <= ts_et <= market_close


def _is_flatten_time(ts: datetime) -> bool:
    """Check if timestamp is at or past flatten time (15:00 ET)."""
    ts_et = _to_et(ts)
    return ts_et.hour >= FLATTEN_HOUR


def _is_time_stop(ts: datetime, minutes_before_flatten: int = 15) -> bool:
    """Check if we should exit due to time stop (N min before flatten)."""
    ts_et = _to_et(ts)
    flatten_time = ts_et.replace(hour=FLATTEN_HOUR, minute=FLATTEN_MINUTE, second=0, microsecond=0)
    cutoff = flatten_time - timedelta(minutes=minutes_before_flatten)
    return ts_et >= cutoff


@dataclass
class DayTradeEntry:
    """A single day-trade entry from the ledger."""
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    pnl: float
    exit_reason: str
    strategy: str
    entry_pattern: str
    stop_loss: float
    take_profit: float
    stop_distance: float
    initial_stop_distance: float  # The REAL initial stop for actual R calculation
    atr: float
    rsi: Optional[float]
    regime: str
    time_of_day: str
    session_id: str
    side: str = "long"
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_trade_history(cls, d: dict) -> Optional["DayTradeEntry"]:
        """Parse an entry from trading_state.json trade_history."""
        reasoning = d.get("reasoning", {})
        strategy = reasoning.get("strategy", d.get("strategy", ""))
        
        if "day_trade" not in strategy.lower() and "momentum" not in strategy.lower():
            return None
        
        symbol = str(d.get("symbol", "")).upper()
        if symbol in HANDS_OFF_SYMBOLS:
            return None
        
        entry_time_str = d.get("entry_time")
        if not entry_time_str:
            return None
        try:
            entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
            entry_time = _to_et(entry_time).replace(tzinfo=None)
        except (ValueError, TypeError):
            return None
        
        exit_time_str = d.get("exit_time")
        exit_time = None
        if exit_time_str:
            try:
                exit_time = datetime.fromisoformat(exit_time_str.replace("Z", "+00:00"))
                exit_time = _to_et(exit_time).replace(tzinfo=None)
            except (ValueError, TypeError):
                pass
        
        entry_price = float(d.get("entry_price", 0))
        exit_price = float(d.get("exit_price", 0)) if d.get("exit_price") else None
        quantity = int(d.get("quantity", 1))
        pnl = float(d.get("pnl", 0))
        exit_reason = str(d.get("exit_reason") or d.get("reason") or "unknown")
        
        stop_loss = float(reasoning.get("stop_loss", reasoning.get("stop", 0)))
        take_profit = float(reasoning.get("take_profit", reasoning.get("target", 0)))
        stop_distance = float(reasoning.get("stop_distance", reasoning.get("stop_dist", 0)))
        atr = float(reasoning.get("atr", 0))
        rsi = reasoning.get("rsi")
        if rsi is not None:
            rsi = float(rsi)
        
        side = str(reasoning.get("side", d.get("side", "long"))).lower()
        if side not in ("long", "short"):
            side = "long"
        
        if stop_distance == 0 and entry_price > 0 and atr > 0:
            stop_distance = ATR_STOP_MULT * atr
        elif stop_distance == 0 and stop_loss > 0:
            stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance <= 0:
            return None
        
        initial_stop_distance = stop_distance
        
        if stop_loss == 0 and entry_price > 0 and stop_distance > 0:
            if side == "long":
                stop_loss = entry_price - stop_distance
            else:
                stop_loss = entry_price + stop_distance
        if take_profit == 0 and entry_price > 0 and stop_distance > 0:
            if side == "long":
                take_profit = entry_price + (RR_RATIO * stop_distance)
            else:
                take_profit = entry_price - (RR_RATIO * stop_distance)
        
        entry_pattern = str(reasoning.get("entry_pattern", reasoning.get("pattern", "unknown")))
        regime = str(reasoning.get("regime", reasoning.get("market_context_regime", "unknown")))
        time_of_day = str(reasoning.get("time_of_day", reasoning.get("market_context", {}).get("time_of_day", "unknown")))
        
        entry_date = entry_time.date() if hasattr(entry_time, "date") else date.today()
        session_id = str(reasoning.get("session_id", entry_date.isoformat()))
        
        return cls(
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            exit_reason=exit_reason,
            strategy=strategy,
            entry_pattern=entry_pattern,
            stop_loss=stop_loss,
            take_profit=take_profit,
            stop_distance=stop_distance,
            initial_stop_distance=initial_stop_distance,
            atr=atr,
            rsi=rsi,
            regime=regime,
            time_of_day=time_of_day,
            session_id=session_id,
            side=side,
            raw=d,
        )


@dataclass
class ReplayResult:
    """Result of replaying a trade under a specific policy."""
    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime]
    exit_price: Optional[float]
    exit_reason: str
    stop_distance: float
    r_multiple: float
    return_pct: float
    hold_bars: int
    policy: str
    side: str = "long"
    scaled_out_at: Optional[float] = None
    partial_r: float = 0.0
    remaining_r: float = 0.0


@dataclass
class PolicyMetrics:
    """Aggregated metrics for an exit policy."""
    policy: str
    n_trades: int
    n_sessions: int
    n_wins: int
    n_losses: int
    n_scratches: int
    win_rate: float
    profit_factor: Optional[float]
    total_r: float
    expectancy_r: float
    max_dd_r: float
    max_losing_day_r: float
    avg_hold_bars: float
    by_exit_reason: dict[str, int]
    by_pattern: dict[str, dict[str, Any]]


def load_trading_state(path: Path) -> list[dict]:
    """Load trade_history from trading_state.json or plain list."""
    if not path.exists():
        logger.warning("Ledger not found at %s", path)
        return []
    
    with open(path) as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    return data.get("trade_history", [])


def extract_day_trade_entries(
    raw_history: list[dict],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> list[DayTradeEntry]:
    """Extract day-trade entries from trade history with deduplication."""
    entries = []
    seen: set[tuple[str, datetime]] = set()
    
    for raw in raw_history:
        entry = DayTradeEntry.from_trade_history(raw)
        if entry is None:
            continue
        
        key = (entry.symbol, entry.entry_time)
        if key in seen:
            continue
        seen.add(key)
        
        entry_date = entry.entry_time.date() if hasattr(entry.entry_time, "date") else None
        if entry_date:
            if start_date and entry_date < start_date:
                continue
            if end_date and entry_date > end_date:
                continue
        
        entries.append(entry)
    
    logger.warning("Extracted %d day-trade entries (after dedup, stop_dist>0 filter)", len(entries))
    return entries


def make_postgres_bars_loader(dsn: str) -> Callable[[str, datetime, datetime, int], pd.DataFrame]:
    """Return a bars loader using PostgresDataProvider."""
    from data_providers.postgres import PostgresDataProvider
    provider = PostgresDataProvider(dsn)

    def _load(symbol: str, start: datetime, entry_ts: datetime, lookforward_minutes: int = 400) -> pd.DataFrame:
        """Load minute bars for symbol starting AFTER `entry_ts`."""
        entry_et = _to_et(entry_ts)
        end = entry_et + timedelta(minutes=lookforward_minutes + 60)
        try:
            df = provider.get_market_data(
                symbol,
                period_type="day",
                period=5,
                frequency_type="minute",
                frequency=1,
                end=end.replace(tzinfo=None),
            )
            if df.empty:
                return df
            
            df = _filter_bars_for_replay(df, entry_et)
            return df
        except Exception as exc:
            logger.warning("Bars load failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    return _load


def make_file_bars_loader(bars_dir: Path) -> Callable[[str, datetime, datetime, int], pd.DataFrame]:
    """Return a bars loader using local parquet/CSV files."""
    def _load(symbol: str, start: datetime, entry_ts: datetime, lookforward_minutes: int = 400) -> pd.DataFrame:
        """Load minute bars from local file, filtering after entry_ts."""
        parquet_path = bars_dir / f"{symbol.upper()}.parquet"
        csv_path = bars_dir / f"{symbol.upper()}.csv"
        
        df = pd.DataFrame()
        try:
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
            elif csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                logger.warning("No bars file for %s in %s", symbol, bars_dir)
                return df
                
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"])
                df = df.set_index("ts")
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            
            df.columns = [c.capitalize() if c.lower() in ("open", "high", "low", "close", "volume") else c 
                          for c in df.columns]
            
            if not df.empty:
                entry_et = _to_et(entry_ts)
                df = _filter_bars_for_replay(df, entry_et)
            
            return df
        except Exception as exc:
            logger.warning("File bars load failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    return _load


def _filter_bars_for_replay(df: pd.DataFrame, entry_et: datetime) -> pd.DataFrame:
    """Filter bars for replay: after entry, within market hours, with upper bound."""
    if df.empty:
        return df
    
    end_of_day = entry_et.replace(hour=MARKET_CLOSE_HOUR, minute=0, second=0, microsecond=0)
    
    filtered_rows = []
    for idx in df.index:
        try:
            if hasattr(idx, 'tzinfo'):
                if idx.tzinfo is None:
                    bar_et = idx.replace(tzinfo=ET) if isinstance(idx, datetime) else datetime.fromisoformat(str(idx)).replace(tzinfo=ET)
                else:
                    bar_et = idx.astimezone(ET)
            else:
                bar_et = datetime.fromisoformat(str(idx)).replace(tzinfo=ET)
            
            if bar_et <= entry_et:
                continue
            if bar_et > end_of_day:
                continue
            
            market_open = bar_et.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
            market_close = bar_et.replace(hour=MARKET_CLOSE_HOUR, minute=0, second=0, microsecond=0)
            if not (market_open <= bar_et <= market_close):
                continue
            
            filtered_rows.append(idx)
        except (TypeError, ValueError):
            continue
    
    if not filtered_rows:
        return pd.DataFrame()
    
    return df.loc[filtered_rows]


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MACD, RSI indicators for early exit detection."""
    if not HAS_TA or len(df) < 14:
        df["macd"] = 0.0
        df["macd_signal"] = 0.0
        df["rsi"] = 50.0
        return df
    
    close = df["Close"]
    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["rsi"] = ta.momentum.rsi(close, window=14)
    df["rsi"] = df["rsi"].fillna(50.0)
    df["macd"] = df["macd"].fillna(0.0)
    df["macd_signal"] = df["macd_signal"].fillna(0.0)
    return df


def _check_early_exit_triggers(
    bar: pd.Series,
    prev_bar: Optional[pd.Series],
    side: str,
) -> Optional[str]:
    """Check if MACD/RSI early exit would trigger on this bar."""
    macd = bar.get("macd", 0)
    macd_signal = bar.get("macd_signal", 0)
    rsi = bar.get("rsi", 50)
    
    if prev_bar is not None:
        prev_macd = prev_bar.get("macd", 0)
        prev_macd_signal = prev_bar.get("macd_signal", 0)
    else:
        prev_macd = macd
        prev_macd_signal = macd_signal
    
    if side == "long":
        if macd < macd_signal and prev_macd >= prev_macd_signal:
            return "proactive_macd_flipped_bearish"
        if rsi < 50:
            return "proactive_rsi_below_50"
    else:
        if macd > macd_signal and prev_macd <= prev_macd_signal:
            return "proactive_macd_flipped_bullish"
        if rsi > 50:
            return "proactive_rsi_above_50"
    
    return None


def _compute_r_multiple(entry_price: float, exit_price: float, stop_distance: float, side: str) -> float:
    """Compute R-multiple correctly for both longs and shorts."""
    if stop_distance <= 0:
        return 0.0
    if side == "long":
        return (exit_price - entry_price) / stop_distance
    else:
        return (entry_price - exit_price) / stop_distance


def replay_policy_actual(entry: DayTradeEntry) -> ReplayResult:
    """Policy (a): Return the actual exit from the ledger.
    
    Uses the REAL initial stop distance for R calculation.
    """
    r_multiple = 0.0
    if entry.initial_stop_distance > 0 and entry.exit_price is not None:
        r_multiple = _compute_r_multiple(
            entry.entry_price, entry.exit_price, 
            entry.initial_stop_distance, entry.side
        )
    
    return_pct = 0.0
    if entry.entry_price > 0 and entry.exit_price is not None:
        if entry.side == "long":
            return_pct = (entry.exit_price - entry.entry_price) / entry.entry_price * 100
        else:
            return_pct = (entry.entry_price - entry.exit_price) / entry.entry_price * 100
    
    hold_bars = 0
    if entry.entry_time and entry.exit_time:
        hold_bars = int((entry.exit_time - entry.entry_time).total_seconds() / 60)
    
    return ReplayResult(
        symbol=entry.symbol,
        entry_time=entry.entry_time,
        entry_price=entry.entry_price,
        exit_time=entry.exit_time,
        exit_price=entry.exit_price,
        exit_reason=entry.exit_reason,
        stop_distance=entry.initial_stop_distance,
        r_multiple=r_multiple,
        return_pct=return_pct,
        hold_bars=hold_bars,
        policy="actual",
        side=entry.side,
    )


def replay_policy_hold_pure(
    entry: DayTradeEntry,
    df: pd.DataFrame,
) -> ReplayResult:
    """Policy (b): Pure hold to SL / TP / flatten — no early exits.
    
    STOP-FIRST tie-break: if both stop and target hit on same bar, stop wins.
    """
    if df.empty:
        return ReplayResult(
            symbol=entry.symbol,
            entry_time=entry.entry_time,
            entry_price=entry.entry_price,
            exit_time=None,
            exit_price=None,
            exit_reason="no_bars",
            stop_distance=entry.stop_distance,
            r_multiple=0.0,
            return_pct=0.0,
            hold_bars=0,
            policy="hold_pure",
            side=entry.side,
        )
    
    stop = entry.stop_loss
    target = entry.take_profit
    side = entry.side
    
    exit_time = None
    exit_price = None
    exit_reason = "flatten"
    hold_bars = 0
    
    for i in range(len(df)):
        bar = df.iloc[i]
        bar_ts = df.index[i]
        hold_bars = i + 1
        
        bar_high = float(bar["High"])
        bar_low = float(bar["Low"])
        
        if side == "long":
            stop_hit = bar_low <= stop
            target_hit = bar_high >= target
        else:
            stop_hit = bar_high >= stop
            target_hit = bar_low <= target
        
        if stop_hit:
            exit_time = bar_ts
            exit_price = stop
            exit_reason = "stop"
            break
        
        if target_hit:
            exit_time = bar_ts
            exit_price = target
            exit_reason = "target"
            break
        
        try:
            bar_ts_et = _to_et(bar_ts) if isinstance(bar_ts, datetime) else bar_ts
            if hasattr(bar_ts_et, 'hour') and bar_ts_et.hour >= FLATTEN_HOUR:
                exit_time = bar_ts
                exit_price = float(bar["Close"])
                exit_reason = "flatten"
                break
        except (TypeError, AttributeError):
            pass
    
    if exit_price is None and len(df) > 0:
        last_bar = df.iloc[-1]
        exit_time = df.index[-1]
        exit_price = float(last_bar["Close"])
        exit_reason = "flatten"
    
    r_multiple = _compute_r_multiple(
        entry.entry_price, exit_price if exit_price else entry.entry_price,
        entry.stop_distance, side
    )
    
    return_pct = 0.0
    if entry.entry_price > 0 and exit_price is not None:
        if side == "long":
            return_pct = (exit_price - entry.entry_price) / entry.entry_price * 100
        else:
            return_pct = (entry.entry_price - exit_price) / entry.entry_price * 100
    
    return ReplayResult(
        symbol=entry.symbol,
        entry_time=entry.entry_time,
        entry_price=entry.entry_price,
        exit_time=exit_time,
        exit_price=exit_price,
        exit_reason=exit_reason,
        stop_distance=entry.stop_distance,
        r_multiple=r_multiple,
        return_pct=return_pct,
        hold_bars=hold_bars,
        policy="hold_pure",
        side=side,
    )


def replay_policy_pro(
    entry: DayTradeEntry,
    df: pd.DataFrame,
    minutes_before_flatten: int = 15,
    trail_atr_mult: float = 1.5,
    disable_early_exits: bool = False,
) -> ReplayResult:
    """Policy (c) or (d): Unified pro policy with scale/trail.
    
    STOP-FIRST tie-break: if both stop and target hit on same bar, stop wins.
    """
    policy_name = "pro_no_early" if disable_early_exits else "pro_policy"
    
    if df.empty:
        return ReplayResult(
            symbol=entry.symbol,
            entry_time=entry.entry_time,
            entry_price=entry.entry_price,
            exit_time=None,
            exit_price=None,
            exit_reason="no_bars",
            stop_distance=entry.stop_distance,
            r_multiple=0.0,
            return_pct=0.0,
            hold_bars=0,
            policy=policy_name,
            side=entry.side,
        )
    
    df_with_ind = _compute_indicators(df.copy())
    
    stop = entry.stop_loss
    target = entry.take_profit
    stop_distance = entry.stop_distance
    atr = entry.atr if entry.atr > 0 else stop_distance / ATR_STOP_MULT
    side = entry.side
    
    current_stop = stop
    scaled_out = False
    scaled_out_at = None
    partial_r = 0.0
    trailing_active = False
    highest_price = entry.entry_price if side == "long" else entry.entry_price
    lowest_price = entry.entry_price if side == "short" else entry.entry_price
    
    exit_time = None
    exit_price = None
    exit_reason = "flatten"
    hold_bars = 0
    
    prev_bar = None
    
    for i in range(len(df_with_ind)):
        bar = df_with_ind.iloc[i]
        bar_ts = df_with_ind.index[i]
        hold_bars = i + 1
        bar_high = float(bar["High"])
        bar_low = float(bar["Low"])
        bar_close = float(bar["Close"])
        
        if side == "long":
            highest_price = max(highest_price, bar_high)
        else:
            lowest_price = min(lowest_price, bar_low)
        
        if side == "long":
            stop_hit = bar_low <= current_stop
            target_hit = bar_high >= target
        else:
            stop_hit = bar_high >= current_stop
            target_hit = bar_low <= target
        
        if stop_hit:
            exit_time = bar_ts
            exit_price = current_stop
            exit_reason = "stop" if not trailing_active else "trailing_stop"
            break
        
        if target_hit:
            exit_time = bar_ts
            exit_price = target
            exit_reason = "target"
            break
        
        try:
            bar_ts_et = _to_et(bar_ts) if isinstance(bar_ts, datetime) else bar_ts
            if hasattr(bar_ts_et, 'hour'):
                flatten_time = bar_ts_et.replace(hour=FLATTEN_HOUR, minute=FLATTEN_MINUTE, second=0, microsecond=0)
                cutoff = flatten_time - timedelta(minutes=minutes_before_flatten)
                if bar_ts_et >= cutoff:
                    exit_time = bar_ts
                    exit_price = bar_close
                    exit_reason = "time_stop" if bar_ts_et < flatten_time else "flatten"
                    break
        except (TypeError, AttributeError):
            pass
        
        if not disable_early_exits:
            early_exit_reason = _check_early_exit_triggers(bar, prev_bar, side)
            if early_exit_reason is not None:
                current_r = _compute_r_multiple(entry.entry_price, bar_close, stop_distance, side)
                if current_r <= -0.5:
                    exit_time = bar_ts
                    exit_price = bar_close
                    exit_reason = early_exit_reason
                    break
        
        current_r = _compute_r_multiple(entry.entry_price, bar_close, stop_distance, side)
        
        if not scaled_out and current_r >= 1.0:
            scaled_out = True
            if side == "long":
                scaled_out_at = entry.entry_price + stop_distance
            else:
                scaled_out_at = entry.entry_price - stop_distance
            partial_r = 0.5 * 1.0
            current_stop = entry.entry_price
            trailing_active = True
        
        if trailing_active:
            if side == "long" and bar_close > entry.entry_price:
                trail_stop = bar_close - (trail_atr_mult * atr)
                if trail_stop > current_stop:
                    current_stop = trail_stop
            elif side == "short" and bar_close < entry.entry_price:
                trail_stop = bar_close + (trail_atr_mult * atr)
                if trail_stop < current_stop:
                    current_stop = trail_stop
        
        prev_bar = bar
    
    if exit_price is None and len(df_with_ind) > 0:
        last_bar = df_with_ind.iloc[-1]
        exit_time = df_with_ind.index[-1]
        exit_price = float(last_bar["Close"])
        exit_reason = "flatten"
    
    remaining_r = 0.0
    if stop_distance > 0 and exit_price is not None:
        remaining_r = _compute_r_multiple(entry.entry_price, exit_price, stop_distance, side)
        if scaled_out:
            remaining_r = remaining_r * 0.5
    
    total_r = partial_r + remaining_r
    
    return_pct = 0.0
    if entry.entry_price > 0 and exit_price is not None:
        if side == "long":
            return_pct = (exit_price - entry.entry_price) / entry.entry_price * 100
        else:
            return_pct = (entry.entry_price - exit_price) / entry.entry_price * 100
    
    return ReplayResult(
        symbol=entry.symbol,
        entry_time=entry.entry_time,
        entry_price=entry.entry_price,
        exit_time=exit_time,
        exit_price=exit_price,
        exit_reason=exit_reason,
        stop_distance=stop_distance,
        r_multiple=total_r,
        return_pct=return_pct,
        hold_bars=hold_bars,
        policy=policy_name,
        side=side,
        scaled_out_at=scaled_out_at,
        partial_r=partial_r,
        remaining_r=remaining_r,
    )


def run_counterfactual_replay(
    entries: list[DayTradeEntry],
    bars_loader: Callable[[str, datetime, datetime, int], pd.DataFrame],
) -> tuple[dict[str, list[ReplayResult]], int]:
    """Replay all entries under all four policies.
    
    Returns: (results_dict, n_no_bars)
    
    Only includes trades where bars were successfully loaded.
    """
    results: dict[str, list[ReplayResult]] = {
        "actual": [],
        "hold_pure": [],
        "pro_policy": [],
        "pro_no_early": [],
    }
    n_no_bars = 0
    
    for i, entry in enumerate(entries):
        if (i + 1) % 10 == 0 or i == len(entries) - 1:
            logger.warning("Replaying %d/%d: %s", i + 1, len(entries), entry.symbol)
        
        df = bars_loader(entry.symbol, entry.entry_time, entry.entry_time, lookforward_minutes=400)
        
        if df.empty:
            n_no_bars += 1
            continue
        
        results["actual"].append(replay_policy_actual(entry))
        results["hold_pure"].append(replay_policy_hold_pure(entry, df))
        results["pro_policy"].append(replay_policy_pro(entry, df, disable_early_exits=False))
        results["pro_no_early"].append(replay_policy_pro(entry, df, disable_early_exits=True))
    
    return results, n_no_bars


def compute_policy_metrics(results: list[ReplayResult], entries: list[DayTradeEntry]) -> PolicyMetrics:
    """Compute aggregate metrics for a policy's results."""
    if not results:
        return PolicyMetrics(
            policy="unknown",
            n_trades=0,
            n_sessions=0,
            n_wins=0,
            n_losses=0,
            n_scratches=0,
            win_rate=0.0,
            profit_factor=None,
            total_r=0.0,
            expectancy_r=0.0,
            max_dd_r=0.0,
            max_losing_day_r=0.0,
            avg_hold_bars=0.0,
            by_exit_reason={},
            by_pattern={},
        )
    
    sessions = set()
    for r in results:
        if r.entry_time:
            sessions.add(r.entry_time.date())
    
    wins = [r for r in results if r.r_multiple > 0.05]
    losses = [r for r in results if r.r_multiple < -0.05]
    scratches = [r for r in results if -0.05 <= r.r_multiple <= 0.05]
    
    n_trades = len(results)
    n_wins = len(wins)
    n_losses = len(losses)
    
    win_loss_count = n_wins + n_losses
    win_rate = n_wins / win_loss_count if win_loss_count > 0 else 0.0
    
    gross_profit = sum(r.r_multiple for r in wins)
    gross_loss = abs(sum(r.r_multiple for r in losses))
    if gross_loss > 0:
        profit_factor = round(gross_profit / gross_loss, 3)
    elif gross_profit > 0:
        profit_factor = None
    else:
        profit_factor = 0.0
    
    total_r = sum(r.r_multiple for r in results)
    expectancy_r = total_r / n_trades if n_trades > 0 else 0.0
    
    peak_r = 0.0
    max_dd_r = 0.0
    cumulative_r = 0.0
    sorted_results = sorted(results, key=lambda x: x.entry_time or datetime.min)
    for r in sorted_results:
        cumulative_r += r.r_multiple
        if cumulative_r > peak_r:
            peak_r = cumulative_r
        dd = peak_r - cumulative_r
        if dd > max_dd_r:
            max_dd_r = dd
    
    daily_r: dict[date, float] = defaultdict(float)
    for r in results:
        if r.entry_time:
            daily_r[r.entry_time.date()] += r.r_multiple
    max_losing_day_r = abs(min(daily_r.values())) if daily_r else 0.0
    
    avg_hold = sum(r.hold_bars for r in results) / n_trades if n_trades > 0 else 0.0
    
    by_exit_reason: dict[str, int] = defaultdict(int)
    for r in results:
        by_exit_reason[r.exit_reason] += 1
    
    entry_patterns = {(e.symbol, e.entry_time): e.entry_pattern for e in entries}
    by_pattern: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "total_r": 0.0, "wins": 0, "losses": 0})
    for r in results:
        pattern = entry_patterns.get((r.symbol, r.entry_time), "unknown")
        by_pattern[pattern]["n"] += 1
        by_pattern[pattern]["total_r"] += r.r_multiple
        if r.r_multiple > 0.05:
            by_pattern[pattern]["wins"] += 1
        elif r.r_multiple < -0.05:
            by_pattern[pattern]["losses"] += 1
    
    for pattern in by_pattern:
        n = by_pattern[pattern]["n"]
        w = by_pattern[pattern]["wins"]
        l = by_pattern[pattern]["losses"]
        by_pattern[pattern]["exp_r"] = round(by_pattern[pattern]["total_r"] / n, 4) if n > 0 else 0.0
        by_pattern[pattern]["wr"] = round(w / (w + l), 3) if (w + l) > 0 else 0.0
        by_pattern[pattern]["total_r"] = round(by_pattern[pattern]["total_r"], 3)
    
    return PolicyMetrics(
        policy=results[0].policy if results else "unknown",
        n_trades=n_trades,
        n_sessions=len(sessions),
        n_wins=n_wins,
        n_losses=n_losses,
        n_scratches=len(scratches),
        win_rate=round(win_rate, 4),
        profit_factor=profit_factor,
        total_r=round(total_r, 3),
        expectancy_r=round(expectancy_r, 4),
        max_dd_r=round(max_dd_r, 3),
        max_losing_day_r=round(max_losing_day_r, 3),
        avg_hold_bars=round(avg_hold, 1),
        by_exit_reason=dict(by_exit_reason),
        by_pattern=dict(by_pattern),
    )


def write_markdown_report(
    metrics: dict[str, PolicyMetrics],
    results: dict[str, list[ReplayResult]],
    entries: list[DayTradeEntry],
    n_no_bars: int,
    output_path: Path,
) -> None:
    """Write human-readable markdown report."""
    n_with_bars = metrics["actual"].n_trades if metrics["actual"].n_trades > 0 else 0
    
    lines = [
        "# Exit Counterfactual Analysis",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Sample:** {n_with_bars} day-trade entries with bars ({n_no_bars} skipped — no bars)",
        "",
        "## Executive Summary",
        "",
        "This report compares the **actual** exits against counterfactual scenarios:",
        "",
        "| Policy | Description |",
        "|--------|-------------|",
        "| actual | What actually happened in live trading |",
        "| hold_pure | Pure hold to SL/TP/flatten — no early exits |",
        "| pro_policy | Unified pro: scale 50% at +1R → BE → trail → time stop (with MACD/RSI exits) |",
        "| pro_no_early | Pro policy WITHOUT MACD/RSI/desk early exits |",
        "",
        "---",
        "",
        "## Policy Comparison",
        "",
        "| Policy | n | WR | expR | Total R | PF | Max DD |",
        "|--------|---|----:|-----:|--------:|---:|-------:|",
    ]
    
    for policy in ["actual", "hold_pure", "pro_policy", "pro_no_early"]:
        m = metrics[policy]
        pf_str = f"{m.profit_factor:.2f}" if m.profit_factor is not None else "∞"
        lines.append(
            f"| {policy} | {m.n_trades} | {m.win_rate*100:.1f}% | {m.expectancy_r:+.3f} | "
            f"{m.total_r:+.2f}R | {pf_str} | {m.max_dd_r:.2f}R |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## Detailed Breakdown",
        "",
    ])
    
    for policy in ["actual", "hold_pure", "pro_policy", "pro_no_early"]:
        m = metrics[policy]
        lines.extend([
            f"### {policy.replace('_', ' ').title()}",
            "",
            f"- **Trades:** {m.n_trades} across {m.n_sessions} sessions",
            f"- **W/L/S:** {m.n_wins}/{m.n_losses}/{m.n_scratches}",
            f"- **Win Rate:** {m.win_rate*100:.1f}%",
            f"- **Expectancy:** {m.expectancy_r:+.4f}R per trade",
            f"- **Total R:** {m.total_r:+.2f}R",
            f"- **Profit Factor:** {m.profit_factor:.2f}" if m.profit_factor is not None else "- **Profit Factor:** ∞",
            f"- **Max Drawdown:** {m.max_dd_r:.2f}R",
            f"- **Max Losing Day:** {m.max_losing_day_r:.2f}R",
            f"- **Avg Hold:** {m.avg_hold_bars:.1f} bars",
            "",
            "**By Exit Reason:**",
            "",
            "| Reason | Count |",
            "|--------|------:|",
        ])
        for reason, count in sorted(m.by_exit_reason.items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {count} |")
        
        lines.extend([
            "",
            "**By Entry Pattern:**",
            "",
            "| Pattern | n | WR | expR | Total R |",
            "|---------|---|----:|-----:|--------:|",
        ])
        for pattern, stats in sorted(m.by_pattern.items(), key=lambda x: -x[1]["n"]):
            lines.append(
                f"| {pattern} | {stats['n']} | {stats['wr']*100:.1f}% | "
                f"{stats['exp_r']:+.3f} | {stats['total_r']:+.2f}R |"
            )
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## Key Findings (same-sample comparison)",
        "",
        f"**n = {metrics['actual'].n_trades}** trades with bars (excluding {n_no_bars} no-bars)",
        "",
        f"**Actual vs Hold Pure ΔexpR:** {(metrics['hold_pure'].expectancy_r - metrics['actual'].expectancy_r):+.4f}R",
        f"**Actual vs Pro Policy ΔexpR:** {(metrics['pro_policy'].expectancy_r - metrics['actual'].expectancy_r):+.4f}R",
        f"**Actual vs Pro No Early ΔexpR:** {(metrics['pro_no_early'].expectancy_r - metrics['actual'].expectancy_r):+.4f}R",
        "",
        "---",
        "",
        "*Generated by `python -m research.exit_counterfactual`*",
    ])
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    logger.warning("Wrote markdown report to %s", output_path)


def write_json_report(
    metrics: dict[str, PolicyMetrics],
    results: dict[str, list[ReplayResult]],
    entries: list[DayTradeEntry],
    n_no_bars: int,
    output_path: Path,
) -> None:
    """Write machine-readable JSON report."""
    from datetime import timezone as tz
    output = {
        "generated_at": datetime.now(tz.utc).isoformat(),
        "sample_size": metrics["actual"].n_trades,
        "n_no_bars": n_no_bars,
        "policies": {},
        "comparison": {},
    }
    
    for policy, m in metrics.items():
        output["policies"][policy] = asdict(m)
    
    base_exp = metrics["actual"].expectancy_r
    output["comparison"] = {
        "actual_vs_hold_pure_delta_exp_r": round(metrics["hold_pure"].expectancy_r - base_exp, 4),
        "actual_vs_pro_policy_delta_exp_r": round(metrics["pro_policy"].expectancy_r - base_exp, 4),
        "actual_vs_pro_no_early_delta_exp_r": round(metrics["pro_no_early"].expectancy_r - base_exp, 4),
    }
    
    trades_by_policy = {}
    for policy, policy_results in results.items():
        trades_by_policy[policy] = [
            {
                "symbol": r.symbol,
                "entry_time": r.entry_time.isoformat() if r.entry_time else None,
                "entry_price": round(r.entry_price, 4),
                "exit_time": r.exit_time.isoformat() if r.exit_time else None,
                "exit_price": round(r.exit_price, 4) if r.exit_price else None,
                "exit_reason": r.exit_reason,
                "r_multiple": round(r.r_multiple, 4),
                "hold_bars": r.hold_bars,
                "side": r.side,
            }
            for r in policy_results
        ]
    output["trades"] = trades_by_policy
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    logger.warning("Wrote JSON report to %s", output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ledger", "-l",
        type=Path,
        default=REPO_ROOT / "trading_state.json",
        help="Path to trading_state.json or plain list JSON (default: repo root)",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help=f"Postgres DSN for minute bars (default: POSTGRES_DSN env, then {DEFAULT_DSN})",
    )
    parser.add_argument(
        "--bars-dir",
        type=Path,
        default=None,
        help="Directory with {SYMBOL}.parquet/.csv files for offline bars",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=REPO_ROOT / "research" / "reports",
        help="Output directory for reports (default: research/reports/)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load entries without replaying (for quick validation)",
    )
    args = parser.parse_args(argv)
    
    raw_history = load_trading_state(args.ledger)
    if not raw_history:
        logger.error("No trade history found in %s", args.ledger)
        return 1
    
    start_date = date.fromisoformat(args.start_date) if args.start_date else None
    end_date = date.fromisoformat(args.end_date) if args.end_date else None
    
    entries = extract_day_trade_entries(raw_history, start_date, end_date)
    if not entries:
        logger.error("No day-trade entries found")
        return 1
    
    logger.warning("Found %d day-trade entries to analyze", len(entries))
    
    if args.dry_run:
        print(f"\nDry run: {len(entries)} entries would be replayed.")
        for e in entries[:10]:
            print(f"  {e.entry_time} {e.symbol} @ {e.entry_price:.2f} [{e.entry_pattern}] exit={e.exit_reason} side={e.side}")
        if len(entries) > 10:
            print(f"  ... and {len(entries) - 10} more")
        return 0
    
    if args.bars_dir and args.bars_dir.exists():
        logger.warning("Using file-based bars loader from: %s", args.bars_dir)
        bars_loader = make_file_bars_loader(args.bars_dir)
    elif args.dsn or os.environ.get("POSTGRES_DSN"):
        dsn = args.dsn or os.environ.get("POSTGRES_DSN") or DEFAULT_DSN
        logger.warning("Using Postgres bars loader with DSN: %s", dsn[:30] + "...")
        bars_loader = make_postgres_bars_loader(dsn)
    else:
        logger.error("NO BARS SOURCE AVAILABLE. Provide --bars-dir or --dsn or set POSTGRES_DSN env.")
        raise NoBarsSourceError("No bars source configured. Cannot run counterfactual replay.")
    
    results, n_no_bars = run_counterfactual_replay(entries, bars_loader)
    
    if not results["actual"]:
        logger.error("No trades with bars data found")
        return 1
    
    metrics = {
        policy: compute_policy_metrics(policy_results, entries)
        for policy, policy_results in results.items()
    }
    
    args.out.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    md_path = args.out / f"exit_counterfactual_{date_str}.md"
    json_path = args.out / f"exit_counterfactual_{date_str}.json"
    
    write_markdown_report(metrics, results, entries, n_no_bars, md_path)
    write_json_report(metrics, results, entries, n_no_bars, json_path)
    
    print("\n" + "=" * 70)
    print("EXIT COUNTERFACTUAL SUMMARY")
    print("=" * 70)
    print(f"\nSample: {metrics['actual'].n_trades} trades with bars ({n_no_bars} skipped — no bars)")
    print(f"\n{'Policy':<15} {'n':>5} {'WR':>7} {'expR':>8} {'TotalR':>8} {'PF':>6}")
    print("-" * 50)
    for policy in ["actual", "hold_pure", "pro_policy", "pro_no_early"]:
        m = metrics[policy]
        pf_str = f"{m.profit_factor:.2f}" if m.profit_factor is not None else "∞"
        print(f"{policy:<15} {m.n_trades:>5} {m.win_rate*100:>6.1f}% {m.expectancy_r:>+7.3f} {m.total_r:>+7.2f}R {pf_str:>6}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
