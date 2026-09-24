#!/usr/bin/env python3
"""Exit Counterfactual Research Harness — replays day-trade entries under different exit policies.

v-exit-counterfactual-2026-09-24. Uses minute_bars (Stage A backtest infrastructure) to replay
historical managed day-trade entries under multiple exit policies:

  (a) actual     — the exit that occurred in the live ledger
  (b) hold_pure  — pure hold to SL / TP / flatten (no early exits)
  (c) pro_policy — unified pro policy:
                   - structure-based initial stop
                   - scale 50% at +1R, move stop to breakeven
                   - trail remainder (ATR or swing low)
                   - time stop N minutes before flatten
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
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone, date, time
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("exit_counterfactual")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

HANDS_OFF_SYMBOLS = frozenset({"MU", "HQGE", "SPCX"})
MAX_HOLD_BARS = 120  # 120 × 1-min = 2 hours max hold for replay
FLATTEN_HOUR = 15  # 3 PM ET
FLATTEN_MINUTE = 0
ATR_STOP_MULT = 1.5
RR_RATIO = 2.0
DEFAULT_SLIPPAGE_PCT = 0.0005


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
    atr: float
    rsi: Optional[float]
    regime: str
    time_of_day: str
    session_id: str
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
        except (ValueError, TypeError):
            return None
        
        exit_time_str = d.get("exit_time")
        exit_time = None
        if exit_time_str:
            try:
                exit_time = datetime.fromisoformat(exit_time_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
        
        entry_price = float(d.get("entry_price", 0))
        exit_price = float(d.get("exit_price", 0)) if d.get("exit_price") else None
        quantity = int(d.get("quantity", 1))
        pnl = float(d.get("pnl", 0))
        exit_reason = str(d.get("exit_reason", "unknown"))
        
        stop_loss = float(reasoning.get("stop_loss", reasoning.get("stop", 0)))
        take_profit = float(reasoning.get("take_profit", reasoning.get("target", 0)))
        stop_distance = float(reasoning.get("stop_distance", reasoning.get("stop_dist", 0)))
        atr = float(reasoning.get("atr", 0))
        rsi = reasoning.get("rsi")
        if rsi is not None:
            rsi = float(rsi)
        
        if stop_distance == 0 and entry_price > 0 and atr > 0:
            stop_distance = ATR_STOP_MULT * atr
        elif stop_distance == 0 and stop_loss > 0:
            stop_distance = abs(entry_price - stop_loss)
        
        if stop_loss == 0 and entry_price > 0 and stop_distance > 0:
            stop_loss = entry_price - stop_distance
        if take_profit == 0 and entry_price > 0 and stop_distance > 0:
            take_profit = entry_price + (RR_RATIO * stop_distance)
        
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
            atr=atr,
            rsi=rsi,
            regime=regime,
            time_of_day=time_of_day,
            session_id=session_id,
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
    profit_factor: float
    total_r: float
    expectancy_r: float
    max_dd_r: float
    max_dd_pct: float
    max_losing_day_r: float
    avg_hold_bars: float
    by_exit_reason: dict[str, int]
    by_pattern: dict[str, dict[str, Any]]


def load_trading_state(path: Path) -> list[dict]:
    """Load trade_history from trading_state.json."""
    if not path.exists():
        logger.warning("trading_state.json not found at %s", path)
        return []
    
    with open(path) as f:
        data = json.load(f)
    
    return data.get("trade_history", [])


def extract_day_trade_entries(
    raw_history: list[dict],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> list[DayTradeEntry]:
    """Extract day-trade entries from trade history."""
    entries = []
    for raw in raw_history:
        entry = DayTradeEntry.from_trade_history(raw)
        if entry is None:
            continue
        
        entry_date = entry.entry_time.date() if hasattr(entry.entry_time, "date") else None
        if entry_date:
            if start_date and entry_date < start_date:
                continue
            if end_date and entry_date > end_date:
                continue
        
        entries.append(entry)
    
    logger.info("Extracted %d day-trade entries", len(entries))
    return entries


def make_postgres_bars_loader(dsn: str) -> Callable[[str, datetime, int], pd.DataFrame]:
    """Return a bars loader using PostgresDataProvider."""
    from data_providers.postgres import PostgresDataProvider
    provider = PostgresDataProvider(dsn)

    def _load(symbol: str, start: datetime, lookforward_minutes: int = 300) -> pd.DataFrame:
        """Load minute bars for symbol starting at `start`."""
        end = start + timedelta(minutes=lookforward_minutes + 60)
        try:
            df = provider.get_market_data(
                symbol,
                period_type="day",
                period=5,
                frequency_type="minute",
                frequency=1,
                end=end,
            )
            if df.empty:
                return df
            df = df[df.index > start]
            return df
        except Exception as exc:
            logger.debug("Bars load failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    return _load


def make_file_bars_loader(bars_dir: Path) -> Callable[[str, datetime, int], pd.DataFrame]:
    """Return a bars loader using local parquet/CSV files."""
    def _load(symbol: str, start: datetime, lookforward_minutes: int = 300) -> pd.DataFrame:
        """Load minute bars from local file."""
        parquet_path = bars_dir / f"{symbol.upper()}.parquet"
        csv_path = bars_dir / f"{symbol.upper()}.csv"
        
        df = pd.DataFrame()
        try:
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
            elif csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
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
                end = start + timedelta(minutes=lookforward_minutes)
                df = df[(df.index > start) & (df.index <= end)]
            
            return df
        except Exception as exc:
            logger.debug("File bars load failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    return _load


def _is_flatten_time(ts: datetime) -> bool:
    """Check if timestamp is at or past flatten time (15:00 ET)."""
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        ts_et = ts.astimezone(et) if ts.tzinfo else ts.replace(tzinfo=timezone.utc).astimezone(et)
        return ts_et.hour >= FLATTEN_HOUR
    except Exception:
        hour_utc = ts.hour if ts.tzinfo is None else ts.utctimetuple().tm_hour
        return hour_utc >= 19


def _is_time_stop(ts: datetime, minutes_before_flatten: int = 15) -> bool:
    """Check if we should exit due to time stop (N min before flatten)."""
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        ts_et = ts.astimezone(et) if ts.tzinfo else ts.replace(tzinfo=timezone.utc).astimezone(et)
        flatten_time = ts_et.replace(hour=FLATTEN_HOUR, minute=FLATTEN_MINUTE, second=0)
        cutoff = flatten_time - timedelta(minutes=minutes_before_flatten)
        return ts_et >= cutoff
    except Exception:
        return False


def replay_policy_actual(entry: DayTradeEntry) -> ReplayResult:
    """Policy (a): Return the actual exit from the ledger."""
    r_multiple = 0.0
    if entry.stop_distance > 0 and entry.exit_price is not None:
        r_multiple = (entry.exit_price - entry.entry_price) / entry.stop_distance
    
    return_pct = 0.0
    if entry.entry_price > 0 and entry.exit_price is not None:
        return_pct = (entry.exit_price - entry.entry_price) / entry.entry_price * 100
    
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
        stop_distance=entry.stop_distance,
        r_multiple=r_multiple,
        return_pct=return_pct,
        hold_bars=hold_bars,
        policy="actual",
    )


def replay_policy_hold_pure(
    entry: DayTradeEntry,
    bars_loader: Callable[[str, datetime, int], pd.DataFrame],
) -> ReplayResult:
    """Policy (b): Pure hold to SL / TP / flatten — no early exits."""
    df = bars_loader(entry.symbol, entry.entry_time, lookforward_minutes=400)
    
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
        )
    
    stop = entry.stop_loss if entry.stop_loss > 0 else entry.entry_price - entry.stop_distance
    target = entry.take_profit if entry.take_profit > 0 else entry.entry_price + (RR_RATIO * entry.stop_distance)
    
    exit_time = None
    exit_price = None
    exit_reason = "timeout"
    hold_bars = 0
    
    for i in range(min(len(df), MAX_HOLD_BARS)):
        bar = df.iloc[i]
        bar_ts = df.index[i]
        hold_bars = i + 1
        
        if _is_flatten_time(bar_ts):
            exit_time = bar_ts
            exit_price = float(bar["Close"])
            exit_reason = "flatten"
            break
        
        bar_high = float(bar["High"])
        bar_low = float(bar["Low"])
        
        stop_hit = bar_low <= stop
        target_hit = bar_high >= target
        
        if stop_hit and target_hit:
            exit_time = bar_ts
            if bar["Close"] > bar["Open"]:
                exit_price = target
                exit_reason = "target"
            else:
                exit_price = stop
                exit_reason = "stop"
            break
        elif stop_hit:
            exit_time = bar_ts
            exit_price = stop
            exit_reason = "stop"
            break
        elif target_hit:
            exit_time = bar_ts
            exit_price = target
            exit_reason = "target"
            break
    
    if exit_price is None:
        if len(df) > 0:
            last_bar = df.iloc[min(len(df) - 1, MAX_HOLD_BARS - 1)]
            exit_time = df.index[min(len(df) - 1, MAX_HOLD_BARS - 1)]
            exit_price = float(last_bar["Close"])
        else:
            exit_price = entry.entry_price
    
    r_multiple = 0.0
    if entry.stop_distance > 0 and exit_price is not None:
        r_multiple = (exit_price - entry.entry_price) / entry.stop_distance
    
    return_pct = 0.0
    if entry.entry_price > 0 and exit_price is not None:
        return_pct = (exit_price - entry.entry_price) / entry.entry_price * 100
    
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
    )


def replay_policy_pro(
    entry: DayTradeEntry,
    bars_loader: Callable[[str, datetime, int], pd.DataFrame],
    minutes_before_flatten: int = 15,
    trail_atr_mult: float = 1.5,
    disable_early_exits: bool = False,
) -> ReplayResult:
    """Policy (c) or (d): Unified pro policy with scale/trail.
    
    - Structure-based initial stop
    - Scale 50% at +1R, move stop to breakeven
    - Trail remainder by ATR
    - Time stop N minutes before flatten
    
    If disable_early_exits=True, this becomes policy (d) — ignores early exits.
    """
    df = bars_loader(entry.symbol, entry.entry_time, lookforward_minutes=400)
    
    if df.empty:
        policy_name = "pro_no_early" if disable_early_exits else "pro_policy"
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
        )
    
    stop = entry.stop_loss if entry.stop_loss > 0 else entry.entry_price - entry.stop_distance
    target = entry.take_profit if entry.take_profit > 0 else entry.entry_price + (RR_RATIO * entry.stop_distance)
    stop_distance = entry.stop_distance
    atr = entry.atr if entry.atr > 0 else stop_distance / ATR_STOP_MULT
    
    current_stop = stop
    scaled_out = False
    scaled_out_at = None
    partial_r = 0.0
    trailing_active = False
    highest_price = entry.entry_price
    
    exit_time = None
    exit_price = None
    exit_reason = "timeout"
    hold_bars = 0
    
    policy_name = "pro_no_early" if disable_early_exits else "pro_policy"
    
    for i in range(min(len(df), MAX_HOLD_BARS)):
        bar = df.iloc[i]
        bar_ts = df.index[i]
        hold_bars = i + 1
        bar_high = float(bar["High"])
        bar_low = float(bar["Low"])
        bar_close = float(bar["Close"])
        
        highest_price = max(highest_price, bar_high)
        
        if _is_time_stop(bar_ts, minutes_before_flatten):
            exit_time = bar_ts
            exit_price = bar_close
            exit_reason = "time_stop"
            break
        
        if _is_flatten_time(bar_ts):
            exit_time = bar_ts
            exit_price = bar_close
            exit_reason = "flatten"
            break
        
        if bar_low <= current_stop:
            exit_time = bar_ts
            exit_price = current_stop
            exit_reason = "stop" if not trailing_active else "trailing_stop"
            break
        
        if bar_high >= target:
            exit_time = bar_ts
            exit_price = target
            exit_reason = "target"
            break
        
        current_r = (bar_close - entry.entry_price) / stop_distance if stop_distance > 0 else 0
        
        if not scaled_out and current_r >= 1.0:
            scaled_out = True
            scaled_out_at = bar_close
            partial_r = 0.5 * 1.0
            current_stop = entry.entry_price
            trailing_active = True
        
        if trailing_active and bar_close > entry.entry_price:
            trail_stop = bar_close - (trail_atr_mult * atr)
            if trail_stop > current_stop:
                current_stop = trail_stop
    
    if exit_price is None:
        if len(df) > 0:
            last_bar = df.iloc[min(len(df) - 1, MAX_HOLD_BARS - 1)]
            exit_time = df.index[min(len(df) - 1, MAX_HOLD_BARS - 1)]
            exit_price = float(last_bar["Close"])
        else:
            exit_price = entry.entry_price
    
    remaining_r = 0.0
    if stop_distance > 0:
        remaining_r = (exit_price - entry.entry_price) / stop_distance
        if scaled_out:
            remaining_r = remaining_r * 0.5
    
    total_r = partial_r + remaining_r
    
    return_pct = 0.0
    if entry.entry_price > 0:
        return_pct = (exit_price - entry.entry_price) / entry.entry_price * 100
    
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
        scaled_out_at=scaled_out_at,
        partial_r=partial_r,
        remaining_r=remaining_r,
    )


def run_counterfactual_replay(
    entries: list[DayTradeEntry],
    bars_loader: Callable[[str, datetime, int], pd.DataFrame],
) -> dict[str, list[ReplayResult]]:
    """Replay all entries under all four policies."""
    results: dict[str, list[ReplayResult]] = {
        "actual": [],
        "hold_pure": [],
        "pro_policy": [],
        "pro_no_early": [],
    }
    
    for i, entry in enumerate(entries):
        if (i + 1) % 10 == 0 or i == len(entries) - 1:
            logger.info("Replaying %d/%d: %s", i + 1, len(entries), entry.symbol)
        
        results["actual"].append(replay_policy_actual(entry))
        results["hold_pure"].append(replay_policy_hold_pure(entry, bars_loader))
        results["pro_policy"].append(replay_policy_pro(entry, bars_loader, disable_early_exits=False))
        results["pro_no_early"].append(replay_policy_pro(entry, bars_loader, disable_early_exits=True))
    
    return results


def compute_policy_metrics(results: list[ReplayResult], entries: list[DayTradeEntry]) -> PolicyMetrics:
    """Compute aggregate metrics for a policy's results."""
    valid = [r for r in results if r.exit_reason != "no_bars"]
    
    if not valid:
        return PolicyMetrics(
            policy=results[0].policy if results else "unknown",
            n_trades=0,
            n_sessions=0,
            n_wins=0,
            n_losses=0,
            n_scratches=0,
            win_rate=0.0,
            profit_factor=0.0,
            total_r=0.0,
            expectancy_r=0.0,
            max_dd_r=0.0,
            max_dd_pct=0.0,
            max_losing_day_r=0.0,
            avg_hold_bars=0.0,
            by_exit_reason={},
            by_pattern={},
        )
    
    sessions = set()
    for r in valid:
        if r.entry_time:
            sessions.add(r.entry_time.date())
    
    wins = [r for r in valid if r.r_multiple > 0.05]
    losses = [r for r in valid if r.r_multiple < -0.05]
    scratches = [r for r in valid if -0.05 <= r.r_multiple <= 0.05]
    
    n_trades = len(valid)
    n_wins = len(wins)
    n_losses = len(losses)
    
    win_loss_count = n_wins + n_losses
    win_rate = n_wins / win_loss_count if win_loss_count > 0 else 0.0
    
    gross_profit = sum(r.r_multiple for r in wins)
    gross_loss = abs(sum(r.r_multiple for r in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    
    total_r = sum(r.r_multiple for r in valid)
    expectancy_r = total_r / n_trades if n_trades > 0 else 0.0
    
    peak_r = 0.0
    max_dd_r = 0.0
    cumulative_r = 0.0
    sorted_results = sorted(valid, key=lambda x: x.entry_time or datetime.min)
    for r in sorted_results:
        cumulative_r += r.r_multiple
        if cumulative_r > peak_r:
            peak_r = cumulative_r
        dd = peak_r - cumulative_r
        if dd > max_dd_r:
            max_dd_r = dd
    
    max_dd_pct = max_dd_r / peak_r if peak_r > 0 else 0.0
    
    daily_r: dict[date, float] = defaultdict(float)
    for r in valid:
        if r.entry_time:
            daily_r[r.entry_time.date()] += r.r_multiple
    max_losing_day_r = abs(min(daily_r.values())) if daily_r else 0.0
    
    avg_hold = sum(r.hold_bars for r in valid) / n_trades if n_trades > 0 else 0.0
    
    by_exit_reason: dict[str, int] = defaultdict(int)
    for r in valid:
        by_exit_reason[r.exit_reason] += 1
    
    entry_patterns = {(e.symbol, e.entry_time): e.entry_pattern for e in entries}
    by_pattern: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "total_r": 0.0, "wins": 0, "losses": 0})
    for r in valid:
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
        policy=valid[0].policy if valid else "unknown",
        n_trades=n_trades,
        n_sessions=len(sessions),
        n_wins=n_wins,
        n_losses=n_losses,
        n_scratches=len(scratches),
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 3) if profit_factor != float("inf") else None,
        total_r=round(total_r, 3),
        expectancy_r=round(expectancy_r, 4),
        max_dd_r=round(max_dd_r, 3),
        max_dd_pct=round(max_dd_pct, 4),
        max_losing_day_r=round(max_losing_day_r, 3),
        avg_hold_bars=round(avg_hold, 1),
        by_exit_reason=dict(by_exit_reason),
        by_pattern=dict(by_pattern),
    )


def write_markdown_report(
    metrics: dict[str, PolicyMetrics],
    results: dict[str, list[ReplayResult]],
    entries: list[DayTradeEntry],
    output_path: Path,
) -> None:
    """Write human-readable markdown report."""
    lines = [
        "# Exit Counterfactual Analysis",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Sample:** {len(entries)} day-trade entries",
        "",
        "## Executive Summary",
        "",
        "This report compares the **actual** exits against counterfactual scenarios:",
        "",
        "| Policy | Description |",
        "|--------|-------------|",
        "| actual | What actually happened in live trading |",
        "| hold_pure | Pure hold to SL/TP/flatten — no early exits |",
        "| pro_policy | Unified pro: scale 50% at +1R → BE → trail → time stop |",
        "| pro_no_early | Pro policy but ignoring MACD/RSI/desk early exits |",
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
            f"- **Profit Factor:** {m.profit_factor:.2f}" if m.profit_factor else "- **Profit Factor:** ∞",
            f"- **Max Drawdown:** {m.max_dd_r:.2f}R ({m.max_dd_pct*100:.1f}%)",
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
        "## Key Findings",
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
    
    logger.info("Wrote markdown report to %s", output_path)


def write_json_report(
    metrics: dict[str, PolicyMetrics],
    results: dict[str, list[ReplayResult]],
    entries: list[DayTradeEntry],
    output_path: Path,
) -> None:
    """Write machine-readable JSON report."""
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(entries),
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
            }
            for r in policy_results
        ]
    output["trades"] = trades_by_policy
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    logger.info("Wrote JSON report to %s", output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ledger", "-l",
        type=Path,
        default=REPO_ROOT / "trading_state.json",
        help="Path to trading_state.json (default: repo root)",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Postgres DSN for minute bars (default: POSTGRES_DSN env var)",
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
    
    logger.info("Found %d day-trade entries to analyze", len(entries))
    
    if args.dry_run:
        print(f"\nDry run: {len(entries)} entries would be replayed.")
        for e in entries[:10]:
            print(f"  {e.entry_time} {e.symbol} @ {e.entry_price:.2f} [{e.entry_pattern}] exit={e.exit_reason}")
        if len(entries) > 10:
            print(f"  ... and {len(entries) - 10} more")
        return 0
    
    if args.bars_dir and args.bars_dir.exists():
        logger.info("Using file-based bars loader from: %s", args.bars_dir)
        bars_loader = make_file_bars_loader(args.bars_dir)
    else:
        dsn = args.dsn or os.environ.get("POSTGRES_DSN")
        if not dsn:
            logger.warning("No bars source available (no --bars-dir, no POSTGRES_DSN). "
                          "Using actual-only mode (no counterfactual replay).")
            results = {
                "actual": [replay_policy_actual(e) for e in entries],
                "hold_pure": [replay_policy_actual(e) for e in entries],
                "pro_policy": [replay_policy_actual(e) for e in entries],
                "pro_no_early": [replay_policy_actual(e) for e in entries],
            }
        else:
            bars_loader = make_postgres_bars_loader(dsn)
            results = run_counterfactual_replay(entries, bars_loader)
    
    if 'results' not in locals():
        results = run_counterfactual_replay(entries, bars_loader)
    
    metrics = {
        policy: compute_policy_metrics(policy_results, entries)
        for policy, policy_results in results.items()
    }
    
    args.out.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    md_path = args.out / f"exit_counterfactual_{date_str}.md"
    json_path = args.out / f"exit_counterfactual_{date_str}.json"
    
    write_markdown_report(metrics, results, entries, md_path)
    write_json_report(metrics, results, entries, json_path)
    
    print("\n" + "=" * 70)
    print("EXIT COUNTERFACTUAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Policy':<15} {'n':>5} {'WR':>7} {'expR':>8} {'TotalR':>8} {'PF':>6}")
    print("-" * 50)
    for policy in ["actual", "hold_pure", "pro_policy", "pro_no_early"]:
        m = metrics[policy]
        pf_str = f"{m.profit_factor:.2f}" if m.profit_factor else "∞"
        print(f"{policy:<15} {m.n_trades:>5} {m.win_rate*100:>6.1f}% {m.expectancy_r:>+7.3f} {m.total_r:>+7.2f}R {pf_str:>6}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
