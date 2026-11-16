"""Scalp reversal backtest — standalone 1-min intraday module.

Detects RSI-extreme + BB-touch + volume-spike reversals on 1-min bars,
enters at bar close, exits via quick_profit / stop / target / timeout / eod.
Enforces a global N-slot concurrent-position cap across all symbols and
reports dollar P&L against a fixed margin capital allocation.

Does NOT modify engine.py, trades.py, or metrics.py.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import DEFAULT_DSN, compute_indicators
from backtest.metrics import summarize
from backtest.trades import Trade

logger = logging.getLogger("backtest.scalp_reversal")
DEFAULT_SYMBOLS = ["NVDA", "TSLA", "AMD", "QQQ", "SMCI", "IREN", "MU"]
CAPITAL_USD = 100_000
MAX_SLOTS = 10
SLOT_SIZE_USD = CAPITAL_USD // MAX_SLOTS  # 10_000


@dataclass(frozen=True)
class ScalpParams:
    rsi_long_thresh: float = 30.0
    rsi_short_thresh: float = 75.0
    vol_min: float = 2.0
    atr_stop_mult: float = 1.0
    atr_target_mult: float = 2.0
    quick_exit_bars: int = 3
    min_profit_pct: float = 0.15
    max_hold_bars: int = 15


def load_bars(symbol: str, days: int, dsn: str) -> pd.DataFrame:
    """Load 1-min RTH bars from Postgres, returned with UTC DatetimeIndex."""
    from data_providers.postgres import PostgresDataProvider
    provider = PostgresDataProvider(dsn)
    period_type = "day" if days < 30 else "month"
    period = days if period_type == "day" else max(1, days // 30)
    df = provider.get_market_data(
        symbol,
        period_type=period_type,
        period=period,
        frequency_type="minute",
        frequency=1,
    )
    if df.empty:
        return df
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    et = df.index.tz_convert("America/New_York")
    minutes = et.hour * 60 + et.minute
    mask = (minutes >= 9 * 60 + 30) & (minutes <= 15 * 60 + 55)
    return df[mask]


def replay_symbol_scalp(
    symbol: str,
    df: pd.DataFrame,
    params: ScalpParams,
) -> list[Trade]:
    """Replay 1-min bars for one symbol; return all trades."""
    if df.empty or len(df) < 51:
        return []

    ind = compute_indicators(df)
    trades: list[Trade] = []

    # Two independent position slots (one per side, no pyramiding).
    open_long: dict | None = None
    open_short: dict | None = None

    # Precompute ET dates for per-day logic.
    if df.index.tz is None:
        et_index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        et_index = df.index.tz_convert("America/New_York")
    et_dates = et_index.normalize()

    # Per-day bar counter to skip first 15 bars of each RTH session.
    day_bar_count: dict = {}

    required = {"rsi", "macd", "macd_signal", "bb_lower", "bb_middle",
                "bb_upper", "atr", "adx", "sma_20", "sma_50",
                "volume_ratio", "high_20", "low_20"}

    for i in range(50, len(df)):
        bar = df.iloc[i]
        indicators = ind.iloc[i].to_dict()

        if any(pd.isna(indicators.get(k)) for k in required):
            continue

        bar_date = et_dates[i]
        day_bar_count[bar_date] = day_bar_count.get(bar_date, 0) + 1
        if day_bar_count[bar_date] <= 15:
            # Skip first 15 bars of each session (RSI instability at open).
            # Still flush EOD for any position open at day boundary.
            pass

        # --- EOD flush: last bar of a calendar day ---
        is_last_bar_of_day = (
            i == len(df) - 1
            or et_dates[i + 1] != bar_date
        )

        close_px = float(bar["Close"])
        bar_ts = df.index[i]

        # --- Manage open long ---
        if open_long is not None:
            open_long["hold"] += 1
            pos = open_long
            direction = 1
            entry = pos["entry"]
            unrealized_pct = (close_px - entry) / entry * 100.0

            exit_reason: str | None = None
            exit_price: float | None = None

            if (pos["hold"] <= params.quick_exit_bars
                    and unrealized_pct >= params.min_profit_pct):
                exit_reason = "quick_profit"
                exit_price = close_px
            elif float(bar["Low"]) <= pos["stop"]:
                exit_reason = "stop"
                exit_price = pos["stop"]
            elif float(bar["High"]) >= pos["target"]:
                exit_reason = "target"
                exit_price = pos["target"]
            elif pos["hold"] >= params.max_hold_bars:
                exit_reason = "timeout"
                exit_price = close_px
            elif is_last_bar_of_day:
                exit_reason = "eod"
                exit_price = close_px

            if exit_reason is not None:
                trades.append(Trade(
                    strategy="scalp_reversal",
                    symbol=symbol,
                    side="long",
                    entry_time=pos["entry_time"],
                    entry_price=entry,
                    exit_time=bar_ts,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    hold_bars=pos["hold"],
                    entry_indicators=pos["entry_indicators"],
                ))
                open_long = None

        # --- Manage open short ---
        if open_short is not None:
            open_short["hold"] += 1
            pos = open_short
            entry = pos["entry"]
            unrealized_pct = (entry - close_px) / entry * 100.0

            exit_reason = None
            exit_price = None

            if (pos["hold"] <= params.quick_exit_bars
                    and unrealized_pct >= params.min_profit_pct):
                exit_reason = "quick_profit"
                exit_price = close_px
            elif float(bar["High"]) >= pos["stop"]:
                exit_reason = "stop"
                exit_price = pos["stop"]
            elif float(bar["Low"]) <= pos["target"]:
                exit_reason = "target"
                exit_price = pos["target"]
            elif pos["hold"] >= params.max_hold_bars:
                exit_reason = "timeout"
                exit_price = close_px
            elif is_last_bar_of_day:
                exit_reason = "eod"
                exit_price = close_px

            if exit_reason is not None:
                trades.append(Trade(
                    strategy="scalp_reversal",
                    symbol=symbol,
                    side="short",
                    entry_time=pos["entry_time"],
                    entry_price=entry,
                    exit_time=bar_ts,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    hold_bars=pos["hold"],
                    entry_indicators=pos["entry_indicators"],
                ))
                open_short = None

        # --- Skip signal check during first-15-bar window OR last bar of session ---
        if day_bar_count[bar_date] <= 15 or is_last_bar_of_day:
            continue

        atr = float(indicators["atr"])
        vol_ratio = float(indicators["volume_ratio"])
        rsi = float(indicators["rsi"])
        bb_lower = float(indicators["bb_lower"])
        bb_upper = float(indicators["bb_upper"])

        def _snapshot() -> dict:
            return {
                k: float(v)
                for k, v in indicators.items()
                if v is not None
                and not isinstance(v, (list, tuple, np.ndarray))
                and not pd.isna(v)
            }

        # --- Long signal ---
        if (open_long is None
                and rsi < params.rsi_long_thresh
                and close_px <= bb_lower
                and vol_ratio >= params.vol_min):
            open_long = {
                "entry": close_px,
                "stop": close_px - atr * params.atr_stop_mult,
                "target": close_px + atr * params.atr_target_mult,
                "entry_time": bar_ts,
                "hold": 0,
                "entry_indicators": _snapshot(),
            }

        # --- Short signal ---
        if (open_short is None
                and rsi > params.rsi_short_thresh
                and close_px >= bb_upper
                and vol_ratio >= params.vol_min):
            open_short = {
                "entry": close_px,
                "stop": close_px + atr * params.atr_stop_mult,
                "target": close_px - atr * params.atr_target_mult,
                "entry_time": bar_ts,
                "hold": 0,
                "entry_indicators": _snapshot(),
            }

    # Final EOD flush for anything still open at the very last bar.
    final_bar = df.iloc[-1]
    final_ts = df.index[-1]
    final_px = float(final_bar["Close"])

    for side, pos in (("long", open_long), ("short", open_short)):
        if pos is None:
            continue
        trades.append(Trade(
            strategy="scalp_reversal",
            symbol=symbol,
            side=side,
            entry_time=pos["entry_time"],
            entry_price=pos["entry"],
            exit_time=final_ts,
            exit_price=final_px,
            exit_reason="eod",
            hold_bars=pos["hold"],
            entry_indicators=pos["entry_indicators"],
        ))

    return trades


def allocate_slots(
    trades: list[Trade],
    max_slots: int = MAX_SLOTS,
) -> list[Trade]:
    """Return the subset of trades that fit within max_slots concurrent positions.

    Trades are processed in (entry_time, symbol) order — alphabetical symbol
    is the tiebreak when two trades have the same entry_time. A greedy
    earliest-free-slot algorithm decides allocation.
    """
    if not trades:
        return []

    sorted_trades = sorted(trades, key=lambda t: (t.entry_time, t.symbol))

    # slot_free_at[i] = earliest time slot i becomes available again.
    min_ts = pd.Timestamp.min.tz_localize("UTC")
    slot_free_at: list[pd.Timestamp] = [min_ts] * max_slots

    result: list[Trade] = []
    for trade in sorted_trades:
        entry = trade.entry_time
        if not isinstance(entry, pd.Timestamp):
            entry = pd.Timestamp(entry, tz="UTC")

        # Find the slot that frees up soonest.
        idx = int(np.argmin([s.value for s in slot_free_at]))
        if slot_free_at[idx] <= entry:
            result.append(trade)
            exit_t = trade.exit_time
            if not isinstance(exit_t, pd.Timestamp):
                exit_t = pd.Timestamp(exit_t, tz="UTC")
            slot_free_at[idx] = exit_t

    return result


def run(
    symbols: list[str] | None = None,
    days: int = 90,
    params: ScalpParams | None = None,
    dsn: str = DEFAULT_DSN,
    max_slots: int = MAX_SLOTS,
    capital_usd: int = CAPITAL_USD,
) -> dict[str, Any]:
    """Run the scalp reversal backtest across all symbols and return report dict."""
    symbols = symbols or DEFAULT_SYMBOLS
    params = params or ScalpParams()
    slot_size = capital_usd / max_slots

    all_trades: list[Trade] = []
    for sym in symbols:
        logger.info("loading %s", sym)
        try:
            df = load_bars(sym, days, dsn)
        except Exception as exc:
            logger.warning("load_bars(%s) failed: %s", sym, exc)
            continue
        sym_trades = replay_symbol_scalp(sym, df, params)
        all_trades.extend(sym_trades)

    executed = allocate_slots(all_trades, max_slots)

    # Dollar P&L per executed trade.
    dollar_pnls: list[float] = []
    for t in executed:
        shares = int(slot_size / t.entry_price) if t.entry_price > 0 else 0
        direction = 1 if t.side == "long" else -1
        dollar_pnls.append(shares * (t.exit_price - t.entry_price) * direction)

    # Exit reason breakdown (summarize() doesn't count quick_profit).
    all_reasons = {"quick_profit", "stop", "target", "timeout", "eod"}
    exit_counts = {r: sum(1 for t in executed if t.exit_reason == r) for r in all_reasons}

    overall = summarize(executed)
    # Patch exit_breakdown to include quick_profit.
    overall["exit_breakdown"] = {
        r: round(100 * exit_counts[r] / len(executed), 1) if executed else 0.0
        for r in all_reasons
    }

    def _patch_exit_bd(m: dict, side_trades: list[Trade]) -> dict:
        n = len(side_trades)
        m["exit_breakdown"]["quick_profit"] = round(
            100 * sum(1 for t in side_trades if t.exit_reason == "quick_profit") / n, 1
        ) if n else 0.0
        return m

    by_side = {
        "long": _patch_exit_bd(
            summarize([t for t in executed if t.side == "long"]),
            [t for t in executed if t.side == "long"],
        ),
        "short": _patch_exit_bd(
            summarize([t for t in executed if t.side == "short"]),
            [t for t in executed if t.side == "short"],
        ),
    }

    per_symbol: dict[str, Any] = {}
    for sym in symbols:
        sym_exec = [t for t in executed if t.symbol == sym]
        if sym_exec:
            sym_pnl = sum(
                int(slot_size / t.entry_price) * (t.exit_price - t.entry_price)
                * (1 if t.side == "long" else -1)
                for t in sym_exec if t.entry_price > 0
            )
            per_symbol[sym] = {**summarize(sym_exec), "dollar_pnl": round(sym_pnl, 2)}

    return {
        "run_date": datetime.now().isoformat(timespec="seconds"),
        "params": asdict(params),
        "symbols": symbols,
        "days": days,
        "capital_usd": capital_usd,
        "max_slots": max_slots,
        "slot_size_usd": slot_size,
        "overall": overall,
        "by_side": by_side,
        "by_exit_reason": exit_counts,
        "per_symbol": per_symbol,
        "total_trades_before_cap": len(all_trades),
        "dropped_trades": len(all_trades) - len(executed),
        "total_dollar_pnl": round(sum(dollar_pnls), 2),
    }
