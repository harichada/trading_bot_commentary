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
    # ADX gate: skip signal when trend strength exceeds this (999 = disabled)
    adx_max: float = 999.0
    # Candle confirmation: require the signal bar to show a reversal candle shape
    # Long: bullish close (close > open) or hammer (lower wick >= body * wick_ratio)
    # Short: bearish close (close < open) or shooting star (upper wick >= body * wick_ratio)
    require_reversal_candle: bool = False
    candle_wick_ratio: float = 1.5
    # Delayed entry: wait one bar after the signal for price to confirm the turn
    # Long confirms if close[i+1] > close[i]; short if close[i+1] < close[i]
    # Pending signals are cleared at EOD and at the start of each new session
    entry_confirm: bool = False
    # RSI turn confirmation: only enter when RSI is at extreme AND ticking back
    # Long: rsi < long_thresh AND rsi > prev_rsi (momentum inflecting up)
    # Short: rsi > short_thresh AND rsi < prev_rsi (momentum inflecting down)
    rsi_turn: bool = False
    # BB-reclaim confirmation: require price to have touched the band on the previous
    # bar and closed back inside it on the signal bar (the actual reversal candle).
    # Long:  prev_close <= bb_lower[i-1]  AND  close > bb_lower[i]
    # Short: prev_close >= bb_upper[i-1]  AND  close < bb_upper[i]
    bb_reclaim: bool = False
    # Post-stop cool-down: after a stop-out, block re-entry on the same side for
    # this many bars. Eliminates cascade re-entries without filtering the first signal.
    # 0 = disabled (default). 5–15 covers the typical cascade window.
    cooldown_bars: int = 0
    # Side filter: "both" | "long_only" | "short_only"
    sides: str = "both"
    # Per-stock daily direction gate: gainer (open > prev close) → short_only;
    # loser (open < prev close) → long_only. Mirrors the live screener logic:
    # buy panic (oversold losers), sell euphoria (overbought gainers).
    # Intersected with `sides` — e.g. sides=long_only + direction_gate on a
    # gainer day → no trades that day.
    direction_gate: bool = False
    # From-open drop gate: skip long signals when price has fallen more than this
    # percentage below the RTH session open. 0 = disabled.
    # E.g. 1.5 → skip longs once stock is down >1.5% from open (falling-knife filter).
    from_open_drop_max: float = 0.0


def _fetch_today_live(symbol: str) -> pd.DataFrame:
    """Fetch today's 1-min RTH bars from the live data priority chain.

    1. SchwabBarCache — zero-lag bars from the bot's own authenticated stream
    2. yfinance — fallback when the stream hasn't populated the cache yet
       (e.g. bot just started, Schwab not connected)

    Returns a UTC-aware DataFrame, empty if both sources fail.
    """
    # 1. Try Schwab stream cache
    try:
        from core.schwab_bar_cache import SchwabBarCache
        df = SchwabBarCache.instance().get_today_df(symbol)
        if not df.empty:
            logger.debug("load_bars %s: %d today bars from Schwab cache", symbol, len(df))
            return df
    except Exception as exc:
        logger.debug("load_bars schwab_bar_cache error for %s: %s", symbol, exc)

    # 2. yfinance fallback — use Ticker.history() (per-object, no cross-symbol cache)
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        raw = ticker.history(period="1d", interval="1m")
        if raw.empty:
            return raw
        raw = raw[["Open", "High", "Low", "Close", "Volume"]]
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("America/New_York")
        raw.index = raw.index.tz_convert("UTC")
        et = raw.index.tz_convert("America/New_York")
        minutes = et.hour * 60 + et.minute
        mask = (minutes >= 9 * 60 + 30) & (minutes <= 15 * 60 + 55)
        df = raw[mask]
        if not df.empty:
            df = df[~df.index.duplicated(keep="last")]
            logger.debug("load_bars %s: %d today bars from yfinance fallback", symbol, len(df))
        return df
    except Exception as exc:
        logger.warning("load_bars yfinance fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame()


def load_bars(symbol: str, days: int, dsn: str, end_date: str | None = None) -> pd.DataFrame:
    """Load 1-min RTH bars with a UTC-aware DatetimeIndex.

    When end_date (YYYY-MM-DD) is given, returns bars in the window
    [end_date - days, end_date) from Postgres only — no live feed.
    This enables out-of-sample backtests on arbitrary historical windows.

    Default (no end_date): merges Postgres history with today's live feed.
    """
    from data_providers.postgres import PostgresDataProvider
    provider = PostgresDataProvider(dsn)

    if end_date is not None:
        # Historical window: fetch extra data so the indicator warm-up
        # (50 bars) is available even after clipping to the target window.
        fetch_days = days + 60
        period_type = "day" if fetch_days < 30 else "month"
        period = fetch_days if period_type == "day" else max(1, fetch_days // 30)
        hist = provider.get_market_data(
            symbol,
            period_type=period_type,
            period=period,
            frequency_type="minute",
            frequency=1,
        )
        if not hist.empty:
            hist = hist[~hist.index.duplicated(keep="last")]
            et = hist.index.tz_convert("America/New_York")
            minutes = et.hour * 60 + et.minute
            mask = (minutes >= 9 * 60 + 30) & (minutes <= 15 * 60 + 55)
            hist = hist[mask]
        if hist.empty:
            return hist
        # Include all bars on the end_date day (treat as end-of-day, not midnight)
        end_cutoff = pd.Timestamp(end_date, tz="America/New_York") + pd.Timedelta(days=1)
        end_cutoff = end_cutoff.tz_convert("UTC")
        start_cutoff = end_cutoff - pd.Timedelta(days=days + 1)
        sliced = hist[(hist.index >= start_cutoff) & (hist.index < end_cutoff)]
        return sliced[~sliced.index.duplicated(keep="last")]

    period_type = "day" if days < 30 else "month"
    period = days if period_type == "day" else max(1, days // 30)

    # Fetch historical bars from Postgres (up to and including yesterday)
    hist = provider.get_market_data(
        symbol,
        period_type=period_type,
        period=period,
        frequency_type="minute",
        frequency=1,
    )
    if not hist.empty:
        hist = hist[~hist.index.duplicated(keep="last")]
        et = hist.index.tz_convert("America/New_York")
        minutes = et.hour * 60 + et.minute
        mask = (minutes >= 9 * 60 + 30) & (minutes <= 15 * 60 + 55)
        hist = hist[mask]

    # Fetch today's live bars (Schwab cache → yfinance fallback)
    today = _fetch_today_live(symbol)

    if hist.empty and today.empty:
        return pd.DataFrame()
    if hist.empty:
        return today[~today.index.duplicated(keep="last")] if not today.empty else today
    if today.empty:
        return hist

    # Sanity-check: today's price range must be consistent with recent history.
    # A >2x median ratio indicates wrong-symbol data from the live feed.
    hist_median = hist["Close"].median()
    today_median = today["Close"].median()
    if hist_median > 0 and today_median > 0:
        ratio = max(hist_median, today_median) / min(hist_median, today_median)
        if ratio > 2.0:
            logger.warning(
                "load_bars %s: today median price %.4f vs hist median %.4f "
                "(ratio %.2fx) — discarding today data as wrong-symbol feed",
                symbol, today_median, hist_median, ratio,
            )
            return hist

    # Merge: drop any Postgres rows that overlap with today's live data,
    # then concat so the live bars are always the freshest tail.
    today_start = today.index[0]
    hist_trimmed = hist[hist.index < today_start]
    combined = pd.concat([hist_trimmed, today]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def replay_symbol_scalp(
    symbol: str,
    df: pd.DataFrame,
    params: ScalpParams,
) -> list[Trade]:
    """Replay 1-min bars for one symbol; return all trades."""
    if df.empty or len(df) < 51:
        return []

    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep="last")]
    ind = compute_indicators(df)
    trades: list[Trade] = []

    # Two independent position slots (one per side, no pyramiding).
    open_long: dict | None = None
    open_short: dict | None = None
    # Pending entries: set when entry_confirm=True; confirmed or discarded next bar.
    # Each pending stores: signal_px, atr, entry_indicators (from signal bar)
    pending_long: dict | None = None
    pending_short: dict | None = None
    # Post-stop cool-down counters: bars remaining before re-entry allowed.
    long_cd: int = 0
    short_cd: int = 0

    # Precompute ET dates for per-day logic.
    if df.index.tz is None:
        et_index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        et_index = df.index.tz_convert("America/New_York")
    et_dates = et_index.normalize()

    # Per-day bar counter to skip first 15 bars of each RTH session.
    day_bar_count: dict = {}
    # Per-day RTH open price (first bar's Open in the session).
    rth_open: dict = {}
    # Per-day direction gate result: "long_only" | "short_only" | "both"
    # Populated at the start of each session when direction_gate=True.
    day_direction: dict = {}

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
        if day_bar_count[bar_date] == 1:
            # New session: discard any pending entry that crossed a day boundary.
            pending_long = None
            pending_short = None
            rth_open[bar_date] = float(bar["Open"])
            # Per-stock direction gate: compare today's RTH open to prior day's last close.
            # Gainer (open > prev close) → short_only; loser → long_only.
            if params.direction_gate and i > 0:
                prev_close_px = float(df.iloc[i - 1]["Close"])
                today_open_px = float(bar["Open"])
                _move = (today_open_px - prev_close_px) / prev_close_px if prev_close_px > 0 else 0
                day_direction[bar_date] = (
                    "short_only" if _move > 0 else ("long_only" if _move < 0 else "both")
                )
            else:
                day_direction[bar_date] = "both"  # gate inactive; params.sides handles it
        if day_bar_count[bar_date] <= 15:
            # Skip first 15 bars of each session (RSI instability at open).
            # Still flush EOD for any position open at day boundary.
            pass

        # --- EOD flush: last bar of a calendar day ---
        # For live/shadow replay the last bar is the most recent available
        # minute (e.g. 15:30), NOT actual market close (15:55).  Only flush
        # when the bar time is >= 15:55 ET or when the next bar crosses into
        # a new calendar day (multi-day backtests).
        bar_et_minutes = et_index[i].hour * 60 + et_index[i].minute
        is_last_bar_of_day = (
            (i == len(df) - 1 and bar_et_minutes >= 15 * 60 + 55)
            or (i < len(df) - 1 and et_dates[i + 1] != bar_date)
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

            # Adverse intrabar path: stop takes priority over everything,
            # including quick_profit. If Low touched the stop during the bar,
            # we were stopped out before the bar closed — even if the close
            # was profitable. Target beats quick_profit for the same reason:
            # we'd have exited at the target price intrabar, not at close.
            if float(bar["Low"]) <= pos["stop"]:
                exit_reason = "stop"
                exit_price = pos["stop"]
            elif float(bar["High"]) >= pos["target"]:
                exit_reason = "target"
                exit_price = pos["target"]
            elif (pos["hold"] <= params.quick_exit_bars
                    and unrealized_pct >= params.min_profit_pct):
                exit_reason = "quick_profit"
                exit_price = close_px
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
                if exit_reason == "stop":
                    long_cd = params.cooldown_bars

        # --- Manage open short ---
        if open_short is not None:
            open_short["hold"] += 1
            pos = open_short
            entry = pos["entry"]
            unrealized_pct = (entry - close_px) / entry * 100.0

            exit_reason = None
            exit_price = None

            # Adverse intrabar path: same logic as long — stop first,
            # then target, then quick_profit at close.
            if float(bar["High"]) >= pos["stop"]:
                exit_reason = "stop"
                exit_price = pos["stop"]
            elif float(bar["Low"]) <= pos["target"]:
                exit_reason = "target"
                exit_price = pos["target"]
            elif (pos["hold"] <= params.quick_exit_bars
                    and unrealized_pct >= params.min_profit_pct):
                exit_reason = "quick_profit"
                exit_price = close_px
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
                if exit_reason == "stop":
                    short_cd = params.cooldown_bars

        # --- Decrement cool-down counters each bar ---
        if long_cd > 0:
            long_cd -= 1
        if short_cd > 0:
            short_cd -= 1

        # --- Confirm or discard pending entries (entry_confirm mode) ---
        # Pending is set on the signal bar; this bar is the confirmation bar.
        # Long confirms if price closed higher than signal bar; else discard.
        # Always clears after one bar so it never carries into a 3rd bar.
        if pending_long is not None and open_long is None:
            if close_px > pending_long["signal_px"]:
                atr_p = pending_long["atr"]
                open_long = {
                    "entry": close_px,
                    "stop": close_px - atr_p * params.atr_stop_mult,
                    "target": close_px + atr_p * params.atr_target_mult,
                    "entry_time": bar_ts,
                    "hold": 0,
                    "entry_indicators": pending_long["entry_indicators"],
                }
            pending_long = None

        if pending_short is not None and open_short is None:
            if close_px < pending_short["signal_px"]:
                atr_p = pending_short["atr"]
                open_short = {
                    "entry": close_px,
                    "stop": close_px + atr_p * params.atr_stop_mult,
                    "target": close_px - atr_p * params.atr_target_mult,
                    "entry_time": bar_ts,
                    "hold": 0,
                    "entry_indicators": pending_short["entry_indicators"],
                }
            pending_short = None

        # --- Skip signal check during first-15-bar window OR last bar of session ---
        if day_bar_count[bar_date] <= 15 or is_last_bar_of_day:
            continue

        atr = float(indicators["atr"])
        adx = float(indicators["adx"])
        vol_ratio = float(indicators["volume_ratio"])
        rsi = float(indicators["rsi"])
        prev_rsi = float(ind.iloc[i - 1]["rsi"]) if i > 0 else rsi
        bb_lower = float(indicators["bb_lower"])
        bb_upper = float(indicators["bb_upper"])

        # ADX gate: skip when market is strongly trending (reversals fail in trends)
        if adx > params.adx_max:
            continue

        open_px = float(bar["Open"])
        high_px = float(bar["High"])
        low_px  = float(bar["Low"])

        # Candle shape flags (computed once, used in both signal checks below)
        if params.require_reversal_candle:
            body        = abs(close_px - open_px)
            lower_wick  = min(open_px, close_px) - low_px
            upper_wick  = high_px - max(open_px, close_px)
            wr          = params.candle_wick_ratio
            # Long confirmation: bar closed up, OR lower wick dominates (hammer)
            is_rev_long  = (close_px > open_px) or (body > 0 and lower_wick >= body * wr)
            # Short confirmation: bar closed down, OR upper wick dominates (shooting star)
            is_rev_short = (close_px < open_px) or (body > 0 and upper_wick >= body * wr)
        else:
            is_rev_long = is_rev_short = True

        def _snapshot() -> dict:
            return {
                k: float(v)
                for k, v in indicators.items()
                if v is not None
                and not isinstance(v, (list, tuple, np.ndarray))
                and not pd.isna(v)
            }

        snap = _snapshot()

        rsi_turn_long  = (not params.rsi_turn) or (rsi > prev_rsi)
        rsi_turn_short = (not params.rsi_turn) or (rsi < prev_rsi)

        prev_close    = float(df.iloc[i - 1]["Close"]) if i > 0 else close_px
        prev_bb_lower = float(ind.iloc[i - 1]["bb_lower"]) if i > 0 else bb_lower
        prev_bb_upper = float(ind.iloc[i - 1]["bb_upper"]) if i > 0 else bb_upper

        # BB-reclaim replaces the band-touch condition when enabled:
        # baseline: close <= bb_lower (entering AS price breaks the band)
        # reclaim:  prev_close <= prev_bb_lower AND close > bb_lower
        #           (entering AS price recovers back inside — the actual reversal bar)
        if params.bb_reclaim:
            bb_long_cond  = prev_close <= prev_bb_lower and close_px > bb_lower
            bb_short_cond = prev_close >= prev_bb_upper and close_px < bb_upper
        else:
            bb_long_cond  = close_px <= bb_lower
            bb_short_cond = close_px >= bb_upper

        # From-open drop gate: skip longs when stock has fallen too far from session open.
        rth_open_px = rth_open.get(bar_date)
        from_open_pct = ((close_px - rth_open_px) / rth_open_px * 100.0
                         if rth_open_px else 0.0)
        long_blocked_by_drop = (
            params.from_open_drop_max > 0
            and rth_open_px is not None
            and from_open_pct < -params.from_open_drop_max
        )

        # Direction gate intersection: params.sides (admin) ∩ day_direction (per-stock).
        _ddir = day_direction.get(bar_date, "both")
        _allow_long  = params.sides in ("both", "long_only")  and _ddir in ("both", "long_only")
        _allow_short = params.sides in ("both", "short_only") and _ddir in ("both", "short_only")

        # --- Long signal ---
        if (_allow_long
                and open_long is None
                and pending_long is None
                and long_cd == 0
                and not long_blocked_by_drop
                and rsi < params.rsi_long_thresh
                and bb_long_cond
                and vol_ratio >= params.vol_min
                and is_rev_long
                and rsi_turn_long):
            if params.entry_confirm:
                pending_long = {"signal_px": close_px, "atr": atr, "entry_indicators": snap}
            else:
                open_long = {
                    "entry": close_px,
                    "stop": close_px - atr * params.atr_stop_mult,
                    "target": close_px + atr * params.atr_target_mult,
                    "entry_time": bar_ts,
                    "hold": 0,
                    "entry_indicators": snap,
                }

        # --- Short signal ---
        if (_allow_short
                and open_short is None
                and pending_short is None
                and short_cd == 0
                and rsi > params.rsi_short_thresh
                and bb_short_cond
                and vol_ratio >= params.vol_min
                and is_rev_short
                and rsi_turn_short):
            if params.entry_confirm:
                pending_short = {"signal_px": close_px, "atr": atr, "entry_indicators": snap}
            else:
                open_short = {
                    "entry": close_px,
                    "stop": close_px + atr * params.atr_stop_mult,
                    "target": close_px - atr * params.atr_target_mult,
                    "entry_time": bar_ts,
                    "hold": 0,
                    "entry_indicators": snap,
                }

    # Final EOD flush — only when the last bar is at or after actual market
    # close (15:55 ET).  During live/shadow replay the last bar is the most
    # recent minute, NOT end-of-day; flushing at 15:08 would stamp every
    # still-open position with exit_reason="eod" even though the market is
    # still open and the position may resolve cleanly before 15:55.
    final_bar = df.iloc[-1]
    final_ts = df.index[-1]
    try:
        final_et = final_ts.tz_convert("America/New_York")
        final_minutes = final_et.hour * 60 + final_et.minute
    except Exception:
        # Naive index — localize as UTC first
        final_et = final_ts.tz_localize("UTC").tz_convert("America/New_York")
        final_minutes = final_et.hour * 60 + final_et.minute
    is_actual_eod = final_minutes >= 15 * 60 + 55   # 15:55 ET
    logger.info(
        "replay_eod_check %s last_bar=%s final_et=%s min=%d is_eod=%s open_long=%s open_short=%s",
        symbol, final_ts, final_et, final_minutes, is_actual_eod,
        open_long is not None, open_short is not None,
    )

    if is_actual_eod:
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


def _load_and_replay(
    sym: str,
    days: int,
    dsn: str,
    params: ScalpParams,
    end_date: str | None,
) -> list[Trade]:
    """Load bars and replay for one symbol — designed for thread-pool execution."""
    try:
        df = load_bars(sym, days, dsn, end_date=end_date)
        return replay_symbol_scalp(sym, df, params)
    except Exception as exc:
        logger.warning("load_bars(%s) failed: %s", sym, exc)
        return []


def _filter_consec_losses(
    trades: list[Trade],
    slot_size: float,
    max_consec: int = 3,
) -> list[Trade]:
    """Per-symbol: once a symbol accumulates max_consec consecutive losing
    trades on a given day, all subsequent same-day entries are dropped.
    Consecutive count resets to 0 on any winning trade.

    Within a single symbol there is no position overlap, so entry order
    equals exit order — a plain single-pass sort by entry_time is correct.
    """
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")

    # Group by symbol for independent per-symbol processing.
    by_sym: dict[str, list[Trade]] = {}
    for t in trades:
        by_sym.setdefault(t.symbol, []).append(t)

    result: list[Trade] = []
    for sym_trades in by_sym.values():
        consec: dict[str, int] = {}   # date → consecutive loss count
        blocked: dict[str, bool] = {} # date → blocked flag

        for trade in sorted(sym_trades, key=lambda t: t.entry_time):
            entry_et = trade.entry_time
            if hasattr(entry_et, "tz_convert"):
                entry_et = entry_et.tz_convert(_ET)
            date = entry_et.date().isoformat()

            if blocked.get(date):
                continue

            result.append(trade)

            # Update consecutive loss count using this trade's P&L.
            direction = 1 if trade.side == "long" else -1
            shares = max(1, int(slot_size / trade.entry_price)) if trade.entry_price > 0 else 1
            pnl = shares * (trade.exit_price - trade.entry_price) * direction

            if pnl < 0:
                consec[date] = consec.get(date, 0) + 1
                if consec[date] >= max_consec:
                    blocked[date] = True
            else:
                consec[date] = 0

    return result


def run(
    symbols: list[str] | None = None,
    days: int = 90,
    params: ScalpParams | None = None,
    dsn: str = DEFAULT_DSN,
    max_slots: int = MAX_SLOTS,
    capital_usd: int = CAPITAL_USD,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Run the scalp reversal backtest across all symbols and return report dict."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    symbols = symbols or DEFAULT_SYMBOLS
    params = params or ScalpParams()
    slot_size = capital_usd / max_slots

    all_trades: list[Trade] = []
    workers = min(len(symbols), 8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_load_and_replay, sym, days, dsn, params, end_date): sym
            for sym in symbols
        }
        for fut in as_completed(futures):
            all_trades.extend(fut.result())

    # Drop same-day entries after 3 consecutive losses per symbol.
    all_trades = _filter_consec_losses(all_trades, slot_size)

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
        if "exit_breakdown" not in m:
            m["exit_breakdown"] = {}
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
        "end_date": end_date,
    }
