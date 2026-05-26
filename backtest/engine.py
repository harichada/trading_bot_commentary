"""Per-strategy P&L attribution backtest engine.

Replays historical ``minute_bars`` (loaded from Postgres by default)
through the three indicator-driven strategies in isolation — no ML
veto, no brain gate, no risk manager. News strategy is excluded
(requires live RSS fetch).

Exit model: stop or target hit first, else 60-bar timeout at close,
else EOD flush at the final bar close.

Behaviour is byte-identical to the previous monolithic
``backtest_strategies.py`` with one additive change: the engine now
snapshots ``MarketData.indicators`` into each ``Trade.entry_indicators``
at the moment the strategy emitted a signal.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import ta
from dotenv import load_dotenv

from backtest.metrics import summarize, summarize_per_symbol
from backtest.trades import BarsLoader, ProgressCb, ProgressEvent, Trade

load_dotenv()

logger = logging.getLogger("backtest")

DEFAULT_DSN = "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev"
DEFAULT_TOP_N = 30
DEFAULT_DAYS = 90
DEFAULT_FREQUENCY = 5
MAX_HOLD_BARS = 60  # timeout (e.g. 60 * 5min = 5h for 5-min bars)


def seed_run(seed: Optional[int]) -> None:
    """Seed ``random`` + ``numpy.random`` once per run.

    The replay loop has no RNG of its own — this is defensive plumbing
    so any future strategy/brain code path that consults an RNG stays
    reproducible. The determinism CI test asserts this wiring exists.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed like df, with columns every strategy reads."""
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]
    ind = pd.DataFrame(index=df.index)
    ind["sma_20"] = close.rolling(20).mean()
    ind["sma_50"] = close.rolling(50).mean()
    ind["rsi"] = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close)
    ind["macd"] = macd.macd()
    ind["macd_signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    ind["bb_upper"] = bb.bollinger_hband()
    ind["bb_middle"] = bb.bollinger_mavg()
    ind["bb_lower"] = bb.bollinger_lband()
    ind["atr"] = ta.volatility.average_true_range(high, low, close, window=14)
    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    ind["adx"] = adx.adx()
    pivot = (high + low + close) / 3
    ind["resistance_1"] = 2 * pivot - low.rolling(20).min().shift(1)
    ind["support_1"] = 2 * pivot - high.rolling(20).max().shift(1)
    # 20-bar high/low — used by breakout-long and breakdown-short strategies.
    # ``shift(1)`` so the "20-bar high" excludes the current bar (avoids lookahead).
    ind["high_20"] = high.rolling(20).max().shift(1)
    ind["low_20"] = low.rolling(20).min().shift(1)
    ind["volume_ratio"] = volume / (volume.rolling(20).mean() + 1e-10)
    # v-oversold-v2-bt-indicators-2026-05-19: scalar `prev_low` for v2's
    # Gate C bar-structure check. `lows_50` (a 50-bar array) is added
    # per-bar inside replay_symbol since it can't live in a flat DataFrame.
    ind["prev_low"] = low.shift(1)
    return ind


async def _run_strategy(strategy, market_data):
    return await strategy.generate_signal_with_commentary(market_data)


def load_strategies(force_v2: Optional[bool] = None) -> list[tuple[str, Any]]:
    """Mirror of engine.py's strategy selection.

    v-oversold-v2-bt-2026-05-19: when ``force_v2`` is None (default), reads
    ``Config().USE_OVERSOLD_BOUNCE_V2`` to decide whether to use the v2
    strategy. Pass ``force_v2=True`` or ``False`` to override for A/B
    comparison backtests without changing yaml.
    """
    from core.commentary import CommentarySystem
    from core.config import Config
    from strategies.builtin import (
        BreakoutStrategyWithCommentary,
        MeanReversionStrategyWithCommentary,
        MomentumStrategyWithCommentary,
    )
    cs = CommentarySystem()
    use_v2 = Config().USE_OVERSOLD_BOUNCE_V2 if force_v2 is None else force_v2
    if use_v2:
        from strategies.oversold_bounce_v2 import OversoldBounceV2Strategy
        mean_rev_tuple = ("oversold_v2", OversoldBounceV2Strategy(cs))
    else:
        mean_rev_tuple = ("mean_reversion", MeanReversionStrategyWithCommentary(cs))
    return [
        ("breakout", BreakoutStrategyWithCommentary(cs)),
        mean_rev_tuple,
        ("momentum", MomentumStrategyWithCommentary(cs)),
    ]


async def replay_symbol(
    symbol: str,
    df: pd.DataFrame,
    strategies: list[tuple[str, Any]],
) -> list[Trade]:
    """Iterate bars, invoke each strategy, track entries and exits."""
    from core.models import MarketData, SignalType

    ind = compute_indicators(df)
    trades: list[Trade] = []
    open_pos: dict[str, Optional[dict]] = {name: None for name, _ in strategies}

    # v-oversold-v2-bt-indicators-2026-05-19: precompute the raw low array
    # once so we can slice 50-bar windows per bar without re-indexing df.
    _low_arr = df["Low"].to_numpy()

    for i in range(50, len(df)):  # need enough warm-up for 50-bar SMAs
        bar = df.iloc[i]
        indicators = ind.iloc[i].to_dict()
        if any(pd.isna(v) for k, v in indicators.items()
               if k in ("rsi", "macd", "macd_signal", "bb_lower", "bb_middle",
                        "bb_upper", "atr", "adx", "sma_20", "sma_50",
                        "volume_ratio", "high_20", "low_20")):
            continue
        # v-oversold-v2-bt-indicators-2026-05-19: per-bar lows_50 list for
        # v2's Gate D swing-low check. Use a 50-bar lookback ending at the
        # previous bar (exclusive of current) to avoid lookahead.
        indicators["lows_50"] = _low_arr[max(0, i - 50):i].tolist()

        md = MarketData(
            symbol=symbol,
            timestamp=df.index[i],
            open=float(bar["Open"]),
            high=float(bar["High"]),
            low=float(bar["Low"]),
            close=float(bar["Close"]),
            volume=float(bar["Volume"]),
            indicators=indicators,
            timeframe="5m",
        )

        for name, strategy in strategies:
            pos = open_pos[name]

            if pos is not None:
                pos["hold"] += 1
                if pos["side"] == "long":
                    stop_hit = bar["Low"] <= pos["stop"]
                    target_hit = bar["High"] >= pos["target"]
                else:
                    stop_hit = bar["High"] >= pos["stop"]
                    target_hit = bar["Low"] <= pos["target"]

                exit_reason = None
                exit_price = None
                if stop_hit:
                    exit_price = pos["stop"]
                    exit_reason = "stop"
                elif target_hit:
                    exit_price = pos["target"]
                    exit_reason = "target"
                elif pos["hold"] >= MAX_HOLD_BARS:
                    exit_price = float(bar["Close"])
                    exit_reason = "timeout"

                if exit_reason is not None:
                    trades.append(Trade(
                        strategy=name, symbol=symbol, side=pos["side"],
                        entry_time=pos["entry_time"], entry_price=pos["entry"],
                        exit_time=df.index[i], exit_price=exit_price,
                        exit_reason=exit_reason, hold_bars=pos["hold"],
                        entry_indicators=pos.get("entry_indicators", {}),
                    ))
                    open_pos[name] = None
                    pos = None

            if pos is None:
                try:
                    signal = await _run_strategy(strategy, md)
                except Exception:
                    signal = None
                if signal is None:
                    continue
                open_pos[name] = {
                    "side": "long" if signal.signal_type == SignalType.BUY else "short",
                    "entry": float(signal.entry_price),
                    "stop": float(signal.stop_loss),
                    "target": float(signal.take_profit),
                    "entry_time": df.index[i],
                    "hold": 0,
                    # Snapshot indicators at signal time. Cast to floats to
                    # keep the dict JSON / CSV-friendly downstream.
                    # v-oversold-v2-bt-indicators-2026-05-19: skip list/array
                    # indicators (lows_50) — they can't be cast to float and
                    # pd.isna() raises on them.
                    "entry_indicators": {
                        k: float(v) for k, v in indicators.items()
                        if v is not None
                        and not isinstance(v, (list, tuple, np.ndarray))
                        and not pd.isna(v)
                    },
                }

    if len(df) > 0:
        final = df.iloc[-1]
        for name in open_pos:
            pos = open_pos[name]
            if pos is None:
                continue
            trades.append(Trade(
                strategy=name, symbol=symbol, side=pos["side"],
                entry_time=pos["entry_time"], entry_price=pos["entry"],
                exit_time=df.index[-1], exit_price=float(final["Close"]),
                exit_reason="eod", hold_bars=pos["hold"],
                entry_indicators=pos.get("entry_indicators", {}),
            ))

    return trades


def _default_bars_loader(dsn: str) -> BarsLoader:
    """Wrap a ``PostgresDataProvider`` in the ``BarsLoader`` signature."""
    from data_providers.postgres import PostgresDataProvider
    provider = PostgresDataProvider(dsn)

    def _load(symbol: str, days: int, frequency: int) -> pd.DataFrame:
        period_type = "day" if days < 30 else "month"
        period = days if period_type == "day" else max(1, days // 30)
        return provider.get_market_data(
            symbol, period_type=period_type, period=period,
            frequency_type="minute", frequency=frequency,
        )
    return _load


def _default_symbols_picker(dsn: str, top_n: int, days: int) -> list[str]:
    from train_ml_model import top_symbols_by_volume
    return top_symbols_by_volume(dsn, top_n, days)


async def _emit(progress_cb: Optional[ProgressCb], event: ProgressEvent) -> None:
    if progress_cb is None:
        return
    result = progress_cb(event)
    if asyncio.iscoroutine(result):
        await result


async def _run_backtest_with_trades(
    *,
    symbols: Optional[list[str]] = None,
    top_n: Optional[int] = None,
    days: int = DEFAULT_DAYS,
    frequency: int = DEFAULT_FREQUENCY,
    dsn: Optional[str] = None,
    bars_loader: Optional[BarsLoader] = None,
    progress_cb: Optional[ProgressCb] = None,
    run_id: Optional[str] = None,
    seed: Optional[int] = None,
    force_v2: Optional[bool] = None,
) -> tuple[dict[str, Any], list[Trade]]:
    """Internal entry point: returns ``(report, trades)``.

    The public ``run_backtest`` wrapper only returns the JSON-serialisable
    ``report`` to preserve the existing API contract (``api/backtest.py``
    serialises it with ``json.dumps(... default=str)`` — embedding
    dataclasses would break that). Callers that need the raw trade list
    (CLI for CSV export, determinism tests) use this private helper.
    """
    if (symbols is None) == (top_n is None):
        raise ValueError("provide exactly one of: symbols, top_n")
    if days < 1 or days > 365:
        raise ValueError("days must be in [1, 365]")
    if frequency not in (1, 5, 15):
        raise ValueError("frequency must be one of {1, 5, 15}")

    seed_run(seed)

    effective_dsn = dsn or os.environ.get("POSTGRES_DSN", DEFAULT_DSN)

    if bars_loader is None:
        bars_loader = _default_bars_loader(effective_dsn)

    if symbols is None:
        symbols = _default_symbols_picker(effective_dsn, top_n, days)
    else:
        symbols = [s.upper() for s in symbols]

    strategies = load_strategies(force_v2=force_v2)
    logger.info("backtest_start run_id=%s symbols=%d days=%d frequency=%dmin",
                run_id, len(symbols), days, frequency)

    all_trades: list[Trade] = []
    t0 = datetime.now()
    total = len(symbols)

    for i, symbol in enumerate(symbols, 1):
        await _emit(progress_cb, ProgressEvent(
            stage="loading_data", symbols_done=i - 1,
            symbols_total=total, current_symbol=symbol,
        ))
        try:
            df = bars_loader(symbol, days, frequency)
        except Exception as e:
            logger.error("load_failed symbol=%s err=%s", symbol, e)
            continue
        if df.empty or len(df) < 100:
            continue

        await _emit(progress_cb, ProgressEvent(
            stage="replaying", symbols_done=i - 1,
            symbols_total=total, current_symbol=symbol,
        ))
        trades = await replay_symbol(symbol, df, strategies)
        all_trades.extend(trades)
        if i % 5 == 0 or i == total:
            logger.info("progress done=%d/%d cumulative_trades=%d",
                        i, total, len(all_trades))

    # Final progress beat — all symbols processed.
    if total > 0:
        await _emit(progress_cb, ProgressEvent(
            stage="replaying", symbols_done=total,
            symbols_total=total, current_symbol=symbols[-1],
        ))

    by_strategy: dict[str, list[Trade]] = {name: [] for name, _ in strategies}
    for t in all_trades:
        by_strategy[t.strategy].append(t)

    def _split_summary(trades: list[Trade]) -> dict:
        longs = [t for t in trades if t.side == "long"]
        shorts = [t for t in trades if t.side == "short"]
        all_summary = summarize(trades)
        # Add per_symbol drill-down only on the ``all`` aggregate. The long/short
        # mirrors stay slim because the UI doesn't drill into them.
        if trades:
            all_summary["per_symbol"] = summarize_per_symbol(trades)
        return {
            "all": all_summary,
            "long": summarize(longs),
            "short": summarize(shorts),
        }

    elapsed = round((datetime.now() - t0).total_seconds(), 1)
    report: dict[str, Any] = {
        "run_id": run_id or f"bt_{t0.strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "symbols": symbols,
        "symbol_count": len(symbols),
        "days": days,
        "frequency_minutes": frequency,
        "total_trades": len(all_trades),
        "elapsed_seconds": elapsed,
        "per_strategy": {name: _split_summary(trades)
                         for name, trades in by_strategy.items()},
    }
    logger.info("backtest_done run_id=%s elapsed_s=%.1f total_trades=%d",
                report["run_id"], elapsed, len(all_trades))
    return report, all_trades


async def run_backtest(
    *,
    symbols: Optional[list[str]] = None,
    top_n: Optional[int] = None,
    days: int = DEFAULT_DAYS,
    frequency: int = DEFAULT_FREQUENCY,
    dsn: Optional[str] = None,
    bars_loader: Optional[BarsLoader] = None,
    progress_cb: Optional[ProgressCb] = None,
    run_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Run the per-strategy backtest and return a JSON-serialisable report.

    Exactly one of ``symbols`` or ``top_n`` must be provided. ``bars_loader``
    is injectable for tests; it defaults to a Postgres-backed loader using
    ``dsn`` (or ``POSTGRES_DSN`` env var, or ``DEFAULT_DSN``).

    ``seed`` is optional; when provided it seeds ``random`` and
    ``numpy.random`` before the replay loop runs. The replay loop itself
    is deterministic given identical inputs — seeding is plumbing for
    forward-compat with any strategy/brain code paths that might consult
    an RNG.

    Returns only the report dict (no trades) to preserve binary compat
    with ``api/backtest.py``. Callers needing both use
    ``_run_backtest_with_trades`` directly.
    """
    report, _ = await _run_backtest_with_trades(
        symbols=symbols, top_n=top_n, days=days, frequency=frequency,
        dsn=dsn, bars_loader=bars_loader, progress_cb=progress_cb,
        run_id=run_id, seed=seed,
    )
    return report
