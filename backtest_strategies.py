"""Per-strategy P&L attribution backtest.

Replays historical minute_bars from Postgres through the three
indicator-driven strategies (breakout, mean_reversion, momentum) and
reports per-strategy trade quality. Each strategy trades in isolation —
no ML veto, no brain gate, no risk manager. The question we answer is:
"what is this strategy's raw signal quality?"

News strategy is excluded (requires live RSS fetch; can't be replayed
deterministically from bars alone).

Exit model per trade:
- Stop loss or take profit hit first -> realized P&L = target hit
- 60-bar timeout -> exit at bar-60 close
- No overnight holds (close any open position at end of the symbol's
  data window)

Metrics per strategy:
- n_trades, win_rate, avg_return_pct, median_return_pct
- total_return_pct, max_drawdown_pct, sharpe (per-trade return / std)
- avg_hold_bars, stop_hit_pct, target_hit_pct, timeout_pct

Usage:
    python backtest_strategies.py --top-n 30 --days 90
    python backtest_strategies.py --symbols NVDA TSLA --days 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import ta
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backtest")

DEFAULT_DSN = "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev"
DEFAULT_TOP_N = 30
DEFAULT_DAYS = 90
DEFAULT_FREQUENCY = 5
MAX_HOLD_BARS = 60  # timeout (e.g. 60 * 5min = 5h for 5-min bars)


@dataclass
class Trade:
    strategy: str
    symbol: str
    side: str  # "long" | "short"
    entry_time: Any
    entry_price: float
    exit_time: Any
    exit_price: float
    exit_reason: str  # "stop" | "target" | "timeout" | "eod"
    hold_bars: int

    @property
    def return_pct(self) -> float:
        direction = 1 if self.side == "long" else -1
        return direction * (self.exit_price - self.entry_price) / self.entry_price * 100


# ---------------------------------------------------------------------------
# Precompute all indicators used by strategies, vectorised
# ---------------------------------------------------------------------------

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
    # Pivot-based resistance / support (daily session rolling 20)
    pivot = (high + low + close) / 3
    ind["resistance_1"] = 2 * pivot - low.rolling(20).min().shift(1)
    ind["support_1"] = 2 * pivot - high.rolling(20).max().shift(1)
    # 20-bar high/low — used by breakout-long and breakdown-short strategies.
    # shift(1) so the "20-bar high" excludes the current bar (avoids lookahead).
    ind["high_20"] = high.rolling(20).max().shift(1)
    ind["low_20"] = low.rolling(20).min().shift(1)
    # Volume ratio
    ind["volume_ratio"] = volume / (volume.rolling(20).mean() + 1e-10)
    return ind


# ---------------------------------------------------------------------------
# Strategy replay — call the real async strategy classes per bar
# ---------------------------------------------------------------------------

async def _run_strategy(strategy, market_data):
    return await strategy.generate_signal_with_commentary(market_data)


def load_strategies() -> list[tuple[str, Any]]:
    from core.commentary import CommentarySystem
    from strategies.builtin import (
        BreakoutStrategyWithCommentary,
        MeanReversionStrategyWithCommentary,
        MomentumStrategyWithCommentary,
    )
    # Silent commentary so backtest doesn't spam UI events
    cs = CommentarySystem()
    return [
        ("breakout",       BreakoutStrategyWithCommentary(cs)),
        ("mean_reversion", MeanReversionStrategyWithCommentary(cs)),
        ("momentum",       MomentumStrategyWithCommentary(cs)),
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
    # Per-strategy open position state
    open_pos: dict[str, Optional[dict]] = {name: None for name, _ in strategies}

    for i in range(50, len(df)):  # need enough warm-up for 50-bar SMAs
        bar = df.iloc[i]
        indicators = ind.iloc[i].to_dict()
        if any(pd.isna(v) for k, v in indicators.items()
               if k in ("rsi", "macd", "macd_signal", "bb_lower", "bb_middle",
                        "bb_upper", "atr", "adx", "sma_20", "sma_50",
                        "volume_ratio", "high_20", "low_20")):
            continue

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

            # Check exit on current bar if position is open
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
                    ))
                    open_pos[name] = None
                    pos = None

            # Consider new entry only if flat
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
                }

    # Close any leftover positions at final bar
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
            ))

    return trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def summarize(trades: list[Trade]) -> dict[str, Any]:
    if not trades:
        return {"n_trades": 0}
    returns = np.array([t.return_pct for t in trades])
    wins = returns > 0

    # Cross-symbol aggregate: sum of per-trade returns (each trade sized
    # equally). Compounding isn't meaningful across parallel symbol books.
    summed = float(np.sum(returns))

    # Running cumulative sum drawdown (simple additive equity curve)
    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    drawdown = cum - peak

    # Per-trade Sharpe (mean / std)
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10))
    reasons = [t.exit_reason for t in trades]

    wins_r = returns[wins]
    losses_r = returns[~wins]

    return {
        "n_trades": len(trades),
        "win_rate_pct": round(100 * float(np.mean(wins)), 2),
        "avg_return_pct": round(float(np.mean(returns)), 3),
        "median_return_pct": round(float(np.median(returns)), 3),
        "avg_win_pct":  round(float(np.mean(wins_r))  if len(wins_r)   else 0.0, 3),
        "avg_loss_pct": round(float(np.mean(losses_r)) if len(losses_r) else 0.0, 3),
        "sum_return_pct": round(summed, 2),
        "max_drawdown_pct": round(float(drawdown.min()), 2),
        "per_trade_sharpe": round(sharpe, 3),
        "profit_factor": round(
            float(wins_r.sum()) / max(1e-10, float(-losses_r.sum())), 3
        ) if len(losses_r) > 0 else None,
        "avg_hold_bars": round(float(np.mean([t.hold_bars for t in trades])), 1),
        "exit_breakdown": {
            "stop":    round(100 * reasons.count("stop")    / len(trades), 1),
            "target":  round(100 * reasons.count("target")  / len(trades), 1),
            "timeout": round(100 * reasons.count("timeout") / len(trades), 1),
            "eod":     round(100 * reasons.count("eod")     / len(trades), 1),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sym = p.add_mutually_exclusive_group()
    sym.add_argument("--symbols", nargs="+")
    sym.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--days", type=int, default=DEFAULT_DAYS)
    p.add_argument("--frequency", type=int, default=DEFAULT_FREQUENCY)
    p.add_argument("--dsn", default=os.environ.get("POSTGRES_DSN", DEFAULT_DSN))
    p.add_argument("--report-path", default="backtest_report.json")
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    from data_providers.postgres import PostgresDataProvider
    from train_ml_model import top_symbols_by_volume

    provider = PostgresDataProvider(args.dsn)
    strategies = load_strategies()

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = top_symbols_by_volume(args.dsn, args.top_n, args.days)
    logger.info("backtest_start symbols=%d days=%d frequency=%dmin",
                len(symbols), args.days, args.frequency)

    period_type = "day" if args.days < 30 else "month"
    period = args.days if period_type == "day" else max(1, args.days // 30)

    all_trades: list[Trade] = []
    t0 = datetime.now()
    for i, symbol in enumerate(symbols, 1):
        try:
            df = provider.get_market_data(
                symbol, period_type=period_type, period=period,
                frequency_type="minute", frequency=args.frequency,
            )
        except Exception as e:
            logger.error("load_failed symbol=%s err=%s", symbol, e)
            continue
        if df.empty or len(df) < 100:
            continue
        trades = await replay_symbol(symbol, df, strategies)
        all_trades.extend(trades)
        if i % 5 == 0 or i == len(symbols):
            logger.info("progress done=%d/%d cumulative_trades=%d",
                        i, len(symbols), len(all_trades))

    # Aggregate — keep (strategy, all), (strategy, long), (strategy, short)
    # so v-short-mirrors report can verify long-side stats didn't regress.
    by_strategy: dict[str, list[Trade]] = {name: [] for name, _ in strategies}
    for t in all_trades:
        by_strategy[t.strategy].append(t)

    def _split_summary(trades: list[Trade]) -> dict:
        longs = [t for t in trades if t.side == "long"]
        shorts = [t for t in trades if t.side == "short"]
        return {
            "all": summarize(trades),
            "long": summarize(longs),
            "short": summarize(shorts),
        }

    report = {
        "timestamp": datetime.now().isoformat(),
        "symbols": symbols,
        "symbol_count": len(symbols),
        "days": args.days,
        "frequency_minutes": args.frequency,
        "total_trades": len(all_trades),
        "elapsed_seconds": round((datetime.now() - t0).total_seconds(), 1),
        "per_strategy": {name: _split_summary(trades)
                         for name, trades in by_strategy.items()},
    }

    logger.info("backtest_done elapsed_s=%.1f total_trades=%d",
                report["elapsed_seconds"], report["total_trades"])

    print("\n" + "=" * 70)
    print(f"BACKTEST REPORT — {len(symbols)} symbols × {args.days} days "
          f"× {args.frequency}min bars")
    print("=" * 70)

    def _print_row(label: str, s: dict) -> None:
        if s["n_trades"] == 0:
            print(f"  {label:<7} (no trades)")
            return
        pf = s.get("profit_factor")
        pf_str = f"{pf:.2f}" if pf is not None else "N/A"
        print(f"  {label:<7} n={s['n_trades']:<5} "
              f"win%={s['win_rate_pct']:<5} "
              f"avg={s['avg_return_pct']:+.3f}% "
              f"pf={pf_str:<5} "
              f"sumR={s['sum_return_pct']:+7.2f}% "
              f"mdd={s['max_drawdown_pct']:+7.2f}% "
              f"exits(s/t/to)="
              f"{s['exit_breakdown']['stop']}/"
              f"{s['exit_breakdown']['target']}/"
              f"{s['exit_breakdown']['timeout']}%")

    for name in by_strategy:
        print(f"\n{name.upper()}")
        by_side = report["per_strategy"][name]
        _print_row("all",   by_side["all"])
        _print_row("long",  by_side["long"])
        _print_row("short", by_side["short"])
    print("=" * 70 + "\n")

    Path(args.report_path).write_text(json.dumps(report, indent=2, default=str))
    logger.info("report_written path=%s", args.report_path)
    return 0


def main() -> int:
    args = _parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    import sys
    sys.exit(main())
