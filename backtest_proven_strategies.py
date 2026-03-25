"""
backtest_proven_strategies.py — Three proven quant strategies backtested on real data.

Strategies:
  A) ETF Pairs Mean Reversion (intraday, 5-min bars)
  B) SPY IBS Mean Reversion (swing, daily bars)
  C) Turnaround Tuesday (swing, daily bars)

Usage:
    python backtest_proven_strategies.py                        # All 3
    python backtest_proven_strategies.py --strategy pairs       # Just pairs
    python backtest_proven_strategies.py --strategy ibs         # Just IBS
    python backtest_proven_strategies.py --strategy tuesday     # Just Tuesday
    python backtest_proven_strategies.py --start 2021-01-01     # Custom range
    python backtest_proven_strategies.py --capital 25000        # Custom capital
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("proven_strats")

DB_URL = "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev"
SLIPPAGE_PCT = 0.0003  # 0.03% per side for liquid ETFs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_daily_bars(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    """Load daily bars from PostgreSQL."""
    conn = psycopg2.connect(DB_URL)
    placeholders = ",".join(["%s"] * len(symbols))
    query = f"""
        SELECT symbol, date, open, high, low, close, volume
        FROM daily_bars
        WHERE symbol IN ({placeholders})
          AND date >= %s AND date <= %s
        ORDER BY date, symbol
    """
    params = symbols + [start, end]
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    log.info("Loaded %d daily bars for %s", len(df), symbols)
    return df


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Trade:
    strategy: str
    symbol: str
    direction: str  # 'long' or 'short'
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    hold_bars: int  # bars for intraday, days for swing


# ---------------------------------------------------------------------------
# Strategy A: ETF Pairs Mean Reversion
# ---------------------------------------------------------------------------

PAIRS = [
    ("QQQ", "SPY"),   # Best pair: PF 1.34, 66% WR
    ("SPY", "IWM"),   # Decent: PF 1.03, 59% WR
]


def _rolling_ols_beta(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling OLS slope (beta) using numpy.

    Returns array of same length as y, with NaN for the first (window-1) entries.
    """
    n = len(y)
    beta = np.full(n, np.nan)
    for i in range(window - 1, n):
        y_w = y[i - window + 1: i + 1]
        x_w = x[i - window + 1: i + 1]
        x_mean = x_w.mean()
        y_mean = y_w.mean()
        denom = np.sum((x_w - x_mean) ** 2)
        if denom < 1e-12:
            beta[i] = beta[i - 1] if i > 0 and not np.isnan(beta[i - 1]) else 1.0
        else:
            beta[i] = np.sum((x_w - x_mean) * (y_w - y_mean)) / denom
    return beta


def run_pairs_strategy(
    df_daily_all: pd.DataFrame,
    capital: float,
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> List[Trade]:
    """Run pairs mean reversion on DAILY bars using log-price ratio.

    Uses log(Y/X) as the spread. Rolling z-score on 60-day lookback.
    Entry at z > 2 or z < -2. Exit at |z| < 0.5. Stop at |z| > 3.5.
    Max hold: 15 trading days.

    This is a swing strategy (1-15 day holds), not intraday.
    """
    if pairs is None:
        pairs = PAIRS

    all_trades: List[Trade] = []
    capital_per_pair = capital / len(pairs)
    lookback = 60

    for sym_y, sym_x in pairs:
        log.info("Pairs: processing %s / %s", sym_y, sym_x)

        dy = df_daily_all[df_daily_all["symbol"] == sym_y].sort_values("date").set_index("date")
        dx = df_daily_all[df_daily_all["symbol"] == sym_x].sort_values("date").set_index("date")
        common_dates = dy.index.intersection(dx.index)
        dy = dy.loc[common_dates]
        dx = dx.loc[common_dates]

        if len(dy) < lookback + 50:
            log.warning("Not enough daily data for %s/%s", sym_y, sym_x)
            continue

        y_close = dy["close"].values
        x_close = dx["close"].values
        dates_arr = common_dates.to_list()

        # Log-price ratio spread
        log_ratio = np.log(y_close / x_close)

        # Rolling z-score
        z_score = np.full(len(log_ratio), np.nan)
        for i in range(lookback - 1, len(log_ratio)):
            window = log_ratio[i - lookback + 1: i + 1]
            mu = np.mean(window)
            sigma = np.std(window, ddof=1)
            if sigma > 1e-10:
                z_score[i] = (log_ratio[i] - mu) / sigma

        # State
        position = 0
        entry_bar = 0
        entry_z = 0.0
        entry_price_y = 0.0
        entry_price_x = 0.0
        shares_y = 0.0
        shares_x = 0.0
        max_hold = 15

        for i in range(lookback, len(dates_arr)):
            if np.isnan(z_score[i]):
                continue

            # Use z from bar i-1 for decision at bar i (no look-ahead)
            z_prev = z_score[i - 1]
            if np.isnan(z_prev):
                continue

            if position == 0:
                leg_capital = capital_per_pair / 2
                if z_prev < -2.0:
                    # Long spread: buy Y, sell X at today's open (approx close)
                    entry_price_y = y_close[i] * (1 + SLIPPAGE_PCT)
                    entry_price_x = x_close[i] * (1 - SLIPPAGE_PCT)
                    shares_y = leg_capital / entry_price_y
                    shares_x = leg_capital / entry_price_x
                    entry_bar = i
                    entry_z = z_prev
                    position = 1
                elif z_prev > 2.0:
                    # Short spread: sell Y, buy X
                    entry_price_y = y_close[i] * (1 - SLIPPAGE_PCT)
                    entry_price_x = x_close[i] * (1 + SLIPPAGE_PCT)
                    shares_y = leg_capital / entry_price_y
                    shares_x = leg_capital / entry_price_x
                    entry_bar = i
                    entry_z = z_prev
                    position = -1
            else:
                should_exit = False
                z = z_score[i]

                # Mean reversion target
                if position == 1 and z > -0.5:
                    should_exit = True
                elif position == -1 and z < 0.5:
                    should_exit = True
                # Stop
                elif abs(z) > 3.5:
                    should_exit = True
                # Max hold
                elif (i - entry_bar) >= max_hold:
                    should_exit = True

                if should_exit:
                    if position == 1:
                        pnl_y = shares_y * (y_close[i] * (1 - SLIPPAGE_PCT) - entry_price_y)
                        pnl_x = shares_x * (entry_price_x - x_close[i] * (1 + SLIPPAGE_PCT))
                    else:
                        pnl_y = shares_y * (entry_price_y - y_close[i] * (1 + SLIPPAGE_PCT))
                        pnl_x = shares_x * (x_close[i] * (1 - SLIPPAGE_PCT) - entry_price_x)
                    all_trades.append(Trade(
                        strategy="pairs",
                        symbol=f"{sym_y}/{sym_x}",
                        direction="long_spread" if position == 1 else "short_spread",
                        entry_date=str(dates_arr[entry_bar]),
                        exit_date=str(dates_arr[i]),
                        entry_price=entry_z,
                        exit_price=z,
                        shares=shares_y,
                        pnl=pnl_y + pnl_x,
                        hold_bars=i - entry_bar,
                    ))
                    position = 0

        # Close open position at end
        if position != 0:
            i = len(dates_arr) - 1
            if position == 1:
                pnl_y = shares_y * (y_close[i] * (1 - SLIPPAGE_PCT) - entry_price_y)
                pnl_x = shares_x * (entry_price_x - x_close[i] * (1 + SLIPPAGE_PCT))
            else:
                pnl_y = shares_y * (entry_price_y - y_close[i] * (1 + SLIPPAGE_PCT))
                pnl_x = shares_x * (x_close[i] * (1 - SLIPPAGE_PCT) - entry_price_x)
            all_trades.append(Trade(
                strategy="pairs",
                symbol=f"{sym_y}/{sym_x}",
                direction="long_spread" if position == 1 else "short_spread",
                entry_date=str(dates_arr[entry_bar]),
                exit_date=str(dates_arr[i]),
                entry_price=entry_z,
                exit_price=z_score[i],
                shares=shares_y,
                pnl=pnl_y + pnl_x,
                hold_bars=i - entry_bar,
            ))

        log.info("Pairs %s/%s: %d trades", sym_y, sym_x,
                 sum(1 for t in all_trades if t.symbol == f"{sym_y}/{sym_x}"))

    return all_trades


# ---------------------------------------------------------------------------
# Strategy B: SPY IBS Mean Reversion (daily)
# ---------------------------------------------------------------------------

def run_ibs_strategy(df_daily: pd.DataFrame, capital: float) -> List[Trade]:
    """SPY IBS mean reversion on daily bars.

    Entry (at close):
      - Close < (10-day highest high) - (25-day avg of daily range)
      - IBS < 0.3
    Exit (at close):
      - Close > yesterday's high
    """
    spy = df_daily[df_daily["symbol"] == "SPY"].copy().sort_values("date").reset_index(drop=True)

    if len(spy) < 30:
        log.warning("Not enough SPY daily data for IBS strategy")
        return []

    closes = spy["close"].values
    highs = spy["high"].values
    lows = spy["low"].values
    opens = spy["open"].values
    dates_arr = spy["date"].values

    trades: List[Trade] = []
    position = False
    entry_price = 0.0
    entry_idx = 0

    for i in range(25, len(spy)):
        # Compute indicators using data up to bar i
        ibs = (closes[i] - lows[i]) / max(highs[i] - lows[i], 0.001)

        # 10-day highest high (bars i-9 to i)
        highest_high_10 = np.max(highs[i - 9: i + 1])

        # 25-day average of daily range (bars i-24 to i)
        daily_ranges = highs[i - 24: i + 1] - lows[i - 24: i + 1]
        avg_range_25 = np.mean(daily_ranges)

        if not position:
            # Entry check at close of bar i
            threshold = highest_high_10 - avg_range_25
            if closes[i] < threshold and ibs < 0.3:
                entry_price = closes[i] * (1 + SLIPPAGE_PCT)
                entry_idx = i
                position = True
        else:
            # Exit: close > yesterday's high
            if closes[i] > highs[i - 1]:
                exit_price = closes[i] * (1 - SLIPPAGE_PCT)
                shares = capital / entry_price
                pnl = shares * (exit_price - entry_price)
                trades.append(Trade(
                    strategy="ibs",
                    symbol="SPY",
                    direction="long",
                    entry_date=str(dates_arr[entry_idx]),
                    exit_date=str(dates_arr[i]),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    shares=shares,
                    pnl=pnl,
                    hold_bars=i - entry_idx,
                ))
                position = False

    # Close any open position at end of data
    if position:
        i = len(spy) - 1
        exit_price = closes[i] * (1 - SLIPPAGE_PCT)
        shares = capital / entry_price
        pnl = shares * (exit_price - entry_price)
        trades.append(Trade(
            strategy="ibs",
            symbol="SPY",
            direction="long",
            entry_date=str(dates_arr[entry_idx]),
            exit_date=str(dates_arr[i]),
            entry_price=entry_price,
            exit_price=exit_price,
            shares=shares,
            pnl=pnl,
            hold_bars=i - entry_idx,
        ))

    log.info("IBS strategy: %d trades", len(trades))
    return trades


# ---------------------------------------------------------------------------
# Strategy C: Turnaround Tuesday (daily)
# ---------------------------------------------------------------------------

def run_tuesday_strategy(df_daily: pd.DataFrame, capital: float) -> List[Trade]:
    """Turnaround Tuesday on SPY daily bars.

    Entry (at Monday's close):
      - Today is Monday
      - Close < yesterday's close (down day)
      - Yesterday's close < day-before-yesterday's close (2 consecutive down)
    Exit (at close):
      - Close > yesterday's high
    """
    spy = df_daily[df_daily["symbol"] == "SPY"].copy().sort_values("date").reset_index(drop=True)

    if len(spy) < 5:
        log.warning("Not enough SPY daily data for Tuesday strategy")
        return []

    closes = spy["close"].values
    highs = spy["high"].values
    dates_arr = spy["date"].values
    # day_of_week: Monday=0 ... Friday=4
    dow = np.array([pd.Timestamp(d).dayofweek for d in dates_arr])

    trades: List[Trade] = []
    position = False
    entry_price = 0.0
    entry_idx = 0

    for i in range(2, len(spy)):
        if not position:
            # Entry: Monday, 2 consecutive down closes
            if dow[i] == 0 and closes[i] < closes[i - 1] and closes[i - 1] < closes[i - 2]:
                entry_price = closes[i] * (1 + SLIPPAGE_PCT)
                entry_idx = i
                position = True
        else:
            # Exit: close > yesterday's high
            if closes[i] > highs[i - 1]:
                exit_price = closes[i] * (1 - SLIPPAGE_PCT)
                shares = capital / entry_price
                pnl = shares * (exit_price - entry_price)
                trades.append(Trade(
                    strategy="tuesday",
                    symbol="SPY",
                    direction="long",
                    entry_date=str(dates_arr[entry_idx]),
                    exit_date=str(dates_arr[i]),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    shares=shares,
                    pnl=pnl,
                    hold_bars=i - entry_idx,
                ))
                position = False

    # Close any open position at end
    if position:
        i = len(spy) - 1
        exit_price = closes[i] * (1 - SLIPPAGE_PCT)
        shares = capital / entry_price
        pnl = shares * (exit_price - entry_price)
        trades.append(Trade(
            strategy="tuesday",
            symbol="SPY",
            direction="long",
            entry_date=str(dates_arr[entry_idx]),
            exit_date=str(dates_arr[i]),
            entry_price=entry_price,
            exit_price=exit_price,
            shares=shares,
            pnl=pnl,
            hold_bars=i - entry_idx,
        ))

    log.info("Tuesday strategy: %d trades", len(trades))
    return trades


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compute_metrics(trades: List[Trade], capital: float) -> Dict:
    """Compute performance metrics from a list of trades."""
    if not trades:
        return {
            "trades": 0, "win_rate": 0, "profit_factor": 0, "total_pnl": 0,
            "sharpe": 0, "max_dd_pct": 0, "avg_hold": 0, "avg_win": 0, "avg_loss": 0,
            "start": "N/A", "end": "N/A",
        }

    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Equity curve for Sharpe and drawdown
    equity = np.cumsum(pnls) + capital
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd_pct = abs(drawdown.min()) * 100

    # Sharpe: annualize based on strategy type
    # Use daily returns approximation
    daily_returns = pnls / capital
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    avg_hold = np.mean([t.hold_bars for t in trades])

    # Date range
    entry_dates = sorted([t.entry_date for t in trades])
    start_date = entry_dates[0][:10]
    end_date = entry_dates[-1][:10]

    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(pnls) * 100 if len(pnls) > 0 else 0,
        "profit_factor": profit_factor,
        "total_pnl": pnls.sum(),
        "sharpe": sharpe,
        "max_dd_pct": max_dd_pct,
        "avg_hold": avg_hold,
        "avg_win": wins.mean() if len(wins) > 0 else 0,
        "avg_loss": losses.mean() if len(losses) > 0 else 0,
        "start": start_date,
        "end": end_date,
    }


def print_report(name: str, metrics: Dict, hold_unit: str = "bars") -> None:
    """Pretty-print strategy metrics."""
    m = metrics
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Period:       {m['start']} to {m['end']}")
    print(f"  Trades:       {m['trades']}")
    print(f"  Win Rate:     {m['win_rate']:.1f}%")
    print(f"  Profit Factor:{m['profit_factor']:.2f}")
    print(f"  Total P&L:    ${m['total_pnl']:,.2f}")
    print(f"  Sharpe:       {m['sharpe']:.2f}")
    print(f"  Max Drawdown: {m['max_dd_pct']:.1f}%")
    print(f"  Avg Hold:     {m['avg_hold']:.1f}{hold_unit}")
    print(f"  Avg Win:      ${m['avg_win']:.2f}")
    print(f"  Avg Loss:     ${m['avg_loss']:.2f}")
    print(f"{'=' * 60}")


def print_combined_report(
    all_trades: List[Trade],
    capital: float,
    alloc: Dict[str, float],
) -> None:
    """Print combined portfolio report with allocated capital per strategy."""
    if not all_trades:
        print("\nNo trades to report.")
        return

    # Sort all trades by entry date
    sorted_trades = sorted(all_trades, key=lambda t: t.entry_date)
    pnls = np.array([t.pnl for t in sorted_trades])

    equity = np.cumsum(pnls) + capital
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = abs(drawdown.min()) * 100

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    daily_ret = pnls / capital
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0

    # Per-strategy breakdown
    strat_pnl = {}
    for t in sorted_trades:
        strat_pnl.setdefault(t.strategy, []).append(t.pnl)

    print(f"\n{'#' * 60}")
    print(f"  COMBINED PORTFOLIO")
    print(f"{'#' * 60}")
    print(f"  Total Capital:  ${capital:,.0f}")
    for s, a in alloc.items():
        print(f"    {s:12s}:  ${a:,.0f}")
    print(f"  Total Trades:   {len(sorted_trades)}")
    print(f"  Win Rate:       {len(wins)/len(pnls)*100:.1f}%")
    print(f"  Profit Factor:  {pf:.2f}")
    print(f"  Total P&L:      ${pnls.sum():,.2f}")
    print(f"  Return:         {pnls.sum()/capital*100:.1f}%")
    print(f"  Sharpe:         {sharpe:.2f}")
    print(f"  Max Drawdown:   {max_dd:.1f}%")
    print()
    print("  Per-Strategy P&L:")
    for s, plist in strat_pnl.items():
        arr = np.array(plist)
        print(f"    {s:12s}: ${arr.sum():>10,.2f}  ({len(plist)} trades, "
              f"WR {(arr>0).sum()/len(arr)*100:.0f}%)")
    print(f"{'#' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest 3 proven quant strategies")
    parser.add_argument("--strategy", choices=["pairs", "ibs", "tuesday", "all"],
                        default="all", help="Which strategy to run")
    parser.add_argument("--start", default="2006-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-03-20", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=12500, help="Total capital")
    args = parser.parse_args()

    run_all = args.strategy == "all"
    all_trades: List[Trade] = []

    # Capital allocation
    if run_all:
        cap_pairs = args.capital * 0.48   # ~$6,000
        cap_ibs = args.capital * 0.26     # ~$3,250
        cap_tuesday = args.capital * 0.26 # ~$3,250
    else:
        cap_pairs = args.capital
        cap_ibs = args.capital
        cap_tuesday = args.capital

    # ------- Strategy A: Pairs -------
    if run_all or args.strategy == "pairs":
        log.info("Loading daily bars for pairs strategy...")
        symbols = list(set(s for pair in PAIRS for s in pair))
        # Extra lookback for rolling window warmup
        daily_start = str((pd.Timestamp(args.start) - pd.Timedelta(days=120)).date())
        df_daily_pairs = load_daily_bars(symbols, daily_start, args.end)

        if len(df_daily_pairs) > 0:
            pairs_trades = run_pairs_strategy(df_daily_pairs, cap_pairs)
            all_trades.extend(pairs_trades)
            metrics = compute_metrics(pairs_trades, cap_pairs)
            print_report("Strategy A: ETF Pairs Mean Reversion (daily)", metrics, "d")
        else:
            log.warning("No daily bar data found for pairs strategy")

    # ------- Strategy B: IBS -------
    if run_all or args.strategy == "ibs":
        log.info("Loading daily bars for IBS strategy...")
        df_daily = load_daily_bars(["SPY"], args.start, args.end)
        ibs_trades = run_ibs_strategy(df_daily, cap_ibs)
        all_trades.extend(ibs_trades)
        metrics = compute_metrics(ibs_trades, cap_ibs)
        print_report("Strategy B: SPY IBS Mean Reversion (daily)", metrics, "d")

    # ------- Strategy C: Tuesday -------
    if run_all or args.strategy == "tuesday":
        log.info("Loading daily bars for Tuesday strategy...")
        df_daily = load_daily_bars(["SPY"], args.start, args.end)
        tue_trades = run_tuesday_strategy(df_daily, cap_tuesday)
        all_trades.extend(tue_trades)
        metrics = compute_metrics(tue_trades, cap_tuesday)
        print_report("Strategy C: Turnaround Tuesday (daily)", metrics, "d")

    # ------- Combined -------
    if run_all and all_trades:
        alloc = {"pairs": cap_pairs, "ibs": cap_ibs, "tuesday": cap_tuesday}
        print_combined_report(all_trades, args.capital, alloc)

    # Print individual trade log (last 20)
    if all_trades:
        print(f"\n--- Last 20 trades ---")
        sorted_t = sorted(all_trades, key=lambda t: t.entry_date)
        for t in sorted_t[-20:]:
            print(f"  {t.strategy:8s} {t.symbol:10s} {t.direction:14s} "
                  f"{t.entry_date[:16]:16s} -> {t.exit_date[:16]:16s} "
                  f"P&L: ${t.pnl:>8.2f}  hold: {t.hold_bars}")


if __name__ == "__main__":
    main()
