"""
Backtrader-based gap fade backtester with chart generation.

Runs on 1-20 user-selected symbols for visual analysis (equity curves,
drawdown plots, per-symbol trade charts).  The custom SQL-based
GapFadeBacktester in gap_fade_app.py remains the primary engine for
full-universe runs.

Usage:
    python bt_backtest.py                     # quick test with AAPL + TSLA
    python bt_backtest.py NVDA MSFT 2023-01-01 2024-06-01
"""
from __future__ import annotations

import logging
import os
import re
import psycopg2
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    import backtrader as bt
    HAS_BACKTRADER = True
except ImportError:
    HAS_BACKTRADER = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. SQLite data loader
# ---------------------------------------------------------------------------

DB_URL_DEFAULT = os.environ.get(
    'DATABASE_URL', 'postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev'
)


def load_symbol_data(
    symbol: str,
    start: str,
    end: str,
    db_url: str = DB_URL_DEFAULT,
) -> Optional["bt.feeds.PandasData"]:
    """Load daily bars from PostgreSQL and return a backtrader PandasData feed.

    Returns *None* if the symbol has no data in the requested range.
    """
    if not HAS_BACKTRADER:
        raise RuntimeError('backtrader is not installed')

    conn = psycopg2.connect(db_url)
    try:
        df = pd.read_sql_query(
            'SELECT date, open, high, low, close, volume '
            'FROM daily_bars WHERE symbol = %s AND date BETWEEN %s AND %s '
            'ORDER BY date',
            conn,
            params=(symbol.upper(), start, end),
        )
    finally:
        conn.close()

    if df.empty:
        logger.warning('No data for %s in %s – %s', symbol, start, end)
        return None

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.columns = [c.lower() for c in df.columns]

    feed = bt.feeds.PandasData(dataname=df, name=symbol)
    return feed


# ---------------------------------------------------------------------------
# 2. Gap Fade strategy for backtrader
# ---------------------------------------------------------------------------

class GapFadeBTStrategy(bt.Strategy):
    """Gap fade (short gap-ups / long gap-downs) implemented as a
    backtrader strategy for charting and analyzer support.

    Simplifications vs the production SQL engine:
    - No re-entry after stop-out
    - No LLM supervisor / catalyst detection / market regime filter
    - Fixed risk_pct sizing (no Kelly / per-slot equity cap)
    """

    params = dict(
        # Gap detection
        gap_threshold=0.07,
        max_gap_pct=0.50,
        vol_ratio_max=3.0,
        vol_window=20,
        # Stops
        stop_pct=0.015,
        adaptive_stops=True,
        stop_gap_fraction=0.25,
        stop_min_pct=0.015,
        stop_max_pct=0.05,
        # Targets
        partial_cover_frac=0.33,
        # Gap-down longs
        trade_gap_downs=False,
        gap_down_threshold=0.05,
        # Sizing
        risk_pct=0.02,
        max_positions=5,
        # Slippage / costs
        slippage_pct=0.0015,
    )

    # -- lifecycle ----------------------------------------------------------

    def __init__(self):
        # Per-data tracking
        self._vol_sma: Dict[str, Any] = {}
        self._prev_close: Dict[str, float] = {}
        self._positions_meta: Dict[str, dict] = {}  # symbol → {entry, stop, target, partial_target, direction, partial_done, hwm}
        self._open_count = 0

        # Trade log for chart markers
        self.trade_log: List[dict] = []  # {date, symbol, side, price, action}

        # Equity curve (recorded at each bar)
        self.equity_curve: List[dict] = []  # {date, value}

        for d in self.datas:
            name = d._name
            self._vol_sma[name] = bt.indicators.SMA(d.volume, period=self.p.vol_window)

    # -- main loop ----------------------------------------------------------

    def next(self):
        # Record equity
        dt = self.datas[0].datetime.date(0)
        self.equity_curve.append({
            'date': dt,
            'value': self.broker.getvalue(),
        })

        for d in self.datas:
            name = d._name
            pos = self.getposition(d)

            # --- exit logic for existing positions -------------------------
            if pos.size != 0:
                meta = self._positions_meta.get(name)
                if not meta:
                    continue
                self._check_exit(d, pos, meta)
                continue

            # --- entry logic -----------------------------------------------
            if self._open_count >= self.p.max_positions:
                continue

            # Need at least vol_window+1 bars to compute gap
            if len(d) < self.p.vol_window + 2:
                continue

            prev_close = d.close[-1]
            if prev_close <= 0:
                continue

            gap_pct = (d.open[0] - prev_close) / prev_close

            # Check volume filter
            avg_vol = self._vol_sma[name][0]
            if avg_vol > 0 and d.volume[0] > 0:
                vol_ratio = d.volume[0] / avg_vol
            else:
                vol_ratio = 0

            # --- Gap-up short ---
            if (gap_pct >= self.p.gap_threshold
                    and gap_pct <= self.p.max_gap_pct
                    and (avg_vol <= 0 or vol_ratio <= self.p.vol_ratio_max)):
                self._enter_short(d, gap_pct, prev_close)

            # --- Gap-down long ---
            elif (self.p.trade_gap_downs
                  and gap_pct <= -self.p.gap_down_threshold
                  and abs(gap_pct) <= self.p.max_gap_pct
                  and (avg_vol <= 0 or vol_ratio <= self.p.vol_ratio_max)):
                self._enter_long(d, gap_pct, prev_close)

    # -- entry helpers ------------------------------------------------------

    def _compute_stop(self, gap_pct_abs: float) -> float:
        if self.p.adaptive_stops:
            stop = gap_pct_abs * self.p.stop_gap_fraction
            return max(self.p.stop_min_pct, min(stop, self.p.stop_max_pct))
        return self.p.stop_pct

    def _size_for_risk(self, price: float, stop_dist_pct: float) -> int:
        equity = self.broker.getvalue()
        risk_dollars = equity * self.p.risk_pct
        stop_dist = price * stop_dist_pct
        if stop_dist <= 0:
            return 0
        shares = int(risk_dollars / stop_dist)
        return max(1, shares)

    def _enter_short(self, d, gap_pct: float, prev_close: float):
        name = d._name
        entry = d.open[0]
        stop_dist = self._compute_stop(abs(gap_pct))
        size = self._size_for_risk(entry, stop_dist)
        if size <= 0:
            return

        self.sell(data=d, size=size)

        stop_price = entry * (1 + stop_dist)
        partial_target = entry - (entry - prev_close) * 0.5
        full_target = prev_close

        self._positions_meta[name] = {
            'direction': 'short',
            'entry': entry,
            'stop': stop_price,
            'partial_target': partial_target,
            'full_target': full_target,
            'size': size,
            'partial_done': False,
            'hwm': entry,
        }
        self._open_count += 1
        self.trade_log.append({
            'date': d.datetime.date(0),
            'symbol': name,
            'side': 'short',
            'price': entry,
            'action': 'entry',
        })

    def _enter_long(self, d, gap_pct: float, prev_close: float):
        name = d._name
        entry = d.open[0]
        stop_dist = self._compute_stop(abs(gap_pct))
        size = self._size_for_risk(entry, stop_dist)
        if size <= 0:
            return

        self.buy(data=d, size=size)

        stop_price = entry * (1 - stop_dist)
        partial_target = entry + (prev_close - entry) * 0.5
        full_target = prev_close

        self._positions_meta[name] = {
            'direction': 'long',
            'entry': entry,
            'stop': stop_price,
            'partial_target': partial_target,
            'full_target': full_target,
            'size': size,
            'partial_done': False,
            'hwm': entry,
        }
        self._open_count += 1
        self.trade_log.append({
            'date': d.datetime.date(0),
            'symbol': name,
            'side': 'long',
            'price': entry,
            'action': 'entry',
        })

    # -- exit logic ---------------------------------------------------------

    def _check_exit(self, d, pos, meta: dict):
        name = d._name
        direction = meta['direction']
        h, l, c = d.high[0], d.low[0], d.close[0]

        if direction == 'short':
            # Update high water mark (lowest price = best for short)
            if l < meta['hwm']:
                meta['hwm'] = l

            # Stop hit?
            if h >= meta['stop']:
                self.close(data=d)
                self._on_exit(d, meta['stop'], 'stop')
                return

            # Partial cover
            if not meta['partial_done'] and l <= meta['partial_target']:
                cover_size = max(1, int(meta['size'] * self.p.partial_cover_frac))
                if cover_size < abs(pos.size):
                    self.buy(data=d, size=cover_size)
                    meta['partial_done'] = True
                    # Move stop to breakeven
                    meta['stop'] = meta['entry']
                    self.trade_log.append({
                        'date': d.datetime.date(0),
                        'symbol': name,
                        'side': 'short',
                        'price': meta['partial_target'],
                        'action': 'partial',
                    })

            # Full target (gap fill)
            if l <= meta['full_target']:
                self.close(data=d)
                self._on_exit(d, meta['full_target'], 'target')
                return

        else:  # long
            if h > meta['hwm']:
                meta['hwm'] = h

            # Stop hit?
            if l <= meta['stop']:
                self.close(data=d)
                self._on_exit(d, meta['stop'], 'stop')
                return

            # Partial cover
            if not meta['partial_done'] and h >= meta['partial_target']:
                cover_size = max(1, int(meta['size'] * self.p.partial_cover_frac))
                if cover_size < abs(pos.size):
                    self.sell(data=d, size=cover_size)
                    meta['partial_done'] = True
                    meta['stop'] = meta['entry']
                    self.trade_log.append({
                        'date': d.datetime.date(0),
                        'symbol': name,
                        'side': 'long',
                        'price': meta['partial_target'],
                        'action': 'partial',
                    })

            # Full target
            if h >= meta['full_target']:
                self.close(data=d)
                self._on_exit(d, meta['full_target'], 'target')
                return

    def _on_exit(self, d, price: float, reason: str):
        name = d._name
        meta = self._positions_meta.pop(name, {})
        self._open_count = max(0, self._open_count - 1)
        self.trade_log.append({
            'date': d.datetime.date(0),
            'symbol': name,
            'side': meta.get('direction', 'short'),
            'price': price,
            'action': reason,
        })

    def notify_trade(self, trade):
        if trade.isclosed:
            logger.debug(
                'Trade closed: %s pnl=%.2f',
                trade.data._name,
                trade.pnl,
            )

    def stop(self):
        """Called when backtest ends — close any remaining positions."""
        for d in self.datas:
            if self.getposition(d).size != 0:
                name = d._name
                self.close(data=d)
                meta = self._positions_meta.pop(name, {})
                self._open_count = max(0, self._open_count - 1)
                self.trade_log.append({
                    'date': d.datetime.date(0),
                    'symbol': name,
                    'side': meta.get('direction', 'short'),
                    'price': d.close[0],
                    'action': 'eod_close',
                })


# ---------------------------------------------------------------------------
# 3. Chart generation
# ---------------------------------------------------------------------------

CHART_STYLE = {
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#444',
    'axes.labelcolor': '#ccc',
    'text.color': '#ccc',
    'xtick.color': '#999',
    'ytick.color': '#999',
    'grid.color': '#333',
    'grid.alpha': 0.5,
}


def _apply_style():
    plt.rcParams.update(CHART_STYLE)


def generate_equity_chart(strat: GapFadeBTStrategy, output_dir: str) -> str:
    """Generate equity curve PNG. Returns file path."""
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    ec = strat.equity_curve
    if not ec:
        return ''

    dates = [e['date'] for e in ec]
    values = [e['value'] for e in ec]
    start_val = values[0] if values else 0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, values, color='#00d4aa', linewidth=1.5, label='Portfolio Value')
    ax.axhline(start_val, color='#666', linestyle='--', linewidth=0.8, alpha=0.7)

    # Fill green above start, red below
    ax.fill_between(dates, values, start_val,
                     where=[v >= start_val for v in values],
                     color='#00d4aa', alpha=0.15)
    ax.fill_between(dates, values, start_val,
                     where=[v < start_val for v in values],
                     color='#ff4757', alpha=0.15)

    final_val = values[-1]
    total_return = (final_val - start_val) / start_val * 100
    n_trades = len([t for t in strat.trade_log if t['action'] == 'entry'])

    ax.set_title(
        f'Equity Curve  |  {dates[0]} → {dates[-1]}  |  '
        f'Return: {total_return:+.1f}%  |  Trades: {n_trades}',
        fontsize=13, fontweight='bold',
    )
    ax.set_ylabel('Portfolio Value ($)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)
    ax.grid(True)
    ax.legend(loc='upper left')

    path = os.path.join(output_dir, 'equity_curve.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path


def generate_drawdown_chart(strat: GapFadeBTStrategy, output_dir: str) -> str:
    """Generate drawdown chart PNG. Returns file path."""
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    ec = strat.equity_curve
    if not ec:
        return ''

    dates = [e['date'] for e in ec]
    values = np.array([e['value'] for e in ec])

    # Compute drawdown
    running_max = np.maximum.accumulate(values)
    dd = (values - running_max) / running_max * 100  # in percent

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dates, dd, 0, color='#ff4757', alpha=0.4)
    ax.plot(dates, dd, color='#ff4757', linewidth=1.0)

    # Annotate max drawdown
    max_dd_idx = np.argmin(dd)
    max_dd_val = dd[max_dd_idx]
    ax.annotate(
        f'Max DD: {max_dd_val:.1f}%',
        xy=(dates[max_dd_idx], max_dd_val),
        xytext=(30, -20),
        textcoords='offset points',
        fontsize=10,
        color='#ff4757',
        fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#ff4757', lw=1.5),
    )

    ax.set_title('Drawdown', fontsize=13, fontweight='bold')
    ax.set_ylabel('Drawdown (%)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)
    ax.grid(True)

    path = os.path.join(output_dir, 'drawdown.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path


def generate_trade_chart(
    strat: GapFadeBTStrategy,
    data_feed: "bt.feeds.PandasData",
    output_dir: str,
) -> str:
    """Generate per-symbol trade chart with buy/sell markers. Returns file path."""
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    name = data_feed._name
    # Extract OHLC from the data feed
    dates = []
    closes = []
    for i in range(-len(data_feed) + 1, 1):
        try:
            dt = data_feed.datetime.date(i)
            cl = data_feed.close[i]
            dates.append(dt)
            closes.append(cl)
        except IndexError:
            break

    if not dates:
        return ''

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, closes, color='#8884d8', linewidth=1.2, label=f'{name} Close')

    # Overlay trade markers
    symbol_trades = [t for t in strat.trade_log if t['symbol'] == name]
    for t in symbol_trades:
        if t['action'] == 'entry':
            if t['side'] == 'short':
                ax.scatter(t['date'], t['price'], marker='v', color='#ff4757',
                          s=80, zorder=5, label='Short Entry' if 'Short Entry' not in ax.get_legend_handles_labels()[1] else '')
            else:
                ax.scatter(t['date'], t['price'], marker='^', color='#00d4aa',
                          s=80, zorder=5, label='Long Entry' if 'Long Entry' not in ax.get_legend_handles_labels()[1] else '')
        elif t['action'] in ('stop', 'target', 'eod_close'):
            ax.scatter(t['date'], t['price'], marker='o', color='#4a9eff',
                      s=60, zorder=5, label='Exit' if 'Exit' not in ax.get_legend_handles_labels()[1] else '')
        elif t['action'] == 'partial':
            ax.scatter(t['date'], t['price'], marker='D', color='#ffa502',
                      s=50, zorder=5, label='Partial' if 'Partial' not in ax.get_legend_handles_labels()[1] else '')

    n_sym_trades = len([t for t in symbol_trades if t['action'] == 'entry'])
    ax.set_title(f'{name}  |  {n_sym_trades} Trades', fontsize=13, fontweight='bold')
    ax.set_ylabel('Price ($)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)
    ax.grid(True)
    ax.legend(loc='upper left', fontsize=9)

    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    path = os.path.join(output_dir, f'trades_{safe_name}.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 4. Runner
# ---------------------------------------------------------------------------

def run_backtrader_backtest(
    symbols: List[str],
    start_date: str,
    end_date: str,
    config: Dict[str, Any],
    db_url: str = DB_URL_DEFAULT,
    charts_dir: str = 'bt_charts',
) -> dict:
    """Run backtrader backtest and generate charts.

    Args:
        symbols: List of ticker symbols (max 20).
        start_date: 'YYYY-MM-DD' start.
        end_date: 'YYYY-MM-DD' end.
        config: Dict of strategy param overrides (keys match GapFadeBTStrategy.params).
        db_url: PostgreSQL connection URL.
        charts_dir: Directory for PNG output.

    Returns:
        dict with keys: status, metrics, charts, trades, errors
    """
    if not HAS_BACKTRADER:
        return {'status': 'error', 'error': 'backtrader is not installed. Run: pip install backtrader'}

    if not symbols:
        return {'status': 'error', 'error': 'No symbols provided'}
    if len(symbols) > 20:
        return {'status': 'error', 'error': f'Too many symbols ({len(symbols)}). Maximum is 20.'}

    cerebro = bt.Cerebro()

    # Broker settings
    initial_capital = config.get('initial_capital', 25_000)
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.set_shortcash(True)  # allow shorting

    slippage = config.get('slippage_pct', 0.0015)
    cerebro.broker.set_slippage_perc(slippage)
    cerebro.broker.setcommission(commission=0.0)  # commission-free

    # Load data feeds
    loaded = []
    errors = []
    for sym in symbols:
        sym = sym.strip().upper()
        if not sym:
            continue
        try:
            feed = load_symbol_data(sym, start_date, end_date, db_url)
            if feed is not None:
                cerebro.adddata(feed)
                loaded.append(sym)
            else:
                errors.append(f'{sym}: no data in range')
        except Exception as e:
            errors.append(f'{sym}: {e}')

    if not loaded:
        return {'status': 'error', 'error': 'No symbols had data in the requested range', 'errors': errors}

    # Strategy params from config
    strat_params = {}
    valid_params = set(GapFadeBTStrategy.params._getkeys())
    for k, v in config.items():
        if k in valid_params:
            strat_params[k] = v

    cerebro.addstrategy(GapFadeBTStrategy, **strat_params)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                        timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # Run
    logger.info('Running backtrader: %d symbols, %s → %s', len(loaded), start_date, end_date)
    results = cerebro.run()
    strat = results[0]

    # Extract metrics
    metrics = _extract_metrics(strat, initial_capital)

    # Generate charts
    os.makedirs(charts_dir, exist_ok=True)
    charts = {}

    eq_path = generate_equity_chart(strat, charts_dir)
    if eq_path:
        charts['equity_curve'] = os.path.basename(eq_path)

    dd_path = generate_drawdown_chart(strat, charts_dir)
    if dd_path:
        charts['drawdown'] = os.path.basename(dd_path)

    # Per-symbol trade charts (up to 10 most traded)
    symbol_trade_counts = {}
    for t in strat.trade_log:
        if t['action'] == 'entry':
            symbol_trade_counts[t['symbol']] = symbol_trade_counts.get(t['symbol'], 0) + 1

    top_symbols = sorted(symbol_trade_counts, key=symbol_trade_counts.get, reverse=True)[:10]
    for d in strat.datas:
        if d._name in top_symbols:
            tp = generate_trade_chart(strat, d, charts_dir)
            if tp:
                charts[f'trades_{d._name}'] = os.path.basename(tp)

    # Trade summary
    trade_summary = _build_trade_summary(strat)

    return {
        'status': 'ok',
        'metrics': metrics,
        'charts': charts,
        'trades': trade_summary,
        'symbols_loaded': loaded,
        'errors': errors,
    }


def _extract_metrics(strat: GapFadeBTStrategy, initial_capital: float) -> dict:
    """Pull metrics from backtrader analyzers."""
    metrics = {}

    # Sharpe
    try:
        sharpe = strat.analyzers.sharpe.get_analysis()
        metrics['sharpe_ratio'] = round(sharpe.get('sharperatio', 0) or 0, 3)
    except Exception:
        metrics['sharpe_ratio'] = 0

    # Drawdown
    try:
        dd = strat.analyzers.drawdown.get_analysis()
        metrics['max_drawdown_pct'] = round(dd.get('max', {}).get('drawdown', 0) or 0, 2)
        metrics['max_drawdown_len'] = dd.get('max', {}).get('len', 0) or 0
    except Exception:
        metrics['max_drawdown_pct'] = 0
        metrics['max_drawdown_len'] = 0

    # Trade analyzer
    try:
        ta = strat.analyzers.trades.get_analysis()
        total = ta.get('total', {}).get('total', 0) or 0
        won = ta.get('won', {}).get('total', 0) or 0
        lost = ta.get('lost', {}).get('total', 0) or 0
        metrics['total_trades'] = total
        metrics['won'] = won
        metrics['lost'] = lost
        metrics['win_rate'] = round(won / total * 100, 1) if total > 0 else 0

        pnl_won = ta.get('won', {}).get('pnl', {}).get('total', 0) or 0
        pnl_lost = ta.get('lost', {}).get('pnl', {}).get('total', 0) or 0
        metrics['gross_profit'] = round(pnl_won, 2)
        metrics['gross_loss'] = round(pnl_lost, 2)
        metrics['net_pnl'] = round(pnl_won + pnl_lost, 2)
        metrics['profit_factor'] = round(abs(pnl_won / pnl_lost), 2) if pnl_lost != 0 else 0

        avg_won = ta.get('won', {}).get('pnl', {}).get('average', 0) or 0
        avg_lost = ta.get('lost', {}).get('pnl', {}).get('average', 0) or 0
        metrics['avg_win'] = round(avg_won, 2)
        metrics['avg_loss'] = round(avg_lost, 2)
    except Exception:
        metrics.update({
            'total_trades': 0, 'won': 0, 'lost': 0, 'win_rate': 0,
            'gross_profit': 0, 'gross_loss': 0, 'net_pnl': 0,
            'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0,
        })

    # Returns
    try:
        ret = strat.analyzers.returns.get_analysis()
        metrics['total_return_pct'] = round((ret.get('rtot', 0) or 0) * 100, 2)
    except Exception:
        metrics['total_return_pct'] = 0

    # Final equity
    final_value = strat.broker.getvalue()
    metrics['final_equity'] = round(final_value, 2)
    metrics['initial_capital'] = initial_capital

    return metrics


def _build_trade_summary(strat: GapFadeBTStrategy) -> List[dict]:
    """Build a simplified trade list from trade_log."""
    entries = {}
    trades = []
    for t in strat.trade_log:
        sym = t['symbol']
        if t['action'] == 'entry':
            entries[sym] = t
        elif t['action'] in ('stop', 'target', 'eod_close') and sym in entries:
            entry = entries.pop(sym)
            pnl_pct = 0
            if entry['side'] == 'short':
                pnl_pct = (entry['price'] - t['price']) / entry['price'] * 100
            else:
                pnl_pct = (t['price'] - entry['price']) / entry['price'] * 100

            trades.append({
                'symbol': sym,
                'side': entry['side'],
                'entry_date': str(entry['date']),
                'exit_date': str(t['date']),
                'entry_price': round(entry['price'], 2),
                'exit_price': round(t['price'], 2),
                'pnl_pct': round(pnl_pct, 2),
                'exit_reason': t['action'],
            })
    return trades


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    symbols = ['AAPL', 'TSLA']
    start = '2024-01-01'
    end = '2025-01-01'

    # Parse CLI args: bt_backtest.py [SYM1 SYM2 ...] [start] [end]
    args = sys.argv[1:]
    sym_args = [a for a in args if not re.match(r'\d{4}-\d{2}-\d{2}', a)]
    date_args = [a for a in args if re.match(r'\d{4}-\d{2}-\d{2}', a)]
    if sym_args:
        symbols = [s.upper() for s in sym_args]
    if len(date_args) >= 1:
        start = date_args[0]
    if len(date_args) >= 2:
        end = date_args[1]

    print(f'Running backtrader backtest: {symbols} from {start} to {end}')
    result = run_backtrader_backtest(symbols, start, end, {})

    if result['status'] == 'ok':
        print('\n--- Metrics ---')
        for k, v in result['metrics'].items():
            print(f'  {k}: {v}')
        print(f'\n--- Charts ---')
        for k, v in result['charts'].items():
            print(f'  {k}: {v}')
        print(f'\n--- Trades ({len(result["trades"])}) ---')
        for t in result['trades'][:20]:
            print(f'  {t["symbol"]} {t["side"]} {t["entry_date"]}→{t["exit_date"]} '
                  f'{t["pnl_pct"]:+.2f}% ({t["exit_reason"]})')
        if result.get('errors'):
            print(f'\n--- Errors ---')
            for e in result['errors']:
                print(f'  {e}')
    else:
        print(f'ERROR: {result.get("error")}')
