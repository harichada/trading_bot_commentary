"""Tests for bt_backtest.py — backtrader-based gap fade visual backtester."""
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(__file__), 'gap_fade_prices.db')
HAS_LIVE_DB = os.path.exists(DB_PATH)

CHARTS_DIR = tempfile.mkdtemp(prefix='bt_charts_test_')


def _make_test_db(path: str, symbols=('AAPL', 'TSLA'), days=200):
    """Create a minimal SQLite DB with synthetic daily bars."""
    conn = sqlite3.connect(path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS daily_bars (
            symbol TEXT NOT NULL,
            date   TEXT NOT NULL,
            open   REAL,
            high   REAL,
            low    REAL,
            close  REAL,
            volume REAL,
            PRIMARY KEY (symbol, date)
        ) WITHOUT ROWID
    ''')
    base_date = datetime(2024, 1, 2)
    for sym in symbols:
        base_price = 150 if sym == 'AAPL' else 250
        price = base_price
        rows = []
        for i in range(days):
            dt = base_date + timedelta(days=i)
            if dt.weekday() >= 5:
                continue
            # Simulate occasional gaps (every ~20 bars)
            if i > 0 and i % 20 == 0:
                gap = price * 0.08  # 8% gap up
                open_px = price + gap
            elif i > 0 and i % 30 == 0:
                gap = price * -0.06  # 6% gap down
                open_px = price + gap
            else:
                open_px = price + np.random.uniform(-1, 1)
            high = max(open_px, price) + abs(np.random.normal(0, 1.5))
            low = min(open_px, price) - abs(np.random.normal(0, 1.5))
            close = low + (high - low) * np.random.uniform(0.3, 0.7)
            vol = np.random.randint(500_000, 5_000_000)
            rows.append((sym, dt.strftime('%Y-%m-%d'), round(open_px, 2),
                         round(high, 2), round(low, 2), round(close, 2), vol))
            price = close
        conn.executemany(
            'INSERT OR REPLACE INTO daily_bars (symbol, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)',
            rows,
        )
    conn.commit()
    conn.close()
    return path


@pytest.fixture(scope='module')
def test_db():
    """Provide a test database path — use live DB if available, else synthetic."""
    if HAS_LIVE_DB:
        return DB_PATH
    path = os.path.join(tempfile.mkdtemp(), 'test_prices.db')
    return _make_test_db(path)


@pytest.fixture(scope='module')
def synthetic_db():
    """Always create a synthetic DB for controlled tests."""
    path = os.path.join(tempfile.mkdtemp(), 'synthetic_prices.db')
    return _make_test_db(path, symbols=('TEST1', 'TEST2'), days=250)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_has_backtrader():
    """Verify backtrader is importable."""
    from bt_backtest import HAS_BACKTRADER
    assert HAS_BACKTRADER is True, 'backtrader must be installed for these tests'


def test_load_symbol_data(test_db):
    """load_symbol_data returns a PandasData feed for a valid symbol."""
    from bt_backtest import load_symbol_data
    syms = ('AAPL',) if HAS_LIVE_DB else ('AAPL',)
    feed = load_symbol_data(syms[0], '2024-01-01', '2025-01-01', db_path=test_db)
    assert feed is not None


def test_load_symbol_data_no_data(test_db):
    """load_symbol_data returns None for a symbol with no data."""
    from bt_backtest import load_symbol_data
    feed = load_symbol_data('ZZZZZZ', '2024-01-01', '2025-01-01', db_path=test_db)
    assert feed is None


def test_strategy_params_mapping():
    """GapFadeBTStrategy accepts all documented params."""
    from bt_backtest import GapFadeBTStrategy
    import backtrader as bt
    valid_keys = set(GapFadeBTStrategy.params._getkeys())
    expected = {
        'gap_threshold', 'max_gap_pct', 'vol_ratio_max', 'vol_window',
        'stop_pct', 'adaptive_stops', 'stop_gap_fraction', 'stop_min_pct', 'stop_max_pct',
        'partial_cover_frac', 'trade_gap_downs', 'gap_down_threshold',
        'risk_pct', 'max_positions', 'slippage_pct',
    }
    assert expected.issubset(valid_keys), f'Missing params: {expected - valid_keys}'


def test_run_backtrader_empty_symbols():
    """Runner returns error for empty symbols list."""
    from bt_backtest import run_backtrader_backtest
    result = run_backtrader_backtest([], '2024-01-01', '2025-01-01', {})
    assert result['status'] == 'error'
    assert 'No symbols' in result['error']


def test_run_backtrader_too_many_symbols():
    """Runner returns error for >20 symbols."""
    from bt_backtest import run_backtrader_backtest
    syms = [f'SYM{i}' for i in range(25)]
    result = run_backtrader_backtest(syms, '2024-01-01', '2025-01-01', {})
    assert result['status'] == 'error'
    assert '20' in result['error']


def test_run_backtrader_missing_db():
    """Runner returns error when database doesn't exist."""
    from bt_backtest import run_backtrader_backtest
    result = run_backtrader_backtest(['AAPL'], '2024-01-01', '2025-01-01', {},
                                     db_path='/nonexistent/path.db')
    assert result['status'] == 'error'
    assert 'not found' in result['error']


def test_run_full_backtest(synthetic_db):
    """Full backtrader run with synthetic data produces metrics and charts."""
    from bt_backtest import run_backtrader_backtest
    charts_dir = os.path.join(CHARTS_DIR, 'full_test')
    result = run_backtrader_backtest(
        ['TEST1', 'TEST2'],
        '2024-01-01', '2024-12-31',
        {'gap_threshold': 0.06},
        db_path=synthetic_db,
        charts_dir=charts_dir,
    )
    assert result['status'] == 'ok', f'Backtest failed: {result}'
    assert 'metrics' in result
    assert 'charts' in result
    assert 'trades' in result
    assert len(result['symbols_loaded']) == 2

    # Metrics should have expected keys
    m = result['metrics']
    for key in ('sharpe_ratio', 'max_drawdown_pct', 'total_trades', 'win_rate',
                'total_return_pct', 'final_equity'):
        assert key in m, f'Missing metric: {key}'


def test_charts_created(synthetic_db):
    """Chart PNG files are actually created on disk."""
    from bt_backtest import run_backtrader_backtest
    charts_dir = os.path.join(CHARTS_DIR, 'chart_test')
    result = run_backtrader_backtest(
        ['TEST1', 'TEST2'],
        '2024-01-01', '2024-12-31',
        {'gap_threshold': 0.06},
        db_path=synthetic_db,
        charts_dir=charts_dir,
    )
    assert result['status'] == 'ok'
    charts = result.get('charts', {})

    # Equity curve should always be generated
    assert 'equity_curve' in charts
    eq_path = os.path.join(charts_dir, charts['equity_curve'])
    assert os.path.isfile(eq_path), f'Equity chart not found at {eq_path}'
    assert os.path.getsize(eq_path) > 1000, 'Equity chart too small (likely corrupt)'

    # Drawdown should be generated
    assert 'drawdown' in charts
    dd_path = os.path.join(charts_dir, charts['drawdown'])
    assert os.path.isfile(dd_path)


def test_run_with_config_overrides(synthetic_db):
    """Config overrides are passed through to the strategy."""
    from bt_backtest import run_backtrader_backtest
    charts_dir = os.path.join(CHARTS_DIR, 'config_test')
    result = run_backtrader_backtest(
        ['TEST1'],
        '2024-01-01', '2024-12-31',
        {
            'gap_threshold': 0.05,
            'max_positions': 2,
            'adaptive_stops': False,
            'stop_pct': 0.03,
        },
        db_path=synthetic_db,
        charts_dir=charts_dir,
    )
    assert result['status'] == 'ok'


def test_no_data_in_range(synthetic_db):
    """Runner handles date range with no data gracefully."""
    from bt_backtest import run_backtrader_backtest
    result = run_backtrader_backtest(
        ['TEST1'],
        '2020-01-01', '2020-06-01',
        {},
        db_path=synthetic_db,
    )
    assert result['status'] == 'error'
    assert 'no data' in result['error'].lower() or 'No symbols' in result['error']


@pytest.mark.skipif(not HAS_LIVE_DB, reason='No live gap_fade_prices.db')
def test_run_live_db_aapl():
    """Integration test with real AAPL data (only runs if live DB exists)."""
    from bt_backtest import run_backtrader_backtest
    charts_dir = os.path.join(CHARTS_DIR, 'live_test')
    result = run_backtrader_backtest(
        ['AAPL'],
        '2024-01-01', '2024-12-31',
        {},
        db_path=DB_PATH,
        charts_dir=charts_dir,
    )
    assert result['status'] == 'ok'
    assert result['metrics']['total_trades'] >= 0
    assert 'equity_curve' in result.get('charts', {})


def test_graceful_no_backtrader():
    """When backtrader is unavailable, runner returns a clear error."""
    # We can't actually unimport backtrader, so test the error path directly
    from bt_backtest import run_backtrader_backtest
    with patch('bt_backtest.HAS_BACKTRADER', False):
        result = run_backtrader_backtest(['AAPL'], '2024-01-01', '2025-01-01', {})
        assert result['status'] == 'error'
        assert 'not installed' in result['error']
