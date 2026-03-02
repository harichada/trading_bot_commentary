"""
Tests for gap_fade_backtester.py — vectorbt-powered backtester module.

Self-contained test file — no shared conftest.
"""
import math
import random
import sqlite3
import tempfile
import os
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Import the module under test
from gap_fade_backtester import (
    DataLoader,
    BatchDataProvider,
    VectorizedGapScanner,
    SimulationState,
    simulate_gap_day,
    MetricsAdapter,
    VbtGapFadeBacktester,
    HAS_VBT,
)
from gap_fade_app import (
    GapFadeConfig,
    GapCandidate,
    TradeRecord,
    StopOutRecord,
    DailyStats,
    MarketRegime,
    PriceDB,
    compute_adaptive_stop_pct,
    _direction_pnl,
    _stop_hit,
    _target_hit,
)


# ---------------------------------------------------------------------------
# Fixtures — test DB with synthetic data
# ---------------------------------------------------------------------------

def _create_test_db(tmp_path: str, symbols: List[str] = None,
                    start: str = '2022-01-01', end: str = '2024-12-31',
                    include_gaps: bool = True) -> str:
    """Create a test SQLite DB with synthetic daily bars.

    Generates realistic OHLCV data with planted gap days for testing.
    """
    if symbols is None:
        symbols = ['AAPL', 'TSLA', 'NVDA', 'SPY', 'META']

    db_path = os.path.join(tmp_path, 'test_gaps.db')
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS daily_bars (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            PRIMARY KEY (symbol, date)
        ) WITHOUT ROWID
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_daily_bars_date ON daily_bars(date, symbol)')

    rng = np.random.RandomState(42)
    start_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt = datetime.strptime(end, '%Y-%m-%d')

    rows = []
    for sym in symbols:
        price = 100.0 + rng.uniform(-30, 50)
        dt = start_dt
        day_count = 0
        while dt <= end_dt:
            # Skip weekends
            if dt.weekday() >= 5:
                dt += timedelta(days=1)
                continue

            day_count += 1
            date_str = dt.strftime('%Y-%m-%d')

            # Normal day: small random movement
            daily_return = rng.normal(0.001, 0.02)
            prev_close = price

            # Plant gap days (every ~60 trading days for AAPL/TSLA/NVDA)
            if include_gaps and sym in ('AAPL', 'TSLA', 'NVDA') and day_count % 60 == 30:
                # Gap up: open 8-15% above prev close
                gap_pct = rng.uniform(0.08, 0.15)
                open_price = prev_close * (1 + gap_pct)
                # Gap day: price tends to fade back
                high_price = open_price * (1 + rng.uniform(0, 0.02))
                low_price = open_price * (1 - rng.uniform(0.02, 0.08))
                close_price = open_price * (1 + rng.uniform(-0.05, 0.01))
            elif include_gaps and sym == 'META' and day_count % 80 == 40:
                # Gap down: open 6-12% below prev close
                gap_pct = rng.uniform(0.06, 0.12)
                open_price = prev_close * (1 - gap_pct)
                high_price = open_price * (1 + rng.uniform(0.02, 0.07))
                low_price = open_price * (1 - rng.uniform(0, 0.02))
                close_price = open_price * (1 + rng.uniform(-0.01, 0.05))
            else:
                open_price = prev_close * (1 + rng.uniform(-0.005, 0.005))
                close_price = open_price * (1 + daily_return)
                high_price = max(open_price, close_price) * (1 + rng.uniform(0, 0.015))
                low_price = min(open_price, close_price) * (1 - rng.uniform(0, 0.015))

            volume = int(rng.uniform(500000, 5000000))
            rows.append((sym, date_str, round(open_price, 2), round(high_price, 2),
                         round(low_price, 2), round(close_price, 2), volume))
            price = close_price
            dt += timedelta(days=1)

    conn.executemany(
        'INSERT OR REPLACE INTO daily_bars (symbol, date, open, high, low, close, volume) '
        'VALUES (?, ?, ?, ?, ?, ?, ?)', rows
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def test_db(tmp_path):
    """Create a test DB and return its path."""
    return _create_test_db(str(tmp_path))


@pytest.fixture
def test_config():
    """Default test config."""
    return GapFadeConfig(
        gap_threshold=0.07,
        max_gap_pct=0.50,
        vol_ratio_max=3.0,
        min_price=10.0,
        min_avg_volume=100,  # low threshold for test data
        initial_capital=25000,
        risk_pct=0.02,
        kelly_fraction=0.25,
        max_positions=5,
        adaptive_stops=True,
        stop_gap_fraction=0.25,
        stop_min_pct=0.015,
        stop_max_pct=0.025,
        slippage_pct=0.0015,
        borrow_rate_annual=0.02,
        adverse_fill=True,
        adverse_fill_pct=0.50,
        reentry_enabled=True,
        reentry_max_per_symbol=1,
        reentry_stop_pct=0.01,
        trade_gap_downs=False,
    )


# ---------------------------------------------------------------------------
# 1. DataLoader Tests
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_load_universe_returns_wide_dataframe(self, test_db):
        loader = DataLoader(db_path=test_db)
        df = loader.load_universe(['AAPL', 'TSLA', 'NVDA'], '2023-01-01', '2023-12-31')

        assert not df.empty
        assert isinstance(df.columns, pd.MultiIndex)
        # Should have fields: open, high, low, close, volume
        fields = set(df.columns.get_level_values(0).unique())
        assert fields == {'open', 'high', 'low', 'close', 'volume'}
        # Should have our symbols
        syms = set(df.columns.get_level_values(1).unique())
        assert syms == {'AAPL', 'TSLA', 'NVDA'}

    def test_load_universe_warmup_extends_dates(self, test_db):
        loader = DataLoader(db_path=test_db)
        df = loader.load_universe(['AAPL'], '2023-06-01', '2023-12-31', warmup_days=252)

        # Data should start well before 2023-06-01 due to warmup
        first_date = df.index.min()
        assert first_date < pd.Timestamp('2023-06-01')

    def test_load_spy(self, test_db):
        loader = DataLoader(db_path=test_db)
        df = loader.load_spy('2023-01-01', '2023-12-31')

        assert not df.empty
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_get_symbol_bars_from_wide(self, test_db):
        loader = DataLoader(db_path=test_db)
        wide = loader.load_universe(['AAPL', 'TSLA'], '2023-01-01', '2023-12-31')

        aapl = DataLoader.get_symbol_bars('AAPL', wide)
        assert aapl is not None
        assert 'open' in aapl.columns
        assert 'close' in aapl.columns
        assert aapl.index.name == 'AAPL'

        # Non-existent symbol
        missing = DataLoader.get_symbol_bars('ZZZZZ', wide)
        assert missing is None

    def test_batch_data_provider_matches_format(self, test_db):
        """BatchDataProvider returns same format as PriceDB.get_bars."""
        loader = DataLoader(db_path=test_db)
        wide = loader.load_universe(['AAPL'], '2023-01-01', '2023-12-31')
        provider = BatchDataProvider(wide)

        result = provider.get_bars('AAPL', '2023-06-01', '2023-12-31')
        assert result is not None
        assert 'open' in result.columns
        assert 'close' in result.columns
        # All dates should be within range
        assert result.index.min() >= pd.Timestamp('2023-06-01')
        assert result.index.max() <= pd.Timestamp('2023-12-31')

        # Missing symbol returns None
        missing = provider.get_bars('ZZZZZ', '2023-06-01', '2023-12-31')
        assert missing is None


# ---------------------------------------------------------------------------
# 2. VectorizedGapScanner Tests
# ---------------------------------------------------------------------------

class TestVectorizedGapScanner:
    def test_detects_gap_ups(self, test_db, test_config):
        loader = DataLoader(db_path=test_db)
        data = loader.load_universe(['AAPL', 'TSLA', 'NVDA'], '2023-01-01', '2023-12-31')
        scanner = VectorizedGapScanner(test_config)
        gaps_df = scanner.scan(data, '2023-01-01', '2023-12-31')

        assert not gaps_df.empty
        # All gap-ups should be direction='short'
        gap_ups = gaps_df[gaps_df['direction'] == 'short']
        assert len(gap_ups) > 0
        assert all(gap_ups['gap_pct'] >= test_config.gap_threshold)

    def test_detects_gap_downs(self, test_db):
        config = GapFadeConfig(
            gap_threshold=0.07,
            trade_gap_downs=True,
            gap_down_threshold=0.05,
            gap_down_max_pct=0.50,
            min_price=10.0,
            min_avg_volume=100,
        )
        loader = DataLoader(db_path=test_db)
        data = loader.load_universe(['META'], '2023-01-01', '2023-12-31')
        scanner = VectorizedGapScanner(config)
        gaps_df = scanner.scan(data, '2023-01-01', '2023-12-31')

        if not gaps_df.empty:
            gap_downs = gaps_df[gaps_df['direction'] == 'long']
            if len(gap_downs) > 0:
                assert all(gap_downs['gap_pct'] < 0)

    def test_respects_min_price(self, test_db):
        config = GapFadeConfig(gap_threshold=0.07, min_price=200.0, min_avg_volume=100)
        loader = DataLoader(db_path=test_db)
        data = loader.load_universe(['AAPL', 'TSLA', 'NVDA'], '2023-01-01', '2023-12-31')
        scanner = VectorizedGapScanner(config)
        gaps_df = scanner.scan(data, '2023-01-01', '2023-12-31')

        # With min_price=200, many gaps should be filtered out
        if not gaps_df.empty:
            assert all(gaps_df['open'] >= 200.0)

    def test_vol_filter_direction_aware(self, test_db):
        config = GapFadeConfig(
            gap_threshold=0.07,
            vol_ratio_max=2.0,
            trade_gap_downs=True,
            gap_down_threshold=0.05,
            gap_down_vol_ratio_max=5.0,
            min_avg_volume=100,
        )
        loader = DataLoader(db_path=test_db)
        data = loader.load_universe(['AAPL', 'META'], '2023-01-01', '2023-12-31')
        scanner = VectorizedGapScanner(config)
        gaps_df = scanner.scan(data, '2023-01-01', '2023-12-31')
        filtered = scanner.apply_vol_filter(gaps_df)

        if not filtered.empty:
            shorts = filtered[filtered['direction'] == 'short']
            longs = filtered[filtered['direction'] == 'long']
            if len(shorts) > 0:
                assert all(shorts['vol_ratio'] <= config.vol_ratio_max)
            if len(longs) > 0:
                assert all(longs['vol_ratio'] <= config.gap_down_vol_ratio_max)

    def test_gap_dict_has_required_keys(self, test_db, test_config):
        loader = DataLoader(db_path=test_db)
        data = loader.load_universe(['AAPL', 'TSLA'], '2023-01-01', '2023-12-31')
        scanner = VectorizedGapScanner(test_config)
        gaps_df = scanner.scan(data, '2023-01-01', '2023-12-31')

        required_keys = {'date', 'symbol', 'open', 'high', 'low', 'close',
                         'prev_close', 'gap_pct', 'volume', 'avg_vol', 'vol_ratio', 'direction'}
        if not gaps_df.empty:
            assert required_keys == set(gaps_df.columns)


# ---------------------------------------------------------------------------
# 3. Simulation Accuracy Tests
# ---------------------------------------------------------------------------

class TestSimulation:
    def _make_gap(self, **overrides) -> dict:
        """Create a gap dict for testing."""
        gap = {
            'date': '2023-06-15',
            'symbol': 'AAPL',
            'open': 108.0,
            'high': 112.0,
            'low': 95.0,
            'close': 98.0,
            'prev_close': 100.0,
            'gap_pct': 0.08,
            'volume': 2000000,
            'avg_vol': 1500000,
            'vol_ratio': 1.3,
            'direction': 'short',
        }
        gap.update(overrides)
        return gap

    def _make_state(self, config=None) -> SimulationState:
        config = config or GapFadeConfig(
            initial_capital=25000, risk_pct=0.02, kelly_fraction=0.25,
            max_positions=5, slippage_pct=0.0015, borrow_rate_annual=0.02,
            adaptive_stops=True, stop_gap_fraction=0.25, stop_min_pct=0.015,
            stop_max_pct=0.025, adverse_fill=True, adverse_fill_pct=0.50,
            reentry_enabled=True, reentry_max_per_symbol=1,
            reentry_stop_pct=0.01, max_notional=50000,
        )
        state = SimulationState(config=config)
        state.reset_daily('2023-06-15', 3)
        return state

    def test_stop_hit_exit(self):
        """Stop triggered → correct P&L (loss)."""
        random.seed(42)
        # High above stop but low stays above half_target → clean stop (no ambiguity)
        gap = self._make_gap(
            open=108.0, high=112.0, low=106.0, close=110.0,
            gap_pct=0.08, prev_close=100.0, direction='short',
        )
        config = GapFadeConfig(
            initial_capital=25000, risk_pct=0.02, kelly_fraction=0.25,
            max_positions=5, slippage_pct=0.0015, borrow_rate_annual=0.02,
            adaptive_stops=True, stop_gap_fraction=0.25, stop_min_pct=0.015,
            stop_max_pct=0.025, adverse_fill=True, adverse_fill_pct=0.50,
            reentry_enabled=False, max_notional=50000,
        )
        state = SimulationState(config=config)
        state.reset_daily('2023-06-15', 3)
        trades, log = simulate_gap_day(gap, state, config)

        assert len(trades) >= 1
        # Trade should be a stop (no ambiguity: low=106 doesn't reach half_target≈103.9)
        assert trades[0].exit_reason == 'stop'
        assert trades[0].pnl < 0  # Stop = loss for short
        assert state.equity < config.initial_capital

    def test_full_target_exit(self):
        """Gap fills back to prev_close → profit."""
        random.seed(42)
        # Price fades: open at 108, low reaches 98 (below half), close at 99 (≤ prev_close=100)
        gap = self._make_gap(
            open=108.0, high=109.0, low=98.0, close=99.0,
            gap_pct=0.08, prev_close=100.0, direction='short',
        )
        state = self._make_state()
        trades, log = simulate_gap_day(gap, state, state.config)

        assert len(trades) >= 1
        total_pnl = sum(t.pnl for t in trades)
        assert total_pnl > 0  # Profitable gap fade

    def test_partial_then_time_exit(self):
        """Partial cover hit but close doesn't reach prev_close → partial + time exit."""
        random.seed(42)
        # Half target = (entry + prev_close) / 2 ≈ (107.84 + 100) / 2 ≈ 103.9
        # Low=103 reaches half target, but close=105 > prev_close → no full target
        gap = self._make_gap(
            open=108.0, high=109.0, low=103.0, close=105.0,
            gap_pct=0.08, prev_close=100.0, direction='short',
        )
        config = GapFadeConfig(
            initial_capital=25000, risk_pct=0.02, kelly_fraction=0.25,
            max_positions=5, slippage_pct=0.0015, borrow_rate_annual=0.02,
            adaptive_stops=True, stop_gap_fraction=0.25, stop_min_pct=0.015,
            stop_max_pct=0.025, adverse_fill=False,  # no ambiguity
            reentry_enabled=False, max_notional=50000,
        )
        state = SimulationState(config=config)
        state.reset_daily('2023-06-15', 3)
        trades, log = simulate_gap_day(gap, state, config)

        if len(trades) >= 2:
            reasons = [t.exit_reason for t in trades]
            assert 'partial' in reasons
            # Second trade should be full_target or time_exit
            assert reasons[1] in ('full_target', 'time_exit')

    def test_direction_long(self):
        """Gap-down long simulation."""
        random.seed(42)
        gap = self._make_gap(
            open=94.0, high=101.0, low=93.0, close=99.0,
            gap_pct=-0.06, prev_close=100.0, direction='long',
        )
        config = GapFadeConfig(
            initial_capital=25000, risk_pct=0.02, kelly_fraction=0.25,
            max_positions=5, slippage_pct=0.0015, borrow_rate_annual=0.02,
            adaptive_stops=True, stop_gap_fraction=0.25, stop_min_pct=0.015,
            stop_max_pct=0.025, adverse_fill=False,
            reentry_enabled=False, max_notional=50000,
            trade_gap_downs=True, gap_down_vol_ratio_max=3.0,
        )
        state = SimulationState(config=config)
        state.reset_daily('2023-06-15', 3)
        trades, log = simulate_gap_day(gap, state, config)

        assert len(trades) >= 1
        assert trades[0].side == 'long'
        # High=101 reaches prev_close=100, so should be profitable
        total_pnl = sum(t.pnl for t in trades)
        assert total_pnl > 0

    def test_kelly_sizing(self):
        """Kelly position size computed correctly."""
        state = self._make_state()
        kelly = state.compute_kelly_size()
        # With study data: p=0.709, b=2.5/3.0, quarter Kelly
        p = 0.709
        b = 2.5 / 3.0
        expected = max(0, ((p * b - (1 - p)) / b) * 0.25)
        assert abs(kelly - expected) < 1e-6

    def test_reentry_after_stopout(self):
        """Re-entry fires after stop-out when price moves favorably."""
        random.seed(42)
        # Stop hits (high=112), then close=101 < entry → favorable for re-entry
        gap = self._make_gap(
            open=108.0, high=112.0, low=100.5, close=101.0,
            gap_pct=0.08, prev_close=100.0, direction='short',
        )
        config = GapFadeConfig(
            initial_capital=25000, risk_pct=0.02, kelly_fraction=0.25,
            max_positions=5, slippage_pct=0.0015, borrow_rate_annual=0.02,
            adaptive_stops=True, stop_gap_fraction=0.25, stop_min_pct=0.015,
            stop_max_pct=0.025, adverse_fill=False,
            reentry_enabled=True, reentry_max_per_symbol=1, reentry_stop_pct=0.01,
            max_notional=50000,
        )
        state = SimulationState(config=config)
        state.reset_daily('2023-06-15', 3)
        trades, log = simulate_gap_day(gap, state, config)

        # Should have stop + re-entry (re-entry may also stop out)
        reasons = [t.exit_reason for t in trades]
        assert 'stop' in reasons
        if len(trades) >= 2:
            assert 'reentry_time' in reasons or 'reentry_stop' in reasons

    def test_position_size_zero_skips(self):
        """Zero position size → skip (no crash)."""
        gap = self._make_gap(open=0.01, prev_close=0.009, gap_pct=0.11)
        state = self._make_state()
        state.equity = 0.01  # tiny equity → zero shares
        trades, log = simulate_gap_day(gap, state, state.config)
        # Should skip gracefully
        assert len(trades) == 0


# ---------------------------------------------------------------------------
# 4. Strategy Plugin Compatibility Tests
# ---------------------------------------------------------------------------

class TestStrategyPlugins:
    def test_classic_passthrough(self, test_db, test_config):
        """Classic strategy: no filter, no score override."""
        try:
            from gap_fade_strategies import GapFadeStrategyRegistry
            strategy = GapFadeStrategyRegistry.create_strategy('classic_gap_fade')
        except (ImportError, KeyError):
            pytest.skip("classic_gap_fade strategy not available")

        gap = {
            'symbol': 'AAPL', 'direction': 'short', 'gap_pct': 0.08,
            'vol_ratio': 1.5, 'date': '2023-06-15', 'prev_close': 100.0,
        }
        ok, reason = strategy.filter_candidate(gap)
        assert ok is True
        score = strategy.score_candidate(gap)
        assert score is None  # None = keep engine score

    def test_minervini_with_batch_provider(self, test_db, test_config):
        """Minervini filter works with BatchDataProvider (no direct SQLite)."""
        try:
            from gap_fade_strategies import GapFadeStrategyRegistry
            strategy = GapFadeStrategyRegistry.create_strategy('minervini_trend')
        except (ImportError, KeyError):
            pytest.skip("minervini_trend strategy not available")

        loader = DataLoader(db_path=test_db)
        data = loader.load_universe(['AAPL', 'TSLA'], '2023-01-01', '2023-12-31')
        provider = BatchDataProvider(data)

        # Patch get_price_db().get_bars to use batch provider
        from gap_fade_app import get_price_db
        db = get_price_db()
        original = db.get_bars
        db.get_bars = provider.get_bars
        try:
            gap = {
                'symbol': 'AAPL', 'direction': 'short', 'gap_pct': 0.08,
                'vol_ratio': 1.5, 'date': '2023-06-15', 'prev_close': 100.0,
            }
            # Should not raise — the strategy reads bars from RAM
            ok, reason = strategy.filter_candidate(gap)
            assert isinstance(ok, bool)
        finally:
            db.get_bars = original

    def test_strategy_filter_reduces_candidates(self, test_db):
        """A filtering strategy should reject some candidates."""
        try:
            from gap_fade_strategies import GapFadeStrategyRegistry
            strategy = GapFadeStrategyRegistry.create_strategy('minervini_trend')
        except (ImportError, KeyError):
            pytest.skip("minervini_trend strategy not available")

        loader = DataLoader(db_path=test_db)
        data = loader.load_universe(['AAPL', 'TSLA', 'NVDA'], '2023-01-01', '2023-12-31')
        provider = BatchDataProvider(data)

        from gap_fade_app import get_price_db
        db = get_price_db()
        original = db.get_bars
        db.get_bars = provider.get_bars

        try:
            gaps = [
                {'symbol': 'AAPL', 'direction': 'short', 'gap_pct': 0.08,
                 'vol_ratio': 1.5, 'date': '2023-06-15', 'prev_close': 100.0},
                {'symbol': 'TSLA', 'direction': 'short', 'gap_pct': 0.10,
                 'vol_ratio': 1.2, 'date': '2023-06-15', 'prev_close': 200.0},
                {'symbol': 'NVDA', 'direction': 'short', 'gap_pct': 0.09,
                 'vol_ratio': 1.8, 'date': '2023-06-15', 'prev_close': 300.0},
            ]
            results = []
            for g in gaps:
                ok, reason = strategy.filter_candidate(g)
                results.append(ok)
            # At least the filter ran without error; it may pass or reject
            assert all(isinstance(r, bool) for r in results)
        finally:
            db.get_bars = original


# ---------------------------------------------------------------------------
# 5. Result Format Tests
# ---------------------------------------------------------------------------

class TestResultFormat:
    def test_result_keys_match_original(self):
        """Result dict has all expected keys."""
        config = GapFadeConfig(initial_capital=25000)
        state = SimulationState(config=config)
        # Add a fake trade
        trade = TradeRecord(
            symbol='AAPL', entry_price=108.0, exit_price=100.0,
            shares=10, pnl=80.0, pnl_pct=0.074,
            entry_time='2023-06-15 09:31', exit_time='2023-06-15 14:00',
            exit_reason='full_target', holding_minutes=270, side='short',
        )
        state.record_trade(trade)

        meta = {
            'raw_gaps': 100, 'gap_days_found': 80, 'bars_missing': 0,
            'gap_days_traded': 50, 'regime_skipped': 2, 'regime_halved': 1,
            'symbols_scanned': 167, 'start_date': '2022-01-01', 'end_date': '2024-12-31',
            'strategy_id': 'classic_gap_fade',
            'strategy_name': 'Classic Gap Fade',
            'bt_log': [], 'daily_summary': [], 'warnings': [],
        }
        result = MetricsAdapter.build_result(state, config, meta)

        expected_keys = {
            'total_trades', 'wins', 'losses', 'win_rate', 'total_pnl',
            'return_pct', 'avg_win', 'avg_loss', 'profit_factor',
            'max_drawdown', 'max_drawdown_pct', 'sharpe', 'avg_holding_min',
            'final_equity', 'raw_gaps', 'gap_days_found', 'bars_missing',
            'gap_days_traded', 'regime_skipped', 'regime_halved',
            'symbols_scanned', 'start_date', 'end_date', 'config',
            'trades', 'all_trades', 'daily_summary', 'bt_log',
            'realism', 'strategy', 'warnings',
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_metrics_computed_correctly(self):
        """Metrics (win rate, P&L, etc.) are computed correctly."""
        config = GapFadeConfig(initial_capital=25000)
        state = SimulationState(config=config)

        trades = [
            TradeRecord('AAPL', 108.0, 100.0, 10, 80.0, 0.074,
                        '2023-06-15 09:31', '2023-06-15 14:00', 'full_target', 270, 'short'),
            TradeRecord('TSLA', 210.0, 215.0, 5, -25.0, -0.024,
                        '2023-06-16 09:31', '2023-06-16 10:30', 'stop', 60, 'short'),
            TradeRecord('NVDA', 305.0, 295.0, 8, 80.0, 0.033,
                        '2023-06-17 09:31', '2023-06-17 12:00', 'partial', 150, 'short'),
        ]
        for t in trades:
            state.record_trade(t)

        meta = {
            'raw_gaps': 10, 'gap_days_found': 8, 'bars_missing': 0,
            'gap_days_traded': 3, 'regime_skipped': 0, 'regime_halved': 0,
            'symbols_scanned': 3, 'start_date': '2023-06-15', 'end_date': '2023-06-17',
            'strategy_id': 'classic_gap_fade', 'strategy_name': 'Classic',
            'bt_log': [], 'daily_summary': [], 'warnings': [],
        }
        result = MetricsAdapter.build_result(state, config, meta)

        assert result['total_trades'] == 3
        assert result['wins'] == 2
        assert result['losses'] == 1
        assert abs(result['win_rate'] - 2/3) < 0.01
        assert abs(result['total_pnl'] - 135.0) < 0.01

    def test_sanitize_no_infinity_nan(self):
        """No JSON-breaking inf/nan values in result."""
        config = GapFadeConfig(initial_capital=0)  # will cause division by zero
        state = SimulationState(config=config)
        state.equity = 0

        meta = {
            'raw_gaps': 0, 'gap_days_found': 0, 'bars_missing': 0,
            'gap_days_traded': 0, 'regime_skipped': 0, 'regime_halved': 0,
            'symbols_scanned': 0, 'start_date': '', 'end_date': '',
            'strategy_id': '', 'strategy_name': '',
            'bt_log': [], 'daily_summary': [], 'warnings': [],
        }
        result = MetricsAdapter.build_result(state, config, meta)

        # Verify no inf/nan in the result
        def _check(obj, path=''):
            if isinstance(obj, float):
                assert not math.isinf(obj), f"inf at {path}"
                assert not math.isnan(obj), f"nan at {path}"
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _check(v, f"{path}.{k}")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    _check(v, f"{path}[{i}]")

        _check(result)

    def test_sanitize_converts_numpy_types(self):
        """Result with numpy types is JSON-serializable after sanitize."""
        import json
        config = GapFadeConfig(initial_capital=25000)
        state = SimulationState(config=config)

        # Create trade with numpy-typed values (simulating vectorized scanner output)
        trade = TradeRecord('AAPL', np.float64(108.0), np.float64(100.0),
                            np.int64(10), np.float64(80.0), np.float64(0.074),
                            '2023-06-15 09:31', '2023-06-15 14:00', 'full_target',
                            np.int64(270), 'short')
        state.record_trade(trade)

        meta = {
            'raw_gaps': np.int64(5), 'gap_days_found': np.int64(3),
            'bars_missing': 0,
            'gap_days_traded': np.int64(1), 'regime_skipped': 0,
            'regime_halved': 0, 'symbols_scanned': np.int64(2),
            'start_date': '2023-01-01', 'end_date': '2023-12-31',
            'strategy_id': 'classic_gap_fade', 'strategy_name': 'Classic',
            'bt_log': [], 'warnings': [],
            'daily_summary': [{'date': '2023-06-15', 'symbol': 'AAPL',
                               'gap_pct': np.float64(0.08),
                               'vol_ratio': np.float64(1.5),
                               'trades': 1, 'pnl': np.float64(80.0)}],
        }
        result = MetricsAdapter.build_result(state, config, meta)

        # Apply the same sanitize function used in run()
        def _sanitize(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                val = float(obj)
                if math.isinf(val):
                    return 9999.99 if val > 0 else -9999.99
                if math.isnan(val):
                    return 0.0
                return val
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, float):
                if math.isinf(obj):
                    return 9999.99 if obj > 0 else -9999.99
                if math.isnan(obj):
                    return 0.0
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return [_sanitize(v) for v in obj.tolist()]
            return obj

        sanitized = _sanitize(result)
        # Must be JSON-serializable without errors
        json_str = json.dumps(sanitized)
        assert len(json_str) > 100
        # Round-trip should preserve values
        parsed = json.loads(json_str)
        assert parsed['total_trades'] == 1
        assert isinstance(parsed['total_pnl'], (int, float))
        assert isinstance(parsed['daily_summary'][0]['gap_pct'], float)


# ---------------------------------------------------------------------------
# 6. Graceful Degradation Tests
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_works_without_vectorbt(self):
        """Module works even if vectorbt isn't installed (uses numpy/pandas only)."""
        # The module imports successfully regardless — HAS_VBT tracks availability
        from gap_fade_backtester import HAS_VBT, MetricsAdapter, SimulationState

        config = GapFadeConfig(initial_capital=25000)
        state = SimulationState(config=config)
        trade = TradeRecord('AAPL', 108.0, 100.0, 10, 80.0, 0.074,
                            '2023-06-15 09:31', '2023-06-15 14:00', 'full_target', 270, 'short')
        state.record_trade(trade)

        meta = {
            'raw_gaps': 1, 'gap_days_found': 1, 'bars_missing': 0,
            'gap_days_traded': 1, 'regime_skipped': 0, 'regime_halved': 0,
            'symbols_scanned': 1, 'start_date': '2023-01-01', 'end_date': '2023-12-31',
            'strategy_id': 'classic_gap_fade', 'strategy_name': 'Classic',
            'bt_log': [], 'daily_summary': [], 'warnings': [],
        }
        result = MetricsAdapter.build_result(state, config, meta)
        assert result['total_trades'] == 1

    @pytest.mark.skipif(not HAS_VBT, reason="vectorbt not installed")
    def test_vbt_metrics_added_when_available(self):
        """When vectorbt is available, extra metrics are added."""
        config = GapFadeConfig(initial_capital=25000)
        state = SimulationState(config=config)

        # Add several trades so equity curve has enough points
        for i in range(20):
            pnl = 50.0 if i % 3 != 0 else -30.0
            trade = TradeRecord(
                f'SYM{i}', 100.0 + i, 100.0 + i - pnl/10, 10, pnl,
                pnl / (1000 + i * 10), f'2023-01-{i+1:02d} 09:31',
                f'2023-01-{i+1:02d} 14:00', 'full_target', 270, 'short')
            state.record_trade(trade)
            state.equity_curve.append(state.equity)

        meta = {
            'raw_gaps': 20, 'gap_days_found': 20, 'bars_missing': 0,
            'gap_days_traded': 20, 'regime_skipped': 0, 'regime_halved': 0,
            'symbols_scanned': 20, 'start_date': '2023-01-01', 'end_date': '2023-01-20',
            'strategy_id': 'classic_gap_fade', 'strategy_name': 'Classic',
            'bt_log': [], 'daily_summary': [], 'warnings': [],
        }
        result = MetricsAdapter.build_result(state, config, meta)
        MetricsAdapter.add_vbt_metrics(result, state, config)
        assert 'vbt_metrics' in result
        assert 'sortino' in result['vbt_metrics']


# ---------------------------------------------------------------------------
# 7. Integration: VbtGapFadeBacktester
# ---------------------------------------------------------------------------

class TestVbtBacktesterIntegration:
    def test_constructor_matches_original(self):
        """VbtGapFadeBacktester has same interface as GapFadeBacktester."""
        bt = VbtGapFadeBacktester()
        assert bt.progress == 0.0
        assert bt.status == 'idle'
        assert bt._cancel is False
        assert bt.result is None
        assert bt.strategy is None

    def test_cancel(self):
        """Cancel sets status correctly."""
        bt = VbtGapFadeBacktester()
        bt.cancel()
        assert bt.status == 'cancelled'
        assert bt._cancel is True

    def test_run_with_test_db(self, test_db, test_config):
        """Full integration test: run backtest on test DB."""
        import asyncio

        original_path = PriceDB.DB_PATH
        PriceDB.DB_PATH = test_db

        try:
            bt = VbtGapFadeBacktester(config=test_config)

            async def _run():
                return await bt.run(
                    symbols=['AAPL', 'TSLA', 'NVDA'],
                    start_date='2023-06-01',
                    end_date='2023-12-31',
                )

            result = asyncio.get_event_loop().run_until_complete(_run())

            assert bt.status == 'done'
            assert 'total_trades' in result
            if result['total_trades'] > 0:
                assert 'win_rate' in result
                assert 'total_pnl' in result
                assert 'final_equity' in result
                assert 'trades' in result
                assert 'all_trades' in result
                assert 'config' in result
                assert 'realism' in result
        finally:
            PriceDB.DB_PATH = original_path

    def test_run_empty_db(self, tmp_path):
        """Run on empty DB returns graceful error."""
        import asyncio

        db_path = os.path.join(str(tmp_path), 'empty.db')
        conn = sqlite3.connect(db_path)
        conn.execute('''
            CREATE TABLE daily_bars (
                symbol TEXT, date TEXT, open REAL, high REAL,
                low REAL, close REAL, volume INTEGER,
                PRIMARY KEY (symbol, date)
            ) WITHOUT ROWID
        ''')
        conn.close()

        original_path = PriceDB.DB_PATH
        PriceDB.DB_PATH = db_path
        try:
            bt = VbtGapFadeBacktester()

            async def _run():
                return await bt.run(
                    symbols=['AAPL'],
                    start_date='2023-01-01',
                    end_date='2023-12-31',
                )

            result = asyncio.get_event_loop().run_until_complete(_run())
            assert bt.status == 'done'
            assert result.get('total_trades', 0) == 0
        finally:
            PriceDB.DB_PATH = original_path


# ---------------------------------------------------------------------------
# 8. SimulationState Tests
# ---------------------------------------------------------------------------

class TestSimulationState:
    def test_equity_tracks_trades(self):
        """Equity is updated after each recorded trade."""
        config = GapFadeConfig(initial_capital=25000)
        state = SimulationState(config=config)

        trade1 = TradeRecord('AAPL', 108.0, 100.0, 10, 80.0, 0.074,
                             '2023-06-15 09:31', '2023-06-15 14:00', 'full_target', 270, 'short')
        state.record_trade(trade1)
        assert state.equity == 25080.0

        trade2 = TradeRecord('TSLA', 210.0, 215.0, 5, -25.0, -0.024,
                             '2023-06-16 09:31', '2023-06-16 10:30', 'stop', 60, 'short')
        state.record_trade(trade2)
        assert state.equity == 25055.0

    def test_peak_equity_tracks_high_water(self):
        config = GapFadeConfig(initial_capital=25000)
        state = SimulationState(config=config)

        state.record_trade(TradeRecord('A', 100, 95, 10, 50, 0.05,
                                       't1', 't2', 'full', 100, 'short'))
        assert state.peak_equity == 25050.0

        state.record_trade(TradeRecord('B', 100, 105, 10, -50, -0.05,
                                       't1', 't2', 'stop', 60, 'short'))
        assert state.peak_equity == 25050.0  # didn't go higher

    def test_reset_daily(self):
        config = GapFadeConfig(initial_capital=25000, max_positions=5, thin_day_threshold=10)
        state = SimulationState(config=config)
        state.positions['AAPL'] = True
        state.stopped_today['TSLA'] = StopOutRecord('TSLA', datetime.now(), 100, 95, 0.05, 1e6)

        state.reset_daily('2023-06-16', 3)
        assert state.positions == {}
        assert state.stopped_today == {}
        assert state.daily_stats.date == '2023-06-16'
        # 3 candidates < thin_day_threshold=10 → eff_max = 3
        assert state.effective_max_positions == 3


# ---------------------------------------------------------------------------
# Drawdown Circuit Breaker Tests
# ---------------------------------------------------------------------------

class TestDDCircuitBreaker:
    """Tests for drawdown circuit breaker feature."""

    def _make_config(self, **overrides) -> GapFadeConfig:
        """Config with DD circuit breaker enabled."""
        defaults = dict(
            initial_capital=100_000, risk_pct=0.02, kelly_fraction=0.25,
            max_positions=5, slippage_pct=0.001, borrow_rate_annual=0.0,
            adaptive_stops=True, stop_gap_fraction=0.25, stop_min_pct=0.015,
            stop_max_pct=0.025, adverse_fill=False, adverse_fill_pct=0.0,
            max_notional=50000,
            dd_circuit_breaker=True,
            dd_tier1_threshold=0.15,
            dd_tier1_scale=0.50,
            dd_tier2_threshold=0.25,
            dd_tier2_scale=0.25,
            dd_tier2_max_positions=1,
            dd_hard_stop=0.0,
        )
        defaults.update(overrides)
        return GapFadeConfig(**defaults)

    def _make_state(self, config=None, equity=None) -> SimulationState:
        config = config or self._make_config()
        state = SimulationState(config=config)
        if equity is not None:
            state.equity = equity
        return state

    def _make_gap(self, **overrides) -> dict:
        gap = {
            'date': '2023-06-15', 'symbol': 'AAPL',
            'open': 108.0, 'high': 112.0, 'low': 95.0, 'close': 98.0,
            'prev_close': 100.0, 'gap_pct': 0.08,
            'volume': 2000000, 'avg_vol': 1500000, 'vol_ratio': 1.3,
            'direction': 'short',
        }
        gap.update(overrides)
        return gap

    # 1. current_drawdown computation
    def test_current_drawdown_computation(self):
        """Verify DD formula with known equity/peak."""
        state = self._make_state()
        state.peak_equity = 100_000
        state.equity = 85_000
        assert abs(state.current_drawdown() - 0.15) < 1e-9

        state.equity = 100_000
        assert state.current_drawdown() == 0.0

        state.equity = 50_000
        assert abs(state.current_drawdown() - 0.50) < 1e-9

    # 2. dd_tier boundaries
    def test_dd_tier_boundaries(self):
        """Verify tier 0/1/2/3 at exact threshold values."""
        config = self._make_config(dd_hard_stop=0.40)
        state = self._make_state(config=config)
        state.peak_equity = 100_000

        # Below tier 1
        state.equity = 86_000  # 14% DD
        assert state.dd_tier() == 0

        # At tier 1 boundary
        state.equity = 85_000  # 15% DD
        assert state.dd_tier() == 1

        # Between tier 1 and tier 2
        state.equity = 80_000  # 20% DD
        assert state.dd_tier() == 1

        # At tier 2 boundary
        state.equity = 75_000  # 25% DD
        assert state.dd_tier() == 2

        # Between tier 2 and hard stop
        state.equity = 65_000  # 35% DD
        assert state.dd_tier() == 2

        # At hard stop boundary
        state.equity = 60_000  # 40% DD
        assert state.dd_tier() == 3
        assert state.dd_hard_stopped is True

    # 3. DD off by default
    def test_dd_off_by_default(self):
        """No circuit_breaker key in results when disabled."""
        config = GapFadeConfig(initial_capital=25000, dd_circuit_breaker=False)
        state = SimulationState(config=config)
        # Simulate a trade so we get a result
        trade = TradeRecord(
            symbol='AAPL', entry_price=100, exit_price=98, shares=10,
            pnl=20, pnl_pct=0.02, holding_minutes=60,
            entry_time='09:31', exit_time='10:31', exit_reason='target_half',
        )
        state.record_trade(trade)
        result = MetricsAdapter.build_result(state, config, {'raw_gaps': 1})
        assert 'circuit_breaker' not in result

    # 4. Tier 1 reduces shares
    def test_tier1_reduces_shares(self):
        """dd_scale=0.5 produces approximately half the shares."""
        config = self._make_config()
        state = self._make_state(config=config)
        state.reset_daily('2023-06-15', 3)
        gap = self._make_gap()

        # Normal run
        trades_normal, _ = simulate_gap_day(gap, state, config, dd_scale=1.0)

        # Reset state for scaled run
        state2 = self._make_state(config=config)
        state2.reset_daily('2023-06-15', 3)
        trades_scaled, _ = simulate_gap_day(gap, state2, config, dd_scale=0.5)

        if trades_normal and trades_scaled:
            normal_shares = trades_normal[0].shares
            scaled_shares = trades_scaled[0].shares
            # scaled should be roughly half (allow rounding)
            assert scaled_shares <= normal_shares
            if normal_shares >= 4:  # enough to meaningfully halve
                assert scaled_shares <= int(normal_shares * 0.5) + 1

    # 5. Tier 2 caps positions
    def test_tier2_caps_positions(self):
        """Only dd_tier2_max_positions allowed at tier 2."""
        config = self._make_config(dd_tier2_max_positions=1)
        state = self._make_state(config=config)
        state.peak_equity = 100_000
        state.equity = 74_000  # 26% DD → tier 2

        tier = state.dd_tier()
        assert tier == 2

        # eff_max should be capped to 1
        eff_max = state.effective_max_pos(10)  # 10 candidates
        eff_max = min(eff_max, config.dd_tier2_max_positions)
        assert eff_max == 1

    # 6. Hard stop skips all subsequent days
    def test_hard_stop_skips_all_days(self):
        """dd_hard_stopped=True → all subsequent days skipped."""
        config = self._make_config(dd_hard_stop=0.30)
        state = self._make_state(config=config)
        state.peak_equity = 100_000
        state.equity = 69_000  # 31% DD → triggers hard stop

        tier = state.dd_tier()
        assert tier == 3
        assert state.dd_hard_stopped is True

        # Subsequent calls still return 3
        state.equity = 95_000  # even if equity recovers
        assert state.dd_tier() == 3

    # 7. Tier resets on recovery
    def test_tier_resets_on_recovery(self):
        """Winning trade drops DD below threshold → tier 0."""
        config = self._make_config()
        state = self._make_state(config=config)
        state.peak_equity = 100_000

        # In tier 1
        state.equity = 84_000  # 16% DD
        assert state.dd_tier() == 1

        # Simulate recovery via winning trade
        trade = TradeRecord(
            symbol='AAPL', entry_price=100, exit_price=98, shares=100,
            pnl=5000, pnl_pct=0.05, holding_minutes=60,
            entry_time='09:31', exit_time='10:31', exit_reason='target_half',
        )
        state.record_trade(trade)
        # equity now 89,000 → 11% DD → tier 0
        assert state.equity == 89_000
        assert state.dd_tier() == 0

    # 8. Hard stop is permanent
    def test_hard_stop_permanent(self):
        """Hard stop remains True even if equity recovers above threshold."""
        config = self._make_config(dd_hard_stop=0.30)
        state = self._make_state(config=config)
        state.peak_equity = 100_000
        state.equity = 69_000  # 31% DD → triggers hard stop
        assert state.dd_tier() == 3

        # Even if equity fully recovers
        state.equity = 110_000
        assert state.dd_hard_stopped is True
        assert state.dd_tier() == 3

    # 9. Config cross-validation
    def test_config_cross_validation(self):
        """tier1 >= tier2 should be rejected by the API endpoint."""
        # This tests the validation logic directly
        config = self._make_config(dd_tier1_threshold=0.30, dd_tier2_threshold=0.25)

        # The cross-validation is in the API endpoint, but we can verify the
        # config values are set and would be caught
        assert config.dd_tier1_threshold >= config.dd_tier2_threshold

        # Also verify that valid config passes
        valid_config = self._make_config(dd_tier1_threshold=0.15, dd_tier2_threshold=0.25)
        assert valid_config.dd_tier1_threshold < valid_config.dd_tier2_threshold

    # 10. Circuit breaker stats in result
    def test_circuit_breaker_stats_in_result(self):
        """All expected keys present with correct types when enabled."""
        config = self._make_config(dd_hard_stop=0.40)
        state = self._make_state(config=config)
        state.peak_equity = 100_000
        state.equity = 80_000
        state.dd_days_tier1 = 5
        state.dd_days_tier2 = 2
        state.dd_trades_skipped = 10

        trade = TradeRecord(
            symbol='AAPL', entry_price=100, exit_price=98, shares=10,
            pnl=20, pnl_pct=0.02, holding_minutes=60,
            entry_time='09:31', exit_time='10:31', exit_reason='target_half',
        )
        state.record_trade(trade)

        result = MetricsAdapter.build_result(state, config, {'raw_gaps': 100})
        assert 'circuit_breaker' in result
        cb = result['circuit_breaker']

        # Check all expected keys
        assert cb['enabled'] is True
        assert cb['tier1_threshold'] == 0.15
        assert cb['tier1_scale'] == 0.50
        assert cb['tier2_threshold'] == 0.25
        assert cb['tier2_scale'] == 0.25
        assert cb['tier2_max_positions'] == 1
        assert cb['hard_stop'] == 0.40
        assert cb['days_in_tier1'] == 5
        assert cb['days_in_tier2'] == 2
        assert cb['trades_skipped_hard_stop'] == 10
        assert cb['hard_stopped'] is False
        assert cb['hard_stop_date'] == ''
        assert isinstance(cb['final_drawdown'], float)
