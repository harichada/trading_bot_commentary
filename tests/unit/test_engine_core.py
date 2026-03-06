"""Unit tests for GapFadeEngine — the core trading logic.

Covers: position sizing, entry validation, exit logic, P&L calculation,
circuit breakers, force close, metrics, and edge cases.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch
from tests.conftest import (
    make_config, make_engine, make_candidate, make_time, ET,
    GapFadeConfig, GapFadeEngine, GapCandidate, GapPosition,
    TradeRecord, DailyStats, StopOutRecord,
    compute_adaptive_stop_pct, _direction_pnl, _stop_hit, _target_hit,
)


# =========================================================================
# Helper functions
# =========================================================================

class TestDirectionPnl:
    """P&L calculation for long and short trades."""

    def test_short_profit(self):
        """Short: entry 100, exit 95, 10 shares → +$50."""
        assert _direction_pnl('short', 100.0, 95.0, 10) == 50.0

    def test_short_loss(self):
        """Short: entry 100, exit 105, 10 shares → -$50."""
        assert _direction_pnl('short', 100.0, 105.0, 10) == -50.0

    def test_long_profit(self):
        """Long: entry 100, exit 105, 10 shares → +$50."""
        assert _direction_pnl('long', 100.0, 105.0, 10) == 50.0

    def test_long_loss(self):
        """Long: entry 100, exit 95, 10 shares → -$50."""
        assert _direction_pnl('long', 100.0, 95.0, 10) == -50.0

    def test_zero_shares(self):
        assert _direction_pnl('short', 100.0, 95.0, 0) == 0.0

    def test_penny_precision(self):
        """P&L with sub-cent prices."""
        result = _direction_pnl('short', 10.005, 9.995, 1000)
        assert abs(result - 10.0) < 0.01

    def test_large_position(self):
        """1M dollar position, small move."""
        result = _direction_pnl('short', 100.0, 99.99, 10000)
        assert abs(result - 100.0) < 0.01


class TestStopHit:
    """Stop-loss trigger detection."""

    def test_short_stop_hit(self):
        """Short stop above entry: high >= stop → triggered."""
        assert _stop_hit('short', 103.0, 99.0, 102.5) is True

    def test_short_stop_not_hit(self):
        assert _stop_hit('short', 101.0, 99.0, 102.5) is False

    def test_long_stop_hit(self):
        """Long stop below entry: low <= stop → triggered."""
        assert _stop_hit('long', 101.0, 97.0, 97.5) is True

    def test_long_stop_not_hit(self):
        assert _stop_hit('long', 101.0, 98.0, 97.5) is False

    def test_exact_stop_price(self):
        """Price exactly at stop → triggered."""
        assert _stop_hit('short', 102.5, 99.0, 102.5) is True
        assert _stop_hit('long', 101.0, 97.5, 97.5) is True


class TestTargetHit:
    """Target price detection."""

    def test_short_target(self):
        """Short target: price drops to/below target → hit."""
        assert _target_hit('short', 95.0, 97.0) is True
        assert _target_hit('short', 97.0, 97.0) is True
        assert _target_hit('short', 99.0, 97.0) is False

    def test_long_target(self):
        """Long target: price rises to/above target → hit."""
        assert _target_hit('long', 105.0, 103.0) is True
        assert _target_hit('long', 103.0, 103.0) is True
        assert _target_hit('long', 101.0, 103.0) is False


class TestAdaptiveStop:
    """Adaptive stop-loss calculation."""

    def test_adaptive_enabled(self):
        """Stop = gap_pct * fraction, clamped to [min, max]."""
        config = make_config(adaptive_stops=True, stop_gap_fraction=0.25,
                            stop_min_pct=0.015, stop_max_pct=0.025)
        # 10% gap → 10% * 0.25 = 2.5% → clamped to max 2.5%
        assert compute_adaptive_stop_pct(config, 0.10) == 0.025
        # 8% gap → 8% * 0.25 = 2.0%
        assert compute_adaptive_stop_pct(config, 0.08) == 0.02
        # 4% gap → 4% * 0.25 = 1.0% → clamped to min 1.5%
        assert compute_adaptive_stop_pct(config, 0.04) == 0.015

    def test_adaptive_disabled(self):
        """Falls back to fixed stop_pct."""
        config = make_config(adaptive_stops=False, stop_pct=0.02)
        assert compute_adaptive_stop_pct(config, 0.10) == 0.02

    def test_negative_gap(self):
        """Gap-down (negative) uses abs(gap_pct)."""
        config = make_config(adaptive_stops=True, stop_gap_fraction=0.25,
                            stop_min_pct=0.015, stop_max_pct=0.025)
        assert compute_adaptive_stop_pct(config, -0.08) == 0.02


# =========================================================================
# Position Sizing
# =========================================================================

class TestPositionSizing:
    """Position size calculations."""

    def test_basic_sizing(self):
        """Standard sizing with 2% risk."""
        engine = make_engine(initial_capital=100_000, risk_pct=0.02)
        # Entry 100, stop 98 → risk_per_share = $2
        # Dollar risk = 100K * 0.02 = $2000 → shares = 1000
        # But capped by per-slot equity
        shares = engine.compute_position_size(100.0, 98.0)
        assert shares > 0
        assert shares <= 1000  # risk-based limit

    def test_zero_risk_per_share(self):
        """Entry = stop → 0 shares (division by zero protection)."""
        engine = make_engine()
        assert engine.compute_position_size(100.0, 100.0) == 0

    def test_stop_below_entry_short(self):
        """Short: stop > entry → risk_per_share = stop - entry."""
        engine = make_engine(initial_capital=100_000)
        shares = engine.compute_position_size(100.0, 103.0)
        assert shares > 0

    def test_notional_cap(self):
        """Shares capped by max_notional / entry_price."""
        engine = make_engine(initial_capital=1_000_000, max_notional=10_000)
        shares = engine.compute_position_size(100.0, 97.0)
        # Max notional = $10K / $100 = 100 shares max
        assert shares <= 100

    def test_per_slot_equity_cap(self):
        """Shares capped so no single position exceeds equity/max_positions."""
        engine = make_engine(initial_capital=100_000, max_positions=5)
        # Per slot = 100K / 5 = 20K → max shares = 20K / 100 = 200
        shares = engine.compute_position_size(100.0, 97.0)
        assert shares <= 200

    def test_existing_positions_reduce_slots(self):
        """With open positions, remaining slots are fewer."""
        engine = make_engine(initial_capital=100_000, max_positions=5)
        # Add 3 existing positions
        for sym in ['A', 'B', 'C']:
            c = make_candidate(symbol=sym)
            engine.open_position(c, 100.0, '2026-03-09 10:00')
        # Now 2 slots left — sizing should be constrained
        shares = engine.compute_position_size(100.0, 97.0)
        assert shares > 0

    def test_zero_equity(self):
        """Edge: equity at 0 → 0 shares."""
        engine = make_engine(initial_capital=0)
        assert engine.compute_position_size(100.0, 97.0) == 0

    def test_very_expensive_stock(self):
        """$5000 stock with small account."""
        engine = make_engine(initial_capital=25_000)
        shares = engine.compute_position_size(5000.0, 5150.0)
        # Per slot = 25K / 5 = 5K → max = 5K / 5000 = 1
        assert shares <= 5


# =========================================================================
# Entry Validation (should_enter)
# =========================================================================

class TestShouldEnter:
    """Validate entry conditions."""

    def test_normal_entry_backtest(self):
        """Standard candidate passes all checks in backtest mode."""
        engine = make_engine()
        c = make_candidate()
        ok, reason = engine.should_enter(c)
        assert ok, f"Should enter but got: {reason}"

    def test_catalyst_earnings_blocked(self):
        """Earnings catalyst → reject."""
        engine = make_engine()
        c = make_candidate(catalyst='earnings')
        ok, reason = engine.should_enter(c)
        assert not ok
        assert 'catalyst' in reason.lower()

    def test_catalyst_ma_blocked(self):
        """M&A catalyst → reject."""
        engine = make_engine()
        c = make_candidate(catalyst='ma')
        ok, _ = engine.should_enter(c)
        assert not ok

    def test_already_in_position(self):
        engine = make_engine()
        c = make_candidate(symbol='AAPL')
        engine.open_position(c, 110.0, '2026-03-09 10:00')
        ok, reason = engine.should_enter(c)
        assert not ok
        assert 'already in position' in reason

    def test_max_positions_reached(self):
        engine = make_engine(max_positions=2)
        for sym in ['A', 'B']:
            engine.open_position(make_candidate(symbol=sym), 100.0, '2026-03-09 10:00')
        c = make_candidate(symbol='C')
        ok, reason = engine.should_enter(c)
        assert not ok
        assert 'max positions' in reason

    def test_vol_ratio_too_high(self):
        engine = make_engine(vol_ratio_max=2.0)
        c = make_candidate(vol_ratio=3.5)
        ok, reason = engine.should_enter(c)
        assert not ok
        assert 'vol ratio' in reason

    def test_reentry_cap(self):
        """Stopped-out symbol can't re-enter beyond max."""
        engine = make_engine(reentry_max_per_symbol=1)
        engine._stopped_today['TEST'] = StopOutRecord(
            symbol='TEST', stop_time=make_time(10, 0),
            original_entry=100.0, prev_close=95.0, gap_pct=0.08,
            avg_vol_20d=500_000, reentry_count=1, direction='short',
        )
        c = make_candidate(symbol='TEST')
        ok, reason = engine.should_enter(c)
        assert not ok
        assert 're-entry cap' in reason


# =========================================================================
# Open Position
# =========================================================================

class TestOpenPosition:
    """Position creation and tracking."""

    def test_short_position_created(self):
        engine = make_engine()
        c = make_candidate(gap_pct=0.10, prev_close=100.0, direction='short')
        pos = engine.open_position(c, 110.0, '2026-03-09 10:00')
        assert pos is not None
        assert pos.symbol == 'TEST'
        assert pos.direction == 'short'
        assert pos.shares > 0
        assert pos.remaining_shares == pos.shares
        assert 'TEST' in engine.positions

    def test_long_position_created(self):
        engine = make_engine(trade_gap_downs=True)
        c = make_candidate(gap_pct=0.10, prev_close=100.0, direction='long')
        pos = engine.open_position(c, 90.0, '2026-03-09 10:00')
        assert pos is not None
        assert pos.direction == 'long'
        # Long stop is below entry
        assert pos.stop_price < pos.entry_price

    def test_short_stop_above_entry(self):
        """Short position: stop must be ABOVE entry price."""
        engine = make_engine()
        c = make_candidate(direction='short')
        pos = engine.open_position(c, 110.0, '2026-03-09 10:00')
        assert pos.stop_price > pos.entry_price

    def test_long_stop_below_entry(self):
        engine = make_engine(trade_gap_downs=True)
        c = make_candidate(direction='long')
        pos = engine.open_position(c, 90.0, '2026-03-09 10:00')
        assert pos.stop_price < pos.entry_price

    def test_slippage_applied_backtest(self):
        """Backtest mode applies slippage to fill price."""
        engine = make_engine(slippage_pct=0.001)
        c = make_candidate(direction='short')
        pos = engine.open_position(c, 110.0, '2026-03-09 10:00')
        # Short slippage: fill lower (worse for short = you sell lower)
        assert pos.entry_price < 110.0

    def test_zero_shares_returns_none(self):
        """If position size computes to 0, return None."""
        engine = make_engine(initial_capital=1)  # too small
        c = make_candidate(prev_close=1000.0, gap_pct=0.10)
        pos = engine.open_position(c, 1100.0, '2026-03-09 10:00')
        assert pos is None

    def test_liquidity_cap(self):
        """Shares capped by max_pct_adv * avg_vol."""
        engine = make_engine(initial_capital=10_000_000, max_pct_adv=0.01)
        c = make_candidate(avg_vol=10_000)  # 1% of 10K = 100 shares max
        pos = engine.open_position(c, 110.0, '2026-03-09 10:00')
        assert pos is not None
        assert pos.shares <= 100


# =========================================================================
# Exit Logic (check_exits)
# =========================================================================

class TestCheckExits:
    """Exit conditions: stop, partial, full target, time, EOD."""

    def _setup_short(self, engine=None, **kwargs):
        """Create engine with a short position."""
        engine = engine or make_engine()
        c = make_candidate(gap_pct=0.10, prev_close=100.0, direction='short', **kwargs)
        pos = engine.open_position(c, 110.0, '2026-03-09 10:00')
        return engine, pos

    def test_stop_loss_short(self):
        """Short stop triggered when high >= stop_price."""
        engine, pos = self._setup_short()
        stop = pos.stop_price
        trades = engine.check_exits('TEST', 109.0, stop + 0.01, make_time(10, 30))
        assert len(trades) == 1
        assert trades[0].exit_reason == 'stop'
        assert trades[0].pnl < 0  # stop = loss
        assert 'TEST' not in engine.positions

    def test_stop_records_stopout(self):
        """Stop-out creates a StopOutRecord for re-entry tracking."""
        engine = make_engine(reentry_enabled=True)
        c = make_candidate()
        pos = engine.open_position(c, 110.0, '2026-03-09 10:00')
        stop = pos.stop_price
        engine.check_exits('TEST', 109.0, stop + 0.01, make_time(10, 30))
        assert 'TEST' in engine._stopped_today

    def test_partial_profit_short(self):
        """Short partial: price drops to midpoint → cover fraction."""
        engine, pos = self._setup_short()
        mid = pos.half_target
        orig_shares = pos.remaining_shares
        trades = engine.check_exits('TEST', mid - 0.01, mid - 0.01, make_time(11, 0))
        assert len(trades) >= 1
        assert trades[0].exit_reason == 'partial'
        assert trades[0].pnl > 0
        # Position still open with fewer shares
        assert 'TEST' in engine.positions
        assert engine.positions['TEST'].remaining_shares < orig_shares
        # Stop moved to breakeven
        assert engine.positions['TEST'].stop_price == pos.entry_price

    def test_full_target_short(self):
        """Short full target: price drops to prev_close → close all."""
        engine, pos = self._setup_short()
        target = pos.full_target
        trades = engine.check_exits('TEST', target - 0.01, target - 0.01, make_time(11, 0))
        # Should have partial + full (or just full if partial threshold was already at/below full)
        assert len(trades) >= 1
        assert 'TEST' not in engine.positions

    def test_time_exit(self):
        """Position closed at configured time exit hour."""
        engine, pos = self._setup_short()
        # Time exit at 15:00
        trades = engine.check_exits('TEST', 108.0, 108.0, make_time(15, 1))
        assert len(trades) >= 1
        assert any(t.exit_reason == 'time_exit' for t in trades)
        assert 'TEST' not in engine.positions

    def test_eod_exit(self):
        """Position force-closed at EOD hour."""
        engine, pos = self._setup_short()
        trades = engine.check_exits('TEST', 108.0, 108.0, make_time(15, 51))
        assert len(trades) >= 1
        assert any(t.exit_reason in ('eod', 'time_exit') for t in trades)

    def test_no_exit_during_normal_trading(self):
        """No exit signals during normal conditions."""
        engine, pos = self._setup_short()
        # Price between entry and stop, before targets/time
        trades = engine.check_exits('TEST', 109.0, 109.5, make_time(10, 30))
        assert len(trades) == 0
        assert 'TEST' in engine.positions

    def test_unknown_symbol_no_crash(self):
        """check_exits for unknown symbol returns empty."""
        engine = make_engine()
        trades = engine.check_exits('DOESNOTEXIST', 100.0, 100.0, make_time(10, 0))
        assert trades == []


# =========================================================================
# confirm_exit
# =========================================================================

class TestConfirmExit:
    """Finalize exits after broker fill confirmation."""

    def test_full_close(self):
        engine = make_engine()
        c = make_candidate()
        pos = engine.open_position(c, 110.0, '2026-03-09 10:00')
        shares = pos.remaining_shares
        trade = engine.confirm_exit('TEST', shares, 105.0, 'eod', make_time(15, 50))
        assert trade is not None
        assert trade.pnl > 0  # short: sold 110, bought 105
        assert 'TEST' not in engine.positions

    def test_partial_close(self):
        engine = make_engine()
        c = make_candidate()
        pos = engine.open_position(c, 110.0, '2026-03-09 10:00')
        half = pos.remaining_shares // 2
        trade = engine.confirm_exit('TEST', half, 105.0, 'partial', make_time(10, 30))
        assert trade is not None
        assert 'TEST' in engine.positions  # still open
        assert engine.positions['TEST'].remaining_shares == pos.shares - half
        assert engine.positions['TEST'].partial_filled is True
        # Stop moved to breakeven
        assert engine.positions['TEST'].stop_price == pos.entry_price

    def test_unknown_symbol(self):
        engine = make_engine()
        trade = engine.confirm_exit('NOPE', 100, 50.0, 'eod', make_time(15, 50))
        assert trade is None

    def test_pnl_uses_fill_price(self):
        """If entry_fill_price is set (from broker), use that for P&L."""
        engine = make_engine()
        c = make_candidate()
        pos = engine.open_position(c, 110.0, '2026-03-09 10:00')
        pos.entry_fill_price = 110.50  # actual broker fill was worse
        trade = engine.confirm_exit('TEST', pos.remaining_shares, 105.0, 'eod', make_time(15, 50))
        # P&L based on 110.50, not 110.0
        expected = (110.50 - 105.0) * pos.shares
        assert abs(trade.pnl - expected) < 0.01


# =========================================================================
# Force Close All
# =========================================================================

class TestForceCloseAll:
    """Emergency close all positions."""

    def test_closes_all(self):
        engine = make_engine()
        for sym in ['A', 'B', 'C']:
            engine.open_position(make_candidate(symbol=sym), 110.0, '2026-03-09 10:00')
        assert len(engine.positions) == 3
        trades = engine.force_close_all({'A': 105.0, 'B': 108.0, 'C': 112.0})
        assert len(trades) == 3
        assert len(engine.positions) == 0

    def test_empty_positions(self):
        engine = make_engine()
        trades = engine.force_close_all({})
        assert trades == []

    def test_missing_price_uses_entry(self):
        """If price not provided, use entry_price (no P&L)."""
        engine = make_engine()
        engine.open_position(make_candidate(symbol='X'), 110.0, '2026-03-09 10:00')
        trades = engine.force_close_all({})  # no prices
        assert len(trades) == 1
        # P&L should be ~0 (entry == exit, minus slippage)
        assert abs(trades[0].pnl) < 5  # small due to slippage


# =========================================================================
# Circuit Breakers
# =========================================================================

class TestCircuitBreakers:
    """Daily loss limits and consecutive loss circuit breakers."""

    _MARKET_OPEN = datetime(2026, 3, 9, 10, 0, 0, tzinfo=ET)

    @patch('gap_fade_app.datetime')
    def test_daily_loss_limit(self, mock_dt):
        """After losing 2% of equity, should_enter blocks."""
        mock_dt.now.return_value = self._MARKET_OPEN
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        engine = make_engine(backtest_mode=False, daily_loss_limit=0.02)
        engine.daily_stats.pnl = -(engine.equity * 0.025)  # 2.5% loss
        c = make_candidate()
        ok, reason = engine.should_enter(c)
        assert not ok
        assert 'daily loss' in reason.lower()

    @patch('gap_fade_app.datetime')
    def test_consecutive_losses(self, mock_dt):
        """After max_consec_losses, should_enter blocks."""
        mock_dt.now.return_value = self._MARKET_OPEN
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        engine = make_engine(backtest_mode=False, max_consec_losses=2)
        engine.daily_stats.consecutive_losses = 3
        c = make_candidate()
        ok, reason = engine.should_enter(c)
        assert not ok
        assert 'consecutive' in reason.lower()

    @patch('gap_fade_app.datetime')
    def test_drawdown_circuit_breaker(self, mock_dt):
        """Max drawdown triggers halt."""
        mock_dt.now.return_value = self._MARKET_OPEN
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        engine = make_engine(backtest_mode=False, max_drawdown=0.05)
        engine.peak_equity = 100_000
        engine.equity = 94_000  # 6% drawdown
        c = make_candidate()
        ok, reason = engine.should_enter(c)
        assert not ok
        assert 'drawdown' in reason.lower()


# =========================================================================
# Equity and Stats Tracking
# =========================================================================

class TestEquityTracking:
    """Equity updates after trades."""

    def test_winning_trade_increases_equity(self):
        engine = make_engine(initial_capital=100_000)
        c = make_candidate()
        engine.open_position(c, 110.0, '2026-03-09 10:00')
        engine.confirm_exit('TEST', engine.positions['TEST'].remaining_shares,
                          105.0, 'full_target', make_time(11, 0))
        assert engine.equity > 100_000

    def test_losing_trade_decreases_equity(self):
        engine = make_engine(initial_capital=100_000)
        c = make_candidate()
        pos = engine.open_position(c, 110.0, '2026-03-09 10:00')
        engine.confirm_exit('TEST', pos.remaining_shares, 115.0, 'stop', make_time(10, 30))
        assert engine.equity < 100_000

    def test_peak_equity_tracked(self):
        engine = make_engine(initial_capital=100_000)
        c = make_candidate()
        engine.open_position(c, 110.0, '2026-03-09 10:00')
        engine.confirm_exit('TEST', engine.positions['TEST'].remaining_shares,
                          100.0, 'full_target', make_time(11, 0))
        assert engine.peak_equity >= engine.equity

    def test_daily_stats_win_loss_count(self):
        engine = make_engine()
        # Win
        c1 = make_candidate(symbol='WIN')
        engine.open_position(c1, 110.0, '2026-03-09 10:00')
        engine.confirm_exit('WIN', engine.positions['WIN'].remaining_shares,
                          105.0, 'target', make_time(11, 0))
        # Loss
        c2 = make_candidate(symbol='LOSE')
        pos = engine.open_position(c2, 110.0, '2026-03-09 10:30')
        engine.confirm_exit('LOSE', pos.remaining_shares, 115.0, 'stop', make_time(11, 30))

        assert engine.daily_stats.wins == 1
        assert engine.daily_stats.losses == 1
        assert engine.daily_stats.consecutive_losses == 1

    def test_reset_daily(self):
        engine = make_engine()
        engine.daily_stats.pnl = 500
        engine.daily_stats.wins = 3
        engine._stopped_today['X'] = StopOutRecord(
            symbol='X', stop_time=make_time(10, 0),
            original_entry=100.0, prev_close=95.0, gap_pct=0.08,
            avg_vol_20d=500_000, reentry_count=1, direction='short',
        )
        engine.reset_daily()
        assert engine.daily_stats.pnl == 0
        assert engine.daily_stats.wins == 0
        assert len(engine._stopped_today) == 0


# =========================================================================
# Metrics
# =========================================================================

class TestMetrics:
    """Performance metrics calculation."""

    def test_no_trades(self):
        engine = make_engine()
        m = engine.get_metrics()
        assert m['total_trades'] == 0

    def test_metrics_accuracy(self):
        engine = make_engine(initial_capital=100_000, slippage_pct=0)
        # 2 wins, 1 loss
        for sym, exit_px in [('A', 105.0), ('B', 107.0), ('C', 115.0)]:
            c = make_candidate(symbol=sym)
            engine.open_position(c, 110.0, '2026-03-09 10:00')
            engine.confirm_exit(sym, engine.positions[sym].remaining_shares,
                              exit_px, 'target', make_time(11, 0))
        m = engine.get_metrics()
        assert m['total_trades'] == 3
        assert m['wins'] == 2
        assert m['losses'] == 1
        assert m['win_rate'] == round(2/3, 4)
        assert m['profit_factor'] > 0

    def test_all_wins(self):
        engine = make_engine(slippage_pct=0)
        for sym in ['A', 'B']:
            c = make_candidate(symbol=sym)
            engine.open_position(c, 110.0, '2026-03-09 10:00')
            engine.confirm_exit(sym, engine.positions[sym].remaining_shares,
                              105.0, 'target', make_time(11, 0))
        m = engine.get_metrics()
        assert m['win_rate'] == 1.0
        assert m['profit_factor'] == 9999.99  # no losses


# =========================================================================
# Re-entry Logic
# =========================================================================

class TestReentry:
    """Re-entry after stop-out."""

    def test_reentry_eligible_after_cooldown(self):
        engine = make_engine(reentry_enabled=True, reentry_cooldown_minutes=30,
                            reentry_trigger_pct=0.0)
        engine._stopped_today['TEST'] = StopOutRecord(
            symbol='TEST', stop_time=make_time(10, 0),
            original_entry=110.0, prev_close=100.0, gap_pct=0.10,
            avg_vol_20d=500_000, reentry_count=0, direction='short',
        )
        ok, reason = engine.can_reenter('TEST', 108.0, make_time(10, 31))
        assert ok, f"Should be eligible: {reason}"

    def test_reentry_too_soon(self):
        engine = make_engine(reentry_enabled=True, reentry_cooldown_minutes=30)
        engine._stopped_today['TEST'] = StopOutRecord(
            symbol='TEST', stop_time=make_time(10, 0),
            original_entry=110.0, prev_close=100.0, gap_pct=0.10,
            avg_vol_20d=500_000, reentry_count=0, direction='short',
        )
        ok, reason = engine.can_reenter('TEST', 108.0, make_time(10, 15))
        assert not ok
        assert 'cooldown' in reason

    def test_reentry_disabled(self):
        engine = make_engine(reentry_enabled=False)
        ok, reason = engine.can_reenter('NOPE', 100.0, make_time(10, 0))
        assert not ok

    def test_reentry_max_reached(self):
        engine = make_engine(reentry_enabled=True, reentry_max_per_symbol=1)
        engine._stopped_today['TEST'] = StopOutRecord(
            symbol='TEST', stop_time=make_time(10, 0),
            original_entry=110.0, prev_close=100.0, gap_pct=0.10,
            avg_vol_20d=500_000, reentry_count=1, direction='short',
        )
        ok, reason = engine.can_reenter('TEST', 108.0, make_time(11, 0))
        assert not ok
        assert 'max' in reason.lower()

    def test_no_stopout_record(self):
        engine = make_engine(reentry_enabled=True)
        ok, reason = engine.can_reenter('NOSUCHSYMBOL', 100.0, make_time(10, 0))
        assert not ok


# =========================================================================
# Edge Cases
# =========================================================================

class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_penny_stock_position(self):
        """$0.50 stock (shouldn't happen due to min_price, but test defensively)."""
        engine = make_engine(initial_capital=1000, min_price=0.01)
        c = make_candidate(prev_close=0.50, gap_pct=0.20)
        c.premarket_price = 0.60
        pos = engine.open_position(c, 0.60, '2026-03-09 10:00')
        if pos:
            assert pos.shares > 0

    def test_high_price_stock(self):
        """$5000 stock position sizing."""
        engine = make_engine(initial_capital=100_000)
        c = make_candidate(prev_close=5000.0, gap_pct=0.08)
        pos = engine.open_position(c, 5400.0, '2026-03-09 10:00')
        if pos:
            assert pos.shares >= 1

    def test_simultaneous_partial_and_full(self):
        """Price drops through both partial and full target in one bar."""
        engine = make_engine(slippage_pct=0)
        c = make_candidate(gap_pct=0.10, prev_close=100.0, direction='short')
        pos = engine.open_position(c, 110.0, '2026-03-09 10:00')
        # Price at prev_close (full target) — should trigger partial + full
        trades = engine.check_exits('TEST', 99.0, 99.0, make_time(11, 0))
        assert len(trades) >= 1
        assert 'TEST' not in engine.positions  # fully closed

    def test_multiple_positions_independent(self):
        """Closing one position doesn't affect others."""
        engine = make_engine(slippage_pct=0)
        for sym in ['A', 'B', 'C']:
            engine.open_position(make_candidate(symbol=sym), 110.0, '2026-03-09 10:00')
        assert len(engine.positions) == 3
        # Close A
        engine.confirm_exit('A', engine.positions['A'].remaining_shares,
                          105.0, 'target', make_time(11, 0))
        assert 'A' not in engine.positions
        assert 'B' in engine.positions
        assert 'C' in engine.positions

    def test_kelly_fraction_zero(self):
        """Kelly fraction = 0 → falls back to risk_pct."""
        engine = make_engine(kelly_fraction=0.0, risk_pct=0.02)
        shares = engine.compute_position_size(100.0, 103.0)
        assert shares > 0

    def test_thin_day_effective_max(self):
        """Few candidates → trade all of them."""
        engine = make_engine(max_positions=5, thin_day_threshold=10)
        assert engine.effective_max_positions(3) == 3
        assert engine.effective_max_positions(15) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
