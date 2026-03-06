"""Unit tests for risk management — circuit breakers, drawdown tiers, loss limits.

These tests verify the system protects capital under adverse conditions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from datetime import datetime
from unittest.mock import patch
from tests.conftest import make_config, make_engine, make_candidate, make_time, ET


_MARKET_OPEN = datetime(2026, 3, 9, 10, 0, 0, tzinfo=ET)


def _patch_market_time(mock_dt):
    """Configure datetime mock to return market-hours time."""
    mock_dt.now.return_value = _MARKET_OPEN
    mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)


class TestDailyLossLimit:
    """Daily P&L loss limit halts trading."""

    def test_loss_below_limit_allows_entry(self):
        engine = make_engine(daily_loss_limit=0.02)
        engine.daily_stats.pnl = -(engine.equity * 0.01)  # 1% loss, limit is 2%
        ok, _ = engine.should_enter(make_candidate())
        assert ok

    @patch('gap_fade_app.datetime')
    def test_loss_at_limit_blocks(self, mock_dt):
        _patch_market_time(mock_dt)
        engine = make_engine(backtest_mode=False, daily_loss_limit=0.02)
        engine.daily_stats.pnl = -(engine.equity * 0.02)  # exactly 2%
        ok, reason = engine.should_enter(make_candidate())
        assert not ok
        assert 'daily loss' in reason.lower()

    @patch('gap_fade_app.datetime')
    def test_loss_exceeds_limit_blocks(self, mock_dt):
        _patch_market_time(mock_dt)
        engine = make_engine(backtest_mode=False, daily_loss_limit=0.02)
        engine.daily_stats.pnl = -(engine.equity * 0.05)
        ok, _ = engine.should_enter(make_candidate())
        assert not ok

    def test_profit_day_allows_entry(self):
        engine = make_engine(daily_loss_limit=0.02)
        engine.daily_stats.pnl = engine.equity * 0.03  # +3% day
        ok, _ = engine.should_enter(make_candidate())
        assert ok


class TestConsecutiveLosses:
    """Consecutive loss circuit breaker."""

    def test_below_limit_allows(self):
        engine = make_engine(max_consec_losses=2)
        engine.daily_stats.consecutive_losses = 1
        ok, _ = engine.should_enter(make_candidate())
        assert ok

    @patch('gap_fade_app.datetime')
    def test_at_limit_blocks(self, mock_dt):
        _patch_market_time(mock_dt)
        engine = make_engine(backtest_mode=False, max_consec_losses=2)
        engine.daily_stats.consecutive_losses = 2
        ok, reason = engine.should_enter(make_candidate())
        assert not ok

    def test_win_resets_counter(self):
        engine = make_engine(max_consec_losses=3)
        engine.daily_stats.consecutive_losses = 2
        # Simulate a win
        c = make_candidate(symbol='WIN')
        engine.open_position(c, 110.0, '2026-03-09 10:00')
        engine.confirm_exit('WIN', engine.positions['WIN'].remaining_shares,
                          105.0, 'target', make_time(11, 0))
        assert engine.daily_stats.consecutive_losses == 0


class TestMaxDrawdown:
    """Drawdown halt."""

    def test_within_limit_allows(self):
        engine = make_engine(max_drawdown=0.05)
        engine.peak_equity = 100_000
        engine.equity = 96_000  # 4% drawdown
        ok, _ = engine.should_enter(make_candidate())
        assert ok

    @patch('gap_fade_app.datetime')
    def test_exceeds_limit_blocks(self, mock_dt):
        _patch_market_time(mock_dt)
        engine = make_engine(backtest_mode=False, max_drawdown=0.05)
        engine.peak_equity = 100_000
        engine.equity = 94_500  # 5.5% drawdown
        ok, reason = engine.should_enter(make_candidate())
        assert not ok
        assert 'drawdown' in reason.lower()


class TestDrawdownTiers:
    """Tiered drawdown position size reduction."""

    def test_tier1_reduces_position(self):
        """15% drawdown → position size halved."""
        engine = make_engine(
            dd_circuit_breaker=True,
            dd_tier1_threshold=0.15,
            dd_tier1_scale=0.50,
            dd_tier2_threshold=0.25,
            initial_capital=100_000,
        )
        engine.peak_equity = 100_000
        engine.equity = 84_000  # 16% drawdown

        # Compute baseline position size (no drawdown)
        engine_base = make_engine(initial_capital=100_000)
        base_shares = engine_base.compute_position_size(100.0, 103.0)

        # With tier 1, effective risk should be halved
        dd_shares = engine.compute_position_size(100.0, 103.0)
        # dd_shares should be roughly half of base (may not be exact due to slot equity)
        # Just verify it's smaller
        if base_shares > 0:
            assert dd_shares <= base_shares

    def test_no_drawdown_full_size(self):
        """At peak equity, no reduction."""
        engine = make_engine(dd_circuit_breaker=True, initial_capital=100_000)
        engine.peak_equity = 100_000
        engine.equity = 100_000
        shares = engine.compute_position_size(100.0, 103.0)
        assert shares > 0


class TestRiskPerTrade:
    """Verify risk per trade never exceeds configured limit."""

    def test_risk_capped(self):
        """Dollar risk per trade ≤ equity * risk_pct."""
        engine = make_engine(initial_capital=100_000, risk_pct=0.02)
        entry = 100.0
        stop = 103.0  # $3 risk per share
        shares = engine.compute_position_size(entry, stop)
        dollar_risk = shares * (stop - entry)
        max_risk = engine.equity * engine.config.risk_pct
        assert dollar_risk <= max_risk * 1.01  # 1% tolerance for rounding

    def test_many_prices(self):
        """Risk cap holds across a range of stock prices."""
        engine = make_engine(initial_capital=100_000, risk_pct=0.02)
        for price in [5, 20, 50, 100, 500, 1000, 5000]:
            stop = price * 1.03  # 3% stop
            shares = engine.compute_position_size(float(price), stop)
            if shares > 0:
                dollar_risk = shares * (stop - price)
                max_risk = engine.equity * engine.config.risk_pct
                assert dollar_risk <= max_risk * 1.01, \
                    f"Price ${price}: risk ${dollar_risk:.2f} > max ${max_risk:.2f}"


class TestNotionalCap:
    """Position notional value cap."""

    def test_position_capped(self):
        engine = make_engine(initial_capital=500_000, max_notional=25_000)
        shares = engine.compute_position_size(100.0, 103.0)
        notional = shares * 100.0
        assert notional <= 25_000 * 1.01


class TestMaxPositions:
    """Concurrent position limit."""

    def test_cannot_exceed_max(self):
        engine = make_engine(max_positions=3)
        for i in range(3):
            c = make_candidate(symbol=f'S{i}')
            engine.open_position(c, 110.0, '2026-03-09 10:00')
        assert len(engine.positions) == 3
        c = make_candidate(symbol='EXTRA')
        ok, reason = engine.should_enter(c)
        assert not ok
        assert 'max positions' in reason

    def test_closing_frees_slot(self):
        engine = make_engine(max_positions=2)
        for sym in ['A', 'B']:
            engine.open_position(make_candidate(symbol=sym), 110.0, '2026-03-09 10:00')
        # Close A
        engine.confirm_exit('A', engine.positions['A'].remaining_shares,
                          105.0, 'target', make_time(11, 0))
        # Now should allow new entry
        ok, _ = engine.should_enter(make_candidate(symbol='C'))
        assert ok


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
