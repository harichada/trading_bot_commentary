"""EOD edge case tests — the scenarios that caused real production failures.

Every test here represents either a real incident or a plausible failure mode
that could cause overnight position exposure.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from tests.conftest import make_config, make_engine, make_candidate, make_time, ET

_MARKET_OPEN = datetime(2026, 3, 9, 10, 0, 0, tzinfo=ET)


class TestEODPositionEntry:
    """Positions opened near EOD must still be closed."""

    def test_position_at_349_closed_by_engine(self):
        """Position opened at 3:49 PM → engine check_exits at 3:50 closes it."""
        engine = make_engine()
        c = make_candidate(symbol='LATE')
        pos = engine.open_position(c, 110.0, '2026-03-09 15:49')
        # check_exits at 15:50 (EOD exit)
        trades = engine.check_exits('LATE', 109.0, 109.0, make_time(15, 50))
        assert len(trades) >= 1
        assert 'LATE' not in engine.positions

    def test_position_at_1530_time_exit(self):
        """Position opened at 3:00 → time exit triggers."""
        engine = make_engine(time_exit_hour=15, time_exit_min=0)
        c = make_candidate(symbol='TIMEX')
        engine.open_position(c, 110.0, '2026-03-09 10:00')
        trades = engine.check_exits('TIMEX', 109.0, 109.0, make_time(15, 1))
        assert len(trades) >= 1
        assert any(t.exit_reason == 'time_exit' for t in trades)


class TestEODPartialFills:
    """Partially filled positions at EOD."""

    def test_partial_then_eod_closes_remaining(self):
        """Partial fill at 11 AM, EOD closes remaining shares."""
        engine = make_engine(slippage_pct=0)
        c = make_candidate(gap_pct=0.10, prev_close=100.0, direction='short')
        pos = engine.open_position(c, 110.0, '2026-03-09 10:00')
        orig_shares = pos.remaining_shares

        # Partial at midpoint
        mid = pos.half_target
        trades1 = engine.check_exits('TEST', mid - 0.5, mid - 0.5, make_time(11, 0))
        assert len(trades1) >= 1
        assert engine.positions['TEST'].partial_filled
        remaining = engine.positions['TEST'].remaining_shares
        assert remaining < orig_shares

        # EOD close remaining
        trades2 = engine.check_exits('TEST', 108.0, 108.0, make_time(15, 51))
        assert 'TEST' not in engine.positions
        total_shares_closed = sum(t.shares for t in trades1 + trades2)
        assert total_shares_closed == orig_shares


class TestMultiplePositionsEOD:
    """EOD with multiple open positions."""

    def test_all_five_positions_closed(self):
        """5 simultaneous positions all closed at EOD."""
        engine = make_engine(max_positions=5, slippage_pct=0)
        symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
        for sym in symbols:
            engine.open_position(make_candidate(symbol=sym), 110.0, '2026-03-09 10:00')
        assert len(engine.positions) == 5

        # Force close all
        prices = {s: 108.0 for s in symbols}
        trades = engine.force_close_all(prices, reason='eod')
        assert len(trades) == 5
        assert len(engine.positions) == 0

    def test_mixed_pnl_all_closed(self):
        """Mix of winning and losing positions all closed at EOD."""
        engine = make_engine(slippage_pct=0)
        engine.open_position(make_candidate(symbol='WIN'), 110.0, '2026-03-09 10:00')
        engine.open_position(make_candidate(symbol='LOSE'), 110.0, '2026-03-09 10:00')

        trades = engine.force_close_all({'WIN': 105.0, 'LOSE': 115.0})
        assert len(trades) == 2
        assert len(engine.positions) == 0
        win_trade = [t for t in trades if t.symbol == 'WIN'][0]
        lose_trade = [t for t in trades if t.symbol == 'LOSE'][0]
        assert win_trade.pnl > 0
        assert lose_trade.pnl < 0


class TestEODWithCircuitBreaker:
    """Circuit breaker active but EOD must still close."""

    @patch('gap_fade_app.datetime')
    def test_halted_positions_still_close(self, mock_dt):
        """Even if circuit breaker blocks new entries, existing positions close at EOD."""
        mock_dt.now.return_value = _MARKET_OPEN
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        engine = make_engine(backtest_mode=False, daily_loss_limit=0.02)
        # Open a position
        engine.open_position(make_candidate(symbol='STUCK'), 110.0, '2026-03-09 10:00')
        # Trigger circuit breaker
        engine.daily_stats.pnl = -(engine.equity * 0.03)  # 3% loss

        # New entries blocked
        ok, _ = engine.should_enter(make_candidate(symbol='NEW'))
        assert not ok

        # But existing position still closes at EOD
        trades = engine.check_exits('STUCK', 108.0, 108.0, make_time(15, 51))
        assert len(trades) >= 1
        assert 'STUCK' not in engine.positions


class TestEODGapDownLong:
    """EOD handling for long (gap-down) positions."""

    def test_long_position_eod_close(self):
        engine = make_engine(trade_gap_downs=True, slippage_pct=0)
        c = make_candidate(direction='long', gap_pct=0.10, prev_close=100.0)
        pos = engine.open_position(c, 90.0, '2026-03-09 10:00')
        assert pos.direction == 'long'

        trades = engine.force_close_all({c.symbol: 92.0})
        assert len(trades) == 1
        assert trades[0].pnl > 0  # long: 92 > 90


class TestEODStateConsistency:
    """State consistency after EOD close."""

    def test_equity_updated_after_eod(self):
        """Equity reflects all EOD trade P&L."""
        engine = make_engine(initial_capital=100_000, slippage_pct=0)
        for sym in ['A', 'B']:
            engine.open_position(make_candidate(symbol=sym), 110.0, '2026-03-09 10:00')
        trades = engine.force_close_all({'A': 105.0, 'B': 108.0})
        total_pnl = sum(t.pnl for t in trades)
        assert abs(engine.equity - (100_000 + total_pnl)) < 1.0

    def test_trade_log_complete(self):
        """All EOD trades appear in all_trade_log."""
        engine = make_engine()
        for sym in ['X', 'Y', 'Z']:
            engine.open_position(make_candidate(symbol=sym), 110.0, '2026-03-09 10:00')
        before = len(engine.all_trade_log)
        engine.force_close_all({s: 108.0 for s in ['X', 'Y', 'Z']})
        assert len(engine.all_trade_log) == before + 3

    def test_no_orphaned_positions(self):
        """After force_close_all, positions dict is empty."""
        engine = make_engine()
        for i in range(5):
            engine.open_position(make_candidate(symbol=f'S{i}'), 110.0, '2026-03-09 10:00')
        engine.force_close_all({f'S{i}': 108.0 for i in range(5)})
        assert len(engine.positions) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
