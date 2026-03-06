"""Integration test: simulate a complete trading day through the engine.

Walks through: pre-market scan → entries → intraday management → EOD close.
Uses the actual GapFadeEngine with backtest mode (no broker calls).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from datetime import datetime
from unittest.mock import patch
from tests.conftest import make_config, make_engine, make_candidate, make_time, ET

_MARKET_OPEN = datetime(2026, 3, 9, 10, 0, 0, tzinfo=ET)


class TestFullTradingDay:
    """Simulate a realistic trading day end-to-end."""

    def _run_day(self, engine, candidates, entry_time, prices_over_time):
        """Simulate a trading day.

        Args:
            engine: GapFadeEngine in backtest mode
            candidates: list of GapCandidate
            entry_time: datetime for entries
            prices_over_time: list of (datetime, {symbol: (price, high)}) tuples
        Returns:
            dict with day summary
        """
        # Phase 1: Entry
        entered = []
        engine._effective_max_positions = engine.effective_max_positions(len(candidates))
        for c in candidates:
            ok, reason = engine.should_enter(c)
            if ok:
                pos = engine.open_position(c, c.premarket_price, entry_time.strftime('%Y-%m-%d %H:%M'))
                if pos:
                    entered.append(c.symbol)

        # Phase 2: Intraday price checks
        all_trades = []
        for check_time, price_dict in prices_over_time:
            for sym in list(engine.positions.keys()):
                if sym in price_dict:
                    price, high = price_dict[sym]
                    trades = engine.check_exits(sym, price, high, check_time)
                    all_trades.extend(trades)

        # Phase 3: EOD force close
        remaining = list(engine.positions.keys())
        if remaining:
            eod_prices = {}
            for sym in remaining:
                # Use last known price or entry price
                eod_prices[sym] = engine.positions[sym].entry_price * 0.98  # slight profit
            eod_trades = engine.force_close_all(eod_prices, reason='eod')
            all_trades.extend(eod_trades)

        return {
            'entered': entered,
            'trades': all_trades,
            'final_equity': engine.equity,
            'positions_remaining': len(engine.positions),
            'wins': engine.daily_stats.wins,
            'losses': engine.daily_stats.losses,
        }

    def test_normal_day_3_entries(self):
        """Standard day: 3 entries, mix of outcomes."""
        engine = make_engine(max_positions=5, slippage_pct=0)
        candidates = [
            make_candidate(symbol='AAPL', gap_pct=0.08, prev_close=150.0),
            make_candidate(symbol='MSFT', gap_pct=0.10, prev_close=300.0),
            make_candidate(symbol='TSLA', gap_pct=0.12, prev_close=200.0),
        ]

        # Simulate price movements
        prices = [
            # 10:30 — AAPL drops (profit), MSFT flat, TSLA rises (stop risk)
            (make_time(10, 30), {
                'AAPL': (158.0, 163.0),  # was ~162 (8% gap), now 158
                'MSFT': (328.0, 331.0),  # was ~330 (10% gap), flat
                'TSLA': (226.0, 226.0),  # was ~224 (12% gap), rises
            }),
            # 11:00 — AAPL at target
            (make_time(11, 0), {
                'AAPL': (152.0, 158.0),
                'MSFT': (325.0, 330.0),
                'TSLA': (220.0, 225.0),
            }),
            # 14:00 — midday
            (make_time(14, 0), {
                'AAPL': (150.0, 155.0),
                'MSFT': (310.0, 325.0),
                'TSLA': (215.0, 222.0),
            }),
        ]

        result = self._run_day(engine, candidates, make_time(9, 35), prices)

        assert len(result['entered']) == 3
        assert result['positions_remaining'] == 0
        assert len(result['trades']) >= 3  # at least 3 closes (may have partials)
        assert result['final_equity'] != engine.config.initial_capital  # something happened

    def test_all_stops_hit(self):
        """Bad day: all positions hit stops."""
        engine = make_engine(max_positions=3, slippage_pct=0, stop_pct=0.02)
        candidates = [
            make_candidate(symbol=f'BAD{i}', gap_pct=0.10, prev_close=100.0)
            for i in range(3)
        ]

        # Prices go against us — all stops hit
        prices = []
        for sym_candidate in candidates:
            sym = sym_candidate.symbol
        # All stocks gap higher (bad for shorts)
        prices = [
            (make_time(10, 0), {
                f'BAD{i}': (115.0, 115.0) for i in range(3)
            }),
        ]

        result = self._run_day(engine, candidates, make_time(9, 35), prices)

        assert result['positions_remaining'] == 0
        assert result['losses'] >= 1  # at least some stopped out
        assert result['final_equity'] < engine.config.initial_capital

    def test_empty_day_no_candidates(self):
        """No candidates → no trades."""
        engine = make_engine()
        result = self._run_day(engine, [], make_time(9, 35), [])
        assert result['entered'] == []
        assert result['trades'] == []
        assert result['final_equity'] == engine.config.initial_capital

    @patch('gap_fade_app.datetime')
    def test_circuit_breaker_stops_entries(self, mock_dt):
        """After 2 consecutive losses, further entries blocked."""
        mock_dt.now.return_value = _MARKET_OPEN
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        engine = make_engine(backtest_mode=False, max_consec_losses=2, max_positions=5, slippage_pct=0)

        # Enter and stop out 2 positions
        for i in range(2):
            c = make_candidate(symbol=f'STOP{i}', gap_pct=0.10, prev_close=100.0)
            pos = engine.open_position(c, 110.0, '2026-03-09 10:00')
            stop = pos.stop_price
            engine.check_exits(c.symbol, 109.0, stop + 1.0, make_time(10, 15 + i * 15))

        assert engine.daily_stats.consecutive_losses == 2

        # Third entry should be blocked
        c3 = make_candidate(symbol='BLOCKED')
        ok, reason = engine.should_enter(c3)
        assert not ok
        assert 'consecutive' in reason.lower()


class TestMultiDayState:
    """State persistence across trading days."""

    def test_equity_carries_over(self):
        """Equity from day 1 carries into day 2."""
        engine = make_engine(initial_capital=100_000, slippage_pct=0)

        # Day 1: one winning trade
        c = make_candidate(symbol='DAY1')
        engine.open_position(c, 110.0, '2026-03-09 10:00')
        engine.confirm_exit('DAY1', engine.positions['DAY1'].remaining_shares,
                          105.0, 'target', make_time(11, 0))
        day1_equity = engine.equity
        assert day1_equity > 100_000

        # Reset daily (simulating overnight)
        engine.reset_daily()

        # Day 2: equity preserved
        assert engine.equity == day1_equity
        assert engine.daily_stats.pnl == 0  # daily stats reset
        assert len(engine.positions) == 0
        assert len(engine._stopped_today) == 0

    def test_trade_log_persists(self):
        """all_trade_log accumulates across days."""
        engine = make_engine(slippage_pct=0)

        # Day 1
        c1 = make_candidate(symbol='D1')
        engine.open_position(c1, 110.0, '2026-03-09 10:00')
        engine.confirm_exit('D1', engine.positions['D1'].remaining_shares,
                          105.0, 'target', make_time(11, 0))
        engine.reset_daily()

        # Day 2
        c2 = make_candidate(symbol='D2')
        engine.open_position(c2, 110.0, '2026-03-10 10:00')
        engine.confirm_exit('D2', engine.positions['D2'].remaining_shares,
                          107.0, 'target', make_time(11, 0))

        assert len(engine.all_trade_log) == 2  # both days
        assert len(engine.trade_log) == 1  # only day 2


class TestDataIntegrity:
    """Verify data consistency after trading operations."""

    def test_trade_fields_populated(self):
        """All required TradeRecord fields are populated."""
        engine = make_engine(slippage_pct=0)
        c = make_candidate(symbol='FULL', gap_pct=0.10)
        engine.open_position(c, 110.0, '2026-03-09 10:00')
        trade = engine.confirm_exit('FULL', engine.positions['FULL'].remaining_shares,
                                   105.0, 'target', make_time(11, 0))

        assert trade.symbol == 'FULL'
        assert trade.entry_price > 0
        assert trade.exit_price > 0
        assert trade.shares > 0
        assert trade.pnl != 0  # non-zero
        assert trade.entry_time != ''
        assert trade.exit_time != ''
        assert trade.exit_reason == 'target'
        assert trade.side in ('short', 'long')
        assert trade.holding_minutes >= 0

    def test_no_duplicate_trades(self):
        """Same position can't generate duplicate trade records."""
        engine = make_engine(slippage_pct=0)
        c = make_candidate(symbol='ONCE')
        engine.open_position(c, 110.0, '2026-03-09 10:00')
        shares = engine.positions['ONCE'].remaining_shares

        t1 = engine.confirm_exit('ONCE', shares, 105.0, 'target', make_time(11, 0))
        assert t1 is not None

        # Second attempt — position already gone
        t2 = engine.confirm_exit('ONCE', shares, 105.0, 'target', make_time(11, 1))
        assert t2 is None

    def test_equity_matches_trades(self):
        """Final equity = initial + sum of all trade P&L."""
        engine = make_engine(initial_capital=100_000, slippage_pct=0)
        for i, exit_px in enumerate([105.0, 108.0, 115.0, 103.0]):
            sym = f'T{i}'
            c = make_candidate(symbol=sym)
            engine.open_position(c, 110.0, '2026-03-09 10:00')
            engine.confirm_exit(sym, engine.positions[sym].remaining_shares,
                              exit_px, 'target', make_time(11, 0))

        total_pnl = sum(t.pnl for t in engine.all_trade_log)
        assert abs(engine.equity - (100_000 + total_pnl)) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
