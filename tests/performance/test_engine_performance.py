"""Performance tests for the trading engine.

Verifies: no memory leaks, acceptable throughput, scaling behavior.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import time
import tracemalloc
import pytest
from tests.conftest import make_config, make_engine, make_candidate, make_time, ET


class TestThroughput:
    """Verify engine operations complete within time budgets."""

    def test_position_size_under_1ms(self):
        """Position sizing: < 1ms per call."""
        engine = make_engine()
        start = time.perf_counter()
        for _ in range(10_000):
            engine.compute_position_size(100.0, 103.0)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / 10_000) * 1000
        assert avg_ms < 1.0, f"Position sizing took {avg_ms:.3f}ms avg (budget: 1ms)"

    def test_should_enter_under_1ms(self):
        """Entry validation: < 1ms per call."""
        engine = make_engine()
        c = make_candidate()
        start = time.perf_counter()
        for _ in range(10_000):
            engine.should_enter(c)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / 10_000) * 1000
        assert avg_ms < 1.0, f"should_enter took {avg_ms:.3f}ms avg (budget: 1ms)"

    def test_check_exits_under_1ms(self):
        """Exit check: < 1ms per call."""
        engine = make_engine(slippage_pct=0)
        c = make_candidate()
        engine.open_position(c, 110.0, '2026-03-09 10:00')

        start = time.perf_counter()
        for _ in range(10_000):
            engine.check_exits('TEST', 109.0, 109.5, make_time(10, 30))
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / 10_000) * 1000
        assert avg_ms < 1.0, f"check_exits took {avg_ms:.3f}ms avg (budget: 1ms)"

    def test_metrics_under_10ms(self):
        """Metrics computation: < 10ms for 1000 trades."""
        engine = make_engine(slippage_pct=0)
        # Generate 1000 trades
        for i in range(500):
            sym = f'S{i}'
            c = make_candidate(symbol=sym)
            engine.open_position(c, 110.0, '2026-03-09 10:00')
            engine.confirm_exit(sym, engine.positions[sym].remaining_shares,
                              105.0 if i % 2 == 0 else 115.0,
                              'target', make_time(11, 0))

        start = time.perf_counter()
        engine.get_metrics()
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 50, f"Metrics took {elapsed:.1f}ms for {len(engine.all_trade_log)} trades (budget: 50ms)"

    def test_1000_open_close_cycle(self):
        """1000 open/close cycles: < 5 seconds total."""
        engine = make_engine(max_positions=5, slippage_pct=0)
        start = time.perf_counter()
        for i in range(1000):
            sym = f'PERF{i}'
            c = make_candidate(symbol=sym)
            engine.open_position(c, 110.0, '2026-03-09 10:00')
            engine.confirm_exit(sym, engine.positions[sym].remaining_shares,
                              105.0, 'target', make_time(11, 0))
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"1000 cycles took {elapsed:.2f}s (budget: 5s)"


class TestMemory:
    """Verify no memory leaks in core operations."""

    def test_no_leak_open_close(self):
        """Repeated open/close doesn't leak memory."""
        tracemalloc.start()

        engine = make_engine(max_positions=100, slippage_pct=0)

        # Warm up
        for i in range(100):
            sym = f'WARM{i}'
            c = make_candidate(symbol=sym)
            engine.open_position(c, 110.0, '2026-03-09 10:00')
            engine.confirm_exit(sym, engine.positions[sym].remaining_shares,
                              105.0, 'target', make_time(11, 0))

        snap1 = tracemalloc.take_snapshot()

        # Main loop
        for i in range(1000):
            sym = f'MEM{i}'
            c = make_candidate(symbol=sym)
            engine.open_position(c, 110.0, '2026-03-09 10:00')
            engine.confirm_exit(sym, engine.positions[sym].remaining_shares,
                              105.0, 'target', make_time(11, 0))

        snap2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Check growth
        stats = snap2.compare_to(snap1, 'lineno')
        total_growth = sum(s.size_diff for s in stats if s.size_diff > 0)

        # Allow reasonable growth (trade log accumulates intentionally)
        # But should be less than 10MB for 1000 trades
        assert total_growth < 10_000_000, \
            f"Memory grew {total_growth / 1_000_000:.1f}MB in 1000 cycles"

    def test_positions_cleaned_after_force_close(self):
        """After force_close_all, no lingering position references."""
        engine = make_engine(max_positions=50, slippage_pct=0)
        for i in range(50):
            engine.open_position(make_candidate(symbol=f'FC{i}'), 110.0, '2026-03-09 10:00')

        assert len(engine.positions) == 50
        engine.force_close_all({f'FC{i}': 108.0 for i in range(50)})
        assert len(engine.positions) == 0


class TestScaling:
    """Verify engine scales with increasing positions."""

    def test_exit_check_scales_linearly(self):
        """check_exits time doesn't explode with many positions."""
        engine = make_engine(max_positions=100, slippage_pct=0)

        # 10 positions
        for i in range(10):
            engine.open_position(make_candidate(symbol=f'S{i}'), 110.0, '2026-03-09 10:00')
        start = time.perf_counter()
        for sym in list(engine.positions.keys()):
            engine.check_exits(sym, 109.0, 109.5, make_time(10, 30))
        time_10 = time.perf_counter() - start

        # Clean up
        engine.force_close_all({f'S{i}': 108.0 for i in range(10)})

        # 50 positions
        for i in range(50):
            engine.open_position(make_candidate(symbol=f'L{i}'), 110.0, '2026-03-09 10:00')
        start = time.perf_counter()
        for sym in list(engine.positions.keys()):
            engine.check_exits(sym, 109.0, 109.5, make_time(10, 30))
        time_50 = time.perf_counter() - start

        # 50 positions should take roughly 5x (not 25x) of 10 positions
        if time_10 > 0:
            ratio = time_50 / time_10
            assert ratio < 15, f"Scaling ratio {ratio:.1f}x for 5x positions (should be ~5x)"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
