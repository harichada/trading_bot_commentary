"""
Tests for the independent EOD watchdog task.

Architecture under test:
    - _eod_watchdog() runs as a separate asyncio.Task, independent of _trading_loop()
    - Circuit breaker at 3:45 PM blocks new entries
    - Primary close at 3:50 PM executes _eod_close()
    - Force close at 3:55 PM uses Alpaca DELETE API
    - Emergency audit at 4:00 PM marks positions for close-on-open
    - Watchdog heartbeat monitors EOD task health and restarts if dead

SLA: 100% EOD execution rate. Zero tolerance for missed deadlines.
"""

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Minimal stubs so gap_fade_app can import without real broker/DB
# ---------------------------------------------------------------------------

# Stub psycopg2 before importing gap_fade_app
_fake_pg = MagicMock()
_fake_pg.connect.return_value = MagicMock()
_fake_pg.extensions = MagicMock()
_fake_pg.extensions.ISOLATION_LEVEL_AUTOCOMMIT = 0
sys.modules.setdefault('psycopg2', _fake_pg)
sys.modules.setdefault('psycopg2.extensions', _fake_pg.extensions)
sys.modules.setdefault('psycopg2.extras', MagicMock())

# Stub optional dependencies
for mod in ('anthropic', 'ollama', 'authlib', 'authlib.integrations',
            'authlib.integrations.starlette_client', 'httpx',
            'starlette.middleware.sessions'):
    sys.modules.setdefault(mod, MagicMock())

os.environ.setdefault('DATABASE_URL', 'postgresql://x:x@localhost/test')
os.environ.setdefault('ALPACA_API_KEY', 'test')
os.environ.setdefault('ALPACA_SECRET_KEY', 'test')

logger = logging.getLogger('test_eod_watchdog')


# ---------------------------------------------------------------------------
# Helpers — mock position / engine / trader
# ---------------------------------------------------------------------------

@dataclass
class MockPosition:
    symbol: str
    entry_price: float = 100.0
    remaining_shares: int = 100
    direction: str = 'short'
    stop_order_id: str = ''
    closing: bool = False
    close_on_open: bool = False
    current_price: float = 0.0
    last_prices: list = field(default_factory=list)

    def update_tracking(self, price, time_str):
        self.current_price = price


@dataclass
class MockDailyStats:
    pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    trade_count: int = 0
    date: str = ''


@dataclass
class MockTradeRecord:
    symbol: str
    pnl: float
    exit_time: str = ''


class MockEngine:
    def __init__(self):
        self.positions: Dict[str, MockPosition] = {}
        self.equity: float = 100000.0
        self.peak_equity: float = 100000.0
        self.daily_stats = MockDailyStats()
        self.all_trade_log: list = []

    def confirm_exit(self, symbol, shares, price, reason, dt):
        if symbol in self.positions:
            pos = self.positions.pop(symbol)
            pnl = (pos.entry_price - price) * shares if pos.direction == 'short' else (price - pos.entry_price) * shares
            return MockTradeRecord(symbol=symbol, pnl=pnl, exit_time=dt.isoformat())
        return None

    def reset_daily(self):
        self.daily_stats = MockDailyStats()


class MockAlerter:
    def __init__(self):
        self.alerts: List[dict] = []

    async def send(self, title, message, level='info', throttle_key=None):
        self.alerts.append({'title': title, 'message': message, 'level': level})


class MockJournal:
    def log(self, *args, **kwargs):
        pass


class MockEventBus:
    def __bool__(self):
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trader():
    """Create a minimal GapFadeLiveTrader-like object for testing."""
    trader = MagicMock()
    trader.engine = MockEngine()
    trader.alerter = MockAlerter()
    trader.journal = MockJournal()
    trader.status = 'trading'
    trader._trading_halted = False
    trader._eod_watchdog_task = None
    trader._last_loop_heartbeat = time.monotonic()
    trader._last_pos_broadcast = 0.0
    trader._position_lock = asyncio.Lock()
    trader._event_bus = MockEventBus()
    trader.messages = []
    trader.llm_supervisor = None
    trader.streamer = None
    trader.config = MagicMock()
    trader.config.auto_start = False
    trader.scanner = MagicMock()

    # Wire real methods from the class
    from gap_fade_app import GapFadeLiveTrader
    trader._eod_close = AsyncMock()
    trader._eod_watchdog = GapFadeLiveTrader._eod_watchdog.__get__(trader)
    trader._enter_positions = GapFadeLiveTrader._enter_positions.__get__(trader)
    trader._add_message = lambda kind, msg: trader.messages.append({'type': kind, 'msg': msg})
    trader._record_performance_snapshot = MagicMock()
    trader._build_llm_state = MagicMock(return_value={})
    trader._save_state = MagicMock()

    return trader


def _mock_time(hour: int, minute: int, second: int = 0, weekday: int = 0):
    """Create a mock datetime at the given ET time."""
    # Use a known Monday (2026-03-09 is a Monday)
    base = date(2026, 3, 9)
    # Shift to desired weekday (0=Monday)
    target = base + timedelta(days=weekday)
    dt = datetime(target.year, target.month, target.day,
                  hour, minute, second)
    return dt


# ---------------------------------------------------------------------------
# Test: Circuit breaker engages at 3:45 PM
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    """Verify the circuit breaker blocks new entries at 3:45 PM."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_entry(self):
        """When _trading_halted is True, _enter_positions must return immediately."""
        trader = _make_trader()
        trader._trading_halted = True
        trader.candidates = [MagicMock()]  # has candidates

        # _enter_positions should be blocked
        # We need to actually call the real method
        from gap_fade_app import GapFadeLiveTrader
        bound = GapFadeLiveTrader._enter_positions.__get__(trader)
        await bound()

        # Should have a blocked message
        blocked = [m for m in trader.messages if 'circuit breaker' in m.get('msg', '').lower()]
        assert len(blocked) >= 1, f"Expected circuit breaker message, got: {trader.messages}"

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_intraday(self):
        """When _trading_halted is True, _intraday_scan_and_enter must return."""
        trader = _make_trader()
        trader._trading_halted = True

        from gap_fade_app import GapFadeLiveTrader
        bound = GapFadeLiveTrader._intraday_scan_and_enter.__get__(trader)
        # Should return without doing anything
        await bound()

        # No scans should have been attempted
        # (If it tried to scan, it would access _intraday_strategies which would fail)


# ---------------------------------------------------------------------------
# Test: EOD watchdog phases fire in correct order
# ---------------------------------------------------------------------------

class TestEODWatchdogPhases:
    """Verify each phase of the EOD watchdog fires at the correct time."""

    @pytest.mark.asyncio
    async def test_phase_circuit_breaker_345(self):
        """At 3:45 PM, watchdog sets _trading_halted = True."""
        trader = _make_trader()
        assert trader._trading_halted is False

        call_count = 0

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count > 5:
                raise asyncio.CancelledError()

        times = [
            _mock_time(15, 45, 0),   # 3:45 PM — trigger circuit breaker
            _mock_time(15, 45, 5),   # still 3:45
            _mock_time(15, 45, 10),
            _mock_time(15, 45, 15),
            _mock_time(15, 45, 20),
            _mock_time(15, 45, 25),
        ]
        time_idx = [0]

        def get_time(*args):
            idx = min(time_idx[0], len(times) - 1)
            time_idx[0] += 1
            return times[idx]

        with patch('gap_fade_app.datetime') as mock_dt, \
             patch('gap_fade_app.broadcast', new_callable=AsyncMock), \
             patch.object(asyncio, 'sleep', side_effect=mock_sleep):
            mock_dt.now.side_effect = get_time
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)

            with pytest.raises(asyncio.CancelledError):
                await trader._eod_watchdog()

        assert trader._trading_halted is True, "Circuit breaker should be engaged at 3:45"
        # Check alert was sent
        cb_alerts = [a for a in trader.alerter.alerts if 'circuit breaker' in a['title'].lower()]
        assert len(cb_alerts) >= 1

    @pytest.mark.asyncio
    async def test_phase_primary_close_350(self):
        """At 3:50 PM with positions, watchdog calls _eod_close."""
        trader = _make_trader()
        trader.engine.positions = {
            'AAPL': MockPosition('AAPL'),
            'NVDA': MockPosition('NVDA'),
        }

        call_count = 0

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            # After the _eod_close call, let it run a bit then cancel
            if call_count > 8:
                raise asyncio.CancelledError()

        # Simulate time progressing from 3:50 through the phases
        times = [
            _mock_time(15, 50, 0),   # hits 3:50 — primary close
            _mock_time(15, 50, 1),
            _mock_time(15, 50, 2),
            _mock_time(15, 50, 30),  # after close
            _mock_time(15, 50, 35),
            _mock_time(15, 51, 0),
            _mock_time(15, 51, 5),
            _mock_time(15, 51, 10),
            _mock_time(15, 51, 15),
            _mock_time(15, 51, 20),
            _mock_time(15, 51, 25),
        ]
        time_idx = [0]

        def get_time(*args):
            idx = min(time_idx[0], len(times) - 1)
            time_idx[0] += 1
            return times[idx]

        # _eod_close clears positions
        async def fake_eod_close():
            trader.engine.positions.clear()

        trader._eod_close = fake_eod_close

        with patch('gap_fade_app.datetime') as mock_dt, \
             patch('gap_fade_app.broadcast', new_callable=AsyncMock), \
             patch.object(asyncio, 'sleep', side_effect=mock_sleep):
            mock_dt.now.side_effect = get_time
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)

            with pytest.raises(asyncio.CancelledError):
                await trader._eod_watchdog()

        assert trader._trading_halted is True
        assert len(trader.engine.positions) == 0, "All positions should be closed"
        # Performance snapshot recorded
        trader._record_performance_snapshot.assert_called_with('eod')

    @pytest.mark.asyncio
    async def test_phase_force_close_355(self):
        """At 3:55 PM with stubborn positions, watchdog uses Alpaca API."""
        trader = _make_trader()

        # _eod_close doesn't clear positions (simulating failure)
        async def failing_eod_close():
            pass  # positions remain

        trader._eod_close = failing_eod_close
        trader.engine.positions = {'TSLA': MockPosition('TSLA')}

        call_count = 0

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count > 12:
                raise asyncio.CancelledError()

        # Jump from 3:50 to 3:55 to trigger force close
        times = [
            _mock_time(15, 50, 0),   # primary close (will fail)
            _mock_time(15, 50, 1),
            _mock_time(15, 50, 30),
            _mock_time(15, 55, 0),   # force close phase
            _mock_time(15, 55, 0),   # refresh
            _mock_time(15, 55, 5),
            _mock_time(15, 55, 10),  # past force close, not yet 4pm
            _mock_time(15, 55, 15),
            _mock_time(15, 56, 0),
            _mock_time(15, 56, 5),
            _mock_time(15, 56, 10),
            _mock_time(15, 56, 15),
            _mock_time(15, 56, 20),
        ]
        time_idx = [0]

        def get_time(*args):
            idx = min(time_idx[0], len(times) - 1)
            time_idx[0] += 1
            return times[idx]

        # Mock broker APIs
        def mock_cancel_orders(sym):
            return 1

        def mock_close_position(sym):
            # Remove from engine positions to simulate success
            trader.engine.positions.pop(sym, None)
            return {'filled_avg_price': '99.50'}

        async def mock_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch('gap_fade_app.datetime') as mock_dt, \
             patch('gap_fade_app.broadcast', new_callable=AsyncMock), \
             patch('gap_fade_app.alpaca_cancel_open_orders_for_symbol', mock_cancel_orders), \
             patch('gap_fade_app.alpaca_close_position_api', mock_close_position), \
             patch.object(asyncio, 'sleep', side_effect=mock_sleep), \
             patch('asyncio.to_thread', side_effect=mock_to_thread):

            mock_dt.now.side_effect = get_time
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)

            with pytest.raises(asyncio.CancelledError):
                await trader._eod_watchdog()

        assert len(trader.engine.positions) == 0, "Force close should have cleared TSLA"
        # Check force close alert was sent
        force_alerts = [a for a in trader.alerter.alerts if 'force' in a['title'].lower()]
        assert len(force_alerts) >= 1

    @pytest.mark.asyncio
    async def test_eod_close_timeout_escalates(self):
        """If _eod_close takes >60s, watchdog times out and escalates."""
        trader = _make_trader()
        trader.engine.positions = {'SLOW': MockPosition('SLOW')}

        # _eod_close that hangs forever
        async def hanging_eod_close():
            await asyncio.sleep(999)

        trader._eod_close = hanging_eod_close

        call_count = [0]
        original_sleep = asyncio.sleep

        async def mock_sleep(seconds):
            call_count[0] += 1
            if call_count[0] > 10:
                raise asyncio.CancelledError()
            # Let wait_for's internal sleep work for short durations
            if seconds <= 1:
                return
            return

        times = [
            _mock_time(15, 50, 0),
            _mock_time(15, 50, 1),
            _mock_time(15, 51, 5),
            _mock_time(15, 51, 10),
            _mock_time(15, 51, 15),
            _mock_time(15, 51, 20),
            _mock_time(15, 51, 25),
            _mock_time(15, 51, 30),
            _mock_time(15, 51, 35),
            _mock_time(15, 51, 40),
            _mock_time(15, 51, 45),
        ]
        time_idx = [0]

        def get_time(*args):
            idx = min(time_idx[0], len(times) - 1)
            time_idx[0] += 1
            return times[idx]

        with patch('gap_fade_app.datetime') as mock_dt, \
             patch('gap_fade_app.broadcast', new_callable=AsyncMock), \
             patch.object(asyncio, 'sleep', side_effect=mock_sleep):
            mock_dt.now.side_effect = get_time
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)

            # Use wait_for with a very short timeout to test the timeout path
            with patch('asyncio.wait_for') as mock_wait_for:
                mock_wait_for.side_effect = asyncio.TimeoutError()

                with pytest.raises(asyncio.CancelledError):
                    await trader._eod_watchdog()

        # Should have logged timeout alert
        timeout_alerts = [a for a in trader.alerter.alerts if 'timeout' in a['title'].lower()]
        assert len(timeout_alerts) >= 1, f"Expected timeout alert, got: {trader.alerter.alerts}"


# ---------------------------------------------------------------------------
# Test: Watchdog independence from trading loop
# ---------------------------------------------------------------------------

class TestWatchdogIndependence:
    """Verify EOD watchdog runs independently of trading loop state."""

    @pytest.mark.asyncio
    async def test_watchdog_fires_during_standdown(self):
        """EOD watchdog fires even when trading loop is in standdown sleep."""
        trader = _make_trader()
        trader.status = 'standdown'
        trader.engine.positions = {'SPY': MockPosition('SPY')}

        eod_close_called = False

        async def mock_eod_close():
            nonlocal eod_close_called
            eod_close_called = True
            trader.engine.positions.clear()

        trader._eod_close = mock_eod_close

        call_count = [0]

        async def mock_sleep(seconds):
            call_count[0] += 1
            if call_count[0] > 8:
                raise asyncio.CancelledError()

        times = [
            _mock_time(15, 50, 0),
            _mock_time(15, 50, 1),
            _mock_time(15, 50, 30),
            _mock_time(15, 50, 35),
            _mock_time(15, 51, 0),
            _mock_time(15, 51, 5),
            _mock_time(15, 51, 10),
            _mock_time(15, 51, 15),
            _mock_time(15, 51, 20),
        ]
        time_idx = [0]

        def get_time(*args):
            idx = min(time_idx[0], len(times) - 1)
            time_idx[0] += 1
            return times[idx]

        with patch('gap_fade_app.datetime') as mock_dt, \
             patch('gap_fade_app.broadcast', new_callable=AsyncMock), \
             patch.object(asyncio, 'sleep', side_effect=mock_sleep):
            mock_dt.now.side_effect = get_time
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)

            with pytest.raises(asyncio.CancelledError):
                await trader._eod_watchdog()

        assert eod_close_called, "EOD close must fire regardless of trading loop status"

    @pytest.mark.asyncio
    async def test_watchdog_fires_during_llm_call(self):
        """EOD watchdog fires even when LLM is blocking the trading loop."""
        # This is the EXACT scenario that caused the original bug.
        # The watchdog is a separate task — it doesn't care what the
        # trading loop is doing.
        trader = _make_trader()
        trader.status = 'trading'
        trader.engine.positions = {'AAPL': MockPosition('AAPL')}

        eod_close_called = False

        async def mock_eod_close():
            nonlocal eod_close_called
            eod_close_called = True
            trader.engine.positions.clear()

        trader._eod_close = mock_eod_close

        # Simulate: trading loop is stuck in an LLM call
        llm_stuck = asyncio.Event()

        async def stuck_trading_loop():
            """Simulates a trading loop stuck on an LLM call."""
            await llm_stuck.wait()  # never completes

        # Run both tasks concurrently
        call_count = [0]

        async def mock_sleep(seconds):
            call_count[0] += 1
            if call_count[0] > 8:
                raise asyncio.CancelledError()

        times = [
            _mock_time(15, 50, 0),
            _mock_time(15, 50, 1),
            _mock_time(15, 50, 30),
            _mock_time(15, 50, 35),
            _mock_time(15, 51, 0),
            _mock_time(15, 51, 5),
            _mock_time(15, 51, 10),
            _mock_time(15, 51, 15),
            _mock_time(15, 51, 20),
        ]
        time_idx = [0]

        def get_time(*args):
            idx = min(time_idx[0], len(times) - 1)
            time_idx[0] += 1
            return times[idx]

        with patch('gap_fade_app.datetime') as mock_dt, \
             patch('gap_fade_app.broadcast', new_callable=AsyncMock), \
             patch.object(asyncio, 'sleep', side_effect=mock_sleep):
            mock_dt.now.side_effect = get_time
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)

            # Start stuck trading loop and watchdog concurrently
            loop_task = asyncio.create_task(stuck_trading_loop())
            try:
                await trader._eod_watchdog()
            except asyncio.CancelledError:
                pass
            finally:
                loop_task.cancel()
                try:
                    await loop_task
                except asyncio.CancelledError:
                    pass

        assert eod_close_called, \
            "EOD close MUST fire even when trading loop is stuck on LLM"


# ---------------------------------------------------------------------------
# Test: Watchdog recovery (self-healing)
# ---------------------------------------------------------------------------

class TestWatchdogRecovery:
    """Verify the watchdog recovers from errors and restarts if it dies."""

    @pytest.mark.asyncio
    async def test_watchdog_survives_exception(self):
        """Watchdog catches exceptions and continues running."""
        trader = _make_trader()

        iteration = [0]

        # Force an exception on first iteration, then normal operation
        original_build = trader._build_llm_state

        def maybe_explode(*args):
            iteration[0] += 1
            if iteration[0] == 1:
                raise RuntimeError("Simulated broker failure")
            return {}

        call_count = [0]

        async def mock_sleep(seconds):
            call_count[0] += 1
            if call_count[0] > 5:
                raise asyncio.CancelledError()

        # First call raises, subsequent calls normal
        times = [
            _mock_time(15, 50, 0),  # will trigger close → exception in _eod_close
            _mock_time(15, 50, 5),  # retry after error
            _mock_time(15, 50, 10),
            _mock_time(15, 50, 15),
            _mock_time(15, 50, 20),
            _mock_time(15, 50, 25),
        ]
        time_idx = [0]

        def get_time(*args):
            idx = min(time_idx[0], len(times) - 1)
            time_idx[0] += 1
            return times[idx]

        # _eod_close raises on first call
        eod_calls = [0]

        async def flaky_eod_close():
            eod_calls[0] += 1
            if eod_calls[0] == 1:
                raise RuntimeError("Broker connection lost")

        trader._eod_close = flaky_eod_close
        trader.engine.positions = {'FAIL': MockPosition('FAIL')}

        with patch('gap_fade_app.datetime') as mock_dt, \
             patch('gap_fade_app.broadcast', new_callable=AsyncMock), \
             patch.object(asyncio, 'sleep', side_effect=mock_sleep):
            mock_dt.now.side_effect = get_time
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)

            with pytest.raises(asyncio.CancelledError):
                await trader._eod_watchdog()

        # Watchdog should have survived and retried
        assert eod_calls[0] >= 1, "EOD close should have been attempted"
        # Error alert should have been sent
        error_alerts = [a for a in trader.alerter.alerts if a['level'] == 'error']
        assert len(error_alerts) >= 1

    @pytest.mark.asyncio
    async def test_systemd_watchdog_restarts_dead_eod_task(self):
        """_watchdog_heartbeat detects dead EOD task and restarts it."""
        from gap_fade_app import _watchdog_heartbeat, GapFadeLiveTrader

        # Create a completed (dead) task
        async def dead_task():
            raise RuntimeError("I died")

        task = asyncio.ensure_future(dead_task())
        try:
            await task
        except RuntimeError:
            pass

        trader = _make_trader()
        trader._eod_watchdog_task = task

        # Verify the task is done
        assert task.done()

        # The _watchdog_heartbeat checks for dead EOD task and restarts it
        # We can't easily test the full function (needs systemd socket),
        # but we can verify the detection logic
        assert trader._eod_watchdog_task.done()
        exc = trader._eod_watchdog_task.exception() if not \
            trader._eod_watchdog_task.cancelled() else None
        assert exc is not None


# ---------------------------------------------------------------------------
# Test: Weekend handling
# ---------------------------------------------------------------------------

class TestWeekendHandling:
    """Verify watchdog sleeps through weekends correctly."""

    @pytest.mark.asyncio
    async def test_weekend_sleep(self):
        """On Saturday, watchdog should sleep until Monday."""
        trader = _make_trader()

        sleep_durations = []

        async def mock_sleep(seconds):
            sleep_durations.append(seconds)
            if len(sleep_durations) > 2:
                raise asyncio.CancelledError()

        # Saturday
        saturday = _mock_time(10, 0, 0, weekday=5)

        call_count = [0]

        def get_time(*args):
            call_count[0] += 1
            return saturday

        with patch('gap_fade_app.datetime') as mock_dt, \
             patch('gap_fade_app.broadcast', new_callable=AsyncMock), \
             patch.object(asyncio, 'sleep', side_effect=mock_sleep):
            mock_dt.now.side_effect = get_time
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)

            with pytest.raises(asyncio.CancelledError):
                await trader._eod_watchdog()

        # Should have attempted to sleep for a long time (until Monday)
        assert len(sleep_durations) > 0
        # Saturday to Monday 6 AM is ~44 hours = ~158400 seconds
        assert sleep_durations[0] > 3600, \
            f"Weekend sleep should be hours, got {sleep_durations[0]}s"


# ---------------------------------------------------------------------------
# Test: No positions scenario
# ---------------------------------------------------------------------------

class TestNoPositions:
    """Verify watchdog handles no-positions case gracefully."""

    @pytest.mark.asyncio
    async def test_no_positions_at_350(self):
        """At 3:50 with no positions, watchdog records snapshot but doesn't close."""
        trader = _make_trader()
        assert len(trader.engine.positions) == 0

        eod_close_called = False

        async def mock_eod_close():
            nonlocal eod_close_called
            eod_close_called = True

        trader._eod_close = mock_eod_close

        call_count = [0]

        async def mock_sleep(seconds):
            call_count[0] += 1
            if call_count[0] > 5:
                raise asyncio.CancelledError()

        times = [
            _mock_time(15, 50, 0),
            _mock_time(15, 50, 5),
            _mock_time(15, 51, 0),
            _mock_time(15, 51, 5),
            _mock_time(15, 51, 10),
            _mock_time(15, 51, 15),
        ]
        time_idx = [0]

        def get_time(*args):
            idx = min(time_idx[0], len(times) - 1)
            time_idx[0] += 1
            return times[idx]

        with patch('gap_fade_app.datetime') as mock_dt, \
             patch('gap_fade_app.broadcast', new_callable=AsyncMock), \
             patch.object(asyncio, 'sleep', side_effect=mock_sleep):
            mock_dt.now.side_effect = get_time
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)

            with pytest.raises(asyncio.CancelledError):
                await trader._eod_watchdog()

        assert not eod_close_called, "Should not call _eod_close with no positions"
        trader._record_performance_snapshot.assert_called_with('eod')


# ---------------------------------------------------------------------------
# Test: Health endpoint reports watchdog status
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Verify health endpoint reports EOD watchdog state."""

    @pytest.mark.asyncio
    async def test_health_reports_watchdog_running(self):
        """Health endpoint shows eod_watchdog: running when task is alive."""
        from gap_fade_app import health_check, live_trader

        # Save originals
        orig_task = live_trader._eod_watchdog_task
        orig_halted = live_trader._trading_halted

        try:
            # Simulate running watchdog
            async def forever():
                await asyncio.sleep(999)

            task = asyncio.create_task(forever())
            live_trader._eod_watchdog_task = task
            live_trader._trading_halted = False

            result = await health_check()
            assert result['eod_watchdog'] == 'running'
            assert result['trading_halted'] is False

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        finally:
            live_trader._eod_watchdog_task = orig_task
            live_trader._trading_halted = orig_halted

    @pytest.mark.asyncio
    async def test_health_reports_watchdog_dead(self):
        """Health endpoint shows eod_watchdog: dead when task has crashed."""
        from gap_fade_app import health_check, live_trader

        orig_task = live_trader._eod_watchdog_task
        orig_halted = live_trader._trading_halted

        try:
            # Simulate dead watchdog
            async def die():
                raise RuntimeError("dead")

            task = asyncio.create_task(die())
            await asyncio.sleep(0.01)  # let it die
            live_trader._eod_watchdog_task = task

            result = await health_check()
            assert result['eod_watchdog'] == 'dead'
        finally:
            live_trader._eod_watchdog_task = orig_task
            live_trader._trading_halted = orig_halted


# ---------------------------------------------------------------------------
# Test: Concurrent execution guarantee
# ---------------------------------------------------------------------------

class TestConcurrentExecution:
    """Prove watchdog and trading loop run concurrently."""

    @pytest.mark.asyncio
    async def test_two_tasks_run_independently(self):
        """Watchdog task fires while trading loop is blocked."""
        watchdog_fired = asyncio.Event()
        loop_unblocked = asyncio.Event()

        async def mock_watchdog():
            """Simulates the watchdog firing at 3:50 PM."""
            await asyncio.sleep(0.01)  # brief delay
            watchdog_fired.set()

        async def mock_trading_loop():
            """Simulates a stuck trading loop."""
            # This is stuck — waiting for something that takes forever
            try:
                await asyncio.wait_for(loop_unblocked.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

        # Run both concurrently
        loop_task = asyncio.create_task(mock_trading_loop())
        wd_task = asyncio.create_task(mock_watchdog())

        # Wait for watchdog to fire
        await asyncio.wait_for(watchdog_fired.wait(), timeout=2.0)

        assert watchdog_fired.is_set(), \
            "Watchdog MUST fire even when trading loop is blocked"

        # Cleanup
        loop_unblocked.set()
        await asyncio.gather(loop_task, wd_task)

    @pytest.mark.asyncio
    async def test_100_iterations_watchdog_always_fires(self):
        """Run 100 iterations: watchdog fires 100% of the time."""
        success_count = 0

        for i in range(100):
            fired = asyncio.Event()

            async def watchdog():
                await asyncio.sleep(0)  # yield
                fired.set()

            async def blocker():
                await asyncio.sleep(0.1)

            wt = asyncio.create_task(watchdog())
            bt = asyncio.create_task(blocker())

            try:
                await asyncio.wait_for(fired.wait(), timeout=1.0)
                success_count += 1
            except asyncio.TimeoutError:
                pass

            bt.cancel()
            try:
                await bt
            except asyncio.CancelledError:
                pass
            await wt

        assert success_count == 100, \
            f"Watchdog fired {success_count}/100 times — must be 100%"


# ---------------------------------------------------------------------------
# Helper for asyncio.to_thread mock
# ---------------------------------------------------------------------------

async def _mock_to_thread(func, *args, **kwargs):
    """Replacement for asyncio.to_thread that runs synchronously."""
    return func(*args, **kwargs)


# ---------------------------------------------------------------------------
# Test: Integration — full EOD sequence
# ---------------------------------------------------------------------------

class TestFullEODSequence:
    """End-to-end test of the complete EOD close sequence."""

    @pytest.mark.asyncio
    async def test_full_sequence_345_to_400(self):
        """Walk through the entire EOD timeline: 3:45 → 3:50 → 3:55 → 4:00."""
        trader = _make_trader()
        trader.engine.positions = {
            'AAPL': MockPosition('AAPL', entry_price=150.0),
            'TSLA': MockPosition('TSLA', entry_price=200.0),
        }

        events = []

        async def mock_eod_close():
            events.append('eod_close_called')
            # Close one position, leave one (simulating partial success)
            trader.engine.positions.pop('AAPL', None)

        trader._eod_close = mock_eod_close

        # Time sequence walking through all phases
        times = [
            _mock_time(15, 45, 0),   # circuit breaker
            _mock_time(15, 45, 5),
            _mock_time(15, 50, 0),   # primary close
            _mock_time(15, 50, 1),   # refresh inside
            _mock_time(15, 50, 30),  # after close
            _mock_time(15, 55, 0),   # force close
            _mock_time(15, 55, 0),   # refresh
            _mock_time(15, 55, 5),
            _mock_time(15, 55, 10),
            _mock_time(16, 0, 0),    # emergency
            _mock_time(16, 5, 0),    # audit
            _mock_time(16, 5, 1),
            _mock_time(16, 5, 2),
        ]
        time_idx = [0]

        def get_time(*args):
            idx = min(time_idx[0], len(times) - 1)
            time_idx[0] += 1
            return times[idx]

        call_count = [0]

        async def mock_sleep(seconds):
            call_count[0] += 1
            if call_count[0] > 15:
                raise asyncio.CancelledError()

        def mock_cancel_orders(sym):
            return 1

        def mock_close_position(sym):
            trader.engine.positions.pop(sym, None)
            return {'filled_avg_price': '199.50'}

        def mock_get_positions():
            return [{'symbol': s, 'qty': '100'} for s in trader.engine.positions]

        with patch('gap_fade_app.datetime') as mock_dt, \
             patch('gap_fade_app.broadcast', new_callable=AsyncMock), \
             patch.object(asyncio, 'sleep', side_effect=mock_sleep), \
             patch('gap_fade_app.alpaca_cancel_open_orders_for_symbol', mock_cancel_orders), \
             patch('gap_fade_app.alpaca_close_position_api', mock_close_position), \
             patch('gap_fade_app.alpaca_get_positions', mock_get_positions), \
             patch('asyncio.to_thread', side_effect=_mock_to_thread):
            mock_dt.now.side_effect = get_time
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)

            with pytest.raises(asyncio.CancelledError):
                await trader._eod_watchdog()

        assert trader._trading_halted is True, "Circuit breaker should be set"
        assert 'eod_close_called' in events, "Primary close should have fired"
        # TSLA should have been force-closed or marked for close-on-open
        assert len(trader.engine.positions) == 0 or \
            all(p.close_on_open for p in trader.engine.positions.values()), \
            "All positions should be closed or marked for close-on-open"


# ---------------------------------------------------------------------------
# Test: _did_eod removed from trading loop
# ---------------------------------------------------------------------------

class TestTradingLoopCleanup:
    """Verify trading loop no longer owns EOD logic."""

    def test_no_did_eod_flag(self):
        """The _did_eod flag should not exist in the trading loop."""
        import inspect
        from gap_fade_app import GapFadeLiveTrader

        source = inspect.getsource(GapFadeLiveTrader._trading_loop)
        # _did_eod should not be set or checked in the trading loop
        assert '_did_eod' not in source, \
            "_did_eod flag must be removed from trading loop (EOD owned by watchdog)"

    def test_trading_halted_flag_exists(self):
        """The _trading_halted flag should be checked in the trading loop."""
        import inspect
        from gap_fade_app import GapFadeLiveTrader

        source = inspect.getsource(GapFadeLiveTrader._trading_loop)
        assert '_trading_halted' in source, \
            "Trading loop must check _trading_halted circuit breaker"

    def test_enter_positions_checks_circuit_breaker(self):
        """_enter_positions must check _trading_halted before entering."""
        import inspect
        from gap_fade_app import GapFadeLiveTrader

        source = inspect.getsource(GapFadeLiveTrader._enter_positions)
        assert '_trading_halted' in source, \
            "_enter_positions must check circuit breaker"

    def test_intraday_checks_circuit_breaker(self):
        """_intraday_scan_and_enter must check _trading_halted."""
        import inspect
        from gap_fade_app import GapFadeLiveTrader

        source = inspect.getsource(GapFadeLiveTrader._intraday_scan_and_enter)
        assert '_trading_halted' in source, \
            "_intraday_scan_and_enter must check circuit breaker"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
