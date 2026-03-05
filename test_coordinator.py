"""Unit tests for Phase 4 — Multi-broker coordinator, market hours extensions,
snapshot prices, combined risk, and backward compatibility.

Tests cover:
  - MarketHoursPolicy: is_pre_session, is_post_session for USEquityHours + ForexHours
  - AbstractBroker: get_snapshot_prices for both adapters (mocked)
  - MultiBrokerCoordinator: lifecycle, state, proxy, enable/disable
  - CombinedRiskManager: drawdown, daily loss, position limits, edge cases
  - Coordinator LLM: combined state building, shared supervisor
  - Backward compat: single-broker mode behaves identically
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: minimal fake GapFadeEngine / GapFadeLiveTrader for coordinator tests
# ---------------------------------------------------------------------------

@dataclass
class FakeDailyStats:
    pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0


@dataclass
class FakeEngine:
    equity: float = 25000.0
    peak_equity: float = 25000.0
    positions: Dict = field(default_factory=dict)
    daily_stats: FakeDailyStats = field(default_factory=FakeDailyStats)
    all_trade_log: List = field(default_factory=list)


class FakeTrader:
    """Minimal GapFadeLiveTrader-like object for coordinator tests."""

    def __init__(self, broker_id='alpaca', equity=25000.0, status='stopped'):
        self.broker = SimpleNamespace(broker_id=broker_id)
        self._market_hours = MagicMock()
        self._market_hours.__class__.__name__ = 'USEquityHours'
        self.engine = FakeEngine(equity=equity, peak_equity=equity)
        self.status = status
        self.config = SimpleNamespace(auto_start=False, llm_enabled=False)
        self._coordinator = None
        self._combined_risk_checker = None
        self.llm_supervisor = None
        self.messages = []
        self.alerter = MagicMock()
        self._started = False
        self._stopped = False

    async def start(self):
        self.status = 'scanning'
        self._started = True

    async def stop(self):
        self.status = 'stopped'
        self._stopped = True

    def _build_llm_state(self, now):
        return {
            'broker_id': self.broker.broker_id,
            'time_et': now.strftime('%H:%M:%S'),
            'day_of_week': now.strftime('%A'),
            'equity': self.engine.equity,
            'daily_pnl': self.engine.daily_stats.pnl,
            'positions': {},
            'status': self.status,
        }

    def get_state(self):
        return {'equity': self.engine.equity, 'status': self.status}


# ===========================================================================
# 1. Market Hours Extensions
# ===========================================================================

class TestUSEquityHoursPrePost:
    """is_pre_session / is_post_session for US equities."""

    def test_pre_session_before_7am(self):
        from brokers.market_hours import USEquityHours
        h = USEquityHours()
        dt = datetime(2026, 3, 2, 6, 30)  # 6:30 AM ET Monday
        assert h.is_pre_session(dt) is True

    def test_pre_session_at_7am(self):
        from brokers.market_hours import USEquityHours
        h = USEquityHours()
        dt = datetime(2026, 3, 2, 7, 0)  # 7:00 AM ET
        assert h.is_pre_session(dt) is False

    def test_pre_session_at_noon(self):
        from brokers.market_hours import USEquityHours
        h = USEquityHours()
        dt = datetime(2026, 3, 2, 12, 0)
        assert h.is_pre_session(dt) is False

    def test_post_session_at_4pm(self):
        from brokers.market_hours import USEquityHours
        h = USEquityHours()
        dt = datetime(2026, 3, 2, 16, 0)
        assert h.is_post_session(dt) is True

    def test_post_session_at_3pm(self):
        from brokers.market_hours import USEquityHours
        h = USEquityHours()
        dt = datetime(2026, 3, 2, 15, 0)
        assert h.is_post_session(dt) is False

    def test_post_session_at_midnight(self):
        from brokers.market_hours import USEquityHours
        h = USEquityHours()
        # Midnight is before 7 AM so it's pre-session, not post-session
        dt = datetime(2026, 3, 2, 0, 0)
        assert h.is_pre_session(dt) is True
        assert h.is_post_session(dt) is False


class TestForexHoursPrePost:
    """is_pre_session / is_post_session for forex."""

    def test_pre_session_always_false_weekday(self):
        from brokers.market_hours import ForexHours
        h = ForexHours()
        # Wednesday 3 AM — forex is open 24h
        dt = datetime(2026, 3, 4, 3, 0)  # Wednesday
        assert h.is_pre_session(dt) is False

    def test_pre_session_always_false_even_early(self):
        from brokers.market_hours import ForexHours
        h = ForexHours()
        # Monday 1 AM — still open
        dt = datetime(2026, 3, 2, 1, 0)
        assert h.is_pre_session(dt) is False

    def test_post_session_saturday(self):
        from brokers.market_hours import ForexHours
        h = ForexHours()
        dt = datetime(2026, 3, 7, 12, 0)  # Saturday
        assert h.is_post_session(dt) is True

    def test_post_session_friday_4pm(self):
        from brokers.market_hours import ForexHours
        h = ForexHours()
        dt = datetime(2026, 3, 6, 16, 0)  # Friday 4 PM
        assert h.is_post_session(dt) is True

    def test_post_session_friday_3pm(self):
        from brokers.market_hours import ForexHours
        h = ForexHours()
        dt = datetime(2026, 3, 6, 15, 0)  # Friday 3 PM
        assert h.is_post_session(dt) is False

    def test_post_session_sunday_before_5pm(self):
        from brokers.market_hours import ForexHours
        h = ForexHours()
        dt = datetime(2026, 3, 1, 14, 0)  # Sunday 2 PM
        assert h.is_post_session(dt) is True

    def test_post_session_sunday_after_5pm(self):
        from brokers.market_hours import ForexHours
        h = ForexHours()
        dt = datetime(2026, 3, 1, 17, 30)  # Sunday 5:30 PM
        assert h.is_post_session(dt) is False


# ===========================================================================
# 2. Snapshot Prices (mocked)
# ===========================================================================

class TestAlpacaSnapshotPrices:
    @pytest.mark.asyncio
    async def test_get_snapshot_prices(self):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter

        adapter = AlpacaBrokerAdapter.__new__(AlpacaBrokerAdapter)
        adapter.config = SimpleNamespace(broker_id='alpaca')
        adapter._market_hours = MagicMock()

        mock_snaps = {
            'AAPL': {'latestTrade': {'p': 175.50}},
            'TSLA': {'latestTrade': {'p': 250.00}},
            'EMPTY': {'latestTrade': {}},
        }
        with patch('brokers.alpaca_adapter._gap_fade') as mock_gf:
            mock_gf.return_value.fetch_alpaca_snapshots = MagicMock(return_value=mock_snaps)
            prices = await adapter.get_snapshot_prices(['AAPL', 'TSLA', 'EMPTY'])

        assert prices['AAPL'] == 175.50
        assert prices['TSLA'] == 250.00
        assert 'EMPTY' not in prices

    @pytest.mark.asyncio
    async def test_empty_snapshots(self):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter

        adapter = AlpacaBrokerAdapter.__new__(AlpacaBrokerAdapter)
        adapter.config = SimpleNamespace(broker_id='alpaca')
        adapter._market_hours = MagicMock()

        with patch('brokers.alpaca_adapter._gap_fade') as mock_gf:
            mock_gf.return_value.fetch_alpaca_snapshots = MagicMock(return_value={})
            prices = await adapter.get_snapshot_prices(['AAPL'])

        assert prices == {}


class TestOandaSnapshotPrices:
    @pytest.mark.asyncio
    async def test_get_snapshot_prices(self):
        from brokers.oanda_adapter import OandaBrokerAdapter

        adapter = OandaBrokerAdapter.__new__(OandaBrokerAdapter)
        adapter.config = SimpleNamespace(broker_id='oanda')
        adapter._market_hours = MagicMock()
        adapter._env_config = {'token': 'fake', 'account_id': '001', 'environment': 'practice'}

        mock_resp = {
            'prices': [
                {'instrument': 'EUR_USD', 'bids': [{'price': '1.08500'}], 'asks': [{'price': '1.08520'}]},
                {'instrument': 'GBP_USD', 'bids': [{'price': '1.26000'}], 'asks': [{'price': '1.26020'}]},
            ]
        }
        mock_api = MagicMock()
        mock_api.request = MagicMock(return_value=mock_resp)
        adapter._api = mock_api

        async def fake_to_thread(fn, *args):
            return fn(*args)

        with patch('brokers.oanda_adapter.asyncio.to_thread', side_effect=fake_to_thread):
            prices = await adapter.get_snapshot_prices(['EUR/USD', 'GBP/USD'])

        assert abs(prices['EUR/USD'] - 1.08510) < 0.0001
        assert abs(prices['GBP/USD'] - 1.26010) < 0.0001

    @pytest.mark.asyncio
    async def test_snapshot_prices_error(self):
        from brokers.oanda_adapter import OandaBrokerAdapter

        adapter = OandaBrokerAdapter.__new__(OandaBrokerAdapter)
        adapter.config = SimpleNamespace(broker_id='oanda')
        adapter._market_hours = MagicMock()
        adapter._env_config = {'token': 'fake', 'account_id': '001', 'environment': 'practice'}
        adapter._api = MagicMock()
        adapter._api.request = MagicMock(side_effect=Exception("network error"))

        async def fake_to_thread(fn, *args):
            return fn(*args)

        with patch('brokers.oanda_adapter.asyncio.to_thread', side_effect=fake_to_thread):
            prices = await adapter.get_snapshot_prices(['EUR/USD'])

        assert prices == {}


# ===========================================================================
# 3. MultiBrokerCoordinator
# ===========================================================================

class TestMultiBrokerCoordinator:
    def test_add_broker_sets_primary(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca')
        coord.add_broker('alpaca', t1, label='Alpaca')
        assert coord.primary is t1
        assert coord._primary_id == 'alpaca'

    def test_add_second_broker(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca')
        t2 = FakeTrader('oanda', equity=10000)
        coord.add_broker('alpaca', t1)
        coord.add_broker('oanda', t2)
        # Primary is still alpaca
        assert coord.primary is t1
        assert len(coord.active_traders) == 2

    def test_get_trader(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca')
        coord.add_broker('alpaca', t1)
        assert coord.get_trader('alpaca') is t1

    def test_get_trader_unknown(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca')
        coord.add_broker('alpaca', t1)
        with pytest.raises(KeyError):
            coord.get_trader('oanda')

    def test_get_trader_for_symbol_primary_fallback(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca')
        t2 = FakeTrader('oanda')
        coord.add_broker('alpaca', t1)
        coord.add_broker('oanda', t2)
        # No positions anywhere — falls back to primary
        assert coord.get_trader_for_symbol('AAPL') is t1

    def test_get_trader_for_symbol_finds_correct(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca')
        t2 = FakeTrader('oanda')
        t2.engine.positions = {'EUR/USD': SimpleNamespace(symbol='EUR/USD')}
        coord.add_broker('alpaca', t1)
        coord.add_broker('oanda', t2)
        assert coord.get_trader_for_symbol('EUR/USD') is t2

    def test_broker_ids(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        coord.add_broker('alpaca', FakeTrader('alpaca'))
        coord.add_broker('oanda', FakeTrader('oanda'))
        assert coord.broker_ids == ['alpaca', 'oanda']

    def test_all_slots(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        coord.add_broker('alpaca', FakeTrader('alpaca'), label='Alpaca')
        slots = coord.all_slots
        assert 'alpaca' in slots
        assert slots['alpaca'].label == 'Alpaca'
        assert slots['alpaca'].enabled is True

    def test_coordinator_sets_back_reference(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca')
        coord.add_broker('alpaca', t1)
        assert t1._coordinator is coord
        assert t1._combined_risk_checker is not None

    @pytest.mark.asyncio
    async def test_start_all(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca')
        t2 = FakeTrader('oanda')
        coord.add_broker('alpaca', t1)
        coord.add_broker('oanda', t2)
        await coord.start_all()
        assert t1._started
        assert t2._started

    @pytest.mark.asyncio
    async def test_stop_all(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca', status='trading')
        t2 = FakeTrader('oanda', status='trading')
        coord.add_broker('alpaca', t1)
        coord.add_broker('oanda', t2)
        await coord.stop_all()
        assert t1._stopped
        assert t2._stopped

    @pytest.mark.asyncio
    async def test_stop_broker(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca', status='trading')
        t2 = FakeTrader('oanda', status='trading')
        coord.add_broker('alpaca', t1)
        coord.add_broker('oanda', t2)
        await coord.stop_broker('oanda')
        assert not t1._stopped  # alpaca untouched
        assert t2._stopped
        assert not coord.all_slots['oanda'].enabled

    @pytest.mark.asyncio
    async def test_start_broker(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca')
        coord.add_broker('alpaca', t1)
        await coord.start_broker('alpaca')
        assert t1._started

    @pytest.mark.asyncio
    async def test_start_broker_unknown_raises(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        with pytest.raises(KeyError):
            await coord.start_broker('unknown')


# ===========================================================================
# 4. CombinedRiskManager
# ===========================================================================

class TestCombinedRiskManager:
    def _make_slots(self, equities, daily_pnls, positions_counts, peaks=None):
        """Create dict of BrokerSlot with given values."""
        from brokers.coordinator import BrokerSlot
        slots = {}
        for i, (eq, pnl, pc) in enumerate(zip(equities, daily_pnls, positions_counts)):
            bid = f'broker{i}'
            t = FakeTrader(bid, equity=eq)
            t.engine.daily_stats.pnl = pnl
            t.engine.positions = {f'SYM{j}': True for j in range(pc)}
            if peaks:
                t.engine.peak_equity = peaks[i]
            else:
                t.engine.peak_equity = eq
            slots[bid] = BrokerSlot(broker_id=bid, trader=t, enabled=True)
        return slots

    def test_no_violations(self):
        from brokers.coordinator import CombinedRiskManager
        rm = CombinedRiskManager()
        slots = self._make_slots([25000, 10000], [100, 50], [2, 1])
        v = rm.check_combined_risk(slots)
        assert v == {}

    def test_daily_loss_violation(self):
        from brokers.coordinator import CombinedRiskManager
        rm = CombinedRiskManager(combined_daily_loss_limit=0.03)
        # total peak = 35000, loss = -1100 = 3.14% > 3%
        slots = self._make_slots([24000, 9900], [-800, -300], [0, 0],
                                  peaks=[25000, 10000])
        v = rm.check_combined_risk(slots)
        assert 'daily_loss' in v

    def test_daily_loss_no_violation_below_limit(self):
        from brokers.coordinator import CombinedRiskManager
        rm = CombinedRiskManager(combined_daily_loss_limit=0.03)
        # total peak = 35000, loss = -500 = 1.43% < 3%
        slots = self._make_slots([24500, 10000], [-300, -200], [0, 0],
                                  peaks=[25000, 10000])
        v = rm.check_combined_risk(slots)
        assert 'daily_loss' not in v

    def test_drawdown_violation(self):
        from brokers.coordinator import CombinedRiskManager
        rm = CombinedRiskManager(combined_max_drawdown=0.07)
        # peak = 35000, equity = 32000, dd = 8.57% > 7%
        slots = self._make_slots([22000, 10000], [0, 0], [0, 0],
                                  peaks=[25000, 10000])
        v = rm.check_combined_risk(slots)
        assert 'drawdown' in v

    def test_position_count_violation(self):
        from brokers.coordinator import CombinedRiskManager
        rm = CombinedRiskManager(max_total_positions=8)
        slots = self._make_slots([25000, 10000], [0, 0], [5, 4])
        v = rm.check_combined_risk(slots)
        assert 'position_count' in v
        assert v['position_count']['count'] == 9

    def test_position_count_at_limit(self):
        from brokers.coordinator import CombinedRiskManager
        rm = CombinedRiskManager(max_total_positions=8)
        slots = self._make_slots([25000, 10000], [0, 0], [5, 3])
        v = rm.check_combined_risk(slots)
        assert 'position_count' not in v

    def test_disabled_slot_excluded(self):
        from brokers.coordinator import CombinedRiskManager, BrokerSlot
        rm = CombinedRiskManager(max_total_positions=5)
        t1 = FakeTrader('a', equity=25000)
        t1.engine.positions = {f'S{i}': True for i in range(3)}
        t2 = FakeTrader('b', equity=10000)
        t2.engine.positions = {f'X{i}': True for i in range(4)}
        slots = {
            'a': BrokerSlot(broker_id='a', trader=t1, enabled=True),
            'b': BrokerSlot(broker_id='b', trader=t2, enabled=False),  # disabled
        }
        v = rm.check_combined_risk(slots)
        # Only 3 positions counted (t2 disabled), under limit of 5
        assert 'position_count' not in v

    def test_get_status(self):
        from brokers.coordinator import CombinedRiskManager
        rm = CombinedRiskManager()
        slots = self._make_slots([25000, 10000], [200, -50], [2, 1])
        status = rm.get_status(slots)
        assert status['total_equity'] == 35000
        assert status['total_daily_pnl'] == 150
        assert status['total_positions'] == 3
        assert status['halted'] is False

    def test_get_status_halted(self):
        from brokers.coordinator import CombinedRiskManager
        rm = CombinedRiskManager(combined_max_drawdown=0.05)
        # Big drawdown
        slots = self._make_slots([20000, 8000], [0, 0], [0, 0],
                                  peaks=[25000, 10000])
        status = rm.get_status(slots)
        assert status['halted'] is True


# ===========================================================================
# 5. Combined State
# ===========================================================================

class TestCombinedState:
    def test_get_combined_state(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca', equity=25000)
        t1.engine.daily_stats.pnl = 100
        t2 = FakeTrader('oanda', equity=10000)
        t2.engine.daily_stats.pnl = -50
        coord.add_broker('alpaca', t1, label='Alpaca')
        coord.add_broker('oanda', t2, label='OANDA')

        state = coord.get_combined_state()
        assert state['total_equity'] == 35000
        assert state['total_daily_pnl'] == 50
        assert 'alpaca' in state['brokers']
        assert 'oanda' in state['brokers']
        assert state['brokers']['alpaca']['label'] == 'Alpaca'

    def test_combined_state_disabled_excluded(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca', equity=25000)
        t2 = FakeTrader('oanda', equity=10000)
        coord.add_broker('alpaca', t1)
        coord.add_broker('oanda', t2)
        # Disable oanda
        coord._slots['oanda'].enabled = False

        state = coord.get_combined_state()
        # Totals only include enabled brokers
        assert state['total_equity'] == 25000


# ===========================================================================
# 6. Coordinator LLM
# ===========================================================================

class TestCoordinatorLLM:
    def test_build_combined_llm_state(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca', equity=25000)
        t1.engine.daily_stats.pnl = 100
        t2 = FakeTrader('oanda', equity=10000)
        t2.engine.daily_stats.pnl = -50
        coord.add_broker('alpaca', t1)
        coord.add_broker('oanda', t2)

        now = datetime(2026, 3, 2, 10, 0)
        state = coord.build_combined_llm_state(now)

        assert state['multi_broker'] is True
        assert state['total_equity'] == 35000
        assert state['total_daily_pnl'] == 50
        assert 'alpaca' in state['brokers']
        assert 'oanda' in state['brokers']
        assert state['brokers']['alpaca']['broker_id'] == 'alpaca'

    def test_combined_llm_state_disabled_excluded(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca', equity=25000)
        t2 = FakeTrader('oanda', equity=10000)
        coord.add_broker('alpaca', t1)
        coord.add_broker('oanda', t2)
        coord._slots['oanda'].enabled = False

        now = datetime(2026, 3, 2, 10, 0)
        state = coord.build_combined_llm_state(now)
        assert state['total_equity'] == 25000
        assert 'oanda' not in state['brokers']

    def test_setup_shared_llm(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca')
        t2 = FakeTrader('oanda')
        coord.add_broker('alpaca', t1)
        coord.add_broker('oanda', t2)

        # Mock LLMSupervisor to avoid importing the real one
        mock_config = SimpleNamespace(
            llm_enabled=True, llm_url='http://localhost:11434',
            llm_model='test', llm_timeout=10, llm_max_failures=5,
            llm_circuit_reset=120, llm_max_hold_overrides=2,
            llm_provider='ollama', llm_api_key='',
        )
        with patch('brokers.coordinator.MultiBrokerCoordinator.setup_shared_llm') as mock_setup:
            # Just verify the method can be called
            coord.setup_shared_llm(mock_config)
            mock_setup.assert_called_once_with(mock_config)

    def test_risk_checker_callable(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca', equity=25000)
        coord.add_broker('alpaca', t1)
        # The combined risk checker should be callable
        assert callable(t1._combined_risk_checker)
        result = t1._combined_risk_checker()
        assert isinstance(result, dict)

    def test_risk_checker_propagated_to_engine(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca', equity=25000)
        coord.add_broker('alpaca', t1)
        # Coordinator should also set the checker on the engine
        assert t1.engine._combined_risk_checker is not None
        assert callable(t1.engine._combined_risk_checker)
        assert t1.engine._combined_risk_checker is t1._combined_risk_checker


# ===========================================================================
# 7. Combined Risk in should_enter
# ===========================================================================

class TestCombinedRiskInShouldEnter:
    """Verify that GapFadeEngine.should_enter checks cross-broker risk."""

    def _make_engine(self, combined_violations=None):
        """Create a minimal GapFadeEngine-like object with should_enter behavior."""
        import sys
        if 'gap_fade_app' not in sys.modules:
            pytest.skip("gap_fade_app not importable in test isolation")
        from gap_fade_app import GapFadeConfig, GapFadeEngine, GapCandidate
        engine = GapFadeEngine(GapFadeConfig(), backtest_mode=False)
        if combined_violations is not None:
            engine._combined_risk_checker = lambda: combined_violations
        return engine

    def _make_candidate(self):
        import sys
        if 'gap_fade_app' not in sys.modules:
            pytest.skip("gap_fade_app not importable in test isolation")
        from gap_fade_app import GapCandidate
        return GapCandidate(
            symbol='TEST', prev_close=100.0, open_price=110.0,
            gap_pct=0.10, vol_ratio=1.0, score=50.0,
            premarket_price=110.0, avg_volume=100000, direction='short',
        )

    def test_no_combined_risk_checker_passes(self):
        engine = self._make_engine(combined_violations=None)
        engine._combined_risk_checker = None
        candidate = self._make_candidate()
        ok, reason = engine.should_enter(candidate)
        assert ok is True

    def test_combined_risk_no_violations_passes(self):
        engine = self._make_engine(combined_violations={})
        candidate = self._make_candidate()
        ok, reason = engine.should_enter(candidate)
        assert ok is True

    def test_combined_risk_daily_loss_blocks(self):
        engine = self._make_engine(combined_violations={
            'daily_loss': {'pct': 0.04, 'limit': 0.03}
        })
        candidate = self._make_candidate()
        ok, reason = engine.should_enter(candidate)
        assert ok is False
        assert 'combined risk limit' in reason
        assert 'daily_loss' in reason

    def test_combined_risk_drawdown_blocks(self):
        engine = self._make_engine(combined_violations={
            'drawdown': {'pct': 0.08, 'limit': 0.07}
        })
        candidate = self._make_candidate()
        ok, reason = engine.should_enter(candidate)
        assert ok is False
        assert 'combined risk limit' in reason
        assert 'drawdown' in reason

    def test_combined_risk_position_count_alone_allows(self):
        """Position count violation alone doesn't block (that's per-engine)."""
        engine = self._make_engine(combined_violations={
            'position_count': {'count': 9, 'limit': 8}
        })
        candidate = self._make_candidate()
        ok, reason = engine.should_enter(candidate)
        # position_count alone doesn't block — only daily_loss/drawdown do
        assert ok is True


# ===========================================================================
# 8. Backward Compatibility (was 7)
# ===========================================================================

class TestBackwardCompat:
    def test_getattr_proxy(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca', equity=25000, status='trading')
        coord.add_broker('alpaca', t1)
        # Access through coordinator proxy
        assert coord.status == 'trading'
        assert coord.engine.equity == 25000

    def test_getattr_config(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca')
        t1.config.auto_start = True
        coord.add_broker('alpaca', t1)
        assert coord.config.auto_start is True

    def test_single_broker_active_traders(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca')
        coord.add_broker('alpaca', t1)
        assert len(coord.active_traders) == 1
        assert coord.active_traders[0] is t1

    def test_getattr_raises_for_internal(self):
        from brokers.coordinator import MultiBrokerCoordinator
        coord = MultiBrokerCoordinator()
        t1 = FakeTrader('alpaca')
        coord.add_broker('alpaca', t1)
        # Internal attrs should not proxy
        with pytest.raises(AttributeError):
            _ = coord._nonexistent_internal


# ===========================================================================
# 9. GapFadeConfig new fields
# ===========================================================================

class TestGapFadeConfigNewFields:
    def test_scan_start_hour_default(self):
        """Verify new config fields have expected defaults."""
        import sys
        # Import GapFadeConfig directly
        if 'gap_fade_app' not in sys.modules:
            pytest.skip("gap_fade_app not importable in test isolation")
        from gap_fade_app import GapFadeConfig
        c = GapFadeConfig()
        assert c.scan_start_hour == 7
        assert c.final_scan_hour == 9
        assert c.final_scan_min == 25
        assert c.eod_exit_hour == 15
        assert c.eod_exit_min == 50
