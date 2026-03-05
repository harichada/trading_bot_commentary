"""Unit tests for brokers package — Phase 1.

Tests cover:
  - Data classes (OrderResult, BrokerConfig)
  - Market hours policy (USEquityHours)
  - AlpacaBrokerAdapter delegation (mocked underlying functions)
  - AlpacaTickStreamerAdapter delegation
"""

import asyncio
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


class TestOrderResult:
    def test_defaults(self):
        from brokers.base import OrderResult
        r = OrderResult()
        assert r.order_id == ''
        assert r.status == ''
        assert r.filled_qty == 0
        assert r.filled_avg_price == 0.0
        assert not r.is_filled
        assert not r.is_error

    def test_filled(self):
        from brokers.base import OrderResult
        r = OrderResult(order_id='abc', status='filled', filled_qty=100,
                        filled_avg_price=25.50, symbol='AAPL', side='sell')
        assert r.is_filled
        assert not r.is_error
        assert r.filled_qty == 100

    def test_error_statuses(self):
        from brokers.base import OrderResult
        for status in ('error', 'rejected', 'cancelled', 'timeout'):
            r = OrderResult(status=status, error='something went wrong')
            assert r.is_error
            assert not r.is_filled

    def test_float_qty(self):
        from brokers.base import OrderResult
        r = OrderResult(filled_qty=10000.5)
        assert r.filled_qty == 10000.5


class TestBrokerConfig:
    def test_defaults(self):
        from brokers.base import BrokerConfig
        c = BrokerConfig()
        assert c.broker_id == ''
        assert c.environment == 'paper'
        assert c.extra == {}

    def test_custom(self):
        from brokers.base import BrokerConfig
        c = BrokerConfig(
            broker_id='oanda', api_key='k', api_secret='s',
            base_url='https://api-fxpractice.oanda.com',
            account_id='101-001-123',
            environment='paper',
            extra={'stream_url': 'https://stream-fxpractice.oanda.com'},
        )
        assert c.broker_id == 'oanda'
        assert c.account_id == '101-001-123'
        assert 'stream_url' in c.extra


# ---------------------------------------------------------------------------
# Market hours tests
# ---------------------------------------------------------------------------


class TestUSEquityHours:
    def setup_method(self):
        from brokers.market_hours import USEquityHours
        self.mh = USEquityHours()

    # -- is_market_day -------------------------------------------------------

    def test_weekday_is_market_day(self):
        # Wednesday 2026-03-04
        now = datetime(2026, 3, 4, 10, 0)
        is_day, _ = self.mh.is_market_day(now)
        assert is_day

    def test_saturday_not_market_day(self):
        now = datetime(2026, 3, 7, 10, 0)  # Saturday
        is_day, next_open = self.mh.is_market_day(now)
        assert not is_day
        assert next_open.weekday() == 0  # Monday
        assert next_open.hour == 7

    def test_sunday_not_market_day(self):
        now = datetime(2026, 3, 8, 10, 0)  # Sunday
        is_day, next_open = self.mh.is_market_day(now)
        assert not is_day
        assert next_open.weekday() == 0

    def test_christmas_observed(self):
        # Dec 25 2026 is a Friday → observed on Friday
        now = datetime(2026, 12, 25, 10, 0)
        is_day, next_open = self.mh.is_market_day(now)
        assert not is_day
        assert next_open.date() == date(2026, 12, 28)  # Monday

    def test_new_years_2027_observed(self):
        # Jan 1 2027 is a Friday → observed on Friday itself
        now = datetime(2027, 1, 1, 10, 0)
        is_day, next_open = self.mh.is_market_day(now)
        assert not is_day

    def test_mlk_day_2026(self):
        # 3rd Monday of January 2026 = Jan 19
        now = datetime(2026, 1, 19, 10, 0)
        is_day, _ = self.mh.is_market_day(now)
        assert not is_day

    def test_thanksgiving_2026(self):
        # 4th Thursday of November 2026 = Nov 26
        now = datetime(2026, 11, 26, 10, 0)
        is_day, _ = self.mh.is_market_day(now)
        assert not is_day

    def test_good_friday_2026(self):
        # Easter 2026 is April 5 → Good Friday is April 3
        now = datetime(2026, 4, 3, 10, 0)
        is_day, _ = self.mh.is_market_day(now)
        assert not is_day

    # -- is_in_session -------------------------------------------------------

    def test_in_session_at_open(self):
        now = datetime(2026, 3, 4, 9, 30)
        assert self.mh.is_in_session(now)

    def test_not_in_session_before_open(self):
        now = datetime(2026, 3, 4, 9, 29)
        assert not self.mh.is_in_session(now)

    def test_not_in_session_at_close(self):
        now = datetime(2026, 3, 4, 16, 0)
        assert not self.mh.is_in_session(now)

    def test_in_session_just_before_close(self):
        now = datetime(2026, 3, 4, 15, 59)
        assert self.mh.is_in_session(now)

    def test_not_in_session_weekend(self):
        now = datetime(2026, 3, 7, 12, 0)  # Saturday
        assert not self.mh.is_in_session(now)

    # -- is_entry_allowed ----------------------------------------------------

    def test_entry_allowed_normal(self):
        now = datetime(2026, 3, 4, 10, 0)
        ok, reason = self.mh.is_entry_allowed(now, 11, 30)
        assert ok
        assert reason == ''

    def test_entry_blocked_before_open(self):
        now = datetime(2026, 3, 4, 9, 25)
        ok, reason = self.mh.is_entry_allowed(now, 11, 30)
        assert not ok
        assert 'not open yet' in reason

    def test_entry_blocked_after_cutoff(self):
        now = datetime(2026, 3, 4, 11, 30)
        ok, reason = self.mh.is_entry_allowed(now, 11, 30)
        assert not ok
        assert 'cutoff' in reason

    # -- is_eod_close --------------------------------------------------------

    def test_eod_before(self):
        now = datetime(2026, 3, 4, 15, 49)
        assert not self.mh.is_eod_close(now, 15, 50)

    def test_eod_at(self):
        now = datetime(2026, 3, 4, 15, 50)
        assert self.mh.is_eod_close(now, 15, 50)

    def test_eod_after(self):
        now = datetime(2026, 3, 4, 16, 0)
        assert self.mh.is_eod_close(now, 15, 50)


# ---------------------------------------------------------------------------
# Calendar helper tests
# ---------------------------------------------------------------------------


class TestCalendarHelpers:
    def test_nth_weekday(self):
        from brokers.market_hours import _nth_weekday
        # 3rd Monday of January 2026
        assert _nth_weekday(2026, 1, 0, 3) == date(2026, 1, 19)
        # 1st Monday of September 2026
        assert _nth_weekday(2026, 9, 0, 1) == date(2026, 9, 7)
        # 4th Thursday of November 2026
        assert _nth_weekday(2026, 11, 3, 4) == date(2026, 11, 26)

    def test_last_weekday(self):
        from brokers.market_hours import _last_weekday
        # Last Monday of May 2026
        assert _last_weekday(2026, 5, 0) == date(2026, 5, 25)
        # Last Monday of December 2026
        assert _last_weekday(2026, 12, 0) == date(2026, 12, 28)

    def test_good_friday(self):
        from brokers.market_hours import _good_friday
        # Easter 2026 is April 5 → Good Friday = April 3
        assert _good_friday(2026) == date(2026, 4, 3)
        # Easter 2025 is April 20 → Good Friday = April 18
        assert _good_friday(2025) == date(2025, 4, 18)


# ---------------------------------------------------------------------------
# AlpacaBrokerAdapter delegation tests (mocked gap_fade_app)
# ---------------------------------------------------------------------------


def _make_mock_gf():
    """Build a mock gap_fade_app module with all functions the adapter calls."""
    gf = MagicMock()

    # _get_alpaca_config → returns a config dict
    gf._get_alpaca_config.return_value = {
        'api_key': 'test', 'secret_key': 'test',
        'base_url': 'https://paper-api.alpaca.markets',
        'data_url': 'https://data.alpaca.markets',
    }

    # alpaca_get_account
    gf.alpaca_get_account.return_value = {
        'equity': '25000.00', 'buying_power': '50000.00',
        'cash': '25000.00', 'portfolio_value': '25000.00',
    }

    # alpaca_get_positions
    gf.alpaca_get_positions.return_value = [
        {'symbol': 'AAPL', 'qty': '50', 'avg_entry_price': '150.00', 'side': 'long'},
        {'symbol': 'MSFT', 'qty': '30', 'avg_entry_price': '300.00', 'side': 'short'},
    ]

    # alpaca_submit_and_confirm — returns a SimpleNamespace mimicking OrderResult
    gf.alpaca_submit_and_confirm = AsyncMock(return_value=SimpleNamespace(
        order_id='ord-123', status='filled', filled_qty=50,
        filled_avg_price=150.25, symbol='AAPL', side='sell', error='',
    ))

    # alpaca_place_stop_order
    gf.alpaca_place_stop_order.return_value = {
        'id': 'stop-456', 'status': 'new', 'symbol': 'AAPL',
        'stop_price': '155.00', 'limit_price': '155.47',
    }

    # alpaca_cancel_order
    gf.alpaca_cancel_order.return_value = True

    # alpaca_replace_stop_order — async
    gf.alpaca_replace_stop_order = AsyncMock(return_value={
        'id': 'stop-789', 'status': 'new', 'symbol': 'AAPL',
    })

    # alpaca_check_shortable
    gf.alpaca_check_shortable.return_value = (True, True)

    # alpaca_get_order
    gf.alpaca_get_order.return_value = {
        'id': 'stop-456', 'status': 'filled', 'filled_qty': '50',
        'filled_avg_price': '155.00', 'symbol': 'AAPL',
    }

    # fetch_alpaca_assets
    gf.fetch_alpaca_assets.return_value = ['AAPL', 'MSFT', 'GOOG']

    # fetch_alpaca_bars
    gf.fetch_alpaca_bars.return_value = MagicMock()  # DataFrame mock

    # fetch_alpaca_bars_multi
    gf.fetch_alpaca_bars_multi.return_value = {'AAPL': MagicMock()}

    # AlpacaTickStreamer class
    gf.AlpacaTickStreamer = MagicMock

    return gf


@pytest.fixture
def mock_gf():
    """Patch the lazy-import in alpaca_adapter to return our mock."""
    gf = _make_mock_gf()
    with patch('brokers.alpaca_adapter._gf', gf):
        yield gf


class TestAlpacaBrokerAdapter:
    def test_broker_id(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        adapter = AlpacaBrokerAdapter()
        assert adapter.broker_id == 'alpaca'

    def test_is_configured(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        adapter = AlpacaBrokerAdapter()
        assert adapter.is_configured()

    def test_is_not_configured(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        mock_gf._get_alpaca_config.return_value = None
        adapter = AlpacaBrokerAdapter()
        assert not adapter.is_configured()

    @pytest.mark.asyncio
    async def test_get_account(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        adapter = AlpacaBrokerAdapter()
        acct = await adapter.get_account()
        assert acct is not None
        assert acct['equity'] == '25000.00'
        mock_gf.alpaca_get_account.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_positions_normalizes(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        adapter = AlpacaBrokerAdapter()
        positions = await adapter.get_positions()
        assert len(positions) == 2
        p = positions[0]
        assert p['symbol'] == 'AAPL'
        assert p['qty'] == 50.0
        assert isinstance(p['qty'], float)
        assert p['broker_id'] == 'alpaca'
        assert 'raw' in p

    @pytest.mark.asyncio
    async def test_submit_order(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        from brokers.base import OrderResult
        adapter = AlpacaBrokerAdapter()
        result = await adapter.submit_order('AAPL', 50, 'sell')
        assert isinstance(result, OrderResult)
        assert result.is_filled
        assert result.order_id == 'ord-123'
        assert result.filled_qty == 50.0
        mock_gf.alpaca_submit_and_confirm.assert_awaited_once_with(
            'AAPL', 50, 'sell',
            order_type='market', limit_price=None, timeout_sec=30.0,
        )

    @pytest.mark.asyncio
    async def test_submit_order_with_limit(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        adapter = AlpacaBrokerAdapter()
        await adapter.submit_order('AAPL', 100.0, 'buy',
                                   order_type='limit', limit_price=149.50,
                                   timeout_sec=15.0)
        mock_gf.alpaca_submit_and_confirm.assert_awaited_once_with(
            'AAPL', 100, 'buy',
            order_type='limit', limit_price=149.50, timeout_sec=15.0,
        )

    @pytest.mark.asyncio
    async def test_submit_order_truncates_float_qty(self, mock_gf):
        """Alpaca uses int qty — adapter should int() the float."""
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        adapter = AlpacaBrokerAdapter()
        await adapter.submit_order('AAPL', 50.7, 'sell')
        mock_gf.alpaca_submit_and_confirm.assert_awaited_once()
        call_args = mock_gf.alpaca_submit_and_confirm.call_args
        assert call_args[0][1] == 50  # qty arg is int(50.7) = 50

    @pytest.mark.asyncio
    async def test_place_stop_order(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        adapter = AlpacaBrokerAdapter()
        result = await adapter.place_stop_order('AAPL', 50, 155.0, 0.003, 'short')
        assert result['id'] == 'stop-456'
        mock_gf.alpaca_place_stop_order.assert_called_once_with(
            'AAPL', 50, 155.0, 0.003, 'short',
        )

    @pytest.mark.asyncio
    async def test_cancel_order(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        adapter = AlpacaBrokerAdapter()
        ok = await adapter.cancel_order('ord-123')
        assert ok is True
        mock_gf.alpaca_cancel_order.assert_called_once_with('ord-123')

    @pytest.mark.asyncio
    async def test_replace_stop_order(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        adapter = AlpacaBrokerAdapter()
        result = await adapter.replace_stop_order('stop-old', 'AAPL', 50, 152.0)
        assert result['id'] == 'stop-789'
        mock_gf.alpaca_replace_stop_order.assert_awaited_once_with(
            'stop-old', 'AAPL', 50, 152.0, 0.003,
        )

    @pytest.mark.asyncio
    async def test_check_tradeable(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        adapter = AlpacaBrokerAdapter()
        shortable, etb = await adapter.check_tradeable('AAPL')
        assert shortable is True
        assert etb is True
        mock_gf.alpaca_check_shortable.assert_called_once_with('AAPL')

    @pytest.mark.asyncio
    async def test_get_tradeable_symbols(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        adapter = AlpacaBrokerAdapter()
        syms = await adapter.get_tradeable_symbols(min_price=5.0)
        assert syms == ['AAPL', 'MSFT', 'GOOG']
        mock_gf.fetch_alpaca_assets.assert_called_once_with(5.0)

    @pytest.mark.asyncio
    async def test_fetch_bars(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        adapter = AlpacaBrokerAdapter()
        df = await adapter.fetch_bars('AAPL', '2026-01-01', '2026-02-01')
        assert df is not None
        mock_gf.fetch_alpaca_bars.assert_called_once_with(
            'AAPL', '2026-01-01', '2026-02-01', '1Day',
        )

    @pytest.mark.asyncio
    async def test_fetch_bars_multi(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        adapter = AlpacaBrokerAdapter()
        result = await adapter.fetch_bars_multi(['AAPL'], '2026-01-01', '2026-02-01')
        assert 'AAPL' in result
        mock_gf.fetch_alpaca_bars_multi.assert_called_once()

    def test_market_hours_returns_us_equity(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        from brokers.market_hours import USEquityHours
        adapter = AlpacaBrokerAdapter()
        mh = adapter.get_market_hours()
        assert isinstance(mh, USEquityHours)

    @pytest.mark.asyncio
    async def test_get_order(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        adapter = AlpacaBrokerAdapter()
        result = await adapter.get_order('stop-456')
        assert result is not None
        assert result['status'] == 'filled'
        assert result['id'] == 'stop-456'
        mock_gf.alpaca_get_order.assert_called_once_with('stop-456')

    @pytest.mark.asyncio
    async def test_get_order_not_found(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter
        mock_gf.alpaca_get_order.return_value = None
        adapter = AlpacaBrokerAdapter()
        result = await adapter.get_order('nonexistent')
        assert result is None

    def test_create_tick_streamer(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaBrokerAdapter, AlpacaTickStreamerAdapter
        adapter = AlpacaBrokerAdapter()
        streamer = adapter.create_tick_streamer(['AAPL', 'MSFT'])
        assert isinstance(streamer, AlpacaTickStreamerAdapter)


# ---------------------------------------------------------------------------
# TickStreamer adapter delegation tests
# ---------------------------------------------------------------------------


class TestAlpacaTickStreamerAdapter:
    def test_init_delegates(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaTickStreamerAdapter
        streamer = AlpacaTickStreamerAdapter(['AAPL'])
        assert streamer._inner is not None

    def test_latest_prices_proxied(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaTickStreamerAdapter
        streamer = AlpacaTickStreamerAdapter(['AAPL'])
        # MagicMock instances expose .latest_prices as a Mock
        streamer._inner.latest_prices = {'AAPL': 150.0}
        assert streamer.latest_prices == {'AAPL': 150.0}

    def test_connected_proxied(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaTickStreamerAdapter
        streamer = AlpacaTickStreamerAdapter(['AAPL'])
        streamer._inner.connected = True
        assert streamer.connected is True
        streamer._inner.connected = False
        assert streamer.connected is False

    def test_symbols_proxied(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaTickStreamerAdapter
        streamer = AlpacaTickStreamerAdapter(['AAPL', 'MSFT'])
        streamer._inner.symbols = ['AAPL', 'MSFT']
        assert streamer.symbols == ['AAPL', 'MSFT']

    def test_symbols_fallback_no_attr(self, mock_gf):
        from brokers.alpaca_adapter import AlpacaTickStreamerAdapter
        streamer = AlpacaTickStreamerAdapter(['AAPL'])
        # If inner doesn't have symbols attr, should return []
        if hasattr(streamer._inner, 'symbols'):
            del streamer._inner.symbols
        assert streamer.symbols == []


# ---------------------------------------------------------------------------
# Abstract broker cannot be instantiated
# ---------------------------------------------------------------------------


class TestAbstractBrokerABC:
    def test_cannot_instantiate(self):
        from brokers.base import AbstractBroker, BrokerConfig
        with pytest.raises(TypeError):
            AbstractBroker(BrokerConfig())

    def test_tick_streamer_cannot_instantiate(self):
        from brokers.base import TickStreamer
        with pytest.raises(TypeError):
            TickStreamer()
